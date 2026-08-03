"""
app.connectors.plugins.google_merchant_feed
=============================================

Connector for Google Merchant Center product feeds (RSS 2.0 with the
``g:`` namespace) — the reference implementation of a feed-based
connector, demonstrating the platform's feed-first priority.

This file is auto-discovered by
:meth:`app.connectors.registry.ConnectorRegistry.discover` purely by
existing inside ``app/connectors/plugins/`` and defining a
:class:`~app.connectors.base.BaseConnector` subclass — nothing else
needs to register it.

Design notes
------------
* Feeds for a single merchant can contain millions of ``<item>``
  elements. Loading the whole document into memory
  (``xml.etree.ElementTree.parse``) does not scale to the platform's
  100M-product target, so this connector uses ``iterparse`` in a
  background thread: each ``<item>`` is normalized into a
  :class:`~app.connectors.base.NormalizedProduct` immediately, handed
  to the event loop through a thread-safe queue, and only then is the
  raw XML element cleared. Clearing happens strictly *after* extraction
  — clearing child elements before their parent ``<item>`` has been
  read would silently discard every field, which is a real bug this
  design specifically avoids.
* XML parsing is CPU-bound and synchronous, so it runs in a dedicated
  background thread rather than directly on the event loop, which
  would otherwise block the platform from servicing other concurrent
  imports for the duration of the parse. The thread streams results
  back via ``asyncio.Queue`` (using ``run_coroutine_threadsafe``), so
  callers of :meth:`import_products` receive products incrementally,
  not all at once after the whole feed has been parsed.
* Feed connectors generally cannot cheaply refresh a handful of SKUs
  the way a REST API can — the entire feed has to be re-fetched and
  re-parsed regardless of how many SKUs changed. ``update_prices``
  documents this trade-off rather than pretending to be as cheap as an
  API-based connector's equivalent method.
* This connector's :class:`~app.connectors.base.ConnectorPolicy` sets
  ``may_store_reviews=False`` because the standard Google Merchant
  feed schema does not carry review data; :meth:`download_reviews`
  enforces that at runtime per the base class contract.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import threading
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import AsyncIterator, Optional
from xml.etree.ElementTree import Element, iterparse

import httpx

from app.connectors.base import (
    BaseConnector,
    ConnectorAuthenticationError,
    ConnectorCredentials,
    ConnectorHealthStatus,
    ConnectorImportError,
    ConnectorPolicyViolationError,
    ImportResult,
    NormalizedPrice,
    NormalizedProduct,
    NormalizedReview,
)

logger = logging.getLogger(__name__)

# Google Merchant feeds use this namespace for all product-specific tags.
_GOOGLE_SHOPPING_NAMESPACE = "http://base.google.com/ns/1.0"
_NS = {"g": _GOOGLE_SHOPPING_NAMESPACE}


class GoogleMerchantFeedConnector(BaseConnector):
    """Connector for a single merchant's Google Merchant Center product
    feed (RSS 2.0 with ``g:`` namespaced product fields).

    Expects ``ConnectorCredentials.extra["feed_url"]`` to be set to the
    feed's URL when :meth:`authenticate` is called — Google Merchant
    feeds are typically fetched over plain HTTPS with the URL itself
    acting as the access credential (optionally with basic auth for
    private feeds, supported via ``extra["basic_auth_user"]`` /
    ``extra["basic_auth_password"]``).
    """

    connector_key = "google_merchant_feed"
    display_name = "Google Merchant Feed"

    def __init__(self, policy) -> None:  # noqa: ANN001 - policy typed in base
        super().__init__(policy)
        self._feed_url: Optional[str] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._auth: Optional[tuple[str, str]] = None

    async def authenticate(self, credentials: ConnectorCredentials) -> None:
        feed_url = credentials.extra.get("feed_url")
        if not feed_url:
            raise ConnectorAuthenticationError(
                "GoogleMerchantFeedConnector requires credentials.extra['feed_url'] "
                "to be set to the merchant's feed URL."
            )
        self._feed_url = feed_url

        basic_user = credentials.extra.get("basic_auth_user")
        basic_password = credentials.extra.get("basic_auth_password")
        self._auth = (basic_user, basic_password) if basic_user and basic_password else None

        self._http_client = httpx.AsyncClient(auth=self._auth, timeout=60.0)
        self._authenticated = True
        self._logger.info("Authenticated against feed URL: %s", self._feed_url)

    async def health_check(self) -> ConnectorHealthStatus:
        if not self._authenticated or self._http_client is None or self._feed_url is None:
            return ConnectorHealthStatus(
                is_healthy=False,
                checked_at=datetime.now(timezone.utc),
                details="Connector has not been authenticated yet.",
            )
        try:
            response = await self._http_client.head(self._feed_url)
            is_healthy = response.status_code < 400
            return ConnectorHealthStatus(
                is_healthy=is_healthy,
                checked_at=datetime.now(timezone.utc),
                details=f"HTTP {response.status_code} from feed URL.",
            )
        except httpx.HTTPError as exc:
            self._logger.warning("Health check failed for %s: %s", self._feed_url, exc)
            return ConnectorHealthStatus(
                is_healthy=False,
                checked_at=datetime.now(timezone.utc),
                details=f"Request error: {exc}",
            )

    async def search(self, query: str, limit: int = 25) -> AsyncIterator[NormalizedProduct]:
        """Search this feed for products whose title contains ``query``.

        Feed-based connectors have no server-side search endpoint, so
        this necessarily parses the full feed and filters client-side.
        For a large feed this is significantly more expensive than an
        API connector's search — callers doing interactive, user-facing
        search should query the platform's own search index instead of
        calling this method directly; it exists primarily for admin
        tooling and for the import pipeline's own use.
        """
        self._ensure_authenticated()
        query_lower = query.lower()
        matched = 0
        async for product in self.import_products():
            if query_lower in product.title.lower():
                yield product
                matched += 1
                if matched >= limit:
                    return

    async def import_products(
        self, since: Optional[datetime] = None
    ) -> AsyncIterator[NormalizedProduct]:
        """Stream every item in the feed as a :class:`NormalizedProduct`.

        Note: standard Google Merchant feeds do not carry a reliable
        per-item last-modified timestamp, so ``since`` is accepted for
        interface compatibility but this connector always performs a
        full import; incremental filtering (skipping unchanged items)
        happens downstream in the Universal Import Engine by comparing
        each item's content fingerprint against the last known one.
        """
        self._ensure_authenticated()
        assert self._http_client is not None and self._feed_url is not None

        tmp_path = await self._download_feed_to_tempfile()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        stop_event = threading.Event()

        def produce() -> None:
            """Runs in a background thread. Parses the feed with
            iterparse, normalizing each <item> and clearing it
            immediately afterward — clearing happens only once this
            item's data has already been extracted into a
            NormalizedProduct, which is what an earlier version of
            this method got wrong (it cleared child elements before
            their parent <item> was read, silently discarding every
            field). Normalized results are handed to the event loop
            via the thread-safe queue so the caller gets them as a
            true incremental stream rather than all-at-once.

            Checks ``stop_event`` between items so that a caller who
            stops consuming early (e.g. :meth:`search` or
            :meth:`download_images` returning after the first match)
            causes this thread to exit promptly instead of continuing
            to parse — and hold the temp file open — after nobody is
            listening.
            """
            try:
                context = iterparse(str(tmp_path), events=("end",))
                for _event, element in context:
                    if stop_event.is_set():
                        return
                    tag = element.tag.rsplit("}", 1)[-1]
                    if tag != "item":
                        continue
                    try:
                        product = self._normalize_item(element)
                        result: tuple[str, object] = ("ok", product)
                    except ConnectorImportError as exc:
                        result = ("error", str(exc))
                    finally:
                        # Safe to clear now: _normalize_item has already
                        # read everything it needs from this element.
                        element.clear()
                    if stop_event.is_set():
                        return
                    try:
                        asyncio.run_coroutine_threadsafe(queue.put(result), loop).result()
                    except RuntimeError:
                        # Event loop already closed (consumer/process shutting
                        # down) — nothing left to deliver to, stop quietly.
                        return
            except Exception as exc:  # pragma: no cover - defensive
                try:
                    asyncio.run_coroutine_threadsafe(queue.put(("fatal", str(exc))), loop).result()
                except RuntimeError:
                    pass
            finally:
                try:
                    asyncio.run_coroutine_threadsafe(queue.put(("_done", None)), loop).result()
                except RuntimeError:
                    pass

        worker = threading.Thread(target=produce, daemon=True, name="feed-parser")
        worker.start()
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "_done":
                    break
                if kind == "ok":
                    yield payload  # type: ignore[misc]
                elif kind == "error":
                    self._logger.warning("Skipping malformed feed item: %s", payload)
                elif kind == "fatal":
                    raise ConnectorImportError(f"Feed parsing failed: {payload}")
        finally:
            stop_event.set()
            # Joining from a separate thread (not directly on the event-loop
            # thread) matters here: run_coroutine_threadsafe's internal
            # bookkeeping — which the producer thread's last .result() call
            # is waiting on — is itself completed via callbacks scheduled on
            # this loop. A synchronous, blocking worker.join() call made
            # directly on the loop thread would prevent the loop from ever
            # running those callbacks, deadlocking both threads until the
            # timeout. Awaiting the join from a thread-pool thread instead
            # lets the loop keep servicing those callbacks concurrently.
            await asyncio.to_thread(worker.join, 5)
            if worker.is_alive():
                self._logger.warning(
                    "Feed parser thread for %s did not stop within timeout; "
                    "temp file %s left in place to avoid disrupting it.",
                    self._feed_url,
                    tmp_path,
                )
            else:
                tmp_path.unlink(missing_ok=True)

    async def update_prices(self, merchant_skus: list[str]) -> ImportResult:
        """Refresh price/availability for the given SKUs.

        As documented on the class: this connector cannot cheaply
        fetch a subset of SKUs, so it re-downloads and re-parses the
        entire feed and filters to the requested SKUs. Callers driving
        frequent price refreshes for feed-based merchants should batch
        many SKUs into one call rather than calling this repeatedly.
        """
        self._ensure_authenticated()
        wanted = set(merchant_skus)
        found_skus: set[str] = set()
        result = ImportResult(started_at=datetime.now(timezone.utc))

        # Downstream persistence of each refreshed price happens in the import
        # pipeline, which consumes this connector's NormalizedProduct stream
        # via import_products(); this method's own job is only to confirm,
        # per requested SKU, whether current feed data for it was found.
        async for product in self.import_products():
            if product.merchant_sku in wanted:
                found_skus.add(product.merchant_sku)
                result.record_success()

        for missing_sku in wanted - found_skus:
            result.record_failure(f"{missing_sku}: not present in current feed")

        result.finished_at = datetime.now(timezone.utc)
        return result

    async def download_images(self, merchant_sku: str) -> list[str]:
        """Return the source image URLs (main + additional) for one
        item, located by re-scanning the feed for a matching ``g:id``.
        """
        self._ensure_authenticated()
        async for product in self.import_products():
            if product.merchant_sku == merchant_sku:
                return product.image_urls
        return []

    async def download_reviews(self, merchant_sku: str) -> AsyncIterator[NormalizedReview]:
        if not self.policy.may_store_reviews:
            raise ConnectorPolicyViolationError(
                "GoogleMerchantFeedConnector's policy has may_store_reviews=False "
                "(the standard feed schema does not provide review data)."
            )
        return  # pragma: no cover - unreachable; keeps this an async generator
        yield  # pragma: no cover

    async def generate_affiliate_links(
        self, merchant_product_urls: list[str]
    ) -> dict[str, str]:
        """Append a generic tracking parameter to each product URL.

        Google Merchant feeds themselves carry no affiliate tagging
        scheme (Google Shopping is not an affiliate program); this
        implementation appends a ``utm_source``/``utm_medium`` pair so
        click traffic is at least attributable in analytics. Merchants
        who run a real affiliate program alongside their feed should
        use that program's own connector (e.g. CJ, Awin) for actual
        commissioned affiliate links.
        """
        tagged: dict[str, str] = {}
        for url in merchant_product_urls:
            separator = "&" if "?" in url else "?"
            tagged[url] = f"{url}{separator}utm_source=affiliate_factory&utm_medium=feed"
        return tagged

    # -- Internal helpers --------------------------------------------------

    async def _download_feed_to_tempfile(self) -> Path:
        """Stream the feed to a temp file rather than buffering it in
        memory, since feeds can be very large."""
        assert self._http_client is not None and self._feed_url is not None
        fd, tmp_name = tempfile.mkstemp(suffix=".xml")
        tmp_path = Path(tmp_name)
        try:
            async with self._http_client.stream("GET", self._feed_url) as response:
                if response.status_code >= 400:
                    raise ConnectorImportError(
                        f"Feed download failed with HTTP {response.status_code} "
                        f"for URL {self._feed_url}"
                    )
                with open(fd, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
        except httpx.HTTPError as exc:
            tmp_path.unlink(missing_ok=True)
            raise ConnectorImportError(f"Feed download failed: {exc}") from exc
        return tmp_path

    def _normalize_item(self, item: Element) -> NormalizedProduct:
        def g(tag: str) -> Optional[str]:
            found = item.find(f"g:{tag}", _NS)
            return found.text.strip() if found is not None and found.text else None

        merchant_sku = g("id")
        title = item.findtext("title") or g("title")
        link = item.findtext("link") or g("link")
        price_raw = g("price")

        if not merchant_sku or not title or not link or not price_raw:
            raise ConnectorImportError(
                f"Feed item missing one of required fields (id/title/link/price): "
                f"id={merchant_sku!r} title={title!r} link={link!r} price={price_raw!r}"
            )

        image_urls = [url for url in [g("image_link")] if url]
        additional_images = item.findall("g:additional_image_link", _NS)
        image_urls.extend(el.text.strip() for el in additional_images if el.text)

        specifications: dict[str, str] = {}
        for spec_tag in ("color", "size", "material", "pattern"):
            value = g(spec_tag)
            if value:
                specifications[spec_tag] = value

        category_text = g("product_type") or g("google_product_category")
        category_path = [part.strip() for part in category_text.split(">")] if category_text else []

        gtin = g("gtin")
        # GTIN length conventionally distinguishes the underlying identifier
        # scheme: 12 digits is a UPC-A, 13 digits is an EAN-13.
        upc = gtin if gtin and len(gtin) == 12 else None
        ean = gtin if gtin and len(gtin) == 13 else None

        return NormalizedProduct(
            merchant_sku=merchant_sku,
            title=title,
            merchant_product_url=link,
            price=self._parse_price(price_raw),
            brand_name=g("brand"),
            upc=upc,
            ean=ean,
            gtin=gtin,
            mpn=g("mpn"),
            description=item.findtext("description") or g("description"),
            category_path=category_path,
            image_urls=image_urls,
            specifications=specifications,
            availability_raw=g("availability"),
        )

    @staticmethod
    def _parse_price(raw: str) -> NormalizedPrice:
        """Parse Google's price format, e.g. "999.00 USD", into a
        :class:`NormalizedPrice`."""
        parts = raw.strip().split()
        amount_text = parts[0] if parts else raw
        currency_code = parts[1] if len(parts) > 1 else "USD"
        try:
            amount = Decimal(amount_text)
        except InvalidOperation as exc:
            raise ConnectorImportError(f"Could not parse price value: {raw!r}") from exc
        return NormalizedPrice(amount=amount, currency_code=currency_code)

