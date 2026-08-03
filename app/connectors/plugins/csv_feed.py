"""
app.connectors.plugins.csv_feed
==================================

Connector for generic CSV product feeds -- common among smaller
merchants who can export a spreadsheet but don't produce a full
Google-Merchant-style XML feed.

This is the platform's second connector, and intentionally reuses the
same download -> background-thread-parse -> async-queue-stream
architecture already built and tested in
``app.connectors.plugins.google_merchant_feed`` (including the fixes
for the data-loss, thread-leak, and event-loop-deadlock bugs found
while testing that connector) -- that pattern is now the platform's
standard shape for any streaming, feed-based connector.

Column mapping
--------------
Real-world CSV exports use wildly inconsistent column names. Rather
than hardcoding one fixed schema, this connector accepts an optional
column mapping via ``credentials.extra["column_mapping"]`` -- a JSON
object mapping the platform's canonical field names to whatever column
headers a specific merchant's file actually uses, e.g.::

    {"sku": "Product Code", "title": "Product Name", "price": "Price (USD)"}

Fields not present in the mapping fall back to a sensible default
column name (see ``_DEFAULT_COLUMN_MAPPING``), so a well-behaved feed
using standard header names needs no mapping at all.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import tempfile
import threading
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import AsyncIterator, Optional

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

_DEFAULT_COLUMN_MAPPING: dict[str, str] = {
    "sku": "sku",
    "title": "title",
    "url": "url",
    "price": "price",
    "currency": "currency",
    "brand": "brand",
    "upc": "upc",
    "ean": "ean",
    "gtin": "gtin",
    "mpn": "mpn",
    "image_url": "image_url",
    "additional_image_urls": "additional_image_urls",
    "category": "category",
    "description": "description",
    "availability": "availability",
}


class CsvFeedConnector(BaseConnector):
    """Connector for a merchant's CSV product feed.

    Expects ``ConnectorCredentials.extra["feed_url"]``. Optionally
    accepts ``extra["column_mapping"]`` (a JSON string) to remap
    non-standard column headers, and ``extra["delimiter"]`` to
    override the default comma delimiter (e.g. a tab character, for
    tab-separated exports -- common enough from spreadsheet tools to
    be worth a first-class option rather than requiring a separate
    connector).
    """

    connector_key = "csv_feed"
    display_name = "Generic CSV Feed"

    def __init__(self, policy) -> None:  # noqa: ANN001 - policy typed in base
        super().__init__(policy)
        self._feed_url: Optional[str] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._column_mapping: dict[str, str] = dict(_DEFAULT_COLUMN_MAPPING)
        self._delimiter: str = ","

    async def authenticate(self, credentials: ConnectorCredentials) -> None:
        feed_url = credentials.extra.get("feed_url")
        if not feed_url:
            raise ConnectorAuthenticationError(
                "CsvFeedConnector requires credentials.extra['feed_url'] to be set."
            )
        self._feed_url = feed_url

        mapping_json = credentials.extra.get("column_mapping")
        if mapping_json:
            try:
                overrides = json.loads(mapping_json)
            except json.JSONDecodeError as exc:
                raise ConnectorAuthenticationError(
                    f"credentials.extra['column_mapping'] is not valid JSON: {exc}"
                ) from exc
            if not isinstance(overrides, dict):
                raise ConnectorAuthenticationError(
                    "credentials.extra['column_mapping'] must be a JSON object."
                )
            self._column_mapping.update(overrides)

        self._delimiter = credentials.extra.get("delimiter", ",")

        basic_user = credentials.extra.get("basic_auth_user")
        basic_password = credentials.extra.get("basic_auth_password")
        auth = (basic_user, basic_password) if basic_user and basic_password else None

        self._http_client = httpx.AsyncClient(auth=auth, timeout=60.0)
        self._authenticated = True
        self._logger.info("Authenticated against CSV feed URL: %s", self._feed_url)

    async def health_check(self) -> ConnectorHealthStatus:
        if not self._authenticated or self._http_client is None or self._feed_url is None:
            return ConnectorHealthStatus(
                is_healthy=False,
                checked_at=datetime.now(timezone.utc),
                details="Connector has not been authenticated yet.",
            )
        try:
            response = await self._http_client.head(self._feed_url)
            return ConnectorHealthStatus(
                is_healthy=response.status_code < 400,
                checked_at=datetime.now(timezone.utc),
                details=f"HTTP {response.status_code} from feed URL.",
            )
        except httpx.HTTPError as exc:
            return ConnectorHealthStatus(
                is_healthy=False,
                checked_at=datetime.now(timezone.utc),
                details=f"Request error: {exc}",
            )

    async def search(self, query: str, limit: int = 25) -> AsyncIterator[NormalizedProduct]:
        """See GoogleMerchantFeedConnector.search -- same client-side
        filtering trade-off applies here for the same reason (no
        server-side search endpoint exists for a static CSV file)."""
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
        """Stream every row in the CSV as a NormalizedProduct.

        Uses the same background-thread-plus-async-queue architecture
        as the Google Merchant connector: parsing happens in a worker
        thread (CSV parsing is CPU/IO-bound and synchronous), results
        stream back through a thread-safe queue, and a ``stop_event``
        lets early-terminating callers (``search``, ``download_images``)
        stop the worker promptly instead of leaking it. The join is
        awaited via ``asyncio.to_thread`` rather than called directly
        on the event-loop thread -- calling it directly would deadlock,
        since the producer's remaining queue puts depend on the loop
        continuing to run while the join waits. See
        ``google_merchant_feed``'s module docstring for the full
        rationale; this is the same fix applied there.
        """
        self._ensure_authenticated()
        assert self._http_client is not None and self._feed_url is not None

        tmp_path = await self._download_feed_to_tempfile()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        stop_event = threading.Event()

        def produce() -> None:
            try:
                with open(tmp_path, "r", newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f, delimiter=self._delimiter)
                    for row in reader:
                        if stop_event.is_set():
                            return
                        try:
                            product = self._normalize_row(row)
                            result: tuple[str, object] = ("ok", product)
                        except ConnectorImportError as exc:
                            result = ("error", str(exc))
                        if stop_event.is_set():
                            return
                        try:
                            asyncio.run_coroutine_threadsafe(queue.put(result), loop).result()
                        except RuntimeError:
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

        worker = threading.Thread(target=produce, daemon=True, name="csv-feed-parser")
        worker.start()
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "_done":
                    break
                if kind == "ok":
                    yield payload  # type: ignore[misc]
                elif kind == "error":
                    self._logger.warning("Skipping malformed CSV row: %s", payload)
                elif kind == "fatal":
                    raise ConnectorImportError(f"CSV parsing failed: {payload}")
        finally:
            stop_event.set()
            await asyncio.to_thread(worker.join, 5)
            if worker.is_alive():
                self._logger.warning(
                    "CSV parser thread for %s did not stop within timeout; "
                    "temp file %s left in place to avoid disrupting it.",
                    self._feed_url,
                    tmp_path,
                )
            else:
                tmp_path.unlink(missing_ok=True)

    async def update_prices(self, merchant_skus: list[str]) -> ImportResult:
        """See GoogleMerchantFeedConnector.update_prices -- same
        full-refresh trade-off applies (a CSV file has no way to
        cheaply fetch a subset of rows)."""
        self._ensure_authenticated()
        wanted = set(merchant_skus)
        found_skus: set[str] = set()
        result = ImportResult(started_at=datetime.now(timezone.utc))

        async for product in self.import_products():
            if product.merchant_sku in wanted:
                found_skus.add(product.merchant_sku)
                result.record_success()

        for missing_sku in wanted - found_skus:
            result.record_failure(f"{missing_sku}: not present in current feed")

        result.finished_at = datetime.now(timezone.utc)
        return result

    async def download_images(self, merchant_sku: str) -> list[str]:
        self._ensure_authenticated()
        async for product in self.import_products():
            if product.merchant_sku == merchant_sku:
                return product.image_urls
        return []

    async def download_reviews(self, merchant_sku: str) -> AsyncIterator[NormalizedReview]:
        if not self.policy.may_store_reviews:
            raise ConnectorPolicyViolationError(
                "CsvFeedConnector's policy has may_store_reviews=False "
                "(a plain CSV product feed does not carry review data)."
            )
        return  # pragma: no cover - unreachable; keeps this an async generator
        yield  # pragma: no cover

    async def generate_affiliate_links(
        self, merchant_product_urls: list[str]
    ) -> dict[str, str]:
        """CSV feeds carry no affiliate tagging scheme of their own;
        same generic UTM-tagging fallback as the Google feed
        connector -- see that class for the full rationale."""
        tagged: dict[str, str] = {}
        for url in merchant_product_urls:
            separator = "&" if "?" in url else "?"
            tagged[url] = f"{url}{separator}utm_source=affiliate_factory&utm_medium=csv_feed"
        return tagged

    # -- Internal helpers --------------------------------------------------

    async def _download_feed_to_tempfile(self) -> Path:
        assert self._http_client is not None and self._feed_url is not None
        fd, tmp_name = tempfile.mkstemp(suffix=".csv")
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

    def _normalize_row(self, row: dict[str, str]) -> NormalizedProduct:
        def col(field: str) -> Optional[str]:
            header = self._column_mapping.get(field, field)
            value = row.get(header)
            return value.strip() if value else None

        merchant_sku = col("sku")
        title = col("title")
        url = col("url")
        price_raw = col("price")

        if not merchant_sku or not title or not url or not price_raw:
            raise ConnectorImportError(
                f"CSV row missing one of required fields (sku/title/url/price): "
                f"sku={merchant_sku!r} title={title!r} url={url!r} price={price_raw!r}"
            )

        try:
            price_amount = Decimal(price_raw.replace(",", "").lstrip("$"))
        except InvalidOperation as exc:
            raise ConnectorImportError(f"Could not parse price value: {price_raw!r}") from exc

        image_url = col("image_url")
        image_urls = [image_url] if image_url else []
        additional_images_raw = col("additional_image_urls")
        if additional_images_raw:
            image_urls.extend(
                part.strip() for part in additional_images_raw.split("|") if part.strip()
            )

        category_raw = col("category")
        category_path = [part.strip() for part in category_raw.split(">")] if category_raw else []

        return NormalizedProduct(
            merchant_sku=merchant_sku,
            title=title,
            merchant_product_url=url,
            price=NormalizedPrice(amount=price_amount, currency_code=col("currency") or "USD"),
            brand_name=col("brand"),
            upc=col("upc"),
            ean=col("ean"),
            gtin=col("gtin"),
            mpn=col("mpn"),
            description=col("description"),
            category_path=category_path,
            image_urls=image_urls,
            availability_raw=col("availability"),
        )

