"""
app.connectors.plugins.ebay_api
==================================

Connector for the eBay Browse API — the platform's first
official-API-based connector (Priority 2 in the architecture, behind
feeds), using OAuth2 client credentials authentication.

Scope and limitations (read before use)
----------------------------------------
Unlike a feed, eBay's Browse API is search-oriented: there is no
"give me your entire catalog" endpoint available to this OAuth grant
type. Consequently:

* :meth:`import_products` requires at least one seed keyword or
  category ID to search against (via ``credentials.extra["seed_queries"]``
  or ``credentials.extra["category_ids"]``) — it cannot enumerate "all
  eBay listings" the way a feed connector enumerates "all items in
  this file". This is a real constraint of the API, not an
  implementation shortcut.
* :meth:`update_prices` looks up each requested SKU individually via
  eBay's ``getItem`` endpoint (one HTTP call per SKU) rather than a
  batch endpoint, since the Browse API has no bulk-by-ID lookup for
  this credential type.
* :meth:`download_reviews` is unsupported (raises
  :class:`~app.connectors.base.ConnectorPolicyViolationError`) — the
  Browse API does not expose review data.

Design notes
------------
* OAuth token acquisition and refresh is handled internally
  (:meth:`_ensure_valid_token`), called before every API request —
  callers never need to think about token lifetime.
* Credentials (``app_id`` / ``cert_id``) are expected via
  :class:`~app.connectors.base.ConnectorCredentials`, which the import
  pipeline resolves from :class:`app.core.config.ConnectorSecuritySettings`
  (env vars locally, Vault/Secrets Manager in production) — this
  module never reads environment variables directly.
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import AsyncIterator, Optional

import httpx

from app.connectors.base import (
    BaseConnector,
    ConnectorAuthenticationError,
    ConnectorCredentials,
    ConnectorHealthStatus,
    ConnectorImportError,
    ConnectorPolicyViolationError,
    ConnectorRateLimitError,
    ImportResult,
    NormalizedPrice,
    NormalizedProduct,
    NormalizedReview,
)

logger = logging.getLogger(__name__)

_OAUTH_TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
_BROWSE_API_BASE = "https://api.ebay.com/buy/browse/v1"
_DEFAULT_SCOPE = "https://api.ebay.com/oauth/api_scope"
_DEFAULT_MARKETPLACE = "EBAY_US"
# Refresh the token this many seconds before it actually expires, so a
# request that starts just before expiry doesn't fail mid-flight.
_TOKEN_REFRESH_SAFETY_MARGIN_SECONDS = 60


class EbayConnector(BaseConnector):
    """Connector for eBay's Browse API using OAuth2 client credentials.

    Expects ``ConnectorCredentials.extra``:

    * ``app_id`` (required) — eBay production App ID / Client ID.
    * ``cert_id`` (required) — eBay production Cert ID / Client Secret.
    * ``marketplace_id`` (optional, default ``"EBAY_US"``).
    * ``seed_queries`` (optional) — comma-separated search terms used
      by :meth:`import_products` when no explicit query is given to
      :meth:`search`.
    * ``campaign_id`` (optional) — eBay Partner Network campaign ID,
      used by :meth:`generate_affiliate_links` to build real
      commissioned links; without it, links are returned untagged.
    """

    connector_key = "ebay_api"
    display_name = "eBay (Browse API)"

    def __init__(self, policy) -> None:  # noqa: ANN001 - policy typed in base
        super().__init__(policy)
        self._app_id: Optional[str] = None
        self._cert_id: Optional[str] = None
        self._marketplace_id: str = _DEFAULT_MARKETPLACE
        self._seed_queries: list[str] = []
        self._campaign_id: Optional[str] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None

    async def authenticate(self, credentials: ConnectorCredentials) -> None:
        app_id = credentials.extra.get("app_id")
        cert_id = credentials.extra.get("cert_id")
        if not app_id or not cert_id:
            raise ConnectorAuthenticationError(
                "EbayConnector requires credentials.extra['app_id'] and "
                "credentials.extra['cert_id']."
            )
        self._app_id = app_id
        self._cert_id = cert_id
        self._marketplace_id = credentials.extra.get("marketplace_id", _DEFAULT_MARKETPLACE)
        seed_queries_raw = credentials.extra.get("seed_queries", "")
        self._seed_queries = [q.strip() for q in seed_queries_raw.split(",") if q.strip()]
        self._campaign_id = credentials.extra.get("campaign_id")

        self._http_client = httpx.AsyncClient(timeout=30.0)
        await self._fetch_oauth_token()
        self._authenticated = True
        self._logger.info(
            "Authenticated against eBay Browse API (marketplace=%s).", self._marketplace_id
        )

    async def _fetch_oauth_token(self) -> None:
        """Request a fresh OAuth2 access token via the client
        credentials grant. Called at authentication time and again
        automatically whenever the current token has expired.
        """
        assert self._http_client is not None and self._app_id and self._cert_id

        basic_auth = base64.b64encode(f"{self._app_id}:{self._cert_id}".encode()).decode()
        try:
            response = await self._http_client.post(
                _OAUTH_TOKEN_URL,
                headers={
                    "Authorization": f"Basic {basic_auth}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={"grant_type": "client_credentials", "scope": _DEFAULT_SCOPE},
            )
        except httpx.HTTPError as exc:
            raise ConnectorAuthenticationError(f"eBay OAuth token request failed: {exc}") from exc

        if response.status_code != 200:
            raise ConnectorAuthenticationError(
                f"eBay OAuth token request returned HTTP {response.status_code}: {response.text}"
            )

        body = response.json()
        try:
            self._access_token = body["access_token"]
            expires_in_seconds = int(body["expires_in"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ConnectorAuthenticationError(
                f"eBay OAuth token response missing expected fields: {exc}"
            ) from exc

        self._token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds)

    async def _ensure_valid_token(self) -> None:
        """Refresh the OAuth token if it's expired or about to expire.
        Called before every Browse API request so callers never need
        to think about token lifetime themselves."""
        if (
            self._token_expires_at is None
            or datetime.now(timezone.utc)
            >= self._token_expires_at - timedelta(seconds=_TOKEN_REFRESH_SAFETY_MARGIN_SECONDS)
        ):
            await self._fetch_oauth_token()

    async def health_check(self) -> ConnectorHealthStatus:
        if not self._authenticated:
            return ConnectorHealthStatus(
                is_healthy=False,
                checked_at=datetime.now(timezone.utc),
                details="Connector has not been authenticated yet.",
            )
        try:
            await self._ensure_valid_token()
            return ConnectorHealthStatus(
                is_healthy=True,
                checked_at=datetime.now(timezone.utc),
                details="OAuth token is valid.",
            )
        except ConnectorAuthenticationError as exc:
            return ConnectorHealthStatus(
                is_healthy=False, checked_at=datetime.now(timezone.utc), details=str(exc)
            )

    async def _api_get(self, path: str, params: dict) -> dict:
        """Shared GET-request helper: ensures a valid token, sets the
        required headers, and translates HTTP-level failures into the
        platform's connector exception types."""
        assert self._http_client is not None
        await self._ensure_valid_token()

        try:
            response = await self._http_client.get(
                f"{_BROWSE_API_BASE}{path}",
                params=params,
                headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "X-EBAY-C-MARKETPLACE-ID": self._marketplace_id,
                },
            )
        except httpx.HTTPError as exc:
            raise ConnectorImportError(f"eBay API request to {path} failed: {exc}") from exc

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise ConnectorRateLimitError(
                "eBay API rate limit hit.",
                retry_after_seconds=int(retry_after) if retry_after else None,
            )
        if response.status_code >= 400:
            raise ConnectorImportError(
                f"eBay API request to {path} returned HTTP {response.status_code}: {response.text}"
            )

        return response.json()

    async def search(self, query: str, limit: int = 25) -> AsyncIterator[NormalizedProduct]:
        """Search eBay listings via the Browse API's item_summary
        search endpoint."""
        self._ensure_authenticated()
        capped_limit = min(limit, 200)  # eBay's own per-request maximum
        body = await self._api_get(
            "/item_summary/search", {"q": query, "limit": str(capped_limit)}
        )
        for item_summary in body.get("itemSummaries", []):
            try:
                yield self._normalize_item_summary(item_summary)
            except ConnectorImportError as exc:
                self._logger.warning("Skipping malformed eBay search result: %s", exc)

    async def import_products(
        self, since: Optional[datetime] = None
    ) -> AsyncIterator[NormalizedProduct]:
        """Iterate results across every configured seed query.

        As documented on the class: eBay's Browse API has no bulk
        catalog export for this credential type, so a full import
        means iterating ``credentials.extra["seed_queries"]``. If none
        were configured at authentication time, this yields nothing
        and logs a warning rather than raising — an empty result set
        is a valid outcome for a connector with no seed queries
        configured yet, not necessarily an error condition.
        """
        self._ensure_authenticated()
        if not self._seed_queries:
            self._logger.warning(
                "EbayConnector.import_products called with no seed_queries configured; "
                "yielding nothing. Set credentials.extra['seed_queries'] to search terms."
            )
            return

        for seed_query in self._seed_queries:
            async for product in self.search(seed_query, limit=200):
                yield product

    async def update_prices(self, merchant_skus: list[str]) -> ImportResult:
        """Refresh price/availability for specific eBay item IDs via
        the getItem endpoint — one request per SKU, since the Browse
        API has no batch-by-ID lookup for this credential type."""
        self._ensure_authenticated()
        result = ImportResult(started_at=datetime.now(timezone.utc))

        for item_id in merchant_skus:
            try:
                item = await self._api_get(f"/item/{item_id}", {})
                self._normalize_item_detail(item)  # validates the item is well-formed
                result.record_success()
            except ConnectorImportError as exc:
                result.record_failure(f"{item_id}: {exc}")

        result.finished_at = datetime.now(timezone.utc)
        return result

    async def download_images(self, merchant_sku: str) -> list[str]:
        self._ensure_authenticated()
        item = await self._api_get(f"/item/{merchant_sku}", {})
        image_urls: list[str] = []
        main_image = item.get("image", {}).get("imageUrl")
        if main_image:
            image_urls.append(main_image)
        for additional in item.get("additionalImages", []):
            url = additional.get("imageUrl")
            if url:
                image_urls.append(url)
        return image_urls

    async def download_reviews(self, merchant_sku: str) -> AsyncIterator[NormalizedReview]:
        raise ConnectorPolicyViolationError(
            "EbayConnector does not support review data via the Browse API."
        )
        yield  # pragma: no cover - unreachable; keeps this an async generator

    async def generate_affiliate_links(
        self, merchant_product_urls: list[str]
    ) -> dict[str, str]:
        """Tag URLs with the eBay Partner Network campaign ID if one
        was configured (``credentials.extra["campaign_id"]``);
        otherwise falls back to generic UTM tagging like the feed
        connectors, since a link without a campaign ID earns no
        commission but should still be trackable in analytics.
        """
        tagged: dict[str, str] = {}
        for url in merchant_product_urls:
            separator = "&" if "?" in url else "?"
            if self._campaign_id:
                tagged[url] = f"{url}{separator}campid={self._campaign_id}&mkevt=1"
            else:
                tagged[url] = f"{url}{separator}utm_source=affiliate_factory&utm_medium=ebay_api"
        return tagged

    # -- Internal helpers --------------------------------------------------

    def _normalize_item_summary(self, item_summary: dict) -> NormalizedProduct:
        """Normalize one entry from the search endpoint's
        ``itemSummaries`` array."""
        item_id = item_summary.get("itemId")
        title = item_summary.get("title")
        item_web_url = item_summary.get("itemWebUrl")
        price_block = item_summary.get("price")

        if not item_id or not title or not item_web_url or not price_block:
            raise ConnectorImportError(
                f"eBay item summary missing required fields: itemId={item_id!r} "
                f"title={title!r} itemWebUrl={item_web_url!r} price={price_block!r}"
            )

        try:
            amount = Decimal(str(price_block["value"]))
        except (KeyError, InvalidOperation) as exc:
            raise ConnectorImportError(f"Could not parse eBay item price: {price_block!r}") from exc

        image_url = item_summary.get("image", {}).get("imageUrl")
        category_path = [
            c["categoryName"] for c in item_summary.get("categories", []) if c.get("categoryName")
        ]
        condition = item_summary.get("condition")

        return NormalizedProduct(
            merchant_sku=item_id,
            title=title,
            merchant_product_url=item_web_url,
            price=NormalizedPrice(amount=amount, currency_code=price_block.get("currency", "USD")),
            brand_name=item_summary.get("brand"),
            description=condition,  # search results carry condition, not a full description
            category_path=category_path,
            image_urls=[image_url] if image_url else [],
            availability_raw=item_summary.get("buyingOptions", [None])[0],
        )

    def _normalize_item_detail(self, item: dict) -> NormalizedProduct:
        """Normalize a full item detail response from the getItem
        endpoint (used by :meth:`update_prices`)."""
        item_id = item.get("itemId")
        title = item.get("title")
        item_web_url = item.get("itemWebUrl")
        price_block = item.get("price")

        if not item_id or not title or not item_web_url or not price_block:
            raise ConnectorImportError(
                f"eBay item detail missing required fields for itemId={item_id!r}"
            )

        try:
            amount = Decimal(str(price_block["value"]))
        except (KeyError, InvalidOperation) as exc:
            raise ConnectorImportError(f"Could not parse eBay item price: {price_block!r}") from exc

        return NormalizedProduct(
            merchant_sku=item_id,
            title=title,
            merchant_product_url=item_web_url,
            price=NormalizedPrice(amount=amount, currency_code=price_block.get("currency", "USD")),
            brand_name=item.get("brand"),
            description=item.get("description"),
            availability_raw=item.get("estimatedAvailabilities", [{}])[0].get(
                "estimatedAvailabilityStatus"
            ),
        )

