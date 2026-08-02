"""
app.connectors.base
====================

Abstract base class and shared types for the platform's Connector SDK.

Every product source — official API, affiliate network feed, merchant
feed, or manual import — is implemented as a subclass of
:class:`BaseConnector`. This is the contract that makes the platform's
"unlimited merchants, feed-first" requirement possible: the Universal
Import Engine, the scheduler, and the admin panel all talk to every
connector through this one interface and never need to know whether a
given merchant is actually an XML feed, a REST API, or a CSV drop in an
SFTP folder.

Design notes
------------
* Methods are declared with Python's ``abc`` module so a connector that
  forgets to implement one fails at class-definition time (import time),
  not three months later when the scheduler happens to call it for the
  first time in production.
* Every method that talks to an external source returns a well-typed
  result object (``ImportResult``, ``ConnectorHealthStatus``, etc.)
  rather than a raw dict — this is what lets the rest of the codebase
  handle "not authenticated" versus "rate limited" versus "partial
  import failure" as distinct, type-checked cases instead of parsing
  strings.
* :class:`ConnectorPolicy` encodes the compliance constraints a given
  affiliate network or API imposes (cache TTL, whether images/reviews
  may be stored at all, rate limits). This is deliberately a first-class
  concept, not a comment — several networks contractually forbid
  long-term caching of prices or bulk storage of review text, and a
  connector that ignores its own policy risks the platform's API access
  being revoked.
* ``NormalizedProduct`` is the common currency every connector must
  produce, regardless of source format — it is what the Universal
  Import Engine's normalization stage consumes. Connectors do NOT write
  directly to the database; they hand normalized data to the import
  pipeline, keeping persistence and matching logic out of every
  individual integration.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConnectorError(Exception):
    """Base exception for all connector failures. Callers (the import
    pipeline, the scheduler) catch this to distinguish connector-level
    failures from unrelated application bugs."""


class ConnectorAuthenticationError(ConnectorError):
    """Raised when a connector cannot authenticate against its source
    (expired credentials, revoked API key, failed OAuth handshake)."""


class ConnectorRateLimitError(ConnectorError):
    """Raised when a connector's source signals a rate limit has been
    hit. Carries ``retry_after_seconds`` so the scheduler can back off
    intelligently rather than guessing a delay."""

    def __init__(self, message: str, retry_after_seconds: Optional[int] = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class ConnectorImportError(ConnectorError):
    """Raised for failures during data retrieval/parsing (malformed
    feed, unexpected schema change, partial download)."""


class ConnectorPolicyViolationError(ConnectorError):
    """Raised when a connector or the pipeline attempts an operation
    forbidden by that source's :class:`ConnectorPolicy` (e.g., trying
    to persist reviews from a source whose policy disallows it)."""


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConnectorPolicy:
    """Compliance and operational constraints for one connector,
    typically sourced from that merchant/network's terms of service.

    Enforced centrally (by the import pipeline and the connector
    registry) rather than left to each connector implementation to
    remember, so a new connector is compliant by construction.
    """

    price_cache_ttl_seconds: int = 300
    may_store_images: bool = True
    may_store_reviews: bool = True
    max_requests_per_minute: int = 60
    requires_attribution: bool = False
    attribution_text: Optional[str] = None


@dataclass(frozen=True)
class ConnectorCredentials:
    """Resolved credentials for one connector instance. Values are
    populated by the secrets-resolution layer (Vault / AWS Secrets
    Manager / environment, per
    :class:`app.core.config.ConnectorSecuritySettings`) and handed to
    :meth:`BaseConnector.authenticate` — individual connectors never
    read environment variables or secret stores directly.
    """

    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    account_id: Optional[str] = None
    extra: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedPrice:
    """A single price point in the common currency every connector
    must produce, regardless of source format."""

    amount: Decimal
    currency_code: str = "USD"
    shipping_amount: Optional[Decimal] = None
    tax_amount: Optional[Decimal] = None


@dataclass(frozen=True)
class NormalizedProduct:
    """The common product representation every connector must emit,
    consumed by the Universal Import Engine's normalization stage.
    This is intentionally source-agnostic — it carries no trace of
    whether it came from an XML feed or a REST API response.
    """

    merchant_sku: str
    title: str
    merchant_product_url: str
    price: NormalizedPrice
    brand_name: Optional[str] = None
    upc: Optional[str] = None
    ean: Optional[str] = None
    gtin: Optional[str] = None
    mpn: Optional[str] = None
    description: Optional[str] = None
    category_path: list[str] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)
    specifications: dict[str, str] = field(default_factory=dict)
    availability_raw: Optional[str] = None
    rating_average: Optional[Decimal] = None
    rating_count: Optional[int] = None
    raw_source_payload: Optional[dict] = None


@dataclass(frozen=True)
class NormalizedReview:
    """A single customer review in common form, emitted by
    :meth:`BaseConnector.download_reviews`."""

    merchant_sku: str
    rating: Decimal
    author_name: Optional[str] = None
    title: Optional[str] = None
    body: Optional[str] = None
    reviewed_at: Optional[datetime] = None


@dataclass
class ImportResult:
    """Summary of one call to :meth:`BaseConnector.import_products` or
    :meth:`BaseConnector.update_prices`, used by the scheduler to
    populate ``SchedulerJobRun`` rows and by the admin dashboard to
    show per-connector import health."""

    items_processed: int = 0
    items_succeeded: int = 0
    items_failed: int = 0
    error_messages: list[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def record_success(self) -> None:
        self.items_processed += 1
        self.items_succeeded += 1

    def record_failure(self, error_message: str) -> None:
        self.items_processed += 1
        self.items_failed += 1
        self.error_messages.append(error_message)


@dataclass(frozen=True)
class ConnectorHealthStatus:
    """Result of :meth:`BaseConnector.health_check`, surfaced directly
    in the admin panel's connector list."""

    is_healthy: bool
    checked_at: datetime
    details: Optional[str] = None


# ---------------------------------------------------------------------------
# Base connector
# ---------------------------------------------------------------------------


class BaseConnector(abc.ABC):
    """Abstract base class every product-source connector must extend.

    Subclasses are auto-discovered from the ``app/connectors/plugins/``
    package (see ``app.connectors.registry``) — dropping a new file
    that defines a ``BaseConnector`` subclass there is sufficient to
    register it; no change to this file or the core application is
    required, satisfying the platform's plugin-architecture
    requirement.

    Every subclass must set :attr:`connector_key` to a short, unique,
    stable identifier (e.g. ``"amazon_pa_api"``, ``"cj_affiliate"``) —
    this key is stored on ``Merchant.connector_key`` and used to route
    scheduler jobs to the correct connector instance.
    """

    #: Unique, stable identifier for this connector. Must be overridden.
    connector_key: str = ""

    #: Human-readable display name shown in the admin panel.
    display_name: str = ""

    def __init__(self, policy: ConnectorPolicy) -> None:
        if not self.connector_key:
            raise ValueError(
                f"{type(self).__name__} must set a non-empty 'connector_key' class attribute."
            )
        self.policy = policy
        self._logger = logging.getLogger(f"connectors.{self.connector_key}")
        self._authenticated = False

    # -- Lifecycle -----------------------------------------------------

    @abc.abstractmethod
    async def authenticate(self, credentials: ConnectorCredentials) -> None:
        """Establish (or refresh) authentication against the source.

        Implementations must raise :class:`ConnectorAuthenticationError`
        on failure rather than returning a falsy value, so callers can
        distinguish "not authenticated" from "authenticated but empty
        result" unambiguously.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def health_check(self) -> ConnectorHealthStatus:
        """Perform a lightweight check that the source is reachable and
        credentials are currently valid (e.g., a cheap authenticated
        ping endpoint, or a HEAD request against a feed URL).

        Must not raise for an unhealthy-but-expected state (e.g., a
        feed temporarily returning 503); instead return
        ``ConnectorHealthStatus(is_healthy=False, ...)`` with details.
        Reserve raising for programming errors.
        """
        raise NotImplementedError

    # -- Discovery -------------------------------------------------------

    @abc.abstractmethod
    async def search(self, query: str, limit: int = 25) -> AsyncIterator[NormalizedProduct]:
        """Search the source directly for products matching ``query``.

        Used for on-demand lookups (e.g., an admin manually searching a
        single merchant) as opposed to :meth:`import_products`, which
        performs bulk ingestion. Yields :class:`NormalizedProduct`
        instances rather than returning a list, so callers can start
        processing results before the full search completes.
        """
        raise NotImplementedError
        yield  # pragma: no cover - makes this an async generator for type checkers

    # -- Bulk ingestion --------------------------------------------------

    @abc.abstractmethod
    async def import_products(self, since: Optional[datetime] = None) -> AsyncIterator[NormalizedProduct]:
        """Perform a bulk (full or incremental) product import.

        When ``since`` is provided, implementations should perform an
        incremental import (only products changed since that
        timestamp) where the source supports it, and must document in
        their class docstring if the source only supports full
        imports. Yields products one at a time so the import pipeline
        can stream-process a feed with millions of rows without
        holding it entirely in memory.
        """
        raise NotImplementedError
        yield  # pragma: no cover

    @abc.abstractmethod
    async def update_prices(self, merchant_skus: list[str]) -> ImportResult:
        """Refresh price/availability for a specific batch of already-
        imported SKUs. This is the method the scheduler calls on the
        platform's price-refresh cadence, and is expected to be cheaper
        per-item than a full :meth:`import_products` run.
        """
        raise NotImplementedError

    # -- Enrichment --------------------------------------------------------

    @abc.abstractmethod
    async def download_images(self, merchant_sku: str) -> list[str]:
        """Return source image URLs for one product.

        Implementations return raw source URLs only — downloading,
        optimizing, converting to WebP, and uploading to CDN-backed
        storage is the Image Engine's responsibility, not the
        connector's, keeping this method fast and side-effect-free.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def download_reviews(self, merchant_sku: str) -> AsyncIterator[NormalizedReview]:
        """Yield reviews for one product, if the source provides them
        and :attr:`ConnectorPolicy.may_store_reviews` is ``True`` for
        this connector.

        Implementations must check ``self.policy.may_store_reviews``
        themselves and raise :class:`ConnectorPolicyViolationError`
        immediately if called while that flag is ``False``, rather than
        silently returning nothing (which would mask a caller bug).
        """
        raise NotImplementedError
        yield  # pragma: no cover

    @abc.abstractmethod
    async def generate_affiliate_links(self, merchant_product_urls: list[str]) -> dict[str, str]:
        """Convert plain merchant product URLs into affiliate-tagged
        destination URLs, returned as a mapping of input URL to
        affiliate URL. The redirect/click-tracking layer wraps these
        further; this method's job is only to apply the merchant's own
        affiliate tagging scheme (e.g., appending a tag/tracking
        parameter, or calling a link-generation API).
        """
        raise NotImplementedError

    # -- Shared helpers available to all subclasses -----------------------

    def _ensure_authenticated(self) -> None:
        """Guard clause subclasses should call at the top of any method
        that requires prior authentication, keeping that check
        consistent across every connector instead of duplicated
        ad-hoc checks."""
        if not self._authenticated:
            raise ConnectorAuthenticationError(
                f"Connector '{self.connector_key}' must call authenticate() "
                "before this operation."
            )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} connector_key={self.connector_key!r}>"
