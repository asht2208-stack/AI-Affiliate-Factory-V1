"""
app.core.config
================

Centralized, validated application configuration for the AI Affiliate
Factory platform.

Every other subsystem (database layer, search layer, cache layer, object
storage layer, connector SDK, scheduler, admin API, public API) reads its
configuration exclusively through the :class:`Settings` object exported
from this module. No module should read ``os.environ`` directly outside
of this file — centralizing configuration here is what lets the rest of
the codebase stay environment-agnostic and unit-testable (settings can be
constructed directly with keyword arguments in tests, bypassing the
environment entirely).

Design notes
------------
* Configuration is loaded via ``pydantic-settings``, which gives us
  type coercion, validation, and clear startup failures when a required
  value is missing or malformed — failing fast at process start is far
  preferable to a connector silently receiving ``None`` for a database
  URL three layers deep.
* Secrets (API keys, database passwords, signing keys) are typed as
  ``pydantic.SecretStr`` so they are never accidentally written to logs,
  tracebacks, or ``repr()`` output. Call ``.get_secret_value()`` only at
  the point of use (e.g., building a connection string or an HTTP auth
  header).
* Settings are organized into nested groups (``DatabaseSettings``,
  ``SearchSettings``, etc.) rather than one flat namespace, both for
  readability and so individual subsystems can depend on just the slice
  of configuration they need (dependency inversion — a connector class
  should type-hint ``ConnectorSecuritySettings``, not the entire
  ``Settings`` object).
* ``get_settings`` is wrapped in ``functools.lru_cache`` so the
  environment is parsed once per process, not once per request; tests
  that need a fresh environment should call ``get_settings.cache_clear()``
  before re-invoking it.
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import (
    AnyHttpUrl,
    Field,
    PostgresDsn,
    RedisDsn,
    SecretStr,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Deployment environment. Drives log verbosity, debug toggles, and
    whether destructive admin operations (e.g., ``DROP`` in migrations)
    are permitted."""

    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Allowed application log levels, constrained to the values Python's
    ``logging`` module actually recognizes so a typo in an env var fails
    at startup instead of silently defaulting."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseSettings(BaseSettings):
    """Connection and pooling configuration for the primary PostgreSQL
    system-of-record database.

    Pooling defaults are conservative; production deployments handling
    500+ concurrent connector imports should override
    ``pool_max_overflow`` via environment variables rather than editing
    this file, keeping the codebase identical across environments.
    """

    model_config = SettingsConfigDict(env_prefix="DB_")

    dsn: PostgresDsn = Field(
        ...,
        description=(
            "Full PostgreSQL DSN, e.g. "
            "postgresql+asyncpg://user:pass@host:5432/affiliate_factory"
        ),
    )
    pool_size: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Baseline number of persistent connections per worker process.",
    )
    pool_max_overflow: int = Field(
        default=40,
        ge=0,
        le=400,
        description="Additional connections allowed above pool_size under burst load.",
    )
    pool_timeout_seconds: int = Field(
        default=30,
        ge=1,
        description="Seconds to wait for a connection from the pool before raising.",
    )
    echo_sql: bool = Field(
        default=False,
        description="If true, logs every SQL statement. Never enable in production.",
    )
    statement_timeout_ms: int = Field(
        default=30_000,
        ge=100,
        description=(
            "Server-side statement timeout in milliseconds. Protects the "
            "database from runaway queries issued by ad-hoc admin reports."
        ),
    )


class SearchSettings(BaseSettings):
    """Configuration for the dedicated search/facet index (OpenSearch),
    which sits alongside — not instead of — the PostgreSQL system of
    record. Product search, autocomplete, and faceting are served from
    here so query load never contends with transactional import writes.
    """

    model_config = SettingsConfigDict(env_prefix="SEARCH_")

    hosts: list[AnyHttpUrl] = Field(
        ...,
        description="One or more OpenSearch node URLs, e.g. https://opensearch:9200",
    )
    username: Optional[str] = Field(default=None)
    password: Optional[SecretStr] = Field(default=None)
    verify_tls: bool = Field(
        default=True,
        description="Disable only for local development against a self-signed node.",
    )
    products_index_name: str = Field(default="products")
    products_index_shards: int = Field(
        default=6,
        ge=1,
        description=(
            "Primary shard count for the products index. Sized for the "
            "100M-product target; re-sharding after go-live requires a "
            "reindex, so this should be set deliberately, not left default "
            "in production."
        ),
    )
    products_index_replicas: int = Field(default=1, ge=0)
    request_timeout_seconds: int = Field(default=10, ge=1)


class CacheSettings(BaseSettings):
    """Configuration for the Redis cache/session/rate-limit layer."""

    model_config = SettingsConfigDict(env_prefix="CACHE_")

    dsn: RedisDsn = Field(..., description="Redis connection URL.")
    default_ttl_seconds: int = Field(
        default=300,
        ge=1,
        description="Default cache-entry lifetime when a caller doesn't specify one.",
    )
    price_quote_ttl_seconds: int = Field(
        default=60,
        ge=1,
        description=(
            "TTL for cached merchant price quotes. Kept short and separate "
            "from default_ttl_seconds because several affiliate networks "
            "contractually forbid displaying prices older than a few minutes."
        ),
    )


class ObjectStorageSettings(BaseSettings):
    """Configuration for S3-compatible object storage used for product
    images, backups, and the event-sourced import log."""

    model_config = SettingsConfigDict(env_prefix="STORAGE_")

    endpoint_url: AnyHttpUrl = Field(
        ..., description="S3-compatible endpoint, e.g. https://s3.us-east-1.amazonaws.com"
    )
    region: str = Field(default="us-east-1")
    access_key_id: SecretStr = Field(...)
    secret_access_key: SecretStr = Field(...)
    images_bucket: str = Field(default="aff-factory-images")
    backups_bucket: str = Field(default="aff-factory-backups")
    import_events_bucket: str = Field(default="aff-factory-import-events")
    cdn_base_url: Optional[AnyHttpUrl] = Field(
        default=None,
        description=(
            "Public CDN URL fronting images_bucket. When unset, image URLs "
            "fall back to the raw endpoint_url, which is discouraged in "
            "production due to latency and egress cost."
        ),
    )


class TaskQueueSettings(BaseSettings):
    """Configuration for the Celery-based background task system that
    drives feed imports, price updates, image processing, and retries."""

    model_config = SettingsConfigDict(env_prefix="QUEUE_")

    broker_url: str = Field(
        ..., description="Message broker URL, e.g. redis://queue:6379/1 or amqp://..."
    )
    result_backend_url: str = Field(
        ..., description="Backend for storing task results and states."
    )
    default_queue_name: str = Field(default="default")
    import_queue_name: str = Field(default="imports")
    max_retries: int = Field(default=5, ge=0)
    retry_backoff_seconds: int = Field(
        default=60,
        ge=1,
        description="Base delay for exponential backoff between task retries.",
    )
    task_soft_time_limit_seconds: int = Field(default=600, ge=1)
    task_hard_time_limit_seconds: int = Field(default=900, ge=1)


class SecuritySettings(BaseSettings):
    """Application-level security configuration: token signing, CORS,
    and rate limiting for public-facing endpoints."""

    model_config = SettingsConfigDict(env_prefix="SECURITY_")

    jwt_signing_key: SecretStr = Field(
        ..., description="Secret key used to sign session/API JWTs."
    )
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_ttl_minutes: int = Field(default=30, ge=1)
    jwt_refresh_token_ttl_days: int = Field(default=14, ge=1)
    allowed_cors_origins: list[str] = Field(default_factory=list)
    public_api_rate_limit_per_minute: int = Field(
        default=120,
        ge=1,
        description="Per-IP request budget for unauthenticated public API traffic.",
    )
    affiliate_link_signing_key: SecretStr = Field(
        ...,
        description=(
            "Separate signing key for outbound affiliate redirect links, "
            "kept distinct from jwt_signing_key so rotating one never "
            "invalidates the other."
        ),
    )


class SecretsBackend(str, Enum):
    """Where connector credentials (merchant/API keys) are ultimately
    resolved from. ``ENV`` is intended for local development only —
    production deployments should use ``VAULT`` or a cloud secrets
    manager so credential rotation doesn't require a redeploy."""

    ENV = "env"
    VAULT = "vault"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"


class ConnectorSecuritySettings(BaseSettings):
    """Configuration governing how the Connector SDK resolves and
    handles per-merchant credentials."""

    model_config = SettingsConfigDict(env_prefix="CONNECTOR_SECRETS_")

    backend: SecretsBackend = Field(default=SecretsBackend.ENV)
    vault_addr: Optional[AnyHttpUrl] = Field(default=None)
    vault_token: Optional[SecretStr] = Field(default=None)
    vault_secrets_path_prefix: str = Field(default="secret/connectors")
    aws_secrets_manager_region: Optional[str] = Field(default=None)

    @field_validator("vault_addr")
    @classmethod
    def _require_vault_addr_when_backend_is_vault(
        cls, value: Optional[AnyHttpUrl], info
    ) -> Optional[AnyHttpUrl]:
        """Ensure that choosing the Vault backend doesn't silently fall
        back to unresolved credentials at runtime — fail at startup
        instead, when the operator can still see the error clearly."""
        backend = info.data.get("backend")
        if backend == SecretsBackend.VAULT and value is None:
            raise ValueError(
                "CONNECTOR_SECRETS_VAULT_ADDR is required when "
                "CONNECTOR_SECRETS_BACKEND=vault"
            )
        return value


class Settings(BaseSettings):
    """Top-level application settings, composed of the grouped settings
    above. This is the single object the rest of the application should
    depend on, obtained via :func:`get_settings`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: Environment = Field(default=Environment.LOCAL)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    app_name: str = Field(default="AI Affiliate Factory")
    app_base_url: AnyHttpUrl = Field(default="http://localhost:8000")

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)  # type: ignore[arg-type]
    search: SearchSettings = Field(default_factory=SearchSettings)  # type: ignore[arg-type]
    cache: CacheSettings = Field(default_factory=CacheSettings)  # type: ignore[arg-type]
    storage: ObjectStorageSettings = Field(default_factory=ObjectStorageSettings)  # type: ignore[arg-type]
    queue: TaskQueueSettings = Field(default_factory=TaskQueueSettings)  # type: ignore[arg-type]
    security: SecuritySettings = Field(default_factory=SecuritySettings)  # type: ignore[arg-type]
    connector_secrets: ConnectorSecuritySettings = Field(
        default_factory=ConnectorSecuritySettings
    )

    @field_validator("log_level")
    @classmethod
    def _warn_on_debug_in_production(cls, value: LogLevel, info) -> LogLevel:
        """Guard against accidentally shipping verbose debug logging (which
        can leak request payloads) to a production deployment."""
        environment = info.data.get("environment")
        if environment == Environment.PRODUCTION and value == LogLevel.DEBUG:
            logger.warning(
                "log_level=DEBUG requested in a PRODUCTION environment; "
                "this may log sensitive request/response payloads. "
                "Proceeding, but this should be confirmed intentional."
            )
        return value

    def is_production(self) -> bool:
        """Convenience predicate used throughout the codebase to gate
        production-only behavior (e.g., disabling destructive admin
        endpoints, enforcing TLS verification)."""
        return self.environment == Environment.PRODUCTION


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton.

    Cached with ``lru_cache`` so environment parsing and validation
    happens exactly once per process rather than on every call site.
    Application code should always obtain settings through this
    function (e.g., via FastAPI's dependency injection:
    ``Depends(get_settings)``) rather than instantiating ``Settings()``
    directly, which keeps call sites trivially mockable in unit tests.

    Raises
    ------
    pydantic.ValidationError
        If any required environment variable is missing or fails
        validation. This is intentionally allowed to propagate and
        crash application startup — a misconfigured production
        deployment should fail loudly before serving traffic, not
        silently run with partial configuration.
    """
    try:
        settings = Settings()  # type: ignore[call-arg]
    except Exception:
        logger.critical(
            "Failed to load application settings from environment. "
            "Check that all required environment variables are set.",
            exc_info=True,
        )
        raise
    logger.info(
        "Loaded settings for environment=%s app_name=%s",
        settings.environment.value,
        settings.app_name,
    )
    return settings

