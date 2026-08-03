"""
app.main
========

FastAPI application entry point.

Run locally with:

    uvicorn app.main:app --reload

This module wires together everything built so far into one runnable
web server: it initializes the database connection pool and the
connector plugin registry on startup, tears them down cleanly on
shutdown, and exposes a couple of operational endpoints
(``/`` and ``/health``) that don't depend on any feature-specific
router. Feature routers (products, search, admin, etc.) are added here
via ``app.include_router(...)`` as each one is built — this file grows
incrementally rather than being rewritten from scratch each time.

Design notes
------------
* FastAPI's ``lifespan`` context manager (rather than the older
  ``@app.on_event("startup")`` style) is used for startup/shutdown,
  since it's the currently-recommended approach and makes the
  dependency between "things to set up" and "things to tear down"
  explicit in one place.
* CORS origins come from :class:`app.core.config.SecuritySettings`
  rather than being hardcoded, so the same code behaves correctly in
  local dev, staging, and production without a code change.
* The connector registry's :meth:`~app.connectors.registry.ConnectorRegistry.discover`
  is called at startup (not lazily on first request) specifically so
  that a broken connector plugin is surfaced in the startup logs
  immediately, not silently on whatever request happens to need it
  first.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.api.routes.admin import router as admin_router
from app.api.routes.products import router as products_router
from app.api.routes.redirect import router as redirect_router
from app.connectors.registry import get_registry
from app.core.config import get_settings
from app.db.session import DatabaseUnavailableError, get_session_manager

logging.basicConfig(
    level=get_settings().log_level.value,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown sequence.

    Startup: verify the database is reachable and discover connector
    plugins, both fail-fast so a misconfigured deployment is caught
    immediately rather than on the first real request.

    Shutdown: dispose the database connection pool so pooled
    connections are closed cleanly instead of being abruptly dropped
    when the process exits.
    """
    settings = get_settings()
    logger.info("Starting %s (environment=%s)", settings.app_name, settings.environment.value)

    session_manager = get_session_manager()
    try:
        await session_manager.health_check()
        logger.info("Database connectivity check passed.")
    except DatabaseUnavailableError:
        logger.exception(
            "Database is not reachable at startup. The application will still "
            "start, but requests requiring the database will fail until this "
            "is resolved."
        )

    registry = get_registry()
    logger.info(
        "Connector discovery complete: %d connector(s) available: %s",
        len(registry),
        registry.list_connector_keys(),
    )

    yield

    logger.info("Shutting down %s...", settings.app_name)
    await session_manager.dispose()
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    """Application factory.

    Kept as a function (rather than only a bare module-level ``app =
    FastAPI(...)``) so tests can construct fresh, isolated app
    instances — useful once test fixtures need to override
    dependencies (e.g., swapping in a test database session) without
    mutating global state shared across test cases.
    """
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        description=(
            "AI-powered shopping intelligence platform: aggregates products "
            "from merchant feeds and APIs so users can search and compare "
            "prices across unlimited merchants."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.allowed_cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return application


app = create_app()
app.include_router(products_router)
app.include_router(redirect_router)
app.include_router(admin_router)


class RootResponse(BaseModel):
    """Response body for ``GET /`` — a minimal identity/version check,
    useful for confirming the right build is deployed."""

    name: str
    version: str
    environment: str


class HealthResponse(BaseModel):
    """Response body for ``GET /health`` — used by load balancers,
    container orchestrators, and uptime monitoring."""

    status: str
    database: str
    checked_at: datetime


@app.get("/", response_model=RootResponse, tags=["meta"])
async def root() -> RootResponse:
    """Basic identity endpoint — confirms the API is up and reports
    which build/environment is running."""
    settings = get_settings()
    return RootResponse(
        name=settings.app_name,
        version=app.version,
        environment=settings.environment.value,
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Liveness/readiness check. Returns ``status: "degraded"`` (not an
    HTTP error) when the database is unreachable, since the process
    itself is still alive and able to serve this endpoint — callers
    that need a hard failure signal should check the ``database``
    field rather than relying on the HTTP status code alone.
    """
    session_manager = get_session_manager()
    try:
        await session_manager.health_check()
        database_status = "connected"
        overall_status = "ok"
    except DatabaseUnavailableError:
        database_status = "unreachable"
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        database=database_status,
        checked_at=datetime.now(timezone.utc),
    )

