"""
app.db.session
===============

Async SQLAlchemy engine and session lifecycle management for the primary
PostgreSQL system-of-record database.

This module owns exactly one thing: turning validated
:class:`~app.core.config.DatabaseSettings` into a working, pooled,
async database engine and a way to obtain sessions from it — both for
FastAPI request handlers (via dependency injection) and for background
Celery tasks / the scheduler (via the explicit context manager).

Design notes
------------
* SQLAlchemy 2.0 async style (``AsyncEngine`` / ``AsyncSession``) is used
  throughout, matching the FastAPI-async architecture decision.
* The engine and session factory are wrapped in a single
  :class:`DatabaseSessionManager` rather than left as bare module
  globals. This makes the object trivially replaceable in tests (a test
  fixture can construct its own manager pointed at a throwaway
  database) instead of monkeypatching module-level state.
* ``get_db_session`` is written as an async generator so it can be used
  directly as a FastAPI dependency (``Depends(get_db_session)``); it
  commits on clean exit, rolls back on exception, and always closes the
  session — callers never need to remember to do any of that
  themselves.
* Connection pool parameters, statement timeout, and SQL echo behavior
  all come from :class:`DatabaseSettings`, so pool sizing for a
  500-merchant deployment is a configuration change, not a code change.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import AsyncIterator

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import DatabaseSettings, Settings, get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Shared declarative base for every ORM model in the platform.

    Centralizing this here (rather than each model module declaring its
    own base) is what allows Alembic autogeneration and
    ``Base.metadata.create_all`` in tests to see every model as long as
    the model module has been imported somewhere in the process —
    typically via ``app.db.models`` re-exporting all model classes.
    """

    pass


class DatabaseUnavailableError(RuntimeError):
    """Raised when the database cannot be reached at all (as opposed to
    a query failing for business-logic reasons). Callers such as the
    admin dashboard's health endpoint and the scheduler's startup check
    catch this specifically to distinguish "database is down" from
    "a particular query had a problem."
    """


class DatabaseSessionManager:
    """Owns the async engine and session factory for one database
    connection configuration.

    Instances are cheap to construct but expensive to use incorrectly —
    creating more than one engine per physical database in a single
    process silently doubles your connection pool usage, so application
    code should obtain the process-wide instance via
    :func:`get_session_manager` rather than constructing this directly,
    except in tests where an isolated instance is exactly what you want.
    """

    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings
        self._engine: AsyncEngine = create_async_engine(
            str(settings.dsn),
            echo=settings.echo_sql,
            pool_size=settings.pool_size,
            max_overflow=settings.pool_max_overflow,
            pool_timeout=settings.pool_timeout_seconds,
            pool_pre_ping=True,
            connect_args={
                "server_settings": {
                    "statement_timeout": str(settings.statement_timeout_ms),
                }
            },
        )
        self._session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            autoflush=False,
        )
        logger.info(
            "Initialized DatabaseSessionManager (pool_size=%s, max_overflow=%s)",
            settings.pool_size,
            settings.pool_max_overflow,
        )

    @property
    def engine(self) -> AsyncEngine:
        """Expose the underlying engine for callers that need it directly
        (e.g., Alembic's env.py, or the connector health-check runner)."""
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Provide a transactional session as an async context manager.

        Intended for non-request call sites (Celery tasks, the
        scheduler, connector import jobs) where FastAPI's dependency
        injection isn't available:

            async with session_manager.session() as db:
                await db.execute(...)

        Commits on clean exit, rolls back and re-raises on any
        exception, and always closes the session afterward.
        """
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            logger.exception("Session rolled back due to an unhandled exception.")
            raise
        finally:
            await session.close()

    async def health_check(self) -> bool:
        """Run a trivial query to confirm the database is reachable and
        accepting connections.

        Used by the admin dashboard's ``/health`` endpoint and by the
        scheduler at startup to fail fast rather than begin queuing
        import jobs against a database that isn't actually up.

        Returns
        -------
        bool
            ``True`` if the database responded successfully.

        Raises
        ------
        DatabaseUnavailableError
            If the connection attempt or query fails for any reason.
        """
        try:
            async with self._engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError as exc:
            logger.error("Database health check failed: %s", exc, exc_info=True)
            raise DatabaseUnavailableError(
                "Database health check failed; see logs for details."
            ) from exc

    async def dispose(self) -> None:
        """Close all pooled connections.

        Must be called during graceful application shutdown (FastAPI
        ``lifespan`` shutdown phase, Celery worker shutdown signal) to
        avoid leaking connections when the process exits.
        """
        await self._engine.dispose()
        logger.info("Database engine disposed; all pooled connections closed.")


@lru_cache(maxsize=1)
def get_session_manager(settings: Settings | None = None) -> DatabaseSessionManager:
    """Return the process-wide :class:`DatabaseSessionManager` singleton.

    Cached with ``lru_cache`` so the engine and its connection pool are
    created exactly once per process. Accepts an optional ``settings``
    override purely so tests can pre-seed a distinct manager; normal
    application code should call this with no arguments.

    Parameters
    ----------
    settings
        Application settings to source the database DSN and pool
        configuration from. Defaults to :func:`app.core.config.get_settings`.
    """
    resolved_settings = settings or get_settings()
    return DatabaseSessionManager(resolved_settings.database)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a request-scoped database session.

    Usage::

        @router.get("/products/{product_id}")
        async def get_product(
            product_id: UUID,
            db: AsyncSession = Depends(get_db_session),
        ) -> ProductResponse:
            ...

    Commits when the request handler completes without raising, rolls
    back on any exception, and always closes the session — mirroring
    :meth:`DatabaseSessionManager.session` but shaped as a plain async
    generator, which is the form FastAPI's dependency system expects.
    """
    manager = get_session_manager()
    async with manager.session() as session:
        yield session
