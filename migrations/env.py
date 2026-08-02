"""
migrations.env
==============

Alembic environment configuration.

This module is executed by Alembic (not imported by the application),
and is what connects Alembic's migration machinery to the platform's
own configuration and ORM models rather than duplicating either.

Design notes
------------
* The database URL is obtained from
  :func:`app.core.config.get_settings` rather than being duplicated in
  ``alembic.ini`` — there is exactly one place the database connection
  string is configured, avoiding the classic failure mode where
  migrations silently run against the wrong database because
  ``alembic.ini`` and the application's ``.env`` drifted out of sync.
* ``import app.db.models`` (even though nothing in this file appears
  to use it) is required: it is what causes every ORM model class to
  register itself on ``Base.metadata``, which is what Alembic's
  autogenerate diffing needs to see in order to detect new/changed
  tables.
* The engine used for migrations is async (matching the rest of the
  platform), using SQLAlchemy's ``async_engine_from_config`` plus
  ``connection.run_sync(...)`` to run Alembic's own (synchronous)
  migration machinery inside that async connection.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Import the application's settings loader and declarative base.
from app.core.config import get_settings
from app.db.session import Base

# Importing this subpackage registers every ORM model onto Base.metadata,
# which is what makes them visible to Alembic's autogenerate diffing.
import app.db.models  # noqa: F401

# Alembic Config object, providing access to values within alembic.ini.
config = context.config

# Configure Python logging per alembic.ini's [loggers] section, if present.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The metadata object Alembic compares the live database against when
# autogenerating a migration.
target_metadata = Base.metadata


def get_database_url() -> str:
    """Resolve the real database URL from application settings.

    Kept as its own function (rather than inlined) so both
    ``run_migrations_offline`` and ``run_migrations_online`` share one
    code path for obtaining it, and so a future change to how the URL
    is resolved (e.g., adding secrets-manager lookup) only needs to
    happen in one place.
    """
    return str(get_settings().database.dsn)


def run_migrations_offline() -> None:
    """Generate SQL without a live database connection ('offline'
    mode) — used for ``alembic upgrade --sql`` style output that a DBA
    might review before applying by hand.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_sync_migrations(connection: Connection) -> None:
    """The actual (synchronous) Alembic migration logic, run inside an
    async connection via ``connection.run_sync``."""
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Apply migrations against a live database ('online' mode) — the
    normal path used by ``alembic upgrade head`` in every environment.
    """
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_database_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(_run_sync_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
