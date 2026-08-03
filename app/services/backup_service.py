"""
app.services.backup_service
==============================

Database backup creation, storage, and restore.

This replaces the architecture's original "every save creates a
backup" idea (reviewed and revised at the start of this project — see
the architecture-review notes) with periodic, complete database
snapshots via ``pg_dump``, stored in object storage alongside a
checksum and metadata row. Per-write versioning is instead handled by
the platform's own append-only tables (``PriceHistory``, and the
import pipeline's own logging) — this service is specifically for
whole-database disaster-recovery snapshots, not per-record undo.

Design notes
------------
* Uses ``pg_dump``'s custom format (``-F c``), which is compressed and
  restorable with ``pg_restore`` selectively (single table, schema
  only, etc.) — plain SQL dumps don't offer that flexibility.
* Shells out to the real ``pg_dump`` / ``pg_restore`` binaries via
  ``asyncio.create_subprocess_exec`` with an explicit argument list
  (never ``shell=True``, never string-interpolated commands) — this is
  what makes it safe regardless of what characters appear in the
  database name or credentials.
* The database password is passed via the ``PGPASSWORD`` environment
  variable for the subprocess only (not the parent process's own
  environment), which is the standard, credential-safe way to
  authenticate a non-interactive ``pg_dump`` run.
* :meth:`BackupService.restore_backup` is destructive by nature —
  it requires an explicit ``confirm=True`` argument and logs at
  CRITICAL level before running, so an accidental call (e.g., a typo'd
  script) can't silently wipe a database.
* Like the image and search services, this module returns plain
  dataclasses and does not itself touch the ``BackupRecord`` ORM model
  or open a database session — persisting the resulting metadata is
  the caller's responsibility (typically a scheduled job), keeping
  this service's own dependencies limited to subprocess + object
  storage.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

from app.core.config import DatabaseSettings, ObjectStorageSettings

logger = logging.getLogger(__name__)


class BackupToolNotFoundError(RuntimeError):
    """Raised when ``pg_dump`` or ``pg_restore`` is not installed /
    not on PATH. Raised eagerly (at the start of the operation) rather
    than letting the subprocess call fail with a cryptic OS error."""


class BackupCreationError(RuntimeError):
    """Raised when ``pg_dump`` exits with a non-zero status."""


class BackupRestoreError(RuntimeError):
    """Raised when ``pg_restore`` exits with a non-zero status, or when
    :meth:`BackupService.restore_backup` is called without explicit
    confirmation."""


class BackupNotFoundError(RuntimeError):
    """Raised when a requested backup's data cannot be found in object
    storage (e.g., a stale ``BackupRecord`` whose file was deleted
    out-of-band)."""


@dataclass(frozen=True)
class ConnectionParams:
    """Parsed, pg_dump/pg_restore-compatible connection parameters,
    extracted from the application's own SQLAlchemy-style DSN so there
    is exactly one place the real connection details are configured
    (see :class:`app.core.config.DatabaseSettings`) — this module never
    duplicates them."""

    host: str
    port: int
    username: str
    password: str
    database: str


def _parse_connection_params(dsn: str) -> ConnectionParams:
    """Parse a SQLAlchemy-style DSN (e.g.
    ``postgresql+asyncpg://user:pass@host:5432/dbname``) into plain
    connection parameters usable as ``pg_dump``/``pg_restore`` CLI
    arguments, which don't understand the ``+asyncpg`` driver suffix.
    """
    parsed = urlparse(dsn)
    if not parsed.hostname or not parsed.path.lstrip("/"):
        raise ValueError(f"Could not parse database DSN into host/database: {dsn!r}")

    return ConnectionParams(
        host=parsed.hostname,
        port=parsed.port or 5432,
        username=unquote(parsed.username) if parsed.username else "postgres",
        password=unquote(parsed.password) if parsed.password else "",
        database=parsed.path.lstrip("/"),
    )


@dataclass(frozen=True)
class BackupArtifact:
    """Result of creating a backup: everything needed to persist a
    ``BackupRecord`` row, but not the row itself (see module
    docstring for why)."""

    storage_key: str
    size_bytes: int
    checksum_sha256: str
    created_at: datetime


@dataclass(frozen=True)
class BackupListing:
    """One entry when listing backups directly from object storage
    (used as a cross-check against the database's own
    ``BackupRecord`` table, e.g. to detect drift between the two)."""

    storage_key: str
    size_bytes: int
    last_modified: datetime


class BackupService:
    """Creates, lists, and restores whole-database backups.

    Like the other process-wide services in this codebase, application
    code should obtain the shared instance via :func:`get_backup_service`.
    """

    def __init__(self, db_settings: DatabaseSettings, storage_settings: ObjectStorageSettings) -> None:
        self._db_settings = db_settings
        self._storage_settings = storage_settings
        self._s3_client = None

    def _get_s3_client(self):
        if self._s3_client is None:
            import boto3

            self._s3_client = boto3.client(
                "s3",
                endpoint_url=str(self._storage_settings.endpoint_url),
                aws_access_key_id=self._storage_settings.access_key_id.get_secret_value(),
                aws_secret_access_key=self._storage_settings.secret_access_key.get_secret_value(),
                region_name=self._storage_settings.region,
            )
        return self._s3_client

    @staticmethod
    def _require_tool(binary_name: str) -> str:
        path = shutil.which(binary_name)
        if path is None:
            raise BackupToolNotFoundError(
                f"'{binary_name}' was not found on PATH. Install the PostgreSQL "
                f"client tools (they ship with any standard postgresql-client "
                f"package) to enable backups."
            )
        return path

    async def create_backup(self, *, triggered_by: str) -> BackupArtifact:
        """Run ``pg_dump`` against the configured database, upload the
        result to object storage, and return the metadata needed to
        record it.

        Parameters
        ----------
        triggered_by
            Human-readable origin of this backup (e.g. ``"scheduled"``
            or an admin username) — stored on the eventual
            ``BackupRecord`` row by the caller, not used internally
            here beyond logging.
        """
        pg_dump_path = self._require_tool("pg_dump")
        params = _parse_connection_params(str(self._db_settings.dsn))

        backup_id = uuid.uuid4()
        with tempfile.TemporaryDirectory() as tmp_dir:
            dump_path = Path(tmp_dir) / f"{backup_id}.dump"

            command = [
                pg_dump_path,
                "-h", params.host,
                "-p", str(params.port),
                "-U", params.username,
                "-d", params.database,
                "-F", "c",  # custom format: compressed, selectively restorable
                "-f", str(dump_path),
                "--no-password",  # never prompt interactively; auth via PGPASSWORD only
            ]

            logger.info("Starting database backup (triggered_by=%s)...", triggered_by)
            process = await asyncio.create_subprocess_exec(
                *command,
                env={**os.environ, "PGPASSWORD": params.password},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise BackupCreationError(
                    f"pg_dump exited with code {process.returncode}: {stderr.decode(errors='replace')}"
                )

            size_bytes = dump_path.stat().st_size
            checksum = await asyncio.to_thread(self._compute_sha256, dump_path)

            created_at = datetime.now(timezone.utc)
            storage_key = f"backups/{created_at:%Y/%m/%d}/{backup_id}.dump"

            await asyncio.to_thread(self._upload_file, dump_path, storage_key)

        logger.info(
            "Backup created: key=%s size=%d bytes checksum=%s", storage_key, size_bytes, checksum
        )
        return BackupArtifact(
            storage_key=storage_key,
            size_bytes=size_bytes,
            checksum_sha256=checksum,
            created_at=created_at,
        )

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _upload_file(self, path: Path, key: str) -> None:
        client = self._get_s3_client()
        client.upload_file(str(path), self._storage_settings.backups_bucket, key)

    def _download_file(self, key: str, destination: Path) -> None:
        client = self._get_s3_client()
        from botocore.exceptions import ClientError

        try:
            client.download_file(self._storage_settings.backups_bucket, key, str(destination))
        except ClientError as exc:
            raise BackupNotFoundError(f"Backup '{key}' could not be downloaded: {exc}") from exc

    async def list_backups_in_storage(self) -> list[BackupListing]:
        """List backup objects directly from object storage. Useful as
        a periodic consistency check against the database's own
        ``BackupRecord`` rows, catching cases where a file was deleted
        out-of-band (e.g., by a storage lifecycle policy) without the
        database being told.
        """
        client = self._get_s3_client()

        def _list() -> list[BackupListing]:
            paginator = client.get_paginator("list_objects_v2")
            results: list[BackupListing] = []
            for page in paginator.paginate(
                Bucket=self._storage_settings.backups_bucket, Prefix="backups/"
            ):
                for obj in page.get("Contents", []):
                    results.append(
                        BackupListing(
                            storage_key=obj["Key"],
                            size_bytes=obj["Size"],
                            last_modified=obj["LastModified"],
                        )
                    )
            return results

        return await asyncio.to_thread(_list)

    async def verify_backup_checksum(self, storage_key: str, expected_checksum: str) -> bool:
        """Download a backup to a temp location and confirm its
        checksum still matches what was recorded at creation time —
        used to periodically audit backup integrity rather than
        discovering a corrupted backup only at restore time, when it's
        too late to matter."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = Path(tmp_dir) / "verify.dump"
            await asyncio.to_thread(self._download_file, storage_key, local_path)
            actual_checksum = await asyncio.to_thread(self._compute_sha256, local_path)
            return actual_checksum == expected_checksum

    async def restore_backup(self, storage_key: str, *, confirm: bool) -> None:
        """Restore the database from a stored backup. **Destructive**:
        this will overwrite existing data in the target database.

        Requires ``confirm=True`` — calling this without it raises
        immediately rather than doing anything, specifically to make
        an accidental/automated call fail loudly instead of silently
        wiping data.
        """
        if not confirm:
            raise BackupRestoreError(
                "restore_backup() called without confirm=True. This operation "
                "overwrites the live database and requires explicit confirmation."
            )

        pg_restore_path = self._require_tool("pg_restore")
        params = _parse_connection_params(str(self._db_settings.dsn))

        logger.critical(
            "DESTRUCTIVE OPERATION: restoring database '%s' on host '%s' from "
            "backup '%s'. This will overwrite existing data.",
            params.database,
            params.host,
            storage_key,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = Path(tmp_dir) / "restore.dump"
            await asyncio.to_thread(self._download_file, storage_key, local_path)

            command = [
                pg_restore_path,
                "-h", params.host,
                "-p", str(params.port),
                "-U", params.username,
                "-d", params.database,
                "--clean",  # drop existing objects before recreating them
                "--if-exists",
                "--no-password",
                str(local_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *command,
                env={**os.environ, "PGPASSWORD": params.password},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise BackupRestoreError(
                    f"pg_restore exited with code {process.returncode}: {stderr.decode(errors='replace')}"
                )

        logger.info("Restore from backup '%s' completed successfully.", storage_key)


_service_instance: Optional[BackupService] = None


def get_backup_service() -> BackupService:
    """Return the process-wide :class:`BackupService` singleton,
    constructed from application settings on first access."""
    global _service_instance
    if _service_instance is None:
        from app.core.config import get_settings

        settings = get_settings()
        _service_instance = BackupService(settings.database, settings.storage)
    return _service_instance

