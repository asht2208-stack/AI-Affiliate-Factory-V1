"""
scripts.test_pipeline
======================

End-to-end smoke test for the import pipeline: feed -> connector ->
repository -> database.

What this proves when it runs successfully
--------------------------------------------
1. Your database connection settings (``.env``) are correct.
2. The database is reachable and the migrated tables exist.
3. The Google Merchant Feed connector can download and parse a feed.
4. The repository layer can persist that data as real rows.
5. Running it a second time updates prices instead of duplicating rows
   (proving the "safe to re-run" behavior described in
   ``ingest_normalized_product``'s docstring).

Usage
-----
Run from the project root, with your virtual environment active and
your ``.env`` configured (see ``.env.example``)::

    python -m scripts.test_pipeline

By default this serves the bundled sample feed
(``scripts/sample_data/sample_feed.xml``) from a local HTTP server so
the test is fully self-contained and needs no real merchant feed or
internet access. To test against a real feed instead::

    python -m scripts.test_pipeline --feed-url https://example.com/feed.xml
"""

from __future__ import annotations

import argparse
import asyncio
import http.server
import logging
import sys
import threading
from pathlib import Path

from sqlalchemy import select

from app.connectors.base import ConnectorCredentials, ConnectorPolicy
from app.connectors.plugins.google_merchant_feed import GoogleMerchantFeedConnector
from app.db.models import Merchant
from app.db.session import DatabaseUnavailableError, get_session_manager
from app.repositories.product_repository import ingest_normalized_product

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger("test_pipeline")

_SAMPLE_FEED_PATH = Path(__file__).parent / "sample_data" / "sample_feed.xml"
_TEST_MERCHANT_SLUG = "test-merchant-google-feed"


def _serve_sample_feed_locally() -> tuple[str, http.server.HTTPServer]:
    """Start a background HTTP server serving the bundled sample feed,
    so the smoke test needs no real network access or real merchant
    feed. Returns the feed URL and the server (caller is responsible
    for calling ``server.shutdown()`` when done).
    """
    directory = str(_SAMPLE_FEED_PATH.parent)

    handler_class = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(  # noqa: E731
        *args, directory=directory, **kwargs
    )
    server = http.server.HTTPServer(("127.0.0.1", 0), handler_class)
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    feed_url = f"http://127.0.0.1:{port}/{_SAMPLE_FEED_PATH.name}"
    logger.info("Serving bundled sample feed locally at %s", feed_url)
    return feed_url, server


async def _get_or_create_test_merchant(session) -> Merchant:
    """Ensure a Merchant row exists to attach this test run's offers
    to, so repeated runs reuse the same merchant instead of piling up
    duplicates."""
    existing = await session.scalar(select(Merchant).where(Merchant.slug == _TEST_MERCHANT_SLUG))
    if existing is not None:
        return existing

    merchant = Merchant(
        name="Test Merchant (Google Feed)",
        slug=_TEST_MERCHANT_SLUG,
        connector_key=GoogleMerchantFeedConnector.connector_key,
    )
    session.add(merchant)
    await session.flush()
    logger.info("Created test merchant: %s", merchant.slug)
    return merchant


async def run(feed_url: str | None) -> int:
    """Run the smoke test. Returns a process exit code (0 = success)."""
    session_manager = get_session_manager()

    logger.info("Checking database connectivity...")
    try:
        await session_manager.health_check()
    except DatabaseUnavailableError:
        logger.error(
            "Could not reach the database. Is it running (docker compose up -d) "
            "and does .env match your docker-compose.yml settings?"
        )
        return 1
    logger.info("Database is reachable.")

    local_server = None
    if feed_url is None:
        if not _SAMPLE_FEED_PATH.exists():
            logger.error("Bundled sample feed not found at %s", _SAMPLE_FEED_PATH)
            return 1
        feed_url, local_server = _serve_sample_feed_locally()

    try:
        connector = GoogleMerchantFeedConnector(policy=ConnectorPolicy())
        await connector.authenticate(ConnectorCredentials(extra={"feed_url": feed_url}))

        health = await connector.health_check()
        if not health.is_healthy:
            logger.error("Connector health check failed: %s", health.details)
            return 1
        logger.info("Connector health check passed.")

        created_or_updated = 0
        skipped = 0

        async with session_manager.session() as db:
            merchant = await _get_or_create_test_merchant(db)

            async for normalized_product in connector.import_products():
                try:
                    offer = await ingest_normalized_product(db, merchant, normalized_product)
                    created_or_updated += 1
                    logger.info(
                        "Ingested: sku=%s title=%r price=%s %s -> offer_id=%s",
                        normalized_product.merchant_sku,
                        normalized_product.title,
                        normalized_product.price.amount,
                        normalized_product.price.currency_code,
                        offer.id,
                    )
                except Exception:
                    skipped += 1
                    logger.exception(
                        "Failed to ingest sku=%s; continuing with remaining products.",
                        normalized_product.merchant_sku,
                    )

        logger.info(
            "Done. %d product(s) ingested/updated, %d skipped due to errors.",
            created_or_updated,
            skipped,
        )
        if created_or_updated == 0:
            logger.warning("No products were ingested — check connector/feed configuration.")
            return 1
        return 0
    finally:
        if local_server is not None:
            local_server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feed-url",
        default=None,
        help="Real feed URL to test against. Omit to use the bundled sample feed.",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(run(args.feed_url))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

