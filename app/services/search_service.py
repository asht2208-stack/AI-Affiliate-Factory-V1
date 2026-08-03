"""
app.services.search_service
==============================

Product search backed by OpenSearch — the dedicated search index the
architecture calls for, sitting alongside (not instead of) the
PostgreSQL system of record.

Why a separate index at all
----------------------------
``app.api.routes.products.list_products`` already supports a basic
``search`` filter directly against PostgreSQL (a simple ``ILIKE``).
That works for moderate catalog sizes but does not scale to the
platform's 100M-product target and cannot support faceting,
relevance ranking, typo-tolerance, or autocomplete — this module is
what actually delivers those, and is meant to eventually replace
``list_products``'s direct-database search path (kept for now since
it is simple and still useful before this index is populated).

Design notes
------------
* ``opensearch-py``'s client is synchronous; every call here is
  wrapped in ``asyncio.to_thread``, consistent with how this codebase
  already handles other sync-library boundaries (Pillow in
  ``image_service``, boto3 in the same module).
* The index mapping keeps a raw ``title`` field for relevance-scored
  full-text search AND a ``title.autocomplete`` sub-field using an
  edge-ngram analyzer, so one indexed document serves both "search for
  wireless headphones" and "as-you-type suggest wirel..." without a
  second index.
* Indexing is idempotent: :meth:`SearchService.index_product` always
  uses the product's own UUID as the OpenSearch document ID, so
  re-indexing an updated product overwrites the existing document
  rather than creating a duplicate.
* This module does not decide *when* to index — that's the import
  pipeline's job (calling ``index_product`` after
  ``ingest_normalized_product`` persists a product). Keeping indexing
  triggered by the pipeline rather than polled keeps search results
  fresh within seconds of an import, not on some fixed batch cadence.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from app.core.config import SearchSettings

logger = logging.getLogger(__name__)

_INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,  # overridden per-environment via SearchSettings at index-create time
        "number_of_replicas": 1,
        "analysis": {
            "filter": {
                "autocomplete_filter": {
                    "type": "edge_ngram",
                    "min_gram": 2,
                    "max_gram": 20,
                }
            },
            "analyzer": {
                "autocomplete_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "autocomplete_filter"],
                },
                "autocomplete_search_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase"],
                },
            },
        },
    },
    "mappings": {
        "properties": {
            "title": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "autocomplete": {
                        "type": "text",
                        "analyzer": "autocomplete_analyzer",
                        "search_analyzer": "autocomplete_search_analyzer",
                    }
                },
            },
            "description": {"type": "text"},
            "brand_name": {"type": "keyword"},
            "category_path": {"type": "keyword"},
            "lowest_total_price": {"type": "scaled_float", "scaling_factor": 100},
            "currency_code": {"type": "keyword"},
            "merchant_count": {"type": "integer"},
            "canonical_image_url": {"type": "keyword", "index": False},
            "created_at": {"type": "date"},
        }
    },
}


@dataclass(frozen=True)
class SearchHit:
    """One product result from a search query."""

    product_id: uuid.UUID
    title: str
    brand_name: Optional[str]
    canonical_image_url: Optional[str]
    lowest_total_price: Optional[Decimal]
    currency_code: Optional[str]
    merchant_count: int
    relevance_score: float


@dataclass(frozen=True)
class SearchResults:
    hits: list[SearchHit]
    total: int


class SearchService:
    """Thin async wrapper around a synchronous OpenSearch client.

    Like the other process-wide services in this codebase (database
    session manager, image storage, connector registry), application
    code should obtain the shared instance via :func:`get_search_service`
    rather than constructing this repeatedly.
    """

    def __init__(self, settings: SearchSettings) -> None:
        self._settings = settings
        self._client = None  # constructed lazily; see _get_client

    def _get_client(self):
        """Lazily construct the OpenSearch client so importing this
        module (or unit-testing code that only touches, e.g., result
        parsing) never requires ``opensearch-py`` to be installed or a
        real cluster to be reachable."""
        if self._client is None:
            from opensearchpy import OpenSearch  # imported here, not at module level

            auth = (
                (self._settings.username, self._settings.password.get_secret_value())
                if self._settings.username and self._settings.password
                else None
            )
            self._client = OpenSearch(
                hosts=[str(host) for host in self._settings.hosts],
                http_auth=auth,
                verify_certs=self._settings.verify_tls,
                timeout=self._settings.request_timeout_seconds,
            )
        return self._client

    async def ensure_index(self) -> None:
        """Create the products index with its mapping if it doesn't
        already exist. Safe to call on every application startup —
        does nothing if the index is already present.
        """
        client = self._get_client()
        index_name = self._settings.products_index_name

        def _create_if_missing() -> None:
            if client.indices.exists(index=index_name):
                return
            mapping = dict(_INDEX_MAPPING)
            mapping["settings"] = {
                **_INDEX_MAPPING["settings"],
                "number_of_shards": self._settings.products_index_shards,
                "number_of_replicas": self._settings.products_index_replicas,
            }
            client.indices.create(index=index_name, body=mapping)
            logger.info("Created OpenSearch index '%s'.", index_name)

        await asyncio.to_thread(_create_if_missing)

    async def index_product(
        self,
        *,
        product_id: uuid.UUID,
        title: str,
        description: str | None,
        brand_name: str | None,
        category_path: list[str],
        lowest_total_price: Decimal | None,
        currency_code: str | None,
        merchant_count: int,
        canonical_image_url: str | None,
        created_at,
    ) -> None:
        """Index (or re-index) one product. Uses ``product_id`` as the
        OpenSearch document ID so this is safe to call repeatedly for
        the same product as its price/merchant data changes.
        """
        client = self._get_client()
        document = {
            "title": title,
            "description": description,
            "brand_name": brand_name,
            "category_path": category_path,
            "lowest_total_price": float(lowest_total_price) if lowest_total_price is not None else None,
            "currency_code": currency_code,
            "merchant_count": merchant_count,
            "canonical_image_url": canonical_image_url,
            "created_at": created_at.isoformat() if created_at else None,
        }

        def _index() -> None:
            client.index(
                index=self._settings.products_index_name,
                id=str(product_id),
                body=document,
                refresh=False,  # near-real-time is fine; forcing refresh on every write is expensive at scale
            )

        await asyncio.to_thread(_index)

    async def delete_product(self, product_id: uuid.UUID) -> None:
        """Remove a product from the index (e.g., delisted / merged
        into another master product during matching-engine cleanup)."""
        client = self._get_client()

        def _delete() -> None:
            from opensearchpy.exceptions import NotFoundError

            try:
                client.delete(index=self._settings.products_index_name, id=str(product_id))
            except NotFoundError:
                # Already absent — deleting something that isn't there
                # is not an error condition for this method's caller.
                pass

        await asyncio.to_thread(_delete)

    async def search(
        self,
        query: str,
        *,
        page: int = 1,
        page_size: int = 24,
        brand_name: str | None = None,
        category: str | None = None,
    ) -> SearchResults:
        """Full-text search with optional brand/category filters,
        ranked by relevance."""
        client = self._get_client()

        must_clauses: list[dict] = [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "title.autocomplete", "description"],
                    "fuzziness": "AUTO",
                }
            }
        ]
        filter_clauses: list[dict] = []
        if brand_name:
            filter_clauses.append({"term": {"brand_name": brand_name}})
        if category:
            filter_clauses.append({"term": {"category_path": category}})

        request_body = {
            "query": {"bool": {"must": must_clauses, "filter": filter_clauses}},
            "from": (page - 1) * page_size,
            "size": page_size,
        }

        def _search():
            return client.search(index=self._settings.products_index_name, body=request_body)

        response = await asyncio.to_thread(_search)

        hits = [
            SearchHit(
                product_id=uuid.UUID(hit["_id"]),
                title=hit["_source"]["title"],
                brand_name=hit["_source"].get("brand_name"),
                canonical_image_url=hit["_source"].get("canonical_image_url"),
                lowest_total_price=(
                    Decimal(str(hit["_source"]["lowest_total_price"]))
                    if hit["_source"].get("lowest_total_price") is not None
                    else None
                ),
                currency_code=hit["_source"].get("currency_code"),
                merchant_count=hit["_source"].get("merchant_count", 0),
                relevance_score=hit["_score"] or 0.0,
            )
            for hit in response["hits"]["hits"]
        ]
        total = response["hits"]["total"]["value"]

        return SearchResults(hits=hits, total=total)

    async def autocomplete(self, prefix: str, *, limit: int = 8) -> list[str]:
        """Return up to ``limit`` title suggestions for an as-you-type
        search box, using the index's edge-ngram autocomplete field."""
        client = self._get_client()
        request_body = {
            "query": {"match": {"title.autocomplete": prefix}},
            "size": limit,
            "_source": ["title"],
        }

        def _search():
            return client.search(index=self._settings.products_index_name, body=request_body)

        response = await asyncio.to_thread(_search)
        # Deduplicate while preserving relevance order — multiple
        # variants of the same product can otherwise suggest the same
        # title twice.
        seen: set[str] = set()
        suggestions: list[str] = []
        for hit in response["hits"]["hits"]:
            title = hit["_source"]["title"]
            if title not in seen:
                seen.add(title)
                suggestions.append(title)
        return suggestions


_service_instance: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """Return the process-wide :class:`SearchService` singleton,
    constructed from application settings on first access. See
    :func:`app.services.image_service.get_image_storage_service` for
    why this uses a plain module-level cache rather than
    ``lru_cache``."""
    global _service_instance
    if _service_instance is None:
        from app.core.config import get_settings

        _service_instance = SearchService(get_settings().search)
    return _service_instance

