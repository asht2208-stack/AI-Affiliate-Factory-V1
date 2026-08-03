"""
app.api.routes.products
=========================

Product listing and detail endpoints — this is the route layer that
finally makes the platform's core promise ("search one product,
compare prices across merchants") a real, callable API.

Endpoints
---------
* ``GET /products`` — paginated, filterable product listing.
* ``GET /products/{product_id}`` — full detail page: every variant,
  every merchant's current offer, and a precomputed price-comparison
  summary (lowest/highest price, savings amount, savings percentage).

Design notes
------------
* Queries use ``selectinload`` to eagerly fetch variants, offers,
  merchants, coupons, and images in a bounded number of additional
  queries — this avoids both the N+1 query problem (one query per
  related row) and accidentally triggering a lazy load on an
  ``AsyncSession``, which raises at runtime rather than silently
  working the way it would on a sync session.
* Price-comparison math (lowest/highest/savings) is computed in Python
  after loading a product's offers rather than in SQL, since the
  result set per product (offers across merchants for one product) is
  small and bounded — this keeps the query simple and the comparison
  logic easy to read and unit-test in isolation from the database.
* Affiliate URLs currently point at a placeholder redirect path
  (``/redirect/offer/{offer_id}``). The actual redirect/click-tracking
  service (which resolves that path to the real merchant URL while
  logging the click) is a separate, not-yet-built module — this route
  deliberately does not fabricate a working redirect, it just
  reserves the URL shape that service will implement.
"""

from __future__ import annotations

import math
import uuid
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.schemas.product import (
    CouponOut,
    MerchantOfferOut,
    PaginatedResponse,
    PriceComparisonSummary,
    ProductDetailOut,
    ProductListItemOut,
    ProductVariantOut,
    SpecificationOut,
)
from app.db.models import Category, MasterProduct, MerchantOffer, ProductVariant
from app.db.session import get_db_session

router = APIRouter(prefix="/products", tags=["products"])


def _build_affiliate_url(offer_id: uuid.UUID) -> str:
    """Placeholder affiliate redirect URL shape. See module docstring:
    the real redirect/click-tracking service is a separate, future
    module; this reserves the URL it will serve."""
    return f"/redirect/offer/{offer_id}"


def _offer_to_schema(offer: MerchantOffer) -> MerchantOfferOut:
    """Build the API schema for one offer, pulling in its merchant and
    coupons (both must already be eagerly loaded by the caller's
    query — this function does not perform any I/O itself)."""
    return MerchantOfferOut.build(
        offer_id=offer.id,
        merchant_name=offer.merchant.name,
        merchant_logo_url=offer.merchant.logo_url,
        current_price=offer.current_price,
        currency_code=offer.currency_code,
        shipping_price=offer.shipping_price,
        tax_price=offer.tax_price,
        availability=offer.availability,
        estimated_delivery_days=offer.estimated_delivery_days,
        rating_average=offer.rating_average,
        rating_count=offer.rating_count,
        last_checked_at=offer.last_checked_at,
        coupons=[CouponOut.model_validate(c) for c in offer.coupons],
        affiliate_url=_build_affiliate_url(offer.id),
    )


def _offer_total_price(offer: MerchantOffer) -> Decimal:
    """Total price used for comparison/sorting: price + shipping + tax.
    Centralized here so list and detail endpoints compute "cheapest"
    identically."""
    return offer.current_price + (offer.shipping_price or Decimal("0")) + (offer.tax_price or Decimal("0"))


def _build_price_comparison(variants: list[ProductVariant]) -> PriceComparisonSummary | None:
    """Compute the lowest/highest price and savings across every
    active offer on every variant of a product. Returns ``None`` if
    the product currently has no active offers (e.g., freshly created
    but not yet matched to any merchant)."""
    active_offers = [
        offer for variant in variants for offer in variant.offers if offer.is_active
    ]
    if not active_offers:
        return None

    priced = [(offer, _offer_total_price(offer)) for offer in active_offers]
    cheapest_offer, lowest_price = min(priced, key=lambda pair: pair[1])
    _, highest_price = max(priced, key=lambda pair: pair[1])

    savings_amount = highest_price - lowest_price
    savings_percentage = (
        (savings_amount / highest_price * 100) if highest_price > 0 else Decimal("0")
    )

    return PriceComparisonSummary(
        lowest_total_price=lowest_price,
        highest_total_price=highest_price,
        currency_code=cheapest_offer.currency_code,
        savings_amount=savings_amount,
        savings_percentage=savings_percentage.quantize(Decimal("0.01")),
        merchant_count=len({offer.merchant_id for offer in active_offers}),
        lowest_price_merchant_name=cheapest_offer.merchant.name,
    )


def _variant_detail_query():
    """Shared eager-loading strategy for endpoints that need full
    variant -> offer -> merchant/coupons -> image data, plus the
    product's brand. Defined once so the list and detail endpoints
    can't silently drift into different (and differently N+1-prone)
    loading strategies.

    Includes ``MasterProduct.brand``: accessing ``product.brand``
    without it being eagerly loaded here would trigger an implicit
    lazy load, which raises at runtime on an ``AsyncSession`` rather
    than quietly working the way it would on a sync session.
    """
    return (
        selectinload(MasterProduct.brand),
        selectinload(MasterProduct.variants).selectinload(ProductVariant.offers).selectinload(
            MerchantOffer.merchant
        ),
        selectinload(MasterProduct.variants).selectinload(ProductVariant.offers).selectinload(
            MerchantOffer.coupons
        ),
        selectinload(MasterProduct.variants).selectinload(ProductVariant.images),
    )


@router.get("", response_model=PaginatedResponse)
async def list_products(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=24, ge=1, le=100),
    search: str | None = Query(default=None, description="Case-insensitive title substring match."),
    category_id: uuid.UUID | None = Query(default=None),
    brand_id: uuid.UUID | None = Query(default=None),
    db: AsyncSession = Depends(get_db_session),
) -> PaginatedResponse:
    """Paginated product listing, optionally filtered by search text,
    category, or brand.

    Note on scale: this endpoint queries the primary PostgreSQL
    database directly. The architecture calls for a dedicated search
    index (OpenSearch) to serve this kind of query at scale with
    faceting/autocomplete — that is a planned, separate module. This
    endpoint is correct and useful today for moderate catalog sizes,
    and is expected to be superseded (not necessarily removed) once
    the search-index module exists.
    """
    filters = []
    if search:
        filters.append(MasterProduct.title.ilike(f"%{search}%"))
    if category_id:
        filters.append(MasterProduct.category_id == category_id)
    if brand_id:
        filters.append(MasterProduct.brand_id == brand_id)

    # Built as two independent queries (rather than deriving the count
    # query from the eager-loaded one) so the count query stays a plain,
    # cheap SELECT COUNT — piggybacking it on a query that carries
    # selectinload options would pull those loader strategies into a
    # subquery unnecessarily, with no benefit since we only need a number.
    count_query = select(func.count()).select_from(MasterProduct).where(*filters)
    total = (await db.execute(count_query)).scalar_one()

    paged_query = (
        select(MasterProduct)
        .options(*_variant_detail_query())
        .where(*filters)
        .order_by(MasterProduct.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    products = (await db.execute(paged_query)).scalars().unique().all()

    items: list[ProductListItemOut] = []
    for product in products:
        comparison = _build_price_comparison(list(product.variants))
        items.append(
            ProductListItemOut(
                id=product.id,
                title=product.title,
                brand_name=product.brand.name if product.brand else None,
                canonical_image_url=product.canonical_image_url,
                lowest_total_price=comparison.lowest_total_price if comparison else None,
                currency_code=comparison.currency_code if comparison else None,
                merchant_count=comparison.merchant_count if comparison else 0,
            )
        )

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=max(1, math.ceil(total / page_size)),
    )


async def _resolve_category_path(db: AsyncSession, category_id: uuid.UUID | None) -> list[str]:
    """Walk a category's ancestor chain (leaf -> root) into a
    display-ready breadcrumb list, e.g.
    ``["Electronics", "Audio", "Headphones"]``.

    Uses explicit, individually-awaited ``session.get()`` calls rather
    than the ORM's ``Category.parent`` relationship attribute:
    traversing a lazy relationship attribute-by-attribute is not safe
    on an ``AsyncSession`` (it would attempt an implicit synchronous
    load and raise at runtime). Category trees are shallow in practice
    (a handful of levels), so the extra round trips this costs are
    negligible; a ``seen_ids`` guard also protects against an
    unexpected cycle in the data from causing an infinite loop.
    """
    path: list[str] = []
    seen_ids: set[uuid.UUID] = set()
    current_id = category_id

    while current_id is not None and current_id not in seen_ids:
        category = await db.get(Category, current_id)
        if category is None:
            break
        path.insert(0, category.name)
        seen_ids.add(current_id)
        current_id = category.parent_id

    return path


@router.get("/{product_id}", response_model=ProductDetailOut)
async def get_product(
    product_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
) -> ProductDetailOut:
    """Full product detail page: every variant, every merchant's
    current offer for it, and a precomputed price-comparison summary —
    this is the endpoint a "compare prices across Amazon, eBay,
    Walmart..." page is built from.
    """
    query = (
        select(MasterProduct)
        .where(MasterProduct.id == product_id)
        .options(
            selectinload(MasterProduct.specifications),
            *_variant_detail_query(),
        )
    )
    product = (await db.execute(query)).scalar_one_or_none()
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found.")

    category_path = await _resolve_category_path(db, product.category_id)

    variants_out = [
        ProductVariantOut(
            id=variant.id,
            variant_label=variant.variant_label,
            images=[
                {
                    "id": img.id,
                    "cdn_url": img.cdn_url,
                    "thumbnail_cdn_url": img.thumbnail_cdn_url,
                    "display_order": img.display_order,
                }
                for img in sorted(variant.images, key=lambda i: i.display_order)
            ],
            offers=[
                _offer_to_schema(offer)
                for offer in sorted(variant.offers, key=_offer_total_price)
                if offer.is_active
            ],
        )
        for variant in product.variants
    ]

    return ProductDetailOut(
        id=product.id,
        title=product.title,
        description=product.description,
        brand_name=product.brand.name if product.brand else None,
        category_path=category_path,
        specifications=[SpecificationOut.model_validate(s) for s in product.specifications],
        variants=variants_out,
        price_comparison=_build_price_comparison(list(product.variants)),
    )

