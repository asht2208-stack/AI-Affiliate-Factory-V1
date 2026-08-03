"""
app.api.schemas.product
=========================

Pydantic response schemas for product-facing API endpoints.

These are deliberately separate from the ORM models in
``app.db.models`` — the database schema and the public API shape are
allowed to diverge (e.g., internal fields like ``commission_rate`` are
never exposed here), and keeping them as distinct classes is what
makes that safe. Every schema sets ``model_config =
ConfigDict(from_attributes=True)`` so it can be built directly from an
ORM instance via ``ProductDetailResponse.model_validate(master_product)``
without manual field-by-field mapping in most cases.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import CouponType, OfferAvailability


class CouponOut(BaseModel):
    """A discount coupon available on one merchant's offer."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    code: str | None
    coupon_type: CouponType
    discount_value: Decimal | None
    description: str | None
    is_verified: bool


class ProductImageOut(BaseModel):
    """One product image, with both full-size and thumbnail CDN URLs."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    cdn_url: str | None
    thumbnail_cdn_url: str | None
    display_order: int


class MerchantOfferOut(BaseModel):
    """One merchant's listing for a product variant — the core unit of
    the price comparison view. Deliberately omits internal-only fields
    such as ``commission_rate`` (affiliate economics) and
    ``merchant_sku`` (the merchant's internal identifier), which have
    no reason to be exposed to end users.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    merchant_name: str = Field(..., description="Populated from the related Merchant, not a raw column.")
    merchant_logo_url: str | None = None
    current_price: Decimal
    currency_code: str
    shipping_price: Decimal | None
    tax_price: Decimal | None
    total_price: Decimal = Field(
        ..., description="current_price + shipping_price + tax_price, precomputed for display."
    )
    availability: OfferAvailability
    estimated_delivery_days: int | None
    rating_average: Decimal | None
    rating_count: int | None
    last_checked_at: datetime | None
    coupons: list[CouponOut] = Field(default_factory=list)
    affiliate_url: str = Field(
        ..., description="Signed tracking redirect URL, not the raw merchant URL."
    )

    @classmethod
    def build(
        cls,
        *,
        offer_id: uuid.UUID,
        merchant_name: str,
        merchant_logo_url: str | None,
        current_price: Decimal,
        currency_code: str,
        shipping_price: Decimal | None,
        tax_price: Decimal | None,
        availability: OfferAvailability,
        estimated_delivery_days: int | None,
        rating_average: Decimal | None,
        rating_count: int | None,
        last_checked_at: datetime | None,
        coupons: list[CouponOut],
        affiliate_url: str,
    ) -> "MerchantOfferOut":
        """Construct with a precomputed ``total_price``.

        A plain ``model_validate(offer)`` can't populate
        ``merchant_name``, ``total_price``, or ``affiliate_url``
        directly since they either come from a related table or are
        derived — the route handler assembles these explicitly and
        calls this constructor rather than relying on attribute
        auto-mapping for this particular schema.
        """
        total = current_price + (shipping_price or Decimal("0")) + (tax_price or Decimal("0"))
        return cls(
            id=offer_id,
            merchant_name=merchant_name,
            merchant_logo_url=merchant_logo_url,
            current_price=current_price,
            currency_code=currency_code,
            shipping_price=shipping_price,
            tax_price=tax_price,
            total_price=total,
            availability=availability,
            estimated_delivery_days=estimated_delivery_days,
            rating_average=rating_average,
            rating_count=rating_count,
            last_checked_at=last_checked_at,
            coupons=coupons,
            affiliate_url=affiliate_url,
        )


class ProductVariantOut(BaseModel):
    """One sellable variant of a product (e.g. a specific color/size),
    with every merchant currently offering it."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    variant_label: str
    images: list[ProductImageOut] = Field(default_factory=list)
    offers: list[MerchantOfferOut] = Field(default_factory=list)


class SpecificationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    spec_key: str
    spec_value: str


class PriceComparisonSummary(BaseModel):
    """Precomputed summary shown at the top of a product detail page —
    this is the "how much they saved" figure from the spec: the gap
    between the cheapest and most expensive current offer.
    """

    lowest_total_price: Decimal
    highest_total_price: Decimal
    currency_code: str
    savings_amount: Decimal = Field(
        ..., description="highest_total_price - lowest_total_price."
    )
    savings_percentage: Decimal = Field(
        ..., description="savings_amount as a percentage of highest_total_price."
    )
    merchant_count: int
    lowest_price_merchant_name: str


class ProductListItemOut(BaseModel):
    """One row in a product search/listing page — intentionally light
    (no full offer list) since a listing page shows many products at
    once."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    brand_name: str | None
    canonical_image_url: str | None
    lowest_total_price: Decimal | None
    currency_code: str | None
    merchant_count: int


class ProductDetailOut(BaseModel):
    """Full product detail page payload: the product itself, its
    category breadcrumb, every variant with every merchant offer
    (the actual price-comparison table), and a precomputed savings
    summary.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    description: str | None
    brand_name: str | None
    category_path: list[str] = Field(default_factory=list)
    specifications: list[SpecificationOut] = Field(default_factory=list)
    variants: list[ProductVariantOut] = Field(default_factory=list)
    price_comparison: PriceComparisonSummary | None = None


class PaginatedResponse(BaseModel):
    """Generic pagination envelope used by every list endpoint."""

    items: list[ProductListItemOut]
    total: int
    page: int
    page_size: int
    total_pages: int

