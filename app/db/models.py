"""
app.db.models
=============

SQLAlchemy ORM models for the AI Automation Affiliate V3 platform.

This module defines the full relational schema for the system-of-record
database. It implements the Master Product model described in the
architecture: one :class:`MasterProduct` aggregates one or more
:class:`ProductVariant` rows (color/size/storage variations of the same
underlying product), and each variant can have unlimited
:class:`MerchantOffer` rows — one per merchant currently selling it.

Design notes
------------
* Every table uses a UUID primary key (``uuid4``) rather than an
  auto-incrementing integer. At the target scale (100M+ products across
  500+ merchants, imported from many independent feed sources) UUIDs
  avoid primary-key collisions across parallel import workers and make
  it safe to generate an ID for a row before it has been persisted
  (useful in the Universal Import Engine's fingerprinting step).
* :class:`TimestampMixin` centralizes ``created_at`` / ``updated_at``
  so every table gets consistent, server-side-generated timestamps
  without repeating the column definitions.
* ``PriceHistory`` is intentionally a narrow, append-only table (few
  columns, all indexed) since it is the highest-write-volume table in
  the system — every price check on every merchant offer inserts a
  row here. Retention/downsampling of old rows is handled by a
  scheduled job, not by this model.
* Relationships use ``lazy="selectin"`` only where the related data is
  almost always needed alongside the parent (e.g., a variant's images);
  high-fanout relationships (a merchant offer's full price history)
  are left as the default lazy load so a routine product-detail query
  doesn't accidentally pull years of price data.
* This module intentionally contains no query logic — only schema and
  the minimal validation SQLAlchemy itself supports (nullability,
  uniqueness, foreign keys, check constraints). Query/business logic
  belongs in a repository layer (``app/repositories/``), keeping this
  file a pure, easily-testable schema definition.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class TimestampMixin:
    """Adds ``created_at`` / ``updated_at`` columns, both server-side
    generated so application code never needs to remember to set them
    and clock skew between app servers can't produce inconsistent
    timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class UUIDPrimaryKeyMixin:
    """Adds a UUID primary key generated client-side, so a valid ID
    exists on an object before it's flushed to the database (needed by
    the import pipeline, which references a product's ID while still
    building related rows in the same transaction)."""

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )


class MatchConfidence(str, enum.Enum):
    """How the AI Matching Engine arrived at a product's grouping into
    a MasterProduct, used to drive manual-review queues in the admin
    panel — low-confidence matches should be surfaced for human review
    rather than silently trusted."""

    EXACT_IDENTIFIER = "exact_identifier"  # matched on UPC/EAN/GTIN/MPN
    HIGH_CONFIDENCE = "high_confidence"    # embedding + attribute match above threshold
    MEDIUM_CONFIDENCE = "medium_confidence"
    LOW_CONFIDENCE = "low_confidence"      # queued for manual review
    MANUAL = "manual"                      # confirmed or created by a human admin


class OfferAvailability(str, enum.Enum):
    """Normalized stock status across merchants, since every feed and
    API represents availability differently (booleans, string enums,
    quantity thresholds) — normalization happens once, in the import
    pipeline, and every downstream consumer reads this enum."""

    IN_STOCK = "in_stock"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    PREORDER = "preorder"
    DISCONTINUED = "discontinued"
    UNKNOWN = "unknown"


class JobStatus(str, enum.Enum):
    """Lifecycle states for a scheduled/background job run, used by
    both the scheduler and the admin dashboard's job monitor."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRYING = "retrying"


class CouponType(str, enum.Enum):
    PERCENTAGE_OFF = "percentage_off"
    FIXED_AMOUNT_OFF = "fixed_amount_off"
    FREE_SHIPPING = "free_shipping"
    BUNDLE = "bundle"


class Brand(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A product brand/manufacturer, deduplicated across all sources."""

    __tablename__ = "brands"
    __table_args__ = (UniqueConstraint("normalized_name", name="uq_brands_normalized_name"),)

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Lowercased, whitespace-collapsed name used for dedup matching.",
    )
    logo_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    master_products: Mapped[list["MasterProduct"]] = relationship(back_populates="brand")


class Category(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Hierarchical product category (self-referential tree)."""

    __tablename__ = "categories"
    __table_args__ = (
        UniqueConstraint("slug", name="uq_categories_slug"),
        Index("ix_categories_parent_id", "parent_id"),
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), nullable=False)
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("categories.id", ondelete="SET NULL"), nullable=True
    )

    parent: Mapped["Category | None"] = relationship(remote_side="Category.id")
    master_products: Mapped[list["MasterProduct"]] = relationship(back_populates="category")


class MasterProduct(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """The canonical, merchant-agnostic product entity — the "Apple
    iPhone 15 Pro Max" that every merchant's individual listing rolls
    up into. Created and merged by the AI Matching Engine."""

    __tablename__ = "master_products"
    __table_args__ = (
        Index("ix_master_products_brand_id", "brand_id"),
        Index("ix_master_products_category_id", "category_id"),
    )

    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    brand_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("brands.id", ondelete="SET NULL"), nullable=True
    )
    category_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("categories.id", ondelete="SET NULL"), nullable=True
    )
    match_confidence: Mapped[MatchConfidence] = mapped_column(
        SAEnum(MatchConfidence, name="match_confidence"),
        nullable=False,
        default=MatchConfidence.MANUAL,
    )
    match_score: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 4),
        nullable=True,
        doc="0.0000-1.0000 confidence score produced by the matching engine.",
    )
    canonical_image_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    needs_manual_review: Mapped[bool] = mapped_column(default=False, nullable=False)

    brand: Mapped["Brand | None"] = relationship(back_populates="master_products")
    category: Mapped["Category | None"] = relationship(back_populates="master_products")
    variants: Mapped[list["ProductVariant"]] = relationship(
        back_populates="master_product", cascade="all, delete-orphan"
    )
    specifications: Mapped[list["ProductSpecification"]] = relationship(
        back_populates="master_product", cascade="all, delete-orphan"
    )


class ProductVariant(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A specific sellable configuration of a MasterProduct (e.g.,
    "256GB / Natural Titanium"). Every MerchantOffer points to exactly
    one variant, never directly to a MasterProduct — this is what lets
    the platform represent color/size/storage variation correctly."""

    __tablename__ = "product_variants"
    __table_args__ = (
        Index("ix_product_variants_master_product_id", "master_product_id"),
        Index("ix_product_variants_upc", "upc"),
        Index("ix_product_variants_ean", "ean"),
        Index("ix_product_variants_mpn", "mpn"),
    )

    master_product_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("master_products.id", ondelete="CASCADE"), nullable=False
    )
    variant_label: Mapped[str] = mapped_column(
        String(255), nullable=False, doc='e.g. "256GB / Natural Titanium"'
    )
    upc: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ean: Mapped[str | None] = mapped_column(String(64), nullable=True)
    gtin: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mpn: Mapped[str | None] = mapped_column(String(128), nullable=True)

    master_product: Mapped["MasterProduct"] = relationship(back_populates="variants")
    offers: Mapped[list["MerchantOffer"]] = relationship(
        back_populates="variant", cascade="all, delete-orphan"
    )
    images: Mapped[list["ProductImage"]] = relationship(
        back_populates="variant", cascade="all, delete-orphan", lazy="selectin"
    )
    reviews: Mapped[list["ProductReview"]] = relationship(
        back_populates="variant", cascade="all, delete-orphan"
    )


class Merchant(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A merchant/retailer whose products are sold through the
    platform (Amazon, eBay, Best Buy, etc.). One row per merchant,
    referenced by every offer, connector-run, and affiliate link."""

    __tablename__ = "merchants"
    __table_args__ = (UniqueConstraint("slug", name="uq_merchants_slug"),)

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), nullable=False)
    connector_key: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        doc="Identifier matching the connector plugin that services this merchant.",
    )
    logo_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)

    offers: Mapped[list["MerchantOffer"]] = relationship(back_populates="merchant")


class MerchantOffer(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A specific merchant's listing for a specific product variant —
    the row that carries current price, stock, and the affiliate
    link. This is the entity users actually compare across merchants."""

    __tablename__ = "merchant_offers"
    __table_args__ = (
        UniqueConstraint(
            "merchant_id", "merchant_sku", name="uq_merchant_offers_merchant_sku"
        ),
        Index("ix_merchant_offers_variant_id", "variant_id"),
        Index("ix_merchant_offers_merchant_id", "merchant_id"),
        CheckConstraint("current_price >= 0", name="ck_merchant_offers_price_nonnegative"),
    )

    variant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("product_variants.id", ondelete="CASCADE"), nullable=False
    )
    merchant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("merchants.id", ondelete="CASCADE"), nullable=False
    )
    merchant_sku: Mapped[str] = mapped_column(
        String(255), nullable=False, doc="The merchant's own identifier for this listing."
    )
    merchant_product_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    current_price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency_code: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    shipping_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    tax_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    availability: Mapped[OfferAvailability] = mapped_column(
        SAEnum(OfferAvailability, name="offer_availability"),
        nullable=False,
        default=OfferAvailability.UNKNOWN,
    )
    estimated_delivery_days: Mapped[int | None] = mapped_column(nullable=True)
    rating_average: Mapped[Decimal | None] = mapped_column(Numeric(3, 2), nullable=True)
    rating_count: Mapped[int | None] = mapped_column(nullable=True)
    commission_rate: Mapped[Decimal | None] = mapped_column(
        Numeric(6, 4), nullable=True, doc="Affiliate commission rate, e.g. 0.0450 for 4.5%."
    )
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(
        default=True,
        nullable=False,
        doc="False once a merchant stops selling this SKU; retained for price-history continuity.",
    )

    variant: Mapped["ProductVariant"] = relationship(back_populates="offers")
    merchant: Mapped["Merchant"] = relationship(back_populates="offers")
    price_history: Mapped[list["PriceHistory"]] = relationship(back_populates="offer")
    coupons: Mapped[list["Coupon"]] = relationship(back_populates="offer")
    affiliate_links: Mapped[list["AffiliateLink"]] = relationship(back_populates="offer")


class PriceHistory(Base, UUIDPrimaryKeyMixin):
    """Append-only record of every observed price for a merchant
    offer. Deliberately has no ``updated_at`` — rows are never
    updated, only inserted, since this table is the immutable audit
    trail behind the platform's "24h / 7d / 30d / all-time" price
    charts. Retention (downsampling old rows to daily/monthly
    aggregates) is handled by a scheduled maintenance job, not here.
    """

    __tablename__ = "price_history"
    __table_args__ = (
        Index("ix_price_history_offer_id_recorded_at", "offer_id", "recorded_at"),
    )

    offer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("merchant_offers.id", ondelete="CASCADE"), nullable=False
    )
    price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    shipping_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    availability: Mapped[OfferAvailability] = mapped_column(
        SAEnum(OfferAvailability, name="offer_availability"), nullable=False
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    offer: Mapped["MerchantOffer"] = relationship(back_populates="price_history")


class ProductImage(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A processed product image (downloaded, optimized, converted to
    WebP, thumbnailed) served from CDN-backed object storage."""

    __tablename__ = "product_images"
    __table_args__ = (Index("ix_product_images_variant_id", "variant_id"),)

    variant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("product_variants.id", ondelete="CASCADE"), nullable=False
    )
    source_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    cdn_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    thumbnail_cdn_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    width_px: Mapped[int | None] = mapped_column(nullable=True)
    height_px: Mapped[int | None] = mapped_column(nullable=True)
    perceptual_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, doc="Used by the import pipeline to deduplicate identical images."
    )
    display_order: Mapped[int] = mapped_column(default=0, nullable=False)

    variant: Mapped["ProductVariant"] = relationship(back_populates="images")


class ProductSpecification(Base, UUIDPrimaryKeyMixin):
    """A single key/value technical specification attached to a
    MasterProduct (e.g., "Screen Size" / "6.7 in"), used both for
    display and as a matching-engine attribute signal."""

    __tablename__ = "product_specifications"
    __table_args__ = (
        Index("ix_product_specifications_master_product_id", "master_product_id"),
        UniqueConstraint(
            "master_product_id", "spec_key", name="uq_product_specifications_key"
        ),
    )

    master_product_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("master_products.id", ondelete="CASCADE"), nullable=False
    )
    spec_key: Mapped[str] = mapped_column(String(255), nullable=False)
    spec_value: Mapped[str] = mapped_column(String(1024), nullable=False)

    master_product: Mapped["MasterProduct"] = relationship(back_populates="specifications")


class ProductReview(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A single customer review imported from a merchant or review
    provider, attached at the variant level."""

    __tablename__ = "product_reviews"
    __table_args__ = (Index("ix_product_reviews_variant_id", "variant_id"),)

    variant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("product_variants.id", ondelete="CASCADE"), nullable=False
    )
    source: Mapped[str] = mapped_column(
        String(128), nullable=False, doc="Connector key the review was imported from."
    )
    author_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    rating: Mapped[Decimal] = mapped_column(Numeric(3, 2), nullable=False)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    variant: Mapped["ProductVariant"] = relationship(back_populates="reviews")


class Coupon(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A discount code or offer associated with a specific merchant
    offer, surfaced on the product comparison page."""

    __tablename__ = "coupons"
    __table_args__ = (Index("ix_coupons_offer_id", "offer_id"),)

    offer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("merchant_offers.id", ondelete="CASCADE"), nullable=False
    )
    code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    coupon_type: Mapped[CouponType] = mapped_column(
        SAEnum(CouponType, name="coupon_type"), nullable=False
    )
    discount_value: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    description: Mapped[str | None] = mapped_column(String(512), nullable=True)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_verified: Mapped[bool] = mapped_column(default=False, nullable=False)

    offer: Mapped["MerchantOffer"] = relationship(back_populates="coupons")


class AffiliateLink(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A generated, trackable affiliate redirect link for a merchant
    offer. The signed short URL (built from ``link_token``) is what's
    shown to users; the redirect service resolves it back to this row
    to log a click before forwarding to ``destination_url``."""

    __tablename__ = "affiliate_links"
    __table_args__ = (
        UniqueConstraint("link_token", name="uq_affiliate_links_token"),
        Index("ix_affiliate_links_offer_id", "offer_id"),
    )

    offer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("merchant_offers.id", ondelete="CASCADE"), nullable=False
    )
    link_token: Mapped[str] = mapped_column(String(64), nullable=False)
    destination_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    click_count: Mapped[int] = mapped_column(default=0, nullable=False)
    last_clicked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    offer: Mapped["MerchantOffer"] = relationship(back_populates="affiliate_links")


class User(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A registered platform user (for wishlists, saved searches, and
    admin-panel access via ``is_admin``)."""

    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("email", name="uq_users_email"),)

    email: Mapped[str] = mapped_column(String(320), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_admin: Mapped[bool] = mapped_column(default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)


class SearchHistoryEntry(Base, UUIDPrimaryKeyMixin):
    """A single logged search query, used for autocomplete ranking and
    trending-search features. Nullable ``user_id`` supports anonymous
    search logging."""

    __tablename__ = "search_history"
    __table_args__ = (Index("ix_search_history_user_id", "user_id"),)

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    query_text: Mapped[str] = mapped_column(String(512), nullable=False)
    result_count: Mapped[int] = mapped_column(nullable=False, default=0)
    searched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class SchedulerJobRun(Base, UUIDPrimaryKeyMixin):
    """One execution record for a background job (feed import, price
    update, image processing, backup, cleanup). The admin dashboard's
    Scheduler view reads this table directly; Celery's own result
    backend is not used for long-term history since it's typically
    TTL-expired for operational reasons."""

    __tablename__ = "scheduler_job_runs"
    __table_args__ = (
        Index("ix_scheduler_job_runs_job_name_started_at", "job_name", "started_at"),
    )

    job_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus, name="job_status"), nullable=False, default=JobStatus.PENDING
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    items_processed: Mapped[int] = mapped_column(default=0, nullable=False)
    items_failed: Mapped[int] = mapped_column(default=0, nullable=False)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class BackupRecord(Base, UUIDPrimaryKeyMixin):
    """Metadata about one database/backup snapshot stored in object
    storage. The snapshot bytes live in the ``backups_bucket``
    configured in :class:`app.core.config.ObjectStorageSettings`; this
    row is what the admin panel's restore/compare/export UI queries."""

    __tablename__ = "backup_records"
    __table_args__ = (Index("ix_backup_records_created_at", "created_at"),)

    storage_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    size_bytes: Mapped[int] = mapped_column(nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    triggered_by: Mapped[str] = mapped_column(
        String(128), nullable=False, doc='"scheduled", "manual", or a username.'
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
