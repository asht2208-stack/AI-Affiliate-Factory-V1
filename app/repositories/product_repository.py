"""
app.repositories.product_repository
=====================================

Persistence layer between connector output and the database.

This is the first stage of the platform's Universal Import Engine: it
takes a :class:`~app.connectors.base.NormalizedProduct` (whatever a
connector produced, regardless of source) and turns it into the
corresponding rows in ``master_products``, ``product_variants``,
``merchant_offers``, and ``price_history``.

Scope note
----------
This module implements **exact-identifier matching only** (UPC / EAN /
GTIN / MPN) when deciding whether a variant already exists — matching
:class:`~app.db.models.MatchConfidence.EXACT_IDENTIFIER` from the
architecture. It deliberately does NOT implement the fuzzy/embedding-
based matching tier (RapidFuzz + sentence embeddings) described for
the AI Matching Engine — that is a distinct, larger piece of work
planned as its own module, not something this file pretends to cover.
Two products from different merchants that lack a shared identifier
and only "look similar" will currently be created as two separate
master products; that is a known, intentional limitation of this
file, not a bug.

Design notes
------------
* All operations take an already-open ``AsyncSession`` rather than
  managing their own transaction — callers (the import pipeline
  orchestrator, test scripts) control transaction boundaries via
  :meth:`app.db.session.DatabaseSessionManager.session`, so multiple
  repository calls can be composed into one atomic transaction.
* ``ingest_normalized_product`` is the single public entry point most
  callers need; the smaller ``get_or_create_*`` helpers are exposed
  too since the admin panel and future matching-engine work will need
  to call them independently.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.connectors.base import NormalizedProduct
from app.db.models import (
    Brand,
    Category,
    MasterProduct,
    MatchConfidence,
    Merchant,
    MerchantOffer,
    OfferAvailability,
    PriceHistory,
    ProductVariant,
)

logger = logging.getLogger(__name__)

#: Maps a Google-Merchant-style raw availability string to the
#: platform's normalized enum. Extended as more connectors' vocabularies
#: are encountered; unrecognized values fall back to UNKNOWN rather than
#: raising, since a merchant using an unexpected string should not abort
#: an otherwise-valid import.
_AVAILABILITY_MAP: dict[str, OfferAvailability] = {
    "in stock": OfferAvailability.IN_STOCK,
    "in_stock": OfferAvailability.IN_STOCK,
    "limited availability": OfferAvailability.LOW_STOCK,
    "low stock": OfferAvailability.LOW_STOCK,
    "out of stock": OfferAvailability.OUT_OF_STOCK,
    "out_of_stock": OfferAvailability.OUT_OF_STOCK,
    "preorder": OfferAvailability.PREORDER,
    "pre order": OfferAvailability.PREORDER,
    "discontinued": OfferAvailability.DISCONTINUED,
}


def normalize_availability(raw: str | None) -> OfferAvailability:
    """Map a connector's raw availability string to the platform's enum.

    Unrecognized or missing values map to ``UNKNOWN`` rather than
    raising — availability is display/filtering data, not something
    that should block an otherwise-valid price/stock update just
    because one merchant used an unfamiliar phrase.
    """
    if raw is None:
        return OfferAvailability.UNKNOWN
    return _AVAILABILITY_MAP.get(raw.strip().lower(), OfferAvailability.UNKNOWN)


async def get_or_create_brand(session: AsyncSession, brand_name: str | None) -> Brand | None:
    """Find an existing brand by normalized name, or create one.

    Returns ``None`` if ``brand_name`` is empty/whitespace — not every
    connector reliably provides a brand, and a product without one is
    still valid to import.
    """
    if not brand_name or not brand_name.strip():
        return None

    normalized_name = " ".join(brand_name.strip().lower().split())
    existing = await session.scalar(
        select(Brand).where(Brand.normalized_name == normalized_name)
    )
    if existing is not None:
        return existing

    brand = Brand(name=brand_name.strip(), normalized_name=normalized_name)
    session.add(brand)
    await session.flush()  # assigns brand.id without committing the transaction
    logger.info("Created new brand: %s", brand.name)
    return brand


async def get_or_create_category_path(
    session: AsyncSession, category_path: list[str]
) -> Category | None:
    """Find-or-create a chain of categories from a path like
    ``["Electronics", "Audio", "Headphones"]``, returning the deepest
    (most specific) category. Returns ``None`` for an empty path.

    Each level is looked up under its specific parent, so
    "Audio" under "Electronics" is a distinct row from an unrelated
    "Audio" category that might exist under a different top-level
    category.
    """
    if not category_path:
        return None

    parent: Category | None = None
    for level_name in category_path:
        level_name = level_name.strip()
        if not level_name:
            continue
        slug = _slugify(level_name)
        existing = await session.scalar(
            select(Category).where(
                Category.slug == slug,
                Category.parent_id == (parent.id if parent else None),
            )
        )
        if existing is not None:
            parent = existing
            continue

        category = Category(name=level_name, slug=slug, parent_id=parent.id if parent else None)
        session.add(category)
        await session.flush()
        logger.info("Created new category: %s (parent=%s)", level_name, parent.name if parent else None)
        parent = category

    return parent


async def find_variant_by_identifiers(
    session: AsyncSession,
    *,
    upc: str | None,
    ean: str | None,
    gtin: str | None,
    mpn: str | None,
) -> ProductVariant | None:
    """Look for an existing variant sharing any of the given exact
    product identifiers. This is the platform's
    :attr:`~app.db.models.MatchConfidence.EXACT_IDENTIFIER` matching
    tier — the only tier this module implements (see module
    docstring). Checks identifiers in descending order of
    reliability (GTIN, then EAN, then UPC, then MPN), returning the
    first match found.
    """
    for column, value in (
        (ProductVariant.gtin, gtin),
        (ProductVariant.ean, ean),
        (ProductVariant.upc, upc),
        (ProductVariant.mpn, mpn),
    ):
        if not value:
            continue
        match = await session.scalar(select(ProductVariant).where(column == value))
        if match is not None:
            return match
    return None


async def ingest_normalized_product(
    session: AsyncSession, merchant: Merchant, normalized: NormalizedProduct
) -> MerchantOffer:
    """Persist one connector-produced product into the database,
    creating or updating whatever's needed: brand, category chain,
    master product, variant, merchant offer, and a price-history row.

    This is the main entry point the import pipeline calls once per
    product yielded by a connector's ``import_products()``.

    Behavior
    --------
    * If a :class:`MerchantOffer` already exists for this
      ``(merchant, merchant_sku)`` pair, it is updated in place (price
      refresh) rather than duplicated — this is what makes it safe to
      call this function repeatedly for the same product over time.
    * If not, an existing :class:`ProductVariant` is looked up by
      exact identifier match (see :func:`find_variant_by_identifiers`)
      so the same physical product from a second merchant attaches to
      the same master product instead of creating a duplicate.
    * If no matching variant exists either, a new
      :class:`MasterProduct` + :class:`ProductVariant` are created,
      tagged ``MatchConfidence.MANUAL`` initially, or
      ``MatchConfidence.EXACT_IDENTIFIER`` if the match came from a
      shared identifier.
    * A :class:`PriceHistory` row is always appended when the offer's
      price actually changes (or on first creation), preserving the
      "store every price change forever" requirement.
    """
    existing_offer = await session.scalar(
        select(MerchantOffer).where(
            MerchantOffer.merchant_id == merchant.id,
            MerchantOffer.merchant_sku == normalized.merchant_sku,
        )
    )
    availability = normalize_availability(normalized.availability_raw)

    if existing_offer is not None:
        price_changed = existing_offer.current_price != normalized.price.amount
        existing_offer.current_price = normalized.price.amount
        existing_offer.currency_code = normalized.price.currency_code
        existing_offer.shipping_price = normalized.price.shipping_amount
        existing_offer.tax_price = normalized.price.tax_amount
        existing_offer.availability = availability
        existing_offer.merchant_product_url = normalized.merchant_product_url
        existing_offer.rating_average = normalized.rating_average
        existing_offer.rating_count = normalized.rating_count
        existing_offer.is_active = True

        if price_changed:
            session.add(
                PriceHistory(
                    offer_id=existing_offer.id,
                    price=normalized.price.amount,
                    shipping_price=normalized.price.shipping_amount,
                    availability=availability,
                )
            )
            logger.info(
                "Updated price for existing offer merchant_sku=%s merchant=%s: %s",
                normalized.merchant_sku,
                merchant.slug,
                normalized.price.amount,
            )
        return existing_offer

    variant = await find_variant_by_identifiers(
        session,
        upc=normalized.upc,
        ean=normalized.ean,
        gtin=normalized.gtin,
        mpn=normalized.mpn,
    )
    match_confidence = MatchConfidence.EXACT_IDENTIFIER

    if variant is None:
        brand = await get_or_create_brand(session, normalized.brand_name)
        category = await get_or_create_category_path(session, normalized.category_path)

        master_product = MasterProduct(
            title=normalized.title,
            description=normalized.description,
            brand_id=brand.id if brand else None,
            category_id=category.id if category else None,
            match_confidence=MatchConfidence.MANUAL,
            canonical_image_url=normalized.image_urls[0] if normalized.image_urls else None,
        )
        session.add(master_product)
        await session.flush()

        variant = ProductVariant(
            master_product_id=master_product.id,
            variant_label=normalized.title,
            upc=normalized.upc,
            ean=normalized.ean,
            gtin=normalized.gtin,
            mpn=normalized.mpn,
        )
        session.add(variant)
        await session.flush()
        match_confidence = MatchConfidence.MANUAL
        logger.info(
            "Created new master product + variant for merchant_sku=%s: %s",
            normalized.merchant_sku,
            normalized.title,
        )
    else:
        logger.info(
            "Matched merchant_sku=%s to existing variant %s via exact identifier.",
            normalized.merchant_sku,
            variant.id,
        )

    offer = MerchantOffer(
        variant_id=variant.id,
        merchant_id=merchant.id,
        merchant_sku=normalized.merchant_sku,
        merchant_product_url=normalized.merchant_product_url,
        current_price=normalized.price.amount,
        currency_code=normalized.price.currency_code,
        shipping_price=normalized.price.shipping_amount,
        tax_price=normalized.price.tax_amount,
        availability=availability,
        rating_average=normalized.rating_average,
        rating_count=normalized.rating_count,
    )
    session.add(offer)
    await session.flush()

    session.add(
        PriceHistory(
            offer_id=offer.id,
            price=normalized.price.amount,
            shipping_price=normalized.price.shipping_amount,
            availability=availability,
        )
    )

    # Record how this variant was matched, for admin-panel visibility,
    # without downgrading a variant that was already matched with
    # higher confidence by an earlier import. Deliberately uses
    # session.get() rather than the variant.master_product relationship
    # attribute: accessing a lazy relationship on an AsyncSession would
    # attempt an implicit synchronous load, which raises at runtime.
    # session.get() is async-safe and checks the identity map first, so
    # it costs nothing extra when master_product was already loaded or
    # created earlier in this same call.
    master_product_obj = await session.get(MasterProduct, variant.master_product_id)
    if master_product_obj is not None:
        current = master_product_obj.match_confidence
        should_upgrade = current in (MatchConfidence.LOW_CONFIDENCE, MatchConfidence.MEDIUM_CONFIDENCE)
        if match_confidence == MatchConfidence.EXACT_IDENTIFIER and should_upgrade:
            master_product_obj.match_confidence = MatchConfidence.EXACT_IDENTIFIER

    return offer


def _slugify(value: str) -> str:
    """Minimal, dependency-free slugify: lowercase, spaces/underscores
    to hyphens, strip anything not alphanumeric or hyphen. Good enough
    for category names; swapped for a more robust library-based
    implementation if internationalized category names become a
    requirement."""
    cleaned = "".join(
        ch if ch.isalnum() else "-" for ch in value.strip().lower().replace("_", "-")
    )
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")

