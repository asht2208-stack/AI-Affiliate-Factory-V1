"""
app.api.routes.redirect
=========================

Affiliate redirect and click-tracking endpoint.

This resolves the URL shape ``app.api.routes.products`` reserved
(``/redirect/offer/{offer_id}``) into an actual redirect to the
merchant, while logging the click — the two things a real affiliate
system needs: users end up on the merchant's page, and the platform
has a record of the click for commission reconciliation and fraud
review.

Design notes
------------
* A dedicated :class:`~app.db.models.AffiliateLink` row is
  find-or-created per offer rather than redirecting directly from the
  route handler with no persistent record — this is what makes click
  counts and "last clicked" timestamps possible, and gives future
  fraud-detection logic (unusually high click rates, etc.) something
  to query against.
* The click count increment and redirect happen in the same request,
  but the increment is not allowed to block or fail the redirect: if
  logging the click raises for any reason, the user still gets
  redirected to the merchant (a lost analytics data point is a much
  smaller problem than a broken "Buy Now" button), and the error is
  logged for investigation instead.
* This intentionally does NOT yet call a connector's
  ``generate_affiliate_links`` to re-tag the URL at click time — the
  merchant URL stored on the offer is used as-is. Applying a
  connector's own affiliate tagging scheme at import time (so the
  stored URL is already correctly tagged) is a natural extension of
  the import pipeline, not this redirect endpoint, and is left for
  that future work rather than duplicated here.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import AffiliateLink, MerchantOffer
from app.db.session import get_db_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/redirect", tags=["redirect"])


async def _get_or_create_affiliate_link(db: AsyncSession, offer: MerchantOffer) -> AffiliateLink:
    """Find this offer's existing AffiliateLink row, or create one.

    A fresh, unique ``link_token`` (used elsewhere for shareable short
    links, not needed for this redirect path itself) is generated with
    ``secrets.token_urlsafe`` rather than ``uuid4`` specifically
    because it's drawn from a cryptographically secure random source —
    appropriate for a value that will appear in public-facing URLs.
    """
    existing = await db.scalar(select(AffiliateLink).where(AffiliateLink.offer_id == offer.id))
    if existing is not None:
        return existing

    link = AffiliateLink(
        offer_id=offer.id,
        link_token=secrets.token_urlsafe(24),
        destination_url=offer.merchant_product_url,
    )
    db.add(link)
    await db.flush()
    return link


@router.get("/offer/{offer_id}")
async def redirect_to_offer(
    offer_id: uuid.UUID,
    db: AsyncSession = Depends(get_db_session),
) -> RedirectResponse:
    """Redirect to the merchant's product page for this offer, logging
    the click first.

    Returns a 404 if the offer doesn't exist or is no longer active
    (a merchant that has stopped selling a SKU shouldn't keep sending
    traffic to a dead or unrelated listing), and a 302 redirect to the
    merchant on success.
    """
    offer = await db.scalar(
        select(MerchantOffer)
        .where(MerchantOffer.id == offer_id)
        .options(selectinload(MerchantOffer.merchant))
    )
    if offer is None or not offer.is_active:
        raise HTTPException(status_code=404, detail="Offer not found or no longer available.")

    try:
        link = await _get_or_create_affiliate_link(db, offer)
        link.click_count += 1
        link.last_clicked_at = datetime.now(timezone.utc)
    except Exception:
        # Click logging must never be allowed to break the actual
        # redirect — a user clicking "Buy Now" reaching the merchant
        # is the important outcome; losing one click-count increment
        # is a recoverable, logged inconvenience, not a user-facing
        # failure. The rollback matters here: a failed flush leaves the
        # session in a state where the automatic commit that happens
        # after this request returns (see DatabaseSessionManager.session)
        # would itself raise, turning a harmless analytics miss into a
        # broken redirect. Rolling back clears that state so the request
        # can still complete cleanly.
        await db.rollback()
        logger.exception(
            "Failed to record click for offer_id=%s; redirecting anyway.", offer_id
        )

    return RedirectResponse(url=offer.merchant_product_url, status_code=302)

