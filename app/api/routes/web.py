"""
app.api.routes.web
====================

The public-facing HTML shop page — what an actual visitor sees when
they open the site, as opposed to the JSON API meant for programmatic
clients (``app.api.routes.products``).

Design notes
------------
* Deliberately reuses ``_variant_detail_query`` and
  ``_build_price_comparison`` from ``app.api.routes.products`` rather
  than reimplementing the same eager-loading strategy and price-math
  here. Both routes need to answer the same question ("what's the
  lowest price and how many merchants sell this") — duplicating that
  logic would risk the two pages silently disagreeing with each other
  over time as one gets updated and the other doesn't.
* Rendered with Jinja2 (via Starlette's ``Jinja2Templates``), which
  auto-escapes all interpolated values by default — this matters here
  specifically because product titles/descriptions originate from
  merchant feeds, which are untrusted external input; auto-escaping is
  what prevents a malicious feed from injecting a `<script>` tag into
  every visitor's browser.
"""

from __future__ import annotations

import math
from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.products import _build_price_comparison, _variant_detail_query
from app.core.config import get_settings
from app.db.models import MasterProduct
from app.db.session import get_db_session

router = APIRouter(tags=["web"])

_templates_dir = Path(__file__).resolve().parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))
# Explicitly force autoescaping on rather than relying on an assumed
# library default. This is a real security control: product titles and
# descriptions originate from merchant feeds, which are untrusted input
# — without autoescaping, a feed containing e.g. a <script> tag in a
# product title would execute in every visitor's browser.
templates.env.autoescape = True


@router.get("/shop", response_class=HTMLResponse)
async def shop_page(
    request: Request,
    page: int = Query(default=1, ge=1),
    db: AsyncSession = Depends(get_db_session),
) -> HTMLResponse:
    """Render the public shop homepage: a paginated grid of products
    with their lowest current price and merchant count.

    Shares its data-loading and price-comparison logic with the JSON
    ``GET /products`` endpoint (see module docstring) — this route's
    own job is purely presentation.
    """
    page_size = 24
    settings = get_settings()

    total = await db.scalar(select(func.count()).select_from(MasterProduct))

    query = (
        select(MasterProduct)
        .options(*_variant_detail_query())
        .order_by(MasterProduct.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    products = (await db.execute(query)).scalars().unique().all()

    items = []
    for product in products:
        comparison = _build_price_comparison(list(product.variants))
        items.append(
            {
                "id": product.id,
                "title": product.title,
                "brand_name": product.brand.name if product.brand else None,
                "canonical_image_url": product.canonical_image_url,
                "lowest_total_price": comparison.lowest_total_price if comparison else None,
                "currency_code": comparison.currency_code if comparison else None,
                "merchant_count": comparison.merchant_count if comparison else 0,
            }
        )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": settings.app_name,
            "items": items,
            "total": total or 0,
            "page": page,
            "total_pages": max(1, math.ceil((total or 0) / page_size)),
        },
    )
