"""
app.api.routes.admin
======================

Admin authentication and dashboard endpoints.

This is the first slice of the Admin Panel described in the
architecture: login (issuing the JWTs built in
``app.services.auth_service``), a "who am I" check, and a summary
stats endpoint (product/merchant/offer counts, registered connectors).
Deeper admin views (per-merchant import health, manual review queue
for low-confidence product matches, backup management) build on this
same authentication dependency in future route files.

Design notes
------------
* ``require_admin_user`` is a FastAPI dependency, not a decorator or
  inline check — this is what lets every future protected admin route
  simply declare ``current_user: User = Depends(require_admin_user)``
  and get consistent 401/403 behavior for free, rather than
  reimplementing token parsing in each route.
* Login intentionally returns the same generic error for "no such
  user" and "wrong password" — distinguishing the two in the response
  would let an attacker enumerate valid admin email addresses.
* The dashboard stats endpoint runs a handful of ``COUNT`` queries
  rather than loading full tables — correct at the current scale, and
  flagged in its own docstring as a candidate for a materialized-view
  or cached-counter replacement once catalog size makes live counts
  slow at the platform's 100M-product target.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.connectors.registry import get_registry
from app.core.config import get_settings
from app.db.models import MasterProduct, Merchant, MerchantOffer, User
from app.db.session import get_db_session
from app.services.auth_service import (
    InvalidTokenError,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
    verify_password,
)

router = APIRouter(prefix="/admin", tags=["admin"])
_bearer_scheme = HTTPBearer(auto_error=False)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class CurrentUserResponse(BaseModel):
    id: str
    email: str
    display_name: str | None
    is_admin: bool


class DashboardStatsResponse(BaseModel):
    total_products: int
    total_merchants: int
    total_active_offers: int
    registered_connectors: list[str]
    generated_at: datetime


async def require_admin_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db_session),
) -> User:
    """FastAPI dependency: verifies the bearer token, loads the
    corresponding user, and confirms admin status.

    Every future protected admin route depends on this rather than
    duplicating token/user lookup logic — that's what keeps 401
    ("no/invalid token") and 403 ("valid user, not an admin") behavior
    identical across every admin endpoint instead of drifting apart
    route by route.
    """
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing bearer token.")

    settings = get_settings()
    try:
        payload = decode_access_token(credentials.credentials, settings.security)
    except InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    user = await db.get(User, payload.user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="User no longer exists or is inactive.")
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required.")

    return user


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    """Authenticate with email/password and receive access + refresh
    tokens. Only succeeds for users with ``is_admin=True`` — this
    endpoint is specifically the admin login, not general user login
    (a separate, non-admin login endpoint would be added alongside
    this one if/when the platform gains customer accounts).
    """
    settings = get_settings()
    user = await db.scalar(select(User).where(User.email == body.email.lower()))

    # Deliberately generic error for both "no such user" and "wrong
    # password" -- see module docstring on why these aren't distinguished.
    invalid_credentials = HTTPException(status_code=401, detail="Invalid email or password.")

    if user is None or not user.is_active:
        raise invalid_credentials
    if not verify_password(body.password, user.hashed_password):
        raise invalid_credentials
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="This account does not have admin access.")

    return TokenResponse(
        access_token=create_access_token(user.id, user.is_admin, settings.security),
        refresh_token=create_refresh_token(user.id, user.is_admin, settings.security),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    """Exchange a valid refresh token for a new access + refresh token
    pair, without requiring the password again."""
    settings = get_settings()
    try:
        payload = decode_refresh_token(body.refresh_token, settings.security)
    except InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    user = await db.get(User, payload.user_id)
    if user is None or not user.is_active or not user.is_admin:
        raise HTTPException(status_code=401, detail="User no longer has valid admin access.")

    return TokenResponse(
        access_token=create_access_token(user.id, user.is_admin, settings.security),
        refresh_token=create_refresh_token(user.id, user.is_admin, settings.security),
    )


@router.get("/me", response_model=CurrentUserResponse)
async def get_current_user(current_user: User = Depends(require_admin_user)) -> CurrentUserResponse:
    """Return the currently authenticated admin's own profile — used
    by the admin panel's frontend to confirm a stored token is still
    valid and to display "logged in as..."."""
    return CurrentUserResponse(
        id=str(current_user.id),
        email=current_user.email,
        display_name=current_user.display_name,
        is_admin=current_user.is_admin,
    )


@router.get("/dashboard", response_model=DashboardStatsResponse)
async def get_dashboard_stats(
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(require_admin_user),
) -> DashboardStatsResponse:
    """Top-level admin dashboard numbers: catalog size, merchant
    count, active offer count, and which connectors are currently
    registered.

    Scale note: these are live ``COUNT`` queries against the primary
    database. That's correct and fast at moderate catalog sizes; at
    the platform's 100M-product target, this should be replaced with
    periodically-refreshed cached counters (e.g., updated by the
    scheduler) rather than a live count on every dashboard load — noted
    here rather than silently left for someone to discover via a slow
    page load later.
    """
    total_products = await db.scalar(select(func.count()).select_from(MasterProduct))
    total_merchants = await db.scalar(select(func.count()).select_from(Merchant))
    total_active_offers = await db.scalar(
        select(func.count()).select_from(MerchantOffer).where(MerchantOffer.is_active.is_(True))
    )

    registry = get_registry()

    return DashboardStatsResponse(
        total_products=total_products or 0,
        total_merchants=total_merchants or 0,
        total_active_offers=total_active_offers or 0,
        registered_connectors=registry.list_connector_keys(),
        generated_at=datetime.now(timezone.utc),
    )

