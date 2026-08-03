"""
app.services.auth_service
============================

Password hashing and JWT token issuance/verification.

This module is the platform's authentication primitive layer — it
does not define any HTTP endpoints itself (those come later, in an
``app/api/routes/auth.py`` login/refresh router); it provides the
building blocks that layer will call: turn a password into a stored
hash, verify a password against that hash, and issue/verify signed
JWTs for access and refresh tokens.

Design notes
------------
* Password hashing uses ``passlib``'s ``CryptContext`` configured for
  bcrypt specifically (not left as "whatever passlib defaults to"),
  since bcrypt's per-hash salt and configurable work factor are the
  actual security properties being relied on here — an implicit
  default could silently change to something weaker in a future
  passlib version.
* Access and refresh tokens are deliberately separate JWTs with a
  ``token_type`` claim, rather than one token used for both purposes.
  This is what makes it possible for
  :func:`decode_access_token` to reject a refresh token presented as
  an access token (and vice versa) — a real and common token-misuse
  bug class if the two are not distinguishable from the token's own
  claims.
* Signing keys come from :class:`app.core.config.SecuritySettings`
  (``jwt_signing_key``), never hardcoded — see that module's docstring
  for why secrets are centralized there.
* All token errors funnel through :class:`InvalidTokenError`, a single
  exception type the API layer can catch once (mapping it to an HTTP
  401) rather than needing to know about every distinct way a JWT can
  be invalid (expired vs malformed vs wrong signature vs wrong type).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

import jwt
from passlib.context import CryptContext

from app.core.config import SecuritySettings

logger = logging.getLogger(__name__)

_password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"


class InvalidTokenError(Exception):
    """Raised for any reason a presented token should be rejected:
    expired, malformed, wrong signature, or wrong token type for the
    context it was used in. Callers (API dependencies) should catch
    this one type and respond with HTTP 401 rather than needing to
    distinguish the underlying cause for the client.
    """


@dataclass(frozen=True)
class TokenPayload:
    """Decoded, verified contents of a JWT issued by this module."""

    user_id: uuid.UUID
    is_admin: bool
    token_type: TokenType
    issued_at: datetime
    expires_at: datetime


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password for storage. Never store or log
    ``plain_password`` itself anywhere else in the codebase — this
    function's return value is the only form of a user's password
    that should ever reach the database."""
    return _password_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check a plaintext password against a stored hash. Returns
    ``False`` for a malformed/corrupt stored hash rather than raising,
    since that should be treated identically to "wrong password" from
    the caller's perspective — never leak *why* a login attempt
    failed to the client.
    """
    try:
        return _password_context.verify(plain_password, hashed_password)
    except (ValueError, TypeError) as exc:
        logger.warning("Password verification failed due to malformed hash: %s", exc)
        return False


def _create_token(
    *,
    user_id: uuid.UUID,
    is_admin: bool,
    token_type: TokenType,
    expires_in: timedelta,
    settings: SecuritySettings,
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "is_admin": is_admin,
        "token_type": token_type.value,
        "iat": now,
        "exp": now + expires_in,
    }
    return jwt.encode(payload, settings.jwt_signing_key.get_secret_value(), algorithm=settings.jwt_algorithm)


def create_access_token(user_id: uuid.UUID, is_admin: bool, settings: SecuritySettings) -> str:
    """Issue a short-lived access token, used to authenticate ordinary
    API requests. Lifetime comes from
    ``settings.jwt_access_token_ttl_minutes``."""
    return _create_token(
        user_id=user_id,
        is_admin=is_admin,
        token_type=TokenType.ACCESS,
        expires_in=timedelta(minutes=settings.jwt_access_token_ttl_minutes),
        settings=settings,
    )


def create_refresh_token(user_id: uuid.UUID, is_admin: bool, settings: SecuritySettings) -> str:
    """Issue a long-lived refresh token, used only to obtain new access
    tokens (never sent as a normal API auth header). Lifetime comes
    from ``settings.jwt_refresh_token_ttl_days``."""
    return _create_token(
        user_id=user_id,
        is_admin=is_admin,
        token_type=TokenType.REFRESH,
        expires_in=timedelta(days=settings.jwt_refresh_token_ttl_days),
        settings=settings,
    )


def _decode_token(token: str, settings: SecuritySettings) -> TokenPayload:
    try:
        raw = jwt.decode(token, settings.jwt_signing_key.get_secret_value(), algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise InvalidTokenError("Token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise InvalidTokenError(f"Token is invalid: {exc}") from exc

    try:
        return TokenPayload(
            user_id=uuid.UUID(raw["sub"]),
            is_admin=bool(raw["is_admin"]),
            token_type=TokenType(raw["token_type"]),
            issued_at=datetime.fromtimestamp(raw["iat"], tz=timezone.utc),
            expires_at=datetime.fromtimestamp(raw["exp"], tz=timezone.utc),
        )
    except (KeyError, ValueError) as exc:
        # A structurally valid, correctly-signed JWT that is nonetheless
        # missing/malformed expected claims — treat identically to any
        # other invalid token rather than letting a KeyError/ValueError
        # escape to a caller that only expects InvalidTokenError.
        raise InvalidTokenError(f"Token is missing or has malformed claims: {exc}") from exc


def decode_access_token(token: str, settings: SecuritySettings) -> TokenPayload:
    """Decode and verify a token, requiring it to be an access token.
    Raises :class:`InvalidTokenError` if it's structurally invalid,
    expired, or is actually a refresh token presented in the wrong
    context."""
    payload = _decode_token(token, settings)
    if payload.token_type is not TokenType.ACCESS:
        raise InvalidTokenError("Expected an access token but received a different token type.")
    return payload


def decode_refresh_token(token: str, settings: SecuritySettings) -> TokenPayload:
    """Decode and verify a token, requiring it to be a refresh token."""
    payload = _decode_token(token, settings)
    if payload.token_type is not TokenType.REFRESH:
        raise InvalidTokenError("Expected a refresh token but received a different token type.")
    return payload

