"""
Admin authentication for the Monitor dashboard.

The Monitor page shows operational data (feedback stats, retrain history,
endpoint traffic) that is meant for the team, not for visitors. This module
gives the API a minimal login: credentials are checked against environment
variables and a signed, expiring token is handed back. The frontend stores
that token and shows the dashboard only while it is valid.

Design notes:
    - No database and no new dependency. Tokens are "<expiry>.<signature>"
      where the signature is an HMAC-SHA256 over the expiry timestamp.
      Verifying = recompute the signature and check the clock.
    - The signing secret defaults to a random value per process, so tokens
      naturally die when the server restarts. Set ADMIN_TOKEN_SECRET if you
      want logins to survive restarts.
    - Comparisons use hmac.compare_digest to avoid timing leaks.

Environment variables:
    ADMIN_USERNAME       login name           (default: "admin")
    ADMIN_PASSWORD       login password       (required; no default)
    ADMIN_TOKEN_SECRET   token signing key    (default: random per process)
    ADMIN_TOKEN_TTL_H    login lifetime hours (default: 8)

Set these env vars in Portainer or docker-compose. When ADMIN_PASSWORD is not
set, a random password is generated per process so no usable credential is
baked into the repo; set ADMIN_PASSWORD explicitly to log in.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time

# Resolved once at import. A restart picks up new env values.
_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
# No hardcoded default: if ADMIN_PASSWORD is unset, use a random per-process
# value so the repo never ships a working login credential.
_PASSWORD = os.getenv("ADMIN_PASSWORD") or secrets.token_hex(16)
_SECRET = (os.getenv("ADMIN_TOKEN_SECRET") or secrets.token_hex(32)).encode()
_TTL_SECONDS = int(float(os.getenv("ADMIN_TOKEN_TTL_H", "8")) * 3600)


def check_credentials(username: str, password: str) -> bool:
    """True if the given username/password match the configured admin."""
    user_ok = hmac.compare_digest(username.encode(), _USERNAME.encode())
    pass_ok = hmac.compare_digest(password.encode(), _PASSWORD.encode())
    return user_ok and pass_ok


def create_token() -> tuple[str, int]:
    """
    Issue a signed admin token.

    Returns:
        (token, expires_at) where expires_at is a unix timestamp.
    """
    expires_at = int(time.time()) + _TTL_SECONDS
    payload = str(expires_at)
    signature = hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}", expires_at


def verify_token(token: str) -> bool:
    """True if the token's signature is ours and it has not expired."""
    if not token or "." not in token:
        return False
    payload, _, signature = token.partition(".")
    if not payload.isdigit():
        return False
    expected = hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return False
    return int(payload) > time.time()
