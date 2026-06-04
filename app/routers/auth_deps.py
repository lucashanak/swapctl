"""FastAPI dependency for optional API key authentication on audio endpoints.

If no enabled API keys are configured, all requests pass through (public mode).
If >=1 enabled key exists, requests must supply either:
  - Authorization: Bearer <key>  header, or
  - ?api_key=<key>               query param.

Session-authenticated users (browser UI) always pass regardless.
"""
from __future__ import annotations

from fastapi import HTTPException, Request

from ..api_keys import keys_required, verify_key


async def require_api_key_if_configured(request: Request) -> None:
    """FastAPI dependency: enforce API key when keys are configured.

    Pass-through conditions (no 401):
      1. No enabled keys in registry (public mode).
      2. Request has a valid session cookie (browser login).
      3. A valid API key is supplied via Bearer header or ?api_key= param.
    """
    # Public mode: no keys configured.
    if not keys_required():
        return

    # Session-authenticated browser users pass through.
    if request.session.get("authed"):
        return

    # Extract candidate key from header or query param.
    raw_key: str | None = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        raw_key = auth_header[7:].strip()
    if not raw_key:
        raw_key = request.query_params.get("api_key") or None

    if raw_key and verify_key(raw_key) is not None:
        return

    raise HTTPException(
        status_code=401,
        detail="API key required. Supply 'Authorization: Bearer <key>' or '?api_key=<key>'.",
    )
