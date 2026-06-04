"""API key management for swapctl audio endpoints.

Keys are stored as sha256 hashes in the registry JSON under `api_keys`.
Raw keys are shown exactly once on creation; only the hash is persisted.

Key format: sk-<32 random url-safe chars>

If no enabled keys exist, endpoints are public (backward-compatible).
If >=1 enabled key exists, endpoints require Authorization: Bearer <key>
or ?api_key=<key> query param.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone

from .registry import _now, _read_raw, _write_raw


def _sha256(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_keys(data: dict) -> list[dict]:
    return list(data.get("api_keys") or [])


def list_keys() -> list[dict]:
    """Return all key records (no raw value), sorted by created_at desc."""
    data = _read_raw()
    keys = _get_keys(data)
    return sorted(keys, key=lambda k: k.get("created_at", ""), reverse=True)


def create_key(label: str) -> tuple[str, dict]:
    """Generate a new API key.

    Returns (raw_key, record).  raw_key is the only time the plaintext is
    available — only the sha256 hash is stored.
    """
    raw = "sk-" + secrets.token_urlsafe(32)
    record: dict = {
        "id": secrets.token_hex(16),
        "label": label,
        "key_hash": _sha256(raw),
        "created_at": _now(),
        "last_used_at": None,
        "enabled": True,
    }
    data = _read_raw()
    keys = _get_keys(data)
    keys.append(record)
    data["api_keys"] = keys
    _write_raw(data)
    return raw, record


def revoke_key(key_id: str) -> None:
    """Set enabled=False for the key with matching id. No-op if not found."""
    data = _read_raw()
    keys = _get_keys(data)
    for k in keys:
        if k.get("id") == key_id:
            k["enabled"] = False
            break
    data["api_keys"] = keys
    _write_raw(data)


def verify_key(raw_key: str) -> dict | None:
    """Check raw_key against stored hashes; update last_used_at on match.

    Returns the matching record or None.
    """
    h = _sha256(raw_key)
    data = _read_raw()
    keys = _get_keys(data)
    matched = None
    for k in keys:
        if k.get("enabled") and k.get("key_hash") == h:
            matched = k
            break
    if matched is None:
        return None
    matched["last_used_at"] = _now()
    data["api_keys"] = keys
    _write_raw(data)
    return matched


def keys_required() -> bool:
    """True if at least one enabled key exists (auth enforcement is active)."""
    data = _read_raw()
    return any(k.get("enabled") for k in _get_keys(data))


__all__ = [
    "list_keys", "create_key", "revoke_key", "verify_key", "keys_required",
]
