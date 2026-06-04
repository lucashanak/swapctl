"""API key management endpoints (create, list, revoke)."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..api_keys import create_key, list_keys, revoke_key

router = APIRouter(prefix="/api/keys", tags=["api-keys"])


class CreateKeyRequest(BaseModel):
    label: str


@router.get("")
def get_keys():
    """List all API keys (no raw values)."""
    return JSONResponse({"keys": list_keys()})


@router.post("")
def post_create_key(req: CreateKeyRequest):
    """Create a new API key. Returns raw key once — store it now."""
    raw, record = create_key(req.label)
    return JSONResponse({"key": raw, "record": record})


@router.delete("/{key_id}")
def delete_key(key_id: str):
    """Revoke (disable) a key by its id."""
    revoke_key(key_id)
    return JSONResponse({"ok": True, "id": key_id})
