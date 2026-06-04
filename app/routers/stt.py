"""OpenAI-compatible STT (speech-to-text) endpoint — proxies to whisper-server via llama-swap."""
from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..config import settings
from .auth_deps import require_api_key_if_configured

router = APIRouter(prefix="/v1/audio", tags=["stt"])


@router.post("/transcriptions")
async def transcriptions(
    model: str = Form(...),
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    response_format: str | None = Form(default=None),
    _auth=Depends(require_api_key_if_configured),
):
    """Proxy OpenAI-compat STT request to llama-swap whisper-server."""
    audio_bytes = await file.read()
    files = {"file": (file.filename or "audio.wav", audio_bytes, file.content_type or "audio/wav")}
    data: dict[str, str] = {"model": model}
    if language:
        data["language"] = language
    if response_format:
        data["response_format"] = response_format
    swap = settings.swap_endpoint.rstrip("/")
    try:
        with httpx.Client(timeout=300.0) as c:
            r = c.post(f"{swap}/v1/audio/transcriptions", files=files, data=data)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"whisper-server unreachable: {exc}") from exc
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return JSONResponse(r.json())
