"""OpenAI-compatible TTS endpoints + voice catalog management."""
from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .. import tts
from ..config import settings
from .auth_deps import require_api_key_if_configured

log = logging.getLogger("swapctl.tts_router")

router = APIRouter(prefix="/v1/audio", tags=["tts"])

_XTTS_MODEL_ID = "xtts-v2"


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    voice: str
    input: str
    response_format: str = "wav"
    language: str | None = None


class InstallRequest(BaseModel):
    voice: str


# ── public endpoints (no auth — openwebui compat) ─────────────────────

@router.get("/voices")
def get_voices():
    return JSONResponse({"voices": tts.list_voices()})


@router.post("/speech")
def post_speech(req: SpeechRequest, _auth=Depends(require_api_key_if_configured)):
    fmt = req.response_format or "wav"

    # Route voices prefixed "xtts:" to the XTTS-v2 service via llama-swap
    if req.voice.startswith("xtts:"):
        xtts_voice = req.voice[5:] or "default"
        xtts_url = f"{settings.swap_endpoint}/v1/audio/speech"
        payload = {
            "model": _XTTS_MODEL_ID,
            "voice": xtts_voice,
            "input": req.input,
            "language": req.language or "auto",
            "response_format": fmt,
        }
        try:
            resp = httpx.post(xtts_url, json=payload, timeout=120.0)
            resp.raise_for_status()
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail=f"XTTS timeout: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"XTTS error {exc.response.status_code}: {exc.response.text[:200]}",
            ) from exc
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"XTTS unreachable: {exc}") from exc
        content_type = resp.headers.get("content-type", "audio/wav")
        return Response(content=resp.content, media_type=content_type)

    # Piper path (unchanged)
    try:
        wav_bytes = tts.synthesize(req.input, req.voice)
        audio_bytes, mime_type = tts._encode_wav(wav_bytes, fmt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(content=audio_bytes, media_type=mime_type)


# ── auth-protected endpoints ──────────────────────────────────────────

@router.get("/catalog")
def get_catalog(language: str | None = Query(default=None)):
    """Return full voice catalog with installed flag. Optionally filter by language family."""
    try:
        from ..piper_catalog import fetch_catalog
        catalog = fetch_catalog()
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if language:
        catalog = [v for v in catalog if v.get("language_family") == language or v.get("language", "").startswith(language)]
    return JSONResponse({"voices": catalog, "count": len(catalog)})


@router.post("/install")
def install_voice(req: InstallRequest):
    """Download and install a voice. 404 if not in catalog, 409 if already installed."""
    voice_key = req.voice
    if voice_key in tts.list_voices():
        raise HTTPException(status_code=409, detail=f"voice already installed: {voice_key!r}")
    try:
        result = tts.install_voice(voice_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return JSONResponse(result)


@router.delete("/voices/{voice_key}")
def delete_voice(voice_key: str):
    """Remove an installed voice."""
    try:
        tts.remove_voice(voice_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JSONResponse({"ok": True, "voice": voice_key})
