"""XTTS-v2 HTTP server — OpenAI-compatible /v1/audio/speech endpoint.

Loads tts_models/multilingual/multi-dataset/xtts_v2 once on startup (~30s CPU).
Listens on 127.0.0.1:9300.
"""
from __future__ import annotations

import io
import logging
import subprocess

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from scipy.io.wavfile import write as wav_write

log = logging.getLogger("xtts-server")
logging.basicConfig(level=logging.INFO)

# Bypass Coqui TOS interactive prompt (we accept non-commercial CPML)
import os
os.environ.setdefault("COQUI_TOS_AGREED", "1")

app = FastAPI(title="xtts-v2", version="1.0.0")
tts = None  # loaded on startup


class SpeechRequest(BaseModel):
    model: str = "xtts-v2"
    voice: str = "default"
    input: str
    language: str = "auto"
    response_format: str = "wav"


_FFMPEG_FORMATS: dict[str, tuple[list[str], str]] = {
    "wav":  ([], "audio/wav"),
    "mp3":  (["-f", "mp3", "-acodec", "libmp3lame", "-b:a", "64k"], "audio/mpeg"),
    "opus": (["-f", "opus", "-acodec", "libopus", "-b:a", "32k"], "audio/opus"),
    "aac":  (["-f", "adts", "-acodec", "aac", "-b:a", "64k"], "audio/aac"),
    "flac": (["-f", "flac", "-acodec", "flac"], "audio/flac"),
    "ogg":  (["-f", "ogg", "-acodec", "libvorbis", "-q:a", "4"], "audio/ogg"),
}

_CS_CHARS = set("ěščřžýáíéůúďťňĚŠČŘŽÝÁÍÉŮÚĎŤŇ")


def detect_lang(text: str) -> str:
    if any(c in _CS_CHARS for c in text):
        return "cs"
    return "en"


def _encode_wav(wav_bytes: bytes, fmt: str) -> tuple[bytes, str]:
    if fmt not in _FFMPEG_FORMATS:
        raise ValueError(f"unsupported format: {fmt!r}")
    extra_args, mime_type = _FFMPEG_FORMATS[fmt]
    if fmt == "wav":
        return wav_bytes, mime_type
    cmd = ["ffmpeg", "-y", "-i", "pipe:0"] + extra_args + ["pipe:1"]
    result = subprocess.run(cmd, input=wav_bytes, capture_output=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg exited {result.returncode}: "
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )
    return result.stdout, mime_type


@app.on_event("startup")
def load_model():
    global tts
    log.info("Loading XTTS-v2 model (CPU, ~30s)...")
    from TTS.api import TTS as _TTS  # noqa: N811
    tts = _TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    speakers = tts.speakers or []
    log.info("XTTS-v2 loaded. Speakers: %d, Languages: %s", len(speakers), tts.languages)


@app.get("/health")
def health():
    return JSONResponse({"ok": True, "loaded": tts is not None})


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if tts is None:
        return Response(content="model not loaded", status_code=503)

    lang = req.language if req.language not in ("auto", "", None) else detect_lang(req.input)

    speakers = tts.speakers or []
    voice = req.voice or "default"
    if voice in ("default", ""):
        speaker = speakers[0] if speakers else None
    elif voice in speakers:
        speaker = voice
    else:
        # try language shortcuts: "cz"/"cs" → first Czech-friendly speaker
        speaker = speakers[0] if speakers else None
        log.warning("Voice %r not in speakers list, using %r", voice, speaker)

    log.info("Synthesizing: lang=%s speaker=%r text=%r", lang, speaker, req.input[:60])
    audio = tts.tts(text=req.input, speaker=speaker, language=lang)

    sample_rate = tts.synthesizer.output_sample_rate
    buf = io.BytesIO()
    wav_write(buf, sample_rate, np.array(audio, dtype=np.float32))
    wav_bytes = buf.getvalue()

    fmt = req.response_format or "wav"
    try:
        audio_bytes, mime_type = _encode_wav(wav_bytes, fmt)
    except ValueError as exc:
        return Response(content=str(exc), status_code=400)
    except RuntimeError as exc:
        return Response(content=str(exc), status_code=500)

    return Response(content=audio_bytes, media_type=mime_type)
