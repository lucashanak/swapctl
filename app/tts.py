"""Piper TTS integration — voice management and synthesis.

In-process synthesis via piper-tts (ONNX, no subprocess overhead).
Voices are lazy-loaded on first use and kept in a module-level dict cache.
RAM cost: ~150 MB per medium-quality voice — they stay loaded until process restart.

Falls back to subprocess piper binary if the piper-tts Python package is unavailable.
"""
from __future__ import annotations

import io
import logging
import re
import shutil
import wave
from pathlib import Path

import httpx

log = logging.getLogger("swapctl.tts")

_PIPER_BIN = Path("/opt/piper/piper/piper")
_VOICES_DIR = Path("/models/voices")
_HF_RESOLVE_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
_SAFE_KEY_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

# Module-level voice cache: voice_id -> PiperVoice instance (in-process).
# Populated lazily on first synthesize() call per voice.
_LOADED: dict[str, "object"] = {}

# Try to import piper-tts package; fall back to subprocess if unavailable.
try:
    from piper import PiperVoice as _PiperVoice  # type: ignore[import]
    _PIPER_PKG_AVAILABLE = True
except ImportError:
    _PiperVoice = None  # type: ignore[assignment,misc]
    _PIPER_PKG_AVAILABLE = False
    log.warning(
        "piper-tts Python package not found — falling back to subprocess piper binary. "
        "Install with: pip install piper-tts"
    )


def list_voices() -> list[str]:
    """Return voice IDs (dir names) that contain a matching .onnx + .onnx.json pair."""
    if not _VOICES_DIR.is_dir():
        return []
    voices = []
    for d in sorted(_VOICES_DIR.iterdir()):
        if not d.is_dir():
            continue
        onnx = d / f"{d.name}.onnx"
        meta = d / f"{d.name}.onnx.json"
        if onnx.is_file() and meta.is_file():
            voices.append(d.name)
    return voices


def _synthesize_inprocess(text: str, voice_id: str) -> bytes:
    """Synthesize using piper-tts in-process (fast path, no subprocess)."""
    if voice_id not in _LOADED:
        onnx_path = str(_VOICES_DIR / voice_id / f"{voice_id}.onnx")
        log.info("Loading piper voice %s into cache (first use)", voice_id)
        _LOADED[voice_id] = _PiperVoice.load(onnx_path)  # type: ignore[union-attr]
    voice = _LOADED[voice_id]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_writer:
        voice.synthesize_wav(text, wav_writer)  # type: ignore[union-attr]
    return buf.getvalue()


def _synthesize_subprocess(text: str, voice_id: str) -> bytes:
    """Synthesize via subprocess piper binary (slow path, ~2-3s startup per call)."""
    import subprocess
    model_path = _VOICES_DIR / voice_id / f"{voice_id}.onnx"
    result = subprocess.run(
        [str(_PIPER_BIN), "--model", str(model_path), "--output_file", "-"],
        input=text.encode("utf-8"),
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"piper exited {result.returncode}: "
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )
    return result.stdout


def synthesize(text: str, voice_id: str) -> bytes:
    """Synthesize text to WAV bytes for the given voice_id.

    Uses piper-tts in-process when available (fast; ~0.3s after first load).
    Falls back to subprocess piper binary otherwise (~2-3s per call).

    Raises ValueError for unknown voice, RuntimeError on synthesis failure.
    """
    if voice_id not in list_voices():
        raise ValueError(f"unknown voice: {voice_id!r}")
    if _PIPER_PKG_AVAILABLE:
        return _synthesize_inprocess(text, voice_id)
    return _synthesize_subprocess(text, voice_id)


def install_voice(voice_key: str) -> dict:
    """Download .onnx and .onnx.json for voice_key from HuggingFace into _VOICES_DIR.

    Returns {"voice": voice_key, "size_bytes": total_bytes}.
    Raises RuntimeError on network failure, ValueError if key is unsafe or not in catalog.
    """
    if not _SAFE_KEY_RE.match(voice_key):
        raise ValueError(f"unsafe voice key: {voice_key!r}")

    from .piper_catalog import fetch_catalog
    catalog = fetch_catalog()
    entry = next((v for v in catalog if v["key"] == voice_key), None)
    if entry is None:
        raise KeyError(f"voice not found in catalog: {voice_key!r}")

    dest_dir = _VOICES_DIR / voice_key
    dest_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    files_to_fetch = [p for p in entry["files"] if p.endswith(".onnx") or p.endswith(".onnx.json")]
    for rel_path in files_to_fetch:
        url = _HF_RESOLVE_BASE + rel_path
        filename = Path(rel_path).name
        dest_file = dest_dir / filename
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
                resp.raise_for_status()
                with dest_file.open("wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        fh.write(chunk)
            total_bytes += dest_file.stat().st_size
        except httpx.HTTPError as exc:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise RuntimeError(f"Network error downloading {filename}: {exc}") from exc

    return {"voice": voice_key, "size_bytes": total_bytes}


def remove_voice(voice_key: str) -> None:
    """Remove installed voice directory. Raises ValueError for unsafe key, FileNotFoundError if not installed."""
    if not _SAFE_KEY_RE.match(voice_key):
        raise ValueError(f"unsafe voice key: {voice_key!r}")
    voice_dir = _VOICES_DIR / voice_key
    if not voice_dir.is_dir():
        raise FileNotFoundError(f"voice not installed: {voice_key!r}")
    # Evict from in-process cache if loaded
    _LOADED.pop(voice_key, None)
    shutil.rmtree(voice_dir)


_FFMPEG_FORMATS: dict[str, tuple[list[str], str]] = {
    "wav":  ([], "audio/wav"),
    "mp3":  (["-f", "mp3", "-acodec", "libmp3lame", "-b:a", "64k"], "audio/mpeg"),
    "opus": (["-f", "opus", "-acodec", "libopus", "-b:a", "32k"], "audio/opus"),
    "aac":  (["-f", "adts", "-acodec", "aac", "-b:a", "64k"], "audio/aac"),
    "flac": (["-f", "flac", "-acodec", "flac"], "audio/flac"),
    "ogg":  (["-f", "ogg", "-acodec", "libvorbis", "-q:a", "4"], "audio/ogg"),
    "pcm":  (["-f", "s16le", "-acodec", "pcm_s16le"], "audio/L16; rate=22050; channels=1"),
}


def _encode_wav(wav_bytes: bytes, fmt: str) -> tuple[bytes, str]:
    """Convert WAV bytes to the requested format using ffmpeg.

    Returns (encoded_bytes, mime_type).  wav is a passthrough (no subprocess).
    Raises ValueError for unknown format, RuntimeError on ffmpeg failure.
    """
    import subprocess

    if fmt not in _FFMPEG_FORMATS:
        raise ValueError(f"unsupported response_format: {fmt!r}")

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


def _default_test_text(language: str) -> str:
    """Return a default test sentence for the given language family code."""
    _texts = {
        "cs": "Ahoj, jmenuju se Jirka. Toto je test českého hlasu.",
        "en": "Hello, this is a test of the English voice.",
        "de": "Hallo, dies ist ein Test der deutschen Stimme.",
        "fr": "Bonjour, ceci est un test de la voix française.",
        "es": "Hola, esta es una prueba de la voz en español.",
    }
    return _texts.get(language, _texts["en"])
