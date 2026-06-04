"""Piper TTS voice catalog — fetches and caches the rhasspy/piper-voices index."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("swapctl.piper_catalog")

_HF_VOICES_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json"
_CACHE_PATH = Path("/var/lib/swapctl/piper_catalog.json")
_CACHE_TTL_S = 24 * 3600
_VOICES_DIR = Path("/models/voices")


def _load_cache() -> tuple[list[dict], float] | None:
    """Return (catalog, fetched_at) from disk cache, or None if missing/corrupt."""
    try:
        raw = json.loads(_CACHE_PATH.read_text())
        return raw["catalog"], raw["fetched_at"]
    except Exception:
        return None


def _save_cache(catalog: list[dict]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps({"catalog": catalog, "fetched_at": time.time()}))
    except Exception as exc:
        log.warning("could not write piper catalog cache: %s", exc)


def _flatten(raw: dict[str, Any]) -> list[dict]:
    """Flatten voices.json dict into list of voice dicts."""
    out = []
    for key, v in raw.items():
        lang = v.get("language") or {}
        files_map = v.get("files") or {}
        # Only include .onnx and .onnx.json paths (skip MODEL_CARD etc.)
        file_paths = [p for p in files_map if p.endswith(".onnx") or p.endswith(".onnx.json")]
        # Check sample_rate from json file if embedded in files metadata
        # (voices.json doesn't have top-level sample_rate; load from onnx.json at runtime)
        out.append({
            "key": key,
            "name": v.get("name", ""),
            "language": lang.get("code", ""),
            "language_family": lang.get("family", ""),
            "language_name_native": lang.get("name_native", ""),
            "language_name_english": lang.get("name_english", ""),
            "language_country": lang.get("country_english", ""),
            "quality": v.get("quality", ""),
            "num_speakers": v.get("num_speakers", 1),
            "files": file_paths,
        })
    out.sort(key=lambda x: x["key"])
    return out


def _mark_installed(catalog: list[dict]) -> list[dict]:
    installed: set[str] = set()
    if _VOICES_DIR.is_dir():
        for d in _VOICES_DIR.iterdir():
            if d.is_dir() and (d / f"{d.name}.onnx").is_file() and (d / f"{d.name}.onnx.json").is_file():
                installed.add(d.name)
    for v in catalog:
        v["installed"] = v["key"] in installed
    return catalog


def fetch_catalog(force: bool = False) -> list[dict]:
    """Return flattened voice catalog, refreshing from HF if cache is stale (>24h)."""
    cached = _load_cache()
    if not force and cached is not None:
        catalog, fetched_at = cached
        if time.time() - fetched_at < _CACHE_TTL_S:
            return _mark_installed(catalog)

    try:
        resp = httpx.get(_HF_VOICES_URL, follow_redirects=True, timeout=20)
        resp.raise_for_status()
        raw = resp.json()
    except httpx.HTTPError as exc:
        log.warning("HF catalog fetch failed: %s", exc)
        if cached is not None:
            log.info("serving stale catalog from cache")
            return _mark_installed(cached[0])
        raise RuntimeError(f"Network error fetching voice catalog: {exc}") from exc

    catalog = _flatten(raw)
    _save_cache(catalog)
    return _mark_installed(catalog)
