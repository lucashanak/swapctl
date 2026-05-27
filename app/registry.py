"""JSON-backed registry of installed models."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import SwapctlError
from .config import settings


@dataclass
class ModelRecord:
    model_id: str
    repo_id: str | None
    filename: str
    mmproj: str | None
    installed_at: str
    hf_last_modified: str | None
    size_bytes: int
    ctx_size: int
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    threads: int = 0
    flash_attn: bool = True
    extra_args: str = ""
    reasoning_format: str = ""
    reasoning_effort: str = ""
    architecture: str = ""
    draft_model: str = ""
    draft_max: int = 0
    draft_min: int = 0
    spec_default: bool = False
    quant: str = ""
    sampling_temp: float = 0.8
    sampling_top_k: int = 40
    sampling_top_p: float = 0.95
    sampling_min_p: float = 0.05
    sampling_repeat_penalty: float = 1.1
    system_prompt: str = ""
    model_tags: list[str] = field(default_factory=list)
    test_results: dict = field(default_factory=dict)
    speed_history: list[dict] = field(default_factory=list)
    binary_override: str = ""
    events: list[dict] = field(default_factory=list)


def _path() -> Path:
    return settings.registry_path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_raw(path: Path | None = None) -> dict:
    target = path or _path()
    if not target.exists():
        return {"models": {}, "events": []}
    try:
        with open(target, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SwapctlError(f"registry read failed: {exc}") from exc
    if "models" not in data:
        data["models"] = {}
    if "events" not in data:
        data["events"] = []
    return data


def _write_raw(data: dict, path: Path | None = None) -> None:
    target = path or _path()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, target)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise SwapctlError(f"registry write failed: {exc}") from exc


def _record_from_dict(model_id: str, d: dict) -> ModelRecord:
    return ModelRecord(
        model_id=model_id,
        repo_id=d.get("repo_id") or None,
        filename=d.get("filename", ""),
        mmproj=d.get("mmproj"),
        installed_at=d.get("installed_at", ""),
        hf_last_modified=d.get("hf_last_modified"),
        size_bytes=int(d.get("size_bytes", 0) or 0),
        ctx_size=int(d.get("ctx_size", 0) or 0),
        cache_type_k=str(d.get("cache_type_k", "q8_0") or "q8_0"),
        cache_type_v=str(d.get("cache_type_v", "q8_0") or "q8_0"),
        threads=int(d.get("threads", 0) or 0),
        flash_attn=bool(d.get("flash_attn", True) if "flash_attn" in d else True),
        extra_args=str(d.get("extra_args", "") or ""),
        reasoning_format=str(d.get("reasoning_format", "") or ""),
        reasoning_effort=str(d.get("reasoning_effort", "") or ""),
        architecture=str(d.get("architecture", "") or ""),
        draft_model=str(d.get("draft_model", "") or ""),
        draft_max=int(d.get("draft_max", 0) or 0),
        draft_min=int(d.get("draft_min", 0) or 0),
        spec_default=bool(d.get("spec_default", False)),
        quant=str(d.get("quant", "") or ""),
        model_tags=list(d.get("model_tags") or []),
        test_results=dict(d.get("test_results") or {}),
        speed_history=list(d.get("speed_history") or []),
        events=list(d.get("events", []) or []),
    )


def get(model_id: str, path: Path | None = None) -> ModelRecord | None:
    """Return record for model_id or None."""
    data = _read_raw(path)
    entry = data["models"].get(model_id)
    if entry is None:
        return None
    return _record_from_dict(model_id, entry)


def upsert(record: ModelRecord, path: Path | None = None) -> None:
    """Insert or update a model record."""
    data = _read_raw(path)
    existed = record.model_id in data["models"]
    payload = asdict(record)
    payload.pop("model_id")
    data["models"][record.model_id] = payload
    event = {
        "ts": _now(),
        "kind": "update" if existed else "install",
        "model_id": record.model_id,
    }
    data["events"].append(event)
    data["events"] = data["events"][-50:]
    _write_raw(data, path)


def delete(model_id: str, path: Path | None = None) -> None:
    """Remove a model record. No-op if absent."""
    data = _read_raw(path)
    if model_id in data["models"]:
        del data["models"][model_id]
        data["events"].append({"ts": _now(), "kind": "remove", "model_id": model_id})
        data["events"] = data["events"][-50:]
        _write_raw(data, path)


def list_all(path: Path | None = None) -> list[ModelRecord]:
    """Return all model records."""
    data = _read_raw(path)
    return [_record_from_dict(mid, d) for mid, d in data["models"].items()]


def recent_events(limit: int = 5, path: Path | None = None) -> list[dict]:
    """Return the most recent N events (newest last)."""
    data = _read_raw(path)
    return list(data.get("events", []))[-limit:]


def get_backends_raw(path: Path | None = None) -> dict:
    """Return the raw `backends` dict from registry.json, or {} if not present."""
    data = _read_raw(path)
    return dict(data.get("backends") or {})


def set_backends_raw(data: dict, path: Path | None = None) -> None:
    """Atomically overwrite the `backends` section of registry.json."""
    raw = _read_raw(path)
    raw["backends"] = dict(data)
    _write_raw(raw, path)


def get_socket_tdp_config(path: Path | None = None) -> dict:
    """Return persisted socket-TDP config or empty dict if never set."""
    data = _read_raw(path)
    return dict(data.get("socket_tdp") or {})


def set_socket_tdp_config(
    stapm_mw: int,
    fast_mw: int,
    slow_mw: int,
    apu_mw: int,
    apply_on_boot: bool,
    stapm_time_s: int | None = None,
    path: Path | None = None,
) -> None:
    data = _read_raw(path)
    payload = {
        "stapm_mw": int(stapm_mw),
        "fast_mw": int(fast_mw),
        "slow_mw": int(slow_mw),
        "apu_mw": int(apu_mw),
        "apply_on_boot": bool(apply_on_boot),
        "updated_at": _now(),
    }
    if stapm_time_s is not None:
        payload["stapm_time_s"] = int(stapm_time_s)
    data["socket_tdp"] = payload
    _write_raw(data, path)


def _ensure_architecture(rec: "ModelRecord") -> str:
    """Return rec.architecture, parsing from GGUF and caching in registry if missing."""
    if rec.architecture:
        return rec.architecture
    try:
        from .gguf_meta import read_architecture
        path = settings.models_dir / rec.model_id / rec.filename
        arch = read_architecture(path) or ""
    except Exception:
        arch = ""
    if arch:
        rec.architecture = arch
        upsert(rec)
    return arch


__all__ = [
    "ModelRecord", "get", "upsert", "delete", "list_all", "recent_events",
    "get_backends_raw", "set_backends_raw",
    "get_socket_tdp_config", "set_socket_tdp_config",
    "_ensure_architecture",
]
