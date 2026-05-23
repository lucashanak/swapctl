"""Background snapshot worker.

Periodically pre-computes data needed by /testing (Models page) and other read-heavy
views, so request handlers can serve from a hot in-memory snapshot instead of doing
GGUF parses, file reads, and process introspection per-request.

Snapshot freshness:
- light data (registry, swap-running, backend assignments, gguf max_ctx, tags): refreshed every 5 s
- HF update info (bulk_repo_last_modified): on-demand only (user clicks Refresh) — too slow to schedule

External callers:
- get_snapshot() — read latest. Always returns a dict (empty if worker not yet started).
- invalidate_now() — wake the worker to refresh ASAP (after install/delete/backend switch).
- start_worker() — call once at FastAPI startup.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

log = logging.getLogger("swapctl.background")

_SNAPSHOT: dict[str, Any] = {
    "records": [],          # list[ModelRecord]
    "max_ctx": {},          # model_id -> int|None
    "tags": {},             # model_id -> list[str]
    "vram": {},             # model_id -> {"vram_mb": int, "vram_estimates": dict}
    "swap_running": [],     # list[str] of model_ids
    "all_backends": {},     # model_id -> backend_id (str)
    "collect_status": {},   # full _collect_status() output
    "ts": 0.0,
}
_LOCK = threading.Lock()
_WAKE = threading.Event()
_INTERVAL_S = 5.0
_STARTED = False


def get_snapshot() -> dict[str, Any]:
    """Return current snapshot (shallow copy of dict). Empty if worker hasn't run."""
    with _LOCK:
        return dict(_SNAPSHOT)


def invalidate_now() -> None:
    """Wake worker so next refresh starts immediately."""
    _WAKE.set()


def _build_snapshot() -> dict[str, Any]:
    """Compute fresh snapshot. May raise — callers handle."""
    from . import registry, swap_config, system
    from .routers.views import _model_max_ctx, _model_tags  # reuse mtime+size cache

    records = registry.list_all()
    try:
        running = system.swap_running_models()
    except Exception:
        running = []
    try:
        backends = swap_config.detect_all_backends()
    except Exception:
        backends = {}

    from . import gguf_meta
    from .config import settings as cfg

    max_ctx = {}
    tags = {}
    vram = {}
    for r in records:
        try:
            max_ctx[r.model_id] = _model_max_ctx(r.model_id)
        except Exception:
            max_ctx[r.model_id] = None
        try:
            tags[r.model_id] = _model_tags(r.model_id)
        except Exception:
            tags[r.model_id] = []
        try:
            model_dir = cfg.models_dir / r.model_id
            gguf_files = sorted(p for p in model_dir.glob("*.gguf") if not p.name.startswith("mmproj"))
            if not gguf_files:
                gguf_files = sorted(p for p in model_dir.rglob("*.gguf") if not p.name.startswith("mmproj"))
            if gguf_files:
                ctx = r.ctx_size or 8192
                vr = gguf_meta.estimate_vram(gguf_files[0], context_sizes=[ctx])
                if vr and vr.get("estimates"):
                    vram[r.model_id] = {
                        "vram_mb": vr["estimates"][0]["total_mb"],
                        "vram_estimates": vr,
                    }
        except Exception:
            pass

    try:
        from .routers.views import _collect_status
        collect_status = _collect_status()
    except Exception:
        collect_status = {}

    return {
        "records": records,
        "max_ctx": max_ctx,
        "tags": tags,
        "vram": vram,
        "swap_running": running,
        "all_backends": backends,
        "collect_status": collect_status,
        "ts": time.time(),
    }


def _refresh_loop() -> None:
    while True:
        try:
            snap = _build_snapshot()
            with _LOCK:
                _SNAPSHOT.update(snap)
        except Exception as exc:
            log.warning("snapshot build failed: %s", exc)
        _WAKE.wait(timeout=_INTERVAL_S)
        _WAKE.clear()


def start_worker() -> None:
    """Spawn the background refresh daemon. Idempotent."""
    global _STARTED
    if _STARTED:
        return
    _STARTED = True
    t = threading.Thread(target=_refresh_loop, name="swapctl-snapshot", daemon=True)
    t.start()
    log.info("snapshot worker started (interval=%ss)", _INTERVAL_S)
