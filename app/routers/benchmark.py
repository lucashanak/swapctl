"""Benchmark page router.

GET  /benchmark                    — HTML page
POST /api/benchmark/run            — JSON body; runs benchmark synchronously
POST /api/benchmark/apply-best/{model_id} — switch model to best backend from history
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ..config import settings
from .. import registry, swap_config, system

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals["root_path"] = settings.root_path
try:
    from .. import github_watch
    templates.env.globals["gh_watch"] = github_watch.load_state
except Exception:
    pass

router = APIRouter(tags=["benchmark"])

log = logging.getLogger("swapctl.benchmark_router")

PREFIX = settings.root_path


# ---------------------------------------------------------------------------
# HTML view
# ---------------------------------------------------------------------------

@router.get("/benchmark", response_class=HTMLResponse)
def benchmark_page(request: Request, model: str | None = None, auto: int = 0):
    from .. import registry

    models = [r.model_id for r in registry.list_all()]
    prefill = model if model in models else ""
    return templates.TemplateResponse(
        request,
        "benchmark.html",
        {
            "active_tab": "benchmark",
            "models": models,
            "prefill_model": prefill,
            "auto_trigger": bool(auto and prefill),
        },
    )


# ---------------------------------------------------------------------------
# API — run benchmark
# ---------------------------------------------------------------------------

class BenchmarkRunRequest(BaseModel):
    model_id: str
    backend_ids: list[str] = []
    mode: Literal["auto", "manual"] = "manual"


@router.post("/api/benchmark/run", status_code=202)
def benchmark_run(req: BenchmarkRunRequest):
    """Start a benchmark in background thread. Returns immediately; client polls /api/benchmark/status."""
    from ..benchmark import auto_bench_backends, start_benchmark_async

    if not req.model_id:
        raise HTTPException(status_code=400, detail="model_id is required")

    if req.mode == "auto":
        backend_ids = auto_bench_backends(req.model_id)
    else:
        backend_ids = req.backend_ids
        if not backend_ids:
            raise HTTPException(status_code=400, detail="at least one backend_id is required")

    started = start_benchmark_async(req.model_id, backend_ids)
    if not started:
        raise HTTPException(status_code=409, detail="A benchmark is already running")

    return JSONResponse({
        "ok": True,
        "started": True,
        "model_id": req.model_id,
        "backend_ids": backend_ids,
        "poll_url": "/api/benchmark/status",
    })


# ---------------------------------------------------------------------------
# API — apply best backend
# ---------------------------------------------------------------------------

class ApplyBestRequest(BaseModel):
    backend_id: str | None = None


@router.post("/api/benchmark/apply-best/{model_id}")
def benchmark_apply_best(model_id: str, req: ApplyBestRequest = ApplyBestRequest()):
    rec = registry.get(model_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"model not found: {model_id}")

    chosen = req.backend_id
    if not chosen:
        # Compute winner from speed_history: highest gen_tps among latest GEN records per backend
        history = rec.speed_history or []
        # Collect latest gen record per backend
        latest_gen: dict[str, dict] = {}
        for entry in history:
            if entry.get("test") != "gen":
                continue
            bid = entry.get("backend_id", "")
            if not bid:
                continue
            # Later entries in the list are newer
            latest_gen[bid] = entry

        if not latest_gen:
            raise HTTPException(
                status_code=400,
                detail="no gen speed history available; run a benchmark first",
            )

        best_backend = max(latest_gen, key=lambda b: latest_gen[b].get("gen_tps") or 0)
        chosen = best_backend

    try:
        swap_config.switch_model_backend(model_id, chosen)
        system.restart_swap()
        system.wait_swap_healthy(timeout_s=60.0)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse({"ok": True, "model_id": model_id, "backend_id": chosen})


# ---------------------------------------------------------------------------
# API — force release stale benchmark lock
# ---------------------------------------------------------------------------

@router.get("/api/benchmark/status")
def benchmark_status():
    """Return live state of any running benchmark."""
    from ..benchmark import bench_state
    return JSONResponse(bench_state())


@router.get("/api/benchmark/history/{model_id}")
def benchmark_history(model_id: str):
    """Return persisted speed_history for a model."""
    rec = registry.get(model_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"model not found: {model_id}")
    return JSONResponse({"model_id": model_id, "history": list(rec.speed_history or [])})


@router.post("/api/benchmark/force-release")
def benchmark_force_release():
    """Force-release a stale benchmark lock and reset live state."""
    from ..benchmark import _BENCHMARK_LOCK, _BENCH_STATE

    released = False
    try:
        _BENCHMARK_LOCK.release()
        released = True
    except RuntimeError:
        released = False  # lock wasn't held
    _BENCH_STATE.update({"running": False, "stage": "released", "current_backend": None})
    return JSONResponse({"ok": True, "released": released})
