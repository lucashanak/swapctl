"""JSON + HTMX partial API endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

from .. import ConfigError, SwapctlError
from ..config import settings

PREFIX = settings.root_path
from ..validation import (
    valid_ctx_size,
    valid_filename,
    valid_model_id,
    valid_repo_id,
)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals["root_path"] = settings.root_path

router = APIRouter(prefix="/api", tags=["api"])


def _wants_html(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/html" in accept.lower()


def _is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request", "").lower() == "true"


@router.post("/install")
async def install(
    request: Request,
    repo_id: str = Form(...),
    filename: str = Form(...),
    model_id: str = Form(...),
    ctx_size: int = Form(default=65536),
    mmproj_filename: str | None = Form(default=None),
):
    from .. import downloader

    try:
        repo_id = valid_repo_id(repo_id)
        filename = valid_filename(filename)
        model_id = valid_model_id(model_id)
        ctx_size = valid_ctx_size(int(ctx_size))
        if mmproj_filename:
            mmproj_filename = valid_filename(mmproj_filename)
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        job_id = downloader.start_install(
            repo_id=repo_id,
            filename=filename,
            mmproj_filename=mmproj_filename or None,
            model_id=model_id,
            ctx_size=ctx_size,
        )
    except SwapctlError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Redirect": f"{PREFIX}/jobs/{job_id}"})
    return JSONResponse({"job_id": job_id})


@router.get("/jobs/{job_id}")
def job_status(request: Request, job_id: str):
    from .. import downloader

    status = downloader.get_status(job_id)
    if not status or status.get("status") == "error" and status.get("error") == "unknown job":
        raise HTTPException(status_code=404, detail="job not found")

    bytes_done = int(status.get("bytes_done", 0) or 0)
    bytes_total = int(status.get("bytes_total", 0) or 0)
    if bytes_done > bytes_total and bytes_total > 0:
        bytes_done = bytes_total
    percent = min(100, int((bytes_done / bytes_total) * 100)) if bytes_total > 0 else 0
    state = status.get("status", "running")

    if _wants_html(request):
        response = templates.TemplateResponse(
            request,
            "partials/job_status.html",
            {
                "job_id": job_id,
                "status": state,
                "percent": percent,
                "bytes_done": bytes_done,
                "bytes_total": bytes_total,
                "error": status.get("error"),
            },
        )
        if state == "done":
            response.headers["HX-Redirect"] = f"{PREFIX}/testing"
        return response

    payload = {
        "job_id": job_id,
        "status": state,
        "percent": percent,
        "bytes_done": bytes_done,
        "bytes_total": bytes_total,
    }
    if status.get("error"):
        payload["error"] = status["error"]
    return JSONResponse(payload)


@router.post("/remove/{model_id}")
def remove_model(request: Request, model_id: str):
    from .. import downloader

    try:
        model_id = valid_model_id(model_id)
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        downloader.remove_model(model_id)
    except SwapctlError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if _is_htmx(request):
        return Response(status_code=200, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True})


@router.post("/load/{model_id}")
def load_model(request: Request, model_id: str):
    import httpx

    try:
        model_id = valid_model_id(model_id)
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    url = settings.swap_endpoint.rstrip("/") + "/v1/chat/completions"
    try:
        resp = httpx.post(
            url,
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            timeout=120,
        )
        resp.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if _is_htmx(request):
        return Response(status_code=200, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True})


_VALID_CACHE_TYPES = {"f16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}


def _resolve_draft_gguf(name: str) -> Path | None:
    """Find a GGUF file for a draft model name. Searches /models/<name>/*.gguf."""
    model_dir = settings.models_dir / name
    if not model_dir.is_dir():
        return None
    for f in sorted(model_dir.iterdir()):
        if f.suffix == ".gguf" and f.is_file():
            return f
    return None


def _list_draft_candidates() -> list[dict]:
    """Return [{name, path, size_mb}] for all model dirs that contain a GGUF."""
    candidates = []
    if not settings.models_dir.is_dir():
        return candidates
    for d in sorted(settings.models_dir.iterdir()):
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix == ".gguf" and f.is_file():
                candidates.append({
                    "name": d.name,
                    "path": str(f),
                    "size_mb": round(f.stat().st_size / 1024**2),
                })
                break
    return candidates


@router.post("/edit/{model_id}")
def edit_model(
    request: Request,
    model_id: str,
    repo_id: str = Form(""),
    ctx_size: int = Form(0),
    cache_type_k: str = Form(""),
    cache_type_v: str = Form(""),
    threads: int = Form(-1),
    flash_attn: str = Form(""),
    extra_args: str = Form(""),
    reasoning_format: str = Form(""),
    reasoning_effort: str = Form(""),
    draft_model: str = Form(""),
    draft_max: int = Form(0),
    draft_min: int = Form(0),
    spec_default: str = Form(""),
    sampling_temp: float = Form(-1),
    sampling_top_k: int = Form(-1),
    sampling_top_p: float = Form(-1),
    sampling_min_p: float = Form(-1),
    sampling_repeat_penalty: float = Form(-1),
    system_prompt: str = Form(""),
):
    from .. import hf_client, registry, swap_config, system

    try:
        model_id = valid_model_id(model_id)
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rec = registry.get(model_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"unknown model: {model_id}")

    new_repo = repo_id.strip() or None
    if new_repo:
        try:
            valid_repo_id(new_repo)
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            lm = hf_client.repo_last_modified(new_repo).isoformat()
        except SwapctlError as exc:
            raise HTTPException(status_code=400, detail=f"HF check failed: {exc}") from exc
        rec.repo_id = new_repo
        rec.hf_last_modified = lm
    else:
        rec.repo_id = None
        rec.hf_last_modified = None

    flags_to_update: dict = {}

    if ctx_size and ctx_size != rec.ctx_size:
        try:
            valid_ctx_size(int(ctx_size))
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        flags_to_update["ctx_size"] = int(ctx_size)
        rec.ctx_size = int(ctx_size)

    if cache_type_k and cache_type_k != rec.cache_type_k:
        if cache_type_k not in _VALID_CACHE_TYPES:
            raise HTTPException(status_code=400, detail=f"invalid cache_type_k: {cache_type_k}")
        flags_to_update["cache_type_k"] = cache_type_k
        rec.cache_type_k = cache_type_k

    if cache_type_v and cache_type_v != rec.cache_type_v:
        if cache_type_v not in _VALID_CACHE_TYPES:
            raise HTTPException(status_code=400, detail=f"invalid cache_type_v: {cache_type_v}")
        flags_to_update["cache_type_v"] = cache_type_v
        rec.cache_type_v = cache_type_v

    if threads >= 0 and threads != rec.threads:
        flags_to_update["threads"] = threads
        rec.threads = threads

    if flash_attn in ("1", "0"):
        fa = flash_attn == "1"
        if fa != rec.flash_attn:
            flags_to_update["flash_attn"] = fa
            rec.flash_attn = fa

    if extra_args != (rec.extra_args or ""):
        flags_to_update["extra_args"] = extra_args
        rec.extra_args = extra_args

    _VALID_REASONING = {"", "deepseek"}
    if reasoning_format not in _VALID_REASONING:
        raise HTTPException(status_code=400, detail=f"invalid reasoning_format: {reasoning_format}")
    if reasoning_format != (rec.reasoning_format or ""):
        flags_to_update["reasoning_format"] = reasoning_format
        rec.reasoning_format = reasoning_format

    _VALID_EFFORT = {"", "low", "medium", "high"}
    if reasoning_effort not in _VALID_EFFORT:
        raise HTTPException(status_code=400, detail=f"invalid reasoning_effort: {reasoning_effort}")
    if reasoning_effort != (rec.reasoning_effort or ""):
        rec.reasoning_effort = reasoning_effort

    sd = spec_default == "1"
    if sd != rec.spec_default:
        flags_to_update["spec_default"] = sd
        rec.spec_default = sd

    draft_model = draft_model.strip()
    if draft_model != (rec.draft_model or ""):
        if draft_model:
            draft_path = _resolve_draft_gguf(draft_model)
            if not draft_path:
                raise HTTPException(status_code=400, detail=f"draft model GGUF not found: {draft_model}")
            flags_to_update["draft_model"] = str(draft_path)
        else:
            flags_to_update["draft_model"] = ""
        rec.draft_model = draft_model
    if draft_max != (rec.draft_max or 0):
        flags_to_update["draft_max"] = draft_max
        rec.draft_max = draft_max
    if draft_min != (rec.draft_min or 0):
        flags_to_update["draft_min"] = draft_min
        rec.draft_min = draft_min

    if sampling_temp >= 0 and sampling_temp != rec.sampling_temp:
        flags_to_update["sampling_temp"] = sampling_temp
        rec.sampling_temp = sampling_temp
    if sampling_top_k >= 0 and sampling_top_k != rec.sampling_top_k:
        flags_to_update["sampling_top_k"] = sampling_top_k
        rec.sampling_top_k = sampling_top_k
    if sampling_top_p >= 0 and sampling_top_p != rec.sampling_top_p:
        flags_to_update["sampling_top_p"] = sampling_top_p
        rec.sampling_top_p = sampling_top_p
    if sampling_min_p >= 0 and sampling_min_p != rec.sampling_min_p:
        flags_to_update["sampling_min_p"] = sampling_min_p
        rec.sampling_min_p = sampling_min_p
    if sampling_repeat_penalty >= 0 and sampling_repeat_penalty != rec.sampling_repeat_penalty:
        flags_to_update["sampling_repeat_penalty"] = sampling_repeat_penalty
        rec.sampling_repeat_penalty = sampling_repeat_penalty

    sys_prompt = system_prompt.strip()
    if sys_prompt != (rec.system_prompt or ""):
        rec.system_prompt = sys_prompt

    if flags_to_update:
        bak = swap_config.backup()
        try:
            swap_config.update_model_flags(model_id, **flags_to_update)
        except (ConfigError, SwapctlError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            system.restart_swap()
            if not system.wait_swap_healthy(timeout_s=15.0):
                raise SwapctlError("llama-swap did not come up after config change")
        except SwapctlError as exc:
            try:
                swap_config.restore_backup()
                system.restart_swap()
                system.wait_swap_healthy(timeout_s=15.0)
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"{exc}; reverted") from exc

    registry.upsert(rec)

    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse(
        {"ok": True, "repo_id": rec.repo_id, "hf_last_modified": rec.hf_last_modified, "ctx_size": rec.ctx_size}
    )


@router.post("/jobs/clear-terminated")
async def clear_terminated_jobs(request: Request):
    from .. import downloader

    n = downloader.clear_terminal_jobs()
    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True, "removed": n})


@router.post("/jobs/{job_id}/clear")
async def clear_one_job(request: Request, job_id: str):
    from .. import downloader

    try:
        removed = downloader.clear_job(job_id)
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="job not found")
    if _is_htmx(request):
        return Response(status_code=204)
    return JSONResponse({"ok": True})


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str):
    from .. import downloader

    ok = downloader.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail="job not running")
    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True})


@router.post("/jobs/{job_id}/pause")
async def pause_job(request: Request, job_id: str):
    from .. import downloader

    ok = downloader.pause_job(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail="job not running")
    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True})


@router.post("/jobs/{job_id}/resume")
async def resume_job(request: Request, job_id: str):
    from .. import downloader

    try:
        resumed_id = downloader.resume_job(job_id)
    except (ConfigError, SwapctlError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True, "job_id": resumed_id})


@router.post("/update/{model_id}")
async def update_model(request: Request, model_id: str):
    from .. import downloader

    try:
        model_id = valid_model_id(model_id)
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        job_id = downloader.start_update(model_id)
    except SwapctlError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Redirect": f"{PREFIX}/jobs/{job_id}"})
    return JSONResponse({"job_id": job_id})


@router.post("/swap-config")
def update_swap_config(
    request: Request,
    keepWarm: str = Form("1"),
    healthCheckTimeout: int = Form(300),
    priorities_json: str = Form("{}"),
    enabled_json: str = Form("{}"),
    max_group_size: int = Form(3),
):
    import json as _json

    from .. import swap_config, system

    try:
        priorities = dict(_json.loads(priorities_json) or {})
        enabled = {str(k): bool(v) for k, v in dict(_json.loads(enabled_json) or {}).items()}
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid payload: {exc}") from exc

    payload = {
        "keepWarm": keepWarm not in ("0", "false", "False", ""),
        "healthCheckTimeout": int(healthCheckTimeout),
        "priorities": priorities,
        "enabled": enabled,
        "max_group_size": max_group_size,
    }

    bak = None
    try:
        bak = swap_config.backup()
        swap_config.set_simple_config(payload)
    except (ConfigError, SwapctlError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _rollback(reason: str):
        try:
            swap_config.restore_backup()
            system.restart_swap()
            system.wait_swap_healthy(timeout_s=15.0)
        except Exception:
            pass
        return HTTPException(
            status_code=500, detail=f"{reason}; previous config restored"
        )

    try:
        system.restart_swap()
    except SwapctlError as exc:
        if bak is not None:
            raise _rollback(f"restart command failed: {exc}") from exc
        raise HTTPException(status_code=500, detail=f"restart failed: {exc}") from exc

    if not system.wait_swap_healthy(timeout_s=15.0):
        raise _rollback("llama-swap did not become healthy within 15s")

    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Redirect": f"{PREFIX}/swap-config"})
    return JSONResponse({"ok": True})


@router.post("/backend")
def switch_all_backends(
    request: Request,
    backend: str = Form(...),
):
    from ..config import _VALID_BACKENDS
    from .. import swap_config, system

    if backend not in _VALID_BACKENDS:
        raise HTTPException(status_code=400, detail=f"invalid backend: {backend!r}, must be one of {_VALID_BACKENDS}")

    bak = swap_config.backup()
    try:
        count = swap_config.switch_backend(backend)
    except (ConfigError, SwapctlError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if count > 0:
        try:
            system.restart_swap()
            if not system.wait_swap_healthy(timeout_s=15.0):
                raise SwapctlError("llama-swap did not come up after backend switch")
        except SwapctlError as exc:
            try:
                swap_config.restore_backup()
                system.restart_swap()
                system.wait_swap_healthy(timeout_s=15.0)
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"{exc}; reverted") from exc

    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True, "backend": backend, "models_updated": count})


@router.post("/backend/{model_id}")
def switch_model_backend(
    request: Request,
    model_id: str,
    backend: str = Form(...),
):
    from ..config import _VALID_BACKENDS
    from .. import swap_config, system

    if backend not in _VALID_BACKENDS:
        raise HTTPException(status_code=400, detail=f"invalid backend: {backend!r}")

    try:
        model_id = valid_model_id(model_id)
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    current = swap_config.detect_model_backend(model_id)
    if current == backend:
        if _is_htmx(request):
            return Response(status_code=204)
        return JSONResponse({"ok": True, "changed": False})

    bak = swap_config.backup()
    try:
        swap_config.switch_model_backend(model_id, backend)
    except (ConfigError, SwapctlError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        system.restart_swap()
        if not system.wait_swap_healthy(timeout_s=15.0):
            raise SwapctlError("llama-swap did not come up after backend switch")
    except SwapctlError as exc:
        try:
            swap_config.restore_backup()
            system.restart_swap()
            system.wait_swap_healthy(timeout_s=15.0)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"{exc}; reverted") from exc

    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True, "model_id": model_id, "backend": backend})


@router.post("/test/run/{model_id}")
def test_run(
    request: Request,
    model_id: str,
    test_type: str = Form("speed"),
    backend: str = Form("vulkan"),
    only_failed: bool = Form(False),
):
    from .. import tester

    try:
        model_id = valid_model_id(model_id)
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if test_type == "all":
        tester.enqueue(model_id, "speed", "vulkan")
        tester.enqueue(model_id, "speed", "hip")
        job_id = tester.enqueue(model_id, "intelligence")
    elif test_type == "intelligence":
        job_id = tester.enqueue(model_id, "intelligence", only_failed=only_failed)
    elif test_type == "speed_spec":
        job_id = tester.enqueue(model_id, "speed_spec")
    else:
        job_id = tester.enqueue(model_id, "speed", backend)

    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Redirect": f"{PREFIX}/testing"})
    return JSONResponse({"job_id": job_id})


@router.post("/test/batch")
def test_batch(
    request: Request,
    test_type: list[str] = Form(default=["speed"]),
    scope: str = Form("all"),
):
    from .. import registry, tester

    types = set(test_type)
    do_vk = "speed" in types
    do_rocm = "speed_rocm" in types
    do_iq = "intelligence" in types

    records = registry.list_all()
    queued = 0
    for r in records:
        tr = r.test_results or {}
        if do_vk:
            if scope != "missing" or not tr.get("speed_vulkan"):
                tester.enqueue(r.model_id, "speed", "vulkan")
                queued += 1
        if do_rocm:
            if scope != "missing" or not tr.get("speed_hip"):
                tester.enqueue(r.model_id, "speed", "hip")
                queued += 1
        if do_iq:
            if scope != "missing" or not tr.get("intelligence"):
                tester.enqueue(r.model_id, "intelligence")
                queued += 1

    if _is_htmx(request):
        if queued:
            return Response(status_code=204, headers={"HX-Redirect": f"{PREFIX}/testing"})
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True, "queued": queued})


@router.post("/test/reset/{model_id}")
def test_reset(
    request: Request,
    model_id: str,
    what: str = Form("all"),
):
    from .. import registry

    try:
        model_id = valid_model_id(model_id)
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    rec = registry.get(model_id)
    if rec is None:
        raise HTTPException(404, "model not found")
    tr = rec.test_results or {}
    if what == "all":
        rec.test_results = {}
    else:
        for key in what.split(","):
            key = key.strip()
            tr.pop(key, None)
        for k in ("best_speed", "best_backend"):
            tr.pop(k, None)
        vk = (tr.get("speed_vulkan") or {}).get("tg_speed", 0) or 0
        hip = (tr.get("speed_hip") or {}).get("tg_speed", 0) or 0
        if hip > vk:
            tr["best_speed"] = hip
            tr["best_backend"] = "hip"
        elif vk > 0:
            tr["best_speed"] = vk
            tr["best_backend"] = "vulkan"
        rec.test_results = tr
    registry.upsert(rec)
    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True})


@router.post("/test/cancel/{job_id}")
def test_cancel(request: Request, job_id: str):
    from .. import tester

    ok = tester.cancel_job(job_id)
    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": ok})


@router.post("/test/cancel-all")
def test_cancel_all(request: Request):
    from .. import tester

    n = tester.cancel_all()
    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True, "cancelled": n})


@router.post("/test/clear")
def test_clear(request: Request):
    from .. import tester

    n = tester.clear_finished()
    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True, "cleared": n})


@router.get("/test/history/{model_id}", response_class=HTMLResponse)
def test_history(request: Request, model_id: str):
    import json as _json
    from .. import tester
    try:
        model_id = valid_model_id(model_id)
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    runs: list[dict] = []
    if tester.HISTORY_PATH.exists():
        try:
            with tester.HISTORY_PATH.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = _json.loads(line)
                    except Exception:
                        continue
                    if e.get("model_id") == model_id:
                        runs.append(e)
        except Exception:
            pass
    runs.reverse()
    return templates.TemplateResponse(request, "partials/test_history.html", {"runs": runs[:50]})


@router.get("/test/iq/{model_id}")
def test_iq_detail(request: Request, model_id: str):
    from .. import registry

    rec = registry.get(model_id)
    if rec is None:
        raise HTTPException(404, "model not found")
    tr = rec.test_results or {}
    iq = tr.get("intelligence")
    if not iq:
        raise HTTPException(404, "no intelligence test results")
    if _is_htmx(request):
        return templates.TemplateResponse(
            request,
            "partials/iq_detail.html",
            {"model_id": model_id, "iq": iq},
        )
    return JSONResponse({"model_id": model_id, "intelligence": iq})


@router.get("/test/export")
def test_export():
    import csv
    import io

    from .. import registry

    categories = ["math", "reasoning", "code", "coding", "knowledge",
                  "czech", "instruction", "summarization", "structured", "sysadmin"]
    records = registry.list_all()

    rows = []
    for r in records:
        tr = r.test_results or {}
        vk = tr.get("speed_vulkan")
        hip = tr.get("speed_hip")
        vk_tg = (vk.get("tg_speed", 0) if isinstance(vk, dict) else 0) or 0
        hip_tg = (hip.get("tg_speed", 0) if isinstance(hip, dict) else 0) or 0
        if hip_tg > vk_tg:
            best_speed, backend = hip_tg, "rocm"
        elif vk_tg > 0:
            best_speed, backend = vk_tg, "vulkan"
        else:
            best_speed, backend = 0, ""
        iq = tr.get("intelligence")
        iq_score = (iq.get("score", 0) if isinstance(iq, dict) else 0) or 0
        iq_passed = iq.get("passed", "") if isinstance(iq, dict) else ""
        iq_total = iq.get("total", "") if isinstance(iq, dict) else ""
        iq_dur = (iq.get("duration", 0) if isinstance(iq, dict) else 0) or 0
        cats = iq.get("categories", {}) if isinstance(iq, dict) else {}
        cat_scores = [cats.get(c, {}).get("score", "") if cats else "" for c in categories]
        size_gb = round(r.size_bytes / 1024**3, 1) if r.size_bytes else ""
        spec = tr.get("speed_spec")
        spec_tg = (spec.get("tg_speed", "") if isinstance(spec, dict) else "") or ""
        rows.append({
            "model_id": r.model_id, "size_gb": size_gb, "quant": r.quant or "",
            "speed": best_speed, "backend": backend, "spec_tg": spec_tg,
            "iq_score": iq_score, "iq_passed": iq_passed, "iq_total": iq_total,
            "iq_dur": iq_dur, "cat_scores": cat_scores,
        })

    speeds = [r["speed"] for r in rows if r["speed"] > 0]
    spd_lo, spd_hi = (min(speeds), max(speeds)) if speeds else (0, 0)
    spd_rng = spd_hi - spd_lo if spd_hi > spd_lo else 1
    durs = [r["iq_dur"] for r in rows if r["iq_dur"] > 0]
    dur_lo, dur_hi = (min(durs), max(durs)) if durs else (0, 0)
    dur_rng = dur_hi - dur_lo if dur_hi > dur_lo else 1

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["model", "size_gb", "quant", "speed_t/s", "backend", "spec_t/s",
                "iq_score", "iq_passed", "iq_total", "iq_time_s",
                "eff"] + [f"iq_{c}" for c in categories])
    for r in sorted(rows, key=lambda r: r["model_id"]):
        dur, spd, iq = r["iq_dur"], r["speed"], r["iq_score"]
        if dur and dur > 0:
            speed_pct = round((1 - (dur - dur_lo) / dur_rng) * 100, 1)
        elif spd > 0:
            speed_pct = round((spd - spd_lo) / spd_rng * 100, 1)
        else:
            speed_pct = 0
        eff = round(iq * 0.7 + speed_pct * 0.3, 1) if iq and (spd or dur) else ""
        w.writerow([r["model_id"], r["size_gb"], r["quant"],
                    r["speed"] or "", r["backend"], r["spec_tg"],
                    r["iq_score"] or "", r["iq_passed"], r["iq_total"],
                    r["iq_dur"] or "", eff] + r["cat_scores"])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=swapctl-test-results.csv"},
    )


@router.get("/status")
def status_json():
    from .. import registry, system

    out: dict = {
        "disk_free_gb": None,
        "disk_total_gb": None,
        "gpu_used_mb": None,
        "gpu_total_mb": None,
        "swap_active": False,
        "swap_running": [],
        "swap_configured": [],
        "swap_loaded": [],
        "installed_count": 0,
    }
    try:
        free, total = system.disk_free_gb(settings.models_dir)
        out["disk_free_gb"] = free
        out["disk_total_gb"] = total
    except OSError:
        pass
    gpu = system.gpu_memory_mb()
    if gpu:
        out["gpu_used_mb"], out["gpu_total_mb"] = gpu
    try:
        out["swap_active"] = system.is_swap_active()
        out["swap_running"] = system.swap_running_models()
        out["swap_configured"] = system.swap_configured_models()
        out["swap_loaded"] = out["swap_running"]
    except (SwapctlError, OSError):
        pass
    try:
        out["installed_count"] = len(registry.list_all())
    except SwapctlError:
        pass
    return JSONResponse(out)


@router.get("/dl-speed")
def get_dl_speed():
    from ..config import get_dl_speed_limit
    return JSONResponse({"limit_mb": get_dl_speed_limit()})


@router.post("/dl-speed")
def set_dl_speed(request: Request, limit_mb: int = Form(default=0)):
    from ..config import set_dl_speed_limit
    set_dl_speed_limit(limit_mb)
    if _is_htmx(request):
        return HTMLResponse(
            f'<span id="dl-speed-val" hx-swap-oob="true">'
            f'{"Full speed" if limit_mb == 0 else f"{limit_mb} MB/s"}</span>'
        )
    return JSONResponse({"limit_mb": limit_mb})


# ── NPU / FLM ──────────────────────────────────────────────────────────


@router.post("/npu/pull")
async def npu_pull(request: Request, model: str = Form(...)):
    from .. import npu
    if not npu.available():
        raise HTTPException(503, "FLM not installed")
    job_id = npu.start_pull(model)
    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"job_id": job_id})


@router.post("/npu/remove/{model:path}")
def npu_remove(request: Request, model: str):
    from .. import npu
    err = npu.remove_model(model)
    if err:
        raise HTTPException(500, err)
    if _is_htmx(request):
        return Response(status_code=200, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True})


@router.post("/npu/serve")
def npu_serve(request: Request, model: str = Form(...), port: int = Form(default=8199)):
    from .. import npu
    err = npu.start_serve(model, port)
    if err:
        raise HTTPException(500, err)
    if _is_htmx(request):
        return Response(status_code=200, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True, "model": model, "port": port})


@router.post("/npu/stop")
def npu_stop(request: Request):
    from .. import npu
    npu.stop_serve()
    if _is_htmx(request):
        return Response(status_code=200, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True})


@router.get("/npu/status")
def npu_status_api():
    from .. import npu
    return JSONResponse({
        "serve": npu.serve_status(),
        "jobs": npu.pull_jobs(),
    })


@router.post("/npu/jobs/{job_id}/cancel")
def npu_cancel_pull(request: Request, job_id: str):
    from .. import npu
    npu.cancel_pull(job_id)
    if _is_htmx(request):
        return Response(status_code=200, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True})


@router.post("/npu/jobs/clear")
def npu_clear_jobs(request: Request):
    from .. import npu
    npu.clear_finished_jobs()
    if _is_htmx(request):
        return Response(status_code=200, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True})


@router.post("/npu/test/{model:path}")
def npu_test(request: Request, model: str):
    from .. import npu
    if not npu.available():
        raise HTTPException(503, "FLM not installed")
    err = npu.run_speed_test(model)
    if err:
        raise HTTPException(409, err)
    if _is_htmx(request):
        return Response(status_code=200, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True, "model": model})


@router.get("/npu/test-status")
def npu_test_status():
    from .. import npu
    return JSONResponse({"testing": npu.get_test_status()})


# ── Parallel slots ────────────────────────────────────────────────────


@router.post("/iq-parallel")
def iq_parallel_set(request: Request, enabled: str = Form("0")):
    from ..config import set_iq_parallel
    set_iq_parallel(enabled in ("1", "on", "true"))
    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True})


@router.post("/parallel-slots")
def parallel_slots_set(request: Request, slots: int = Form(...)):
    from ..config import set_parallel_slots
    from .. import swap_config, system
    set_parallel_slots(slots)
    try:
        count = swap_config.update_parallel_slots(slots)
        if count > 0:
            system.restart_swap()
            system.wait_swap_healthy(timeout_s=15.0)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True, "slots": slots, "models_updated": count})


# ── Auto-update ───────────────────────────────────────────────────────


@router.get("/auto-update")
def auto_update_get():
    from .. import auto_update
    return JSONResponse(auto_update.get_settings())


@router.post("/auto-update")
def auto_update_set(request: Request, enabled: str = Form("0"), interval_hours: int = Form(24)):
    from .. import auto_update
    s = auto_update.set_settings(enabled in ("1", "on", "true"), interval_hours)
    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse(s)


@router.post("/auto-update/run-now")
async def auto_update_run(request: Request):
    from .. import auto_update
    results = await auto_update.run_check_and_queue()
    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"results": results})


# ── Power management ──────────────────────────────────────────────────


@router.get("/power")
def power_state():
    from .. import power
    return JSONResponse(power.get_state())


@router.post("/power/gpu-dpm")
def power_gpu_dpm(request: Request, level: str = Form(...)):
    from .. import power
    err = power.set_gpu_dpm(level)
    if err:
        raise HTTPException(400, err)
    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True, "level": level})


@router.post("/power/gpu-clocks")
def power_gpu_clocks(request: Request, min_mhz: int = Form(...), max_mhz: int = Form(...)):
    from .. import power
    err = power.set_gpu_clocks(min_mhz, max_mhz)
    if err:
        raise HTTPException(400, err)
    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True})


@router.post("/power/gpu-clocks-reset")
def power_gpu_clocks_reset(request: Request):
    from .. import power
    err = power.reset_gpu_clocks()
    if err:
        raise HTTPException(500, err)
    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True})


@router.post("/power/cpu-governor")
def power_cpu_governor(request: Request, governor: str = Form(...)):
    from .. import power
    err = power.set_cpu_governor(governor)
    if err:
        raise HTTPException(400, err)
    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True, "governor": governor})


@router.post("/power/cpu-boost")
def power_cpu_boost(request: Request, enabled: str = Form(...)):
    from .. import power
    err = power.set_cpu_boost(enabled == "1")
    if err:
        raise HTTPException(500, err)
    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True, "boost": enabled == "1"})


@router.post("/power/socket-tdp")
def power_socket_tdp(
    request: Request,
    stapm_w: float = Form(...),
    fast_w: float = Form(...),
    slow_w: float = Form(...),
    apu_w: float = Form(...),
    stapm_time_s: int = Form(0),
    apply_on_boot: str = Form(""),
):
    from .. import power, registry
    for label, val in (("stapm", stapm_w), ("fast", fast_w), ("slow", slow_w), ("apu", apu_w)):
        if not (1.0 <= val <= 200.0):
            raise HTTPException(400, f"{label}_w {val} out of range 1..200 W")
    if stapm_time_s and not (1 <= stapm_time_s <= 3600):
        raise HTTPException(400, f"stapm_time_s {stapm_time_s} out of range 1..3600 s")
    args = [int(round(v * 1000)) for v in (stapm_w, fast_w, slow_w, apu_w)]
    time_arg = stapm_time_s if stapm_time_s > 0 else None
    result = power.set_socket_tdp(*args, stapm_time_s=time_arg)
    if not result.get("ok"):
        raise HTTPException(502, result.get("error") or str(result.get("errors") or "tdpctl failed"))
    registry.set_socket_tdp_config(
        *args,
        apply_on_boot=apply_on_boot in ("1", "true", "on"),
        stapm_time_s=time_arg,
    )
    if _is_htmx(request):
        return Response(status_code=204, headers={"HX-Refresh": "true"})
    return JSONResponse({"ok": True, "actual": result.get("actual")})


@router.get("/vram-estimate/{model_id:path}")
def vram_estimate(request: Request, model_id: str):
    from .. import gguf_meta

    model_dir = settings.models_dir / model_id
    if not model_dir.is_dir():
        raise HTTPException(404, "model not found")
    gguf_files = sorted(
        [p for p in model_dir.glob("*.gguf") if not p.name.startswith("mmproj")]
    )
    if not gguf_files:
        gguf_files = sorted(
            [p for p in model_dir.rglob("*.gguf") if not p.name.startswith("mmproj")]
        )
    if not gguf_files:
        raise HTTPException(404, "no GGUF files found")
    result = gguf_meta.estimate_vram(gguf_files[0])
    if result is None:
        raise HTTPException(500, "could not parse GGUF metadata")
    if _wants_html(request):
        return templates.TemplateResponse(
            request, "partials/vram_estimate.html", {"vram": result, "model_id": model_id},
        )
    return JSONResponse(result)
