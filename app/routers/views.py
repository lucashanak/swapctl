"""HTMX server-rendered view routes."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..config import settings

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals["root_path"] = settings.root_path
try:
    from .. import github_watch
    templates.env.globals["gh_watch"] = github_watch.load_state
except Exception:
    pass

router = APIRouter(tags=["views"])


AUTHORS = ["unsloth", "bartowski", "mradermacher", "ggml-org", "google", "Qwen"]
SORTS = [("downloads", "Downloads"), ("updated", "Updated"), ("created", "Created"), ("likes", "Likes")]


def _collect_status() -> dict:
    """Collect dashboard status. Tolerates missing backend modules."""
    status: dict = {
        "disk_free_gb": None,
        "disk_total_gb": None,
        "gpu_used_mb": None,
        "gpu_total_mb": None,
        "ram_used_mb": None,
        "ram_total_mb": None,
        "cpu_pct": None,
        "swap_active": False,
        "swap_running": [],
        "swap_configured": [],
        "swap_loaded": [],
        "installed_count": 0,
    }
    try:
        from .. import system

        free, total = system.disk_free_gb(settings.models_dir)
        status["disk_free_gb"] = free
        status["disk_total_gb"] = total
        gpu = system.gpu_memory_mb()
        if gpu:
            status["gpu_used_mb"], status["gpu_total_mb"] = gpu
        ram = system.ram_mb()
        if ram:
            status["ram_used_mb"], status["ram_total_mb"] = ram
        status["cpu_pct"] = system.cpu_usage_percent()
        status["gpu_busy_pct"] = system.gpu_busy_percent()
        status["npu_contexts"] = system.npu_active_contexts()
        status["sensors"] = system.sensors()
        status["swap_active"] = system.is_swap_active()
        status["swap_running"] = system.swap_running_models()
        status["swap_configured"] = system.swap_configured_models()
        status["swap_loaded"] = status["swap_running"]
    except Exception:
        pass
    try:
        from .. import registry

        status["installed_count"] = len(registry.list_all())
    except Exception:
        pass
    return status


_MAX_CTX_CACHE: dict[str, tuple[float, int | None]] = {}
_TAGS_CACHE: dict[str, list[str]] = {}


def _model_tags(model_id: str) -> list[str]:
    from .. import gguf_meta

    model_dir = settings.models_dir / model_id
    if not model_dir.exists():
        return []
    gguf_files = [p for p in model_dir.glob("*.gguf") if not p.name.startswith("mmproj")]
    if not gguf_files:
        gguf_files = [p for p in model_dir.rglob("*.gguf") if not p.name.startswith("mmproj")]
    if not gguf_files:
        return []
    target = sorted(gguf_files)[0]
    try:
        stat = target.stat()
    except OSError:
        return []
    cache_key = f"{target}:{stat.st_mtime_ns}:{stat.st_size}"
    cached = _TAGS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    tags = gguf_meta.read_model_tags(target)
    _TAGS_CACHE[cache_key] = tags
    return tags


def _model_max_ctx(model_id: str) -> int | None:
    """Resolve model's max context by scanning gguf in /models/<id>/. Cached by mtime+size."""
    from .. import gguf_meta

    model_dir = settings.models_dir / model_id
    if not model_dir.exists():
        return None
    gguf_files = [p for p in model_dir.glob("*.gguf") if not p.name.startswith("mmproj")]
    if not gguf_files:
        gguf_files = list(model_dir.rglob("*.gguf"))
        gguf_files = [p for p in gguf_files if not p.name.startswith("mmproj")]
    if not gguf_files:
        return None
    target = sorted(gguf_files)[0]
    try:
        stat = target.stat()
    except OSError:
        return None
    cache_key = f"{target}:{stat.st_mtime_ns}:{stat.st_size}"
    cached = _MAX_CTX_CACHE.get(cache_key)
    if cached is not None:
        return cached[1]
    n_ctx = gguf_meta.read_context_length(target)
    _MAX_CTX_CACHE[cache_key] = (0.0, n_ctx)
    return n_ctx


def _list_installed(check_updates: bool = False, force_refresh: bool = False) -> list[dict]:
    try:
        from .. import hf_client, registry

        records = registry.list_all()
        loaded: list[str] = []
        try:
            from .. import system

            loaded = system.swap_running_models()
        except Exception:
            pass
        latest_map: dict = {}
        if check_updates:
            repos = [r.repo_id for r in records if r.repo_id]
            try:
                latest_map = hf_client.bulk_repo_last_modified(repos, force=force_refresh)
            except Exception:
                latest_map = {}
        out: list[dict] = []
        for r in records:
            d = _record_to_dict(r)
            d["loaded"] = d.get("model_id") in loaded
            latest = latest_map.get(r.repo_id) if r.repo_id else None
            d["latest_modified"] = latest.isoformat() if latest else None
            installed_lm = r.hf_last_modified
            d["update_available"] = bool(
                latest and installed_lm and latest.isoformat() > installed_lm
            )
            d["max_ctx"] = _model_max_ctx(r.model_id)
            d["tags"] = _model_tags(r.model_id)
            out.append(d)
        return out
    except Exception:
        return []


def _record_to_dict(r: object) -> dict:
    if isinstance(r, dict):
        return dict(r)
    fields = (
        "model_id",
        "repo_id",
        "filename",
        "mmproj",
        "installed_at",
        "hf_last_modified",
        "size_bytes",
        "ctx_size",
        "quant",
        "cache_type_k",
        "cache_type_v",
        "threads",
        "flash_attn",
        "extra_args",
        "reasoning_format",
        "reasoning_effort",
        "spec_default",
        "draft_model",
        "draft_max",
        "draft_min",
        "sampling_temp",
        "sampling_top_k",
        "sampling_top_p",
        "sampling_min_p",
        "sampling_repeat_penalty",
        "system_prompt",
        "test_results",
    )
    return {f: getattr(r, f, None) for f in fields}


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request, refresh: int = 0) -> HTMLResponse:
    from fastapi.responses import RedirectResponse
    return RedirectResponse(f"{settings.root_path}/testing" + ("?refresh=1" if refresh else ""), status_code=302)


@router.get("/browse", response_class=HTMLResponse)
def browse(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "browse.html",
        {
            "authors": AUTHORS,
            "sorts": SORTS,
            "active_tab": "browse",
        },
    )


@router.post("/browse/search", response_class=HTMLResponse)
def browse_search(
    request: Request,
    q: str = Form(default=""),
    author: list[str] = Form(default=[]),
    sort: str = Form(default="downloads"),
    max_size_gb: int = Form(default=0),
    page: int = Form(default=1),
) -> HTMLResponse:
    page = max(1, int(page or 1))
    limit = 20
    offset = (page - 1) * limit
    results: list = []
    error: str | None = None
    enrich = max_size_gb > 0 or sort in ("size_asc", "size_desc")
    hf_sort = sort if sort in ("downloads", "updated", "created", "likes") else "downloads"
    try:
        from .. import hf_client

        author_arg = author[0] if author and len(author) == 1 else None
        fetch_limit = limit * 3 if enrich else limit * (len(author) if len(author) > 1 else 1)
        items = hf_client.search_models(
            query=q or None,
            author=author_arg,
            library="gguf",
            sort=hf_sort,
            limit=fetch_limit,
            offset=offset,
            enrich_sizes=enrich,
        )
        if len(author) > 1:
            allowed = set(author)
            items = [m for m in items if m.author in allowed]
        if max_size_gb > 0:
            items = [m for m in items if m.min_size_gb is not None and m.min_size_gb <= max_size_gb]
        if sort == "size_asc":
            items.sort(key=lambda m: (m.min_size_gb is None, m.min_size_gb or 0))
        elif sort == "size_desc":
            items.sort(key=lambda m: -(m.min_size_gb or 0))
        results = items[:limit]
    except Exception as exc:
        error = str(exc)

    return templates.TemplateResponse(
        request,
        "partials/results.html",
        {
            "results": results,
            "error": error,
            "page": page,
            "q": q,
            "sort": sort,
            "authors_selected": author,
            "max_size_gb": max_size_gb,
        },
    )


@router.get("/model/{repo_id:path}", response_class=HTMLResponse)
def model_detail(request: Request, repo_id: str) -> HTMLResponse:
    main_groups: list = []
    mmproj_groups: list = []
    error: str | None = None
    try:
        from .. import hf_client

        main_groups, mmproj_groups = hf_client.list_repo_groups(repo_id)
    except Exception as exc:
        error = str(exc)

    from .. import hf_client, system

    gpu = system.gpu_memory_mb()
    gpu_total_gb = int(gpu[1] / 1024) if gpu else 96

    readme_html: str | None = None
    try:
        raw = hf_client.get_readme(repo_id)
        if raw:
            readme_html = hf_client.render_markdown(raw)
    except Exception:
        readme_html = None

    groups_payload = [
        {
            "primary_filename": g.primary_filename,
            "quant": g.quant or "",
            "total_bytes": g.total_bytes,
            "total_gb": round(g.total_bytes / 1024 / 1024 / 1024, 2),
            "part_count": g.part_count,
        }
        for g in main_groups
    ]
    mmproj_payload = [
        {"primary_filename": g.primary_filename, "quant": g.quant or ""}
        for g in mmproj_groups
    ]

    suggested = repo_id.split("/", 1)[-1].lower()
    for sfx in ("-gguf", "-gguf.gguf"):
        if suggested.endswith(sfx):
            suggested = suggested[: -len(sfx)]
    return templates.TemplateResponse(
        request,
        "model_detail.html",
        {
            "repo_id": repo_id,
            "groups": groups_payload,
            "mmproj_groups": mmproj_payload,
            "error": error,
            "suggested_model_id": suggested,
            "ctx_default": settings.ctx_default,
            "gpu_total_gb": gpu_total_gb,
            "readme_html": readme_html,
            "active_tab": "browse",
        },
    )


@router.get("/swap-config", response_class=HTMLResponse)
def swap_config_page(request: Request) -> HTMLResponse:
    from .. import registry, swap_config, system

    try:
        cfg = swap_config.get_simple_config()
    except Exception as exc:
        cfg = {
            "globalTTL": 0,
            "healthCheckTimeout": 300,
            "models": [],
            "priorities": {},
            "pairs": [],
            "error": str(exc),
        }
    try:
        records = registry.list_all()
    except Exception:
        records = []
    registry_payload = [
        {"model_id": r.model_id, "size_gb": round(r.size_bytes / 1024 ** 3, 1) if r.size_bytes else None}
        for r in records
    ]
    gpu = system.gpu_memory_mb()
    gpu_total_gb = int(gpu[1] / 1024) if gpu else 96

    backends = swap_config.detect_all_backends()

    return templates.TemplateResponse(
        request,
        "swap_config.html",
        {
            "cfg": cfg,
            "registry": registry_payload,
            "gpu_total_gb": gpu_total_gb,
            "gpu_budget_gb": settings.gpu_budget_gb,
            "backends": backends,
            "active_tab": "swap",
        },
    )


@router.get("/jobs", response_class=HTMLResponse)
def jobs_list(request: Request) -> HTMLResponse:
    from .. import downloader

    raw = downloader.list_jobs()
    raw.sort(key=lambda j: j.get("started_at", ""), reverse=True)
    jobs = []
    for j in raw:
        bd = int(j.get("bytes_done", 0) or 0)
        bt = int(j.get("bytes_total", 0) or 0)
        pct = min(100, int((bd / bt) * 100)) if bt > 0 else 0
        jobs.append(
            {
                "job_id": j.get("job_id", ""),
                "kind": j.get("kind", ""),
                "model_id": j.get("model_id", ""),
                "status": j.get("status", "running"),
                "percent": pct,
                "bytes_done_gb": round(bd / 1024 ** 3, 2),
                "bytes_total_gb": round(bt / 1024 ** 3, 2),
                "started_at": j.get("started_at", ""),
                "message": j.get("message"),
                "error": j.get("error"),
            }
        )
    from ..config import get_dl_speed_limit
    return templates.TemplateResponse(
        request,
        "jobs.html",
        {"jobs": jobs, "active_tab": "jobs", "dl_speed_limit": get_dl_speed_limit()},
    )


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_progress(request: Request, job_id: str) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "job_progress.html",
        {"job_id": job_id, "active_tab": None},
    )


def _add_effectivity(models: list[dict], speed_fn, iq_fn, duration_fn=None) -> None:
    """Add effectivity = iq*0.7 + speed_pct*0.3.

    speed_pct prefers IQ test duration (inversely normalised — faster is better)
    and falls back to t/s when duration is unavailable."""
    speeds = [(speed_fn(m) or 0) for m in models]
    iqs = [(iq_fn(m) or 0) for m in models]
    durations = [(duration_fn(m) if duration_fn else None) for m in models]

    # Speed normalisation from t/s (higher = better)
    spd_nonzero = [s for s in speeds if s > 0]
    spd_lo = min(spd_nonzero) if spd_nonzero else 0
    spd_hi = max(spd_nonzero) if spd_nonzero else 0
    spd_rng = spd_hi - spd_lo if spd_hi > spd_lo else 1

    # Duration normalisation (lower = better, so invert)
    dur_nonzero = [d for d in durations if d and d > 0]
    dur_lo = min(dur_nonzero) if dur_nonzero else 0
    dur_hi = max(dur_nonzero) if dur_nonzero else 0
    dur_rng = dur_hi - dur_lo if dur_hi > dur_lo else 1

    for m, spd, iq, dur in zip(models, speeds, iqs, durations):
        # Prefer duration-based speed_pct when available
        if dur and dur > 0 and dur_rng > 0:
            m["speed_pct"] = round((1 - (dur - dur_lo) / dur_rng) * 100, 1)
        elif spd > 0:
            m["speed_pct"] = round((spd - spd_lo) / spd_rng * 100, 1)
        else:
            m["speed_pct"] = 0

        if not iq or (not spd and not (dur and dur > 0)):
            m["effectivity"] = 0.0
        else:
            m["effectivity"] = round(iq * 0.7 + m["speed_pct"] * 0.3, 1)


@router.get("/installed")
def installed(request: Request, refresh: int = 0):
    from fastapi.responses import RedirectResponse
    url = f"{settings.root_path}/testing"
    if refresh:
        url += "?refresh=1"
    return RedirectResponse(url, status_code=302)


@router.get("/testing", response_class=HTMLResponse)
def testing(request: Request, refresh: int = 0) -> HTMLResponse:
    from .. import background, hf_client, registry, swap_config, system, tester
    from .api import _list_draft_candidates

    snap = background.get_snapshot()
    records = snap.get("records") or registry.list_all()
    loaded = snap.get("swap_running") or []
    all_backends = snap.get("all_backends") or {}
    snap_max_ctx = snap.get("max_ctx") or {}
    snap_tags = snap.get("tags") or {}

    latest_map: dict = {}
    if refresh:
        repos = [r.repo_id for r in records if r.repo_id]
        try:
            latest_map = hf_client.bulk_repo_last_modified(repos, force=True)
        except Exception:
            pass
        background.invalidate_now()

    models = []
    for r in records:
        tr = r.test_results or {}
        iq = tr.get("intelligence")
        iq_dur = (iq.get("duration") if isinstance(iq, dict) else None) if iq else None
        spec = tr.get("speed_spec")
        latest = latest_map.get(r.repo_id) if r.repo_id else None
        installed_lm = r.hf_last_modified

        current_backend_id = all_backends.get(r.model_id, "")
        speed_history = list(r.speed_history or [])

        # Compute best_speed for sorting/effectivity from speed_history (gen records)
        best_speed = 0.0
        for entry in speed_history:
            if entry.get("test") == "gen":
                tps = entry.get("gen_tps") or 0
                if tps > best_speed:
                    best_speed = tps

        # The per-backend speed test was removed (410), so speed_history only
        # fills for models someone remembered to benchmark and the column read
        # empty for the rest. An intelligence run measures decode on all ~149
        # questions anyway, so use its median when nothing better exists: a
        # ranking that shows intelligence without speed cannot be acted on,
        # because every choice here trades one for the other.
        iq_res = (r.test_results or {}).get("intelligence") or {}
        iq_tps = iq_res.get("decode_tps") or 0
        speed_from_iq = not best_speed and bool(iq_tps)
        if speed_from_iq:
            best_speed = iq_tps
        # Work per second of suite wall clock: comparable across every model,
        # whether or not the server batches the suite's concurrent questions.
        suite_tps = tester.suite_throughput(iq_res)
        speed_batched = speed_from_iq and tester.batched_speed(iq_res)

        models.append({
            "model_id": r.model_id,
            "repo_id": r.repo_id,
            "filename": r.filename,
            "size_bytes": r.size_bytes,
            "quant": r.quant,
            "mmproj": r.mmproj,
            "ctx_size": r.ctx_size,
            "max_ctx": snap_max_ctx.get(r.model_id) if r.model_id in snap_max_ctx else _model_max_ctx(r.model_id),
            "tags": snap_tags.get(r.model_id) if r.model_id in snap_tags else _model_tags(r.model_id),
            "cache_type_k": r.cache_type_k,
            "cache_type_v": r.cache_type_v,
            "threads": r.threads,
            "flash_attn": r.flash_attn,
            "extra_args": r.extra_args,
            "reasoning_format": r.reasoning_format,
            "reasoning_effort": r.reasoning_effort,
            "spec_default": r.spec_default,
            "draft_model": r.draft_model,
            "draft_max": r.draft_max,
            "draft_min": r.draft_min,
            "sampling_temp": getattr(r, "sampling_temp", 0.8),
            "sampling_top_k": getattr(r, "sampling_top_k", 40),
            "sampling_top_p": getattr(r, "sampling_top_p", 0.95),
            "sampling_min_p": getattr(r, "sampling_min_p", 0.05),
            "sampling_repeat_penalty": getattr(r, "sampling_repeat_penalty", 1.1),
            "system_prompt": getattr(r, "system_prompt", ""),
            "hf_last_modified": r.hf_last_modified,
            "update_available": bool(
                latest and installed_lm and latest.isoformat() > installed_lm
            ),
            "loaded": r.model_id in loaded,
            "speed_history": speed_history,
            "current_backend_id": current_backend_id,
            "best_speed": best_speed,
            "speed_from_iq": speed_from_iq,
            "speed_batched": speed_batched,
            "suite_tps": suite_tps,
            "speed_spec": spec,
            "intelligence": iq,
            "iq_duration": iq_dur,
            "agentic": tr.get("agentic"),
        })

    snap_vram = snap.get("vram") or {}
    for m in models:
        v = snap_vram.get(m["model_id"])
        if v:
            m["vram_mb"] = v.get("vram_mb")
            m["vram_estimates"] = v.get("vram_estimates")

    def _test_best_speed(m):
        return m.get("best_speed") or 0

    def _test_iq(m):
        iq = m.get("intelligence")
        return (iq.get("score", 0) if isinstance(iq, dict) else 0) if iq else 0

    def _test_iq_duration(m):
        iq = m.get("intelligence")
        return (iq.get("duration") if isinstance(iq, dict) else None) if iq else None

    _add_effectivity(models, _test_best_speed, _test_iq, _test_iq_duration)
    models.sort(key=lambda m: m["model_id"])
    jobs = tester.list_jobs()
    jobs.sort(key=lambda j: j.get("queued_at", ""), reverse=True)
    iq_categories = ["math", "reasoning", "code", "coding", "knowledge",
                      "czech", "instruction", "summarization", "structured", "sysadmin"]
    draft_candidates = _list_draft_candidates()

    from .. import npu as npu_mod
    npu_installed = []
    npu_test_results = npu_mod.get_test_results()
    npu_testing = npu_mod.get_test_status()
    for m in npu_mod.list_models():
        if not m.get("installed"):
            continue
        name = m.get("model", m.get("name", "?"))
        tr = npu_test_results.get(name, {})
        npu_installed.append({
            "name": name,
            "params": m.get("details", {}).get("parameter_size", ""),
            "quant": m.get("details", {}).get("quantization_level", ""),
            "size_gb": m.get("footprint", 0),
            "ctx": m.get("default_context_length", 0),
            "labels": m.get("label", []),
            "decode_speed": tr.get("decode_speed", 0),
            "prefill_speed": tr.get("prefill_speed", 0),
            "tested_at": tr.get("tested_at"),
            "loaded": f"npu/{name}" in loaded,
        })

    status = snap.get("collect_status") or _collect_status()

    from .. import power, registry
    power_state = power.get_state()
    socket_tdp_config = registry.get_socket_tdp_config()

    return templates.TemplateResponse(
        request,
        "testing.html",
        {"models": models, "jobs": jobs, "queue_len": tester.queue_length(),
         "bench_state": __import__("app.benchmark", fromlist=["bench_state"]).bench_state(),
         "active_tab": "models", "iq_categories": iq_categories,
         "draft_candidates": draft_candidates,
         "npu_models": npu_installed, "npu_testing": npu_testing,
         "status": status, "power": power_state,
         "socket_tdp_config": socket_tdp_config},
    )


@router.get("/npu", response_class=HTMLResponse)
def npu_page(request: Request) -> HTMLResponse:
    from .. import npu as npu_mod
    from .. import system

    npu = system.npu_status()
    return templates.TemplateResponse(
        request,
        "npu.html",
        {
            "npu": npu,
            "active_tab": "npu",
            "serve": npu_mod.serve_status(),
            "pull_jobs": npu_mod.pull_jobs(),
        },
    )


@router.get("/voices", response_class=HTMLResponse)
def voices_page(request: Request) -> HTMLResponse:
    from .. import tts
    from ..piper_catalog import fetch_catalog
    installed_keys = set(tts.list_voices())
    try:
        catalog = fetch_catalog()
    except Exception:
        catalog = []
    installed_voices = [v for v in catalog if v["key"] in installed_keys]
    # Include installed voices not yet in catalog (edge case)
    catalog_keys = {v["key"] for v in catalog}
    for key in installed_keys:
        if key not in catalog_keys:
            installed_voices.append({"key": key, "name": key, "language": "", "language_family": "",
                                     "language_name_native": "", "quality": "", "num_speakers": 1,
                                     "files": [], "installed": True})
    return templates.TemplateResponse(
        request,
        "voices.html",
        {
            "installed_voices": installed_voices,
            "catalog": catalog,
            "active_tab": "voices",
        },
    )


@router.get("/api-docs", response_class=HTMLResponse)
def api_docs(request: Request) -> HTMLResponse:
    import httpx

    from .. import tts as tts_mod

    # Live model list from llama-swap so the docs never go stale.
    llm_models: list[str] | None = None
    try:
        with httpx.Client(timeout=3.0) as c:
            r = c.get(settings.swap_endpoint.rstrip("/") + "/v1/models")
            r.raise_for_status()
            llm_models = sorted(m["id"] for m in r.json().get("data", []))
    except Exception:
        llm_models = None

    # The everyday default first, then the hard-work default; qwen3-coder, the old
    # pick, was deleted on 2026-09-23 and left the example pointing at nothing.
    preferred = ("gufo-ornith-1.5-q6dense", "halogen-flash-next")
    example_model = preferred[0]
    if llm_models:
        example_model = next((m for m in preferred if m in llm_models), llm_models[0])

    # NPU models live inside halogen and answer only while halogen is loaded.
    npu: dict | None = None
    try:
        with httpx.Client(timeout=2.0) as c:
            r = c.get(settings.npu_endpoint + "/health")
            r.raise_for_status()
            info = r.json().get("npu")
            if isinstance(info, dict):
                npu = {"status": info.get("status"), "models": list(info.get("models") or [])}
    except Exception:
        npu = None

    try:
        voices = sorted(tts_mod.list_voices())
    except Exception:
        voices = []
    example_voice = "cs_CZ-jirka-medium"
    if voices and example_voice not in voices:
        example_voice = voices[0]

    # Live image model list from sd-server (stable-diffusion.cpp), so the docs
    # reflect whatever model is currently loaded.
    image_models: list[str] | None = None
    try:
        with httpx.Client(timeout=3.0) as c:
            r = c.get(settings.image_endpoint + "/v1/models")
            r.raise_for_status()
            image_models = sorted(m["id"] for m in r.json().get("data", []))
    except Exception:
        image_models = None
    image_model = image_models[0] if image_models else "sd-cpp-local"

    # Endpoint table autogenerated from the live FastAPI routes.
    from ..main import _is_public  # runtime import — avoids circular import at module load

    root = settings.root_path
    api_endpoints = []
    for route in request.app.routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", None)
        if not methods:
            continue
        if not (path.startswith(f"{root}/api/v1/audio") or path.startswith(f"{root}/api/keys")):
            continue
        doc = (getattr(getattr(route, "endpoint", None), "__doc__", None) or "").strip()
        desc = doc.splitlines()[0] if doc else ""
        for method in sorted(set(methods) - {"HEAD", "OPTIONS"}):
            api_endpoints.append({
                "method": method,
                "path": path,
                "desc": desc,
                "public": _is_public(path),
            })
    api_endpoints.sort(key=lambda e: (e["path"], e["method"]))

    return templates.TemplateResponse(
        request,
        "api_docs.html",
        {
            "active_tab": "api-docs",
            "public_base": settings.public_base,
            "llm_prefix": settings.llm_public_prefix,
            "llm_base": settings.public_base + settings.llm_public_prefix + "/v1",
            "llm_models": llm_models,
            "example_model": example_model,
            "voices": voices,
            "example_voice": example_voice,
            "api_endpoints": api_endpoints,
            "image_prefix": settings.image_public_prefix,
            "image_base": settings.public_base + settings.image_public_prefix + "/v1",
            "image_models": image_models,
            "image_model": image_model,
            "image_steps": 8,
            "npu": npu,
            "npu_base": settings.npu_lan_base,
        },
    )


@router.get("/api-keys", response_class=HTMLResponse)
def api_keys_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "api_keys.html",
        {"active_tab": "api-keys"},
    )


@router.get("/options", response_class=HTMLResponse)
def options_page(request: Request) -> HTMLResponse:
    from .. import auto_update, registry
    from ..config import get_parallel_slots, get_iq_parallel, get_judge_model
    settings_data = auto_update.get_settings()
    models_with_repo = [r for r in registry.list_all() if r.repo_id]
    all_models = sorted(r.model_id for r in registry.list_all())
    return templates.TemplateResponse(
        request,
        "options.html",
        {
            "active_tab": "options",
            "auto_update": settings_data,
            "models_with_repo_count": len(models_with_repo),
            "parallel_slots": get_parallel_slots(),
            "iq_parallel": get_iq_parallel(),
            "judge_model": get_judge_model(),
            "all_models": all_models,
        },
    )
