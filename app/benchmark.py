"""Cross-backend benchmark runner — unified speed measurement.

Runs PP and GEN tests across backends for a given model, persisting results
as ``speed_history`` on the model registry record.

Usage::

    from app.benchmark import run_benchmark, auto_bench_backends
    results = run_benchmark("my-model", ["vulkan", "vulkan-mtp"])
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .backends_registry import BackendsRegistry
from . import registry as reg_mod

log = logging.getLogger("swapctl.benchmark")

# ---------------------------------------------------------------------------
# Module-level concurrency guard
# ---------------------------------------------------------------------------

_BENCHMARK_LOCK = threading.Lock()

# Live state of the current/last benchmark run. Read by /api/benchmark/status and the testing page.
_BENCH_STATE: dict = {
    "running": False,
    "model_id": None,
    "backend_ids": [],
    "current_backend": None,
    "stage": None,         # "starting" | "switching" | "pp" | "gen" | "restoring" | "done"
    "started_at": None,
    "last_result": None,   # last completed run dict (when running=False)
    "last_error": None,
}


def bench_state() -> dict:
    """Return a snapshot of the current benchmark state."""
    return dict(_BENCH_STATE)


def start_benchmark_async(model_id: str, backend_ids: list[str]) -> bool:
    """Spawn benchmark in background thread. Returns False if already running."""
    if not _BENCHMARK_LOCK.acquire(blocking=False):
        return False
    # Lock acquired — release inside thread after work completes
    _BENCHMARK_LOCK.release()

    def _worker():
        try:
            run_benchmark(model_id, backend_ids)
        except Exception as exc:
            log.exception("async benchmark failed: %s", exc)

    t = threading.Thread(target=_worker, daemon=True, name=f"bench-{model_id}")
    t.start()
    return True

# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------

# ~1500 tokens (≈ char_count / 4). Used for prompt-processing speed.
PP_TEST_PROMPT: str = (
    "The history of artificial intelligence is a remarkable journey spanning decades of scientific "
    "discovery, philosophical debate, and engineering ingenuity. The field formally began in 1956 "
    "when John McCarthy, Marvin Minsky, Claude Shannon, and Nathaniel Rochester organised the "
    "Dartmouth Conference, coining the term 'artificial intelligence' and laying out an ambitious "
    "research agenda. Early optimism was high: pioneers believed that human-level machine "
    "intelligence might be achieved within a generation. Those early programs could solve algebra "
    "word problems, prove geometric theorems, and learn to play checkers at a respectable level.\n\n"
    "The first AI winter arrived in the 1970s when progress stalled against the hard limits of "
    "hardware and the combinatorial explosion that plagued search-based methods. Funding dried up "
    "and the field contracted. A second wave came in the 1980s with expert systems—hand-crafted "
    "rule bases that encoded domain knowledge for tasks like medical diagnosis and equipment "
    "configuration. XCON, deployed by Digital Equipment Corporation, saved the company tens of "
    "millions of dollars annually by automatically configuring VAX computer orders. But expert "
    "systems proved brittle: they could not generalise beyond their training domain, and "
    "maintaining vast rule bases was prohibitively expensive. A second AI winter followed.\n\n"
    "The modern renaissance of machine learning began quietly in the 1990s with algorithms like "
    "support vector machines, random forests, and early neural network variants. Yann LeCun "
    "demonstrated convolutional networks for handwritten digit recognition. Geoffrey Hinton and "
    "colleagues developed backpropagation training, but compute and data remained scarce. The "
    "decisive turning point came in 2012 when AlexNet, trained on two NVIDIA GTX 580 GPUs, "
    "crushed the ImageNet competition by a margin that shocked the computer vision community. "
    "Deep learning had arrived. Within a few years, convolutional networks surpassed human-level "
    "accuracy on benchmark image classification tasks.\n\n"
    "Natural language processing underwent its own revolution. Recurrent networks and LSTMs "
    "improved translation and speech recognition through the 2010s. The transformer architecture, "
    "introduced in 2017 by Vaswani and colleagues in 'Attention Is All You Need', fundamentally "
    "changed the field. Self-attention mechanisms allowed models to relate every word to every "
    "other word in a sequence without the vanishing-gradient problems that plagued recurrence. "
    "BERT, GPT-2, and subsequently GPT-3 demonstrated that scaling model size and training data "
    "produced qualitative leaps in language understanding and generation. GPT-3's 175 billion "
    "parameters could write coherent essays, answer questions, and translate between languages—"
    "all without task-specific fine-tuning.\n\n"
    "The release of ChatGPT in November 2022 brought large language models to mass public "
    "awareness. Within two months it had acquired 100 million users, the fastest adoption of any "
    "consumer application in history. Competing models followed rapidly: Google's Bard and later "
    "Gemini, Anthropic's Claude, Meta's LLaMA family released under open weights, and Mistral's "
    "efficient mixture-of-experts models. Inference hardware evolved in parallel: dedicated AI "
    "accelerators, custom silicon from major cloud providers, and consumer GPUs with tensor cores "
    "all contributed to dramatic cost reductions. Quantisation techniques allowed billion-parameter "
    "models to run on consumer hardware with modest VRAM, enabling local inference for "
    "privacy-conscious users. Speculative decoding, mixture-of-experts routing, and flash "
    "attention further improved throughput. The field that once seemed confined to research labs "
    "had become infrastructure embedded in search engines, coding assistants, document editors, "
    "customer service systems, and scientific research pipelines across every domain."
)

# ~50 tokens. Used for generation speed measurement.
GEN_TEST_PROMPT: str = (
    "Write a Python implementation of merge sort with full docstrings and a usage example."
)

# ---------------------------------------------------------------------------
# VRAM helper
# ---------------------------------------------------------------------------

_VRAM_PATH = Path("/sys/class/drm/card0/device/mem_info_gtt_used")


def _read_vram_bytes() -> int | None:
    """Read current GTT memory usage in bytes. Returns None on error."""
    try:
        return int(_VRAM_PATH.read_text().strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Internal chat helper
# ---------------------------------------------------------------------------

def _chat_simple(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float = 300.0,
) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with httpx.Client(timeout=timeout) as client:
        for attempt in range(4):
            resp = client.post(f"{endpoint}/v1/chat/completions", json=body)
            if resp.status_code in (502, 503) and attempt < 3:
                time.sleep(20 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
    raise RuntimeError("exhausted retries")  # unreachable but satisfies type checker


# ---------------------------------------------------------------------------
# Registry persistence helper
# ---------------------------------------------------------------------------

_MAX_HISTORY_PER_KEY = 5


def _append_speed_history(model_id: str, record: dict) -> None:
    """Append a speed record to the model's speed_history, capped at 5 per (backend_id, test)."""
    try:
        rec = reg_mod.get(model_id)
        if rec is None:
            log.warning("benchmark: model %s not found in registry, skipping persist", model_id)
            return
        history = list(rec.speed_history or [])
        history.append(record)
        # Cap to last 5 per (backend_id, test) key
        key = (record.get("backend_id"), record.get("test"))
        keyed = [e for e in history if (e.get("backend_id"), e.get("test")) == key]
        if len(keyed) > _MAX_HISTORY_PER_KEY:
            # Remove oldest excess entries for this key
            excess = len(keyed) - _MAX_HISTORY_PER_KEY
            removed = 0
            kept: list[dict] = []
            for e in history:
                if (e.get("backend_id"), e.get("test")) == key and removed < excess:
                    removed += 1
                else:
                    kept.append(e)
            history = kept
        # Sync best gen result into test_results so installed_row UI displays it
        try:
            best_record = None
            for e in history:
                if e.get('test') == 'gen' and e.get('gen_tps') is not None:
                    if best_record is None or e['gen_tps'] > best_record['gen_tps']:
                        best_record = e
            if best_record is not None:
                tr = dict(rec.test_results or {})
                tr['best_speed'] = round(float(best_record['gen_tps']), 1)
                tr['best_backend'] = best_record.get('backend_id', '')
                rec.test_results = tr
        except Exception:
            log.exception('benchmark: failed to sync test_results for %s', model_id)
        rec.speed_history = history
        reg_mod.upsert(rec)
    except Exception:
        log.exception("benchmark: failed to persist speed history for %s", model_id)


# ---------------------------------------------------------------------------
# Auto backend selection
# ---------------------------------------------------------------------------

def _configured_backend_id(model_id: str) -> str | None:
    """Backend the model is *actually* configured to run on, from its llama-swap cmd.

    This one is known-good by definition, while `BackendsRegistry.compatible()` is
    predicate-based and can miss pinned/specialised runtimes (e.g. a ROCmFPX build
    serving Step-3.7-Flash). Without it, auto-bench can pick only backends that
    cannot load the model at all and store no result.
    """
    try:
        from . import swap_config
        blk = (swap_config.load().get("models") or {}).get(model_id) or {}
        cmd = blk.get("cmd") or ""
        if not cmd:
            return None
        bid = swap_config._detect_backend(cmd)
        return bid if bid and bid != "unknown" else None
    except Exception:
        return None


def auto_bench_backends(model_id: str) -> list[str]:
    """Return ordered list of backends to bench: the model's configured (known-good)
    backend first, then ['vulkan'] + spec-capable compatible backends.

    Spec-capable = backend has non-empty default_flags OR caps contains 'mtp'/'draft'/'gemma4-assistant'.
    """
    try:
        rec = reg_mod.get(model_id)
        br = BackendsRegistry()
        compat = br.compatible(rec) if rec is not None else br.list_all()

        spec_capable_ids = []
        for b in compat:
            if b.id == "vulkan":
                continue
            has_spec_flags = bool(b.default_flags)
            has_spec_caps = any(c in b.caps for c in ("mtp", "draft", "gemma4-assistant"))
            if has_spec_flags or has_spec_caps:
                spec_capable_ids.append(b.id)

        # Skip vulkan baseline if model has baked-in MTP heads or assistant arch —
        # stock vulkan binary can't load those GGUFs at all.
        from .backends_registry import has_mtp_heads
        skip_vulkan_baseline = rec is not None and has_mtp_heads(rec)

        result = [] if skip_vulkan_baseline else ["vulkan"]
        for sid in spec_capable_ids:
            if sid not in result:
                result.append(sid)

        # The backend the model is pinned to always works; put it first so a
        # result is stored even when every predicate-matched backend fails.
        configured = _configured_backend_id(model_id)
        if configured:
            if configured in result:
                result.remove(configured)
            result.insert(0, configured)

        return result or ["vulkan"]
    except Exception:
        log.warning("auto_bench_backends: failed to resolve backends, returning ['vulkan']")
        return ["vulkan"]


# ---------------------------------------------------------------------------
# Main benchmark function
# ---------------------------------------------------------------------------

def run_benchmark(
    model_id: str,
    backend_ids: list[str],
) -> dict:
    """Run PP + GEN benchmark across the specified backends sequentially.

    Returns::

        {
          "model_id": "...",
          "results": [
            {"backend_id": "vulkan", "test": "pp", "pp_tps": 312.0, ...},
            {"backend_id": "vulkan", "test": "gen", "gen_tps": 84.2, ...},
            ...
          ],
        }

    Raises RuntimeError if another benchmark is already running (HTTP 409 caller
    should convert this).
    """
    if not _BENCHMARK_LOCK.acquire(blocking=False):
        raise RuntimeError("A benchmark is already running")

    from datetime import datetime, timezone
    _BENCH_STATE.update({
        "running": True,
        "model_id": model_id,
        "backend_ids": list(backend_ids),
        "current_backend": None,
        "stage": "starting",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "last_result": None,
        "last_error": None,
    })

    result = None
    err = None
    try:
        result = _run_benchmark_locked(model_id, backend_ids)
        return result
    except Exception as exc:
        err = str(exc)
        raise
    finally:
        _BENCH_STATE.update({
            "running": False,
            "stage": "done" if err is None else "error",
            "last_result": result,
            "last_error": err,
        })
        _BENCHMARK_LOCK.release()


def _run_benchmark_locked(
    model_id: str,
    backend_ids: list[str],
) -> dict:
    from .config import settings
    from . import swap_config

    endpoint = settings.swap_endpoint.rstrip("/")

    # Snapshot original backend so we can restore it at the end
    try:
        original_backends: dict[str, str] = swap_config.detect_all_backends()
        original_backend = original_backends.get(model_id, "vulkan")
    except Exception:
        log.warning("benchmark: could not detect original backend for %s, defaulting to vulkan", model_id)
        original_backend = "vulkan"

    # Snapshot the original cmd string verbatim so each backend iteration starts clean.
    # Without this, switch_model_backend's flag-strip logic doesn't catch every prior backend's
    # leftover flags (different backends inject different flag forms), and cmd accumulates cruft
    # across iterations. _ORIGINAL_CMD_SNAPSHOT is restored before each switch_model_backend call.
    _ORIGINAL_CMD_SNAPSHOT = None
    try:
        _orig_data = swap_config.load()
        _ORIGINAL_CMD_SNAPSHOT = (_orig_data.get("models") or {}).get(model_id, {}).get("cmd", "") or None
    except Exception:
        log.warning("benchmark: could not snapshot original cmd for %s", model_id)

    def _restore_cmd_snapshot():
        if not _ORIGINAL_CMD_SNAPSHOT:
            return
        try:
            data = swap_config.load()
            models = data.get("models") or {}
            if model_id in models and isinstance(models[model_id], dict):
                models[model_id]["cmd"] = _ORIGINAL_CMD_SNAPSHOT
                swap_config.save(data)
        except Exception:
            log.exception("benchmark: failed to restore cmd snapshot for %s", model_id)

    results: list[dict] = []

    # Determine whether to restore (avoid unnecessary restart if last backend == original)
    should_restore = bool(original_backend) and (not backend_ids or original_backend != backend_ids[-1])

    try:
        for backend_id in backend_ids:
            log.info("benchmark: switching %s to backend %s", model_id, backend_id)
            vram_start = _read_vram_bytes()

            _restore_cmd_snapshot()
            try:
                swap_config.switch_model_backend(model_id, backend_id)
            except Exception as exc:
                log.error("benchmark: switch to %s failed: %s", backend_id, exc)
                _add_error_records(results, backend_id, str(exc))
                continue

            # Restart llama-swap and wait for it to be healthy
            try:
                from . import system
                system.restart_swap()
                system.wait_swap_healthy(timeout_s=60.0)
            except Exception as exc:
                log.error("benchmark: llama-swap restart failed for backend %s: %s", backend_id, exc)
                _add_error_records(results, backend_id, f"restart failed: {exc}")
                continue

            # Warmup — force model load. If this fails, skip PP/GEN to avoid wasting time
            # on a backend that can't load the model.
            try:
                log.info("benchmark: warming up %s on %s", model_id, backend_id)
                # 180s: large models with --no-mmap + a draft (MTP) can take >60s to
                # cold-load after a backend switch (esp. first Vulkan shader compile).
                _chat_simple(endpoint, model_id, "Say OK", 5, timeout=180.0)
            except Exception as exc:
                log.warning("benchmark: warmup failed for %s / %s: %s — skipping pp/gen", model_id, backend_id, exc)
                _add_error_records(results, backend_id, f"warmup failed: {exc}")
                continue

            vram_after_load = _read_vram_bytes()

            # --- PP test ---
            pp_record = _run_pp_test(model_id, backend_id, endpoint, vram_start, vram_after_load)
            results.append(pp_record)
            _append_speed_history(model_id, pp_record)

            # --- GEN test ---
            gen_record = _run_gen_test(model_id, backend_id, endpoint, vram_start, vram_after_load)
            results.append(gen_record)
            _append_speed_history(model_id, gen_record)

    finally:
        _restore_cmd_snapshot()
        if should_restore:
            try:
                log.info("benchmark: restoring original backend %s for %s", original_backend, model_id)
                swap_config.switch_model_backend(model_id, original_backend)
                from . import system
                system.restart_swap()
                system.wait_swap_healthy(timeout_s=60.0)
            except Exception:
                log.exception("benchmark: failed to restore original backend %s", original_backend)

    return {
        "model_id": model_id,
        "results": results,
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_error_records(results: list[dict], backend_id: str, error: str) -> None:
    now = _now_iso()
    for test in ("pp", "gen"):
        results.append({
            "backend_id": backend_id,
            "test": test,
            "pp_tps": None,
            "gen_tps": None,
            "accept_pct": None,
            "vram_peak_mb": None,
            "tokens": 0,
            "build_fingerprint": "",
            "tested_at": now,
            "error": error,
        })


def _vram_delta_mb(vram_start: int | None, vram_peak: int | None) -> int | None:
    if vram_start is not None and vram_peak is not None and vram_peak >= vram_start:
        return round((vram_peak - vram_start) / 1024 / 1024)
    return None


def _run_pp_test(
    model_id: str,
    backend_id: str,
    endpoint: str,
    vram_start: int | None,
    vram_baseline: int | None,
) -> dict:
    """Run the PP (prompt processing) speed test. Returns a speed_history record."""
    now = _now_iso()
    vram_peak = vram_baseline
    try:
        data = _chat_simple(endpoint, model_id, PP_TEST_PROMPT, max_tokens=10, timeout=120.0)
    except Exception as exc:
        log.warning("benchmark: PP test failed for %s / %s: %s", model_id, backend_id, exc)
        return {
            "backend_id": backend_id, "test": "pp",
            "pp_tps": None, "gen_tps": None, "accept_pct": None,
            "vram_peak_mb": None, "tokens": 0,
            "build_fingerprint": "", "tested_at": now,
            "error": str(exc),
        }

    vram_now = _read_vram_bytes()
    if vram_now is not None and (vram_peak is None or vram_now > vram_peak):
        vram_peak = vram_now

    timings = data.get("timings") or {}
    pp_tps = timings.get("prompt_per_second") or 0.0
    prompt_tokens = int(timings.get("prompt_n") or timings.get("n_prompt_tokens_processed") or 0)
    fingerprint = str(data.get("system_fingerprint") or "")

    return {
        "backend_id": backend_id,
        "test": "pp",
        "pp_tps": round(pp_tps, 1) if pp_tps else None,
        "gen_tps": None,
        "accept_pct": None,
        "vram_peak_mb": _vram_delta_mb(vram_start, vram_peak),
        "tokens": prompt_tokens,
        "build_fingerprint": fingerprint,
        "tested_at": now,
    }


def _run_gen_test(
    model_id: str,
    backend_id: str,
    endpoint: str,
    vram_start: int | None,
    vram_baseline: int | None,
) -> dict:
    """Run the GEN (generation) speed test. Returns a speed_history record."""
    now = _now_iso()
    vram_peak = vram_baseline
    try:
        data = _chat_simple(endpoint, model_id, GEN_TEST_PROMPT, max_tokens=600, timeout=300.0)
    except Exception as exc:
        log.warning("benchmark: GEN test failed for %s / %s: %s", model_id, backend_id, exc)
        return {
            "backend_id": backend_id, "test": "gen",
            "pp_tps": None, "gen_tps": None, "accept_pct": None,
            "vram_peak_mb": None, "tokens": 0,
            "build_fingerprint": "", "tested_at": now,
            "error": str(exc),
        }

    vram_now = _read_vram_bytes()
    if vram_now is not None and (vram_peak is None or vram_now > vram_peak):
        vram_peak = vram_now

    timings = data.get("timings") or {}
    gen_tps = timings.get("predicted_per_second") or 0.0
    draft_n = timings.get("draft_n") or 0
    draft_accepted = timings.get("draft_n_accepted") or 0
    accept_pct = round(draft_accepted / draft_n * 100.0, 1) if draft_n > 0 else None
    completion_tokens = int(timings.get("predicted_n") or timings.get("n_predicted") or 0)
    fingerprint = str(data.get("system_fingerprint") or "")

    return {
        "backend_id": backend_id,
        "test": "gen",
        "pp_tps": None,
        "gen_tps": round(gen_tps, 1) if gen_tps else None,
        "accept_pct": accept_pct,
        "vram_peak_mb": _vram_delta_mb(vram_start, vram_peak),
        "tokens": completion_tokens,
        "build_fingerprint": fingerprint,
        "tested_at": now,
    }
