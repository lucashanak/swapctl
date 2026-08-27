"""Tests for the new speed_history system (spec section H)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helper: create a minimal registry with one model
# ---------------------------------------------------------------------------

def _make_registry(tmp_path: Path, extra_fields: dict | None = None) -> Path:
    registry_path = tmp_path / "registry.json"
    model: dict = {
        "repo_id": None,
        "filename": "model-mtp-Q4_K_M.gguf",
        "mmproj": None,
        "installed_at": "2025-01-01T00:00:00+00:00",
        "hf_last_modified": None,
        "size_bytes": 1000000000,
        "ctx_size": 8192,
        "speed_history": [],
        "test_results": {},
        "events": [],
        "model_tags": [],
    }
    if extra_fields:
        model.update(extra_fields)
    data = {"models": {"test-model": model}, "events": []}
    registry_path.write_text(json.dumps(data))
    return registry_path


# ---------------------------------------------------------------------------
# A) _append_speed_history caps at 5 per (backend_id, test)
# ---------------------------------------------------------------------------

def test_append_speed_history_caps_at_five(tmp_path):
    """_append_speed_history caps at 5 records per (backend_id, test) key."""
    from app.benchmark import _append_speed_history
    import app.registry as real_registry

    reg_path = _make_registry(tmp_path)

    # Patch reg_mod (the module-level alias in benchmark.py) to use our tmp registry path
    with patch("app.benchmark.reg_mod") as mock_reg:
        # Track a real in-memory record
        history_store: list[dict] = []

        def fake_get(model_id):
            rec = MagicMock()
            rec.speed_history = list(history_store)
            return rec

        def fake_upsert(rec):
            history_store.clear()
            history_store.extend(rec.speed_history)

        mock_reg.get.side_effect = fake_get
        mock_reg.upsert.side_effect = fake_upsert

        for i in range(7):
            record = {
                "backend_id": "vulkan",
                "test": "gen",
                "gen_tps": float(10 + i),
                "pp_tps": None,
                "accept_pct": None,
                "vram_peak_mb": None,
                "tokens": 100,
                "build_fingerprint": f"fp{i}",
                "tested_at": f"2025-01-0{i+1}T00:00:00+00:00",
            }
            _append_speed_history("test-model", record)

    vulkan_gen = [e for e in history_store if e["backend_id"] == "vulkan" and e["test"] == "gen"]
    assert len(vulkan_gen) == 5, f"Expected 5 records capped, got {len(vulkan_gen)}"
    tps_values = [e["gen_tps"] for e in vulkan_gen]
    assert min(tps_values) >= 12.0, f"Old records not trimmed: {tps_values}"


def test_append_speed_history_different_keys_not_capped():
    """Records with different (backend_id, test) keys are each capped independently."""
    from app.benchmark import _append_speed_history

    with patch("app.benchmark.reg_mod") as mock_reg:
        history_store: list[dict] = []

        def fake_get(model_id):
            rec = MagicMock()
            rec.speed_history = list(history_store)
            return rec

        def fake_upsert(rec):
            history_store.clear()
            history_store.extend(rec.speed_history)

        mock_reg.get.side_effect = fake_get
        mock_reg.upsert.side_effect = fake_upsert

        for i in range(4):
            _append_speed_history("test-model", {
                "backend_id": "vulkan", "test": "pp",
                "pp_tps": float(100 + i), "gen_tps": None, "accept_pct": None,
                "vram_peak_mb": None, "tokens": 50,
                "build_fingerprint": "", "tested_at": "2025-01-01T00:00:00+00:00",
            })
        for i in range(4):
            _append_speed_history("test-model", {
                "backend_id": "vulkan-mtp", "test": "gen",
                "gen_tps": float(80 + i), "pp_tps": None, "accept_pct": None,
                "vram_peak_mb": None, "tokens": 200,
                "build_fingerprint": "", "tested_at": "2025-01-01T00:00:00+00:00",
            })

    vk_pp = [e for e in history_store if e["backend_id"] == "vulkan" and e["test"] == "pp"]
    mtp_gen = [e for e in history_store if e["backend_id"] == "vulkan-mtp" and e["test"] == "gen"]
    assert len(vk_pp) == 4
    assert len(mtp_gen) == 4


# ---------------------------------------------------------------------------
# C) auto_bench_backends
# ---------------------------------------------------------------------------

def _make_backend_def(bid, caps=None, default_flags=None, model_predicate=None):
    from app.backends_registry import BackendDef
    return BackendDef(
        id=bid,
        binary_path=f"/usr/local/bin/llama-server-{bid}",
        label=bid,
        caps=caps or [],
        default_flags=default_flags or [],
        required_flags=[],
        incompatible_flag_patterns=[],
        model_predicate=model_predicate,
    )


def test_auto_bench_backends_mtp_model():
    """MTP-tagged model returns ['vulkan', 'vulkan-mtp']."""
    from app.benchmark import auto_bench_backends

    vulkan_b = _make_backend_def("vulkan")
    mtp_b = _make_backend_def(
        "vulkan-mtp",
        caps=["vulkan", "mtp"],
        default_flags=["--spec-type draft-mtp"],
        model_predicate="has_mtp_heads",
    )

    mock_model = MagicMock()
    mock_model.model_tags = ["mtp_heads"]
    mock_model.filename = "model-mtp-Q4.gguf"

    with (
        patch("app.benchmark.BackendsRegistry") as mock_br_cls,
        patch("app.benchmark.reg_mod") as mock_reg,
    ):
        mock_br = MagicMock()
        mock_br_cls.return_value = mock_br
        mock_reg.get.return_value = mock_model
        # compatible() returns both backends (model has mtp_heads predicate satisfied)
        mock_br.compatible.return_value = [vulkan_b, mtp_b]

        result = auto_bench_backends("test-model")

    # MTP-tagged model: stock vulkan baseline can't load the GGUF, so skipped.
    assert result == ["vulkan-mtp"], f"Expected ['vulkan-mtp'], got {result}"


def test_auto_bench_backends_non_mtp_model():
    """Non-MTP model returns ['vulkan'] only."""
    from app.benchmark import auto_bench_backends

    vulkan_b = _make_backend_def("vulkan")

    mock_model = MagicMock()
    mock_model.model_tags = []
    mock_model.filename = "model-Q4.gguf"

    with (
        patch("app.benchmark.BackendsRegistry") as mock_br_cls,
        patch("app.benchmark.reg_mod") as mock_reg,
    ):
        mock_br = MagicMock()
        mock_br_cls.return_value = mock_br
        mock_reg.get.return_value = mock_model
        # compatible() returns only vulkan
        mock_br.compatible.return_value = [vulkan_b]

        result = auto_bench_backends("test-model")

    assert result == ["vulkan"], f"Expected ['vulkan'], got {result}"


def test_auto_bench_backends_fallback_on_error():
    """auto_bench_backends returns ['vulkan'] if registry raises."""
    from app.benchmark import auto_bench_backends

    with patch("app.benchmark.BackendsRegistry", side_effect=RuntimeError("unavailable")):
        result = auto_bench_backends("any-model")

    assert result == ["vulkan"]


# ---------------------------------------------------------------------------
# D) apply-best endpoint chooses highest gen_tps
# ---------------------------------------------------------------------------

def test_apply_best_chooses_highest_gen_tps():
    """apply-best should pick the backend with highest gen_tps from speed_history."""
    from fastapi.testclient import TestClient

    speed_history = [
        {"backend_id": "vulkan", "test": "gen", "gen_tps": 70.0,
         "pp_tps": None, "accept_pct": None, "vram_peak_mb": None,
         "tokens": 300, "build_fingerprint": "", "tested_at": "2025-01-01T00:00:00+00:00"},
        {"backend_id": "vulkan-mtp", "test": "gen", "gen_tps": 95.0,
         "pp_tps": None, "accept_pct": None, "vram_peak_mb": None,
         "tokens": 300, "build_fingerprint": "", "tested_at": "2025-01-01T00:00:00+00:00"},
    ]

    mock_rec = MagicMock()
    mock_rec.speed_history = speed_history

    with (
        patch("app.routers.benchmark.registry") as mock_reg,
        patch("app.routers.benchmark.swap_config") as mock_sc,
        patch("app.routers.benchmark.system") as mock_sys,
    ):
        mock_reg.get.return_value = mock_rec
        mock_sc.switch_model_backend.return_value = None
        mock_sys.restart_swap.return_value = None
        mock_sys.wait_swap_healthy.return_value = True

        from app.routers.benchmark import router as bench_router
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(bench_router)
        client = TestClient(app)

        resp = client.post("/api/benchmark/apply-best/test-model", json={})

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["backend_id"] == "vulkan-mtp", (
        f"Expected 'vulkan-mtp' (highest gen_tps=95), got {data['backend_id']}"
    )


# ---------------------------------------------------------------------------
# Concurrency lock still works
# ---------------------------------------------------------------------------

def test_benchmark_lock_still_works():
    from app.benchmark import _BENCHMARK_LOCK, run_benchmark
    import pytest

    _BENCHMARK_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            run_benchmark("some-model", ["vulkan"])
    finally:
        _BENCHMARK_LOCK.release()


def test_auto_bench_backends_includes_pinned_backend():
    """A model pinned to a specialised runtime gets that backend benched first.

    Regression: Step-3.7-Flash runs on a ROCmFPX build that compatible() does not
    return, so auto-bench only tried backends that cannot load it and stored no
    result at all.
    """
    from app.benchmark import auto_bench_backends

    vulkan_b = _make_backend_def("vulkan")

    mock_model = MagicMock()
    mock_model.model_tags = []
    mock_model.filename = "Step-3.7-Flash-ROCmFPX-Q3-00001-of-00009.gguf"

    with (
        patch("app.benchmark.BackendsRegistry") as mock_br_cls,
        patch("app.benchmark.reg_mod") as mock_reg,
        patch("app.benchmark._configured_backend_id", return_value="rocmfpx-mtp"),
    ):
        mock_br = MagicMock()
        mock_br_cls.return_value = mock_br
        mock_reg.get.return_value = mock_model
        mock_br.compatible.return_value = [vulkan_b]  # predicate misses rocmfpx-mtp

        result = auto_bench_backends("step-3.7-flash-q3")

    # Without the fix this is ["vulkan"] only — nothing that can load the model.
    assert result[0] == "rocmfpx-mtp", f"pinned backend must be first, got {result}"
    assert "rocmfpx-mtp" in result


def test_auto_bench_backends_pinned_backend_not_duplicated():
    """If the pinned backend is also predicate-compatible it appears exactly once."""
    from app.benchmark import auto_bench_backends

    vulkan_b = _make_backend_def("vulkan")
    mtp_b = _make_backend_def("vulkan-mtp", caps=["vulkan", "mtp"],
                              default_flags=["--spec-type draft-mtp"])

    mock_model = MagicMock()
    mock_model.model_tags = []
    mock_model.filename = "model-Q4.gguf"

    with (
        patch("app.benchmark.BackendsRegistry") as mock_br_cls,
        patch("app.benchmark.reg_mod") as mock_reg,
        patch("app.benchmark._configured_backend_id", return_value="vulkan-mtp"),
    ):
        mock_br = MagicMock()
        mock_br_cls.return_value = mock_br
        mock_reg.get.return_value = mock_model
        mock_br.compatible.return_value = [vulkan_b, mtp_b]

        result = auto_bench_backends("test-model")

    assert result.count("vulkan-mtp") == 1, f"duplicate backend: {result}"
    assert result[0] == "vulkan-mtp", f"pinned backend must be first, got {result}"
