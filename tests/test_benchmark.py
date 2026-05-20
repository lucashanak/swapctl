"""Tests for benchmark module — updated for the new speed-history system."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Test: test prompt constants exist and are long enough
# ---------------------------------------------------------------------------

def test_pp_test_prompt_exists() -> None:
    from app.benchmark import PP_TEST_PROMPT

    assert isinstance(PP_TEST_PROMPT, str)
    # ~1500 tokens ≈ 6000 chars minimum
    assert len(PP_TEST_PROMPT) >= 3000, "PP_TEST_PROMPT should be ~1500 tokens"


def test_gen_test_prompt_exists() -> None:
    from app.benchmark import GEN_TEST_PROMPT

    assert isinstance(GEN_TEST_PROMPT, str)
    assert len(GEN_TEST_PROMPT) > 10, "GEN_TEST_PROMPT too short"
    assert "merge sort" in GEN_TEST_PROMPT.lower() or "sort" in GEN_TEST_PROMPT.lower()


# ---------------------------------------------------------------------------
# Test: _BENCHMARK_LOCK prevents concurrent runs
# ---------------------------------------------------------------------------

def test_benchmark_lock_prevents_concurrent_runs() -> None:
    """Acquiring _BENCHMARK_LOCK should prevent a second run_benchmark call."""
    from app.benchmark import _BENCHMARK_LOCK

    acquired = _BENCHMARK_LOCK.acquire(blocking=False)
    assert acquired, "_BENCHMARK_LOCK should be acquirable when idle"

    try:
        second = _BENCHMARK_LOCK.acquire(blocking=False)
        assert not second, "Second non-blocking acquire should fail while lock is held"
    finally:
        _BENCHMARK_LOCK.release()

    third = _BENCHMARK_LOCK.acquire(blocking=False)
    assert third, "_BENCHMARK_LOCK should be acquirable after release"
    _BENCHMARK_LOCK.release()


def test_run_benchmark_raises_if_lock_held() -> None:
    """run_benchmark should raise RuntimeError (HTTP 409) if another run is in progress."""
    from app.benchmark import _BENCHMARK_LOCK, run_benchmark
    import pytest

    _BENCHMARK_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            run_benchmark("some-model", ["vulkan"])
    finally:
        _BENCHMARK_LOCK.release()


# ---------------------------------------------------------------------------
# Test: run_benchmark calls switch_model_backend per backend and restores original
# ---------------------------------------------------------------------------

def _fake_chat_response(gen_tps: float = 80.0, pp_tps: float = 300.0) -> dict:
    return {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "system_fingerprint": "test-fp",
        "timings": {
            "predicted_per_second": gen_tps,
            "prompt_per_second": pp_tps,
            "predicted_n": 100,
            "prompt_n": 400,
            "draft_n": 10,
            "draft_n_accepted": 8,
        },
    }


def test_run_benchmark_switches_and_restores(tmp_path) -> None:
    """run_benchmark must call switch_model_backend for each backend and restore original."""
    switch_calls: list[tuple] = []

    def fake_switch(model_id: str, backend_id: str, path=None) -> None:
        switch_calls.append((model_id, backend_id))

    fake_detect_all = {"test-model": "vulkan"}

    with (
        patch("app.benchmark.httpx.Client") as mock_client_cls,
        patch("app.swap_config.switch_model_backend", side_effect=fake_switch),
        patch("app.swap_config.detect_all_backends", return_value=fake_detect_all),
        patch("app.system.restart_swap"),
        patch("app.system.wait_swap_healthy", return_value=True),
        patch("app.benchmark._append_speed_history"),
        patch("app.benchmark._read_vram_bytes", return_value=None),
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _fake_chat_response()
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        from app.benchmark import run_benchmark
        result = run_benchmark("test-model", ["hip", "rock"])

    assert result["model_id"] == "test-model"
    # Each backend runs 2 records (pp + gen)
    assert len(result["results"]) == 4

    switched_backends = [b for _, b in switch_calls]
    assert "hip" in switched_backends
    assert "rock" in switched_backends
    # Original backend (vulkan) should be restored at end
    assert switched_backends[-1] == "vulkan", (
        f"Last switch should restore original backend 'vulkan', got {switched_backends}"
    )


def test_run_benchmark_result_structure() -> None:
    """Result entries must contain expected fields for both pp and gen tests."""
    fake_detect_all = {"test-model": "vulkan"}

    with (
        patch("app.benchmark.httpx.Client") as mock_client_cls,
        patch("app.swap_config.switch_model_backend"),
        patch("app.swap_config.detect_all_backends", return_value=fake_detect_all),
        patch("app.system.restart_swap"),
        patch("app.system.wait_swap_healthy", return_value=True),
        patch("app.benchmark._append_speed_history"),
        patch("app.benchmark._read_vram_bytes", return_value=None),
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _fake_chat_response(85.0, 320.0)
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        from app.benchmark import run_benchmark
        result = run_benchmark("test-model", ["vulkan"])

    assert "results" in result
    assert "model_id" in result
    # Should have 2 records: pp + gen
    assert len(result["results"]) == 2

    pp_entry = next((r for r in result["results"] if r["test"] == "pp"), None)
    gen_entry = next((r for r in result["results"] if r["test"] == "gen"), None)

    assert pp_entry is not None, "Should have a PP record"
    assert gen_entry is not None, "Should have a GEN record"
    assert pp_entry["backend_id"] == "vulkan"
    assert gen_entry["backend_id"] == "vulkan"
    assert "pp_tps" in pp_entry
    assert "gen_tps" in gen_entry
    assert "tested_at" in pp_entry
    assert "build_fingerprint" in gen_entry
