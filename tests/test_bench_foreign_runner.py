"""A container-served model gets its speed measured, once, as configured.

halogen-flash-next runs as `podman run ...`. The benchmark switches backends
per arm, and switch_model_backend now refuses a foreign runner, so every arm
used to end in an error record and the model never got a speed. A restart
per arm would also kill and cold-start the container for nothing. It now gets
one arm, no switch, no restart — the same pp/gen tests every other model gets.
"""
from __future__ import annotations

import pytest

from app import benchmark, swap_config

HALOGEN = "podman run --name halogen --rm -p 9135:8731 ghcr.io/peonist-ai/halogen-flash-server:0.13.8"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    f = tmp_path / "config.yaml"
    f.write_text(f"models:\n  halogen-flash-next:\n    cmd: {HALOGEN}\n    proxy: http://127.0.0.1:9135\n")
    real_load = swap_config.load
    monkeypatch.setattr(swap_config, "load", lambda path=None: real_load(f))
    monkeypatch.setattr(swap_config, "save", lambda data, path=None: None)
    monkeypatch.setattr(swap_config, "detect_all_backends", lambda: {})
    return f


def test_auto_mode_gives_a_container_one_arm(cfg):
    assert benchmark.auto_bench_backends("halogen-flash-next") == ["container"]


def test_one_arm_no_switch_no_restart(cfg, monkeypatch):
    calls = {"switch": 0, "restart": 0}
    monkeypatch.setattr(swap_config, "switch_model_backend",
                        lambda *a, **k: calls.__setitem__("switch", calls["switch"] + 1))
    from app import system
    monkeypatch.setattr(system, "restart_swap", lambda: calls.__setitem__("restart", calls["restart"] + 1))
    monkeypatch.setattr(system, "wait_swap_healthy", lambda **k: None)
    monkeypatch.setattr(benchmark, "_chat_simple", lambda *a, **k: {})
    monkeypatch.setattr(benchmark, "_read_vram_bytes", lambda: 0)
    monkeypatch.setattr(benchmark, "_append_speed_history", lambda *a, **k: None)
    monkeypatch.setattr(benchmark, "_run_pp_test",
                        lambda mid, b, *a, **k: {"backend_id": b, "test": "pp", "pp_tps": 1300.0})
    monkeypatch.setattr(benchmark, "_run_gen_test",
                        lambda mid, b, *a, **k: {"backend_id": b, "test": "gen", "gen_tps": 48.5})

    # Several labels passed in: a container still gets exactly one arm.
    out = benchmark._run_benchmark_locked("halogen-flash-next", ["vulkan", "hip", "rock"])

    assert calls == {"switch": 0, "restart": 0}
    tests = [(r["backend_id"], r["test"]) for r in out["results"]]
    assert tests == [("container", "pp"), ("container", "gen")]
