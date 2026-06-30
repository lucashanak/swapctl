"""Regression: reasoning_format='off' must not emit the bogus
`--reasoning-format off` flag (invalid → llama-server crashes on start).
It should map to `--reasoning-budget 0` instead."""
from __future__ import annotations

from pathlib import Path

from app import swap_config


def _cmd(reasoning_format: str) -> str:
    return swap_config._build_cmd(
        gguf_path=Path("/models/m/model.gguf"),
        mmproj_path=None,
        port=9100,
        ctx_size=65536,
        no_mmap=True,
        reasoning_format=reasoning_format,
    )


def test_off_maps_to_reasoning_budget_not_invalid_flag():
    cmd = _cmd("off")
    assert "--reasoning-format off" not in cmd  # the crashing flag
    assert "--reasoning-budget 0" in cmd


def test_deepseek_still_emits_format_flag():
    cmd = _cmd("deepseek")
    assert "--reasoning-format deepseek" in cmd
    assert "--reasoning-budget" not in cmd


def test_empty_emits_neither():
    cmd = _cmd("")
    assert "--reasoning-format" not in cmd
    assert "--reasoning-budget" not in cmd
