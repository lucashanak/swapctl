"""Tests for app.swap_config round-trip editing."""
from __future__ import annotations

from pathlib import Path

import pytest
from ruamel.yaml import YAML

from app import swap_config


SEED_YAML = """\
# llama-swap config
# global settings
healthCheckTimeout: 120

models:
  # existing model — leave comment intact
  llama3-8b:
    cmd: /usr/local/bin/llama-server --model /models/llama3-8b/model.gguf --port 9100 -ngl 999 --ctx-size 65536
    proxy: http://127.0.0.1:9100
  qwen3-4b:
    cmd: /usr/local/bin/llama-server --model /models/qwen3-4b/model.gguf --port 9101 -ngl 999 --ctx-size 65536
    proxy: http://127.0.0.1:9101
"""


@pytest.fixture
def cfg_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(SEED_YAML)
    return path


def test_load_returns_models_mapping(cfg_path: Path) -> None:
    data = swap_config.load(cfg_path)
    assert "models" in data
    assert "llama3-8b" in data["models"]
    assert "qwen3-4b" in data["models"]


def test_list_model_ids(cfg_path: Path) -> None:
    ids = swap_config.list_model_ids(cfg_path)
    assert ids == ["llama3-8b", "qwen3-4b"]


def test_add_model_then_remove_roundtrip(cfg_path: Path) -> None:
    swap_config.add_model(
        model_id="phi4-mini",
        gguf_path=Path("/models/phi4-mini/model.gguf"),
        mmproj_path=None,
        port=9102,
        ctx_size=65536,
        no_mmap=False,
        path=cfg_path,
    )
    ids_after_add = swap_config.list_model_ids(cfg_path)
    assert "phi4-mini" in ids_after_add
    data = swap_config.load(cfg_path)
    cmd = data["models"]["phi4-mini"]["cmd"]
    assert "--model /models/phi4-mini/model.gguf" in cmd
    assert "--port 9102" in cmd
    assert "--ctx-size 65536" in cmd
    assert "--no-mmap" not in cmd
    assert data["models"]["phi4-mini"]["proxy"] == "http://127.0.0.1:9102"

    swap_config.remove_model("phi4-mini", path=cfg_path)
    ids_after_remove = swap_config.list_model_ids(cfg_path)
    assert "phi4-mini" not in ids_after_remove
    assert "llama3-8b" in ids_after_remove
    assert "qwen3-4b" in ids_after_remove


def test_preserves_comments(cfg_path: Path) -> None:
    swap_config.add_model(
        model_id="phi4-mini",
        gguf_path=Path("/models/phi4-mini/model.gguf"),
        mmproj_path=None,
        port=9102,
        ctx_size=65536,
        no_mmap=False,
        path=cfg_path,
    )
    swap_config.remove_model("phi4-mini", path=cfg_path)
    raw = cfg_path.read_text()
    assert "# llama-swap config" in raw
    assert "# global settings" in raw
    assert "# existing model" in raw


def test_next_free_port_scans_existing(cfg_path: Path) -> None:
    port = swap_config.next_free_port(9100, path=cfg_path)
    assert port == 9102


def test_next_free_port_empty_config(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("models: {}\n")
    port = swap_config.next_free_port(9100, path=path)
    assert port == 9100


def test_next_free_port_respects_base(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "models:\n"
        "  a:\n"
        "    cmd: /x --port 8050\n"
        "    proxy: http://127.0.0.1:8050\n"
    )
    port = swap_config.next_free_port(9100, path=path)
    assert port == 9100


def test_next_free_port_exhaustion_raises(tmp_path: Path) -> None:
    lines = ["models:"]
    for i in range(9100, 9100 + 101):
        lines.append(f"  m{i}:")
        lines.append(f"    cmd: /x --port {i}")
        lines.append(f"    proxy: http://127.0.0.1:{i}")
    path = tmp_path / "full.yaml"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(RuntimeError):
        swap_config.next_free_port(9100, path=path)


def test_add_model_with_mmproj_and_large_file(cfg_path: Path) -> None:
    swap_config.add_model(
        model_id="vision-70b",
        gguf_path=Path("/models/vision-70b/model.gguf"),
        mmproj_path=Path("/models/vision-70b/mmproj.gguf"),
        port=9105,
        ctx_size=131072,
        no_mmap=True,
        path=cfg_path,
    )
    data = swap_config.load(cfg_path)
    cmd = data["models"]["vision-70b"]["cmd"]
    assert "--mmproj /models/vision-70b/mmproj.gguf" in cmd
    assert "--no-mmap" in cmd
    assert "--ctx-size 131072" in cmd


def test_save_is_atomic_and_parses_back(cfg_path: Path) -> None:
    data = swap_config.load(cfg_path)
    swap_config.save(data, cfg_path)
    yaml = YAML()
    reparsed = yaml.load(cfg_path.read_text())
    assert "models" in reparsed
    assert list(reparsed["models"].keys()) == ["llama3-8b", "qwen3-4b"]


def test_validate_happy_and_empty(tmp_path: Path, cfg_path: Path) -> None:
    assert swap_config.validate(cfg_path) is True
    empty = tmp_path / "empty.yaml"
    empty.write_text("models: {}\n")
    assert swap_config.validate(empty) is False


def test_backup_and_restore(cfg_path: Path) -> None:
    original = cfg_path.read_text()
    bak = swap_config.backup(cfg_path)
    assert bak is not None and bak.exists()
    swap_config.remove_model("llama3-8b", path=cfg_path)
    assert "llama3-8b" not in swap_config.list_model_ids(cfg_path)
    swap_config.restore_backup(cfg_path)
    assert cfg_path.read_text() == original


# ---------------------------------------------------------------------------
# Test FIX 5: add_model with vulkan-mtp backend applies backend flags
# ---------------------------------------------------------------------------

def test_add_model_vulkan_mtp_injects_backend_flags(tmp_path: Path) -> None:
    """add_model using the vulkan-mtp binary must inject --spec-type draft-mtp and --parallel 1."""
    import os
    import app.registry as rmod
    from unittest.mock import patch
    from app.backends_registry import BackendsRegistry

    reg_path = tmp_path / "registry.json"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("models: {}\n")

    # Seed backends into tmp registry
    old_path_fn = rmod._path
    rmod._path = lambda: reg_path
    try:
        br = BackendsRegistry()
        # Wire br to use tmp registry
        def _read():
            return rmod.get_backends_raw()
        def _write(data):
            rmod.set_backends_raw(data)
        br._read = _read
        br._write = _write
        br.seed_defaults_if_empty()
    finally:
        rmod._path = old_path_fn

    mtp_binary = "/usr/local/bin/llama-server-vulkan-mtp"

    with (
        patch("app.swap_config.llama_server_binary", return_value=mtp_binary),
        patch("app.registry._path", lambda: reg_path),
    ):
        swap_config.add_model(
            model_id="mtp-model",
            gguf_path=Path("/models/mtp-model/model.gguf"),
            mmproj_path=None,
            port=9150,
            ctx_size=65536,
            no_mmap=False,
            path=cfg_path,
        )

    data = swap_config.load(cfg_path)
    cmd = data["models"]["mtp-model"]["cmd"]

    assert mtp_binary in cmd, f"Expected MTP binary in cmd: {cmd}"
    assert "--spec-type draft-mtp" in cmd, f"Expected --spec-type draft-mtp in cmd: {cmd}"
    assert "--parallel 1" in cmd, f"Expected --parallel 1 in cmd: {cmd}"


# --- KV-aware co-residency budget -----------------------------------------

def test_cmd_kv_params_reads_ctx_and_cache_type() -> None:
    ctx, factor = swap_config._cmd_kv_params(
        "llama-server --ctx-size 262144 --cache-type-k q8_0 --cache-type-v q8_0"
    )
    assert ctx == 262144
    assert factor == 1.0


def test_cmd_kv_params_defaults_to_f16_when_unflagged() -> None:
    """llama.cpp defaults to f16, i.e. 2x the q8_0 baseline the tiers assume —
    an unflagged model must reserve MORE, not less."""
    ctx, factor = swap_config._cmd_kv_params("llama-server --ctx-size 65536")
    assert ctx == 65536
    assert factor == 2.0


def test_cmd_kv_params_accepts_short_flags_and_averages_mixed_types() -> None:
    ctx, factor = swap_config._cmd_kv_params("llama-server -c 32768 -ctk f16 -ctv q8_0")
    assert ctx == 32768
    assert factor == pytest.approx(1.5)


def test_cmd_kv_params_handles_empty_cmd() -> None:
    ctx, factor = swap_config._cmd_kv_params("")
    assert ctx is None
    assert factor == 2.0


def test_kv_reserve_scales_with_cache_type() -> None:
    gib = 1024 ** 3
    assert swap_config._kv_reserve_bytes(262144, 1.0) == 32 * gib
    assert swap_config._kv_reserve_bytes(262144, 2.0) == 64 * gib
    assert swap_config._kv_reserve_bytes(262144, 0.5) == 16 * gib


def test_kv_bytes_falls_back_to_tiers_when_gguf_unreadable(monkeypatch) -> None:
    monkeypatch.setattr(swap_config, "_main_gguf", lambda _mid: None)
    assert swap_config._kv_bytes("absent-model", 65536, 1.0) == 8 * 1024 ** 3
    assert swap_config._kv_bytes("absent-model", 65536, 2.0) == 16 * 1024 ** 3


def test_kv_bytes_prefers_exact_gguf_geometry_over_tiers(monkeypatch, tmp_path: Path) -> None:
    """A hybrid 27B at 256k needs 8 GiB of q8_0 KV — the tier would claim 32."""
    fake = tmp_path / "w.gguf"
    fake.write_bytes(b"x")
    monkeypatch.setattr(swap_config, "_main_gguf", lambda _mid: fake)
    monkeypatch.setattr("app.gguf_meta.kv_bytes_per_token", lambda _p: 64 * 1024)
    assert swap_config._kv_bytes("hybrid", 262144, 1.0) == 8 * 1024 ** 3
    assert swap_config._kv_bytes("hybrid", 262144, 2.0) == 16 * 1024 ** 3
