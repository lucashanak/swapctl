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
