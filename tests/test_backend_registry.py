"""Tests for PR1-PR4: BackendsRegistry, model predicates, and flag injection."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(tmp_path: Path) -> "BackendsRegistry":
    """Return a BackendsRegistry backed by a temp registry.json."""
    from app.backends_registry import BackendsRegistry

    reg_path = tmp_path / "registry.json"
    with patch.dict(os.environ, {"SWAPCTL_REGISTRY_PATH": str(reg_path)}):
        # Re-import settings to pick up the env override
        import importlib
        import app.config as _cfg_mod
        import app.registry as _reg_mod

        # Patch the settings object path so registry reads/writes go to tmp
        original = _reg_mod._path

        def _patched_path():
            return reg_path

        _reg_mod._path = _patched_path
        br = BackendsRegistry()
        # Monkey-patch the internal helpers to use our tmp registry path
        original_read = br._read
        original_write = br._write

        def _read():
            from app.registry import get_backends_raw
            import app.registry as rmod
            old = rmod._path
            rmod._path = _patched_path
            try:
                return get_backends_raw()
            finally:
                rmod._path = old

        def _write(data):
            from app.registry import set_backends_raw
            import app.registry as rmod
            old = rmod._path
            rmod._path = _patched_path
            try:
                set_backends_raw(data)
            finally:
                rmod._path = old

        br._read = _read
        br._write = _write
        return br


def _make_model(filename: str = "model.gguf", model_tags: list | None = None):
    """Build a minimal ModelRecord-like object for testing."""
    from app.registry import ModelRecord
    from dataclasses import asdict
    rec = ModelRecord(
        model_id="test-model",
        repo_id=None,
        filename=filename,
        mmproj=None,
        installed_at="2025-01-01T00:00:00+00:00",
        hf_last_modified=None,
        size_bytes=0,
        ctx_size=65536,
    )
    # Attach model_tags as a plain attribute (not a dataclass field)
    object.__setattr__(rec, "model_tags", model_tags or []) if False else None
    rec.__dict__["model_tags"] = model_tags or []
    return rec


# ---------------------------------------------------------------------------
# Test 1: seed produces 4 default backends
# ---------------------------------------------------------------------------

def test_seed_produces_four_backends(tmp_path: Path) -> None:
    br = _make_registry(tmp_path)
    br.seed_defaults_if_empty()
    backends = br.list_all()
    ids = [b.id for b in backends]
    assert len(backends) == 4, f"expected 4 backends, got {len(backends)}: {ids}"
    assert "vulkan" in ids
    assert "hip" in ids
    assert "rock" in ids
    assert "vulkan-mtp" in ids


def test_seed_is_idempotent(tmp_path: Path) -> None:
    br = _make_registry(tmp_path)
    br.seed_defaults_if_empty()
    br.seed_defaults_if_empty()  # second call should be a no-op
    assert len(br.list_all()) == 4


def test_seed_sets_correct_mtp_fields(tmp_path: Path) -> None:
    br = _make_registry(tmp_path)
    br.seed_defaults_if_empty()
    mtp = br.get("vulkan-mtp")
    assert mtp is not None
    assert mtp.binary_path == "/usr/local/bin/llama-server-vulkan-mtp"
    assert mtp.model_predicate == "has_mtp_heads"
    assert "--parallel 1" in mtp.required_flags
    assert any("spec-type" in f for f in mtp.default_flags)
    assert any("spec-draft-n-max" in f for f in mtp.default_flags)


# ---------------------------------------------------------------------------
# Test 2: compatible() includes vulkan-mtp only for MTP models
# ---------------------------------------------------------------------------

def test_compatible_includes_mtp_for_mtp_model(tmp_path: Path) -> None:
    br = _make_registry(tmp_path)
    br.seed_defaults_if_empty()

    # Model with mtp_heads tag
    model_with_mtp = _make_model(filename="qwen3.6-35b-a3b-mtp-ud-q3-k-xl.gguf", model_tags=["mtp_heads"])
    compat_ids = {b.id for b in br.compatible(model_with_mtp)}
    assert "vulkan-mtp" in compat_ids, "vulkan-mtp should be compatible with MTP model"
    assert "vulkan" in compat_ids
    assert "hip" in compat_ids
    assert "rock" in compat_ids


def test_compatible_excludes_mtp_for_non_mtp_model(tmp_path: Path) -> None:
    br = _make_registry(tmp_path)
    br.seed_defaults_if_empty()

    # Model without mtp_heads tag
    model_no_mtp = _make_model(filename="qwen3-4b-instruct-q8.gguf", model_tags=[])
    compat_ids = {b.id for b in br.compatible(model_no_mtp)}
    assert "vulkan-mtp" not in compat_ids, "vulkan-mtp should NOT be compatible with non-MTP model"
    assert "vulkan" in compat_ids


def test_compatible_uses_filename_heuristic(tmp_path: Path) -> None:
    """has_mtp_heads falls back to filename heuristic when no tags set."""
    br = _make_registry(tmp_path)
    br.seed_defaults_if_empty()

    # Filename contains '-mtp-' — should trigger heuristic
    model_mtp_name = _make_model(filename="some-model-mtp-q4.gguf", model_tags=[])
    from app.backends_registry import has_mtp_heads
    assert has_mtp_heads(model_mtp_name), "filename heuristic should detect -mtp- in filename"

    model_no_mtp = _make_model(filename="llama3-8b-q8.gguf", model_tags=[])
    assert not has_mtp_heads(model_no_mtp), "plain model should not be detected as MTP"


# ---------------------------------------------------------------------------
# Test 3: switching to vulkan-mtp from --parallel 4 cmd
# ---------------------------------------------------------------------------

def _make_swap_config_with_model(tmp_path: Path, model_id: str, cmd: str):
    """Write a minimal llama-swap config.yaml with one model."""
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap

    y = YAML()
    data = CommentedMap()
    data["models"] = CommentedMap()
    block = CommentedMap()
    block["cmd"] = cmd
    block["proxy"] = "http://127.0.0.1:9100"
    data["models"][model_id] = block
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        y.dump(data, f)
    return cfg_path


def test_switch_to_mtp_replaces_parallel_4_with_1(tmp_path: Path) -> None:
    """Switching to vulkan-mtp must inject --parallel 1 and drop --parallel 4."""
    # Seed registry in tmp
    br = _make_registry(tmp_path)
    br.seed_defaults_if_empty()

    model_id = "qwen3-mtp-test"
    orig_cmd = (
        "/usr/local/bin/llama-server-vulkan "
        "--model /models/qwen3-mtp.gguf "
        "--port 9100 -ngl 999 "
        "--cache-type-k q8_0 --cache-type-v q8_0 "
        "--jinja --ctx-size 65536 --parallel 4"
    )
    cfg_path = _make_swap_config_with_model(tmp_path, model_id, orig_cmd)

    # Need swap_config to use our tmp registry
    import app.registry as rmod
    reg_path = tmp_path / "registry.json"
    old_path_fn = rmod._path
    rmod._path = lambda: reg_path

    try:
        from app import swap_config
        swap_config.switch_model_backend(model_id, "vulkan-mtp", path=cfg_path)
    finally:
        rmod._path = old_path_fn

    from app import swap_config as sc
    data = sc.load(cfg_path)
    result_cmd = data["models"][model_id]["cmd"]

    assert "/usr/local/bin/llama-server-vulkan-mtp" in result_cmd
    assert "--parallel 1" in result_cmd
    assert "--parallel 4" not in result_cmd
    # Count occurrences of --parallel to ensure no duplicates
    import re
    parallel_matches = re.findall(r"--parallel\s+\S+", result_cmd)
    assert len(parallel_matches) == 1, f"expected exactly 1 --parallel, got {parallel_matches}"
    assert "--spec-type draft-mtp" in result_cmd
    assert "--spec-draft-n-max 3" in result_cmd


# ---------------------------------------------------------------------------
# Test 4: switching back to vulkan from vulkan-mtp strips MTP flags
# ---------------------------------------------------------------------------

def test_switch_back_to_vulkan_strips_mtp_flags(tmp_path: Path) -> None:
    """Switching from vulkan-mtp back to vulkan removes MTP-injected flags."""
    br = _make_registry(tmp_path)
    br.seed_defaults_if_empty()

    model_id = "qwen3-mtp-roundtrip"
    mtp_cmd = (
        "/usr/local/bin/llama-server-vulkan-mtp "
        "--model /models/qwen3-mtp.gguf "
        "--port 9100 -ngl 999 "
        "--cache-type-k q8_0 "
        "--jinja --ctx-size 65536 "
        "--parallel 1 "
        "--spec-type draft-mtp --spec-draft-n-max 3"
    )
    cfg_path = _make_swap_config_with_model(tmp_path, model_id, mtp_cmd)

    import app.registry as rmod
    reg_path = tmp_path / "registry.json"
    old_path_fn = rmod._path
    rmod._path = lambda: reg_path

    try:
        from app import swap_config
        swap_config.switch_model_backend(model_id, "vulkan", path=cfg_path)
    finally:
        rmod._path = old_path_fn

    from app import swap_config as sc
    data = sc.load(cfg_path)
    result_cmd = data["models"][model_id]["cmd"]

    assert "/usr/local/bin/llama-server-vulkan" in result_cmd
    assert "/usr/local/bin/llama-server-vulkan-mtp" not in result_cmd
    # MTP-specific flags should be gone (they were injected by vulkan-mtp backend)
    assert "--spec-type" not in result_cmd
    assert "--spec-draft-n-max" not in result_cmd
    # --parallel is removed when switching to vulkan (no required_flags for vulkan)
    # This is acceptable behavior — documented in spec: "or leaves them — your choice"
    # We chose to strip them since they were required_flags of the source backend


# ---------------------------------------------------------------------------
# Test FIX 3: model_tags round-trips through registry write/read
# ---------------------------------------------------------------------------

def test_model_tags_roundtrip_registry(tmp_path: Path) -> None:
    """model_tags field must survive a registry upsert + read cycle."""
    import app.registry as rmod
    reg_path = tmp_path / "registry.json"
    old_path_fn = rmod._path
    rmod._path = lambda: reg_path
    try:
        rec = rmod.ModelRecord(
            model_id="mtp-test",
            repo_id=None,
            filename="mtp-model-q4.gguf",
            mmproj=None,
            installed_at="2025-01-01T00:00:00+00:00",
            hf_last_modified=None,
            size_bytes=12345,
            ctx_size=32768,
            model_tags=["mtp_heads", "moe"],
        )
        rmod.upsert(rec)
        loaded = rmod.get("mtp-test")
    finally:
        rmod._path = old_path_fn

    assert loaded is not None
    assert loaded.model_tags == ["mtp_heads", "moe"], (
        f"model_tags not persisted correctly: {loaded.model_tags}"
    )


# ---------------------------------------------------------------------------
# Test FIX 7: POST /backend/{model_id} rejects incompatible backends
# ---------------------------------------------------------------------------

def test_switch_model_backend_api_rejects_incompatible(tmp_path: Path) -> None:
    """POST /api/backend/{model_id} must return 400 when backend is incompatible."""
    import app.registry as rmod
    from app.backends_registry import BackendDef
    from unittest.mock import patch

    # Plain model record (no MTP) — vulkan-mtp should be incompatible
    plain_rec = rmod.ModelRecord(
        model_id="plain-model",
        repo_id=None,
        filename="plain-model-q8.gguf",
        mmproj=None,
        installed_at="2025-01-01T00:00:00+00:00",
        hf_last_modified=None,
        size_bytes=0,
        ctx_size=65536,
        model_tags=[],
    )

    # Backends known to the registry (all 4)
    all_backends = [
        BackendDef(id="vulkan", binary_path="/usr/local/bin/llama-server-vulkan", label="Vulkan"),
        BackendDef(id="hip", binary_path="/usr/local/bin/llama-server-hip", label="HIP"),
        BackendDef(id="rock", binary_path="/usr/local/bin/llama-server-rock", label="Rock"),
        BackendDef(id="vulkan-mtp", binary_path="/usr/local/bin/llama-server-vulkan-mtp", label="Vulkan+MTP",
                   model_predicate="has_mtp_heads"),
    ]
    # Compatible backends for a plain model: all except vulkan-mtp
    compat_backends = [b for b in all_backends if b.id != "vulkan-mtp"]

    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from app.routers.api import router as api_router

    app_fa = FastAPI()
    app_fa.include_router(api_router)

    with (
        patch("app.backends_registry.BackendsRegistry.list_all", return_value=all_backends),
        patch("app.backends_registry.BackendsRegistry.compatible", return_value=compat_backends),
        patch("app.registry.get", return_value=plain_rec),
        patch("app.swap_config.detect_model_backend", return_value="vulkan"),
    ):
        client = TestClient(app_fa, raise_server_exceptions=True)
        resp = client.post(
            "/api/backend/plain-model",
            data={"backend": "vulkan-mtp"},
        )

    assert resp.status_code == 400, (
        f"Expected 400 for incompatible backend, got {resp.status_code}: {resp.text}"
    )
