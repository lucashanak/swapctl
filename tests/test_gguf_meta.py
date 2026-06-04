"""Smoke tests for app.gguf_meta.read_architecture."""
from __future__ import annotations

import struct
from pathlib import Path

import pytest


def _write_minimal_gguf(path: Path, arch: str) -> None:
    """Write a minimal valid GGUF file with general.architecture set to arch."""
    key = b"general.architecture"
    arch_bytes = arch.encode("utf-8")

    def pack_str(s: bytes) -> bytes:
        return struct.pack("<Q", len(s)) + s

    kv_count = 1
    value_type_string = 8

    kv_block = (
        pack_str(key)
        + struct.pack("<I", value_type_string)
        + pack_str(arch_bytes)
    )

    with open(path, "wb") as f:
        f.write(b"GGUF")                        # magic
        f.write(struct.pack("<I", 3))            # version
        f.write(struct.pack("<Q", 0))            # tensor_count
        f.write(struct.pack("<Q", kv_count))     # kv_count
        f.write(kv_block)


def test_read_architecture_from_synthetic_gguf(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf, "llama")
    from app.gguf_meta import read_architecture
    assert read_architecture(gguf) == "llama"


def test_read_architecture_returns_none_on_missing_file(tmp_path: Path) -> None:
    from app.gguf_meta import read_architecture
    result = read_architecture(tmp_path / "nonexistent.gguf")
    assert result is None


def test_read_architecture_returns_none_on_invalid_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"notgguf")
    from app.gguf_meta import read_architecture
    assert read_architecture(bad) is None


def test_read_architecture_custom_arch(tmp_path: Path) -> None:
    gguf = tmp_path / "custom.gguf"
    _write_minimal_gguf(gguf, "gemma4_assistant")
    from app.gguf_meta import read_architecture
    assert read_architecture(gguf) == "gemma4_assistant"


def test_read_architecture_real_model() -> None:
    """Smoke test against a real model file if any exist under /models/."""
    import os
    models_root = Path("/models")
    if not models_root.exists():
        pytest.skip("no /models/ directory on this machine")

    gguf_files = list(models_root.rglob("*.gguf"))
    if not gguf_files:
        pytest.skip("no .gguf files found under /models/")

    from app.gguf_meta import read_architecture
    path = gguf_files[0]
    result = read_architecture(path)
    # result may be None for malformed/partial files; just check it doesn't raise
    assert result is None or isinstance(result, str)
