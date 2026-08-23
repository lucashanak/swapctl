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


def _write_gguf_kv(path: Path, pairs: dict) -> None:
    """Minimal GGUF carrying typed KV pairs: str values as (8, str), ints as (4, int)."""
    def pack_str(s: bytes) -> bytes:
        return struct.pack("<Q", len(s)) + s

    blocks = b""
    for key, (vtype, value) in pairs.items():
        blocks += pack_str(key.encode())
        blocks += struct.pack("<I", vtype)
        blocks += pack_str(value.encode()) if vtype == 8 else struct.pack("<I", int(value))

    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 0))
        f.write(struct.pack("<Q", len(pairs)))
        f.write(blocks)


_QWEN35_GEOMETRY = {
    "general.architecture": (8, "qwen35"),
    "qwen35.block_count": (4, 64),
    "qwen35.attention.head_count": (4, 24),
    "qwen35.attention.head_count_kv": (4, 4),
    "qwen35.attention.key_length": (4, 256),
    "qwen35.attention.value_length": (4, 256),
}


def test_kv_bytes_per_token_hybrid_counts_only_full_attention_layers(tmp_path: Path) -> None:
    """Qwen3.8-27B grows KV in 16 of 64 layers; counting all 64 over-reserves 4x."""
    from app.gguf_meta import kv_bytes_per_token

    gguf = tmp_path / "hybrid.gguf"
    _write_gguf_kv(gguf, {**_QWEN35_GEOMETRY,
                          "qwen35.full_attention_interval": (4, 4)})
    # 4 kv-heads * (256+256) * 2 bytes * (64/4) layers = 65536 B/token = 64 KiB
    assert kv_bytes_per_token(gguf) == 4 * (256 + 256) * 2 * 16
    assert kv_bytes_per_token(gguf) == 64 * 1024


def test_kv_bytes_per_token_dense_counts_every_layer(tmp_path: Path) -> None:
    from app.gguf_meta import kv_bytes_per_token

    gguf = tmp_path / "dense.gguf"
    _write_gguf_kv(gguf, _QWEN35_GEOMETRY)          # no full_attention_interval
    assert kv_bytes_per_token(gguf) == 4 * (256 + 256) * 2 * 64


def test_kv_bytes_per_token_returns_none_without_geometry(tmp_path: Path) -> None:
    from app.gguf_meta import kv_bytes_per_token

    gguf = tmp_path / "bare.gguf"
    _write_minimal_gguf(gguf, "llama")
    assert kv_bytes_per_token(gguf) is None
