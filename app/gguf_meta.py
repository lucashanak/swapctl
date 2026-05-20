"""Minimal GGUF header reader — extracts metadata fields without pulling in `gguf` package."""
from __future__ import annotations

import struct
from pathlib import Path


_TYPE_UINT8 = 0
_TYPE_INT8 = 1
_TYPE_UINT16 = 2
_TYPE_INT16 = 3
_TYPE_UINT32 = 4
_TYPE_INT32 = 5
_TYPE_FLOAT32 = 6
_TYPE_BOOL = 7
_TYPE_STRING = 8
_TYPE_ARRAY = 9
_TYPE_UINT64 = 10
_TYPE_INT64 = 11
_TYPE_FLOAT64 = 12


def _read_value(f, value_type: int):
    if value_type == _TYPE_UINT8:   return struct.unpack("<B", f.read(1))[0]
    if value_type == _TYPE_INT8:    return struct.unpack("<b", f.read(1))[0]
    if value_type == _TYPE_UINT16:  return struct.unpack("<H", f.read(2))[0]
    if value_type == _TYPE_INT16:   return struct.unpack("<h", f.read(2))[0]
    if value_type == _TYPE_UINT32:  return struct.unpack("<I", f.read(4))[0]
    if value_type == _TYPE_INT32:   return struct.unpack("<i", f.read(4))[0]
    if value_type == _TYPE_FLOAT32: return struct.unpack("<f", f.read(4))[0]
    if value_type == _TYPE_BOOL:    return bool(struct.unpack("<B", f.read(1))[0])
    if value_type == _TYPE_STRING:
        slen = struct.unpack("<Q", f.read(8))[0]
        return f.read(slen).decode("utf-8", errors="replace")
    if value_type == _TYPE_ARRAY:
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        if arr_len <= 1024:
            return [_read_value(f, arr_type) for _ in range(arr_len)]
        sizes = {_TYPE_UINT8:1,_TYPE_INT8:1,_TYPE_UINT16:2,_TYPE_INT16:2,
                 _TYPE_UINT32:4,_TYPE_INT32:4,_TYPE_FLOAT32:4,_TYPE_BOOL:1,
                 _TYPE_UINT64:8,_TYPE_INT64:8,_TYPE_FLOAT64:8}
        size = sizes.get(arr_type)
        if size:
            f.read(size * arr_len)
            return None
        return [_read_value(f, arr_type) for _ in range(arr_len)]
    if value_type == _TYPE_UINT64:  return struct.unpack("<Q", f.read(8))[0]
    if value_type == _TYPE_INT64:   return struct.unpack("<q", f.read(8))[0]
    if value_type == _TYPE_FLOAT64: return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"unknown GGUF value type {value_type}")


def read_gguf_metadata(path: Path, interesting_keys: tuple = ("context_length",)) -> dict:
    """Parse GGUF metadata section; return a dict of key → value.

    Stops early once all `interesting_keys` (matched as suffix after the arch prefix) are found
    to avoid reading entire metadata block for huge models.
    """
    out: dict = {}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return out
            _version = struct.unpack("<I", f.read(4))[0]
            _tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]
            remaining = set(interesting_keys)
            for _ in range(kv_count):
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode("utf-8", errors="replace")
                value_type = struct.unpack("<I", f.read(4))[0]
                value = _read_value(f, value_type)
                out[key] = value
                for wanted in list(remaining):
                    if key == wanted or key.endswith("." + wanted):
                        remaining.discard(wanted)
                if not remaining:
                    break
    except (OSError, ValueError, struct.error):
        pass
    return out


def read_context_length(path: Path) -> int | None:
    """Extract n_ctx_train from GGUF file. Returns None on failure."""
    meta = read_gguf_metadata(path, ("context_length",))
    for k, v in meta.items():
        if (k == "context_length" or k.endswith(".context_length")) and isinstance(v, int):
            return v
    return None


_VRAM_KEYS = (
    "general.architecture", "general.name",
    "block_count", "context_length", "embedding_length",
    "attention.head_count", "attention.head_count_kv",
    "attention.key_length", "attention.value_length",
    "attention.sliding_window_size", "attention.sliding_window",
)


def estimate_vram(path: Path, context_sizes: list[int] | None = None, overhead_gib: float = 2.0) -> dict | None:
    """Estimate VRAM for a GGUF model at various context sizes.

    Returns {"model_name", "max_context", "model_size_mb", "estimates": [{"ctx", "kv_mb", "total_mb"}]}
    or None on failure.
    """
    import os
    import re

    meta = read_gguf_metadata(path, _VRAM_KEYS)
    if not meta:
        return None

    arch = None
    for k, v in meta.items():
        if k == "general.architecture" and isinstance(v, str):
            arch = v
            break
    if not arch:
        return None

    def _get(suffix: str) -> int | None:
        for k, v in meta.items():
            if k == suffix or k.endswith("." + suffix):
                if isinstance(v, int):
                    return v
                if isinstance(v, list) and v and isinstance(v[0], int):
                    return max(v)
        return None

    n_layers = _get("block_count")
    n_head = _get("attention.head_count")
    n_head_kv = _get("attention.head_count_kv") or n_head
    n_embd_k = _get("attention.key_length")
    n_embd_v = _get("attention.value_length")
    if not n_embd_k or not n_embd_v:
        n_embd = _get("embedding_length")
        if n_embd and n_head:
            head_dim = n_embd // n_head
            n_embd_k = n_embd_k or head_dim
            n_embd_v = n_embd_v or head_dim
    if not all((n_layers, n_head_kv, n_embd_k, n_embd_v)):
        return None

    training_ctx = _get("context_length") or 0
    swa = _get("attention.sliding_window_size") or _get("attention.sliding_window") or 0
    model_name = meta.get("general.name", path.stem)

    # Multi-part model size
    model_bytes = 0
    match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", str(path), re.IGNORECASE)
    if match:
        base = str(path)[:match.start()]
        total_parts = int(match.group(2))
        for i in range(1, total_parts + 1):
            part = f"{base}-{i:05d}-of-{match.group(2)}.gguf"
            if os.path.exists(part):
                model_bytes += os.path.getsize(part)
    else:
        model_bytes = os.path.getsize(str(path))

    overhead_bytes = int(overhead_gib * 1024**3)

    if swa > 0:
        n_layers_swa, n_layers_full = n_layers, 0
    else:
        n_layers_swa, n_layers_full = 0, n_layers

    bytes_per_token_per_layer = n_head_kv * (n_embd_k + n_embd_v) * 2

    if context_sizes is None:
        context_sizes = [4096, 8192, 16384, 32768, 65536, 131072]
    if training_ctx > 0:
        context_sizes = [c for c in context_sizes if c <= training_ctx]
        if training_ctx not in context_sizes:
            context_sizes.append(training_ctx)
    context_sizes = sorted(set(context_sizes))

    mb = 1024 * 1024
    estimates = []
    for ctx in context_sizes:
        mem_full = ctx * n_layers_full * bytes_per_token_per_layer
        mem_swa = min(ctx, swa) * n_layers_swa * bytes_per_token_per_layer if swa > 0 else 0
        kv_bytes = mem_full + mem_swa
        total = model_bytes + kv_bytes + overhead_bytes
        estimates.append({
            "ctx": ctx,
            "kv_mb": round(kv_bytes / mb),
            "total_mb": round(total / mb),
        })

    return {
        "model_name": model_name,
        "max_context": training_ctx,
        "model_size_mb": round(model_bytes / mb),
        "estimates": estimates,
    }


def read_tensor_names(path: Path) -> list[str]:
    """Read tensor names from GGUF file header.

    Stops after the tensor info section (does not read weight data).
    Returns empty list on any failure.
    """
    names: list[str] = []
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return names
            _version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]
            # Skip all metadata key-value pairs
            for _ in range(kv_count):
                key_len = struct.unpack("<Q", f.read(8))[0]
                f.read(key_len)  # key bytes
                value_type = struct.unpack("<I", f.read(4))[0]
                _read_value(f, value_type)
            # Read tensor info: name, n_dims, dims[], type, offset
            for _ in range(min(tensor_count, 100_000)):
                name_len = struct.unpack("<Q", f.read(8))[0]
                name = f.read(name_len).decode("utf-8", errors="replace")
                names.append(name)
                n_dims = struct.unpack("<I", f.read(4))[0]
                f.read(8 * n_dims)  # dims (uint64 each)
                f.read(4)           # ggml_type (uint32)
                f.read(8)           # offset (uint64)
    except (OSError, ValueError, struct.error):
        pass
    return names


def read_model_tags(path: Path) -> list[str]:
    """Return list of tag strings (e.g. 'moe', 'hybrid', 'mtp_heads') derived from GGUF metadata."""
    meta = read_gguf_metadata(path, ("expert_count", "general.architecture"))
    tags: list[str] = []
    for k, v in meta.items():
        if k.endswith(".expert_count") and isinstance(v, int) and v > 0:
            tags.append("moe")
            break
    for k, v in meta.items():
        if k == "general.architecture" and isinstance(v, str):
            if "hybrid" in v.lower():
                tags.append("hybrid")
            break

    # MTP heads detection — check tensor names for mtp.* pattern
    # Fall back to filename heuristic if tensor read fails or returns nothing
    try:
        tensor_names = read_tensor_names(path)
        has_mtp = any(
            n.startswith("mtp") or ".mtp" in n
            for n in tensor_names
        )
    except Exception:
        has_mtp = False

    if not has_mtp:
        # Filename heuristic
        stem = str(path.name)
        has_mtp = "-mtp-" in stem.lower()

    if has_mtp:
        tags.append("mtp_heads")

    return tags
