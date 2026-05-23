"""Backend registry — wraps registry.json `backends` section.

Defines BackendDef dataclass, model predicates, and BackendsRegistry class.
Default seed: vulkan, hip, rock, vulkan-mtp.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import ModelRecord

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class BackendDef:
    id: str
    binary_path: str
    label: str
    caps: list[str] = field(default_factory=list)
    default_flags: list[str] = field(default_factory=list)
    required_flags: list[str] = field(default_factory=list)
    incompatible_flag_patterns: list[str] = field(default_factory=list)
    model_predicate: str | None = None
    version_hint: str = ""
    builtin: bool = True


# ---------------------------------------------------------------------------
# Predicate functions
# ---------------------------------------------------------------------------

def has_mtp_heads(model: "ModelRecord") -> bool:
    """Return True if the model's GGUF contains MTP prediction head tensors.

    Strategy (in order of reliability):
    1. Check model_tags for 'mtp_heads' (set by gguf_meta.read_model_tags).
    2. Filename heuristic: '-mtp-' or '-MTP-' in the filename.
    """
    # Check tags list set by gguf_meta
    tags = getattr(model, "model_tags", None) or []
    if "mtp_heads" in tags:
        return True

    # Filename heuristic
    filename = (getattr(model, "filename", "") or "").lower()
    if "-mtp-" in filename or "mtp_" in filename or "_mtp" in filename:
        return True

    # Model id heuristic (swapctl directory key often contains -mtp- even if filename doesn't)
    mid = (getattr(model, "model_id", "") or "").lower()
    if "-mtp-" in mid or mid.endswith("-mtp") or mid.startswith("mtp-"):
        return True

    # Explicit MTP draft pairing: model has draft_model pointing to a -mtp- draft
    draft_mid = (getattr(model, "draft_model", "") or "").lower()
    if "-mtp-" in draft_mid or "mtp-" in draft_mid or "-mtp" in draft_mid:
        return True

    return False


def is_moe(model: "ModelRecord") -> bool:
    """Return True if the model is a Mixture-of-Experts model."""
    tags = getattr(model, "model_tags", None) or []
    return "moe" in tags


def is_qwen3_nextn(model: "ModelRecord") -> bool:
    """Return True if model is Qwen3.x — NextN speculative decoding uses same main GGUF as draft.

    Qwen3, Qwen3.5, Qwen3.6 all support NextN. Excludes draft/mtp companion files.
    """
    mid = (getattr(model, "model_id", "") or "").lower()
    fn = (getattr(model, "filename", "") or "").lower()
    if "mtp" in mid or "mtp" in fn or "draft" in mid:
        return False
    return ("qwen3" in mid) or ("qwen3" in fn)


def is_zaya_arch(model: "ModelRecord") -> bool:
    """Return True if the model uses the Zyphra ZAYA1 'zaya' architecture.

    Detected by 'zaya' tag (set by gguf_meta) or 'zaya' substring in model_id/filename.
    """
    tags = getattr(model, "model_tags", None) or []
    if "zaya" in tags:
        return True
    mid = (getattr(model, "model_id", "") or "").lower()
    fn = (getattr(model, "filename", "") or "").lower()
    return "zaya" in mid or "zaya" in fn


def is_gemma4_main(model: "ModelRecord") -> bool:
    """Return True if the model is a Gemma 4 main model (target for the assistant draft).

    Excludes Gemma 4 assistant/draft companion models (which have 'assistant' in id/filename).
    """
    mid = (getattr(model, "model_id", "") or "").lower()
    fn = (getattr(model, "filename", "") or "").lower()
    if "assistant" in mid or "assistant" in fn:
        return False
    return ("gemma-4" in mid) or ("gemma4" in mid) or ("gemma-4" in fn) or ("gemma4" in fn)


PREDICATES: dict[str, object] = {
    "has_mtp_heads": has_mtp_heads,
    "is_moe": is_moe,
    "is_gemma4_main": is_gemma4_main,
    "is_qwen3_nextn": is_qwen3_nextn,
    "is_zaya_arch": is_zaya_arch,
}


# ---------------------------------------------------------------------------
# Default seed data
# ---------------------------------------------------------------------------

_DEFAULT_BACKENDS: list[dict] = [
    {
        "id": "vulkan",
        "binary_path": "/usr/local/bin/llama-server-vulkan",
        "label": "Vulkan",
        "caps": ["vulkan"],
        "default_flags": [],
        "required_flags": [],
        "incompatible_flag_patterns": [],
        "model_predicate": None,
        "version_hint": "",
        "builtin": True,
    },
    {
        "id": "hip",
        "binary_path": "/usr/local/bin/llama-server-hip",
        "label": "HIP",
        "caps": ["hip"],
        "default_flags": [],
        "required_flags": [],
        "incompatible_flag_patterns": [],
        "model_predicate": None,
        "version_hint": "",
        "builtin": True,
    },
    {
        "id": "rock",
        "binary_path": "/usr/local/bin/llama-server-rock",
        "label": "Rock",
        "caps": ["rock"],
        "default_flags": [],
        "required_flags": [],
        "incompatible_flag_patterns": [],
        "model_predicate": None,
        "version_hint": "",
        "builtin": True,
    },
    {
        "id": "vulkan-mtp",
        "binary_path": "/usr/local/bin/llama-server-vulkan-mtp",
        "label": "Vulkan + MTP",
        "caps": ["vulkan", "mtp"],
        "default_flags": ["--spec-type draft-mtp", "--spec-draft-n-max 3"],
        "required_flags": ["--parallel 1"],
        "incompatible_flag_patterns": [r"--parallel\s+([2-9]|\d{2,})"],
        "model_predicate": "has_mtp_heads",
        "version_hint": "am17an b9206",
        "builtin": False,
    },
    {
        "id": "vulkan-atomic-turbo",
        "binary_path": "/usr/local/bin/llama-server-vulkan-atomic",
        "label": "Vulkan + TurboQuant KV",
        "caps": ["vulkan", "turboquant"],
        "default_flags": ["--cache-type-k turbo3", "--cache-type-v turbo3"],
        "required_flags": [],
        "incompatible_flag_patterns": [r"--cache-type-k\s+(?!turbo)", r"--cache-type-v\s+(?!turbo)"],
        "model_predicate": None,
        "version_hint": "AtomicBot/atomic-llama-cpp-turboquant",
        "builtin": False,
    },
    {
        "id": "vulkan-atomic-nextn",
        "binary_path": "/usr/local/bin/llama-server-vulkan-atomic",
        "label": "Vulkan + Qwen NextN",
        "caps": ["vulkan", "nextn"],
        "default_flags": ["--spec-type nextn", "--draft-max 3"],
        "required_flags": ["--parallel 1"],
        "incompatible_flag_patterns": [
            r"--parallel\s+([2-9]|\d{2,})",
            r"--spec-draft-n-max\s+\d+",
            r"--cache-type-k\s+turbo\d",
            r"--cache-type-v\s+turbo\d",
        ],
        "model_predicate": "is_qwen3_nextn",
        "version_hint": "AtomicBot/atomic-llama-cpp-turboquant",
        "builtin": False,
    },
    {
        "id": "vulkan-zaya",
        "binary_path": "/usr/local/bin/llama-server-vulkan-zaya",
        "label": "Vulkan + ZAYA1",
        "caps": ["vulkan", "zaya"],
        "default_flags": [],
        "required_flags": [],
        "incompatible_flag_patterns": [],
        "model_predicate": "is_zaya_arch",
        "version_hint": "Juste-Leo2 PR #23112 draft",
        "builtin": False,
    },
    {
        "id": "hip-zaya",
        "binary_path": "/usr/local/bin/llama-server-hip-zaya",
        "label": "HIP + ZAYA1",
        "caps": ["hip", "zaya"],
        "default_flags": [],
        "required_flags": [],
        "incompatible_flag_patterns": [],
        "model_predicate": "is_zaya_arch",
        "version_hint": "Juste-Leo2 PR #23112 draft, ROCm 7.2.3 gfx1151",
        "builtin": False,
    },
    {
        "id": "vulkan-atomic-gemma4-mtp",
        "binary_path": "/usr/local/bin/llama-server-vulkan-atomic",
        "label": "Vulkan + Gemma4 MTP",
        "caps": ["vulkan", "mtp", "gemma4-mtp"],
        "default_flags": ["--spec-type mtp", "--draft-max 3"],
        "required_flags": ["--parallel 1"],
        "incompatible_flag_patterns": [
            r"--parallel\s+([2-9]|\d{2,})",
            r"--spec-draft-n-max\s+\d+",
            r"--cache-type-k\s+turbo\d",
            r"--cache-type-v\s+turbo\d",
        ],
        "model_predicate": "is_gemma4_main",
        "version_hint": "AtomicBot/atomic-llama-cpp-turboquant",
        "builtin": False,
    },
]


# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------

class BackendsRegistry:
    """Wraps the `backends` section of registry.json.

    Uses module-level get_backends_raw / set_backends_raw from app.registry.
    """

    def _read(self) -> dict:
        from . import registry as _reg
        return _reg.get_backends_raw()

    def _write(self, data: dict) -> None:
        from . import registry as _reg
        _reg.set_backends_raw(data)

    def seed_defaults_if_empty(self) -> None:
        """Populate backends section with defaults if missing.

        Adds any DEFAULT_BACKENDS that are not yet in the runtime registry (does
        not overwrite existing entries — user-edited backends are preserved).
        """
        current = self._read() or {}
        added = False
        for b in _DEFAULT_BACKENDS:
            if b["id"] in current:
                continue
            current[b["id"]] = {k: v for k, v in b.items() if k != "id"}
            added = True
        if added or not self._read():
            self._write(current)

    def list_all(self) -> list[BackendDef]:
        """Return all backends in insertion order (vulkan, hip, rock, custom…)."""
        raw = self._read()
        out: list[BackendDef] = []
        for bid, entry in raw.items():
            out.append(_backend_from_dict(bid, entry))
        return out

    def get(self, backend_id: str) -> BackendDef | None:
        """Return a BackendDef by id or None if not found."""
        raw = self._read()
        entry = raw.get(backend_id)
        if entry is None:
            return None
        return _backend_from_dict(backend_id, entry)

    def compatible(self, model: "ModelRecord") -> list[BackendDef]:
        """Return backends compatible with the given model.

        A backend is compatible if:
        - it has no model_predicate, OR
        - its predicate function returns True for the model.
        """
        result: list[BackendDef] = []
        for b in self.list_all():
            if b.model_predicate is None:
                result.append(b)
            else:
                fn = PREDICATES.get(b.model_predicate)
                if fn is not None and fn(model):  # type: ignore[operator]
                    result.append(b)
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BINARY_PATH_RE = re.compile(
    r"^(?:/usr/local/bin/llama-server[\w-]*|/opt/llama\.cpp[\w./-]*/llama-server[\w-]*)$"
)
_MAX_PATTERN_LEN = 200


def _validate_backend_def(bid: str, binary_path: str, incompatible_flag_patterns: list[str]) -> None:
    """Validate binary_path and incompatible_flag_patterns for a backend definition.

    Raises ValueError on invalid input. Called at construction time so that any
    future API endpoint that writes backends inherits the validation automatically.
    """
    if binary_path and not _BINARY_PATH_RE.match(binary_path):
        raise ValueError(
            f"Backend {bid!r}: binary_path {binary_path!r} does not match allowed pattern. "
            "Must be /usr/local/bin/llama-server[...] or /opt/llama.cpp/.../llama-server[...]"
        )
    for pat in incompatible_flag_patterns:
        if len(pat) > _MAX_PATTERN_LEN:
            raise ValueError(
                f"Backend {bid!r}: incompatible_flag_pattern exceeds {_MAX_PATTERN_LEN} chars: {pat[:50]!r}..."
            )
        try:
            re.compile(pat)
        except re.error as exc:
            raise ValueError(
                f"Backend {bid!r}: invalid regex in incompatible_flag_patterns: {pat!r} — {exc}"
            ) from exc


def _backend_from_dict(bid: str, d: dict) -> BackendDef:
    binary_path = str(d.get("binary_path", ""))
    incompatible_flag_patterns = list(d.get("incompatible_flag_patterns") or [])
    _validate_backend_def(bid, binary_path, incompatible_flag_patterns)
    return BackendDef(
        id=bid,
        binary_path=binary_path,
        label=str(d.get("label", bid)),
        caps=list(d.get("caps") or []),
        default_flags=list(d.get("default_flags") or []),
        required_flags=list(d.get("required_flags") or []),
        incompatible_flag_patterns=incompatible_flag_patterns,
        model_predicate=d.get("model_predicate") or None,
        version_hint=str(d.get("version_hint", "")),
        builtin=bool(d.get("builtin", True)),
    )


__all__ = ["BackendDef", "BackendsRegistry", "PREDICATES", "has_mtp_heads", "is_moe"]
