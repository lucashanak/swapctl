"""Input validation and path safety for untrusted form/route parameters."""
from __future__ import annotations

import re
from pathlib import Path

from . import ConfigError

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_FILENAME_RE = re.compile(
    r"^(?!/)(?!.*\.\.)(?:[A-Za-z0-9][A-Za-z0-9._-]*/)?[A-Za-z0-9][A-Za-z0-9._-]{0,191}\.gguf$"
)

_MAX_CTX = 1_048_576  # 1M tokens
_MIN_CTX = 512
_MAX_INSTALL_BYTES = 200 * 1024 ** 3  # 200 GiB per file


def valid_model_id(value: str) -> str:
    if not isinstance(value, str) or not _MODEL_ID_RE.match(value):
        raise ConfigError(
            "model_id must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
        )
    return value


def valid_repo_id(value: str) -> str:
    if not isinstance(value, str) or not _REPO_ID_RE.match(value):
        raise ConfigError("repo_id must look like 'owner/repo'")
    return value


def valid_filename(value: str) -> str:
    if not isinstance(value, str) or not _FILENAME_RE.match(value):
        raise ConfigError("filename must be a bare *.gguf basename")
    return value


def valid_ctx_size(value: int) -> int:
    if not isinstance(value, int) or value < _MIN_CTX or value > _MAX_CTX:
        raise ConfigError(f"ctx_size must be in [{_MIN_CTX}, {_MAX_CTX}]")
    return value


def safe_child_dir(parent: Path, name: str) -> Path:
    """Resolve `parent/name` and assert it stays strictly below `parent`."""
    parent_r = parent.resolve()
    child = (parent_r / name).resolve()
    try:
        child.relative_to(parent_r)
    except ValueError as exc:
        raise ConfigError(f"path escape: {name}") from exc
    if child == parent_r:
        raise ConfigError(f"path must not equal parent: {name}")
    return child


def max_install_bytes() -> int:
    return _MAX_INSTALL_BYTES
