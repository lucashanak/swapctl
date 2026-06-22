"""Runtime configuration for swapctl."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_path(name: str, default: str) -> Path:
    return Path(_env(name, default))


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        return


_load_env_file(Path("/etc/swapctl.env"))


def _load_or_create_secret(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text().strip()
    except OSError:
        pass
    import secrets as _secrets
    value = _secrets.token_hex(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n")
        path.chmod(0o600)
    except OSError:
        return value
    return value


def _resolve_secret_key() -> str:
    explicit = _env("SWAPCTL_SECRET_KEY", "")
    if explicit:
        return explicit
    return _load_or_create_secret(Path("/var/lib/swapctl/.secret"))


@dataclass(frozen=True)
class Settings:
    models_dir: Path = field(default_factory=lambda: _env_path("SWAPCTL_MODELS_DIR", "/models"))
    swap_config_path: Path = field(default_factory=lambda: _env_path("SWAPCTL_CONFIG_PATH", "/etc/llama-swap/config.yaml"))
    registry_path: Path = field(default_factory=lambda: _env_path("SWAPCTL_REGISTRY_PATH", "/var/lib/swapctl/registry.json"))
    swap_service: str = field(default_factory=lambda: _env("SWAPCTL_SERVICE", "llama-swap"))
    swap_endpoint: str = field(default_factory=lambda: _env("SWAPCTL_ENDPOINT", "http://127.0.0.1:8080"))
    port_base: int = field(default_factory=lambda: _env_int("SWAPCTL_PORT_BASE", 9100))
    ctx_default: int = field(default_factory=lambda: _env_int("SWAPCTL_CTX_DEFAULT", 65536))
    gpu_budget_gb: int = field(default_factory=lambda: _env_int("SWAPCTL_GPU_BUDGET_GB", 70))
    bind_host: str = field(default_factory=lambda: _env("SWAPCTL_BIND_HOST", "0.0.0.0"))
    bind_port: int = field(default_factory=lambda: _env_int("SWAPCTL_BIND_PORT", 8090))
    password: str = field(default_factory=lambda: _env("SWAPCTL_PASSWORD", ""))
    secret_key: str = field(default_factory=_resolve_secret_key)
    session_hours: int = field(default_factory=lambda: _env_int("SWAPCTL_SESSION_HOURS", 720))
    cookie_secure: bool = field(default_factory=lambda: _env("SWAPCTL_COOKIE_SECURE", "1") not in ("0", "false", "False", ""))
    root_path: str = field(default_factory=lambda: _env("SWAPCTL_ROOT_PATH", "").rstrip("/"))
    public_base: str = field(default_factory=lambda: _env("SWAPCTL_PUBLIC_BASE", "https://ai.hanaktech.org").rstrip("/"))
    llm_public_prefix: str = field(default_factory=lambda: "/" + _env("SWAPCTL_LLM_PUBLIC_PREFIX", "llm").strip("/"))
    image_endpoint: str = field(default_factory=lambda: _env("SWAPCTL_IMAGE_ENDPOINT", "http://127.0.0.1:8190").rstrip("/"))
    image_public_prefix: str = field(default_factory=lambda: "/" + _env("SWAPCTL_IMAGE_PUBLIC_PREFIX", "image").strip("/"))


_DL_SPEED_FILE = Path("/var/lib/swapctl/dl_speed_limit")


def get_dl_speed_limit() -> int:
    """Return download speed limit in MB/s. 0 = unlimited."""
    try:
        val = _DL_SPEED_FILE.read_text().strip()
        return max(0, int(val))
    except (OSError, ValueError):
        return 0


def set_dl_speed_limit(mb_per_sec: int) -> None:
    """Set download speed limit in MB/s. 0 = unlimited."""
    _DL_SPEED_FILE.parent.mkdir(parents=True, exist_ok=True)
    _DL_SPEED_FILE.write_text(str(max(0, int(mb_per_sec))) + "\n")


_BACKEND_FILE = Path("/var/lib/swapctl/llama_backend")
_VALID_BACKENDS = ("hip", "vulkan", "rock")
_BACKEND_BINARIES = {
    "hip": "/usr/local/bin/llama-server-hip",
    "vulkan": "/usr/local/bin/llama-server-vulkan",
    "rock": "/usr/local/bin/llama-server-rock",
}


def get_llama_backend() -> str:
    try:
        val = _BACKEND_FILE.read_text().strip()
        if val in _VALID_BACKENDS:
            return val
    except OSError:
        pass
    return _env("SWAPCTL_LLAMA_BACKEND", "vulkan")


def set_llama_backend(backend: str) -> None:
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"invalid backend: {backend!r}, must be one of {_VALID_BACKENDS}")
    _BACKEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    _BACKEND_FILE.write_text(backend + "\n")


def llama_server_binary(backend_id: str | None = None) -> str:
    """Return the binary path for a backend.

    If backend_id is None, uses the currently configured default backend.
    Looks up the registry first so custom backends (e.g. vulkan-mtp) are resolved.
    Falls back to _BACKEND_BINARIES for the three built-in ids.
    """
    bid = backend_id if backend_id is not None else get_llama_backend()
    # Try registry lookup for non-standard backends
    if bid not in _BACKEND_BINARIES:
        try:
            from .backends_registry import BackendsRegistry
            b = BackendsRegistry().get(bid)
            if b is not None:
                return b.binary_path
        except Exception:
            pass
    return _BACKEND_BINARIES.get(bid, _BACKEND_BINARIES[get_llama_backend()])


_PARALLEL_FILE = Path("/var/lib/swapctl/parallel_slots")


def get_parallel_slots() -> int:
    """Return number of parallel slots per model (continuous batching)."""
    try:
        val = _PARALLEL_FILE.read_text().strip()
        return max(1, min(8, int(val)))
    except (OSError, ValueError):
        return 2


def set_parallel_slots(n: int) -> None:
    _PARALLEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PARALLEL_FILE.write_text(str(max(1, min(8, int(n)))) + "\n")


_IQ_PARALLEL_FILE = Path("/var/lib/swapctl/iq_parallel")


def get_iq_parallel() -> bool:
    """If True, IQ test sends questions concurrently (workers = parallel_slots)."""
    try:
        return _IQ_PARALLEL_FILE.read_text().strip() == "1"
    except OSError:
        return False


def set_iq_parallel(enabled: bool) -> None:
    _IQ_PARALLEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    _IQ_PARALLEL_FILE.write_text("1" if enabled else "0")


_JUDGE_MODEL_FILE = Path("/var/lib/swapctl/judge_model")


def get_judge_model() -> str:
    """Return model_id used by IQ test LLM judge for failed-answer review."""
    try:
        val = _JUDGE_MODEL_FILE.read_text().strip()
        if val:
            return val
    except OSError:
        pass
    return "gpt-oss-120b"


def set_judge_model(model_id: str) -> None:
    _JUDGE_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    _JUDGE_MODEL_FILE.write_text(model_id.strip() + "\n")


settings = Settings()

__all__ = ["Settings", "settings"]
