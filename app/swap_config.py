"""Round-trip YAML editing for /etc/llama-swap/config.yaml."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from . import ConfigError
from .config import settings, llama_server_binary, get_llama_backend, set_llama_backend, _BACKEND_BINARIES, get_parallel_slots


_DISABLED_PATH = settings.registry_path.parent / "disabled_models.json"


def _load_disabled() -> dict:
    if not _DISABLED_PATH.exists():
        return {}
    try:
        with open(_DISABLED_PATH) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_disabled(d: dict) -> None:
    try:
        _DISABLED_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=_DISABLED_PATH.name + ".", suffix=".tmp", dir=str(_DISABLED_PATH.parent)
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(d, f, indent=2, sort_keys=True)
            os.replace(tmp, _DISABLED_PATH)
        except OSError:
            try:
                Path(tmp).unlink()
            except OSError:
                pass
    except OSError:
        pass


def _block_to_plain(block) -> dict:
    """Recursively convert a CommentedMap (or dict) to a plain JSON-serializable dict."""
    if isinstance(block, dict):
        return {str(k): _block_to_plain(v) for k, v in block.items()}
    if isinstance(block, list):
        return [_block_to_plain(x) for x in block]
    if isinstance(block, (str, int, float, bool)) or block is None:
        return block
    return str(block)


def list_disabled_ids() -> list[str]:
    return sorted(_load_disabled().keys())

_PORT_RE = re.compile(r"--port\s+(\d+)")

_PORT_CAP = 100


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=4, offset=2)
    y.width = 10_000
    return y


def _path() -> Path:
    return settings.swap_config_path


def load(path: Path | None = None) -> CommentedMap:
    """Read config.yaml and return its ruamel structure."""
    target = path or _path()
    try:
        with open(target, "r") as f:
            data = _yaml().load(f)
    except FileNotFoundError as exc:
        raise ConfigError(f"config not found: {target}") from exc
    except OSError as exc:
        raise ConfigError(f"failed to read config {target}: {exc}") from exc
    if data is None:
        data = CommentedMap()
    if "models" not in data:
        data["models"] = CommentedMap()
    return data


def save(data: CommentedMap, path: Path | None = None) -> None:
    """Atomically write ruamel structure to config path."""
    target = path or _path()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            _yaml().dump(data, f)
        os.replace(tmp, target)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise ConfigError(f"failed to write config {target}: {exc}") from exc


def backup(path: Path | None = None) -> Path | None:
    """Copy current config to <path>.bak (single rolling backup). Returns .bak path or None if no source."""
    target = path or _path()
    if not target.exists():
        return None
    bak = target.with_suffix(target.suffix + ".bak")
    shutil.copy2(target, bak)
    return bak


def restore_backup(path: Path | None = None) -> None:
    """Restore the most recent .bak over the config."""
    target = path or _path()
    bak = target.with_suffix(target.suffix + ".bak")
    if not bak.exists():
        raise ConfigError(f"no backup to restore at {bak}")
    shutil.copy2(bak, target)


def list_model_ids(path: Path | None = None) -> list[str]:
    """Return model ids currently defined under `models:`."""
    data = load(path)
    models = data.get("models") or {}
    return list(models.keys())


def _collect_cmd_strings(data: CommentedMap) -> list[str]:
    out: list[str] = []
    models = data.get("models") or {}
    for block in models.values():
        if not isinstance(block, dict):
            continue
        cmd = block.get("cmd")
        if isinstance(cmd, str):
            out.append(cmd)
        elif isinstance(cmd, list):
            out.append(" ".join(str(x) for x in cmd))
    return out


def next_free_port(base: int | None = None, path: Path | None = None) -> int:
    """Return the next free port scanning --port flags in existing cmd strings."""
    start = base if base is not None else settings.port_base
    data = load(path)
    used: set[int] = set()
    for cmd in _collect_cmd_strings(data):
        for m in _PORT_RE.finditer(cmd):
            try:
                used.add(int(m.group(1)))
            except ValueError:
                continue
    for port in range(start, start + _PORT_CAP + 1):
        if port not in used:
            return port
    raise RuntimeError(f"port allocation exhausted above {start + _PORT_CAP}")


def _build_cmd(
    gguf_path: Path,
    mmproj_path: Path | None,
    port: int,
    ctx_size: int,
    no_mmap: bool,
    cache_type_k: str = "q8_0",
    cache_type_v: str = "q8_0",
    threads: int = 0,
    flash_attn: bool = True,
    extra_args: str = "",
    reasoning_format: str = "",
    draft_model: str = "",
    draft_max: int = 0,
    draft_min: int = 0,
    spec_default: bool = False,
    sampling_temp: float = 0.8,
    sampling_top_k: int = 40,
    sampling_top_p: float = 0.95,
    sampling_min_p: float = 0.05,
    sampling_repeat_penalty: float = 1.1,
) -> str:
    parts = [
        llama_server_binary(),
        f"--model {gguf_path}",
        f"--port {port}",
        "-ngl 999",
        f"--cache-type-k {cache_type_k}",
        f"--cache-type-v {cache_type_v}",
        "--jinja",
        f"--ctx-size {ctx_size}",
        f"--parallel {get_parallel_slots()}",
    ]
    if flash_attn:
        parts.append("--flash-attn on")
    if threads > 0:
        parts.append(f"--threads {threads}")
    if no_mmap:
        parts.append("--no-mmap")
    if mmproj_path is not None:
        parts.append(f"--mmproj {mmproj_path}")
    if reasoning_format:
        parts.append(f"--reasoning-format {reasoning_format}")
    if spec_default:
        parts.append("--spec-default")
    if draft_model:
        parts.append(f"-md {draft_model}")
        parts.append("-ngld 999")
        if draft_max > 0:
            parts.append(f"--draft-max {draft_max}")
        if draft_min > 0:
            parts.append(f"--draft-min {draft_min}")
    if sampling_temp != 0.8:
        parts.append(f"--temp {sampling_temp}")
    if sampling_top_k != 40:
        parts.append(f"--top-k {sampling_top_k}")
    if sampling_top_p != 0.95:
        parts.append(f"--top-p {sampling_top_p}")
    if sampling_min_p != 0.05:
        parts.append(f"--min-p {sampling_min_p}")
    if sampling_repeat_penalty != 1.1:
        parts.append(f"--repeat-penalty {sampling_repeat_penalty}")
    if extra_args.strip():
        parts.append(extra_args.strip())
    return " ".join(parts)


def rebuild_matrix(path: Path | None = None, max_group_size: int = 3) -> None:
    """Regenerate matrix vars, evict_costs, and combo sets from current models."""
    data = load(path)
    models_block = data.get("models")
    if not isinstance(models_block, CommentedMap) or not models_block:
        if "matrix" in data:
            del data["matrix"]
            save(data, path)
        return

    declared = [m for m in models_block.keys()
                if not _is_npu_block(models_block[m])]
    old_matrix = data.get("matrix") or {}
    old_vars = {str(v): str(k) for k, v in (old_matrix.get("vars") or {}).items()}
    old_costs = old_matrix.get("evict_costs") or {}

    used_short: set[str] = set()
    shortcuts: dict[str, str] = {}
    priorities: dict[str, str] = {}
    for m in sorted(declared):
        old_sc = old_vars.get(m)
        if old_sc and old_sc not in used_short:
            sc = old_sc
        else:
            sc = _shortcut_for(m, used_short)
        shortcuts[m] = sc
        used_short.add(sc)
        if old_sc and old_sc in old_costs:
            priorities[m] = _cost_to_priority(old_costs[old_sc])
        else:
            priorities[m] = "medium"

    budget_bytes = settings.gpu_budget_gb * 1024 ** 3
    combos = _auto_generate_combos(declared, budget_bytes, max_group_size)

    vars_d = {shortcuts[m]: m for m in declared}
    costs = {shortcuts[m]: _PRIORITY_TO_COST[priorities[m]] for m in declared}
    sets_d: dict[str, str] = {}
    for m in declared:
        sets_d[f"solo_{shortcuts[m]}"] = shortcuts[m]
    for combo in combos:
        if not all(m in shortcuts for m in combo):
            continue
        scs = [shortcuts[m] for m in combo]
        name = "combo_" + "_".join(scs)
        sets_d[name] = " & ".join(scs)

    matrix = CommentedMap()
    matrix["vars"] = CommentedMap({k: vars_d[k] for k in sorted(vars_d)})
    matrix["evict_costs"] = CommentedMap({k: costs[k] for k in sorted(costs)})
    matrix["sets"] = CommentedMap({k: sets_d[k] for k in sorted(sets_d)})
    data["matrix"] = matrix
    save(data, path)


def add_model(
    model_id: str,
    gguf_path: Path,
    mmproj_path: Path | None,
    port: int,
    ctx_size: int,
    no_mmap: bool,
    cache_type_k: str = "q8_0",
    cache_type_v: str = "q8_0",
    threads: int = 0,
    flash_attn: bool = True,
    extra_args: str = "",
    reasoning_format: str = "",
    draft_model: str = "",
    draft_max: int = 0,
    draft_min: int = 0,
    spec_default: bool = False,
    sampling_temp: float = 0.8,
    sampling_top_k: int = 40,
    sampling_top_p: float = 0.95,
    sampling_min_p: float = 0.05,
    sampling_repeat_penalty: float = 1.1,
    path: Path | None = None,
) -> None:
    """Insert or replace a model block under `models:` and save."""
    if not model_id:
        raise ConfigError("model_id must be non-empty")
    data = load(path)
    models = data.get("models")
    if not isinstance(models, CommentedMap):
        models = CommentedMap()
        data["models"] = models
    block = CommentedMap()
    cmd = _build_cmd(
        gguf_path, mmproj_path, port, ctx_size, no_mmap,
        cache_type_k=cache_type_k, cache_type_v=cache_type_v,
        threads=threads, flash_attn=flash_attn, extra_args=extra_args,
        reasoning_format=reasoning_format,
        draft_model=draft_model, draft_max=draft_max, draft_min=draft_min,
        spec_default=spec_default,
        sampling_temp=sampling_temp, sampling_top_k=sampling_top_k,
        sampling_top_p=sampling_top_p, sampling_min_p=sampling_min_p,
        sampling_repeat_penalty=sampling_repeat_penalty,
    )
    # Apply backend-specific flag injection (FIX 5: registry awareness in _build_cmd path)
    try:
        from .backends_registry import BackendsRegistry
        _backend_id = _detect_backend(cmd)
        _br = BackendsRegistry()
        _backend_def = _br.get(_backend_id)
        if _backend_def is not None:
            cmd = _inject_backend_flags(cmd, _backend_def)
    except Exception:
        pass
    block["cmd"] = cmd
    block["proxy"] = f"http://127.0.0.1:{port}"
    models[model_id] = block
    save(data, path)
    rebuild_matrix(path=path)


def remove_model(model_id: str, path: Path | None = None) -> None:
    """Pop a model block by id and rebuild matrix. No-op if absent."""
    data = load(path)
    models = data.get("models")
    if isinstance(models, CommentedMap) and model_id in models:
        del models[model_id]
        _purge_matrix_refs(data, model_id)
        save(data, path)
        rebuild_matrix(path=path)


def _purge_matrix_refs(data: dict, model_id: str) -> None:
    """Remove all matrix vars, evict_costs, and sets that reference a removed model."""
    matrix = data.get("matrix")
    if not matrix:
        return
    vars_d = matrix.get("vars") or {}
    orphan_keys = [k for k, v in vars_d.items() if v == model_id]
    if not orphan_keys:
        return
    for k in orphan_keys:
        vars_d.pop(k, None)
        costs = matrix.get("evict_costs")
        if costs:
            costs.pop(k, None)
    orphan_set = set(orphan_keys)
    sets = matrix.get("sets")
    if sets:
        to_remove = [name for name, expr in sets.items()
                     if orphan_set & set(_SET_TOKEN_RE.findall(expr))]
        for name in to_remove:
            del sets[name]


_CTX_RE = re.compile(r"(--ctx-size\s+)\d+")


def update_ctx_size(model_id: str, ctx_size: int, path: Path | None = None) -> None:
    """Rewrite --ctx-size in the model's cmd string. Raises ConfigError if not found."""
    data = load(path)
    models = data.get("models") or {}
    if model_id not in models:
        raise ConfigError(f"unknown model: {model_id}")
    block = models[model_id]
    cmd = block.get("cmd", "")
    new_cmd, n = _CTX_RE.subn(lambda m: f"{m.group(1)}{ctx_size}", cmd, count=1)
    if n == 0:
        raise ConfigError(f"no --ctx-size flag in cmd for {model_id!r}")
    block["cmd"] = new_cmd
    save(data, path)


_CACHE_K_RE = re.compile(r"(--cache-type-k\s+)\S+")
_CACHE_V_RE = re.compile(r"(--cache-type-v\s+)\S+")
_THREADS_RE = re.compile(r"--threads\s+\d+")
_FLASH_RE = re.compile(r"--flash-attn\s+\S+")
_REASONING_RE = re.compile(r"--reasoning-format\s+\S+")
_SPEC_DEFAULT_RE = re.compile(r"--spec-default")
_DRAFT_RE = re.compile(r"-md\s+\S+")
_DRAFT_NGLD_RE = re.compile(r"-ngld\s+\d+")
_DRAFT_MAX_RE = re.compile(r"--draft-max\s+\d+")
_DRAFT_MIN_RE = re.compile(r"--draft-min\s+\d+")
_TEMP_RE = re.compile(r"--temp\s+[\d.]+")
_TOP_K_RE = re.compile(r"--top-k\s+\d+")
_TOP_P_RE = re.compile(r"--top-p\s+[\d.]+")
_MIN_P_RE = re.compile(r"--min-p\s+[\d.]+")
_REPEAT_PENALTY_RE = re.compile(r"--repeat-penalty\s+[\d.]+")


def update_model_flags(
    model_id: str,
    *,
    ctx_size: int | None = None,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    threads: int | None = None,
    flash_attn: bool | None = None,
    extra_args: str | None = None,
    reasoning_format: str | None = None,
    draft_model: str | None = None,
    draft_max: int | None = None,
    draft_min: int | None = None,
    spec_default: bool | None = None,
    sampling_temp: float | None = None,
    sampling_top_k: int | None = None,
    sampling_top_p: float | None = None,
    sampling_min_p: float | None = None,
    sampling_repeat_penalty: float | None = None,
    path: Path | None = None,
) -> None:
    """Update individual flags in a model's cmd string. Only touches provided (non-None) values."""
    data = load(path)
    models = data.get("models") or {}
    if model_id not in models:
        raise ConfigError(f"unknown model: {model_id}")
    block = models[model_id]
    cmd = block.get("cmd", "")

    if ctx_size is not None:
        cmd, n = _CTX_RE.subn(lambda m: f"{m.group(1)}{ctx_size}", cmd, count=1)
        if n == 0:
            cmd += f" --ctx-size {ctx_size}"

    if cache_type_k is not None:
        cmd, n = _CACHE_K_RE.subn(lambda m: f"{m.group(1)}{cache_type_k}", cmd, count=1)
        if n == 0:
            cmd += f" --cache-type-k {cache_type_k}"

    if cache_type_v is not None:
        cmd, n = _CACHE_V_RE.subn(lambda m: f"{m.group(1)}{cache_type_v}", cmd, count=1)
        if n == 0:
            cmd += f" --cache-type-v {cache_type_v}"

    if threads is not None:
        if threads > 0:
            repl = f"--threads {threads}"
            cmd, n = _THREADS_RE.subn(repl, cmd, count=1)
            if n == 0:
                cmd += f" --threads {threads}"
        else:
            cmd = _THREADS_RE.sub("", cmd).strip()
            cmd = re.sub(r"  +", " ", cmd)

    if flash_attn is not None:
        if flash_attn:
            cmd, n = _FLASH_RE.subn("--flash-attn on", cmd, count=1)
            if n == 0:
                cmd += " --flash-attn on"
        else:
            cmd = _FLASH_RE.sub("", cmd).strip()
            cmd = re.sub(r"  +", " ", cmd)

    if reasoning_format is not None:
        cmd = _REASONING_RE.sub("", cmd).strip()
        cmd = re.sub(r"  +", " ", cmd)
        if reasoning_format:
            cmd += f" --reasoning-format {reasoning_format}"

    if spec_default is not None:
        cmd = _SPEC_DEFAULT_RE.sub("", cmd).strip()
        cmd = re.sub(r"  +", " ", cmd)
        if spec_default:
            cmd += " --spec-default"

    if draft_model is not None:
        cmd = _DRAFT_RE.sub("", cmd)
        cmd = _DRAFT_NGLD_RE.sub("", cmd)
        cmd = _DRAFT_MAX_RE.sub("", cmd)
        cmd = _DRAFT_MIN_RE.sub("", cmd)
        cmd = re.sub(r"  +", " ", cmd).strip()
        if draft_model:
            cmd += f" -md {draft_model} -ngld 999"
            dm = draft_max if draft_max is not None else 0
            dn = draft_min if draft_min is not None else 0
            if dm > 0:
                cmd += f" --draft-max {dm}"
            if dn > 0:
                cmd += f" --draft-min {dn}"
    else:
        if draft_max is not None:
            cmd = _DRAFT_MAX_RE.sub("", cmd)
            cmd = re.sub(r"  +", " ", cmd).strip()
            if draft_max > 0:
                cmd += f" --draft-max {draft_max}"
        if draft_min is not None:
            cmd = _DRAFT_MIN_RE.sub("", cmd)
            cmd = re.sub(r"  +", " ", cmd).strip()
            if draft_min > 0:
                cmd += f" --draft-min {draft_min}"

    if extra_args is not None:
        for flag in ("--no-mmap", "-b"):
            if flag == "-b":
                pat = re.compile(r"-b\s+\S+")
            else:
                pat = re.compile(re.escape(flag))
            cmd = pat.sub("", cmd)
        cmd = re.sub(r"  +", " ", cmd).strip()
        if extra_args.strip():
            cmd += " " + extra_args.strip()

    _SAMPLING_DEFAULTS = {"temp": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05, "repeat_penalty": 1.1}
    _SAMPLING_RES = {"temp": _TEMP_RE, "top_k": _TOP_K_RE, "top_p": _TOP_P_RE, "min_p": _MIN_P_RE, "repeat_penalty": _REPEAT_PENALTY_RE}
    _SAMPLING_FLAGS = {"temp": "--temp", "top_k": "--top-k", "top_p": "--top-p", "min_p": "--min-p", "repeat_penalty": "--repeat-penalty"}
    for key, val in [("temp", sampling_temp), ("top_k", sampling_top_k), ("top_p", sampling_top_p),
                     ("min_p", sampling_min_p), ("repeat_penalty", sampling_repeat_penalty)]:
        if val is not None:
            cmd = _SAMPLING_RES[key].sub("", cmd)
            cmd = re.sub(r"  +", " ", cmd).strip()
            if val != _SAMPLING_DEFAULTS[key]:
                cmd += f" {_SAMPLING_FLAGS[key]} {val}"

    block["cmd"] = cmd
    save(data, path)


def validate(path: Path | None = None) -> bool:
    """Re-parse config and ensure `models:` mapping exists and is non-empty."""
    try:
        data = load(path)
    except ConfigError:
        return False
    models = data.get("models")
    if not isinstance(models, (dict, CommentedMap)):
        return False
    return len(models) > 0


_VALID_SET_EXPR = re.compile(r"^[A-Za-z0-9_&|!()\s]+$")
_SET_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def get_matrix_and_globals() -> dict:
    """Return current matrix + top-level scalars as plain dicts."""
    data = load()
    matrix = data.get("matrix") or {}
    return {
        "healthCheckTimeout": int(data.get("healthCheckTimeout", 300) or 300),
        "globalTTL": int(data.get("globalTTL", 0) or 0),
        "vars": {str(k): str(v) for k, v in dict(matrix.get("vars") or {}).items()},
        "evict_costs": {str(k): int(v) for k, v in dict(matrix.get("evict_costs") or {}).items()},
        "sets": {str(k): str(v) for k, v in dict(matrix.get("sets") or {}).items()},
    }


def _validate_matrix(vars_d: dict, costs: dict, sets: dict, declared_models: set) -> None:
    for short, mid in vars_d.items():
        if not short or not re.match(r"^[A-Za-z][A-Za-z0-9]{0,7}$", short):
            raise ConfigError(f"invalid var shortcut: {short!r} (must be 1–8 alphanumerics)")
        if mid not in declared_models:
            raise ConfigError(f"var {short!r} → {mid!r} is not a declared model")
    for key in costs:
        if key not in vars_d:
            raise ConfigError(f"evict_costs key {key!r} not declared in vars")
    shortcuts = set(vars_d.keys())
    for name, expr in sets.items():
        if not name or not re.match(r"^[A-Za-z_][A-Za-z0-9_]{0,31}$", name):
            raise ConfigError(f"invalid set name: {name!r}")
        if not isinstance(expr, str) or not _VALID_SET_EXPR.match(expr):
            raise ConfigError(f"set {name!r}: illegal characters in expression")
        for tok in _SET_TOKEN_RE.findall(expr):
            if tok not in shortcuts:
                raise ConfigError(f"set {name!r}: unknown shortcut {tok!r}")


def set_matrix_and_globals(payload: dict) -> None:
    """Validate + write matrix section + top-level scalars. Preserves models and other keys."""
    data = load()
    health = int(payload.get("healthCheckTimeout", 300) or 300)
    gttl = int(payload.get("globalTTL", 0) or 0)
    vars_d = dict(payload.get("vars") or {})
    costs = {k: int(v) for k, v in dict(payload.get("evict_costs") or {}).items()}
    sets = {k: str(v) for k, v in dict(payload.get("sets") or {}).items()}

    declared = set((data.get("models") or {}).keys())
    _validate_matrix(vars_d, costs, sets, declared)

    data["healthCheckTimeout"] = health
    data["globalTTL"] = gttl
    if not vars_d and not costs and not sets:
        if "matrix" in data:
            del data["matrix"]
    else:
        matrix = CommentedMap()
        matrix["vars"] = CommentedMap({k: vars_d[k] for k in sorted(vars_d)})
        matrix["evict_costs"] = CommentedMap({k: costs[k] for k in sorted(costs)})
        matrix["sets"] = CommentedMap({k: sets[k] for k in sorted(sets)})
        data["matrix"] = matrix
    save(data)


_PRIORITY_TO_COST = {"low": 5, "medium": 15, "high": 50}
_COST_TO_PRIORITY_THRESHOLDS = [(8, "low"), (25, "medium")]  # else "high"


def _cost_to_priority(cost: int) -> str:
    for threshold, name in _COST_TO_PRIORITY_THRESHOLDS:
        if cost <= threshold:
            return name
    return "high"


def _shortcut_for(model_id: str, used: set[str]) -> str:
    """llama-swap requires alphanumeric shortcuts (no underscore), 1–8 chars."""
    base = re.sub(r"[^a-z0-9]", "", model_id.lower())[:6] or "m"
    candidate = base
    i = 2
    while candidate in used:
        suffix = str(i)
        candidate = (base[: max(1, 8 - len(suffix))] + suffix)[:8]
        i += 1
    return candidate


def _parse_combos_from_sets(sets: dict, vars_map: dict) -> list[list[str]]:
    """Extract combinations of size >= 2 from pure `a & b [& c ...]` sets."""
    seen: set[frozenset] = set()
    out: list[list[str]] = []
    for expr in sets.values():
        expr_s = str(expr)
        if "|" in expr_s or "(" in expr_s:
            continue
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr_s)
        models_in_expr = [vars_map[t] for t in tokens if t in vars_map]
        if len(models_in_expr) < 2:
            continue
        if len(set(models_in_expr)) != len(models_in_expr):
            continue
        key = frozenset(models_in_expr)
        if key in seen:
            continue
        seen.add(key)
        out.append(sorted(models_in_expr))
    return out


def _model_sizes() -> dict[str, int]:
    """Get model sizes from registry, returns {model_id: size_bytes}."""
    from . import registry
    sizes = {}
    for rec in registry.list_all():
        if rec.size_bytes and rec.size_bytes > 0:
            sizes[rec.model_id] = rec.size_bytes
    return sizes


_COMBO_LIMIT = 950


def _auto_generate_combos(
    model_ids: list[str],
    budget_bytes: int,
    max_group_size: int = 3,
) -> list[list[str]]:
    """Generate all valid concurrent model groups that fit within GPU budget.

    Auto-reduces max_group_size when total combinations would exceed the
    llama-swap limit of ~1000 sets."""
    from itertools import combinations
    from math import comb

    sizes = _model_sizes()
    sized_models = [(m, sizes.get(m, 0)) for m in model_ids if sizes.get(m, 0) > 0]
    n = len(sized_models)
    while max_group_size > 2:
        estimate = sum(comb(n, k) for k in range(2, max_group_size + 1)) + n
        if estimate <= _COMBO_LIMIT:
            break
        max_group_size -= 1
    combos: list[list[str]] = []
    for group_size in range(2, min(max_group_size + 1, n + 1)):
        for group in combinations(sized_models, group_size):
            total = sum(s for _, s in group)
            if total <= budget_bytes:
                combos.append(sorted(m for m, _ in group))
    return combos


def get_simple_config() -> dict:
    """Return simplified view: globals + per-model priority + auto-generated combos.
    Model list = union of currently-declared (config.yaml) and disabled-stash."""
    data = load()
    models_block = data.get("models") or {}
    declared = [m for m in models_block.keys()
                if not _is_npu_block(models_block[m])]
    disabled = _load_disabled()
    all_known = sorted(set(declared) | set(disabled.keys()))
    matrix = data.get("matrix") or {}
    vars_map = {str(k): str(v) for k, v in (matrix.get("vars") or {}).items()}
    costs = {str(k): int(v) for k, v in (matrix.get("evict_costs") or {}).items()}
    priorities: dict[str, str] = {}
    for short, mid in vars_map.items():
        priorities[mid] = _cost_to_priority(costs.get(short, 15))
    for m in all_known:
        priorities.setdefault(m, "medium")
    budget_bytes = settings.gpu_budget_gb * 1024 ** 3
    combos = _auto_generate_combos(
        [m for m in all_known if m in declared], budget_bytes
    )
    enabled = {m: (m in declared) for m in all_known}
    return {
        "globalTTL": int(data.get("globalTTL", 0) or 0),
        "healthCheckTimeout": int(data.get("healthCheckTimeout", 300) or 300),
        "models": all_known,
        "priorities": priorities,
        "combos": combos,
        "enabled": enabled,
    }


def set_simple_config(payload: dict) -> None:
    """Translate simple form input into matrix + write. Combos are auto-generated
    from model sizes and GPU budget. Enabled=false models are parked in
    /var/lib/swapctl/disabled_models.json and removed from config.yaml entirely."""
    data = load()
    disabled_store = _load_disabled()
    models_block = data.get("models")
    if not isinstance(models_block, CommentedMap):
        models_block = CommentedMap()
        data["models"] = models_block
    declared_in_config = [m for m in models_block.keys()
                          if not _is_npu_block(models_block[m])]
    health = int(payload.get("healthCheckTimeout", 300) or 300)
    keep_warm = bool(payload.get("keepWarm", True))
    priorities_in = dict(payload.get("priorities") or {})
    enabled_in = dict(payload.get("enabled") or {})
    all_known = set(declared_in_config) | set(disabled_store.keys())
    new_enabled_set = {m for m in all_known if enabled_in.get(m, True)}

    for m in list(declared_in_config):
        if m not in new_enabled_set:
            disabled_store[m] = _block_to_plain(models_block[m])
            del models_block[m]
    for m in list(disabled_store.keys()):
        if m in new_enabled_set:
            block = disabled_store.pop(m)
            new_block = CommentedMap()
            for k, v in block.items():
                new_block[k] = v
            models_block[m] = new_block

    declared = list(models_block.keys())
    active = sorted(new_enabled_set & set(declared))

    priorities: dict[str, str] = {}
    for m in declared:
        p = priorities_in.get(m, "medium")
        if p not in _PRIORITY_TO_COST:
            p = "medium"
        priorities[m] = p

    used_short: set[str] = set()
    shortcuts: dict[str, str] = {}
    for m in active:
        sc = _shortcut_for(m, used_short)
        shortcuts[m] = sc
        used_short.add(sc)

    budget_bytes = settings.gpu_budget_gb * 1024 ** 3
    max_group = int(payload.get("max_group_size", 3) or 3)
    combos = _auto_generate_combos(active, budget_bytes, max_group)

    vars_d = {shortcuts[m]: m for m in active}
    costs = {shortcuts[m]: _PRIORITY_TO_COST[priorities[m]] for m in active}
    sets_d: dict[str, str] = {}
    for m in active:
        sets_d[f"solo_{shortcuts[m]}"] = shortcuts[m]
    for combo in combos:
        if not all(m in shortcuts for m in combo):
            continue
        scs = [shortcuts[m] for m in combo]
        name = "combo_" + "_".join(scs)
        sets_d[name] = " & ".join(scs)

    _validate_matrix(vars_d, costs, sets_d, set(declared))

    data["healthCheckTimeout"] = health
    data["globalTTL"] = 0 if keep_warm else 600
    if not active:
        if "matrix" in data:
            del data["matrix"]
    else:
        matrix = CommentedMap()
        matrix["vars"] = CommentedMap({k: vars_d[k] for k in sorted(vars_d)})
        matrix["evict_costs"] = CommentedMap({k: costs[k] for k in sorted(costs)})
        matrix["sets"] = CommentedMap({k: sets_d[k] for k in sorted(sets_d)})
        data["matrix"] = matrix
    _save_disabled(disabled_store)
    save(data)


_BINARY_RE = re.compile(r"(/usr/local/bin/llama-server[-\w]*|/opt/llama\.cpp/build[^/]*/bin/llama-server)")


def _binary_to_backend_id(binary_path: str) -> str:
    """Map a binary basename to a registry backend id.

    Falls back to simple basename → id heuristics when registry is unavailable.
    """
    import os as _os
    basename = _os.path.basename(binary_path)
    try:
        from .backends_registry import BackendsRegistry
        reg = BackendsRegistry()
        for b in reg.list_all():
            if _os.path.basename(b.binary_path) == basename:
                return b.id
    except Exception:
        pass
    # Fallback heuristics
    if basename == "llama-server-rock":
        return "rock"
    if basename == "llama-server-vulkan":
        return "vulkan"
    if basename == "llama-server-hip":
        return "hip"
    return "unknown"


def _detect_backend(cmd: str) -> str:
    m = _BINARY_RE.search(cmd)
    if m:
        return _binary_to_backend_id(m.group(1))
    if "build-rock" in cmd:
        return "rock"
    if "build-vulkan" in cmd:
        return "vulkan"
    if "build/bin" in cmd:
        return "hip"
    return "unknown"


def _apply_backend_flags(cmd: str, backend) -> str:  # backend: BackendDef
    """Apply a BackendDef's flag injection rules to cmd:
    1. Strip tokens matching incompatible_flag_patterns (flag + value).
    2. Apply required_flags (replace existing same-flag, else append).
    3. Append default_flags only if flag name not already present.
    """
    # Step 1: strip incompatible patterns (each pattern may consume flag+value)
    for pat_str in backend.incompatible_flag_patterns:
        try:
            pat = re.compile(pat_str)
            # Find the flag name from the pattern prefix (e.g. --parallel)
            flag_match = re.match(r"(--[\w-]+)", pat_str)
            if flag_match:
                flag_name = flag_match.group(1)
                # Strip flag + its value token(s) by matching flag \s+ value
                strip_re = re.compile(re.escape(flag_name) + r"\s+\S+")
                # Only strip if the value matches the incompatible pattern
                def _strip_if_match(m, pat=pat, strip_re=strip_re):
                    return "" if pat.search(m.group(0)) else m.group(0)
                cmd = strip_re.sub(_strip_if_match, cmd)
        except re.error:
            pass
    cmd = re.sub(r"  +", " ", cmd).strip()

    # Step 2: apply required_flags
    for flag_str in backend.required_flags:
        parts = flag_str.split(None, 1)
        if not parts:
            continue
        flag_name = parts[0]
        flag_val = parts[1] if len(parts) > 1 else None
        existing_re = re.compile(re.escape(flag_name) + r"\s+\S+")
        if existing_re.search(cmd):
            if flag_val is not None:
                cmd = existing_re.sub(f"{flag_name} {flag_val}", cmd, count=1)
        else:
            cmd = cmd + " " + flag_str
    cmd = re.sub(r"  +", " ", cmd).strip()

    # Step 3: append default_flags only if flag name absent (word-boundary check)
    def _flag_present(cmd: str, flag_name: str) -> bool:
        return bool(re.search(rf"(?:^|\s){re.escape(flag_name)}(?=\s|$)", cmd))

    for flag_str in backend.default_flags:
        parts = flag_str.split(None, 1)
        if not parts:
            continue
        flag_name = parts[0]
        if not _flag_present(cmd, flag_name):
            cmd = cmd + " " + flag_str
    cmd = re.sub(r"  +", " ", cmd).strip()

    return cmd


def _inject_backend_flags(cmd: str, backend) -> str:
    """Apply a backend's flag injection rules to cmd (strips incompatible, applies required, appends defaults).

    Thin wrapper around _apply_backend_flags that can be called from both
    add_model (via _build_cmd path) and switch_model_backend, ensuring
    consistent flag injection regardless of call site.
    """
    return _apply_backend_flags(cmd, backend)


def _strip_all_backend_flags(cmd: str, backend) -> str:
    """Remove all default_flags and required_flags injected by a backend from cmd."""
    for flag_str in list(backend.required_flags) + list(backend.default_flags):
        parts = flag_str.split(None, 1)
        if not parts:
            continue
        flag_name = parts[0]
        strip_re = re.compile(re.escape(flag_name) + r"\s+\S+")
        cmd = strip_re.sub("", cmd)
    return re.sub(r"  +", " ", cmd).strip()


def detect_model_backend(model_id: str, path: Path | None = None) -> str:
    """Detect which backend a model's cmd currently uses. Returns backend id or 'unknown'."""
    data = load(path)
    models = data.get("models") or {}
    if model_id not in models:
        return "unknown"
    return _detect_backend(models[model_id].get("cmd", ""))


def detect_all_backends(path: Path | None = None) -> dict[str, str]:
    """Return {model_id: backend_id} for all models.

    Uses binary basename → registry id mapping.
    """
    data = load(path)
    models = data.get("models") or {}
    return {
        mid: _detect_backend(block.get("cmd", ""))
        for mid, block in models.items() if isinstance(block, dict)
    }


def switch_model_backend(model_id: str, backend_id: str, path: Path | None = None) -> None:
    """Rewrite a single model's cmd to use the given backend.

    Accepts any backend_id known to BackendsRegistry (including 'vulkan-mtp').
    For backwards compat also accepts the three legacy ids 'hip', 'vulkan', 'rock'.

    Steps:
    1. Resolve BackendDef from registry.
    2. Detect current backend and strip its injected flags first.
    3. Replace binary path.
    4. Strip incompatible flags.
    5. Apply required_flags (replace existing value or append).
    6. Append missing default_flags.
    """
    from .backends_registry import BackendsRegistry, BackendDef
    reg = BackendsRegistry()
    backend = reg.get(backend_id)
    if backend is None:
        # Legacy fallback — accept 'hip', 'vulkan', 'rock' even if registry empty
        if backend_id in _BACKEND_BINARIES:
            backend = BackendDef(
                id=backend_id,
                binary_path=_BACKEND_BINARIES[backend_id],
                label=backend_id,
            )
        else:
            raise ConfigError(f"unknown backend: {backend_id!r}")

    data = load(path)
    models = data.get("models") or {}
    if model_id not in models:
        raise ConfigError(f"unknown model: {model_id}")
    block = models[model_id]
    cmd = block.get("cmd", "")

    # Strip flags from the current backend before switching
    current_id = _detect_backend(cmd)
    if current_id != "unknown" and current_id != backend_id:
        current_backend = reg.get(current_id)
        if current_backend is not None:
            cmd = _strip_all_backend_flags(cmd, current_backend)

    # Replace binary
    new_cmd, n = _BINARY_RE.subn(backend.binary_path, cmd, count=1)
    if n == 0:
        raise ConfigError(f"no llama-server binary found in cmd for {model_id!r}")
    cmd = new_cmd

    # Apply flag injection
    cmd = _inject_backend_flags(cmd, backend)

    block["cmd"] = cmd
    save(data, path)


_PARALLEL_RE = re.compile(r"--parallel\s+\d+")


def update_parallel_slots(slots: int, path: Path | None = None) -> int:
    """Rewrite --parallel N flag in all model cmds. Returns count of models updated."""
    n = max(1, min(8, int(slots)))
    data = load(path)
    models = data.get("models") or {}
    count = 0
    for block in models.values():
        if not isinstance(block, dict):
            continue
        cmd = block.get("cmd", "")
        if "flm serve" in cmd or not cmd:
            continue
        if _PARALLEL_RE.search(cmd):
            new_cmd = _PARALLEL_RE.sub(f"--parallel {n}", cmd)
        else:
            new_cmd = cmd + f" --parallel {n}"
        if new_cmd != cmd:
            block["cmd"] = new_cmd
            count += 1
    if count > 0:
        save(data, path)
    return count


def switch_backend(backend: str, path: Path | None = None) -> int:
    """DEPRECATED: global backend switch. Kept as a no-op stub so tester.py imports don't break.

    Use switch_model_backend() for per-model backend switching instead.
    Returns 0 without modifying any config.
    """
    import logging as _logging
    _logging.getLogger("swapctl.swap_config").warning(
        "switch_backend() is deprecated and is a no-op. "
        "Use switch_model_backend(model_id, backend_id) instead."
    )
    return 0


NPU_PORT_BASE = 9200
NPU_MODEL_PREFIX = "npu/"


def _is_npu_block(block: dict) -> bool:
    cmd = block.get("cmd", "") if isinstance(block, dict) else ""
    return "flm serve" in cmd or "flm " in cmd


def sync_npu_models(path: Path | None = None) -> list[str]:
    """Sync installed FLM (NPU) models into llama-swap config.

    Adds new models, removes uninstalled ones. Returns list of synced model IDs.
    """
    from . import npu as npu_mod

    if not npu_mod.available():
        return []

    installed = npu_mod.installed_models()
    data = load(path)
    models_block = data.get("models")
    if not isinstance(models_block, CommentedMap):
        models_block = CommentedMap()
        data["models"] = models_block

    existing_npu = {
        mid for mid, block in models_block.items()
        if isinstance(block, dict) and _is_npu_block(block)
    }

    npu_ports: dict[str, int] = {}
    for mid in existing_npu:
        block = models_block[mid]
        cmd = block.get("cmd", "")
        m = _PORT_RE.search(cmd)
        if m:
            npu_ports[mid] = int(m.group(1))

    used_ports = set(npu_ports.values())
    next_port = NPU_PORT_BASE
    def _alloc_port() -> int:
        nonlocal next_port
        while next_port in used_ports:
            next_port += 1
        p = next_port
        used_ports.add(p)
        next_port += 1
        return p

    wanted = {f"{NPU_MODEL_PREFIX}{tag}" for tag in installed}
    changed = False

    to_remove = existing_npu - wanted
    for mid in to_remove:
        del models_block[mid]
        changed = True

    for tag in installed:
        mid = f"{NPU_MODEL_PREFIX}{tag}"
        if mid in models_block:
            continue
        port = _alloc_port()
        block = CommentedMap()
        block["cmd"] = f"flm serve {tag} --port {port} --host 127.0.0.1"
        block["proxy"] = f"http://127.0.0.1:{port}"
        block["checkEndpoint"] = "/v1/models"
        models_block[mid] = block
        changed = True

    if changed:
        save(data, path)
        rebuild_matrix(path=path)

    return sorted(wanted & set(models_block.keys()))


__all__ = [
    "load",
    "save",
    "backup",
    "restore_backup",
    "list_model_ids",
    "add_model",
    "remove_model",
    "update_ctx_size",
    "next_free_port",
    "validate",
    "get_matrix_and_globals",
    "set_matrix_and_globals",
    "get_simple_config",
    "set_simple_config",
    "switch_backend",
    "sync_npu_models",
]
