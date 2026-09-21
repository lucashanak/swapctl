"""Static checks over a llama-swap config.

Every rule here exists because the corresponding breakage actually happened and
cost hours to find by hand:

- a model whose ``--port`` disagreed with its ``proxy:`` never came up: llama-swap
  polled a port nothing was listening on and killed the load on timeout;
- two models sharing a ``--port`` meant whichever loaded second answered as the
  first;
- ``--parallel N`` splits ``--ctx-size`` into N slots of ctx/N (measured), so a
  model advertising 65536 served 16384-token requests and a client died with
  ``ContextWindowExceededError`` — no error at load, nothing in the UI;
- a generated block pointed at a binary that cannot read the model's format, so
  the server exited during load with only ``exit status 1`` in the journal;
- a model shipping its own MTP head ran without speculation because nobody
  looked inside the GGUF.

The checks are pure functions over the parsed config so they can be unit-tested
without a node; filesystem- and GGUF-dependent rules are opt-in via flags.
"""
from __future__ import annotations

from .gguf_meta import has_mtp_tensors

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

ERROR = "error"
WARN = "warn"
INFO = "info"

# Below this many tokens per slot a "long context" model is not one. 8192 is the
# smallest context any model in this fleet is useful at.
MIN_CTX_PER_SLOT = 8192

# llama.cpp accepts both `--flag value` and `--flag=value`, so every pattern
# has to allow either separator, and every flag is anchored to a token start so
# it cannot match the tail of a path or a longer flag.
_SEP = r"[\s=]+"
_B = r"(?:^|\s)"
_PORT_RE = re.compile(_B + r"--port" + _SEP + r"(\d+)")
_CTX_RE = re.compile(_B + r"(?:--ctx-size|-c)" + _SEP + r"(\d+)")
_PARALLEL_RE = re.compile(_B + r"(?:--parallel|-np)" + _SEP + r"(\d+)")
_MODEL_RE = re.compile(_B + r"--model" + _SEP + r"(\S+)")
_DRAFT_RE = re.compile(_B + r"(?:--spec-draft-model|--model-draft|-md)" + _SEP + r"(\S+)")
_SPEC_TYPE_RE = re.compile(_B + r"--spec-type" + _SEP + r"(\S+)")
# Removed upstream; llama-server exits with "the argument has been removed"
# before it binds a port. Matched as whole flags so a path cannot trip it.
_DEAD_FLAG_RE = re.compile(_B + r"(--draft(?:-max|-min|-n)?)(?=[\s=]|$)")
# Leading VAR=value tokens (e.g. LD_LIBRARY_PATH=... for the ROCm builds).
_ENV_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Servers this module deliberately has no rules for, as opposed to ones it
# failed to recognise. They speak their own flags: NPU models run `flm serve`,
# and whisper/xtts are separate projects entirely. Keep this list tight —
# anything not on it that looks server-shaped gets reported rather than
# silently skipped.
_NON_LLAMA_BINARIES = ("flm", "whisper-server")


@dataclass(frozen=True)
class Finding:
    level: str
    model_id: str
    rule: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.model_id}: {self.message} ({self.rule})"


def _binary_of(cmd: str) -> str:
    """The executable, skipping any leading VAR=value environment assignments.

    The documented ROCm invocation is `LD_LIBRARY_PATH=/opt/rocm/lib
    /usr/local/bin/llama-server ...`; taking token 0 blindly reported that as a
    missing binary.
    """
    for tok in cmd.split():
        if not _ENV_TOKEN_RE.match(tok):
            return tok
    return ""


def _is_known_non_llama(cmd: str) -> bool:
    """Entries we deliberately have no rules for, as opposed to ones we failed
    to recognise. NPU models run `flm serve`; a `.sh` launcher owns its own
    arguments. Both are expected shapes and must stay silent."""
    name = _binary_of(cmd).rsplit("/", 1)[-1]
    return name in _NON_LLAMA_BINARIES or name.endswith(".sh")


def _is_llama_server(cmd: str) -> bool:
    """Whether the flag rules apply to this entry at all.

    Identified positively, by the binary's name. Deciding it negatively — "has
    no flags, so it's a wrapper" — broke the moment a wrapper took one flag:
    `/opt/xtts/start.sh --device cuda` would have been reported as missing a
    --port it never accepts.
    """
    binary = _binary_of(cmd)
    if not binary:
        return False
    name = binary.rsplit("/", 1)[-1]
    if name in _NON_LLAMA_BINARIES:
        return False
    return name.startswith("llama-server")


def _int_or_none(pattern: re.Pattern, text: str) -> int | None:
    m = pattern.search(text)
    return int(m.group(1)) if m else None


def _proxy_port(proxy: str) -> int | None:
    """Port from a proxy URL, tolerating a path or trailing slash.

    An end-anchored `:(\\d+)$` silently skipped `http://127.0.0.1:9199/`, which
    disabled the port/proxy rule on a purely cosmetic difference.
    """
    if not proxy:
        return None
    try:
        return urlsplit(proxy if "//" in proxy else "//" + proxy).port
    except ValueError:
        return None


def lint_config(
    data: dict,
    *,
    check_paths: bool = False,
    check_gguf: bool = False,
) -> list[Finding]:
    """Return findings for a parsed llama-swap config.

    ``check_paths`` enables filesystem existence checks (binary, --model, -md).
    ``check_gguf`` additionally reads GGUF headers to spot an unused MTP head;
    both are off by default so the rules stay unit-testable.
    """
    models = data.get("models") or {}
    findings: list[Finding] = []
    ports: dict[int, list[str]] = {}
    # Every path some other model points -md at. A sidecar that is in use is
    # not "an unused head"; one that nobody references is an orphan, which is
    # a different finding with different advice.
    drafted: set[str] = set()
    for blk in models.values():
        m = _DRAFT_RE.search(str((blk or {}).get("cmd") or ""))
        if m:
            drafted.add(m.group(1))

    for model_id, block in models.items():
        cmd = str((block or {}).get("cmd") or "").strip()
        proxy = str((block or {}).get("proxy") or "")
        if not cmd:
            findings.append(Finding(ERROR, model_id, "no_cmd", "model block has no cmd"))
            continue

        cmd_port = _int_or_none(_PORT_RE, cmd)
        proxy_port = _proxy_port(proxy)

        # Not every entry is a llama-server invocation: xtts-v2 runs a wrapper
        # script and NPU models run `flm serve`, neither of which takes these
        # flags. Reporting those would teach everyone to ignore the linter.
        # They still own a port, so keep them in the collision check.
        if not _is_llama_server(cmd):
            if proxy_port is not None:
                ports.setdefault(proxy_port, []).append(model_id)
            # Silence here means "not checked", not "fine". An entry that looks
            # server-shaped but whose binary we do not recognise (a typo, a
            # shell wrapper) would otherwise skip every rule without a trace.
            if _is_known_non_llama(cmd):
                continue
            if _PORT_RE.search(cmd) or _MODEL_RE.search(cmd):
                findings.append(Finding(
                    INFO, model_id, "unrecognised_binary",
                    f"flag rules skipped: {_binary_of(cmd)!r} is not a recognised "
                    "llama-server binary",
                ))
            continue

        # A proxy pointing somewhere else still occupies that port at runtime.
        for p in {v for v in (cmd_port, proxy_port) if v is not None}:
            ports.setdefault(p, []).append(model_id)

        if cmd_port is None:
            findings.append(Finding(ERROR, model_id, "no_port", "cmd has no --port"))
        elif proxy_port is not None and proxy_port != cmd_port:
            findings.append(Finding(
                ERROR, model_id, "port_proxy_mismatch",
                f"cmd listens on {cmd_port} but proxy points at {proxy_port}; "
                "llama-swap will wait on the wrong port and kill the load",
            ))

        ctx = _int_or_none(_CTX_RE, cmd)
        par = _int_or_none(_PARALLEL_RE, cmd)
        if par and par > 1 and not ctx:
            # `--ctx-size 0`/absent means "whatever the model declares", which
            # is still divided by the slot count — the gap is real, only its
            # size is unknown here.
            findings.append(Finding(
                WARN, model_id, "context_split",
                f"--parallel {par} splits the model's default context into {par} "
                "slots; requests see a fraction of it",
            ))
        elif ctx and par and par > 1:
            # The defect is the silent gap between declared and effective
            # context, not the absolute slot size: ornith split 65536 into
            # 16384 — comfortably above any floor — and still broke a client
            # that trusted the declared number. So always report the gap, and
            # escalate only when a slot is too small to be useful at all.
            per_slot = ctx // par
            findings.append(Finding(
                WARN if per_slot < MIN_CTX_PER_SLOT else INFO,
                model_id, "context_split",
                f"--parallel {par} splits --ctx-size {ctx} into {par} slots of "
                f"{per_slot} tokens each; requests see {per_slot}, not {ctx}",
            ))

        dead = _DEAD_FLAG_RE.search(cmd)
        if dead:
            findings.append(Finding(
                ERROR, model_id, "dead_flag",
                f"{dead.group(1)} was removed upstream; llama-server exits before "
                "binding a port. Use --spec-draft-n-max / --spec-draft-n-min",
            ))

        if _DRAFT_RE.search(cmd) and not _SPEC_TYPE_RE.search(cmd):
            findings.append(Finding(
                WARN, model_id, "draft_without_spec_type",
                "a draft model is given but --spec-type is missing, so it is ignored",
            ))

        if check_paths:
            findings.extend(_path_findings(model_id, cmd))
        if check_gguf:
            findings.extend(_gguf_findings(model_id, cmd, drafted))

    for port, owners in sorted(ports.items()):
        if len(owners) > 1:
            for model_id in owners:
                findings.append(Finding(
                    ERROR, model_id, "port_collision",
                    f"port {port} is also used by {', '.join(o for o in owners if o != model_id)}",
                ))

    return findings


def _path_findings(model_id: str, cmd: str) -> list[Finding]:
    out: list[Finding] = []
    binary = _binary_of(cmd)
    if binary and not Path(binary).exists():
        out.append(Finding(ERROR, model_id, "binary_missing",
                           f"binary {binary} does not exist"))
    for rule, pattern in (("model_missing", _MODEL_RE), ("draft_missing", _DRAFT_RE)):
        m = pattern.search(cmd)
        if m and not Path(m.group(1)).exists():
            out.append(Finding(ERROR, model_id, rule, f"{m.group(1)} does not exist"))
    return out


def _gguf_findings(model_id: str, cmd: str, drafted: set[str]) -> list[Finding]:
    """Flag a model that carries an MTP head it never uses.

    Reading tensor names touches only the header, but it is still file I/O per
    model, which is why this rule is opt-in.
    """
    m = _MODEL_RE.search(cmd)
    if not m or _SPEC_TYPE_RE.search(cmd):
        return []
    path = Path(m.group(1))
    if not path.exists():
        return []
    try:
        from .gguf_meta import read_tensor_names
        names = read_tensor_names(path)
    except Exception:
        return []
    if str(path) in drafted:
        # Some target points -md here: this entry exists to be a draft, and is
        # in use. Nothing to say.
        return []
    if _is_head_only(names):
        return [Finding(
            INFO, model_id, "orphan_draft",
            "this looks like a standalone MTP/draft sidecar, but no model "
            "points -md at it, so it is dead weight in the config",
        )]
    if has_mtp_tensors(names):
        return [Finding(
            WARN, model_id, "mtp_head_unused",
            "the GGUF contains an MTP head (blk.*.nextn.* or mtp.*) but the cmd "
            "has no --spec-type, so it is left unused. Nothing wires this "
            "automatically: switch the model to an MTP-capable backend, which "
            "owns the spec flags and their tuning",
        )]
    return []


def _is_head_only(tensor_names: list[str]) -> bool:
    """True when the GGUF is a standalone MTP sidecar rather than a model.

    A sidecar carries one block (the head) and a few dozen tensors; a target
    carries hundreds across every layer. Distinguishing them matters because
    four of the six live hits for `mtp_head_unused` were sidecars, and a rule
    that is wrong two thirds of the time stops being read.
    """
    blocks = set()
    for n in tensor_names:
        parts = n.split(".")
        for i, s in enumerate(parts):
            if s == "blk" and i + 1 < len(parts) and parts[i + 1].isdigit():
                blocks.add(int(parts[i + 1]))
    return len(blocks) == 1 and has_mtp_tensors(tensor_names)


def format_findings(findings: list[Finding]) -> str:
    if not findings:
        return "config OK"
    order = {ERROR: 0, WARN: 1, INFO: 2}
    ranked = sorted(findings, key=lambda f: (order.get(f.level, 9), f.model_id))
    return "\n".join(str(f) for f in ranked)


def worst_level(findings: list[Finding]) -> str | None:
    for level in (ERROR, WARN, INFO):
        if any(f.level == level for f in findings):
            return level
    return None
