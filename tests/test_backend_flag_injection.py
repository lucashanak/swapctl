"""What every registered backend injects into a generated command.

`_inject_backend_flags` runs after `_build_cmd`, so a backend can silently undo
what the generator just did. Two of them did: `vulkan-atomic-nextn` and
`vulkan-atomic-gemma4-mtp` carried `--draft-max 3` in `default_flags` *and*
listed the correct `--spec-draft-n-max` as incompatible, so the injector
stripped the working spelling and put back the one llama-server refuses to
start on. Nothing tested the injector's output, so a green suite said the
generator fix worked while every model on those backends stayed dead.

These tests run against the real seed definitions rather than a fixture, so a
future edit to backends_registry that reintroduces a removed flag fails here.
"""
from __future__ import annotations

import pytest

from app import swap_config
from app.backends_registry import _DEFAULT_BACKENDS, BackendDef

# Imported, not copied: a second copy of this pattern is the same drift that
# caused the MTP detector bug this change exists to fix.
from app.config_lint import _DEAD_FLAG_RE as DEAD_FLAG_RE

BASE_CMD = (
    "/usr/local/bin/llama-server-vulkan --model /models/m/m.gguf --port 9200 "
    "-ngl 999 --cache-type-k q8_0 --cache-type-v q8_0 --jinja "
    "--ctx-size 65536 --parallel 1 --flash-attn on"
)


def _backend(d: dict) -> BackendDef:
    return BackendDef(
        id=d["id"],
        binary_path=d["binary_path"],
        label=d.get("label", d["id"]),
        caps=list(d.get("caps") or []),
        default_flags=list(d.get("default_flags") or []),
        required_flags=list(d.get("required_flags") or []),
        incompatible_flag_patterns=list(d.get("incompatible_flag_patterns") or []),
        model_predicate=d.get("model_predicate"),
        version_hint=d.get("version_hint", ""),
        builtin=d.get("builtin", False),
    )


ALL_BACKENDS = [pytest.param(d, id=d["id"]) for d in _DEFAULT_BACKENDS]


@pytest.mark.parametrize("bdef", ALL_BACKENDS)
def test_no_backend_injects_a_removed_flag(bdef: dict):
    out = swap_config._inject_backend_flags(BASE_CMD, _backend(bdef))
    hit = DEAD_FLAG_RE.search(out)
    assert hit is None, f"{bdef['id']} injects {hit.group(0).strip()!r}: {out}"


@pytest.mark.parametrize("bdef", ALL_BACKENDS)
def test_injected_output_passes_the_linter(bdef: dict):
    """A backend must not be able to produce a config we would call broken."""
    from app.config_lint import lint_config, worst_level

    out = swap_config._inject_backend_flags(BASE_CMD, _backend(bdef))
    data = {"models": {"m": {"cmd": out, "proxy": "http://127.0.0.1:9200"}}}
    findings = lint_config(data)
    assert worst_level(findings) != "error", [str(f) for f in findings]


@pytest.mark.parametrize("bdef", ALL_BACKENDS)
def test_spec_flags_survive_their_own_incompatible_patterns(bdef: dict):
    """A backend that asks for a spec flag must not also strip it.

    The two atomic backends listed `--spec-draft-n-max\\s+\\d+` as incompatible
    while requesting a draft length, which is self-contradictory: whichever
    spelling `_build_cmd` used, the result was wrong.
    """
    out = swap_config._inject_backend_flags(BASE_CMD, _backend(bdef))
    for flag in _backend(bdef).default_flags + _backend(bdef).required_flags:
        # the full string, not just the flag name: every name here is already
        # in BASE_CMD, so a name-only check passes no matter what the backend
        # did to the value.
        assert flag in out, f"{bdef['id']} declares {flag!r} but it is absent from {out}"


def test_mtp_backend_keeps_its_tuned_draft_length():
    """vulkan-mtp is tuned to 3; nothing upstream of it may pin a lower value."""
    d = next(b for b in _DEFAULT_BACKENDS if b["id"] == "vulkan-mtp")
    out = swap_config._inject_backend_flags(BASE_CMD, _backend(d))
    assert "--spec-draft-n-max 3" in out


def test_parallel_guard_strips_a_multi_slot_value():
    """The MTP backends require --parallel 1 and must enforce it."""
    d = next(b for b in _DEFAULT_BACKENDS if b["id"] == "vulkan-mtp")
    noisy = BASE_CMD.replace("--parallel 1", "--parallel 4")
    out = swap_config._inject_backend_flags(noisy, _backend(d))
    assert "--parallel 4" not in out
    assert "--parallel 1" in out
