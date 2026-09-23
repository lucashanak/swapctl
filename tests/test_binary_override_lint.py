"""A model pinned to a fork build must not silently run on a mainline one.

k2-horizon-mova-36b-a4b needs the MoVA fork. Its cmd was rewritten to
/usr/local/bin/llama-server-hip at some point, and mainline answers
`unknown model architecture: 'k2-horizon'` and exits. llama-swap logs
`ExitError >> exit status 1` and nothing more, so the model simply stopped
working — it kept its IQ score in the ranking table and could not be started
to earn a speed one. The registry held the right binary the whole time.
"""
from __future__ import annotations

import pytest

from app.config_lint import lint_config

FORK = "/opt/llama.cpp-k2/build/bin/llama-server"
MAINLINE = "/usr/local/bin/llama-server-hip"


def cfg(binary: str) -> dict:
    return {"models": {"k2": {
        "cmd": f"{binary} --model /models/k2/k2.gguf --port 9103 -ngl 999",
        "proxy": "http://127.0.0.1:9103",
    }}}


def rules(findings, rule):
    return [f for f in findings if f.rule == rule]


def test_mismatch_is_reported_as_an_error():
    f = rules(lint_config(cfg(MAINLINE), binary_overrides={"k2": FORK}),
              "binary_override_ignored")
    assert len(f) == 1
    assert f[0].level == "error"
    assert FORK in f[0].message and MAINLINE in f[0].message


def test_matching_binary_is_silent():
    assert not rules(lint_config(cfg(FORK), binary_overrides={"k2": FORK}),
                     "binary_override_ignored")


def test_no_override_recorded_means_no_opinion():
    """Most models have no pin; the rule must not invent one."""
    assert not rules(lint_config(cfg(MAINLINE)), "binary_override_ignored")
    assert not rules(lint_config(cfg(MAINLINE), binary_overrides={}),
                     "binary_override_ignored")


def test_an_empty_pin_is_not_a_pin():
    """registry.binary_override defaults to "" — that must not match everything."""
    assert not rules(lint_config(cfg(MAINLINE), binary_overrides={"k2": ""}),
                     "binary_override_ignored")


def test_pin_for_another_model_does_not_leak():
    assert not rules(lint_config(cfg(MAINLINE), binary_overrides={"other": FORK}),
                     "binary_override_ignored")


def test_leading_env_assignment_does_not_hide_the_binary():
    """The ROCm invocation prefixes LD_LIBRARY_PATH=...; token 0 is not the binary."""
    data = {"models": {"k2": {
        "cmd": f"LD_LIBRARY_PATH=/opt/rocm/lib {FORK} --model /m/k2.gguf --port 9103",
        "proxy": "http://127.0.0.1:9103",
    }}}
    assert not rules(lint_config(data, binary_overrides={"k2": FORK}),
                     "binary_override_ignored")


@pytest.mark.parametrize("cmd", [
    "flm serve gpt-oss:20b --port 9206",
    "/opt/xtts/start.sh --port 9300",
])
def test_non_llama_entries_are_not_checked(cmd):
    """Those never reach the flag rules, and a pin on them would be meaningless."""
    data = {"models": {"k2": {"cmd": cmd, "proxy": "http://127.0.0.1:9206"}}}
    assert not rules(lint_config(data, binary_overrides={"k2": FORK}),
                     "binary_override_ignored")
