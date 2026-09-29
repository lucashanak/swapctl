"""The IQ judge must be something llama-swap can serve, and the score must say
who graded it.

The configured judge was deleted twice (gpt-oss-120b, then qwen3.8-27b). The
warm-up then failed, judging was skipped with a log line and nothing else, and
the stored score was judge-free while looking like any other.
"""
from __future__ import annotations

import pytest

from app import config, tester


@pytest.fixture
def served(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_JUDGE_MODEL_FILE", tmp_path / "judge_model")

    def set_(names):
        monkeypatch.setattr(tester, "_servable_names", lambda: set(names))
    return set_


def test_configured_judge_used_when_servable(served):
    config.set_judge_model("gufo-qwen3.8-27b")
    served({"gufo-qwen3.8-27b", "qwen3.8-flash-next", "ornith"})
    assert tester.resolve_judge("ornith") == "gufo-qwen3.8-27b"


def test_deleted_judge_falls_back_instead_of_silently_skipping(served):
    """What happened when qwen3.8-27b was removed in the UI."""
    config.set_judge_model("qwen3.8-27b")
    served({"qwen3.8-flash-next", "ornith"})
    assert tester.resolve_judge("ornith") == config.FALLBACK_JUDGE_MODEL


def test_an_alias_counts_as_servable(served):
    """The fallback judge is an alias; a registry-only check would miss it."""
    config.set_judge_model("gone")
    served({"qwen38-flash-next-agention", "qwen3.8-flash-next"})  # id and its alias
    assert tester.resolve_judge("x") == "qwen3.8-flash-next"


def test_never_the_model_under_test(served):
    config.set_judge_model("gufo-qwen3.8-27b")
    served({"gufo-qwen3.8-27b", "qwen3.8-flash-next"})
    assert tester.resolve_judge("gufo-qwen3.8-27b") == "qwen3.8-flash-next"


def test_nothing_servable_is_none_not_a_dead_name(served):
    config.set_judge_model("gone")
    served({"some-unrelated-model"})
    assert tester.resolve_judge("x") is None


def test_unreadable_config_keeps_the_old_behaviour(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_JUDGE_MODEL_FILE", tmp_path / "judge_model")
    monkeypatch.setattr(tester, "_servable_names", lambda: None)
    config.set_judge_model("gufo-qwen3.8-27b")
    assert tester.resolve_judge("x") == "gufo-qwen3.8-27b"


def test_servable_names_reads_ids_and_aliases(monkeypatch):
    from app import swap_config
    monkeypatch.setattr(swap_config, "load", lambda path=None: {"models": {
        "qwen38-flash-next-agention": {"aliases": ["qwen3.8-flash-next", "flash-next"]},
        "ornith-1.5-35b-a3b-mtp": {},
    }})
    assert tester._servable_names() == {
        "qwen38-flash-next-agention", "qwen3.8-flash-next", "flash-next", "ornith-1.5-35b-a3b-mtp"}
