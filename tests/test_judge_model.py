"""The judge fallback must name a model the fleet actually serves.

It was hardcoded to gpt-oss-120b. Deleting that model left the constant
pointing at nothing, and the failure is silent: _llm_judge posts to a model
llama-swap cannot start, the review of every failed answer errors out, and
the run still finishes — with a score quietly lower than it should be.
"""
from __future__ import annotations

import pytest

from app import config


@pytest.fixture
def no_override(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_JUDGE_MODEL_FILE", tmp_path / "judge_model")
    return tmp_path / "judge_model"


def test_default_is_used_when_nothing_is_stored(no_override):
    assert config.get_judge_model() == config.DEFAULT_JUDGE_MODEL


def test_default_is_not_a_deleted_model(no_override):
    """Named explicitly so the next purge trips this test instead of the suite."""
    assert "gpt-oss" not in config.get_judge_model()


def test_stored_choice_wins_over_the_default(no_override):
    config.set_judge_model("some-other-model")
    assert config.get_judge_model() == "some-other-model"


def test_blank_file_falls_back_rather_than_returning_empty(no_override):
    no_override.parent.mkdir(parents=True, exist_ok=True)
    no_override.write_text("   \n")
    assert config.get_judge_model() == config.DEFAULT_JUDGE_MODEL


def test_stored_value_is_stripped(no_override):
    config.set_judge_model("  spaced-model  ")
    assert config.get_judge_model() == "spaced-model"


def test_a_model_does_not_grade_its_own_failed_answers(no_override):
    config.set_judge_model("qwen3.8-27b")
    assert config.judge_model_for("qwen3.8-27b") != "qwen3.8-27b"
    assert config.judge_model_for("qwen3.8-27b") == config.FALLBACK_JUDGE_MODEL


def test_any_other_model_is_graded_by_the_configured_judge(no_override):
    config.set_judge_model("qwen3.8-27b")
    assert config.judge_model_for("ornith-1.5-35b-a3b-mtp") == "qwen3.8-27b"


def test_no_model_under_test_means_the_plain_judge(no_override):
    config.set_judge_model("qwen3.8-27b")
    assert config.judge_model_for(None) == "qwen3.8-27b"


def test_fallback_never_returns_the_model_under_test(no_override):
    """When the fallback itself is the judge, the guard has to pick a third."""
    config.set_judge_model(config.FALLBACK_JUDGE_MODEL)
    chosen = config.judge_model_for(config.FALLBACK_JUDGE_MODEL)
    assert chosen != config.FALLBACK_JUDGE_MODEL
    assert chosen == config.DEFAULT_JUDGE_MODEL


def test_the_two_named_judges_are_different_models(no_override):
    """Otherwise the guard above has nowhere to fall back to."""
    assert config.DEFAULT_JUDGE_MODEL != config.FALLBACK_JUDGE_MODEL
