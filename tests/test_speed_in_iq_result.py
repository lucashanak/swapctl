"""Decode speed must survive an intelligence run into the stored result.

It was measured on every question and then dropped: `answer_question` returns
(content, tg, tokens, err), but the response tuple kept only
(q, content, tokens, err). The value reached the progress bar and nothing else,
so the ranking table had intelligence and no speed — and on this fleet the two
trade against each other, which makes an IQ-only ranking unusable for deciding
anything.
"""
from __future__ import annotations

import pytest

from app import tester


def test_empty_run_reports_zero_not_a_crash():
    s = tester._speed_summary([])
    assert s == {"decode_tps": 0.0, "decode_tps_mean": 0.0, "decode_samples": 0}


def test_median_is_the_headline_not_the_mean():
    """One slow question must not drag the figure the ranking shows.

    Real spread on this hardware: flash-next produced 25.8-39.2 t/s across three
    identical runs, and a question with a long prefill reports far lower.
    """
    s = tester._speed_summary([40.0, 39.0, 38.0, 41.0, 2.0])
    assert s["decode_tps"] == 39.0, "median ignores the outlier"
    assert s["decode_tps_mean"] == 32.0, "the mean does not, which is why it is not the headline"
    assert s["decode_samples"] == 5


def test_even_sample_count_averages_the_middle_pair():
    s = tester._speed_summary([10.0, 20.0, 30.0, 40.0])
    assert s["decode_tps"] == 25.0


@pytest.fixture
def bench(monkeypatch):
    questions = [
        {"id": "q_ok", "category": "knowledge", "tier": "core", "difficulty": "easy",
         "prompt": "capital of France?", "max_tokens": 20, "match": "regex",
         "expected": "Paris", "regex": r"(?i)\bparis\b"},
        {"id": "q_ok2", "category": "knowledge", "tier": "core", "difficulty": "easy",
         "prompt": "capital of Italy?", "max_tokens": 20, "match": "regex",
         "expected": "Rome", "regex": r"(?i)\brome\b"},
    ]
    monkeypatch.setattr(tester, "_load_benchmark", lambda: {"questions": questions})
    monkeypatch.setattr(tester, "_detect_thinking_support", lambda *a, **k: True)
    monkeypatch.setattr(tester, "_get_model_sampling", lambda rec: {})
    monkeypatch.setattr(tester, "_llm_judge", lambda *a, **k: False)

    # Keyed on the prompt, not call order: a warm-up "Say OK" runs before the
    # questions and would otherwise eat the first value.
    def fake_chat(endpoint, model_id, prompt, max_tok, **kw):
        if "Say OK" in prompt:
            return {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 1},
                    "timings": {"predicted_per_second": 999.0}}
        france = "France" in prompt
        return {"choices": [{"message": {"content": "Paris" if france else "Rome"},
                             "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5},
                "timings": {"predicted_per_second": 50.0 if france else 30.0}}

    monkeypatch.setattr(tester, "_chat", fake_chat)

    class Rec:
        reasoning_effort = ""
        reasoning_format = "none"
        system_prompt = ""
    return Rec


def test_run_result_carries_decode_speed(bench, monkeypatch):
    """The end-to-end path, not just the helper — that is where it was lost."""
    from app import registry
    monkeypatch.setattr(registry, "get", lambda mid: bench)

    result = tester._run_intelligence_test("fake-model")

    assert result["decode_samples"] == 2
    assert result["decode_tps"] == 40.0          # median of 50 and 30
    assert result["decode_tps_mean"] == 40.0


def test_per_question_detail_carries_its_own_speed(bench, monkeypatch):
    """So a slow category can be told apart from a slow model."""
    from app import registry
    monkeypatch.setattr(registry, "get", lambda mid: bench)

    result = tester._run_intelligence_test("fake-model")
    tps = sorted(d["tps"] for d in result["details"])
    assert tps == [30.0, 50.0]
