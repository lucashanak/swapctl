"""End-to-end shape test for _run_intelligence_test.

Threading the request error through answer_question widened the per-question
tuple from 3 to 4. The main scoring loop was updated, the judge path was not,
and every intelligence run then died with "too many values to unpack" — after
doing all the answering work. Unit tests on answer_question passed the whole
time because none of them ran the pipeline that consumes it.

So this test exercises the real function: answering, scoring, AND judging.
"""
import pytest

from app import tester


@pytest.fixture
def bench(monkeypatch):
    questions = [
        # one that passes outright, one that fails and so reaches the judge
        {"id": "q_ok", "category": "knowledge", "tier": "core", "difficulty": "easy",
         "prompt": "capital of France?", "max_tokens": 20, "match": "regex",
         "expected": "Paris", "regex": r"(?i)\bparis\b"},
        {"id": "q_bad", "category": "reasoning", "tier": "core", "difficulty": "hard",
         "prompt": "explain", "max_tokens": 20, "match": "regex",
         "expected": "nope", "regex": r"this-will-never-match-xyzzy"},
    ]
    monkeypatch.setattr(tester, "_load_benchmark", lambda: {"questions": questions})
    monkeypatch.setattr(tester, "_detect_thinking_support", lambda *a, **k: True)
    monkeypatch.setattr(tester, "_get_model_sampling", lambda rec: {})
    monkeypatch.setattr(tester, "_llm_judge", lambda *a, **k: False)

    def fake_chat(endpoint, model_id, prompt, max_tok, **kw):
        body = "Paris" if "France" in prompt else "some wrong but non-empty answer"
        return {"choices": [{"message": {"content": body}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5},
                "timings": {"predicted_per_second": 40}}

    monkeypatch.setattr(tester, "_chat", fake_chat)

    class Rec:
        reasoning_effort = ""
        reasoning_format = "none"
        system_prompt = ""
    monkeypatch.setattr(tester._reg if hasattr(tester, "_reg") else tester,
                        "_unused", None, raising=False)
    return Rec


def test_full_run_reaches_the_judge_path(bench, monkeypatch):
    """The judge path unpacks each response; a shape change must fail here."""
    from app import registry
    monkeypatch.setattr(registry, "get", lambda mid: bench)

    result = tester._run_intelligence_test("fake-model")

    assert result["total"] == 2
    assert result["passed"] == 1, "one answer matches, one does not"
    assert result["score"] == 50.0
    # the failing answer was non-empty, so it must have gone through the judge
    judged = [d for d in result["details"] if d["id"] == "q_bad"]
    assert judged and judged[0]["judge"] == "fail"
    assert result["errored_questions"] == 0
    assert result["degraded"] is False


def test_timed_out_question_leaves_the_denominator(bench, monkeypatch):
    """A question whose request fails must not count as a wrong answer."""
    from app import registry
    monkeypatch.setattr(registry, "get", lambda mid: bench)

    def half_broken(endpoint, model_id, prompt, max_tok, **kw):
        if "Say OK" in prompt or "France" in prompt:
            return {"choices": [{"message": {"content": "Paris"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 5},
                    "timings": {"predicted_per_second": 40}}
        raise TimeoutError("read timed out")

    monkeypatch.setattr(tester, "_chat", half_broken)

    result = tester._run_intelligence_test("fake-model")

    assert result["errored_questions"] == 1
    assert result["total"] == 1, "the timed-out question left the denominator"
    assert result["score"] == 100.0, "scored on what it actually answered"
    assert result["degraded"] is True, "half the suite lost must be flagged"
