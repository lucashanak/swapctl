"""A failed request must never be scored as a wrong answer.

bonsai-g64-27b reported 37.6 on the intelligence suite. 79 of its 149 answers
were empty: at 9.7 tok/s an uncapped 4096-token answer needs ~420s and the
request timeout was a fixed 300s. Every timeout was graded as a wrong answer,
so the score measured the ceiling rather than the model.
"""
import pytest

from app import tester


def test_timeout_scales_with_the_answer_length():
    """A slow model must be given time proportional to what it was asked for."""
    q_short = {"id": "x", "category": "knowledge", "prompt": "hi", "max_tokens": 50}
    q_long = {"id": "y", "category": "exec", "prompt": "hi", "max_tokens": 4096}

    seen = {}

    def fake_chat(endpoint, model_id, prompt, max_tok, **kw):
        seen[max_tok] = kw.get("timeout")
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1}, "timings": {"predicted_per_second": 10}}

    orig = tester._chat
    tester._chat = fake_chat
    try:
        tester.answer_question(q_short, "http://x", "m", thinking=False)
        tester.answer_question(q_long, "http://x", "m", thinking=False)
    finally:
        tester._chat = orig

    timeouts = sorted(t for t in seen.values() if t)
    assert timeouts[-1] > timeouts[0], "a longer answer must get a longer timeout"
    # 4096 tokens at a 4 tok/s floor is ~17 minutes; 300s would guarantee a timeout
    assert timeouts[-1] >= 900, f"4096-token answer got only {timeouts[-1]}s"


def test_request_failure_is_reported_not_swallowed():
    """The bare 'except Exception: pass' returned '' and lost the reason."""
    q = {"id": "x", "category": "knowledge", "prompt": "hi", "max_tokens": 50}

    def boom(*a, **kw):
        raise TimeoutError("read timed out")

    orig = tester._chat
    tester._chat = boom
    try:
        content, tg, tokens, err = tester.answer_question(q, "http://x", "m", thinking=False)
    finally:
        tester._chat = orig

    assert content == ""
    assert err is not None and "TimeoutError" in err, "the failure must be reported"


def test_context_overflow_counts_against_the_model():
    """A prompt that does not fit the deployed context is a failure, not an excuse.

    qwen3-next-80b is deployed at --ctx-size 16384. The 10 long-context questions
    above that returned 400 and were dropped from the denominator, quietly
    excusing exactly the questions it cannot answer.
    """
    from app.tester import _is_context_error
    assert _is_context_error(
        "http_400: the request exceeds the available context size") is True
    assert _is_context_error("http_400: prompt is too long for n_ctx") is True
    # not a context problem -> still excluded as infrastructure
    assert _is_context_error("http_503: loading model") is False
    assert _is_context_error("TimeoutError: read timed out") is False
    assert _is_context_error("http_500: internal error") is False
