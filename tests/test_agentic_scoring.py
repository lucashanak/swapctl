import pytest


def test_errored_episodes_excluded_from_solve_rate():
    """A transport error is not a failure to solve.

    ornith scored 33.3 because two of three episodes died on an HTTP error
    mid-run and were counted as unsolved. Errored episodes must drop out of
    the denominator, not sink the score.
    """
    from app.tester import _score_agentic
    eps = [
        {"solved": True, "turns": 5, "errored": False, "duration": 60, "tool_calls": 10},
        {"solved": False, "turns": 4, "errored": True, "duration": 30, "tool_calls": 8},
        {"solved": False, "turns": 4, "errored": True, "duration": 30, "tool_calls": 8},
    ]
    res = _score_agentic(eps, scenario="s2")
    assert res["solve_rate"] == 100.0, "one clean episode solved -> 100%, not 33%"
    assert res["errored_episodes"] == 2
    assert res["episodes"] == 3


def test_all_errored_does_not_fabricate_a_score():
    from app.tester import _score_agentic
    eps = [{"solved": False, "turns": 0, "errored": True, "duration": 5, "tool_calls": 0} for _ in range(3)]
    res = _score_agentic(eps, scenario="s2")
    assert res["score"] == 0.0 and res["errored_episodes"] == 3


def test_malformed_tool_call_is_not_charged_to_the_model():
    """Reversed on evidence: this was the harness truncating the model.

    The original version of this test asserted the opposite, on the reasoning
    that a model unable to escape code into JSON cannot edit files. A faithful
    replay disproved it — the arguments were cut off mid-string by our own
    1200-token cap, with finish_reason "length" and 2851 characters of reasoning
    ahead of them. The cap is raised and truncation is now caught before
    parsing; a 500 cannot be attributed after the fact, so it is excluded.
    """
    from app.tester import _score_agentic
    eps = [
        {"solved": True, "turns": 5, "errored": False, "duration": 60, "tool_calls": 10},
        {"solved": True, "turns": 5, "errored": False, "duration": 60, "tool_calls": 10},
        {"solved": False, "turns": 4, "errored": False, "duration": 30, "tool_calls": 8,
         "stop_reason": "malformed_output — http_500: Failed to parse tool call arguments"},
    ]
    res = _score_agentic(eps, scenario="s2")
    assert res["malformed_episodes"] == 1, "still reported, so it stays visible"


def test_error_classification_splits_model_from_infrastructure():
    from app.agentic import _is_infra_error
    parse = '{"error":{"code":500,"message":"Failed to parse tool call arguments as JSON"}}'
    assert _is_infra_error(500, parse) is True, "shown to be our truncation, not the model"
    assert _is_infra_error(503, "loading model") is True
    assert _is_infra_error(502, "bad gateway") is True
    assert _is_infra_error(400, "the request exceeds the available context size") is False
