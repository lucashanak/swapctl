

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
