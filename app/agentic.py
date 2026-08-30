"""Agentic test: can the model drive tools to fix a broken project?

Everything the suite measures elsewhere is a single request/response. This is a
loop: the model is handed a small broken repo and four tools, and has to find the
bugs, edit files and re-run the tests until they pass. The fleet's own history says
this is the axis that separates models when plain coding has saturated — one model
solved a buggy pytest suite in 8 turns where another needed 14, while both scored
identically on static coding tasks.

Runs entirely against the llama-swap OpenAI endpoint, so there is no dependency on
an external agent CLI or gateway: what gets measured is the model's own tool
calling, not a wrapper's.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import httpx

MAX_TURNS = 24
TOOL_TIMEOUT = 20.0


# --- the broken project ---------------------------------------------------
# Three independent bugs, one per function, each with a passing test alongside a
# failing one so the model cannot "fix" things by deleting behaviour:
#   normalise_spaces  - collapses runs but drops the leading/trailing strip
#   parse_range       - off-by-one, excludes the end of an inclusive range
#   safe_div          - returns None instead of the documented default on /0
FILES_S1 = {
    "utils.py": '''\
def normalise_spaces(s):
    """Collapse runs of whitespace to one space and strip the ends."""
    out = []
    prev_space = False
    for ch in s:
        if ch.isspace():
            if not prev_space:
                out.append(" ")
            prev_space = True
        else:
            out.append(ch)
            prev_space = False
    return "".join(out)


def parse_range(spec):
    """'3-6' -> [3, 4, 5, 6]. A bare '4' -> [4]. Inclusive of both ends."""
    if "-" not in spec:
        return [int(spec)]
    lo, hi = spec.split("-", 1)
    return list(range(int(lo), int(hi)))


def safe_div(a, b, default=0.0):
    """Divide, returning `default` when the divisor is zero."""
    if b == 0:
        return None
    return a / b
''',
    "test_utils.py": '''\
from utils import normalise_spaces, parse_range, safe_div


def test_normalise_collapses():
    assert normalise_spaces("a   b") == "a b"


def test_normalise_strips():
    assert normalise_spaces("  a  b  ") == "a b"


def test_parse_single():
    assert parse_range("4") == [4]


def test_parse_inclusive():
    assert parse_range("3-6") == [3, 4, 5, 6]


def test_div_normal():
    assert safe_div(10, 4) == 2.5


def test_div_by_zero_uses_default():
    assert safe_div(1, 0) == 0.0
    assert safe_div(1, 0, default=-1.0) == -1.0
''',
}

SYSTEM = (
    "You are fixing a small Python project. Use the tools to inspect and edit files, "
    "then run the tests. Keep working until every test passes. Do not rewrite the "
    "tests — the tests are correct and the source has bugs. When all tests pass, say "
    "DONE and stop."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List files in the project.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file's full contents.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Overwrite a file with new contents. Provide the COMPLETE file.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run the test suite and return the result.",
        "parameters": {"type": "object", "properties": {}}}},
]


def _run_tests(workdir: str) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", "-x"],
            cwd=workdir, capture_output=True, timeout=TOOL_TIMEOUT,
            env={"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir},
        )
        out = (proc.stdout + proc.stderr).decode(errors="replace")[-1500:]
        return proc.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: the test run exceeded its limit"
    except OSError as exc:
        return False, f"could not run tests: {exc}"


def _dispatch(name: str, args: dict, workdir: str) -> str:
    """Tool results are strings — models handle those far more reliably than JSON."""
    if name == "list_files":
        return "\n".join(sorted(os.listdir(workdir)))
    if name == "read_file":
        path = os.path.basename(str(args.get("path", "")))
        full = os.path.join(workdir, path)
        if not os.path.isfile(full):
            return f"no such file: {path}"
        with open(full) as fh:
            return fh.read()
    if name == "write_file":
        path = os.path.basename(str(args.get("path", "")))
        content = args.get("content")
        if not path or content is None:
            return "write_file needs both path and content"
        if path.startswith("test_"):
            return "refused: the tests are correct and must not be modified"
        with open(os.path.join(workdir, path), "w") as fh:
            fh.write(content)
        return f"wrote {path} ({len(content)} bytes)"
    if name == "run_tests":
        ok, out = _run_tests(workdir)
        return ("ALL TESTS PASS\n" if ok else "TESTS FAILED\n") + out
    return f"unknown tool: {name}"


def run_agentic(endpoint: str, model_id: str, timeout: float = 900.0,
                scenario: str = "s2") -> dict:
    """One episode. Returns solved/turns/tool_calls and why it stopped."""
    workdir = tempfile.mkdtemp(prefix="agentic-")
    try:
        for name, body in SCENARIOS[scenario].items():
            with open(os.path.join(workdir, name), "w") as fh:
                fh.write(body)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                "The test suite in this project fails. Find and fix the bugs in the "
                "source (not the tests) until `run_tests` reports that all tests pass."},
        ]
        t0 = time.time()
        turns = tool_calls = 0
        solved = False
        stop = "max_turns"
        no_tool_turns = 0

        with httpx.Client(timeout=timeout) as client:
            while turns < MAX_TURNS:
                turns += 1
                try:
                    r = client.post(f"{endpoint}/v1/chat/completions", json={
                        "model": model_id, "messages": messages, "tools": TOOLS,
                        "tool_choice": "auto", "max_tokens": 1200, "temperature": 0,
                    })
                    r.raise_for_status()
                    msg = r.json()["choices"][0]["message"]
                except Exception as exc:
                    stop = f"api_error: {type(exc).__name__}"
                    break

                messages.append(msg)
                calls = msg.get("tool_calls") or []
                if not calls:
                    no_tool_turns += 1
                    # two content-only replies in a row means it has stopped driving
                    # the loop; nudge once, then give up rather than burn the budget
                    if no_tool_turns >= 2:
                        stop = "stopped_calling_tools"
                        break
                    messages.append({"role": "user", "content":
                                     "Use the tools. Call run_tests to check the current state."})
                    continue
                no_tool_turns = 0

                for call in calls:
                    tool_calls += 1
                    fn = (call.get("function") or {})
                    name = fn.get("name", "")
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = _dispatch(name, args, workdir)
                    messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                     "content": result[:4000]})
                    if name == "run_tests" and result.startswith("ALL TESTS PASS"):
                        solved = True
                        stop = "solved"
                if solved:
                    break

        # never trust the model's own claim — verify from the filesystem
        verified, _ = _run_tests(workdir)
        return {
            "solved": bool(verified),
            "claimed_solved": solved,
            "turns": turns,
            "tool_calls": tool_calls,
            "duration": round(time.time() - t0, 1),
            "stop_reason": stop,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


FILES_S2 = {
    "defaults.py": '''\
# Applied to any field the config file omits.
DEFAULTS = {
    "workers": 4,
    "timeout": 30,
    "retries": 0,
    "mode": "batch",
}


def apply_defaults(cfg):
    out = dict(DEFAULTS)
    out.update(cfg)
    return out
''',
    "parser.py": '''\
def parse_config(text):
    """Parse 'key = value' lines. '#' starts a comment. Blank lines are skipped.

    Values that look like integers are converted to int.
    """
    cfg = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"bad line: {raw!r}")
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value.lstrip("-").isdigit():
            value = int(value)
        cfg[key] = value
    return cfg
''',
    "validator.py": '''\
MIN_WORKERS = 1
VALID_MODES = ("batch", "stream")


def validate(cfg):
    """Raise ValueError if the config is unusable. Assumes defaults are applied."""
    if "workers" not in cfg:
        raise ValueError("workers missing")
    if not isinstance(cfg["workers"], int) or cfg["workers"] < MIN_WORKERS:
        raise ValueError("workers must be >= 1")
    if cfg.get("mode") not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}")
    return True
''',
    "pipeline.py": '''\
from parser import parse_config
from validator import validate
from defaults import apply_defaults


def load(text):
    """Parse a config, check it, and return it ready to use."""
    cfg = parse_config(text)
    validate(cfg)
    cfg = apply_defaults(cfg)
    return cfg
''',
    "test_pipeline.py": '''\
import pytest
from pipeline import load
from parser import parse_config


def test_parses_plain_values():
    assert parse_config("mode = stream") == {"mode": "stream"}


def test_strips_trailing_comment():
    assert parse_config("workers = 8  # tuned for the big box") == {"workers": 8}


def test_omitted_fields_get_defaults():
    cfg = load("mode = stream")
    assert cfg["workers"] == 4
    assert cfg["retries"] == 3


def test_explicit_values_win_over_defaults():
    cfg = load("workers = 9\\nmode = batch")
    assert cfg["workers"] == 9


def test_rejects_zero_workers():
    with pytest.raises(ValueError):
        load("workers = 0\\nmode = batch")


def test_rejects_unknown_mode():
    with pytest.raises(ValueError):
        load("mode = turbo")


def test_validation_sees_defaulted_mode():
    # mode is omitted, so the default "batch" applies and the config is legal.
    # Relaxing the mode check to make this pass breaks test_rejects_unknown_mode:
    # only running validation AFTER defaults satisfies both.
    cfg = load("workers = 2")
    assert cfg["mode"] == "batch"
''',
}


SCENARIOS = {
    "s1": FILES_S1,   # two files, three bugs each visible in a single read
    "s2": FILES_S2,   # five files; one bug only appears when the modules run together
}
