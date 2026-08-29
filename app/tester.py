"""Background model testing: speed benchmarks (per-backend) and intelligence scoring."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from . import SwapctlError
from .config import settings

log = logging.getLogger("swapctl.tester")

BENCHMARK_PATH = Path(__file__).resolve().parent / "benchmarks" / "intelligence.json"
HISTORY_PATH = Path("/var/lib/swapctl/test_history.jsonl")

_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_queue: list[dict] = []
_cancelled: set[str] = set()
_worker_thread: threading.Thread | None = None


@dataclass
class SpeedResult:
    backend: str
    tg_speed: float
    pp_speed: float
    load_time: int
    tested_at: str
    tests: dict = field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_benchmark() -> dict:
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


# Thinking budgets calibrated to gpt-oss reasoning_effort tier token consumption,
# so IQ test duration is comparable across thinking models regardless of backend.
# MEDIUM = 1024: empirical sweet spot for qwen3.6 from A/B on qwen3.6-35b-a3b-mtp-iq4xs.
#   2048: score 88.5, 1173s (non-binding, model overthought past it)
#   1024: score 89.0,  983s, math 100 / reasoning 90  <- best speed/quality
#    512: score 84.5,  814s, math 70 / reasoning 70    <- gpt-oss token level (~324/101)
#         was too aggressive; qwen3.6 needs more thinking room than gpt-oss-120b.
_TIER_LOW = 512
_TIER_MEDIUM = 1024
_TIER_HIGH = 8192

# A budget of 0 disables thinking outright for most models, but gpt-oss has no
# off-switch — _chat() floors it at reasoning_effort "low". That made five of the
# ten categories a handicap match: measured on `instruction`, thinking is worth
# +25 points to BOTH qwen3.8-27b (70 -> 95) and gpt-oss-120b (70 -> 95), and the
# whole gpt-oss lead in the fleet ranking came from that category alone.
# So the floor is _TIER_LOW everywhere: every model now gets the same small
# allowance that gpt-oss cannot switch off.
_THINKING_BUDGET_BY_CATEGORY = {
    "reasoning": _TIER_MEDIUM,
    "math": _TIER_MEDIUM,
    "code": _TIER_LOW,
    "sysadmin": _TIER_LOW,
    "knowledge": _TIER_LOW,
    "coding": _TIER_LOW,
    "instruction": _TIER_LOW,
    "structured": _TIER_LOW,
    "czech": _TIER_LOW,
    "summarization": _TIER_LOW,
    # multi-hop retrieval over a long document: the reasoning is trivial once both
    # facts are found, so a MEDIUM budget only pays for the search, not for thinking
    "longctx": _TIER_LOW,
}


def _chat(
    endpoint: str, model: str, prompt: str, max_tokens: int,
    timeout: float = 300.0, reasoning_effort: str = "",
    sampling: dict | None = None,
    system_prompt: str = "",
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
) -> dict:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if sampling:
        if "temp" in sampling:
            body["temperature"] = sampling["temp"]
        if "top_k" in sampling:
            body["top_k"] = sampling["top_k"]
        if "top_p" in sampling:
            body["top_p"] = sampling["top_p"]
        if "min_p" in sampling:
            body["min_p"] = sampling["min_p"]
        if "repeat_penalty" in sampling:
            body["repeat_penalty"] = sampling["repeat_penalty"]
    # gpt-oss uses Harmony format — only supports reasoning_effort tier (low/medium/high),
    # ignores thinking_budget_tokens. Map per-category budget to effort tier.
    is_gpt_oss = "gpt-oss" in (model or "").lower()
    if is_gpt_oss:
        # gpt-oss has no off-switch; budget=0 falls back to "low" (~200 token overhead).
        if thinking_budget is None or thinking_budget <= _TIER_LOW:
            effort = "low"
        elif thinking_budget <= _TIER_MEDIUM:
            effort = "medium"
        else:
            effort = "high"
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
        body["reasoning_effort"] = effort
    else:
        if reasoning_effort and reasoning_effort != "auto":
            body["reasoning_effort"] = reasoning_effort
        if thinking_budget == 0:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        elif enable_thinking and thinking_budget is None:
            # auto mode: do not override, let server decide
            pass
        elif enable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": True}
            if thinking_budget > 0:
                body["thinking_budget_tokens"] = thinking_budget
        elif not reasoning_effort:
            body["chat_template_kwargs"] = {"enable_thinking": False}
    with httpx.Client(timeout=timeout) as client:
        for attempt in range(3):
            resp = client.post(f"{endpoint}/v1/chat/completions", json=body)
            if resp.status_code in (502, 503) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()


_PP512_PROMPT = (
    "You are a helpful assistant. Analyze the following passage and then write a brief continuation.\n\n"
    "The history of computing spans several centuries. Charles Babbage designed the Analytical Engine in "
    "the 1830s, which contained many features of modern computers including arithmetic logic, control flow, "
    "and memory. Ada Lovelace wrote what is considered the first computer program for this machine. "
    "In the 1930s and 1940s, Alan Turing formalized the concept of computation with his Turing machine, "
    "while Claude Shannon showed how Boolean algebra could be used to design digital circuits. "
    "The first electronic general-purpose computers like ENIAC (1945) filled entire rooms and used vacuum "
    "tubes. The invention of the transistor at Bell Labs in 1947 revolutionized electronics, leading to "
    "smaller and more reliable computers. The integrated circuit, developed independently by Jack Kilby "
    "and Robert Noyce in 1958-1959, allowed multiple transistors on a single chip. Gordon Moore observed "
    "in 1965 that the number of transistors on a chip doubled approximately every two years, a trend known "
    "as Moore's Law that has held remarkably well for decades. The development of microprocessors in the "
    "early 1970s, particularly the Intel 4004 and later the 8080, made personal computing possible. "
    "The Apple II (1977) and IBM PC (1981) brought computers into homes and offices. The rise of the "
    "internet in the 1990s connected these machines globally, transforming communication, commerce, and "
    "culture. Today, computing has expanded to include smartphones, cloud services, artificial intelligence, "
    "and quantum computing research, continuing the remarkable trajectory that began nearly two centuries ago.\n\n"
    "Continue this narrative with a focus on recent developments:"
)


def _run_speed_test(model_id: str) -> dict:
    endpoint = settings.swap_endpoint.rstrip("/")
    t0 = time.time()
    _chat(endpoint, model_id, "Say OK", 5)
    load_time = round(time.time() - t0)
    t_start = time.time()

    pp_speeds: list[float] = []
    tg_speeds: list[float] = []
    runs: list[dict] = []

    for i in range(3):
        data = _chat(endpoint, model_id, _PP512_PROMPT, 128)
        timings = data.get("timings", {})
        pp_n = timings.get("prompt_n", 0)
        tg_n = timings.get("predicted_n", 0)
        pp = timings.get("prompt_per_second", 0)
        tg = timings.get("predicted_per_second", 0)
        if pp > 0:
            pp_speeds.append(pp)
        if tg > 0:
            tg_speeds.append(tg)
        runs.append({"pp": round(pp, 1), "tg": round(tg, 1), "pp_n": pp_n, "tg_n": tg_n})

    return {
        "tg_speed": round(sum(tg_speeds) / len(tg_speeds), 1) if tg_speeds else 0,
        "pp_speed": round(sum(pp_speeds) / len(pp_speeds), 1) if pp_speeds else 0,
        "load_time": load_time,
        "duration": round(time.time() - t_start),
        "tested_at": _now(),
        "runs": runs,
    }


_CODE_BLOCK = '''def calculate_tax(income, rate):
    result = income * rate
    return result

def calculate_bonus(salary, months):
    result = salary * months * 0.1
    return result

def calculate_discount(price, percent):
    result = price * percent / 100
    return result

def calculate_interest(principal, rate, years):
    result = principal * rate * years
    return result

def calculate_shipping(weight, distance):
    result = weight * 0.5 + distance * 0.1
    return result

def calculate_overtime(hours, rate):
    result = hours * rate * 1.5
    return result'''

_JSON_SCHEMA = '''[
  {"id": 1, "name": "Alice Johnson", "email": "alice@example.com", "role": "engineer", "active": true},
  {"id": 2, "name": "Bob Smith", "email": "bob@example.com", "role": "designer", "active": true},
  {"id": 3, "name": "Carol White", "email": "carol@example.com", "role": "manager", "active": false},
  {"id": 4, "name": "David Brown", "email": "david@example.com", "role": "engineer", "active": true},
  {"id": 5, "name": "Eve Davis", "email": "eve@example.com", "role": "analyst", "active": true}
]'''


def _run_spec_test(model_id: str) -> dict:
    """Speed test with maximally repetitive prompts for ngram speculation.

    Ngram speculation predicts tokens from patterns in already-generated output.
    These prompts produce extremely repetitive output where the predictor
    should achieve high hit rates.
    """
    from . import registry as _reg
    endpoint = settings.swap_endpoint.rstrip("/")
    rec = _reg.get(model_id)
    effort = rec.reasoning_effort if rec else ""

    t0 = time.time()
    _chat(endpoint, model_id, "Say OK", 5)
    load_time = round(time.time() - t0)
    t_start = time.time()

    tests_def = [
        ("numbered_list", "Generate a numbered list from 1 to 100. Each line: 'N. A short random fact about number N.' No introduction, start immediately with '1.'", 3000),
        ("mult_table", "Print a full multiplication table from 1×1 to 15×15. Format each line as: 'A × B = C'. One result per line. No headers, start immediately.", 3000),
        ("json_array", f"Generate a JSON array with 40 entries following this exact schema. Only increment the id (6-45) and vary name/email/role. Return ONLY valid JSON:\n\n```json\n{_JSON_SCHEMA}\n```", 4000),
        ("getters", "Generate Python getter and setter methods for these 15 fields: name, email, age, address, phone, city, country, zipcode, company, title, salary, department, manager, start_date, active. Use @property decorator. No class definition, just the methods.", 4000),
        ("csv_100", "Generate a CSV with exactly 100 rows of fake employee data. Columns: id,first_name,last_name,email,department,salary,hire_date. Start with the header row, then rows 1-100. No explanation.", 4000),
    ]

    pp_speeds: list[float] = []
    tg_speeds: list[float] = []
    tests: dict = {}

    for name, prompt, max_tokens in tests_def:
        data = _chat(endpoint, model_id, prompt, max_tokens, timeout=600.0, reasoning_effort=effort)
        timings = data.get("timings", {})
        content = ""
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            pass
        pp = timings.get("prompt_per_second", 0)
        tg = timings.get("predicted_per_second", 0)
        if pp > 0:
            pp_speeds.append(pp)
        if tg > 0:
            tg_speeds.append(tg)
        tests[name] = {"pp": round(pp, 1), "tg": round(tg, 1), "response": content[:200]}

    return {
        "tg_speed": round(sum(tg_speeds) / len(tg_speeds), 1) if tg_speeds else 0,
        "pp_speed": round(sum(pp_speeds) / len(pp_speeds), 1) if pp_speeds else 0,
        "load_time": load_time,
        "duration": round(time.time() - t_start),
        "tested_at": _now(),
        "tests": tests,
    }


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_HARMONY_FINAL_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(
    r'(?:So (?:the )?(?:answer|response|reply|output) (?:is|would be|should be)[:\s]*["\']?)(.+?)["\']?\s*$',
    re.DOTALL | re.IGNORECASE,
)


def _strip_think_tags(text: str) -> str:
    """Strip thinking blocks, Harmony channel markers, and inline reasoning from model output."""
    # 1) Harmony format: extract <|channel|>final content
    m = _HARMONY_FINAL_RE.search(text)
    if m:
        return m.group(1).strip()

    # 2) <think>...</think> tags
    cleaned = _THINK_RE.sub("", text)
    if cleaned.strip():
        text = cleaned

    start = text.rfind("</think>")
    if start != -1:
        after = text[start + len("</think>"):].strip()
        if after:
            return after

    # 3) Inline reasoning with "So the answer is: X" pattern
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        if candidate:
            return candidate

    return text


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"[  -​  　]", " ", s)


_ANSWER_DELIMITERS = re.compile(
    r"(?:^|\n)\s*(?:(?:so |thus |therefore |hence |the answer(?: is)?|answer:|result:|output:)\s*)",
    re.IGNORECASE,
)


def _extract_answer(text: str) -> str:
    """Try to extract the core answer from a response that may contain reasoning."""
    parts = _ANSWER_DELIMITERS.split(text)
    if len(parts) > 1:
        candidate = parts[-1].strip()
        if candidate:
            return candidate
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) > 1:
        return lines[-1]
    # Single-line: try to extract after common inline delimiters
    for sep in [":", "→", "=>", "is "]:
        idx = text.rfind(sep)
        if idx != -1:
            tail = text[idx + len(sep):].strip().rstrip(".")
            if tail and len(tail) < len(text) * 0.6:
                return tail
    return text


_MCQ_RE = re.compile(
    r"(?:ANSWER|answer|Answer|the answer is)\s*[:=]?\s*\(?([A-Ja-j])\)?|"
    r"\*\*([A-J])\*\*|"
    r"^([A-J])\b|"
    r"\b([A-J])\)?\s*$",
    re.MULTILINE
)


def _score_one(match_type: str, expected: str, question: dict, text: str) -> bool:
    if match_type == "contains":
        return expected in text
    if match_type == "contains_ci":
        return expected.lower() in text.lower()
    if match_type == "exact_ci":
        cleaned = re.sub(r"[^\w]", "", text.lower())
        exp_cleaned = re.sub(r"[^\w]", "", expected.lower())
        return cleaned == exp_cleaned
    if match_type == "regex":
        pattern = question.get("regex", "")
        return bool(re.search(pattern, text, re.DOTALL))
    if match_type == "word_count":
        bounds = question.get("word_count", [1, 100])
        wc = len(text.split())
        return bounds[0] <= wc <= bounds[1]
    if match_type == "mcq":
        for m in _MCQ_RE.finditer(text):
            letter = (m.group(1) or m.group(2) or m.group(3) or m.group(4)).upper()
            if letter == expected.upper():
                return True
        return expected.upper() in text.upper().replace(" ", "")
    if match_type == "json_valid":
        cleaned = re.sub(r'```(?:json)?\s*', '', text).strip().rstrip('`')
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            search_from = 0
            while True:
                start = cleaned.find(start_char, search_from)
                if start == -1:
                    break
                depth = 0
                found_end = -1
                for i, c in enumerate(cleaned[start:], start):
                    if c == start_char:
                        depth += 1
                    elif c == end_char:
                        depth -= 1
                        if depth == 0:
                            found_end = i
                            break
                if found_end == -1:
                    break
                try:
                    parsed = json.loads(cleaned[start:found_end+1])
                    if not expected:
                        return True
                    if isinstance(parsed, dict):
                        keys = [k.strip().strip('"\'') for k in expected.split(",")]
                        if all(k in parsed for k in keys):
                            return True
                    else:
                        return True
                except json.JSONDecodeError:
                    pass
                search_from = found_end + 1
        return False
    if match_type == "contains_all":
        parts = [p.strip() for p in expected.split("|")]
        text_lower = text.lower()
        return all(p.lower() in text_lower for p in parts)
    if match_type == "lines_count":
        lines = [l for l in text.strip().splitlines() if l.strip()]
        expected_count = int(expected)
        return len(lines) == expected_count
    return expected.lower() in text.lower()


def _score_answer(question: dict, response: str) -> bool:
    match_type = question.get("match", "contains")
    expected = question.get("expected", "")
    text = _normalize_whitespace(response.strip())

    if _score_one(match_type, expected, question, text):
        return True

    extracted = _extract_answer(text)
    if extracted != text and _score_one(match_type, expected, question, extracted):
        return True

    return False


def JUDGE_MODEL() -> str:
    from .config import get_judge_model
    return get_judge_model()

_JUDGE_PROMPT = """You are a strict exam grader. Evaluate if the student's response correctly answers the question.

Question: {question}
Expected answer: {expected}
Student's response: {response}

Rules:
- The student may include reasoning, explanations, code formatting (```), or chain-of-thought. Ignore all that — focus ONLY on whether the core answer is correct.
- For code questions: the code must be functionally correct, even if style differs.
- For math: the final numerical answer must be correct.
- For translations: meaning must be preserved, exact wording not required.
- Be strict on factual correctness but lenient on formatting.

Reply with exactly one word: PASS or FAIL"""


def _llm_judge(endpoint: str, question: dict, response: str) -> bool:
    extracted = _extract_answer(response)
    judge_response = extracted[:500] if extracted != response else response[-800:]
    prompt = _JUDGE_PROMPT.format(
        question=question["prompt"],
        expected=question.get("expected", "see question"),
        response=judge_response,
    )
    try:
        data = _chat(endpoint, JUDGE_MODEL(), prompt, 10, timeout=60.0)
        msg = data["choices"][0]["message"]
        verdict = (msg.get("content") or "").strip().upper()
        return "PASS" in verdict
    except Exception:
        return False


def _get_model_sampling(rec) -> dict | None:
    if rec is None:
        return None
    s = {}
    if hasattr(rec, "sampling_temp") and rec.sampling_temp != 0.8:
        s["temp"] = rec.sampling_temp
    if hasattr(rec, "sampling_top_k") and rec.sampling_top_k != 40:
        s["top_k"] = rec.sampling_top_k
    if hasattr(rec, "sampling_top_p") and rec.sampling_top_p != 0.95:
        s["top_p"] = rec.sampling_top_p
    if hasattr(rec, "sampling_min_p") and rec.sampling_min_p != 0.05:
        s["min_p"] = rec.sampling_min_p
    if hasattr(rec, "sampling_repeat_penalty") and rec.sampling_repeat_penalty != 1.1:
        s["repeat_penalty"] = rec.sampling_repeat_penalty
    return s or None


def _detect_thinking_support(endpoint: str, model_id: str) -> bool | None:
    """Probe whether a model emits reasoning content by default.

    Sends a minimal chat request (no chat_template_kwargs to avoid triggering
    segfaults in chat templates that don't define `enable_thinking`) and checks
    if the response carries reasoning_content or <think>/Harmony markers.

    Returns True/False on a successful probe, or None on transport error
    (so the caller can leave the registry flag unset for re-probe next time).
    """
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": "2+2?"}],
        "max_tokens": 50,
        "temperature": 0,
    }
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(f"{endpoint}/v1/chat/completions", json=body)
            if resp.status_code != 200:
                return None
            msg = resp.json()["choices"][0]["message"]
            if (msg.get("reasoning_content") or "").strip():
                return True
            content = msg.get("content") or ""
            return "<think>" in content or "<|channel|>analysis" in content
    except Exception:
        return None


def _run_intelligence_test(model_id: str, job_id: str = "", only_failed: bool = False) -> dict:
    from . import registry as _reg
    endpoint = settings.swap_endpoint.rstrip("/")
    bench = _load_benchmark()
    all_questions = bench["questions"]

    rec = _reg.get(model_id)
    effort = rec.reasoning_effort if rec else ""
    sampling = _get_model_sampling(rec)
    sys_prompt = (rec.system_prompt if rec and rec.system_prompt else "")

    # Auto-detect thinking support when reasoning_format is unset.
    # gpt-oss is always thinking-capable; skip probe.
    if rec and (rec.reasoning_format or "") == "" and "gpt-oss" not in model_id.lower():
        detected = _detect_thinking_support(endpoint, model_id)
        if detected is None:
            raise SwapctlError(
                f"{model_id}: model failed to respond to probe — likely won't load. "
                f"Check llama-swap logs and fix the backend before retrying."
            )
        rec.reasoning_format = "deepseek" if detected else "off"
        _reg.upsert(rec)
        log.info("%s: auto-detected thinking=%s -> reasoning_format=%s",
                 model_id, detected, rec.reasoning_format)

    thinking = bool(rec and rec.reasoning_format and rec.reasoning_format != "off")

    prev_details = []
    failed_ids: set = set()
    if only_failed and rec:
        prev = (rec.test_results or {}).get("intelligence") or {}
        prev_details = prev.get("details") or []
        failed_ids = {d["id"] for d in prev_details if not d.get("passed")}
    if only_failed and failed_ids:
        questions = [q for q in all_questions if q["id"] in failed_ids]
    else:
        questions = all_questions
        only_failed = False

    _chat(endpoint, model_id, "Say OK", 5, enable_thinking=thinking)
    t_start = time.time()

    from .config import get_parallel_slots, get_iq_parallel
    total_q = len(questions)
    workers = max(1, min(get_parallel_slots(), total_q)) if get_iq_parallel() else 1

    def _answer_one(q: dict) -> tuple[dict, str, float, int]:
        max_tok = q.get("max_tokens", 100)
        budget = _THINKING_BUDGET_BY_CATEGORY.get(q.get("category", ""), 128) if thinking else None
        if thinking and budget and budget > 0:
            max_tok = max(max_tok + budget + 100, 512)
        q_sys_prompt = sys_prompt
        if budget == 0 and "<|think|>" in q_sys_prompt:
            q_sys_prompt = q_sys_prompt.replace("<|think|>", "").strip()
        content = ""
        tg = 0.0
        tokens = 0
        try:
            data = _chat(endpoint, model_id, q["prompt"], max_tok, reasoning_effort=effort,
                         sampling=sampling, system_prompt=q_sys_prompt, enable_thinking=thinking,
                         thinking_budget=budget)
            tg = (data.get("timings") or {}).get("predicted_per_second", 0) or 0
            tokens = (data.get("usage") or {}).get("completion_tokens", 0) or 0
            msg = data["choices"][0]["message"]
            content = _strip_think_tags(msg.get("content") or "")
            if not content.strip() and msg.get("reasoning_content"):
                content = _strip_think_tags(msg["reasoning_content"])
            finish = data["choices"][0].get("finish_reason", "")
            if (not content.strip() or finish == "length") and not _score_answer(q, content):
                retry_tok = max_tok * 2
                data2 = _chat(endpoint, model_id, q["prompt"], retry_tok, reasoning_effort=effort,
                              sampling=sampling, system_prompt=q_sys_prompt, enable_thinking=thinking,
                              thinking_budget=budget)
                msg2 = data2["choices"][0]["message"]
                content2 = _strip_think_tags(msg2.get("content") or "")
                if not content2.strip() and msg2.get("reasoning_content"):
                    content2 = _strip_think_tags(msg2["reasoning_content"])
                if content2.strip():
                    content = content2
                    tokens = (data2.get("usage") or {}).get("completion_tokens", 0) or tokens
        except Exception:
            pass
        return (q, content, tg, tokens)

    responses_by_idx: dict[int, tuple[dict, str, int]] = {}
    tg_speeds: list[float] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_answer_one, q): i for i, q in enumerate(questions)}
        for fut in as_completed(futures):
            if job_id and job_id in _cancelled:
                for f in futures:
                    f.cancel()
                raise SwapctlError("cancelled")
            idx = futures[fut]
            q, content, tg, tokens = fut.result()
            responses_by_idx[idx] = (q, content, tokens)
            if tg > 0:
                tg_speeds.append(tg)
            completed += 1
            if job_id:
                elapsed = time.time() - t_start
                avg_per_q = elapsed / completed if completed > 0 else 0
                eta = round(avg_per_q * (total_q - completed) / workers) if avg_per_q else 0
                avg_tg = round(sum(tg_speeds) / len(tg_speeds), 1) if tg_speeds else 0
                with _lock:
                    j = _jobs.get(job_id)
                    if j is not None:
                        j["progress"] = {
                            "step": "answering",
                            "current": completed,
                            "total": total_q,
                            "percent": round(completed / total_q * 100, 1),
                            "tg_speed": avg_tg,
                            "eta_sec": eta,
                        }

    responses: list[tuple[dict, str, int]] = [responses_by_idx[i] for i in range(total_q)]

    empty_count = sum(1 for _, c, _ in responses if not c.strip())
    if empty_count > len(responses) * 0.2:
        log.warning("%s: %d/%d responses empty — model may not be working", model_id, empty_count, len(responses))

    categories: dict[str, dict] = {}
    difficulties: dict[str, dict] = {}
    details: list[dict] = []
    failed_indices: list[int] = []

    for i, (q, content, tokens) in enumerate(responses):
        cat = q["category"]
        diff = q.get("difficulty", "medium")
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "tokens": 0}
        if diff not in difficulties:
            difficulties[diff] = {"total": 0, "passed": 0}

        passed = _score_answer(q, content)
        categories[cat]["total"] += 1
        categories[cat]["tokens"] += tokens
        difficulties[diff]["total"] += 1
        if passed:
            categories[cat]["passed"] += 1
            difficulties[diff]["passed"] += 1

        details.append({
            "id": q["id"],
            "category": cat,
            "difficulty": diff,
            "passed": passed,
            "judge": None,
            "tokens": tokens,
            "response": content[:300],
        })

        if not passed and content.strip():
            failed_indices.append(i)

    if failed_indices:
        log.info("LLM judge: reviewing %d failed answers for %s", len(failed_indices), model_id)
        try:
            _chat(endpoint, JUDGE_MODEL(), "Say OK", 5, timeout=120.0)
        except Exception:
            log.warning("could not load judge model %s", JUDGE_MODEL())
            failed_indices = []

        n_failed = len(failed_indices)
        for ji, i in enumerate(failed_indices):
            if job_id and job_id in _cancelled:
                raise SwapctlError("cancelled")
            if job_id:
                with _lock:
                    j = _jobs.get(job_id)
                    if j is not None:
                        j["progress"] = {"step": "judging", "current": ji, "total": n_failed, "percent": round(ji / n_failed * 100, 1)}
            q, content, _ = responses[i]
            verdict = _llm_judge(endpoint, q, content)
            details[i]["judge"] = "pass" if verdict else "fail"
            if verdict:
                details[i]["passed"] = True
                cat = q["category"]
                diff = q.get("difficulty", "medium")
                categories[cat]["passed"] += 1
                difficulties[diff]["passed"] += 1

    if only_failed and prev_details:
        # merge: keep previously-passing answers, override re-tested ones
        new_by_id = {d["id"]: d for d in details}
        merged_details = []
        categories = {}
        difficulties = {}
        for old in prev_details:
            qid = old["id"]
            d = new_by_id.get(qid, old)
            merged_details.append(d)
            cat = d["category"]
            diff = d.get("difficulty", "medium")
            categories.setdefault(cat, {"total": 0, "passed": 0, "tokens": 0})
            difficulties.setdefault(diff, {"total": 0, "passed": 0})
            categories[cat]["total"] += 1
            categories[cat]["tokens"] += d.get("tokens", 0) or 0
            difficulties[diff]["total"] += 1
            if d.get("passed"):
                categories[cat]["passed"] += 1
                difficulties[diff]["passed"] += 1
        details = merged_details

    total = sum(c["total"] for c in categories.values())
    total_passed = sum(c["passed"] for c in categories.values())
    total_tokens = sum(c.get("tokens", 0) for c in categories.values())
    score = round((total_passed / total) * 100, 1) if total > 0 else 0

    cat_scores = {}
    for cat, vals in categories.items():
        toks = vals.get("tokens", 0)
        cat_scores[cat] = {
            "score": round((vals["passed"] / vals["total"]) * 100, 1) if vals["total"] > 0 else 0,
            "passed": vals["passed"],
            "total": vals["total"],
            "tokens": toks,
            "avg_tokens": round(toks / vals["total"], 1) if vals["total"] > 0 else 0,
        }

    # The benchmark declares a per-category `weight` that nothing has ever read, so
    # every category counted equally in `score` — including knowledge and structured,
    # which every model in the fleet answers perfectly and which therefore contribute
    # no ranking information at all. Report a weighted score alongside the raw one so
    # saturated or use-case-irrelevant categories can be discounted without changing
    # what `score` has always meant.
    weights = {c: float((v or {}).get("weight", 1.0))
               for c, v in (bench.get("categories") or {}).items()}
    num = sum(cat_scores[c]["score"] * weights.get(c, 1.0) for c in cat_scores)
    den = sum(weights.get(c, 1.0) for c in cat_scores)
    weighted_score = round(num / den, 1) if den else score

    diff_scores = {}
    for diff, vals in difficulties.items():
        diff_scores[diff] = {
            "score": round((vals["passed"] / vals["total"]) * 100, 1) if vals["total"] > 0 else 0,
            "passed": vals["passed"],
            "total": vals["total"],
        }

    new_duration = round(time.time() - t_start)
    if only_failed and prev_details and rec:
        prev = (rec.test_results or {}).get("intelligence") or {}
        prev_duration = prev.get("duration")
        duration = prev_duration if prev_duration else new_duration
    else:
        duration = new_duration

    return {
        "score": score,
        "weighted_score": weighted_score,
        "category_weights": weights,
        "passed": total_passed,
        "total": total,
        "duration": duration,
        "total_tokens": total_tokens,
        "avg_tokens": round(total_tokens / total, 1) if total > 0 else 0,
        "tested_at": _now(),
        "categories": cat_scores,
        "difficulties": diff_scores,
        "details": details,
    }


def _append_history(model_id: str, test_type: str, result: dict, backend: str | None = None) -> None:
    from . import registry as _reg
    rec = _reg.get(model_id)
    settings_snapshot = {}
    if rec:
        settings_snapshot = {
            "reasoning_format": rec.reasoning_format,
            "reasoning_effort": rec.reasoning_effort,
            "system_prompt": (rec.system_prompt or "")[:80],
            "temp": rec.sampling_temp,
            "top_k": rec.sampling_top_k,
            "top_p": rec.sampling_top_p,
            "min_p": rec.sampling_min_p,
            "repeat_penalty": rec.sampling_repeat_penalty,
        }
    entry = {
        "ts": _now(),
        "model_id": model_id,
        "test_type": test_type,
        "backend": backend,
        "score": result.get("score"),
        "passed": result.get("passed"),
        "total": result.get("total"),
        "tg_speed": result.get("tg_speed"),
        "pp_speed": result.get("pp_speed"),
        "duration": result.get("duration"),
        "categories": {k: v.get("score") for k, v in (result.get("categories") or {}).items()},
        "settings": settings_snapshot,
    }
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        log.exception("failed to append test history")


def _update_registry(model_id: str, test_type: str, result: dict, backend: str | None = None) -> None:
    from . import registry

    rec = registry.get(model_id)
    if rec is None:
        return

    tr = rec.test_results or {}

    if test_type == "speed":
        key = f"speed_{backend}" if backend else "speed_vulkan"
        tr[key] = result
        best_tg = 0
        best_backend = ""
        for k in ("speed_vulkan", "speed_hip", "speed_rock"):
            s = tr.get(k, {})
            if s.get("tg_speed", 0) > best_tg:
                best_tg = s["tg_speed"]
                best_backend = k.replace("speed_", "")
        tr["best_speed"] = best_tg
        tr["best_backend"] = best_backend
    elif test_type == "speed_spec":
        tr["speed_spec"] = result
    elif test_type == "intelligence":
        tr["intelligence"] = result

    rec.test_results = tr
    registry.upsert(rec)
    _append_history(model_id, test_type, result, backend)


def _process_job(job: dict) -> None:
    model_id = job["model_id"]
    test_type = job["test_type"]
    backend = job.get("backend")

    with _lock:
        job["status"] = "running"
        job["started_at"] = _now()

    try:
        if test_type == "speed":
            if backend and backend != _current_backend():
                _switch_backend(backend)
            result = _run_speed_test(model_id)
            _update_registry(model_id, "speed", result, backend or _current_backend())
        elif test_type == "speed_spec":
            result = _run_spec_test(model_id)
            _update_registry(model_id, "speed_spec", result)
        elif test_type == "intelligence":
            result = _run_intelligence_test(model_id, job_id=job.get("job_id", ""), only_failed=job.get("only_failed", False))
            _update_registry(model_id, "intelligence", result)
        else:
            raise SwapctlError(f"unknown test type: {test_type}")

        with _lock:
            if job["job_id"] in _cancelled:
                _cancelled.discard(job["job_id"])
            else:
                job["status"] = "done"
                job["result"] = result
                job["finished_at"] = _now()

    except Exception as exc:
        log.exception("test failed for %s", model_id)
        with _lock:
            _cancelled.discard(job.get("job_id", ""))
            if job["status"] != "cancelled":
                job["status"] = "error"
                job["error"] = str(exc)
                job["finished_at"] = _now()


def _current_backend() -> str:
    from .config import get_llama_backend
    return get_llama_backend()


def _switch_backend(backend: str) -> None:
    from . import swap_config, system
    swap_config.switch_backend(backend)
    system.restart_swap()
    system.wait_swap_healthy(timeout_s=30.0)


def _worker_loop() -> None:
    while True:
        with _lock:
            if not _queue:
                return
            job = _queue.pop(0)

        _process_job(job)


def _ensure_worker() -> None:
    global _worker_thread
    with _lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
        _worker_thread.start()


def enqueue(
    model_id: str,
    test_type: str = "speed",
    backend: str | None = None,
    only_failed: bool = False,
) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "model_id": model_id,
        "test_type": test_type,
        "backend": backend,
        "only_failed": only_failed,
        "status": "queued",
        "queued_at": _now(),
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
    }
    with _lock:
        _jobs[job_id] = job
        _queue.append(job)
    _ensure_worker()
    return job_id


def enqueue_batch(
    model_ids: list[str],
    test_type: str = "speed",
    backend: str | None = None,
) -> str:
    batch_id = uuid.uuid4().hex[:12]
    job_ids = []
    for mid in model_ids:
        jid = enqueue(mid, test_type, backend)
        with _lock:
            _jobs[jid]["batch_id"] = batch_id
        job_ids.append(jid)

    batch = {
        "job_id": batch_id,
        "model_ids": model_ids,
        "test_type": test_type,
        "backend": backend,
        "status": "running",
        "queued_at": _now(),
        "child_jobs": job_ids,
    }
    with _lock:
        _jobs[batch_id] = batch
    return batch_id


def get_status(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        job = dict(job)
    if "child_jobs" in job:
        children = [get_status(jid) for jid in job["child_jobs"]]
        done = sum(1 for c in children if c and c.get("status") in ("done", "error"))
        total = len(children)
        job["progress"] = f"{done}/{total}"
        if done == total:
            job["status"] = "done"
        job["children"] = children
    return job


def list_jobs() -> list[dict]:
    with _lock:
        return [dict(j) for j in _jobs.values() if "batch_id" not in j]


def queue_length() -> int:
    with _lock:
        return len(_queue)


def cancel_job(job_id: str) -> bool:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return False
        if job.get("status") == "queued":
            if job in _queue:
                _queue.remove(job)
            job["status"] = "cancelled"
            job["finished_at"] = _now()
            return True
        if job.get("status") == "running":
            _cancelled.add(job_id)
            job["status"] = "cancelled"
            job["finished_at"] = _now()
            return True
        if "child_jobs" in job:
            for cid in job["child_jobs"]:
                child = _jobs.get(cid)
                if child and child.get("status") in ("queued", "running"):
                    if child in _queue:
                        _queue.remove(child)
                    if child.get("status") == "running":
                        _cancelled.add(cid)
                    child["status"] = "cancelled"
                    child["finished_at"] = _now()
            job["status"] = "cancelled"
            return True
    return False


def cancel_all() -> int:
    with _lock:
        count = 0
        for job in list(_queue):
            job["status"] = "cancelled"
            job["finished_at"] = _now()
            count += 1
        _queue.clear()
        for job in _jobs.values():
            if job.get("status") == "running":
                _cancelled.add(job["job_id"])
                job["status"] = "cancelled"
                job["finished_at"] = _now()
                count += 1
            elif "child_jobs" in job and job.get("status") not in ("done", "error", "cancelled"):
                job["status"] = "cancelled"
        return count


def clear_finished() -> int:
    with _lock:
        to_remove = [jid for jid, j in _jobs.items() if j.get("status") in ("done", "error", "cancelled")]
        for jid in to_remove:
            del _jobs[jid]
        return len(to_remove)
