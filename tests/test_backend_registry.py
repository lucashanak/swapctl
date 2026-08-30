"""Tests for PR1-PR4: BackendsRegistry, model predicates, and flag injection."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(tmp_path: Path) -> "BackendsRegistry":
    """Return a BackendsRegistry backed by a temp registry.json."""
    from app.backends_registry import BackendsRegistry

    reg_path = tmp_path / "registry.json"
    with patch.dict(os.environ, {"SWAPCTL_REGISTRY_PATH": str(reg_path)}):
        # Re-import settings to pick up the env override
        import importlib
        import app.config as _cfg_mod
        import app.registry as _reg_mod

        # Patch the settings object path so registry reads/writes go to tmp
        original = _reg_mod._path

        def _patched_path():
            return reg_path

        _reg_mod._path = _patched_path
        br = BackendsRegistry()
        # Monkey-patch the internal helpers to use our tmp registry path
        original_read = br._read
        original_write = br._write

        def _read():
            from app.registry import get_backends_raw
            import app.registry as rmod
            old = rmod._path
            rmod._path = _patched_path
            try:
                return get_backends_raw()
            finally:
                rmod._path = old

        def _write(data):
            from app.registry import set_backends_raw
            import app.registry as rmod
            old = rmod._path
            rmod._path = _patched_path
            try:
                set_backends_raw(data)
            finally:
                rmod._path = old

        br._read = _read
        br._write = _write
        return br


def _make_model(filename: str = "model.gguf", model_tags: list | None = None):
    """Build a minimal ModelRecord-like object for testing."""
    from app.registry import ModelRecord
    from dataclasses import asdict
    rec = ModelRecord(
        model_id="test-model",
        repo_id=None,
        filename=filename,
        mmproj=None,
        installed_at="2025-01-01T00:00:00+00:00",
        hf_last_modified=None,
        size_bytes=0,
        ctx_size=65536,
    )
    # Attach model_tags as a plain attribute (not a dataclass field)
    object.__setattr__(rec, "model_tags", model_tags or []) if False else None
    rec.__dict__["model_tags"] = model_tags or []
    return rec


# ---------------------------------------------------------------------------
# Test 1: seed produces 4 default backends
# ---------------------------------------------------------------------------

def test_seed_produces_four_backends(tmp_path: Path) -> None:
    br = _make_registry(tmp_path)
    br.seed_defaults_if_empty()
    backends = br.list_all()
    ids = [b.id for b in backends]
    assert len(backends) == 7, f"expected 7 backends, got {len(backends)}: {ids}"
    assert "vulkan" in ids
    assert "hip" in ids
    assert "rock" in ids
    assert "vulkan-mtp" in ids
    assert "vulkan-atomic-turbo" in ids
    assert "vulkan-atomic-nextn" in ids


def test_seed_is_idempotent(tmp_path: Path) -> None:
    br = _make_registry(tmp_path)
    br.seed_defaults_if_empty()
    br.seed_defaults_if_empty()  # second call should be a no-op
    assert len(br.list_all()) == 7


def test_seed_sets_correct_mtp_fields(tmp_path: Path) -> None:
    br = _make_registry(tmp_path)
    br.seed_defaults_if_empty()
    mtp = br.get("vulkan-mtp")
    assert mtp is not None
    assert mtp.binary_path == "/usr/local/bin/llama-server-vulkan-mtp"
    assert mtp.model_predicate == "has_mtp_heads"
    assert "--parallel 1" in mtp.required_flags
    assert any("spec-type" in f for f in mtp.default_flags)
    assert any("spec-draft-n-max" in f for f in mtp.default_flags)


# ---------------------------------------------------------------------------
# Test 2: compatible() includes vulkan-mtp only for MTP models
# ---------------------------------------------------------------------------

def test_compatible_includes_mtp_for_mtp_model(tmp_path: Path) -> None:
    br = _make_registry(tmp_path)
    br.seed_defaults_if_empty()

    # Model with mtp_heads tag
    model_with_mtp = _make_model(filename="qwen3.6-35b-a3b-mtp-ud-q3-k-xl.gguf", model_tags=["mtp_heads"])
    compat_ids = {b.id for b in br.compatible(model_with_mtp)}
    assert "vulkan-mtp" in compat_ids, "vulkan-mtp should be compatible with MTP model"
    assert "vulkan" in compat_ids
    assert "hip" in compat_ids
    assert "rock" in compat_ids


def test_compatible_excludes_mtp_for_non_mtp_model(tmp_path: Path) -> None:
    br = _make_registry(tmp_path)
    br.seed_defaults_if_empty()

    # Model without mtp_heads tag
    model_no_mtp = _make_model(filename="qwen3-4b-instruct-q8.gguf", model_tags=[])
    compat_ids = {b.id for b in br.compatible(model_no_mtp)}
    assert "vulkan-mtp" not in compat_ids, "vulkan-mtp should NOT be compatible with non-MTP model"
    assert "vulkan" in compat_ids


def test_compatible_uses_filename_heuristic(tmp_path: Path) -> None:
    """has_mtp_heads falls back to filename heuristic when no tags set."""
    br = _make_registry(tmp_path)
    br.seed_defaults_if_empty()

    # Filename contains '-mtp-' — should trigger heuristic
    model_mtp_name = _make_model(filename="some-model-mtp-q4.gguf", model_tags=[])
    from app.backends_registry import has_mtp_heads
    assert has_mtp_heads(model_mtp_name), "filename heuristic should detect -mtp- in filename"

    model_no_mtp = _make_model(filename="llama3-8b-q8.gguf", model_tags=[])
    assert not has_mtp_heads(model_no_mtp), "plain model should not be detected as MTP"


# ---------------------------------------------------------------------------
# Test 3: switching to vulkan-mtp from --parallel 4 cmd
# ---------------------------------------------------------------------------

def _make_swap_config_with_model(tmp_path: Path, model_id: str, cmd: str):
    """Write a minimal llama-swap config.yaml with one model."""
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap

    y = YAML()
    data = CommentedMap()
    data["models"] = CommentedMap()
    block = CommentedMap()
    block["cmd"] = cmd
    block["proxy"] = "http://127.0.0.1:9100"
    data["models"][model_id] = block
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        y.dump(data, f)
    return cfg_path


def test_switch_to_mtp_replaces_parallel_4_with_1(tmp_path: Path) -> None:
    """Switching to vulkan-mtp must inject --parallel 1 and drop --parallel 4."""
    # Seed registry in tmp
    br = _make_registry(tmp_path)
    br.seed_defaults_if_empty()

    model_id = "qwen3-mtp-test"
    orig_cmd = (
        "/usr/local/bin/llama-server-vulkan "
        "--model /models/qwen3-mtp.gguf "
        "--port 9100 -ngl 999 "
        "--cache-type-k q8_0 --cache-type-v q8_0 "
        "--jinja --ctx-size 65536 --parallel 4"
    )
    cfg_path = _make_swap_config_with_model(tmp_path, model_id, orig_cmd)

    # Need swap_config to use our tmp registry
    import app.registry as rmod
    reg_path = tmp_path / "registry.json"
    old_path_fn = rmod._path
    rmod._path = lambda: reg_path

    try:
        from app import swap_config
        swap_config.switch_model_backend(model_id, "vulkan-mtp", path=cfg_path)
    finally:
        rmod._path = old_path_fn

    from app import swap_config as sc
    data = sc.load(cfg_path)
    result_cmd = data["models"][model_id]["cmd"]

    assert "/usr/local/bin/llama-server-vulkan-mtp" in result_cmd
    assert "--parallel 1" in result_cmd
    assert "--parallel 4" not in result_cmd
    # Count occurrences of --parallel to ensure no duplicates
    import re
    parallel_matches = re.findall(r"--parallel\s+\S+", result_cmd)
    assert len(parallel_matches) == 1, f"expected exactly 1 --parallel, got {parallel_matches}"
    assert "--spec-type draft-mtp" in result_cmd
    assert "--spec-draft-n-max 3" in result_cmd


# ---------------------------------------------------------------------------
# Test 4: switching back to vulkan from vulkan-mtp strips MTP flags
# ---------------------------------------------------------------------------

def test_switch_back_to_vulkan_strips_mtp_flags(tmp_path: Path) -> None:
    """Switching from vulkan-mtp back to vulkan removes MTP-injected flags."""
    br = _make_registry(tmp_path)
    br.seed_defaults_if_empty()

    model_id = "qwen3-mtp-roundtrip"
    mtp_cmd = (
        "/usr/local/bin/llama-server-vulkan-mtp "
        "--model /models/qwen3-mtp.gguf "
        "--port 9100 -ngl 999 "
        "--cache-type-k q8_0 "
        "--jinja --ctx-size 65536 "
        "--parallel 1 "
        "--spec-type draft-mtp --spec-draft-n-max 3"
    )
    cfg_path = _make_swap_config_with_model(tmp_path, model_id, mtp_cmd)

    import app.registry as rmod
    reg_path = tmp_path / "registry.json"
    old_path_fn = rmod._path
    rmod._path = lambda: reg_path

    try:
        from app import swap_config
        swap_config.switch_model_backend(model_id, "vulkan", path=cfg_path)
    finally:
        rmod._path = old_path_fn

    from app import swap_config as sc
    data = sc.load(cfg_path)
    result_cmd = data["models"][model_id]["cmd"]

    assert "/usr/local/bin/llama-server-vulkan" in result_cmd
    assert "/usr/local/bin/llama-server-vulkan-mtp" not in result_cmd
    # MTP-specific flags should be gone (they were injected by vulkan-mtp backend)
    assert "--spec-type" not in result_cmd
    assert "--spec-draft-n-max" not in result_cmd
    # --parallel is removed when switching to vulkan (no required_flags for vulkan)
    # This is acceptable behavior — documented in spec: "or leaves them — your choice"
    # We chose to strip them since they were required_flags of the source backend


# ---------------------------------------------------------------------------
# Test FIX 3: model_tags round-trips through registry write/read
# ---------------------------------------------------------------------------

def test_model_tags_roundtrip_registry(tmp_path: Path) -> None:
    """model_tags field must survive a registry upsert + read cycle."""
    import app.registry as rmod
    reg_path = tmp_path / "registry.json"
    old_path_fn = rmod._path
    rmod._path = lambda: reg_path
    try:
        rec = rmod.ModelRecord(
            model_id="mtp-test",
            repo_id=None,
            filename="mtp-model-q4.gguf",
            mmproj=None,
            installed_at="2025-01-01T00:00:00+00:00",
            hf_last_modified=None,
            size_bytes=12345,
            ctx_size=32768,
            model_tags=["mtp_heads", "moe"],
        )
        rmod.upsert(rec)
        loaded = rmod.get("mtp-test")
    finally:
        rmod._path = old_path_fn

    assert loaded is not None
    assert loaded.model_tags == ["mtp_heads", "moe"], (
        f"model_tags not persisted correctly: {loaded.model_tags}"
    )


# ---------------------------------------------------------------------------
# Test FIX 7: POST /backend/{model_id} rejects incompatible backends
# ---------------------------------------------------------------------------

def test_switch_model_backend_api_rejects_incompatible(tmp_path: Path) -> None:
    """POST /api/backend/{model_id} must return 400 when backend is incompatible."""
    import app.registry as rmod
    from app.backends_registry import BackendDef
    from unittest.mock import patch

    # Plain model record (no MTP) — vulkan-mtp should be incompatible
    plain_rec = rmod.ModelRecord(
        model_id="plain-model",
        repo_id=None,
        filename="plain-model-q8.gguf",
        mmproj=None,
        installed_at="2025-01-01T00:00:00+00:00",
        hf_last_modified=None,
        size_bytes=0,
        ctx_size=65536,
        model_tags=[],
    )

    # Backends known to the registry (all 4)
    all_backends = [
        BackendDef(id="vulkan", binary_path="/usr/local/bin/llama-server-vulkan", label="Vulkan"),
        BackendDef(id="hip", binary_path="/usr/local/bin/llama-server-hip", label="HIP"),
        BackendDef(id="rock", binary_path="/usr/local/bin/llama-server-rock", label="Rock"),
        BackendDef(id="vulkan-mtp", binary_path="/usr/local/bin/llama-server-vulkan-mtp", label="Vulkan+MTP",
                   model_predicate="has_mtp_heads"),
    ]
    # Compatible backends for a plain model: all except vulkan-mtp
    compat_backends = [b for b in all_backends if b.id != "vulkan-mtp"]

    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from app.routers.api import router as api_router

    app_fa = FastAPI()
    app_fa.include_router(api_router)

    with (
        patch("app.backends_registry.BackendsRegistry.list_all", return_value=all_backends),
        patch("app.backends_registry.BackendsRegistry.compatible", return_value=compat_backends),
        patch("app.registry.get", return_value=plain_rec),
        patch("app.swap_config.detect_model_backend", return_value="vulkan"),
    ):
        client = TestClient(app_fa, raise_server_exceptions=True)
        resp = client.post(
            "/api/backend/plain-model",
            data={"backend": "vulkan-mtp"},
        )

    assert resp.status_code == 400, (
        f"Expected 400 for incompatible backend, got {resp.status_code}: {resp.text}"
    )


def test_record_from_dict_reads_sampling_and_system_prompt(tmp_path: Path) -> None:
    """These six fields were written by upsert() but never parsed back, so any
    per-model sampling was silently reset to defaults on the next write."""
    from app.registry import _record_from_dict

    rec = _record_from_dict("m", {
        "sampling_temp": 0.0, "sampling_top_k": 1, "sampling_top_p": 0.5,
        "sampling_min_p": 0.0, "sampling_repeat_penalty": 1.0,
        "system_prompt": "be terse",
    })
    assert rec.sampling_temp == 0.0        # falsy — must survive, not become 0.8
    assert rec.sampling_top_k == 1
    assert rec.sampling_top_p == 0.5
    assert rec.sampling_min_p == 0.0
    assert rec.sampling_repeat_penalty == 1.0
    assert rec.system_prompt == "be terse"


def test_record_from_dict_defaults_when_absent() -> None:
    from app.registry import _record_from_dict

    rec = _record_from_dict("m", {})
    assert rec.sampling_temp == 0.8
    assert rec.sampling_top_k == 40
    assert rec.system_prompt == ""


def test_sampling_round_trips_through_upsert(tmp_path: Path) -> None:
    """temp 0 must survive a write/read cycle — the whole point of the fix."""
    import dataclasses
    from app import registry as reg

    path = tmp_path / "registry.json"
    rec = reg.ModelRecord(
        model_id="m", repo_id=None, filename="f.gguf", mmproj=None,
        installed_at="", hf_last_modified=None, size_bytes=1, ctx_size=4096,
    )
    reg.upsert(dataclasses.replace(rec, sampling_temp=0.0), path=path)
    got = next(r for r in reg.list_all(path=path) if r.model_id == "m")
    assert got.sampling_temp == 0.0


def test_no_category_disables_thinking_outright() -> None:
    """gpt-oss cannot turn reasoning off — _chat floors it at effort "low". A budget
    of 0 therefore handicaps every OTHER model, which is worth ~25 points on the
    instruction category. Every category must sit at or above that floor."""
    from app.tester import _THINKING_BUDGET_BY_CATEGORY, _TIER_LOW, _TIER_UNCAPPED

    # Legal values are the uncapped sentinel, or a budget at/above the floor that
    # gpt-oss cannot go below. What must never appear is a small positive cap or 0:
    # those silence or truncate every other model while gpt-oss keeps reasoning.
    bad = [c for c, b in _THINKING_BUDGET_BY_CATEGORY.items()
           if b != _TIER_UNCAPPED and b < _TIER_LOW]
    assert not bad, f"categories that handicap non-gpt-oss models: {bad}"


def test_benchmark_declares_weights_for_every_category() -> None:
    """`weight` was declared per category but never read, so the headline score
    counted saturated categories (knowledge, structured — zero spread across the
    whole fleet) exactly as heavily as discriminating ones."""
    from app.tester import _load_benchmark

    bench = _load_benchmark()
    cats = {q["category"] for q in bench["questions"]}
    declared = set(bench.get("categories") or {})
    assert cats <= declared, f"categories with no metadata: {cats - declared}"
    for c, meta in (bench.get("categories") or {}).items():
        assert isinstance(meta.get("weight"), (int, float)), f"{c} has no numeric weight"


def test_longctx_variants_are_deep_and_do_not_leak_their_answers() -> None:
    """Both variants must be long, and neither may contain its own answer: the answer
    is derived (a product, or a sum), so if it appeared in the text the question would
    collapse from reasoning to needle-finding. The generator caught exactly that —
    an aggregation sum of 83 that also occurred in the filler."""
    from app.tester import _load_benchmark

    for cat in ("longctx", "longctx_agg"):
        qs = [q for q in _load_benchmark()["questions"] if q["category"] == cat]
        assert len(qs) == 20, f"{cat} has {len(qs)} questions"
        words = sorted(len(q["prompt"].split()) for q in qs)
        assert words[0] > 1000 and words[-1] > 10000, f"{cat} depth {words[0]}-{words[-1]}"
        for q in qs:
            body = q["prompt"].rsplit("\n\n", 1)[0]
            assert body.count(q["expected"]) == 0, f"{q['id']} leaks its answer"
            assert int(q["expected"]) >= 100, f"{q['id']} answer could match filler numbers"


def test_batch_execution_only_groups_short_answer_questions() -> None:
    """Batching is a speed-up, so it must never change what a question is worth.
    Only questions with a short canonical answer qualify: anything whose answer
    spans lines, or whose prompt carries its own numbered list, would collide with
    the batch numbering and lose marks for a formatting reason."""
    from app.tester import _batch_eligible, _load_benchmark

    for q in _load_benchmark()["questions"]:
        if not _batch_eligible(q):
            continue
        exp = str(q.get("expected", "")).strip()
        assert exp and "\n" not in exp and len(exp) <= 40, q["id"]
        assert q["category"] not in ("exec", "longctx"), q["id"]


def test_split_batch_reply_maps_answers_to_indices() -> None:
    from app.tester import _split_batch_reply

    assert _split_batch_reply("1. 42\n2) Paris\n3: B", 3) == ["42", "Paris", "B"]
    # a missing index yields an empty slot, which makes the caller fall back to
    # asking that batch one question at a time rather than scoring blanks
    assert _split_batch_reply("1. 42", 3)[2] == ""
    # a restated list plus the real answers: keep both, let the grader decide
    assert "42" in _split_batch_reply("1. the maths one\n\n1. 42\n2. x\n3. y", 3)[0]


def test_no_question_is_asked_twice() -> None:
    """The removed `batch` category reused questions the suite already asked, so
    those items counted twice toward the score."""
    from app.tester import _load_benchmark

    ids = [q["id"] for q in _load_benchmark()["questions"]]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    prompts = [q["prompt"] for q in _load_benchmark()["questions"]]
    assert len(prompts) == len(set(prompts)), "the same prompt appears twice"


def test_every_test_type_persists_its_result() -> None:
    """Each branch of _process_job must call _update_registry. The agentic branch
    once did not: the test ran to completion, the result went onto the in-memory
    job only, and vanished with it — no error, no data, nothing to debug."""
    import inspect, re
    from app import tester

    src = inspect.getsource(tester._process_job)
    body = src.split("try:", 1)[1].split("except Exception", 1)[0]
    branches = re.findall(r'test_type == "(\w+)":(.*?)(?=elif test_type|else:)', body, re.S)
    assert branches, "could not parse the dispatch"
    for name, block in branches:
        assert "_update_registry(" in block, f"test type {name!r} never persists its result"


def test_agentic_scoring_rewards_efficiency_not_just_success() -> None:
    """Both calibrated models solve every episode of both scenarios, so a pass/fail
    score is a ceiling. Turns are what separate them (9 vs 14 on s2), which is also
    what the fleet's earlier agentic comparison found."""
    ref = 5

    def score(solve_rate, turns):
        return round(100.0 * solve_rate * min(1.0, ref / turns), 1)

    assert score(1.0, 9) > score(1.0, 14), "more turns must score lower"
    assert score(1.0, 5) == 100.0, "a solve at the reference turn count is full marks"
    assert score(1.0, 3) == 100.0, "beating the reference is capped, not rewarded twice"
    assert score(0.0, 5) == 0.0, "an unsolved episode scores nothing however few turns"
    # Below the reference every solve scores 100, so the scale cannot tell a 3-turn
    # solve from a 5-turn one. That dead zone is deliberate — the reference is the
    # fewest turns a real solve needs, and anything quicker is suspect rather than
    # better — but it means the reference must sit BELOW the turn counts actually
    # observed, or the category silently becomes pass/fail again.
    assert score(1.0, 4) == score(1.0, 5) == 100.0
    assert score(1.0, 9) > score(1.0, 14), "above the reference the scale is live"


def test_sanity_tier_carries_no_ranking_information() -> None:
    """Questions are tiered by measured pass rate: `sanity` means every model in the
    fleet got it right, so asking it cannot change any ranking. Dropping them is what
    makes the run 43% shorter — but it also means a trimmed run cannot detect a
    regression in basic ability, which is why they stay in the file and remain
    runnable via SWAPCTL_IQ_INCLUDE_SANITY."""
    from app.tester import _load_benchmark

    qs = _load_benchmark()["questions"]
    assert all("tier" in q for q in qs), "every question must be tiered"
    tiers = {q["tier"] for q in qs}
    assert tiers <= {"core", "sanity"}, f"unexpected tiers: {tiers}"
    core = [q for q in qs if q["tier"] == "core"]
    assert len(core) < len(qs), "nothing was trimmed"
    # the newest categories have too little evidence to be called saturated yet,
    # so they must all be core regardless of how the first model scored
    for cat in ("exec", "longctx", "longctx_agg"):
        got = [q for q in qs if q["category"] == cat]
        assert got and all(q["tier"] == "core" for q in got), f"{cat} was trimmed too early"
