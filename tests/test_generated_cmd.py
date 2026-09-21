"""What add_model() emits for a brand-new model.

Three defects lived in the generator for a long time because nobody asserted on
its output, and every new model was repaired by hand instead:

1. `--parallel` came from a global slot count, which splits --ctx-size into
   ctx/N per request. Silent: the config still reads 65536.
2. `--draft-max` / `--draft-min` were removed upstream; llama-server exits with
   "the argument has been removed" before binding a port, so any model added
   with a draft could never start.
3. A GGUF shipping its own MTP head (blk.*.nextn.*) was never recognised as
   one, because gguf_meta only looked for `mtp.*` names. The backends
   registry therefore never steered it to an MTP-capable backend. That is
   fixed in gguf_meta.has_mtp_tensors, not here — _build_cmd must not
   second-guess the registry, which owns the spec flags and their tuning.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import swap_config


SEED_YAML = """\
models:
  existing:
    cmd: /usr/local/bin/llama-server --model /models/existing/m.gguf --port 9100
    proxy: http://127.0.0.1:9100
"""


@pytest.fixture
def cfg_path(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(SEED_YAML)
    return p


def add(cfg_path: Path, **kw) -> str:
    defaults = dict(
        model_id="new-model",
        gguf_path=Path("/models/new-model/m.gguf"),
        mmproj_path=None,
        port=9200,
        ctx_size=65536,
        no_mmap=False,
        path=cfg_path,
    )
    defaults.update(kw)
    model_id = defaults["model_id"]
    swap_config.add_model(**defaults)
    return swap_config.load(cfg_path)["models"][model_id]["cmd"]


def test_parallel_defaults_to_one_regardless_of_the_global(cfg_path: Path, monkeypatch):
    """The global slot count must not reach a freshly generated model.

    It used to: every new model inherited --parallel 4 and therefore a quarter
    of its declared context, which is how claude-local hit
    ContextWindowExceededError at n_ctx 16384 on a model configured for 65536.
    """
    monkeypatch.setattr(swap_config, "get_parallel_slots", lambda: 8)
    cmd = add(cfg_path)
    assert "--parallel 1" in cmd
    assert "--parallel 8" not in cmd


def test_explicit_parallel_is_still_honoured(cfg_path: Path):
    """Concurrency stays available — it just has to be asked for."""
    cmd = add(cfg_path, parallel=8)
    assert "--parallel 8" in cmd


def test_draft_flags_use_the_current_spelling(cfg_path: Path):
    cmd = add(cfg_path, draft_model="/models/new-model/draft.gguf",
              draft_max=3, draft_min=1)
    assert "--spec-draft-n-max 3" in cmd
    assert "--spec-draft-n-min 1" in cmd
    # the removed spellings are fatal at startup, so they must not appear
    assert "--draft-max" not in cmd
    assert "--draft-min" not in cmd


def test_generated_cmd_passes_its_own_lint(cfg_path: Path):
    """The generator must not produce a config its own validator rejects."""
    from app.config_lint import lint_config, worst_level
    add(cfg_path)
    findings = lint_config(swap_config.load(cfg_path))
    mine = [f for f in findings if f.model_id == "new-model"]
    assert worst_level(mine) is None, [str(f) for f in mine]


def test_global_slot_change_reports_what_it_did(cfg_path: Path, caplog):
    """The second writer of --parallel rewrites already-tuned models.

    Raising the global used to cut every unpinned model's usable context with
    no record anywhere. It is still allowed — it just has to say so.
    """
    import logging
    add(cfg_path)
    with caplog.at_level(logging.WARNING, logger="swapctl.swap_config"):
        changed = swap_config.update_parallel_slots(4, path=cfg_path)
    assert changed >= 1
    split = [r for r in caplog.records if "parallel slots ->" in r.getMessage()]
    assert split, "a global slot change must record the context it just took away"
    msgs = " ".join(r.getMessage() for r in split)
    assert "16384" in msgs, msgs          # 65536 / 4, the context actually lost
    assert "new-model" in msgs, msgs      # and which model lost it


def test_lowering_slots_to_one_needs_no_warning(cfg_path: Path, caplog):
    import logging
    add(cfg_path)
    with caplog.at_level(logging.WARNING, logger="swapctl.swap_config"):
        swap_config.update_parallel_slots(1, path=cfg_path)
    assert not [r for r in caplog.records if "parallel slots ->" in r.getMessage()]


def test_slot_change_does_not_blame_models_it_left_alone(cfg_path: Path, caplog, monkeypatch):
    """Models pinned in the registry are skipped, so they must not be reported.

    Warning about every split in the file — including ones this call did not
    cause — is how a warning becomes wallpaper on a 40-model fleet.
    """
    import logging
    from types import SimpleNamespace
    from app import registry

    add(cfg_path, model_id="pinned", port=9201)
    add(cfg_path, model_id="free", port=9202)
    monkeypatch.setattr(
        registry, "get",
        lambda mid: SimpleNamespace(parallel=4) if mid == "pinned" else None)

    with caplog.at_level(logging.WARNING, logger="swapctl.swap_config"):
        swap_config.update_parallel_slots(4, path=cfg_path)
    msgs = " ".join(r.getMessage() for r in caplog.records
                    if "parallel slots ->" in r.getMessage())
    assert "free" in msgs
    assert "pinned" not in msgs
