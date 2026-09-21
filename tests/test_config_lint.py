"""Each test here is a bug that actually shipped and cost hours to find by hand.

The point of the lint pass is to turn those from "discover it on the node, much
later" into "told at install time", so the tests are written as the incidents.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.config_lint import (
    ERROR,
    INFO,
    WARN,
    Finding,
    format_findings,
    lint_config,
    worst_level,
)


def cfg(**models) -> dict:
    return {"models": {k: v for k, v in models.items()}}


def rules(findings: list[Finding]) -> set[str]:
    return {f.rule for f in findings}


def test_clean_config_has_no_findings():
    data = cfg(**{
        "ok-model": {
            "cmd": "/usr/local/bin/llama-server-vulkan --model /models/a/a.gguf --port 9100 "
                   "--ctx-size 65536 --parallel 1",
            "proxy": "http://127.0.0.1:9100",
        }
    })
    assert lint_config(data) == []
    assert worst_level([]) is None
    assert format_findings([]) == "config OK"


def test_port_proxy_mismatch_is_an_error():
    """deepseek-v4-flash-iq3xxs shipped --port 9125 with proxy :9121.

    llama-swap waited on a port nothing was listening on and killed the 98 GB
    load on timeout, twice, before anyone read the config.
    """
    data = cfg(**{
        "deepseek": {
            "cmd": "/usr/local/bin/llama-server-vulkan --model /m/d.gguf --port 9125 --ctx-size 32768",
            "proxy": "http://127.0.0.1:9121",
        }
    })
    found = lint_config(data)
    assert "port_proxy_mismatch" in rules(found)
    assert worst_level(found) == ERROR


def test_port_collision_names_the_other_model():
    data = cfg(**{
        "a": {"cmd": "/usr/local/bin/llama-server-vulkan --model /m/a.gguf --port 9125", "proxy": "http://127.0.0.1:9125"},
        "b": {"cmd": "/usr/local/bin/llama-server-vulkan --model /m/b.gguf --port 9125", "proxy": "http://127.0.0.1:9125"},
    })
    found = [f for f in lint_config(data) if f.rule == "port_collision"]
    assert len(found) == 2
    assert "b" in next(f.message for f in found if f.model_id == "a")


def test_parallel_splits_context_and_is_reported():
    """ornith ran -c 65536 --parallel 4 and served 16384-token requests.

    claude-local died with ContextWindowExceededError; nothing in the config or
    the UI said anything other than 65536.
    """
    data = cfg(**{
        "ornith": {
            "cmd": "/usr/local/bin/llama-server-vulkan --model /m/o.gguf --port 9102 --ctx-size 65536 --parallel 4",
            "proxy": "http://127.0.0.1:9102",
        }
    })
    found = [f for f in lint_config(data) if f.rule == "context_split"]
    assert len(found) == 1
    # 16384 clears any sane absolute floor and still broke a real client, so
    # the gap itself must be reported, not just an undersized slot.
    assert "16384" in found[0].message
    assert "65536" in found[0].message


def test_usable_slots_report_the_gap_without_escalating():
    data = cfg(**{
        "batch": {
            "cmd": "/usr/local/bin/llama-server-vulkan --model /m/b.gguf --port 9103 --ctx-size 131072 --parallel 8",
            "proxy": "http://127.0.0.1:9103",
        }
    })
    found = [f for f in lint_config(data) if f.rule == "context_split"]
    assert len(found) == 1 and found[0].level == INFO


def test_tiny_slots_escalate_to_warn():
    data = cfg(**{
        "sidecar": {
            "cmd": "/usr/local/bin/llama-server-vulkan --model /m/s.gguf --port 9109 --ctx-size 8192 --parallel 4",
            "proxy": "http://127.0.0.1:9109",
        }
    })
    found = [f for f in lint_config(data) if f.rule == "context_split"]
    assert len(found) == 1 and found[0].level == WARN


def test_removed_draft_flags_are_an_error():
    """--draft-max is fatal: llama-server exits before binding a port.

    The generator emitted it for every model installed with a draft, so such a
    model could never have started.
    """
    data = cfg(**{
        "m": {
            "cmd": "/usr/local/bin/llama-server-vulkan --model /m/m.gguf --port 9104 -md /m/d.gguf --draft-max 3",
            "proxy": "http://127.0.0.1:9104",
        }
    })
    found = lint_config(data)
    assert "dead_flag" in rules(found)
    assert worst_level(found) == ERROR


def test_draft_without_spec_type_is_flagged():
    data = cfg(**{
        "m": {
            "cmd": "/usr/local/bin/llama-server-vulkan --model /m/m.gguf --port 9105 -md /m/d.gguf -ngld 999",
            "proxy": "http://127.0.0.1:9105",
        }
    })
    assert "draft_without_spec_type" in rules(lint_config(data))


def test_wrapper_script_entries_are_not_flagged():
    """xtts-v2 runs /opt/xtts/start.sh with the port baked into the script.

    Flagging that as "cmd has no --port" is a false positive, and a linter that
    cries wolf on the live config gets ignored.
    """
    data = cfg(**{
        "xtts-v2": {"cmd": "/opt/xtts/start.sh", "proxy": "http://127.0.0.1:9300"},
    })
    assert lint_config(data) == []


def test_wrapper_still_counts_for_port_collisions():
    """It owns its port even though the port is not on the command line."""
    data = cfg(**{
        "xtts-v2": {"cmd": "/opt/xtts/start.sh", "proxy": "http://127.0.0.1:9300"},
        "other": {"cmd": "/usr/local/bin/llama-server-vulkan --model /m/o.gguf --port 9300",
                  "proxy": "http://127.0.0.1:9300"},
    })
    assert "port_collision" in rules(lint_config(data))


def test_missing_cmd_and_missing_port():
    data = cfg(**{"a": {"proxy": "http://127.0.0.1:9100"},
                  "b": {"cmd": "/usr/local/bin/llama-server-vulkan --model /m/b.gguf", "proxy": "http://127.0.0.1:9101"}})
    found = rules(lint_config(data))
    assert "no_cmd" in found and "no_port" in found


def test_path_checks_are_opt_in(tmp_path: Path):
    data = cfg(**{
        "m": {
            "cmd": f"/usr/local/bin/llama-server-nonexistent --model {tmp_path}/missing.gguf --port 9106",
            "proxy": "http://127.0.0.1:9106",
        }
    })
    assert "binary_missing" not in rules(lint_config(data))
    strict = rules(lint_config(data, check_paths=True))
    assert "binary_missing" in strict and "model_missing" in strict


def test_unused_inline_mtp_head_is_reported(tmp_path: Path, monkeypatch):
    """The Qwen3.8 distill carried blk.40.nextn.* and ran without speculation.

    It is invisible: the config looks complete and the model answers fine.
    """
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    from app import gguf_meta
    monkeypatch.setattr(gguf_meta, "read_tensor_names",
                        lambda p: ["blk.0.attn_q.weight", "blk.40.nextn.eh_proj.weight"])

    data = cfg(**{
        "distill": {
            "cmd": f"/usr/local/bin/llama-server-vulkan --model {gguf} --port 9107 --ctx-size 65536 --parallel 1",
            "proxy": "http://127.0.0.1:9107",
        }
    })
    found = [f for f in lint_config(data, check_gguf=True) if f.rule == "mtp_head_unused"]
    assert len(found) == 1
    # WARN, not INFO: nothing wires this automatically, so it is a to-do for a
    # human. As INFO it sat in a list of ten and nobody acted on it.
    assert found[0].level == WARN
    assert "backend" in found[0].message


def test_inline_mtp_head_not_reported_when_already_used(tmp_path: Path, monkeypatch):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    from app import gguf_meta
    monkeypatch.setattr(gguf_meta, "read_tensor_names",
                        lambda p: ["blk.40.nextn.eh_proj.weight"])

    data = cfg(**{
        "m": {
            "cmd": f"/usr/local/bin/llama-server-vulkan --model {gguf} --port 9108 --spec-type draft-mtp --spec-draft-n-max 2",
            "proxy": "http://127.0.0.1:9108",
        }
    })
    assert "mtp_head_unused" not in rules(lint_config(data, check_gguf=True))


def test_findings_sort_errors_first():
    out = format_findings([
        Finding(INFO, "c", "r3", "info msg"),
        Finding(ERROR, "a", "r1", "err msg"),
        Finding(WARN, "b", "r2", "warn msg"),
    ])
    assert out.splitlines()[0].startswith("[error]")
    assert out.splitlines()[-1].startswith("[info]")


# Each row is a real shape the live config contains or a regex hole that was
# mutation-proven unprotected. Table form so a future "simplification" of any
# single pattern fails here rather than on the node.
@pytest.mark.parametrize("label,cmd,proxy,expected", [
    ("equals-separated flags",
     "/usr/local/bin/llama-server-vulkan --model=/m/a.gguf --port=9100 "
     "--ctx-size=65536 --parallel=4",
     "http://127.0.0.1:9100", {"context_split"}),
    ("proxy with trailing slash still compared",
     "/usr/local/bin/llama-server-vulkan --model /m/a.gguf --port 9125",
     "http://127.0.0.1:9121/", {"port_proxy_mismatch"}),
    ("env-prefixed rocm invocation is a llama-server",
     "LD_LIBRARY_PATH=/opt/rocm/lib /usr/local/bin/llama-server-hip "
     "--model /m/a.gguf --port 9100 --draft-max 3",
     "http://127.0.0.1:9100", {"dead_flag"}),
    ("npu blocks take none of these flags",
     "flm serve some-tag --port 9400", "http://127.0.0.1:9400", set()),
    ("wrapper with a flag is still a wrapper",
     "/opt/xtts/start.sh --device cuda", "http://127.0.0.1:9300", set()),
    ("dead flag inside a path is not a dead flag",
     "/usr/local/bin/llama-server-vulkan --model /m/--draft-max/a.gguf --port 9100",
     "http://127.0.0.1:9100", set()),
    ("parallel with no ctx still splits",
     "/usr/local/bin/llama-server-vulkan --model /m/a.gguf --port 9100 --parallel 4",
     "http://127.0.0.1:9100", {"context_split"}),
    ("unrecognised binary is reported, not silently skipped",
     "/usr/local/bin/lama-server --model /m/a.gguf --port 9100 --draft-max 3",
     "http://127.0.0.1:9100", {"unrecognised_binary"}),
    ("current draft spelling is not mistaken for the removed one",
     "/usr/local/bin/llama-server-vulkan --model /m/a.gguf --port 9100 "
     "-md /m/d.gguf --spec-type draft-mtp --spec-draft-n-max 3",
     "http://127.0.0.1:9100", set()),
])
def test_rule_table(label, cmd, proxy, expected):
    got = rules(lint_config(cfg(**{"m": {"cmd": cmd, "proxy": proxy}})))
    assert got == expected, f"{label}: got {sorted(got)}, want {sorted(expected)}"


def test_known_non_llama_servers_stay_silent():
    """whisper and the NPU runner speak their own flags — no rules apply.

    They must be silent rather than reported as unrecognised, or the useful
    signal (a typo'd binary) drowns.
    """
    data = cfg(**{
        "whisper": {"cmd": "/opt/whisper.cpp/src/build/bin/whisper-server "
                           "--model /models/w/w.bin --port 9210",
                    "proxy": "http://127.0.0.1:9210"},
        "npu": {"cmd": "flm serve tag --port 9400", "proxy": "http://127.0.0.1:9400"},
    })
    assert lint_config(data) == []


def test_a_typo_in_a_llama_binary_is_still_reported():
    """The counterpart: suppression must not become a blanket."""
    data = cfg(**{
        "typo": {"cmd": "/usr/local/bin/lama-server --model /m/a.gguf --port 9100",
                 "proxy": "http://127.0.0.1:9100"},
    })
    assert "unrecognised_binary" in rules(lint_config(data))


def test_sidecar_in_use_is_silent(tmp_path: Path, monkeypatch):
    """A sidecar GGUF *is* the head; it exists so a target can point -md at it.

    Four of the six live hits for this rule were sidecars. A rule that is wrong
    two thirds of the time gets ignored along with its true positives.
    """
    gguf = tmp_path / "head.gguf"
    gguf.write_bytes(b"GGUF")
    from app import config_lint
    monkeypatch.setattr(config_lint, "read_tensor_names", lambda p: [
        "blk.48.nextn.eh_proj.weight", "blk.48.nextn.enorm.weight",
        "blk.48.nextn.hnorm.weight", "blk.48.attn_q.weight",
    ], raising=False)
    from app import gguf_meta
    monkeypatch.setattr(gguf_meta, "read_tensor_names", lambda p: [
        "blk.48.nextn.eh_proj.weight", "blk.48.nextn.enorm.weight",
        "blk.48.nextn.hnorm.weight", "blk.48.attn_q.weight",
    ])
    data = cfg(**{
        "sidecar": {
            "cmd": f"/usr/local/bin/llama-server-vulkan --model {gguf} --port 9114",
            "proxy": "http://127.0.0.1:9114",
        },
        "target": {
            "cmd": f"/usr/local/bin/llama-server-vulkan --model /m/t.gguf --port 9115 "
                   f"-md {gguf} --spec-type draft-mtp",
            "proxy": "http://127.0.0.1:9115",
        },
    })
    assert lint_config(data, check_gguf=True) == []


def test_sidecar_nobody_points_at_is_an_orphan(tmp_path: Path, monkeypatch):
    """step-3.7-flash-mtp-draft sits in the config with no -md referencing it.

    That is worth saying, but "switch to an MTP backend" is the wrong advice
    for a sidecar, so it gets its own finding.
    """
    gguf = tmp_path / "head.gguf"
    gguf.write_bytes(b"GGUF")
    from app import gguf_meta
    monkeypatch.setattr(gguf_meta, "read_tensor_names",
                        lambda p: ["blk.48.nextn.eh_proj.weight", "blk.48.attn_q.weight"])
    data = cfg(**{
        "sidecar": {
            "cmd": f"/usr/local/bin/llama-server-vulkan --model {gguf} --port 9114",
            "proxy": "http://127.0.0.1:9114",
        }
    })
    found = rules(lint_config(data, check_gguf=True))
    assert "orphan_draft" in found
    assert "mtp_head_unused" not in found


def test_real_model_with_a_head_is_still_reported(tmp_path: Path, monkeypatch):
    """The counterpart: a target carries many blocks, so it must still fire."""
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    from app import gguf_meta
    monkeypatch.setattr(gguf_meta, "read_tensor_names", lambda p: [
        f"blk.{i}.attn_q.weight" for i in range(40)
    ] + ["blk.40.nextn.eh_proj.weight"])
    data = cfg(**{
        "target": {
            "cmd": f"/usr/local/bin/llama-server-vulkan --model {gguf} --port 9115",
            "proxy": "http://127.0.0.1:9115",
        }
    })
    assert "mtp_head_unused" in rules(lint_config(data, check_gguf=True))
