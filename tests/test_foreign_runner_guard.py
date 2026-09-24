"""A model that is not run by llama-server has no backend to switch to.

halogen-flash-next is served by a container: its cmd is `podman run ...` and
its OpenAI endpoint is proxied like any other model. Without this guard a
backend switch — a benchmark arm, a UI click — would find no llama-server
binary to splice, fall through to the rebuild-from-registry path, and replace
the container invocation with a llama-server command built from a GGUF record
it does not have. The same applies to `flm serve` (NPU), whisper-server and
the xtts wrapper script.

The old `else: raise` only caught this when the registry record happened to
have no `filename`; one filled-in field and the rebuild ran.
"""
from __future__ import annotations

import pytest

from app.swap_config import _cmd_binary, _is_foreign_runner

HALOGEN = ("podman run --name halogen --rm -p 9135:8731 --device /dev/kfd "
           "-v /models/halogen-models:/models:ro "
           "ghcr.io/peonist-ai/halogen-flash-server:0.13.8")


@pytest.mark.parametrize("cmd", [
    HALOGEN,
    "docker run --rm -p 9135:8731 ghcr.io/x/y:1",
    "flm serve gpt-oss:20b --port 9206",
    "whisper-server --model /models/w/ggml.bin --port 9400",
    "/opt/xtts/start.sh --port 9300",
])
def test_foreign_runners_are_recognised(cmd):
    assert _is_foreign_runner(cmd)


@pytest.mark.parametrize("cmd", [
    "/usr/local/bin/llama-server --model /m/a.gguf --port 9100",
    "/opt/llama.cpp-k2/build/bin/llama-server --model /m/k2.gguf --port 9103",
    "LD_LIBRARY_PATH=/opt/rocm/lib /usr/local/bin/llama-server-hip --port 9101",
])
def test_llama_server_is_not_foreign(cmd):
    assert not _is_foreign_runner(cmd)


def test_env_prefix_does_not_hide_the_runner():
    assert _is_foreign_runner("HALOGEN_CTX=65536 podman run --rm x")
    assert _cmd_binary("HALOGEN_CTX=65536 podman run --rm x") == "podman"


def test_empty_cmd_is_not_foreign():
    """Nothing to refuse; the caller's own no-cmd path handles it."""
    assert not _is_foreign_runner("")


def test_switch_refuses_and_leaves_the_cmd_alone(tmp_path, monkeypatch):
    """The whole point: the entry must survive a backend switch attempt."""
    from app import swap_config

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n"
        "  halogen-flash-next:\n"
        f"    cmd: {HALOGEN}\n"
        "    proxy: http://127.0.0.1:9135\n"
    )
    before = swap_config.load(cfg)["models"]["halogen-flash-next"]["cmd"]

    with pytest.raises(swap_config.ConfigError) as e:
        swap_config.switch_model_backend("halogen-flash-next", "vulkan", path=cfg)
    assert "podman" in str(e.value)

    after = swap_config.load(cfg)["models"]["halogen-flash-next"]["cmd"]
    assert after == before, "the container invocation was rewritten"


def test_survives_even_when_a_registry_record_could_rebuild_it(tmp_path, monkeypatch):
    """The dangerous case the old `else: raise` did not cover.

    With a registry record carrying a `filename`, the rebuild branch is
    reachable: it would build a llama-server command from that GGUF and write
    it over the container invocation. The guard has to fire before that.
    """
    from app import swap_config, registry

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n"
        "  halogen-flash-next:\n"
        f"    cmd: {HALOGEN}\n"
        "    proxy: http://127.0.0.1:9135\n"
    )

    class Rec:
        model_id = "halogen-flash-next"
        filename = "some-model.gguf"       # this is what makes the rebuild reachable
        mmproj = ""
        draft_model = ""
        ctx_size = 65536
        cache_type_k = "q8_0"
        cache_type_v = "q8_0"
        threads = 0
        flash_attn = True
        extra_args = ""
        reasoning_format = ""
        draft_max = 0
        draft_min = 0
        spec_default = False
        sampling_temp = 0.8
        sampling_top_k = 40
        sampling_top_p = 0.95
        sampling_min_p = 0.05
        sampling_repeat_penalty = 1.1
        binary_override = ""
        parallel = 1

    monkeypatch.setattr(registry, "get", lambda mid, path=None: Rec())

    before = swap_config.load(cfg)["models"]["halogen-flash-next"]["cmd"]
    with pytest.raises(swap_config.ConfigError) as e:
        swap_config.switch_model_backend("halogen-flash-next", "vulkan", path=cfg)
    assert "podman" in str(e.value)

    after = swap_config.load(cfg)["models"]["halogen-flash-next"]["cmd"]
    assert after == before
    assert "llama-server" not in after
