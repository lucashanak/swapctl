"""A backend switch must patch the cmd, never rebuild it, for fork builds too.

switch_model_backend prefers patching the binary in place so hand-tuned flags
survive, and only rebuilds from the registry when it cannot find a binary to
splice. _BINARY_RE listed /usr/local/bin and the literal /opt/llama.cpp/build*
and nothing else, so for every fork build — /opt/llama.cpp-k2 (MoVA),
/opt/llama.cpp-lz (flash-next), /opt/strix-llamacpp-v070 (DeepSeek) — it took
the rebuild path.

What that cost, measured on deepseek-v4-flash-iq2m: the rebuilt cmd dropped
-md, -ngld, --spec-type draft-dspark and --spec-draft-n-max 4 (the DSpark
draft is worth ~13 -> 24 t/s), dropped -dev Vulkan0, and pointed --model at a
UD-IQ2_M/ subdirectory that does not exist, so the server exited in 251 ms.
Three of the fleet's models were affected, including the fleet leader.
"""
from __future__ import annotations

import pytest

from app.swap_config import _BINARY_RE, _cmd_binary, _has_llama_binary

FORKS = [
    "/opt/llama.cpp-k2/build/bin/llama-server",
    "/opt/llama.cpp-lz/build/bin/llama-server",
    "/opt/strix-llamacpp-v070/vulkan/bin/llama-server",
    "/opt/llama.cpp-b10360/build-vulkan/bin/llama-server",
    "/usr/local/bin/llama-server",
    "/usr/local/bin/llama-server-vulkan-mtp",
    "/usr/local/bin/llama-server-rocmfpx",
]


@pytest.mark.parametrize("binary", FORKS)
def test_every_build_shape_is_recognised(binary):
    cmd = f"{binary} --model /models/m/m.gguf --port 9100 -ngl 999"
    assert _has_llama_binary(cmd), f"{binary} would trigger a full rebuild"
    assert _cmd_binary(cmd) == binary


@pytest.mark.parametrize("binary", FORKS)
def test_the_binary_is_replaced_and_nothing_else_is(binary):
    """The tuned flags are the whole reason patching beats rebuilding."""
    tail = ("--model /models/ds/ds-00001-of-00003.gguf --port 9124 -dev Vulkan0 "
            "-ngl 999 -fa on -ctk q8_0 -md /models/dr/draft.gguf -ngld 999 "
            "--spec-type draft-dspark --spec-draft-n-max 4")
    cmd = f"{binary} {tail}"
    new, n = _BINARY_RE.subn("/usr/local/bin/llama-server-vulkan", cmd, count=1)
    assert n == 1
    assert new == f"/usr/local/bin/llama-server-vulkan {tail}"


def test_leading_env_assignment_does_not_hide_the_binary():
    cmd = "LD_LIBRARY_PATH=/opt/rocm/lib /usr/local/bin/llama-server --port 9100"
    assert _has_llama_binary(cmd)
    assert _cmd_binary(cmd) == "/usr/local/bin/llama-server"


@pytest.mark.parametrize("cmd", [
    "flm serve gpt-oss:20b --port 9206",
    "/opt/xtts/start.sh --port 9300",
    "whisper-server --model /models/w/ggml.bin --port 9400",
])
def test_non_llama_commands_are_still_not_spliceable(cmd):
    """Those must keep taking the rebuild path, not be patched blindly."""
    assert not _has_llama_binary(cmd)


def test_a_model_path_alone_is_not_a_binary():
    """Deciding on the first token stops a stray path standing in for it."""
    cmd = "/opt/xtts/start.sh --model /weird/llama-server-ish/model.gguf --port 9300"
    assert not _has_llama_binary(cmd)


def test_multiline_cmd_formatting_survives_the_splice():
    """The live config stores cmd as a block scalar with blank lines between
    tokens; a token-rejoin would silently reflow every model's cmd."""
    cmd = ("/opt/llama.cpp-lz/build/bin/llama-server\n\n"
           "--model /models/f/f.gguf\n\n--port 9130\n\n--spec-draft-p-min 0.75")
    new, n = _BINARY_RE.subn("/usr/local/bin/llama-server", cmd, count=1)
    assert n == 1
    assert new.count("\n\n") == cmd.count("\n\n")
    assert "--spec-draft-p-min 0.75" in new
