"""Input validation and path-safety tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from app import ConfigError
from app.validation import (
    safe_child_dir,
    valid_ctx_size,
    valid_filename,
    valid_model_id,
    valid_repo_id,
)


def test_model_id_accepts_sane_ids():
    for name in ["qwen3.6-35b-a3b", "gpt-oss-120b", "gemma_4-26b", "abc", "a1"]:
        assert valid_model_id(name) == name


@pytest.mark.parametrize(
    "bad",
    ["", "..", "../etc", "foo/bar", "foo bar", "\n", "foo\nbar", "/abs", ".hidden", "a" * 65],
)
def test_model_id_rejects_unsafe(bad: str):
    with pytest.raises(ConfigError):
        valid_model_id(bad)


def test_repo_id_accepts_owner_repo():
    assert valid_repo_id("unsloth/gemma-4-26B-A4B-it-GGUF") == "unsloth/gemma-4-26B-A4B-it-GGUF"


@pytest.mark.parametrize(
    "bad",
    ["", "noslash", "a/b/c", "../bad", "owner/repo?evil", "owner /repo", " owner/repo"],
)
def test_repo_id_rejects_unsafe(bad: str):
    with pytest.raises(ConfigError):
        valid_repo_id(bad)


def test_filename_must_end_with_gguf():
    assert valid_filename("gemma-4-26B-A4B-it-UD-Q4_K_M.gguf").endswith(".gguf")


def test_filename_allows_single_subfolder():
    assert valid_filename("BF16/MiniMax-M2.7-BF16-00001-of-00010.gguf")
    assert valid_filename("MXFP4_MOE/MiniMax-M2.7-MXFP4_MOE-00001-of-00004.gguf")
    assert valid_filename("UD-Q4_K_XL/gpt-oss-120b-UD-Q4_K_XL-00001-of-00002.gguf")


@pytest.mark.parametrize(
    "bad",
    [
        "", "foo.bin", "../etc/passwd", "/etc/passwd", "a\nb.gguf",
        "../../x.gguf", ".hidden.gguf", "/BF16/foo.gguf",
        "a/b/c.gguf", "BF16/../evil.gguf", "BF16//inner.gguf",
    ],
)
def test_filename_rejects_unsafe(bad: str):
    with pytest.raises(ConfigError):
        valid_filename(bad)


def test_ctx_size_bounds():
    assert valid_ctx_size(65536) == 65536
    with pytest.raises(ConfigError):
        valid_ctx_size(0)
    with pytest.raises(ConfigError):
        valid_ctx_size(10_000_000)


def test_safe_child_dir_blocks_escape(tmp_path: Path):
    with pytest.raises(ConfigError):
        safe_child_dir(tmp_path, "..")
    with pytest.raises(ConfigError):
        safe_child_dir(tmp_path, "../sibling")
    resolved = safe_child_dir(tmp_path, "good_name")
    assert resolved.parent == tmp_path.resolve()


def test_safe_child_dir_blocks_equal_to_parent(tmp_path: Path):
    with pytest.raises(ConfigError):
        safe_child_dir(tmp_path, ".")
