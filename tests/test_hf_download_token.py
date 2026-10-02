"""The chunked downloader must send the saved HF token, or gated repos 401."""
from __future__ import annotations

from unittest import mock

from app import hf_client


class _Resp:
    status_code = 200
    headers = {"Content-Length": "4"}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield b"data"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def close(self):
        pass


def _run(tmp_path, token):
    seen = {}

    def fake_get(url, headers=None, **kw):
        seen["headers"] = dict(headers or {})
        return _Resp()

    with mock.patch.object(hf_client, "get_token", return_value=token), \
         mock.patch("requests.get", side_effect=fake_get), \
         mock.patch("app.config.get_dl_speed_limit", return_value=0):
        hf_client._download_throttled("org/repo", "model.gguf", tmp_path)
    return seen["headers"]


def test_saved_token_is_sent(tmp_path):
    assert _run(tmp_path, "hf_secret").get("Authorization") == "Bearer hf_secret"


def test_no_token_sends_no_header(tmp_path):
    assert "Authorization" not in _run(tmp_path, None)
