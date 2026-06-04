"""Tests for XTTS dispatch in the /api/v1/audio/speech endpoint."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from app.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _tts_path(path: str) -> str:
    from app.config import settings
    return f"{settings.root_path}/api/v1/audio/{path}"


# ── XTTS dispatch tests ───────────────────────────────────────────────


def test_xtts_voice_prefix_dispatches_to_xtts(client):
    """Voice starting with 'xtts:' must be forwarded to the XTTS service."""
    fake_audio = b"fake-xtts-audio-bytes"
    mock_resp = MagicMock()
    mock_resp.content = fake_audio
    mock_resp.headers = {"content-type": "audio/wav"}
    mock_resp.raise_for_status = MagicMock()

    with patch("app.routers.tts.httpx.post", return_value=mock_resp) as mock_post:
        resp = client.post(
            _tts_path("speech"),
            json={"voice": "xtts:default", "input": "Hello world"},
        )

    assert resp.status_code == 200
    assert resp.content == fake_audio
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
    assert payload["model"] == "xtts-v2"
    assert payload["voice"] == "default"
    assert payload["input"] == "Hello world"


def test_xtts_voice_prefix_strips_prefix(client):
    """'xtts:Ana Florence' must send voice='Ana Florence' to XTTS."""
    mock_resp = MagicMock()
    mock_resp.content = b"audio"
    mock_resp.headers = {"content-type": "audio/wav"}
    mock_resp.raise_for_status = MagicMock()

    with patch("app.routers.tts.httpx.post", return_value=mock_resp) as mock_post:
        client.post(
            _tts_path("speech"),
            json={"voice": "xtts:Ana Florence", "input": "Test"},
        )

    payload = mock_post.call_args[1]["json"]
    assert payload["voice"] == "Ana Florence"


def test_xtts_bare_prefix_uses_default(client):
    """'xtts:' with nothing after the colon must send voice='default'."""
    mock_resp = MagicMock()
    mock_resp.content = b"audio"
    mock_resp.headers = {"content-type": "audio/wav"}
    mock_resp.raise_for_status = MagicMock()

    with patch("app.routers.tts.httpx.post", return_value=mock_resp) as mock_post:
        client.post(
            _tts_path("speech"),
            json={"voice": "xtts:", "input": "Test"},
        )

    payload = mock_post.call_args[1]["json"]
    assert payload["voice"] == "default"


def test_xtts_language_passthrough(client):
    """language field must be forwarded to XTTS payload."""
    mock_resp = MagicMock()
    mock_resp.content = b"audio"
    mock_resp.headers = {"content-type": "audio/wav"}
    mock_resp.raise_for_status = MagicMock()

    with patch("app.routers.tts.httpx.post", return_value=mock_resp) as mock_post:
        client.post(
            _tts_path("speech"),
            json={"voice": "xtts:default", "input": "Ahoj", "language": "cs"},
        )

    payload = mock_post.call_args[1]["json"]
    assert payload["language"] == "cs"


def test_xtts_language_defaults_to_auto(client):
    """When language is omitted, XTTS payload must get language='auto'."""
    mock_resp = MagicMock()
    mock_resp.content = b"audio"
    mock_resp.headers = {"content-type": "audio/wav"}
    mock_resp.raise_for_status = MagicMock()

    with patch("app.routers.tts.httpx.post", return_value=mock_resp) as mock_post:
        client.post(
            _tts_path("speech"),
            json={"voice": "xtts:default", "input": "Hello"},
        )

    payload = mock_post.call_args[1]["json"]
    assert payload["language"] == "auto"


def test_xtts_timeout_returns_504(client):
    """XTTS timeout must return 504."""
    import httpx

    with patch("app.routers.tts.httpx.post", side_effect=httpx.TimeoutException("timeout")):
        resp = client.post(
            _tts_path("speech"),
            json={"voice": "xtts:default", "input": "Test"},
        )
    assert resp.status_code == 504


def test_xtts_upstream_error_returns_502(client):
    """XTTS HTTP error must return 502."""
    import httpx

    mock_req = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "internal error"
    exc = httpx.HTTPStatusError("error", request=mock_req, response=mock_response)

    with patch("app.routers.tts.httpx.post", side_effect=exc):
        resp = client.post(
            _tts_path("speech"),
            json={"voice": "xtts:default", "input": "Test"},
        )
    assert resp.status_code == 502


# ── Piper path unchanged ──────────────────────────────────────────────


def test_non_xtts_voice_uses_piper(client):
    """Voice without 'xtts:' prefix must not call httpx.post."""
    _fake_wav = b"RIFF\x00\x00\x00\x00WAVEfmt "
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _fake_wav

    with patch("app.tts.list_voices", return_value=["cs_CZ-jirka-medium"]), \
         patch("subprocess.run", return_value=mock_result), \
         patch("app.routers.tts.httpx.post") as mock_post:
        resp = client.post(
            _tts_path("speech"),
            json={"voice": "cs_CZ-jirka-medium", "input": "Ahoj"},
        )

    assert resp.status_code == 200
    mock_post.assert_not_called()


def test_piper_unknown_voice_returns_400(client):
    """Unknown non-xtts voice must return 400 (Piper path unchanged)."""
    with patch("app.tts.list_voices", return_value=[]):
        resp = client.post(
            _tts_path("speech"),
            json={"voice": "nonexistent", "input": "hi"},
        )
    assert resp.status_code == 400
