"""Smoke tests for Whisper STT integration."""
from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from app.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _stt_path() -> str:
    from app.config import settings
    return f"{settings.root_path}/api/v1/audio/transcriptions"


def _make_httpx_response(data: dict, status: int = 200):
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = data
    mock.text = str(data)
    return mock


def test_transcriptions_returns_text(client):
    """Endpoint proxies to whisper-server and returns JSON with text field."""
    fake_response = _make_httpx_response({"text": "And so my fellow Americans..."})

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = fake_response

    with patch("app.routers.stt.httpx.Client", return_value=mock_client):
        resp = client.post(
            _stt_path(),
            data={"model": "whisper-large-v3-turbo"},
            files={"file": ("audio.wav", BytesIO(b"RIFF"), "audio/wav")},
        )

    assert resp.status_code == 200
    assert resp.json()["text"] == "And so my fellow Americans..."


def test_transcriptions_propagates_upstream_error(client):
    """If whisper-server returns non-200, endpoint raises matching HTTP error."""
    fake_response = _make_httpx_response({"error": "model not loaded"}, status=503)

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = fake_response

    with patch("app.routers.stt.httpx.Client", return_value=mock_client):
        resp = client.post(
            _stt_path(),
            data={"model": "whisper-large-v3-turbo"},
            files={"file": ("audio.wav", BytesIO(b"RIFF"), "audio/wav")},
        )

    assert resp.status_code == 503


def test_transcriptions_handles_connection_error(client):
    """If whisper-server is unreachable, returns 502."""
    import httpx as _httpx

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.side_effect = _httpx.ConnectError("connection refused")

    with patch("app.routers.stt.httpx.Client", return_value=mock_client):
        resp = client.post(
            _stt_path(),
            data={"model": "whisper-large-v3-turbo"},
            files={"file": ("audio.wav", BytesIO(b"RIFF"), "audio/wav")},
        )

    assert resp.status_code == 502


def test_transcriptions_auth_bypass(client):
    """The transcriptions endpoint must be publicly accessible (no auth required)."""
    from app.main import _PUBLIC_EXACT
    from app.config import settings

    assert f"{settings.root_path}/api/v1/audio/transcriptions" in _PUBLIC_EXACT


def test_transcriptions_optional_params(client):
    """language and response_format are forwarded when provided."""
    fake_response = _make_httpx_response({"text": "hello"})

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = fake_response

    with patch("app.routers.stt.httpx.Client", return_value=mock_client):
        resp = client.post(
            _stt_path(),
            data={"model": "whisper-large-v3-turbo", "language": "en", "response_format": "json"},
            files={"file": ("audio.wav", BytesIO(b"RIFF"), "audio/wav")},
        )

    assert resp.status_code == 200
    call_kwargs = mock_client.post.call_args
    forwarded_data = call_kwargs.kwargs.get("data", call_kwargs.args[1] if len(call_kwargs.args) > 1 else {})
    assert forwarded_data.get("language") == "en"
    assert forwarded_data.get("response_format") == "json"
