"""Smoke tests for Piper TTS integration."""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── unit tests for tts module ─────────────────────────────────────────


def test_synthesize_raises_for_unknown_voice():
    with patch("app.tts.list_voices", return_value=["en_US-ryan-medium"]):
        from app import tts
        with pytest.raises(ValueError, match="unknown voice"):
            tts.synthesize("hello", "nonexistent-voice")


def test_synthesize_returns_wav_bytes():
    _fake_wav = b"RIFF\x00\x00\x00\x00WAVEfmt "
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _fake_wav

    with patch("app.tts.list_voices", return_value=["fake-voice"]), \
         patch("subprocess.run", return_value=mock_result):
        from app import tts
        result = tts.synthesize("hello", "fake-voice")
    assert result == _fake_wav


def test_synthesize_raises_on_nonzero_exit():
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = b""
    mock_result.stderr = b"model not found"

    with patch("app.tts.list_voices", return_value=["fake-voice"]), \
         patch("subprocess.run", return_value=mock_result):
        from app import tts
        with pytest.raises(RuntimeError, match="piper exited 1"):
            tts.synthesize("hello", "fake-voice")


def test_synthesize_caches_voice_model():
    """Second synthesize() call for same voice must not reload the model."""
    import io
    import wave
    from app import tts

    # Build a minimal valid WAV in-memory so wave.open() inside synthesize succeeds
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * 100)
    wav_bytes = wav_buf.getvalue()

    mock_voice = MagicMock()

    def _fake_synthesize(text, wav_writer):
        # Write valid WAV frames into the provided wav_writer
        wav_writer.setnchannels(1)
        wav_writer.setsampwidth(2)
        wav_writer.setframerate(22050)
        wav_writer.writeframes(b"\x00\x00" * 100)

    mock_voice.synthesize.side_effect = _fake_synthesize

    with patch("app.tts.list_voices", return_value=["fake-voice"]), \
         patch("app.tts._PIPER_PKG_AVAILABLE", True), \
         patch("app.tts._PiperVoice") as mock_cls, \
         patch.dict("app.tts._LOADED", {}, clear=True):
        mock_cls.load.return_value = mock_voice
        tts.synthesize("hello", "fake-voice")
        tts.synthesize("world", "fake-voice")
        # PiperVoice.load must be called exactly once (cache hit on second call)
        mock_cls.load.assert_called_once()


# ── endpoint tests ────────────────────────────────────────────────────


@pytest.fixture()
def client():
    from app.main import app
    # Bypass session auth for tests
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _tts_path(path: str) -> str:
    from app.config import settings
    return f"{settings.root_path}/api/v1/audio/{path}"


def test_voices_endpoint(client):
    with patch("app.tts.list_voices", return_value=["cs_CZ-jirka-medium"]):
        resp = client.get(_tts_path("voices"))
    assert resp.status_code == 200
    assert resp.json() == {"voices": ["cs_CZ-jirka-medium"]}


def test_speech_endpoint_returns_wav(client):
    _fake_wav = b"RIFF\x00\x00\x00\x00WAVEfmt "
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _fake_wav

    with patch("app.tts.list_voices", return_value=["cs_CZ-jirka-medium"]), \
         patch("subprocess.run", return_value=mock_result):
        resp = client.post(
            _tts_path("speech"),
            json={"model": "tts-1", "voice": "cs_CZ-jirka-medium", "input": "Ahoj"},
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/wav")
    assert resp.content == _fake_wav


def test_speech_endpoint_unknown_voice(client):
    with patch("app.tts.list_voices", return_value=[]):
        resp = client.post(
            _tts_path("speech"),
            json={"model": "tts-1", "voice": "bogus", "input": "hi"},
        )
    assert resp.status_code == 400


def test_speech_endpoint_unknown_format(client):
    """Completely unknown format should return 400."""
    _fake_wav = b"RIFF\x00\x00\x00\x00WAVEfmt "
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _fake_wav

    with patch("app.tts.list_voices", return_value=["cs_CZ-jirka-medium"]), \
         patch("subprocess.run", return_value=mock_result):
        resp = client.post(
            _tts_path("speech"),
            json={"model": "tts-1", "voice": "cs_CZ-jirka-medium", "input": "hi", "response_format": "xyz123"},
        )
    assert resp.status_code == 400


# ── _encode_wav unit tests ────────────────────────────────────────────


def test_encode_wav_passthrough():
    """wav format must not invoke ffmpeg."""
    from app.tts import _encode_wav

    wav = b"RIFF\x00\x00\x00\x00WAVEfmt "
    with patch("subprocess.run") as mock_run:
        result, mime = _encode_wav(wav, "wav")
    mock_run.assert_not_called()
    assert result == wav
    assert mime == "audio/wav"


def test_encode_wav_unknown_format():
    from app.tts import _encode_wav

    with pytest.raises(ValueError, match="unsupported"):
        _encode_wav(b"data", "xyz")


@pytest.mark.parametrize("fmt,expected_mime,expected_ffmpeg_flag", [
    ("mp3",  "audio/mpeg",  "libmp3lame"),
    ("opus", "audio/opus",  "libopus"),
    ("aac",  "audio/aac",   "aac"),
    ("flac", "audio/flac",  "flac"),
    ("ogg",  "audio/ogg",   "libvorbis"),
    ("pcm",  "audio/L16; rate=22050; channels=1", "pcm_s16le"),
])
def test_encode_wav_calls_ffmpeg(fmt, expected_mime, expected_ffmpeg_flag):
    """Each non-wav format must call ffmpeg with the correct codec flag."""
    from app.tts import _encode_wav

    fake_output = b"encoded-audio-data"
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = fake_output

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result, mime = _encode_wav(b"RIFF\x00\x00\x00\x00WAVEfmt ", fmt)

    assert result == fake_output
    assert mime == expected_mime
    # Verify the correct codec was in the ffmpeg command args
    args_passed = mock_run.call_args[0][0]
    assert expected_ffmpeg_flag in args_passed, f"Expected {expected_ffmpeg_flag} in {args_passed}"


def test_encode_wav_ffmpeg_failure():
    from app.tts import _encode_wav

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = b""
    mock_result.stderr = b"some error"

    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(RuntimeError, match="ffmpeg exited 1"):
            _encode_wav(b"wav-data", "mp3")


def test_speech_endpoint_mp3(client):
    """POST /speech with response_format=mp3 should return audio/mpeg."""
    _fake_wav = b"RIFF\x00\x00\x00\x00WAVEfmt "
    _fake_mp3 = b"\xff\xfb" + b"\x00" * 100  # fake mp3 frame header

    piper_result = MagicMock()
    piper_result.returncode = 0
    piper_result.stdout = _fake_wav

    ffmpeg_result = MagicMock()
    ffmpeg_result.returncode = 0
    ffmpeg_result.stdout = _fake_mp3

    def _fake_subprocess_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg":
            return ffmpeg_result
        return piper_result

    with patch("app.tts.list_voices", return_value=["cs_CZ-jirka-medium"]), \
         patch("subprocess.run", side_effect=_fake_subprocess_run):
        resp = client.post(
            _tts_path("speech"),
            json={"model": "tts-1", "voice": "cs_CZ-jirka-medium", "input": "hi", "response_format": "mp3"},
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/mpeg")


# ── Phase 2: catalog, install, remove ────────────────────────────────

_FAKE_RAW = {
    "cs_CZ-jirka-medium": {
        "key": "cs_CZ-jirka-medium",
        "name": "jirka",
        "language": {
            "code": "cs_CZ",
            "family": "cs",
            "region": "CZ",
            "name_native": "čeština",
            "name_english": "Czech",
            "country_english": "Czech Republic",
        },
        "quality": "medium",
        "num_speakers": 1,
        "speaker_id_map": {},
        "files": {
            "cs/cs_CZ/jirka/medium/cs_CZ-jirka-medium.onnx": {"size_bytes": 60000000, "md5_digest": "abc"},
            "cs/cs_CZ/jirka/medium/cs_CZ-jirka-medium.onnx.json": {"size_bytes": 5000, "md5_digest": "def"},
        },
        "aliases": [],
    }
}


def _make_httpx_response(data: dict):
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = data
    return mock


def test_fetch_catalog_from_cache(tmp_path, monkeypatch):
    import json
    import time
    from app import piper_catalog

    cache_file = tmp_path / "piper_catalog.json"
    flat = piper_catalog._flatten(_FAKE_RAW)
    cache_file.write_text(json.dumps({"catalog": flat, "fetched_at": time.time()}))
    monkeypatch.setattr(piper_catalog, "_CACHE_PATH", cache_file)
    monkeypatch.setattr(piper_catalog, "_VOICES_DIR", tmp_path / "voices")

    with patch("httpx.get") as mock_get:
        result = piper_catalog.fetch_catalog()

    mock_get.assert_not_called()
    assert any(v["key"] == "cs_CZ-jirka-medium" for v in result)


def test_fetch_catalog_network_fetch(tmp_path, monkeypatch):
    from app import piper_catalog

    monkeypatch.setattr(piper_catalog, "_CACHE_PATH", tmp_path / "piper_catalog.json")
    monkeypatch.setattr(piper_catalog, "_VOICES_DIR", tmp_path / "voices")

    with patch("httpx.get", return_value=_make_httpx_response(_FAKE_RAW)):
        result = piper_catalog.fetch_catalog(force=True)

    assert len(result) == 1
    v = result[0]
    assert v["key"] == "cs_CZ-jirka-medium"
    assert v["language_family"] == "cs"
    assert v["installed"] is False


def test_remove_voice_rejects_unsafe_key():
    from app import tts
    with pytest.raises(ValueError, match="unsafe"):
        tts.remove_voice("../etc/passwd")


def test_remove_voice_rejects_slash_key():
    from app import tts
    with pytest.raises(ValueError, match="unsafe"):
        tts.remove_voice("cs/bad")


def test_install_voice_mocked(tmp_path, monkeypatch):
    import httpx
    from app import tts, piper_catalog

    monkeypatch.setattr(tts, "_VOICES_DIR", tmp_path / "voices")
    monkeypatch.setattr(piper_catalog, "_VOICES_DIR", tmp_path / "voices")
    monkeypatch.setattr(piper_catalog, "_CACHE_PATH", tmp_path / "piper_catalog.json")

    # Pre-populate catalog cache
    import json, time
    flat = piper_catalog._flatten(_FAKE_RAW)
    (tmp_path / "piper_catalog.json").write_text(json.dumps({"catalog": flat, "fetched_at": time.time()}))

    fake_content = b"fake-onnx-data"

    class _FakeStream:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def raise_for_status(self): pass
        def iter_bytes(self, chunk_size=None):
            yield fake_content

    monkeypatch.setattr(httpx, "stream", lambda *a, **kw: _FakeStream())

    result = tts.install_voice("cs_CZ-jirka-medium")
    assert result["voice"] == "cs_CZ-jirka-medium"
    assert result["size_bytes"] > 0


def test_catalog_endpoint_installed_flag(tmp_path, monkeypatch):
    """Catalog returns installed=True for cs_CZ-jirka-medium when voice dir exists."""
    import json, time
    from app import piper_catalog

    voices_dir = tmp_path / "voices"
    voice_dir = voices_dir / "cs_CZ-jirka-medium"
    voice_dir.mkdir(parents=True)
    (voice_dir / "cs_CZ-jirka-medium.onnx").write_bytes(b"x")
    (voice_dir / "cs_CZ-jirka-medium.onnx.json").write_text("{}")
    monkeypatch.setattr(piper_catalog, "_VOICES_DIR", voices_dir)
    monkeypatch.setattr(piper_catalog, "_CACHE_PATH", tmp_path / "piper_catalog.json")

    flat = piper_catalog._flatten(_FAKE_RAW)
    (tmp_path / "piper_catalog.json").write_text(json.dumps({"catalog": flat, "fetched_at": time.time()}))

    result = piper_catalog.fetch_catalog()
    cs = next((v for v in result if v["key"] == "cs_CZ-jirka-medium"), None)
    assert cs is not None
    assert cs["installed"] is True
