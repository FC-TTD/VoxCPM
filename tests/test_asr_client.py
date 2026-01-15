import sys
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lib import asr_client


def _write_dummy_audio(path: Path) -> None:
    path.write_bytes(b"RIFF0000WAVEfmt ")


def test_extract_text_variants() -> None:
    assert asr_client.extract_text({"result": [{"clean_text": "hello"}]}) == "hello"
    assert asr_client.extract_text({"result": {"text": "world"}}) == "world"
    assert asr_client.extract_text({"result": []}) == ""


def test_asr_model_calls_requests(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: Dict[str, Any] = {}

    class DummyResponse:
        def json(self) -> Dict[str, Any]:
            return {"result": [{"clean_text": "ok"}]}

    def fake_post(url: str, **kwargs: Any) -> DummyResponse:
        captured["url"] = url
        captured.update(kwargs)
        return DummyResponse()

    monkeypatch.setattr(asr_client.requests, "post", fake_post)
    audio_path = tmp_path / "sample.wav"
    _write_dummy_audio(audio_path)

    with audio_path.open("rb") as audio_data:
        result = asr_client.asr_model(
            audio_data,
            host="http://example.com",
            language="zh",
            timeout=5,
        )

    assert result["result"][0]["clean_text"] == "ok"
    assert captured["url"] == "http://example.com/api/v1/asr"
    assert captured["data"] == {"keys": "wav", "lang": "zh"}
    assert captured["timeout"] == 5


def test_auto_asr_reads_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_asr_model(audio_data, host: str, language: str, timeout: int) -> Dict[str, Any]:
        return {"result": [{"clean_text": "hello"}]}

    monkeypatch.setattr(asr_client, "asr_model", fake_asr_model)
    audio_path = tmp_path / "sample.wav"
    _write_dummy_audio(audio_path)

    assert asr_client.auto_asr(str(audio_path), host="example", language="auto", timeout=3) == "hello"
    assert asr_client.auto_asr(str(tmp_path / "missing.wav")) == ""
