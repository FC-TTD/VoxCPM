from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FUNASR_PATH = ROOT / "funasr" / "__init__.py"


def load_local_funasr_module():
    spec = importlib.util.spec_from_file_location("funasr_local_shim", FUNASR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_automodel_generate_uses_internal_asr_api(monkeypatch, tmp_path):
    funasr = load_local_funasr_module()
    audio_path = tmp_path / "prompt.wav"
    audio_path.write_bytes(b"RIFFfake")

    payload = {
        "result": [{"text": "recognized text", "clean_text": "", "raw_text": ""}]
    }

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["content_type"] = req.headers["Content-type"]
        captured["body"] = req.data
        return FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(funasr.request, "urlopen", fake_urlopen)

    model = funasr.AutoModel(
        model="iic/SenseVoiceSmall",
        disable_update=True,
        log_level="DEBUG",
        device="cpu",
    )
    result = model.generate(input=str(audio_path), language="auto", use_itn=True)

    assert result == [{"text": "recognized text"}]
    assert captured["url"] == "http://asrpri-api/api/v1/asr"
    assert captured["timeout"] == 60.0
    assert "multipart/form-data" in captured["content_type"]
    assert b'name="files"; filename="prompt.wav"' in captured["body"]
    assert b'name="lang"' in captured["body"]
    assert b"auto" in captured["body"]
