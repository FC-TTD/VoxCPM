import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT)) # For api module

# Mock voxcpm module to avoid loading real model
sys.modules["voxcpm"] = MagicMock()
sys.modules["voxcpm.model"] = MagicMock()
class DummyVoxCPM2Model: pass
import types
m = types.ModuleType("voxcpm.model.voxcpm2")
m.VoxCPM2Model = DummyVoxCPM2Model
sys.modules["voxcpm.model.voxcpm2"] = m
sys.modules["voxcpm.model.voxcpm"] = MagicMock()

# Now import app
from api.server import app

# Create client
client = TestClient(app)

@pytest.fixture
def mock_model():
    # Setup the mock model
    mock = MagicMock()
    mock.tts_model = DummyVoxCPM2Model()
    mock.tts_model.sample_rate = 16000
    mock.tts_model._encode_sample_rate = 16000
    mock.generate = MagicMock()
    mock.generate.return_value = np.zeros(16000, dtype=np.float32)
    
    # Setup mock model manager with get() method
    mock_manager = MagicMock()
    mock_manager.get.return_value = mock
    mock_manager.stop.return_value = None
    
    mock_lora_manager = MagicMock()

    with patch("api.server.model_manager", mock_manager), patch("api.server.lora_manager", mock_lora_manager):
        yield mock

def test_health(mock_model):
    # Ensure model is "loaded" (not None) via fixture
    response = client.get("/health")
    assert response.status_code == 200, f"Response: {response.json()}"
    assert "status" in response.json()

def test_generate_no_model():
    with patch("api.server.model_manager", None):
        response = client.post("/generate", data={"text": "hello"})
        assert response.status_code == 503

def test_generate_success(mock_model):
    response = client.post("/generate", data={"text": "hello"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"

def test_generate_with_control(mock_model):
    response = client.post("/generate", data={"text": "hello", "control": "warm female voice"})
    assert response.status_code == 200
    assert mock_model.generate.call_args.kwargs["text"] == "(warm female voice)hello"

def test_generate_with_lora(mock_model):
    with patch("os.path.exists", side_effect=lambda p: True): # Always exists
        response = client.post("/generate", data={"text": "hello", "lora_name": "test_lora"})
        assert response.status_code == 200

def test_generate_with_invalid_lora(mock_model):
     # os.path.exists returns False by default if not patched, or we force it
     with patch("os.path.exists", return_value=False):
        response = client.post("/generate", data={"text": "hello", "lora_name": "non_existent"})
        assert response.status_code == 404, f"Response: {response.json()}"

def test_generate_rejects_prompt_audio_without_prompt_text(mock_model, tmp_path):
    prompt_path = tmp_path / "prompt.wav"
    prompt_path.write_bytes(b"fake")

    with open(prompt_path, "rb") as f:
        response = client.post(
            "/generate",
            data={"text": "hello"},
            files={"prompt_audio": ("prompt.wav", f, "audio/wav")},
        )

    assert response.status_code == 400

def test_generate_rejects_prompt_text_without_prompt_audio(mock_model):
    response = client.post("/generate", data={"text": "hello", "prompt_text": "test prompt"})
    assert response.status_code == 400

def test_generate_with_reference_audio_on_legacy_runtime_returns_400(mock_model, tmp_path):
    mock_model.tts_model.sample_rate = 48000
    mock_model.tts_model._encode_sample_rate = 16000
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(b"fake")

    with patch("api.server.preprocess_audio", return_value=str(reference_path)):
        with open(reference_path, "rb") as f:
            response = client.post(
                "/generate",
                data={"text": "hello"},
                files={"reference_audio": ("reference.wav", f, "audio/wav")},
            )

    assert response.status_code == 400

def test_generate_with_reference_audio_on_v2_runtime(mock_model, tmp_path):
    mock_model.tts_model.sample_rate = 48000
    mock_model.tts_model._encode_sample_rate = 16000
    captured = {}

    def generate_v2(
        *,
        text,
        prompt_wav_path=None,
        prompt_text=None,
        reference_wav_path=None,
        cfg_value=2.0,
        inference_timesteps=10,
        normalize=True,
        denoise=True,
    ):
        captured.update(
            {
                "text": text,
                "prompt_wav_path": prompt_wav_path,
                "prompt_text": prompt_text,
                "reference_wav_path": reference_wav_path,
                "cfg_value": cfg_value,
                "inference_timesteps": inference_timesteps,
                "normalize": normalize,
                "denoise": denoise,
            }
        )
        return np.zeros(48000, dtype=np.float32)

    mock_model.generate = generate_v2
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(b"fake")

    with patch("api.server.preprocess_audio", return_value=str(reference_path)):
        with open(reference_path, "rb") as f:
            response = client.post(
                "/generate",
                data={"text": "hello", "control": "bright energetic"},
                files={"reference_audio": ("reference.wav", f, "audio/wav")},
            )

    assert response.status_code == 200
    assert captured["reference_wav_path"] == str(reference_path)
    assert captured["text"] == "(bright energetic)hello"

def test_preprocess_uses_input_sample_rate_for_reference_audio(mock_model, tmp_path):
    mock_model.tts_model.sample_rate = 48000
    mock_model.tts_model._encode_sample_rate = 16000

    def generate_v2(
        *,
        text,
        prompt_wav_path=None,
        prompt_text=None,
        reference_wav_path=None,
        cfg_value=2.0,
        inference_timesteps=10,
        normalize=True,
        denoise=True,
    ):
        return np.zeros(48000, dtype=np.float32)

    mock_model.generate = generate_v2
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(b"fake")

    with patch("api.server.preprocess_audio", return_value=str(reference_path)) as preprocess_mock:
        with open(reference_path, "rb") as f:
            response = client.post(
                "/generate",
                data={"text": "hello"},
                files={"reference_audio": ("reference.wav", f, "audio/wav")},
            )

    assert response.status_code == 200
    assert preprocess_mock.call_args.args[1] == 16000
