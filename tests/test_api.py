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
sys.modules["voxcpm.model.voxcpm"] = MagicMock()

# Now import app
from api.server import app

# Create client
client = TestClient(app)

@pytest.fixture
def mock_model():
    # Setup the global model in api.server
    mock = MagicMock()
    mock.tts_model.sample_rate = 16000
    mock.generate.return_value = np.zeros(16000, dtype=np.float32)
    
    mock_lora_manager = MagicMock()

    with patch("api.server.model", mock), patch("api.server.lora_manager", mock_lora_manager):
        yield mock

def test_health(mock_model):
    # Ensure model is "loaded" (not None) via fixture
    response = client.get("/health")
    assert response.status_code == 200, f"Response: {response.json()}"
    assert "status" in response.json()

def test_generate_no_model():
    with patch("api.server.model", None):
        response = client.post("/generate", data={"text": "hello"})
        assert response.status_code == 503

def test_generate_success(mock_model):
    response = client.post("/generate", data={"text": "hello"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"

def test_generate_with_lora(mock_model):
    with patch("os.path.exists", side_effect=lambda p: True): # Always exists
        response = client.post("/generate", data={"text": "hello", "lora_name": "test_lora"})
        assert response.status_code == 200

def test_generate_with_invalid_lora(mock_model):
     # os.path.exists returns False by default if not patched, or we force it
     with patch("os.path.exists", return_value=False):
        response = client.post("/generate", data={"text": "hello", "lora_name": "non_existent"})
        assert response.status_code == 404, f"Response: {response.json()}"