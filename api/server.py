import os
import sys
import logging
import asyncio
import tempfile
from contextlib import asynccontextmanager
from typing import Optional
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.concurrency import run_in_threadpool
import soundfile as sf
import torch
import numpy as np
import librosa
import soxr

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from voxcpm import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig
from .lora_manager import LoRAManager

# Logging setup
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("voxcpm-api")

from ttd_fastapi_utils import (
    eq as _eq,
    loudnorm as _loudnorm,
    setup_cuda_health,
    trim_silence as _trim_silence,
    SmartModel,
)

GPU_LOCK = asyncio.Lock()
model_manager: Optional[SmartModel] = None
lora_manager: Optional[LoRAManager] = None

# Environment variables
MODEL_PATH = os.environ.get("VOXCPM_MODEL_PATH", "openbmb/VoxCPM1.5")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def preprocess_audio(file_path: str, target_sr: int = 16000) -> str:
    """
    Load audio, mix to mono, resample to target_sr, and handle errors.
    Returns path to preprocessed temporary WAV file.
    """
    try:
        # librosa handles resampling and mono mixing automatically
        # mono=True mixes to mono (mean)
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        # Check for silence or invalid data
        if len(y) == 0:
            raise ValueError("Audio file is empty")
        if np.all(y == 0):
            logger.warning("Audio file is silent")
        
        # Normalize to [-1, 1] if not already (librosa usually does 0-1 float32)
        max_val = np.abs(y).max()
        if max_val > 1.0:
            y = y / max_val
            
        # Create temp file
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        
        sf.write(temp_path, y, target_sr, subtype='PCM_16')
        return temp_path
    except Exception as e:
        logger.error(f"Preprocessing failed for {file_path}: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager, lora_manager
    device = _resolve_device()
    
    # Initialize with default LoRA config to enable hot-swapping
    lora_config = LoRAConfig(
        enable_lm=True,
        enable_dit=True,
        enable_proj=False
    )
    
    def loader():
        logger.info(f"Loading VoxCPM model from {MODEL_PATH} on {device}...")
        model = VoxCPM.from_pretrained(
            hf_model_id=MODEL_PATH,
            load_denoiser=True, 
            optimize=True,
            lora_config=lora_config,
            device=device
        )
        logger.info("VoxCPM model loaded successfully.")
        return model

    try:
        # Default 2h timeout
        model_manager = SmartModel(loader, timeout_seconds=7200)
        lora_manager = LoRAManager(capacity=5)
        yield
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    finally:
        if model_manager:
            model_manager.stop()
            model_manager = None
    

app = FastAPI(title="VoxCPM API", lifespan=lifespan)

# Setup CUDA health check
cuda_monitor = setup_cuda_health(
    app,
    path="/health",
    ready_predicate=lambda: model_manager is not None,
)

@app.post("/generate")
async def generate(
    text: str = Form(...),
    prompt_audio: Optional[UploadFile] = File(None),
    prompt_text: Optional[str] = Form(None),
    lora_name: Optional[str] = Form(None),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(10),
    normalize: bool = Form(True),
    denoise: bool = Form(True), # Input prompt denoising
    postprocess: bool = Form(True), # Output audio post-processing
    trim_silence: bool = Form(True), # Output silence trimming
):
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    temp_prompt_path = None
    preprocessed_path = None
    try:
        # Get model instance
        model = model_manager.get()
        
        # Handle prompt audio
        if prompt_audio:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                content = await prompt_audio.read()
                f.write(content)
                temp_prompt_path = f.name
            
            # Preprocess audio to avoid tensor errors
            # Target SR depends on model config, usually available in model.tts_model.sample_rate
            # But VoxCPM core might need original SR or specific? 
            # Core handles resampling, but to be safe we use librosa to sanitize first.
            target_sr = model.tts_model.sample_rate if model else 16000 # Fallback
            # Run in threadpool as it is CPU bound
            preprocessed_path = await run_in_threadpool(preprocess_audio, temp_prompt_path, target_sr)
        
        async with GPU_LOCK:
            # Handle LoRA switching via Manager
            target_lora_path = None
            if lora_name and lora_name.lower() != "none":
                potential_paths = [
                    lora_name,
                    os.path.join("lora", lora_name),
                    os.path.join("/app/lora", lora_name),
                ]
                for p in potential_paths:
                    if os.path.exists(p):
                        target_lora_path = p
                        break
                if not target_lora_path:
                    raise HTTPException(status_code=404, detail=f"LoRA '{lora_name}' not found.")

            # Perform switch (Manager handles caching and diff check)
            try:
                await run_in_threadpool(lora_manager.switch_lora, model, target_lora_path)
            except Exception as e:
                logger.error(f"Failed to switch LoRA: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to switch LoRA: {e}")
            
            # Generate
            logger.info(f"Generating TTS for text: {text[:20]}...")
            # Use preprocessed path if available, else temp_prompt_path (which might be raw upload if preprocess failed/skipped? No, if prompt_audio exists, we preprocess)
            # If preprocess failed, it raises HTTPException.
            input_wav_path = preprocessed_path if preprocessed_path else temp_prompt_path
            
            wav_np = await run_in_threadpool(
                model.generate,
                text=text,
                prompt_wav_path=input_wav_path,
                prompt_text=prompt_text,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize,
                denoise=denoise
            )
            
            sr = model.tts_model.sample_rate

            # Post-processing
            if trim_silence:
                try:
                    wav_np = _trim_silence(wav_np, sr)
                except Exception as e:
                    logger.warning(f"Silence trimming failed: {e}")

            if postprocess:
                try:
                    wav_np, _ = _loudnorm(wav_np, sr)
                    wav_np = _eq(wav_np, sr)
                except Exception as e:
                    logger.warning(f"Postprocess failed: {e}")

            # Convert to bytes
            buffer = BytesIO()
            sf.write(buffer, wav_np, sr, format="WAV")
            buffer.seek(0)
            
            return Response(content=buffer.read(), media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [temp_prompt_path, preprocessed_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
