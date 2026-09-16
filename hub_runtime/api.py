"""Copied /generate contract at 8d3237a; lifecycle and GPU access use Hub."""
import os
import logging
import asyncio
import tempfile
from typing import Optional
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.concurrency import run_in_threadpool
import soundfile as sf
import numpy as np
import librosa
from ttd_model_runtime import HubError
from ttd_model_runtime.audio.postprocess import eq as _eq, loudnorm as _loudnorm, trim_silence as _trim_silence
logger = logging.getLogger("voxcpm-api")


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

def _get_input_sample_rate(model):
    return model.input_sample_rate()


def _supports_reference_audio(model):
    return model.supports_reference_audio()


def _build_generation_kwargs(
    model,
    *,
    text: str,
    control: Optional[str],
    prompt_wav_path: Optional[str],
    prompt_text: Optional[str],
    reference_wav_path: Optional[str],
    cfg_value: float,
    inference_timesteps: int,
    normalize: bool,
    denoise: bool,
) -> dict:
    final_text = text
    if control and control.strip():
        final_text = f"({control.strip()}){text}"

    kwargs = {
        "text": final_text,
        "prompt_wav_path": prompt_wav_path,
        "prompt_text": prompt_text,
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
        "normalize": normalize,
        "denoise": denoise,
    }

    if reference_wav_path:
        if not _supports_reference_audio(model):
            raise HTTPException(
                status_code=400,
                detail="Current model runtime does not support reference_audio. Upgrade to a VoxCPM2-compatible runtime first.",
            )
        kwargs["reference_wav_path"] = reference_wav_path

    return kwargs

def build_api(runtime):
    app = FastAPI(title="VoxCPM API")
    GPU_LOCK = asyncio.Lock()

    @app.post("/generate")
    @runtime.task
    async def generate(
        text: str = Form(...),
        prompt_audio: Optional[UploadFile] = File(None),
        prompt_text: Optional[str] = Form(None),
        reference_audio: Optional[UploadFile] = File(None),
        control: Optional[str] = Form(None),
        lora_name: Optional[str] = Form(None),
        cfg_value: float = Form(2.0),
        inference_timesteps: int = Form(10),
        normalize: bool = Form(True),
        denoise: bool = Form(False), # Input prompt denoising
        postprocess: bool = Form(True), # Output audio post-processing
        trim_silence: bool = Form(True), # Output silence trimming
        lufs: float = Form(-23.0),
    ):

        temp_prompt_path = None
        temp_reference_path = None
        preprocessed_path = None
        preprocessed_reference_path = None
        try:
            # Get model instance
            model = runtime.get()

            if prompt_audio and not prompt_text:
                raise HTTPException(status_code=400, detail="prompt_text is required when prompt_audio is provided")
            if prompt_text and not prompt_audio:
                raise HTTPException(status_code=400, detail="prompt_audio is required when prompt_text is provided")
            
            # Handle prompt audio
            if prompt_audio:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    content = await prompt_audio.read()
                    f.write(content)
                    temp_prompt_path = f.name
                
                target_sr = _get_input_sample_rate(model)
                preprocessed_path = await run_in_threadpool(preprocess_audio, temp_prompt_path, target_sr)

            if reference_audio:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    content = await reference_audio.read()
                    f.write(content)
                    temp_reference_path = f.name

                target_sr = _get_input_sample_rate(model)
                preprocessed_reference_path = await run_in_threadpool(preprocess_audio, temp_reference_path, target_sr)
            
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

                # Generate
                logger.info(f"Generating TTS for text: {text[:20]}...")
                input_wav_path = preprocessed_path if preprocessed_path else temp_prompt_path

                generation_kwargs = _build_generation_kwargs(
                    model,
                    text=text,
                    control=control,
                    prompt_wav_path=input_wav_path,
                    prompt_text=prompt_text,
                    reference_wav_path=preprocessed_reference_path if preprocessed_reference_path else temp_reference_path,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    normalize=normalize,
                    denoise=denoise,
                )

                sr, wav_np = await run_in_threadpool(model.generate_managed, target_lora_path, **generation_kwargs)

                # Post-processing
                if trim_silence:
                    try:
                        wav_np = _trim_silence(wav_np, sr)
                    except Exception as e:
                        logger.warning(f"Silence trimming failed: {e}")

                if postprocess:
                    try:
                        wav_np, _ = _loudnorm(wav_np, sr, target_loudness=float(lufs))
                        wav_np = _eq(wav_np, sr)
                    except Exception as e:
                        logger.warning(f"Postprocess failed: {e}")

                # Convert to bytes
                buffer = BytesIO()
                sf.write(buffer, wav_np, sr, format="WAV")
                buffer.seek(0)
                
                return Response(content=buffer.read(), media_type="audio/wav")

        except (HTTPException, HubError):
            raise
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            for p in [temp_prompt_path, temp_reference_path, preprocessed_path, preprocessed_reference_path]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
    return app
