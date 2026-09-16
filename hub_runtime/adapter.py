"""Native VoxCPM and its existing CPU LoRA cache in one assigned GPU child."""
import gc
import os
import threading
from fastapi import HTTPException


def tensor_placement(module):
    try:
        tensor = next(module.parameters())
        return str(tensor.device), str(tensor.dtype)
    except (AttributeError, StopIteration):
        return "unknown", "unknown"


class NativeBackend:
    def __init__(self, model, lora_manager):
        self.model = model
        self.lora_manager = lora_manager
        self._lock = threading.RLock()
        main_device, main_dtype = tensor_placement(model.tts_model)
        vae_device, vae_dtype = tensor_placement(getattr(model.tts_model, "audio_vae", None))
        pipeline = getattr(getattr(model, "denoiser", None), "_pipeline", None)
        self.__hub_device_summary__ = {
            "main_model": main_device, "main_dtype": main_dtype,
            "audio_vae": vae_device, "audio_vae_dtype": vae_dtype,
            "denoiser": str(getattr(pipeline, "device", "unknown")),
        }

    def input_sample_rate(self):
        model = self.model.tts_model
        return int(getattr(model, "_encode_sample_rate", getattr(model, "sample_rate", 16000)))

    def supports_reference_audio(self):
        import inspect
        generate = getattr(self.model, "generate", None)
        try:
            if "reference_wav_path" in inspect.signature(generate).parameters:
                return True
        except (TypeError, ValueError):
            pass
        from voxcpm.model.voxcpm2 import VoxCPM2Model
        model = getattr(self.model, "tts_model", None)
        return isinstance(model, VoxCPM2Model) and type(model).__module__.startswith("voxcpm.")

    def generate_managed(self, lora_path, **kwargs):
        # API preserves its original explicit LoRA selection/disable semantics.
        with self._lock:
            try:
                self.lora_manager.switch_lora(self.model, lora_path)
            except Exception as exc:
                raise HTTPException(500, detail=f"Failed to switch LoRA: {exc}") from exc
            wav = self.model.generate(**kwargs)
            return int(self.model.tts_model.sample_rate), wav

    def generate_ui(self, **kwargs):
        # Native UI shares the active LoRA selected by the API. It does not
        # expose a LoRA selector; do not silently disable that native state.
        with self._lock:
            wav = self.model.generate(**kwargs)
            return int(self.model.tts_model.sample_rate), wav


def load_model():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Managed VoxCPM requires its assigned CUDA GPU")
    from voxcpm import VoxCPM
    from voxcpm.model.voxcpm import LoRAConfig
    from api.lora_manager import LoRAManager
    # Retain native LM BF16, AudioVAE FP32, ZipEnhancer's original device, and
    # CPU LoRA state-dict cache. There is no capacity fallback/device rewrite.
    model = VoxCPM.from_pretrained(
        hf_model_id=os.getenv("VOXCPM_MODEL_DIR", "openbmb/VoxCPM2"),
        load_denoiser=True, optimize=False,
        lora_config=LoRAConfig(enable_lm=True, enable_dit=True, enable_proj=False),
    )
    return NativeBackend(model, LoRAManager(capacity=5))


def completion(model):
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def release(model):
    with model._lock:
        model.model = None
        model.lora_manager.cache.clear()
        model.lora_manager.current_lora_path = None


def cleanup():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
