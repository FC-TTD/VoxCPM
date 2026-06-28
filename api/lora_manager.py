import os
import logging
from collections import OrderedDict
from typing import Optional, Dict, Union
import torch

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

logger = logging.getLogger(__name__)

class LoRAManager:
    """
    Manages LoRA adapters with an LRU cache to avoid repetitive disk I/O.
    Ensures that switching between recently used LoRAs is fast (memory copy only).
    """
    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        # Cache stores state_dicts on CPU to save VRAM
        self.cache: OrderedDict[str, Dict[str, torch.Tensor]] = OrderedDict()
        self.current_lora_path: Optional[str] = None

    def _load_state_dict_from_disk(self, lora_path: str) -> Dict[str, torch.Tensor]:
        """Load LoRA weights from disk (safetensors or bin/ckpt) into CPU memory."""
        if os.path.isdir(lora_path):
            safetensors_file = os.path.join(lora_path, "lora_weights.safetensors")
            ckpt_file = os.path.join(lora_path, "lora_weights.ckpt")
            # Also check .pth
            pth_file = os.path.join(lora_path, "lora_weights.pth")
        else:
            if lora_path.endswith(".safetensors"):
                safetensors_file = lora_path
                ckpt_file = None
                pth_file = None
            else:
                safetensors_file = None
                ckpt_file = lora_path
                pth_file = lora_path

        if safetensors_file and os.path.exists(safetensors_file) and SAFETENSORS_AVAILABLE:
            logger.debug(f"Loading safetensors from {safetensors_file}")
            return load_file(safetensors_file, device="cpu")
        elif ckpt_file and os.path.exists(ckpt_file):
            logger.debug(f"Loading ckpt from {ckpt_file}")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            return ckpt.get("state_dict", ckpt)
        elif pth_file and os.path.exists(pth_file):
            logger.debug(f"Loading pth from {pth_file}")
            ckpt = torch.load(pth_file, map_location="cpu")
            return ckpt.get("state_dict", ckpt)
        else:
            raise FileNotFoundError(f"LoRA checkpoint not found at {lora_path}")

    def _apply_weights(self, model: torch.nn.Module, state_dict: Dict[str, torch.Tensor], device: Union[str, torch.device]):
        """Apply state_dict to model, handling potential torch.compile prefixes."""
        model_params = dict(model.named_parameters())
        # Handle torch.compile's _orig_mod prefix if present in model keys
        key_mapping = {k.replace("._orig_mod.", "."): k for k in model_params if "._orig_mod." in k}
        
        with torch.no_grad():
            for key, value in state_dict.items():
                target_key = key if key in model_params else key_mapping.get(key)
                if target_key:
                    # Copy to VRAM/Device
                    model_params[target_key].data.copy_(value.to(device))
    
    def switch_lora(self, voxcpm_model, lora_path: Optional[str]):
        """
        Switch to the specified LoRA.
        
        Args:
            voxcpm_model: The VoxCPM wrapper instance.
            lora_path: Absolute path to LoRA, or None/Empty to disable LoRA.
        """
        if not lora_path:
            if self.current_lora_path is not None:
                logger.info("Disabling LoRA")
                voxcpm_model.set_lora_enabled(False)
                self.current_lora_path = None
            return

        if self.current_lora_path == lora_path:
            # Already loaded and enabled
            if not voxcpm_model.lora_enabled:
                 voxcpm_model.set_lora_enabled(True)
            return

        logger.info(f"Switching LoRA to {lora_path}")
        
        # 1. Check Cache
        if lora_path in self.cache:
            self.cache.move_to_end(lora_path) # Mark as recently used
            state_dict = self.cache[lora_path]
            logger.debug("LoRA loaded from cache")
        else:
            # 2. Load from disk
            state_dict = self._load_state_dict_from_disk(lora_path)
            # 3. Update Cache
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False) # Evict oldest
            self.cache[lora_path] = state_dict
            logger.debug("LoRA loaded from disk and cached")
        
        # 4. Apply to Model
        # We access the inner tts_model to set weights
        self._apply_weights(voxcpm_model.tts_model, state_dict, voxcpm_model.tts_model.device)
        voxcpm_model.set_lora_enabled(True)
        self.current_lora_path = lora_path
