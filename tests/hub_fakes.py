"""CPU-only fixtures; no pretrained model constructors or production weights."""
from functools import wraps
import inspect
import threading
import numpy as np
import soundfile as sf
import torch
from api.lora_manager import LoRAManager
from hub_runtime.adapter import NativeBackend


class TinyTTS(torch.nn.Module):
    sample_rate = 48000
    _encode_sample_rate = 16000
    device = "cpu"
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2,2))


class FakeNative:
    def __init__(self, calls=None, gate=None):
        self.tts_model = TinyTTS()
        self.lora_enabled = False
        self.calls = calls if calls is not None else []
        self.gate = gate

    def set_lora_enabled(self, enabled):
        self.lora_enabled = enabled

    def _generate(self, *, reference_wav_path=None, **kwargs):
        pass

    def generate(self, reference_wav_path=None, **kwargs):
        if reference_wav_path is not None:
            kwargs["reference_wav_path"] = reference_wav_path
        record = dict(kwargs, active_lora=self.lora_enabled)
        for key in ("prompt_wav_path", "reference_wav_path"):
            if kwargs.get(key):
                wav,sr=sf.read(kwargs[key]);record[key+"_sample_rate"]=sr
        self.calls.append(record)
        if self.gate:
            self.gate[0].set()
            assert self.gate[1].wait(10)
        return (0.2*np.sin(2*np.pi*220*np.arange(48000)/48000)).astype(np.float32)


def load_fake_engine():
    return NativeBackend(FakeNative(), LoRAManager(capacity=5))


class FakeRuntime:
    def __init__(self, backend=None):
        self.backend = backend or load_fake_engine()
        self.active = 0
        self.started = False
        self.calls = 0
        self.lock = threading.Lock()
    def task(self, fn):
        if inspect.iscoroutinefunction(fn):
            @wraps(fn)
            async def wrapped(*args, **kwargs):
                with self.lock: self.active += 1; self.calls += 1
                try: return await fn(*args, **kwargs)
                finally:
                    with self.lock: self.active -= 1
        else:
            @wraps(fn)
            def wrapped(*args, **kwargs):
                with self.lock: self.active += 1; self.calls += 1
                try: return fn(*args, **kwargs)
                finally:
                    with self.lock: self.active -= 1
        return wrapped
    def pending_work(self): return 0
    def get(self):
        assert self.active > 0, "native inference bypassed managed activity"
        return self.backend
    def status(self):
        return {"accepting": True, "healthy": True, "residency": "ready"}
    def start(self): self.started = True
    def stop(self): self.started = False


if __name__ == "__main__":
    import os,uvicorn
    from hub_runtime.__main__ import create_app
    uvicorn.run(create_app(FakeRuntime()),host="127.0.0.1",port=int(os.environ["HUB_TEST_PORT"]))
