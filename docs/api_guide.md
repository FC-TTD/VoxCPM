# VoxCPM API Server Guide

The VoxCPM API Server provides a FastAPI-based interface for high-performance Text-to-Speech generation, compatible with TTD Swarm and featuring dynamic LoRA adapter hot-swapping. It is now aligned with the VoxCPM2 reference-audio and continuation workflow.

## Features

- **Dynamic LoRA Switching**: Load, enable, or disable LoRA adapters on-the-fly via API parameters.
- **Advanced Post-processing**: Built-in silence trimming, loudness normalization, and EQ (pre-emphasis).
- **TTD Swarm Compatible**: Integrated health checks for CUDA status and deployment ready with `stack.yml`.
- **GPU Efficient**: Internal locking mechanism to prevent OOM during concurrent requests.

## Deployment

### Local Start

Ensure you have the dependencies installed:

```bash
uv pip install fastapi uvicorn python-multipart pyloudnorm soxr ttd-fastapi-utils
```

Start the server:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Docker / Swarm

Use the provided `docker/stack.yml` for Swarm deployment:

```bash
docker stack deploy -c docker/stack.yml voxcpm
```

The API will be exposed on port `8000` (or via Caddy at `http://voxcpm-api`).

## API Endpoints

### 1. `POST /generate`

Generate audio from text.

**Parameters (Form Data):**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `text` | string | **Required** | The text to synthesize. |
| `prompt_audio` | file | Optional | Prompt WAV file for continuation / ultimate cloning. Must be paired with `prompt_text`. |
| `prompt_text` | string | Optional | Transcript of the prompt audio. Must be paired with `prompt_audio`. |
| `reference_audio` | file | Optional | Reference WAV file for VoxCPM2 voice cloning. |
| `control` | string | Optional | Natural-language voice/style control. The server prepends it to text as `(control)text`. |
| `lora_name` | string | Optional | Name of LoRA in `/app/lora` or absolute path. Use `None` to disable. |
| `cfg_value` | float | 2.0 | CFG guidance scale. |
| `inference_timesteps` | int | 10 | Diffusion steps. |
| `normalize` | bool | True | Enable text normalization. |
| `denoise` | bool | True | Denoise prompt audio before processing. |
| `postprocess` | bool | True | Enable output loudness norm and EQ. |
| `trim_silence` | bool | True | Trim silence from the start/end of generated audio. |

**Response:**
Returns `audio/wav` file. The output sample rate follows the loaded model, typically 48kHz on VoxCPM2.

**Mode combinations:**

- `text` only: plain TTS / voice design
- `text` + `reference_audio`: VoxCPM2 controllable cloning
- `text` + `prompt_audio` + `prompt_text`: continuation / ultimate cloning
- `text` + `reference_audio` + `prompt_audio` + `prompt_text`: VoxCPM2 reference + continuation combined mode

### 2. `GET /health`

Health check for deployment.

**Response:**

```json
{
  "status": "ok",
  "cuda_available": true
}
```

## Environment Variables

- `VOXCPM_MODEL_PATH`: Path to base model (default: `openbmb/VoxCPM2`).
- `DEVICE`: Torch device (default: auto-detect `cuda` or `cpu`).
