# VoxCPM API Server Guide

The VoxCPM API Server provides a FastAPI-based interface for high-performance Text-to-Speech generation, compatible with TTD Swarm and featuring dynamic LoRA adapter hot-swapping.

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
| `prompt_audio` | file | Optional | Reference WAV file for voice cloning. |
| `prompt_text` | string | Optional | Transcript of the prompt audio. |
| `lora_name` | string | Optional | Name of LoRA in `/app/lora` or absolute path. Use `None` to disable. |
| `cfg_value` | float | 2.0 | CFG guidance scale. |
| `inference_timesteps` | int | 10 | Diffusion steps. |
| `normalize` | bool | True | Enable text normalization. |
| `denoise` | bool | True | Denoise prompt audio before processing. |
| `postprocess` | bool | True | Enable output loudness norm and EQ. |
| `trim_silence` | bool | True | Trim silence from the start/end of generated audio. |
| `lufs` | float | -23.0 | Target loudness for post-processing loudnorm, in LUFS. |

**Response:**
Returns `audio/wav` file.

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

- `VOXCPM_MODEL_PATH`: Path to base model (default: `openbmb/VoxCPM1.5`).
- `DEVICE`: Torch device (default: auto-detect `cuda` or `cpu`).
