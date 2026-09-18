# Managed VoxCPM2

2026-09-16: formally managed on worker; existing domains, native API/UI and Gateway acceptance passed. Current reservation: 8.5 GiB. Full evidence: Hub `docs/proposals/model-compute-pool/worker-expansion-2026-09-16.md`. Runtime source: `b74bee6`; subsequent documentation commits do not change the deployed model image.

This additive runtime is based on local source
`8d3237afbcf2d57ab3aa07534bc168292a51785e`. Its API, UI, entrypoint, LoRA manager
and VoxCPM2 framework match the running `/opt/voxcpm` source and formal container
by SHA256. This local commit is not on GitHub; remote main/api have different
runtime code and must not silently replace this verified source base.

Original image config ID: `sha256:40a1a55356d3d840e61fc7b02d769335c30af93aa519ce7cf2546af210837c10`.
Preserved immutable base: `registry.ttd/voxcpm/fusion@sha256:72a2805c62fabd44ff25cb206882c536760d9344b869959156c5577b595b2044`.
Framework: Python3.11.10, Gradio6.19.0, FastAPI0.138.1, Torch2.5.0+cu124,
Transformers5.12.1. This is not the Qwen Gradio5 runtime.

Run `python -m hub_runtime` with `HUB_SERVICE_ID=voxcpm`, the standard SDK actor
identity and node transport, `HUB_GPU_PROCESS=1`, and empty parent
`CUDA_VISIBLE_DEVICES`. `python -m hub_runtime describe` reports the original
multipart `/generate` API without importing Torch or the retired plugin.

Preserve `http://voxcpm-api`, its `/generate` API and `/gradio/` UI, including the
root redirect, original callback names, file uploads/downloads, logo and I18N.
The UI is copied from the original app and mounted with native Gradio6 options;
SDK attach owns the overall lifecycle, admission, activities, health and unload.
No native UI failure may construct an independent unleased model.

The GPU child uses the original constructor: `load_denoiser=True`,
`optimize=False`, LoRA on LM and DiT but not projection. Do not change native
BF16 LM / FP32 AudioVAE placement or ZipEnhancer's native ModelScope GPU default.
The native LoRAManager keeps its CPU LRU of five state dictionaries. This is
existing model behavior, not an Agent-added capacity workaround. API requests
retain their explicit LoRA switch/disable semantics; the original UI shares the
current LoRA state because it has no separate selector. All GPU inference and
LoRA changes serialize in one child lock. CPU input/output audio handling keeps
the original preprocessing, silence trimming and loudness/EQ algorithms, using
the SDK audio helpers.

The `/generate` contract also supports unified output timing controls: `speed`
(default `1.0`) applies pitch-preserving time-stretch, while
`expected_duration` derives the required stretch factor and takes precedence when
present. Both controls run after silence trimming and loudness/EQ post-processing;
timing metadata is returned in response headers.

The vendored `funasr/` remains required: it is an HTTP compatibility client to
`asrpri-api`, not a local ASR model. Keep that source folder and existing assets
in the runtime image. There is no new local ASR or training allocation.

The base model directory is `/app/models/openbmb__VoxCPM2`, originally linked
from `/TTD-Data/voxcpm/models`. Preserve the existing shared models/cache/local
mounts. Main weights are 4,580,080,592 bytes and AudioVAE is 376,951,122 bytes;
7 GiB is an initial scheduling estimate requiring real model peak calibration,
including denoise/reference paths. The legacy idle process's 526 MiB observation
is not an inference footprint. No second replica or CPU offload is introduced.

`tests/hub_contract_test.py` and `tests/hub_fakes.py` run only CPU fake model
weights, using real native framework/audio/LoRA-cache behavior, SDK ProcessModel
IPC and Gradio queue/files. Mount source and SDK read-only into the fixed base
without GPU or production weight mounts, include the source root, `src/`, and
`tests/` in PYTHONPATH, and run the explicit pytest file. These tests do not
replace real API/UI/GPU and original-domain cutover acceptance.

## Timing release build

Build `deploy/hub/Dockerfile` with immutable `BASE_IMAGE` (the current managed VoxCPM image) and `SDK_IMAGE` (the existing Hub SDK artifact). Keep the original fusion dependencies, weights, placement and vendored FunASR shim in the base. The overlay copies `hub_runtime`, `src/voxcpm` and `api/lora_manager.py`; API/UI and native seed support must ship together. The Dockerfile verifies all native seed signatures without loading weights. Do not use the retired first-adoption script to replace an already managed service; use Hub's controlled model release coordinator.

API speed/expected_duration must be positive finite numbers; invalid values return 422 before runtime admission. Native UI validates timing before entering the managed generation callback. A changed timing stage does not change model device placement or precision.
