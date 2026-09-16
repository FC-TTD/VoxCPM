# VoxCPM2 Hub adoption

2026-09-16 formal adoption and real acceptance completed. Canonical FC-TTD/VoxCPM (OpenBMB fork), local committed baseline `8d3237afbcf2d57ab3aa07534bc168292a51785e`. Six core entry/UI/API/LoRA/framework files were SHA256-matched with the live container and /opt/voxcpm source. The remote main/api revisions are older/different; this local source has not been pushed.

The original `voxcpm-fusion:latest` was preserved by its actual immutable config ID `sha256:40a1a55356d3d840e61fc7b02d769335c30af93aa519ce7cf2546af210837c10`, pushed directly from worker without rebuilding as `registry.ttd/voxcpm/fusion@sha256:72a2805c62fabd44ff25cb206882c536760d9344b869959156c5577b595b2044`. New image extends this fixed base, installs the SDK with no dependency upgrades, and adds hub_runtime only.

Original Python3.11.10, Gradio6.19/I18N, FastAPI0.138.1, Torch2.5/CUDA12.4 and Transformers5.12 stay intact. The original NVIDIA entrypoint and model-directory symlink setup are preserved. /TTD-Data/voxcpm, its cache and local directories keep their original mounts/modes. FunASR remains the original HTTP ASR shim; no new ASR model is installed.

Loader arguments remain `load_denoiser=True,optimize=False,LoRAConfig(enable_lm=True,enable_dit=True,enable_proj=False)`. Native LM BF16 GPU, AudioVAE FP32 GPU and ZipEnhancer GPU placement remain unchanged. Existing LoRA CPU state_dict LRU5 is retained inside the child; it is not a capacity-driven offload. API and UI use one child/lock and preserve original LoRA selection semantics. The old UI error fallback that loaded a separate model is removed.

Original API `/generate`, all Form parameters/control/LoRA/cfg/timesteps, reference/prompt audio handling and output/postprocessing are preserved. Original root redirect and `/gradio/` UI, I18N/theme/CSS, three callback names, audio upload/queue/download and ASR helper remain. API has no native speed/expected_duration contract; Gateway retains that capability boundary rather than inventing timing parameters.

Worker service `voxcpm` uses one actor, explicit three-GPU envelope with preferenceGPU1, private127.0.0.1:13918 and actor-isolated state. Initial budget7GiB based on model/AudioVAE weight sizes plus denoiser/workspace; actual denoise/reference peaks must be measured. `voxcpm-pool-validation` is temporary; business overlay takes original `voxcpm-api`, UI `/gradio/`. Existing Gateway ID remains `voxcpm` (火雀), preserving its separately tuned normalize=false/lufs=-18 defaults and reference/prompt pairing.

18 CPU tests in the fixed original image passed: original schema/defaults/error paths, actual CPU LoRA LRU behavior, constructor placement arguments, actual tensor device summary, real ProcessModel IPC/exit and Gradio6 HTTP queue/reference upload/48kHz WAV download. No weights loaded and no GPU used in these tests. Production publication must additionally verify real output, denoise/reference peaks, Hub unload/generation, original domain and Gateway usage.

Hub scripts/voxcpm_release.py coordinates additive catalog/node setup, preserving sibling actors/leases/policies. Legacy Compose `fusion` is stopped by verified full container ID and image ID with no forced timeout. Rollback keeps the retired Hub registration/token for reconciliation and starts the original container, with fixed original digest in persistent Compose; no catalog/journal reset or pull of latest.


正式运行版本 b74bee6 已通过原域名 API、原生 UI、实际卸载/重载及 Gateway 持久用量验收。当前持久预算 8.5 GiB，实际新租约已使用。完整固定镜像、设备实测和回退边界统一记录在 Hub `docs/proposals/model-compute-pool/worker-expansion-2026-09-16.md`；此目录的初期准备描述保留为实现基线，不能覆盖最终验收。源码仅本地提交，未 Git push。
