# Repository Notes

## FunASR Shim

- This repository intentionally vendors a local `funasr` compatibility shim under `funasr/`.
- Demo-facing ASR code should keep the upstream-style `AutoModel(...).generate(...)` call shape whenever possible.
- The local shim forwards `SenseVoiceSmall`-style ASR requests to the internal service `http://asrpri-api/api/v1/asr`.
- Do not add the external `funasr` package back into project dependencies for the demo path unless the user explicitly asks to restore the original local-model behavior.
- When changing Dockerfiles, deployment sync rules, or runtime entrypoints, be sure to include the `funasr/` folder, which transparently overrides the `funasr` import.

## Deployment Strategy

- The project's default unified deployment mode is the **fusion / docker** mode (API and WebUI combined) by using ./deploy.sh docker.
- Services and images should be consistently named or tagged around `voxcpm-fusion` rather than `voxcpm-demo`.
