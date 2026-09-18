# syntax=docker/dockerfile:1
ARG BASE_IMAGE
ARG SDK_IMAGE
FROM ${SDK_IMAGE} AS runtime_package
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG SDK_IMAGE
ARG SOURCE_REV
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends sox \
    && rm -rf /var/lib/apt/lists/*
RUN /opt/appenv/bin/python -c 'import re,sys; assert all(re.fullmatch(r".+@sha256:[a-f0-9]{64}", v) for v in sys.argv[1:])' "${BASE_IMAGE}" "${SDK_IMAGE}"
COPY --link --from=runtime_package /opt/ttd-hub-runtime /opt/ttd-hub-runtime
ENV PYTHONPATH="/opt/ttd-hub-runtime:/app/src:${PYTHONPATH}"
COPY --link src/voxcpm /app/src/voxcpm
COPY --link api/lora_manager.py /app/api/lora_manager.py
COPY --link hub_runtime /app/hub_runtime
ENV TTD_WEBHOOK_URL="" GRADIO_ANALYTICS_ENABLED="False"
LABEL org.opencontainers.image.source="https://github.com/FC-TTD/VoxCPM" org.opencontainers.image.revision="${SOURCE_REV}"
# No model/weight loading: catch a runtime overlay paired with stale native code.
RUN /opt/appenv/bin/python -c 'import inspect; from voxcpm import VoxCPM; from voxcpm.model.voxcpm import VoxCPMModel; from voxcpm.model.voxcpm2 import VoxCPM2Model; assert "seed" in inspect.signature(VoxCPM._generate).parameters; assert all("seed" in inspect.signature(getattr(c, n)).parameters for c in (VoxCPMModel, VoxCPM2Model) for n in ("_generate", "_generate_with_prompt_cache"))'
RUN /opt/appenv/bin/python -m hub_runtime describe > /app/hub_runtime/contract.json
CMD ["bash", "-lc", "ln -sfn /TTD-Data/voxcpm/models /app/models && exec /opt/appenv/bin/python -m hub_runtime"]
