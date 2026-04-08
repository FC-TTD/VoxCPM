# syntax=docker/dockerfile:1.7-labs
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VOXCPM=0.0.0 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs libsndfile1 ffmpeg curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages /opt/appenv
ENV VIRTUAL_ENV=/opt/appenv
ENV PATH="/opt/appenv/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install . \
    && pip install fastapi uvicorn soxr python-multipart \
    && pip install \
      --index-url http://pypi-server/simple/ \
      --trusted-host pypi-server \
      --extra-index-url https://pypi.org/simple \
      ttd-fastapi-utils

COPY app.py ./
COPY entry.py ./
COPY api ./api
COPY lora_ft_webui.py ./
COPY scripts ./scripts
COPY conf ./conf
COPY examples ./examples
COPY assets ./assets

RUN find /opt/appenv -type d -name "__pycache__" -prune -exec rm -rf {} + || true \
    && rm -rf /root/.cache/pip || true

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "entry:app", "--host", "0.0.0.0", "--port", "8000"]
