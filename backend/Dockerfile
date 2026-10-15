FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
# 默认构建 CPU 镜像，避免为无 CUDA 的容器安装整套 GPU 运行库。
# GPU 部署可在构建时显式指定对应的 PyTorch 安装源。
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url "${PYTORCH_INDEX_URL}" 'torch>=2.0.0' \
    && python -m pip install -r requirements.txt

COPY backend/ ./
RUN mkdir -p /app/runtime/sqlite /app/runtime/chroma /app/runtime/md5 /app/runtime/models /app/logs

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
