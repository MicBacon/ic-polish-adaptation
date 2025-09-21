FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git git-lfs curl ca-certificates \
    libgl1 libglib2.0-0 ffmpeg \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    ultralytics==8.3.202 \
    easyocr \
    transformers \
    accelerate \
    sentence-transformers \
    bitsandbytes \
    opencv-python-headless \
    pillow tqdm numpy scikit-image pyyaml requests psutil