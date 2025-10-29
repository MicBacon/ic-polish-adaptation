FROM nvcr.io/nvidia/pytorch:24.07-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git git-lfs ffmpeg libgl1-mesa-glx \
    openjdk-11-jre-headless \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /workspace

ENV HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NCCL_ASYNC_ERROR_HANDLING=1

COPY . /workspace/

RUN pip install -U pip setuptools wheel && \
    pip install -Ur requirements.txt && \
    pip install flash-attn --no-build-isolation && \
    pip install --no-cache-dir "uvicorn[standard]" fastapi python-multipart


EXPOSE 7862
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7862"]