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

RUN pip install -U pip setuptools wheel && \
    pip install -U \
    "transformers>=4.43,<4.46" \
    "accelerate>=0.33" \
    "peft>=0.11" \
    "datasets>=2.20" \
    evaluate rouge_score nltk bert_score sacrebleu \
    "pycocoevalcap==1.2" \
    open_clip_torch sentence-transformers \
    pillow sentencepiece "timm>=0.9" einops \
    qwen-vl-utils hf-transfer && \
    pip install flash-attn --no-build-isolation

COPY . /workspace/

CMD ["/bin/bash"]