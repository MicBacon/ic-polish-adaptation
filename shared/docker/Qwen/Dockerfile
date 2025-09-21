FROM nvcr.io/nvidia/pytorch:24.07-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git git-lfs ffmpeg libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install \
    && openjdk-8-jre-headless

WORKDIR /workspace

ENV HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN pip install -U pip setuptools wheel && \
    pip install -U \
    "transformers>=4.43" \
    "accelerate>=0.33" \
    "peft>=0.11" \
    "datasets>=2.20" \
    "pycocoevalcap==1.2" \
    "bleu==0.3" \
    evaluate rouge_score nltk bert_score sacrebleu \
    open_clip_torch sentence-transformers \
    pillow sentencepiece "timm>=0.9" einops \
    qwen-vl-utils \
    flash-attn --no-build-isolation

COPY . /workspace/

CMD ["/bin/bash"]