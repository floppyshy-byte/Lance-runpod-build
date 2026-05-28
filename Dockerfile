# =============================================================================
# Lance — RunPod Serverless Worker
# =============================================================================
# Model weights are NOT baked in. We rely on RunPod's cached model storage.
# Add this HF repo ID when creating the endpoint:
#   bytedance-research/Lance
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — compile Python deps
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ffmpeg libsndfile1 \
    python3.11 python3.11-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

ARG LANCE_COMMIT=df23c7438b9a71a6b9d335e09ddf7cfeff1f1a1d
RUN git clone https://github.com/bytedance/Lance.git . \
    && git checkout ${LANCE_COMMIT}

RUN rm -rf .git

# Patch Lance ViT to use pure-PyTorch rotary embedding instead of flash-attn's
# Triton implementation, which segfaults when the Docker image is built for
# CUDA 12.6 but the GPU runs CUDA 12.8+ / 13.0 drivers (triton/compiler crash).
RUN python3 -c "
vit_path = 'modeling/vit/qwen2_5_vl_vit.py'
with open(vit_path) as f:
    src = f.read()

# Remove the flash-attn import (the try/except sets apply_rotary_emb = None on failure)
src = src.replace(
    'from flash_attn.layers.rotary import apply_rotary_emb',
    'apply_rotary_emb = None  # flash-attn patched out (Triton segfault on mismatch CUDA)'
)

# Replace the apply_rotary_pos_emb_flashatt function with pure-PyTorch version.
# Use the function signature line to find and replace the whole body.
import re
pat = r'(def apply_rotary_pos_emb_flashatt\([^)]+\)[^:]*:).*?(?=\nclass |\ndef |\n@|\Z)'
replacement = r'''\1
    cos_half = cos.chunk(2, dim=-1)[0].contiguous()
    sin_half = sin.chunk(2, dim=-1)[0].contiguous()
    # Pure-PyTorch rotary: rotates last half of dims
    q_embed = (q.float() * cos_half + torch.cat((-q.float()[..., q.shape[-1]//2:], q.float()[..., :q.shape[-1]//2]), dim=-1) * sin_half).type_as(q)
    k_embed = (k.float() * cos_half + torch.cat((-k.float()[..., k.shape[-1]//2:], k.float()[..., :k.shape[-1]//2]), dim=-1) * sin_half).type_as(k)
    return q_embed, k_embed'''
src = re.sub(pat, replacement, src, flags=re.DOTALL)

with open(vit_path, 'w') as f:
    f.write(src)
print('Patched Lance ViT rotary embedding to pure-PyTorch')
"

# Create venv (uv pip install requires one by default)
RUN uv venv --python python3.11
ENV PATH="/app/.venv/bin:${PATH}"

# Install PyTorch with CUDA 12.6
RUN uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Install Lance requirements
RUN uv pip install -r requirements.txt

# Install wheel first (flash-attn build needs it)
RUN uv pip install wheel

# Install flash-attn (needs build isolation disabled)
RUN uv pip install flash-attn==2.8.3 --no-build-isolation

# Install runpod
RUN uv pip install runpod

# ---------------------------------------------------------------------------
# Stage 2: Runtime — lean image
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    python3.11 python3.11-venv python3.11-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local/bin/uv /root/.local/bin/uv
COPY --from=builder /app /app

WORKDIR /app

ENV PATH="/app/.venv/bin:/root/.local/bin:${PATH}"

# Environment defaults (override at runtime in RunPod console)
ENV LANCE_MODEL_BASE_DIR=/runpod-volume/checkpoints
ENV HF_HOME=/runpod-volume/huggingface-cache/hub
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache/hub
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV PYTHONFAULTHANDLER=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
ENV TORCH_COMPILE_DISABLE=1

# Copy setup helper and handler LAST so code changes don't invalidate dep layers.
COPY setup_models.py /app/setup_models.py
COPY handler.py /app/handler.py

CMD ["python3", "-X", "faulthandler", "-u", "/app/handler.py"]
