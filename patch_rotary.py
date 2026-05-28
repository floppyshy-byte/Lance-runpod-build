#!/usr/bin/env python3
"""Patch the Lance ViT to avoid flash-attn's Triton rotary embedding.

flash-attn's Triton JIT compiler segfaults (triton/compiler/code_generator.py
ast_to_ttir) when the Docker image is built for CUDA 12.6 but the GPU runs
CUDA 12.8+ or 13.0 drivers.

Instead of rewriting the rotary kernel, we redirect "flash_attention_2" to
Qwen2_5_VLVisionSdpaAttention which uses `apply_rotary_pos_emb_vision` — a
pure PyTorch implementation with proper broadcasting, already in the Lance
source. No Triton involved.
"""

VIT_PATH = "modeling/vit/qwen2_5_vl_vit.py"

with open(VIT_PATH) as f:
    src = f.read()

# Replace the ATTENTION_CLASSES dict to redirect flash_attention_2 → sdpa
src = src.replace(
    '"flash_attention_2": Qwen2_5_VLVisionFlashAttention2,',
    '"flash_attention_2": Qwen2_5_VLVisionSdpaAttention,  # was FlashAttention2, patched for CUDA compat',
)

with open(VIT_PATH, "w") as f:
    f.write(src)

print("Patched Lance ViT: flash_attention_2 -> SdpaAttention (no Triton)")
