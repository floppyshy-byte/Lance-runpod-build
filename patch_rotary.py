#!/usr/bin/env python3
"""Patch the Lance ViT source to use pure-PyTorch rotary embedding.

flash-attn's Triton implementation segfaults (triton/compiler/code_generator.py
ast_to_ttir) when the Docker image is built for CUDA 12.6 but the GPU runs
CUDA 12.8+ or 13.0 drivers.
"""
import re

VIT_PATH = "modeling/vit/qwen2_5_vl_vit.py"

with open(VIT_PATH) as f:
    src = f.read()

# Remove the flash-attn import — the try/except already sets
# apply_rotary_emb = None as a fallback
src = src.replace(
    "from flash_attn.layers.rotary import apply_rotary_emb",
    "apply_rotary_emb = None  # flash-attn patched out (Triton segfault on CUDA mismatch)",
)

# Replace apply_rotary_pos_emb_flashatt with pure-PyTorch version.
# Match from 'def apply_rotary_pos_emb_flashatt' to the next class/def/@/EOF.
pat = r"(def apply_rotary_pos_emb_flashatt\([^)]+\)[^:]*:).*?(?=\nclass |\ndef |\n@|\Z)"
replacement = r"""\1
    # Log shapes for debugging, then apply broadcasting-safe rotary.
    # The original flash-attn apply_rotary_emb handles complex broadcasting
    # that we need to match. Log first so we can see what's mismatched.
    import sys
    print(f'[rotary-debug] q={q.shape} k={k.shape} cos={cos.shape} sin={sin.shape}',
          file=sys.stderr, flush=True)

    # Standard rotary: x_rot = x*cos + rotate_half(x)*sin
    def _rotate_half(x):
        d = x.shape[-1]
        return torch.cat((-x[..., d // 2 :], x[..., : d // 2]), dim=-1)

    q_embed = (q.float() * cos + _rotate_half(q.float()) * sin).type_as(q)
    k_embed = (k.float() * cos + _rotate_half(k.float()) * sin).type_as(k)
    return q_embed, k_embed"""

src = re.sub(pat, replacement, src, flags=re.DOTALL)

with open(VIT_PATH, "w") as f:
    f.write(src)

print("Patched Lance ViT rotary embedding to pure-PyTorch")
