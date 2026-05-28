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
    cos_half = cos.chunk(2, dim=-1)[0].contiguous()
    sin_half = sin.chunk(2, dim=-1)[0].contiguous()
    # Expand half-dim cos/sin to full dim for element-wise multiplication
    cos_full = torch.repeat_interleave(cos_half, 2, dim=-1)
    sin_full = torch.repeat_interleave(sin_half, 2, dim=-1)
    # Pure-PyTorch rotary: x_rot = x*cos + rotate_half(x)*sin
    q_embed = (
        q.float() * cos_full
        + torch.cat(
            (-q.float()[..., q.shape[-1] // 2 :], q.float()[..., : q.shape[-1] // 2]),
            dim=-1,
        )
        * sin_full
    ).type_as(q)
    k_embed = (
        k.float() * cos_full
        + torch.cat(
            (-k.float()[..., k.shape[-1] // 2 :], k.float()[..., : k.shape[-1] // 2]),
            dim=-1,
        )
        * sin_full
    ).type_as(k)
    return q_embed, k_embed"""

src = re.sub(pat, replacement, src, flags=re.DOTALL)

with open(VIT_PATH, "w") as f:
    f.write(src)

print("Patched Lance ViT rotary embedding to pure-PyTorch")
