#!/usr/bin/env python3
"""Layer-by-layer comparison of HF vs Neuron vision encoder on CPU."""

import sys, os, json, torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

# ── Load processor & prepare inputs ──────────────────────────────────────────
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLProcessor, AutoConfig

model_path = os.path.expanduser("~/models/qwen2-vl-7b")
processor = Qwen2VLProcessor.from_pretrained(model_path)
if hasattr(processor, "image_processor"):
    processor.image_processor.max_pixels = 200704
    processor.image_processor.min_pixels = 3136

image_path = os.path.join(os.path.dirname(__file__), "puppy.jpg")
messages = [{"role": "user", "content": [
    {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
    {"type": "text", "text": "Describe this image."},
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, _ = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, return_tensors="pt")

pixel_values = inputs["pixel_values"]  # [1008, 1176]
grid_thw = inputs["image_grid_thw"]    # [[1, 24, 42]]

print(f"pixel_values: {pixel_values.shape}, grid_thw: {grid_thw}")

# ── Load HF vision model ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Loading HF vision model on CPU...")
print("=" * 70)

from transformers import Qwen2VLForConditionalGeneration
hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float32, device_map="cpu"
)
hf_visual = hf_model.model.visual
hf_visual.eval()

# ── Load Neuron vision model on CPU ─────────────────────────────────────────
print("\n" + "=" * 70)
print("Loading Neuron vision model on CPU...")
print("=" * 70)

from neuronx_distributed_inference.models.config import NeuronConfig
from modeling_qwen2vl_vision_neuron import (
    Qwen2VLVisionInferenceConfig,
    NeuronQwen2VLVisionModel,
    Qwen2VLVisionModelWrapper,
)

vision_neuron_config = NeuronConfig(
    tp_degree=1,
    torch_dtype="float32",
    on_cpu=True,
    fused_qkv=False,
    batch_size=1,
    seq_len=1024,
    buckets=[256, 1024],
)

hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
vision_cfg_dict = hf_cfg.vision_config.to_dict() if hasattr(hf_cfg.vision_config, "to_dict") else vars(hf_cfg.vision_config)

vision_config = Qwen2VLVisionInferenceConfig(
    neuron_config=vision_neuron_config,
    embed_dim=vision_cfg_dict.get("embed_dim", 1280),
    depth=vision_cfg_dict.get("depth", 32),
    num_heads=vision_cfg_dict.get("num_heads", 16),
    mlp_ratio=vision_cfg_dict.get("mlp_ratio", 4),
    patch_size=vision_cfg_dict.get("patch_size", 14),
    temporal_patch_size=vision_cfg_dict.get("temporal_patch_size", 2),
    spatial_merge_size=vision_cfg_dict.get("spatial_merge_size", 2),
    in_chans=vision_cfg_dict.get("in_channels", 3),
    text_hidden_size=hf_cfg.text_config.hidden_size if hasattr(hf_cfg, "text_config") else 3584,
)

neuron_model = NeuronQwen2VLVisionModel(vision_config)
neuron_model.eval()

# ── Copy weights from HF to Neuron ──────────────────────────────────────────
print("\n" + "=" * 70)
print("Copying weights from HF to Neuron model...")
print("=" * 70)

hf_sd = hf_visual.state_dict()
neuron_sd = neuron_model.state_dict()

print(f"HF vision state dict keys: {len(hf_sd)}")
print(f"Neuron vision state dict keys: {len(neuron_sd)}")

# Build mapping
new_sd = {}
for hf_key, tensor in hf_sd.items():
    nk = hf_key
    # patch_embed: Conv3d -> Linear
    if hf_key == "patch_embed.proj.weight":
        new_sd[nk] = tensor.reshape(1280, -1).float()
        continue
    # merger mlp renaming
    nk = nk.replace("merger.mlp.0.", "merger.mlp_fc1.")
    nk = nk.replace("merger.mlp.2.", "merger.mlp_fc2.")
    # rotary_pos_emb is not in the Neuron model (computed on CPU)
    if "rotary_pos_emb" in nk:
        continue
    new_sd[nk] = tensor.float()

# Check for missing/extra keys
neuron_keys = set(neuron_sd.keys())
mapped_keys = set(new_sd.keys())
missing = neuron_keys - mapped_keys
extra = mapped_keys - neuron_keys

if missing:
    print(f"MISSING keys in Neuron model (not mapped from HF): {sorted(missing)}")
if extra:
    print(f"EXTRA keys (mapped but not in Neuron model): {sorted(extra)}")
if not missing and not extra:
    print("All keys match perfectly!")

# Load weights
result = neuron_model.load_state_dict(new_sd, strict=False)
print(f"Load result - missing: {result.missing_keys}, unexpected: {result.unexpected_keys}")

# ── Step-by-step comparison ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Comparing patch embedding output")
print("=" * 70)

pv_float = pixel_values.float()

# HF patch embed
hf_patch_out = hf_visual.patch_embed(pv_float)  # [1008, 1280]
print(f"HF patch_embed output: {hf_patch_out.shape}, mean={hf_patch_out.mean():.6f}, std={hf_patch_out.std():.6f}")

# Neuron patch embed
neuron_patch_in = pv_float.unsqueeze(0)  # [1, 1008, 1176]
neuron_patch_out = neuron_model.patch_embed(neuron_patch_in)  # [1, 1008, 1280]
print(f"Neuron patch_embed output: {neuron_patch_out.shape}, mean={neuron_patch_out.mean():.6f}, std={neuron_patch_out.std():.6f}")

diff = (hf_patch_out - neuron_patch_out.squeeze(0)).abs()
print(f"Diff: max={diff.max():.6f}, mean={diff.mean():.6f}")

# ── Compare RoPE computation ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Comparing RoPE computation")
print("=" * 70)

# HF RoPE
hf_rotary = hf_visual.rot_pos_emb(grid_thw)  # [1008, 40]
hf_emb = torch.cat((hf_rotary, hf_rotary), dim=-1)  # [1008, 80]
hf_cos = hf_emb.cos()
hf_sin = hf_emb.sin()
print(f"HF cos: {hf_cos.shape}, range=[{hf_cos.min():.4f}, {hf_cos.max():.4f}]")
print(f"HF sin: {hf_sin.shape}, range=[{hf_sin.min():.4f}, {hf_sin.max():.4f}]")

# Neuron RoPE (from patchify)
spatial_merge_size = 2
pos_ids = []
for t_val, h_val, w_val in grid_thw:
    t_v, h_v, w_v = int(t_val), int(h_val), int(w_val)
    hpos_ids = torch.arange(h_v).unsqueeze(1).expand(-1, w_v)
    hpos_ids = hpos_ids.reshape(
        h_v // spatial_merge_size, spatial_merge_size,
        w_v // spatial_merge_size, spatial_merge_size,
    ).permute(0, 2, 1, 3).flatten()
    wpos_ids = torch.arange(w_v).unsqueeze(0).expand(h_v, -1)
    wpos_ids = wpos_ids.reshape(
        h_v // spatial_merge_size, spatial_merge_size,
        w_v // spatial_merge_size, spatial_merge_size,
    ).permute(0, 2, 1, 3).flatten()
    pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t_v, 1))
pos_ids = torch.cat(pos_ids, dim=0)

head_dim = 80
rope_dim = head_dim // 2  # 40
rope_theta = 10000.0
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float) / rope_dim))
max_grid_size = grid_thw[:, 1:].max().item()
seq = torch.arange(max_grid_size, dtype=torch.float)
freqs = torch.outer(seq, inv_freq)
neuron_rotary = freqs[pos_ids].flatten(1)
neuron_emb = torch.cat((neuron_rotary, neuron_rotary), dim=-1)
neuron_cos = neuron_emb.cos()
neuron_sin = neuron_emb.sin()
print(f"\nNeuron cos: {neuron_cos.shape}, range=[{neuron_cos.min():.4f}, {neuron_cos.max():.4f}]")
print(f"Neuron sin: {neuron_sin.shape}, range=[{neuron_sin.min():.4f}, {neuron_sin.max():.4f}]")

rope_diff = (hf_cos - neuron_cos).abs()
print(f"\ncos diff: max={rope_diff.max():.8f}, mean={rope_diff.mean():.8f}")
rope_diff_sin = (hf_sin - neuron_sin).abs()
print(f"sin diff: max={rope_diff_sin.max():.8f}, mean={rope_diff_sin.mean():.8f}")

# ── Compare after block 0 ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Comparing after block 0")
print("=" * 70)

# HF: block 0 forward
# HF uses cu_seqlens for chunked attention (unbatched, shape [S, D])
hf_hidden = hf_patch_out  # [1008, 1280]
hf_position_embeddings = (hf_cos, hf_sin)

cu_seqlens = torch.repeat_interleave(
    grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
).cumsum(dim=0, dtype=torch.int32)
cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

with torch.no_grad():
    hf_after_block0 = hf_visual.blocks[0](
        hf_hidden, cu_seqlens=cu_seqlens, position_embeddings=hf_position_embeddings
    )
print(f"HF after block 0: {hf_after_block0.shape}, mean={hf_after_block0.mean():.6f}, std={hf_after_block0.std():.6f}")

# Neuron: block 0 forward
# Build attention mask (block-diagonal)
total_patches = neuron_patch_out.shape[1]
mask = torch.zeros(total_patches, total_patches, dtype=torch.int32)
offset = 0
for t_val, h_val, w_val in grid_thw:
    np_val = t_val.item() * h_val.item() * w_val.item()
    mask[offset:offset + np_val, offset:offset + np_val] = 1
    offset += np_val
attn_mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

with torch.no_grad():
    neuron_after_block0 = neuron_model.blocks[0](
        neuron_patch_out, attn_mask, cos=neuron_cos, sin=neuron_sin
    )
print(f"Neuron after block 0: {neuron_after_block0.shape}, mean={neuron_after_block0.mean():.6f}, std={neuron_after_block0.std():.6f}")

diff_b0 = (hf_after_block0 - neuron_after_block0.squeeze(0)).abs()
print(f"Diff after block 0: max={diff_b0.max():.6f}, mean={diff_b0.mean():.6f}")

# ── If block 0 differs, investigate attention sub-layer ──────────────────────
if diff_b0.mean() > 0.01:
    print("\n" + "=" * 70)
    print("Block 0 DIFFERS - investigating attention sub-layer in detail")
    print("=" * 70)

    # HF attention sub-layer (manual)
    hf_block0 = hf_visual.blocks[0]
    hf_normed = hf_block0.norm1(hf_hidden)  # [S, D]
    print(f"HF norm1 output: mean={hf_normed.mean():.6f}, std={hf_normed.std():.6f}")

    # HF QKV
    hf_attn = hf_block0.attn
    seq_length = hf_normed.shape[0]
    hf_qkv_out = hf_attn.qkv(hf_normed)  # [S, 3*D]
    hf_qkv_reshaped = hf_qkv_out.reshape(seq_length, 3, hf_attn.num_heads, -1).permute(1, 0, 2, 3)
    hf_q, hf_k, hf_v = hf_qkv_reshaped.unbind(0)  # each [S, H, Hd]
    print(f"HF Q before RoPE: mean={hf_q.mean():.6f}, std={hf_q.std():.6f}")

    # HF RoPE application
    from qwen2_vl_pytorch import apply_rotary_pos_emb_vision as hf_apply_rope
    hf_q_rot, hf_k_rot = hf_apply_rope(hf_q, hf_k, hf_cos, hf_sin)
    print(f"HF Q after RoPE: mean={hf_q_rot.mean():.6f}, std={hf_q_rot.std():.6f}")

    # Neuron attention sub-layer (manual)
    neuron_block0 = neuron_model.blocks[0]
    neuron_hidden = neuron_patch_out  # [1, S, D]
    neuron_normed = neuron_block0.norm1(neuron_hidden)  # [1, S, D]
    print(f"\nNeuron norm1 output: mean={neuron_normed.mean():.6f}, std={neuron_normed.std():.6f}")

    norm_diff = (hf_normed - neuron_normed.squeeze(0)).abs()
    print(f"Norm1 diff: max={norm_diff.max():.6f}, mean={norm_diff.mean():.6f}")

    # Neuron QKV
    neuron_attn = neuron_block0.attn
    B, S, D = neuron_normed.shape
    neuron_qkv_out = neuron_attn.qkv(neuron_normed)  # [1, S, 3*D]
    neuron_qkv_reshaped = neuron_qkv_out.reshape(B, S, 3, neuron_attn.num_heads, neuron_attn.head_dim)
    neuron_qkv_reshaped = neuron_qkv_reshaped.permute(2, 0, 3, 1, 4)  # [3, B, H, S, Hd]
    neuron_q, neuron_k, neuron_v = neuron_qkv_reshaped.unbind(0)  # each [B, H, S, Hd]
    print(f"Neuron Q before RoPE: mean={neuron_q.mean():.6f}, std={neuron_q.std():.6f}")

    q_diff = (hf_q.unsqueeze(0).transpose(1, 2) - neuron_q).abs()
    print(f"Q before RoPE diff (HF[S,H,Hd]→[1,H,S,Hd] vs Neuron[1,H,S,Hd]): max={q_diff.max():.6f}, mean={q_diff.mean():.6f}")

    # Neuron RoPE application
    from modeling_qwen2vl_vision_neuron import apply_rotary_pos_emb_vision as neuron_apply_rope
    neuron_q_rot, neuron_k_rot = neuron_apply_rope(neuron_q, neuron_k, neuron_cos, neuron_sin)
    print(f"Neuron Q after RoPE: mean={neuron_q_rot.mean():.6f}, std={neuron_q_rot.std():.6f}")

    # Compare Q after RoPE
    # HF Q after RoPE: [S, H, Hd] -> need to match Neuron: [1, H, S, Hd]
    hf_q_for_cmp = hf_q_rot.unsqueeze(0).transpose(1, 2)  # [1, H, S, Hd]
    q_rot_diff = (hf_q_for_cmp - neuron_q_rot).abs()
    print(f"Q after RoPE diff: max={q_rot_diff.max():.6f}, mean={q_rot_diff.mean():.6f}")

    # Compare full attention output
    # HF attention: uses chunked attention with cu_seqlens
    with torch.no_grad():
        hf_attn_out = hf_block0.attn(
            hf_normed, cu_seqlens=cu_seqlens, position_embeddings=hf_position_embeddings
        )
    print(f"\nHF attention output: mean={hf_attn_out.mean():.6f}, std={hf_attn_out.std():.6f}")

    with torch.no_grad():
        neuron_attn_out = neuron_block0.attn(
            neuron_normed, attn_mask, cos=neuron_cos, sin=neuron_sin
        )
    print(f"Neuron attention output: mean={neuron_attn_out.mean():.6f}, std={neuron_attn_out.std():.6f}")

    attn_diff = (hf_attn_out - neuron_attn_out.squeeze(0)).abs()
    print(f"Attention output diff: max={attn_diff.max():.6f}, mean={attn_diff.mean():.6f}")
else:
    print("Block 0 matches well, comparing full encoder output...")

    # Run all blocks through HF
    hf_hidden_all = hf_patch_out
    with torch.no_grad():
        for i, blk in enumerate(hf_visual.blocks):
            hf_hidden_all = blk(hf_hidden_all, cu_seqlens=cu_seqlens, position_embeddings=hf_position_embeddings)
    print(f"HF after all blocks: mean={hf_hidden_all.mean():.6f}, std={hf_hidden_all.std():.6f}")

    # Run all blocks through Neuron
    neuron_hidden_all = neuron_patch_out
    with torch.no_grad():
        for i, blk in enumerate(neuron_model.blocks):
            neuron_hidden_all = blk(neuron_hidden_all, attn_mask, cos=neuron_cos, sin=neuron_sin)
    print(f"Neuron after all blocks: mean={neuron_hidden_all.mean():.6f}, std={neuron_hidden_all.std():.6f}")

    diff_all = (hf_hidden_all - neuron_hidden_all.squeeze(0)).abs()
    print(f"Diff after all blocks: max={diff_all.max():.6f}, mean={diff_all.mean():.6f}")
