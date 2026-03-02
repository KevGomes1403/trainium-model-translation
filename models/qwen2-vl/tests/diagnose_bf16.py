#!/usr/bin/env python3
"""Test bfloat16 precision impact: HF vs Neuron in bf16 on CPU."""

import sys, os, torch
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

pixel_values = inputs["pixel_values"]
grid_thw = inputs["image_grid_thw"]

# ── Load HF model in bfloat16 ───────────────────────────────────────────────
print("=" * 70)
print("Loading HF vision model in BFLOAT16 on CPU...")
print("=" * 70)

from transformers import Qwen2VLForConditionalGeneration
hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, dtype=torch.bfloat16, device_map="cpu"
)
hf_visual = hf_model.model.visual
hf_visual.eval()

# Run HF vision encoder in bfloat16
pv_bf16 = pixel_values.to(torch.bfloat16)
with torch.no_grad():
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    hf_out = hf_visual(pv_bf16, grid_thw=grid_thw)
    if isinstance(hf_out, BaseModelOutputWithPooling):
        hf_merged = hf_out.pooler_output
    else:
        hf_merged = hf_out

print(f"HF bfloat16 output: shape={hf_merged.shape}, mean={hf_merged.float().mean():.6f}, std={hf_merged.float().std():.6f}")

# Run HF vision encoder in float32 for comparison
hf_model_f32 = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, dtype=torch.float32, device_map="cpu"
)
hf_visual_f32 = hf_model_f32.model.visual
hf_visual_f32.eval()

with torch.no_grad():
    hf_out_f32 = hf_visual_f32(pixel_values.float(), grid_thw=grid_thw)
    if isinstance(hf_out_f32, BaseModelOutputWithPooling):
        hf_merged_f32 = hf_out_f32.pooler_output
    else:
        hf_merged_f32 = hf_out_f32

print(f"HF float32 output: shape={hf_merged_f32.shape}, mean={hf_merged_f32.mean():.6f}, std={hf_merged_f32.std():.6f}")

diff_hf_precision = (hf_merged.float() - hf_merged_f32.float()).abs()
print(f"HF bf16 vs f32 diff: max={diff_hf_precision.max():.4f}, mean={diff_hf_precision.mean():.4f}")

# ── Load Neuron model in bfloat16 on CPU ─────────────────────────────────────
print("\n" + "=" * 70)
print("Loading Neuron vision model in BFLOAT16 on CPU...")
print("=" * 70)

from neuronx_distributed_inference.models.config import NeuronConfig
from modeling_qwen2vl_vision_neuron import (
    Qwen2VLVisionInferenceConfig,
    NeuronQwen2VLVisionModel,
    apply_rotary_pos_emb_vision,
)

vision_neuron_config = NeuronConfig(
    tp_degree=1,
    torch_dtype="bfloat16",
    on_cpu=True,
    fused_qkv=False,
    batch_size=1,
    seq_len=1024,
    buckets=[256, 1024],
)

hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
vc = hf_cfg.vision_config.to_dict() if hasattr(hf_cfg.vision_config, "to_dict") else vars(hf_cfg.vision_config)

vision_config = Qwen2VLVisionInferenceConfig(
    neuron_config=vision_neuron_config,
    embed_dim=vc.get("embed_dim", 1280),
    depth=vc.get("depth", 32),
    num_heads=vc.get("num_heads", 16),
    mlp_ratio=vc.get("mlp_ratio", 4),
    patch_size=vc.get("patch_size", 14),
    temporal_patch_size=vc.get("temporal_patch_size", 2),
    spatial_merge_size=vc.get("spatial_merge_size", 2),
    in_chans=vc.get("in_channels", 3),
    text_hidden_size=hf_cfg.text_config.hidden_size if hasattr(hf_cfg, "text_config") else 3584,
)

neuron_model = NeuronQwen2VLVisionModel(vision_config)
neuron_model.eval()

# Copy weights from HF bfloat16 model
hf_sd = hf_visual.state_dict()
new_sd = {}
for hf_key, tensor in hf_sd.items():
    nk = hf_key
    if hf_key == "patch_embed.proj.weight":
        new_sd[nk] = tensor.reshape(1280, -1)
        continue
    nk = nk.replace("merger.mlp.0.", "merger.mlp_fc1.")
    nk = nk.replace("merger.mlp.2.", "merger.mlp_fc2.")
    if "rotary_pos_emb" in nk:
        continue
    new_sd[nk] = tensor
result = neuron_model.load_state_dict(new_sd, strict=False)
print(f"Load result - missing: {result.missing_keys}, unexpected: {result.unexpected_keys}")

# Compute RoPE
spatial_merge_size = 2
pos_ids = []
for t_val, h_val, w_val in grid_thw:
    t_v, h_v, w_v = int(t_val), int(h_val), int(w_val)
    hpos_ids = torch.arange(h_v).unsqueeze(1).expand(-1, w_v)
    hpos_ids = hpos_ids.reshape(h_v // spatial_merge_size, spatial_merge_size,
                                w_v // spatial_merge_size, spatial_merge_size).permute(0, 2, 1, 3).flatten()
    wpos_ids = torch.arange(w_v).unsqueeze(0).expand(h_v, -1)
    wpos_ids = wpos_ids.reshape(h_v // spatial_merge_size, spatial_merge_size,
                                w_v // spatial_merge_size, spatial_merge_size).permute(0, 2, 1, 3).flatten()
    pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t_v, 1))
pos_ids = torch.cat(pos_ids, dim=0)

head_dim = 80
rope_dim = head_dim // 2
inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float) / rope_dim))
max_grid_size = grid_thw[:, 1:].max().item()
seq = torch.arange(max_grid_size, dtype=torch.float)
freqs = torch.outer(seq, inv_freq)
rotary = freqs[pos_ids].flatten(1)
emb = torch.cat((rotary, rotary), dim=-1)
cos_rope = emb.cos().to(torch.bfloat16)
sin_rope = emb.sin().to(torch.bfloat16)

# Run Neuron model on CPU in bfloat16
patch_input = pv_bf16.unsqueeze(0)  # [1, 1008, 1176]
total_patches = patch_input.shape[1]

mask = torch.zeros(total_patches, total_patches, dtype=torch.int32)
offset = 0
for t_val, h_val, w_val in grid_thw:
    np_val = t_val.item() * h_val.item() * w_val.item()
    mask[offset:offset + np_val, offset:offset + np_val] = 1
    offset += np_val
attn_mask = mask.unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    neuron_out = neuron_model(patch_input, attn_mask, cos_rope, sin_rope)

print(f"Neuron bf16 output: shape={neuron_out.shape}, mean={neuron_out.float().mean():.6f}, std={neuron_out.float().std():.6f}")

diff_neuron_hf = (neuron_out.squeeze(0).float() - hf_merged.float()).abs()
print(f"Neuron bf16 vs HF bf16 diff: max={diff_neuron_hf.max():.4f}, mean={diff_neuron_hf.mean():.4f}")

# ── Test with float32 upcast in RoPE ────────────────────────────────────────
print("\n" + "=" * 70)
print("Testing float32 upcast effect in RoPE (block 0 only)")
print("=" * 70)

# Get the hidden state after patch embedding
with torch.no_grad():
    h_bf16 = neuron_model.patch_embed(patch_input)
    h_normed = neuron_model.blocks[0].norm1(h_bf16)

    # QKV
    attn = neuron_model.blocks[0].attn
    B, S, D = h_normed.shape
    qkv_out = attn.qkv(h_normed)
    qkv_r = qkv_out.reshape(B, S, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv_r.unbind(0)

    # RoPE without float32 upcast (current)
    q_nocast, k_nocast = apply_rotary_pos_emb_vision(q, k, cos_rope, sin_rope)

    # RoPE WITH float32 upcast (matching HF)
    orig_q_dtype = q.dtype
    q_f32, k_f32 = q.float(), k.float()
    cos_f32 = cos_rope[None, None, :, :].float()
    sin_f32 = sin_rope[None, None, :, :].float()
    from modeling_qwen2vl_vision_neuron import rotate_half
    q_cast = ((q_f32 * cos_f32) + (rotate_half(q_f32) * sin_f32)).to(orig_q_dtype)
    k_cast = ((k_f32 * cos_f32) + (rotate_half(k_f32) * sin_f32)).to(orig_q_dtype)

    diff_cast = (q_nocast.float() - q_cast.float()).abs()
    print(f"RoPE Q diff (no-cast vs f32-cast): max={diff_cast.max():.6f}, mean={diff_cast.mean():.6f}")
    print(f"Q nocast range: [{q_nocast.float().min():.4f}, {q_nocast.float().max():.4f}]")
    print(f"Q cast   range: [{q_cast.float().min():.4f}, {q_cast.float().max():.4f}]")

# ── Layer-by-layer diff accumulation in bfloat16 ────────────────────────────
print("\n" + "=" * 70)
print("Layer-by-layer diff accumulation (HF bf16 vs Neuron bf16)")
print("=" * 70)

# HF block-by-block
hf_hidden = hf_visual.patch_embed(pv_bf16)
hf_rotary_emb = hf_visual.rot_pos_emb(grid_thw)
hf_emb = torch.cat((hf_rotary_emb, hf_rotary_emb), dim=-1)
hf_pos_emb = (hf_emb.cos(), hf_emb.sin())

cu_seqlens = torch.repeat_interleave(
    grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
).cumsum(dim=0, dtype=torch.int32)
cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

# Neuron block-by-block
neuron_hidden = neuron_model.patch_embed(patch_input)

with torch.no_grad():
    for i in range(min(32, len(hf_visual.blocks))):
        hf_hidden = hf_visual.blocks[i](hf_hidden, cu_seqlens=cu_seqlens, position_embeddings=hf_pos_emb)
        neuron_hidden = neuron_model.blocks[i](neuron_hidden, attn_mask, cos=cos_rope, sin=sin_rope)
        
        diff = (hf_hidden.float() - neuron_hidden.squeeze(0).float()).abs()
        if i < 4 or i >= 28 or i % 8 == 0:
            print(f"  Block {i:2d}: max_diff={diff.max():.4f}, mean_diff={diff.mean():.6f}, "
                  f"hf_std={hf_hidden.float().std():.4f}, neuron_std={neuron_hidden.float().std():.4f}")

# After merger
hf_merged_manual = hf_visual.merger(hf_hidden)
neuron_merged_manual = neuron_model.merger(neuron_hidden)

print(f"\nAfter merger:")
print(f"  HF:     shape={hf_merged_manual.shape}, mean={hf_merged_manual.float().mean():.6f}, std={hf_merged_manual.float().std():.6f}")
print(f"  Neuron: shape={neuron_merged_manual.shape}, mean={neuron_merged_manual.float().mean():.6f}, std={neuron_merged_manual.float().std():.6f}")
diff_merged = (hf_merged_manual.float() - neuron_merged_manual.squeeze(0).float()).abs()
print(f"  Diff:   max={diff_merged.max():.4f}, mean={diff_merged.mean():.6f}")
