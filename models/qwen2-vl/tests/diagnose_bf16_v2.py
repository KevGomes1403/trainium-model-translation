#!/usr/bin/env python3
"""Quick Neuron bf16 on CPU vs HF bf16 comparison."""

import sys, os, torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLProcessor, AutoConfig, Qwen2VLForConditionalGeneration

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

# ── Load HF model in bf16 ───────────────────────────────────────────────────
print("Loading HF vision model bf16...")
hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, dtype=torch.bfloat16, device_map="cpu"
)
hf_visual = hf_model.model.visual
hf_visual.eval()

# ── Load Neuron model in float32 then convert to bf16 ───────────────────────
print("Loading Neuron vision model...")
from neuronx_distributed_inference.models.config import NeuronConfig
from modeling_qwen2vl_vision_neuron import (
    Qwen2VLVisionInferenceConfig, NeuronQwen2VLVisionModel,
)

vision_neuron_config = NeuronConfig(
    tp_degree=1, torch_dtype="float32", on_cpu=True,
    fused_qkv=False, batch_size=1, seq_len=1024, buckets=[256, 1024],
)
hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
vc = hf_cfg.vision_config.to_dict() if hasattr(hf_cfg.vision_config, "to_dict") else vars(hf_cfg.vision_config)
vision_config = Qwen2VLVisionInferenceConfig(
    neuron_config=vision_neuron_config,
    embed_dim=vc.get("embed_dim", 1280), depth=vc.get("depth", 32),
    num_heads=vc.get("num_heads", 16), mlp_ratio=vc.get("mlp_ratio", 4),
    patch_size=vc.get("patch_size", 14), temporal_patch_size=vc.get("temporal_patch_size", 2),
    spatial_merge_size=vc.get("spatial_merge_size", 2), in_chans=vc.get("in_channels", 3),
    text_hidden_size=hf_cfg.text_config.hidden_size if hasattr(hf_cfg, "text_config") else 3584,
)

neuron_model = NeuronQwen2VLVisionModel(vision_config)

# Copy HF weights (converting from bf16 to f32 for loading, then convert model)
hf_sd = hf_visual.state_dict()
new_sd = {}
for hf_key, tensor in hf_sd.items():
    nk = hf_key
    if hf_key == "patch_embed.proj.weight":
        new_sd[nk] = tensor.float().reshape(1280, -1)
        continue
    nk = nk.replace("merger.mlp.0.", "merger.mlp_fc1.")
    nk = nk.replace("merger.mlp.2.", "merger.mlp_fc2.")
    if "rotary_pos_emb" in nk:
        continue
    new_sd[nk] = tensor.float()
neuron_model.load_state_dict(new_sd, strict=False)
neuron_model = neuron_model.to(torch.bfloat16)
neuron_model.eval()

# ── Compute RoPE ────────────────────────────────────────────────────────────
spatial_merge_size = 2
pos_ids = []
for t_val, h_val, w_val in grid_thw:
    t_v, h_v, w_v = int(t_val), int(h_val), int(w_val)
    hpos = torch.arange(h_v).unsqueeze(1).expand(-1, w_v).reshape(
        h_v//spatial_merge_size, spatial_merge_size,
        w_v//spatial_merge_size, spatial_merge_size
    ).permute(0,2,1,3).flatten()
    wpos = torch.arange(w_v).unsqueeze(0).expand(h_v, -1).reshape(
        h_v//spatial_merge_size, spatial_merge_size,
        w_v//spatial_merge_size, spatial_merge_size
    ).permute(0,2,1,3).flatten()
    pos_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t_v, 1))
pos_ids = torch.cat(pos_ids, dim=0)

head_dim, rope_dim = 80, 40
inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float) / rope_dim))
max_gs = grid_thw[:, 1:].max().item()
freqs = torch.outer(torch.arange(max_gs, dtype=torch.float), inv_freq)
rotary = freqs[pos_ids].flatten(1)
emb = torch.cat((rotary, rotary), dim=-1)
cos_rope = emb.cos().to(torch.bfloat16)
sin_rope = emb.sin().to(torch.bfloat16)

# ── Run both in bf16 ────────────────────────────────────────────────────────
pv_bf16 = pixel_values.to(torch.bfloat16)
total_patches = pv_bf16.shape[0]

# HF
hf_pos_emb = (cos_rope, sin_rope)  # Use same cos/sin

cu_seqlens = torch.repeat_interleave(
    grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
).cumsum(dim=0, dtype=torch.int32)
cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

# Neuron
mask = torch.zeros(total_patches, total_patches, dtype=torch.int32)
offset = 0
for t_val, h_val, w_val in grid_thw:
    n = t_val.item() * h_val.item() * w_val.item()
    mask[offset:offset+n, offset:offset+n] = 1
    offset += n
attn_mask = mask.unsqueeze(0).unsqueeze(0)

print("\n" + "=" * 70)
print("Layer-by-layer comparison in BFLOAT16")
print("=" * 70)

# Patch embedding
hf_h = hf_visual.patch_embed(pv_bf16)
neuron_h = neuron_model.patch_embed(pv_bf16.unsqueeze(0))

diff = (hf_h.float() - neuron_h.squeeze(0).float()).abs()
print(f"Patch embed diff: max={diff.max():.6f}, mean={diff.mean():.6f}")

# Block by block
with torch.no_grad():
    for i in range(32):
        hf_h = hf_visual.blocks[i](hf_h, cu_seqlens=cu_seqlens, position_embeddings=hf_pos_emb)
        neuron_h = neuron_model.blocks[i](neuron_h, attn_mask, cos=cos_rope, sin=sin_rope)

        diff = (hf_h.float() - neuron_h.squeeze(0).float()).abs()
        if i < 4 or i >= 28 or i % 8 == 0:
            print(f"Block {i:2d}: max_diff={diff.max():.2f}, mean_diff={diff.mean():.4f}, "
                  f"hf_std={hf_h.float().std():.2f}, neuron_std={neuron_h.float().std():.2f}")

    # Merger
    hf_merged = hf_visual.merger(hf_h)
    neuron_merged = neuron_model.merger(neuron_h)

print(f"\nAfter merger:")
print(f"  HF:     mean={hf_merged.float().mean():.6f}, std={hf_merged.float().std():.4f}")
print(f"  Neuron: mean={neuron_merged.float().mean():.6f}, std={neuron_merged.float().std():.4f}")
diff = (hf_merged.float() - neuron_merged.squeeze(0).float()).abs()
print(f"  Diff:   max={diff.max():.4f}, mean={diff.mean():.6f}")
