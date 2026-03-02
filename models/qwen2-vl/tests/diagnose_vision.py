#!/usr/bin/env python3
"""Diagnose vision encoder issues by comparing HF vs Neuron on CPU."""

import sys, os, torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

# ── Step 1: Check pixel_values shape from processor ──────────────────────────
print("=" * 70)
print("Step 1: Check pixel_values shape from processor")
print("=" * 70)

from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLProcessor

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

print(f"pixel_values shape: {pixel_values.shape}")
print(f"pixel_values dtype: {pixel_values.dtype}")
print(f"grid_thw: {grid_thw}")

t, h, w = grid_thw[0].tolist()
num_patches = t * h * w
print(f"num_patches (t*h*w): {num_patches}")
print(f"Expected input_dim (3*2*14*14): {3 * 2 * 14 * 14}")

# ── Step 2: Compare Conv3d vs Linear patch embedding ─────────────────────────
print()
print("=" * 70)
print("Step 2: Compare Conv3d vs Linear patch embedding")
print("=" * 70)

from safetensors import safe_open
import json

with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
    index = json.load(f)
pe_file = index["weight_map"]["visual.patch_embed.proj.weight"]
pe_path = os.path.join(model_path, pe_file)

with safe_open(pe_path, framework="pt", device="cpu") as f:
    conv_weight = f.get_tensor("visual.patch_embed.proj.weight")

print(f"Conv3d weight shape: {conv_weight.shape}")  # [1280, 3, 2, 14, 14]

# Conv3d approach (HF reference)
conv = nn.Conv3d(3, 1280, kernel_size=[2, 14, 14], stride=[2, 14, 14], bias=False)
conv.weight.data = conv_weight.float()

pv_float = pixel_values.float()
print(f"pixel_values for Conv3d input: shape={pv_float.shape}")
conv_input = pv_float.view(-1, 3, 2, 14, 14)
print(f"Conv3d input after view: {conv_input.shape}")
conv_output = conv(conv_input).view(-1, 1280)
print(f"Conv3d output: {conv_output.shape}, mean={conv_output.mean():.6f}, std={conv_output.std():.6f}")

# Linear approach (Neuron model - current implementation: direct reshape)
linear_weight = conv_weight.float().reshape(1280, -1)
linear = nn.Linear(1176, 1280, bias=False)
linear.weight.data = linear_weight

# Method A: Direct reshape (current Neuron implementation)
linear_input_a = pv_float.reshape(-1, 1176)
print(f"\nLinear input (direct reshape): {linear_input_a.shape}")
linear_output_a = linear(linear_input_a)
print(f"Linear output (direct reshape): mean={linear_output_a.mean():.6f}, std={linear_output_a.std():.6f}")

diff_a = (conv_output - linear_output_a).abs()
print(f"Diff (Conv3d vs Linear direct reshape): max={diff_a.max():.6f}, mean={diff_a.mean():.6f}")

# Method B: Proper reshape through [N, 3, 2, 14, 14] then flatten correctly
conv_input_flat = conv_input.reshape(-1, 1176)  # Same ordering as Conv3d
linear_output_b = linear(conv_input_flat)
print(f"\nLinear output (via Conv3d view then flatten): mean={linear_output_b.mean():.6f}, std={linear_output_b.std():.6f}")

diff_b = (conv_output - linear_output_b).abs()
print(f"Diff (Conv3d vs Linear correct flatten): max={diff_b.max():.6f}, mean={diff_b.mean():.6f}")

if diff_a.mean() > 0.01 and diff_b.mean() < 0.01:
    print("\n*** CONFIRMED: Direct reshape gives WRONG results! ***")
    print("*** The pixel_values must be viewed as [-1, 3, 2, 14, 14] first, ***")
    print("*** then flattened to [-1, 1176] for the linear projection.      ***")
elif diff_a.mean() < 0.01:
    print("\n*** Direct reshape is correct (processor may output compatible format) ***")
else:
    print(f"\n*** Both methods have significant error - investigate further ***")

# ── Step 3: Show the correct patchify approach ───────────────────────────────
print()
print("=" * 70)
print("Step 3: Correct patchify approach")
print("=" * 70)
# The fix: view pixel_values as [-1, C, T, H, W] then flatten to [-1, C*T*H*W]
correct_patches = pv_float.view(-1, 3, 2, 14, 14).reshape(-1, 1176)
wrong_patches = pv_float.reshape(-1, 1176)

print(f"First 10 elements of correct_patches[0]: {correct_patches[0, :10].tolist()}")
print(f"First 10 elements of wrong_patches[0]:   {wrong_patches[0, :10].tolist()}")
print(f"Are they the same? {torch.allclose(correct_patches, wrong_patches)}")

if not torch.allclose(correct_patches, wrong_patches):
    print(f"Element-wise max diff: {(correct_patches - wrong_patches).abs().max():.6f}")
    matching = (correct_patches == wrong_patches).float().mean()
    print(f"Fraction of elements matching: {matching:.4f}")
