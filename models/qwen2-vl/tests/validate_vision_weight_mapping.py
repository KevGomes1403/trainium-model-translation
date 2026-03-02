#!/usr/bin/env python3
"""
Validate the vision weight mapping for Qwen2-VL NxDI translation.

Compares:
  1. HF checkpoint vision keys (visual.* from model.safetensors.index.json)
  2. convert_hf_to_neuron_state_dict output (vision_transformer.* after conversion)
  3. NxDI NeuronQwen2VLVisionModel state dict keys (actual model structure)

Reports mismatches between (2) and (3).
"""

import json
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/home/ubuntu/model-translation/qwen2-vl")

# ---------------------------------------------------------------------------
# Step 1: Read HF checkpoint keys and filter for vision keys
# ---------------------------------------------------------------------------
print("=" * 70)
print("Step 1: Read HF vision keys from checkpoint index")
print("=" * 70)

with open("/home/ubuntu/models/qwen2-vl-7b/model.safetensors.index.json") as f:
    index = json.load(f)

all_hf_keys = set(index["weight_map"].keys())
# Vision keys in HF checkpoint start with "visual."
hf_vision_keys = sorted(k for k in all_hf_keys if k.startswith("visual."))
print(f"Total HF keys: {len(all_hf_keys)}")
print(f"HF vision keys (visual.*): {len(hf_vision_keys)}")
for k in hf_vision_keys:
    print(f"  {k}")

# ---------------------------------------------------------------------------
# Step 2: Simulate the framework's prefix stripping
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("Step 2: After framework strips 'model.' prefix")
print("=" * 70)
# The framework strips "model." prefix. Vision keys don't have "model." prefix,
# so they arrive as-is: "visual.*"
# But lm_head.weight also has no "model." prefix. Let's just focus on visual.* keys.
arriving_vision_keys = sorted(k for k in hf_vision_keys)  # already "visual.*"
print(f"Vision keys arriving at convert function: {len(arriving_vision_keys)}")

# ---------------------------------------------------------------------------
# Step 3: Simulate convert_hf_to_neuron_state_dict on vision keys
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("Step 3: Simulate convert_hf_to_neuron_state_dict (vision portion)")
print("=" * 70)

# Create a fake state dict with dummy tensors for vision keys
fake_state_dict = {}
for k in arriving_vision_keys:
    if k == "visual.patch_embed.proj.weight":
        fake_state_dict[k] = torch.randn(1280, 3, 2, 14, 14)  # Conv3d shape
    elif "qkv.weight" in k:
        fake_state_dict[k] = torch.randn(3 * 1280, 1280)
    elif "qkv.bias" in k:
        fake_state_dict[k] = torch.randn(3 * 1280)
    elif "weight" in k:
        fake_state_dict[k] = torch.randn(10, 10)
    elif "bias" in k:
        fake_state_dict[k] = torch.randn(10)
    else:
        fake_state_dict[k] = torch.randn(10)

# Now run the vision portion of convert_hf_to_neuron_state_dict
PREFIX_IN = "visual."
PREFIX_OUT = "vision_transformer."

# patch_embed: reshape Conv3d weight [1280,3,2,14,14] -> [1280, 1176]
pe_key_in = "visual.patch_embed.proj.weight"
pe_key_out = "vision_transformer.patch_embed.proj.weight"
if pe_key_in in fake_state_dict:
    fake_state_dict[pe_key_out] = fake_state_dict.pop(pe_key_in).reshape(1280, -1)

# Collect remaining visual.* keys for bulk rename
vision_keys = [k for k in list(fake_state_dict.keys()) if k.startswith(PREFIX_IN)]
for key in vision_keys:
    new_key = key.replace(PREFIX_IN, PREFIX_OUT)
    new_key = new_key.replace("merger.mlp.0.", "merger.mlp_fc1.")
    new_key = new_key.replace("merger.mlp.2.", "merger.mlp_fc2.")
    fake_state_dict[new_key] = fake_state_dict.pop(key)

# Vision attention QKV / o_proj renames
vision_depth = 32
vision_tp = 1
embed_dim = 1280

for i in range(vision_depth):
    for suffix in ("weight", "bias"):
        fused_key = f"vision_transformer.blocks.{i}.attn.qkv.{suffix}"
        if fused_key in fake_state_dict:
            fused = fake_state_dict.pop(fused_key)
            chunks = fused.chunk(3, dim=0)
            fake_state_dict[f"vision_transformer.blocks.{i}.attn.qkv_proj.q_proj.{suffix}"] = chunks[0]
            fake_state_dict[f"vision_transformer.blocks.{i}.attn.qkv_proj.k_proj.{suffix}"] = chunks[1]
            fake_state_dict[f"vision_transformer.blocks.{i}.attn.qkv_proj.v_proj.{suffix}"] = chunks[2]

        old = f"vision_transformer.blocks.{i}.attn.proj.{suffix}"
        new = f"vision_transformer.blocks.{i}.attn.o_proj.o_proj.{suffix}"
        if old in fake_state_dict:
            fake_state_dict[new] = fake_state_dict.pop(old)

    fake_state_dict[f"vision_transformer.blocks.{i}.attn.rank_util.rank"] = torch.arange(
        vision_tp, dtype=torch.int32
    )

# Now extract only vision keys from the converted state dict
converted_vision_keys = sorted(k for k in fake_state_dict.keys() if k.startswith("vision_transformer."))
print(f"Converted vision keys (vision_transformer.*): {len(converted_vision_keys)}")
for k in converted_vision_keys:
    print(f"  {k}")

# ---------------------------------------------------------------------------
# Step 4: Instantiate NxDI vision model and get its state dict keys
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("Step 4: Instantiate NeuronQwen2VLVisionModel on CPU")
print("=" * 70)

from neuronx_distributed_inference.models.config import NeuronConfig, InferenceConfig
from modeling_qwen2vl_vision_neuron import (
    Qwen2VLVisionInferenceConfig,
    NeuronQwen2VLVisionModel,
)

vision_neuron_config = NeuronConfig(
    tp_degree=1,
    torch_dtype="bfloat16",
    on_cpu=True,
    fused_qkv=False,
    batch_size=1,
    seq_len=1024,
)

# Use depth=1 first to see the key pattern, then depth=32 for full comparison
vision_config = Qwen2VLVisionInferenceConfig(
    neuron_config=vision_neuron_config,
    embed_dim=1280,
    depth=1,  # just 1 layer to see the structure
    num_heads=16,
    mlp_ratio=4,
    patch_size=14,
    temporal_patch_size=2,
    spatial_merge_size=2,
    in_chans=3,
    hidden_size=3584,
    text_hidden_size=3584,
)

model_1layer = NeuronQwen2VLVisionModel(vision_config)
keys_1layer = sorted(model_1layer.state_dict().keys())
print(f"NxDI vision model keys (depth=1): {len(keys_1layer)}")
for k in keys_1layer:
    print(f"  {k}")

# Now with full 32 layers
print()
vision_config_full = Qwen2VLVisionInferenceConfig(
    neuron_config=vision_neuron_config,
    embed_dim=1280,
    depth=32,
    num_heads=16,
    mlp_ratio=4,
    patch_size=14,
    temporal_patch_size=2,
    spatial_merge_size=2,
    in_chans=3,
    hidden_size=3584,
    text_hidden_size=3584,
)

model_full = NeuronQwen2VLVisionModel(vision_config_full)
neuron_keys = set(model_full.state_dict().keys())
print(f"NxDI vision model keys (depth=32): {len(neuron_keys)}")

# ---------------------------------------------------------------------------
# Step 5: Compare converted keys vs. NxDI model keys
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("Step 5: Compare converted vision keys vs NxDI model state dict keys")
print("=" * 70)

converted_set = set(converted_vision_keys)

# The converted keys have "vision_transformer." prefix.
# The NxDI model keys do NOT have this prefix (they're relative to the model).
# Let's check both with and without prefix.

print()
print("--- Direct comparison (converted keys vs model keys) ---")
in_converted_not_model = sorted(converted_set - neuron_keys)
in_model_not_converted = sorted(neuron_keys - converted_set)

if in_converted_not_model:
    print(f"\nKeys in CONVERTED state dict but NOT in NxDI model ({len(in_converted_not_model)}):")
    for k in in_converted_not_model:
        print(f"  EXTRA: {k}")
else:
    print("\nNo extra keys in converted state dict. All converted keys exist in model.")

if in_model_not_converted:
    print(f"\nKeys in NxDI MODEL but NOT in converted state dict ({len(in_model_not_converted)}):")
    for k in in_model_not_converted:
        print(f"  MISSING: {k}")
else:
    print("\nNo missing keys. All model keys are covered by converted state dict.")

# --- Now try stripping "vision_transformer." prefix and compare ---
print()
print("--- Comparison after stripping 'vision_transformer.' prefix from converted keys ---")
stripped_converted = set()
for k in converted_set:
    if k.startswith("vision_transformer."):
        stripped_converted.add(k[len("vision_transformer."):])
    else:
        stripped_converted.add(k)

in_stripped_not_model = sorted(stripped_converted - neuron_keys)
in_model_not_stripped = sorted(neuron_keys - stripped_converted)

if in_stripped_not_model:
    print(f"\nStripped converted keys NOT in NxDI model ({len(in_stripped_not_model)}):")
    for k in in_stripped_not_model:
        print(f"  EXTRA: {k}")
else:
    print("\nAll stripped converted keys exist in model.")

if in_model_not_stripped:
    print(f"\nNxDI model keys NOT in stripped converted ({len(in_model_not_stripped)}):")
    for k in in_model_not_stripped:
        print(f"  MISSING: {k}")
else:
    print("\nAll model keys are covered.")

# ---------------------------------------------------------------------------
# Step 6: Summary of key structural observations
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("Step 6: Key structural analysis")
print("=" * 70)

# Check the block key pattern
print("\nConverter produces block keys like:")
block_keys_conv = [k for k in converted_set if "blocks." in k]
if block_keys_conv:
    # Show first few unique patterns
    patterns = set()
    for k in block_keys_conv:
        # Replace layer number with {i}
        import re
        p = re.sub(r'blocks\.\d+', 'blocks.{i}', k)
        patterns.add(p)
    for p in sorted(patterns):
        print(f"  {p}")

print("\nNxDI model has block keys like:")
block_keys_model = [k for k in neuron_keys if "blocks." in k]
if block_keys_model:
    patterns = set()
    for k in block_keys_model:
        import re
        p = re.sub(r'blocks\.\d+', 'blocks.{i}', k)
        patterns.add(p)
    for p in sorted(patterns):
        print(f"  {p}")

print()
print("=" * 70)
print("DONE")
print("=" * 70)
