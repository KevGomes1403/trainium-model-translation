#!/usr/bin/env python3
"""Verify that the vision encoder weights on Trainium match expected values."""

import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))

from neuronx_distributed_inference.models.config import NeuronConfig
from transformers import AutoConfig

model_path = os.path.expanduser("~/models/qwen2-vl-7b")

# Load the config
from modeling_qwen2vl_neuron import NeuronQwen2VLForConditionalGeneration

# Simulate what the framework does: load and convert state dict
hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

# Step 1: Load raw HF state dict (same as get_state_dict)
from neuronx_distributed_inference.modules.checkpoint import load_state_dict
raw_sd = load_state_dict(model_path)
print(f"Raw HF state dict: {len(raw_sd)} keys")

# Step 2: Strip "model." prefix (same as get_state_dict)
_STATE_DICT_MODEL_PREFIX = "model."
param_name_list = list(raw_sd.keys())
for param_name in param_name_list:
    if param_name.startswith(_STATE_DICT_MODEL_PREFIX):
        updated = param_name.replace(_STATE_DICT_MODEL_PREFIX, "", 1)
        raw_sd[updated] = raw_sd.pop(param_name)

# Step 3: Apply convert_hf_to_neuron_state_dict
from modeling_qwen2vl_neuron import Qwen2VLInferenceConfig as NeuronQwen2VLInferenceConfig

# Create a minimal config for the conversion
text_neuron_config = NeuronConfig(
    tp_degree=8, torch_dtype="bfloat16", batch_size=1, seq_len=1024,
    on_cpu=False, fused_qkv=True,
)
vision_neuron_config = NeuronConfig(
    tp_degree=8, torch_dtype="bfloat16", batch_size=1, seq_len=1024,
    on_cpu=False, fused_qkv=False,
)
nxd_config = NeuronQwen2VLInferenceConfig.from_pretrained(
    model_path=model_path,
    text_neuron_config=text_neuron_config,
    vision_neuron_config=vision_neuron_config,
    max_context_length=1024,
    max_new_tokens=32,
    max_length=1056,
    buckets=[256, 1024],
    output_logits=False,
    on_device_sampling_config={"do_sample": False, "top_k": 1},
)

converted_sd = NeuronQwen2VLForConditionalGeneration.convert_hf_to_neuron_state_dict(raw_sd, nxd_config)

# Step 4: Filter for vision keys
vision_keys = sorted([k for k in converted_sd if k.startswith(("patch_embed.", "blocks.", "merger."))])
text_keys = sorted([k for k in converted_sd if not k.startswith(("patch_embed.", "blocks.", "merger."))])

print(f"\nVision keys: {len(vision_keys)}")
print(f"Text keys: {len(text_keys)}")
print(f"\nFirst 10 vision keys:")
for k in vision_keys[:10]:
    t = converted_sd[k]
    print(f"  {k}: shape={t.shape}, dtype={t.dtype}, mean={t.float().mean():.6f}, std={t.float().std():.6f}")

print(f"\nLast 10 vision keys:")
for k in vision_keys[-10:]:
    t = converted_sd[k]
    print(f"  {k}: shape={t.shape}, dtype={t.dtype}, mean={t.float().mean():.6f}, std={t.float().std():.6f}")

# Step 5: Now load into CPU Neuron vision model and verify
from modeling_qwen2vl_vision_neuron import (
    Qwen2VLVisionInferenceConfig, NeuronQwen2VLVisionModel,
)

# Cast to bfloat16 (same as _cast_helper in checkpoint_loader_fn)
for key in vision_keys:
    t = converted_sd[key]
    if torch.is_floating_point(t) and t.dtype != torch.bfloat16:
        converted_sd[key] = t.to(torch.bfloat16)

vision_neuron_config_cpu = NeuronConfig(
    tp_degree=1, torch_dtype="bfloat16", on_cpu=True,
    fused_qkv=False, batch_size=1, seq_len=1024, buckets=[256, 1024],
)
vc = hf_cfg.vision_config.to_dict() if hasattr(hf_cfg.vision_config, "to_dict") else vars(hf_cfg.vision_config)
vision_config = Qwen2VLVisionInferenceConfig(
    neuron_config=vision_neuron_config_cpu,
    embed_dim=vc.get("embed_dim", 1280), depth=vc.get("depth", 32),
    num_heads=vc.get("num_heads", 16), mlp_ratio=vc.get("mlp_ratio", 4),
    patch_size=vc.get("patch_size", 14), temporal_patch_size=vc.get("temporal_patch_size", 2),
    spatial_merge_size=vc.get("spatial_merge_size", 2), in_chans=vc.get("in_channels", 3),
    text_hidden_size=hf_cfg.text_config.hidden_size if hasattr(hf_cfg, "text_config") else 3584,
)

neuron_model = NeuronQwen2VLVisionModel(vision_config)

# Load ONLY vision weights
vision_sd = {k: converted_sd[k] for k in vision_keys}
result = neuron_model.load_state_dict(vision_sd, strict=False)
if result.missing_keys:
    print(f"\nMISSING KEYS: {result.missing_keys}")
if result.unexpected_keys:
    print(f"\nUNEXPECTED KEYS: {result.unexpected_keys}")
else:
    print(f"\nAll vision weights loaded successfully!")

# Compute checksum
print("\n" + "=" * 70)
print("Weight checksums (vision model)")
print("=" * 70)
total_checksum = 0
for name, param in neuron_model.named_parameters():
    checksum = param.float().sum().item()
    total_checksum += checksum
    if "blocks.0." in name or "patch_embed" in name or "merger" in name:
        print(f"  {name}: shape={param.shape}, sum={checksum:.4f}, mean={param.float().mean():.6f}")

print(f"\nTotal weight checksum: {total_checksum:.4f}")
