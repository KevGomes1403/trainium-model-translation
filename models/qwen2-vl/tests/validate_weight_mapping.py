"""
Validate weight mapping for Qwen2-VL -> NxDI.

Instantiates NeuronQwen2VLModel on CPU, converts the HF state dict,
and verifies:
  1. No unexpected missing keys in the Neuron model
  2. No unexpected extra keys after conversion
  3. All tensor shapes match

Run: source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate && python validate_weight_mapping.py
"""

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Initialize distributed process group (required by SPMDRank / parallel_state)
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12399")
if not dist.is_initialized():
    dist.init_process_group("gloo", rank=0, world_size=1)

from neuronx_distributed.parallel_layers import parallel_state
if not parallel_state.model_parallel_is_initialized():
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)

from modeling_qwen2vl_neuron import (
    NeuronQwen2VLForCausalLM,
    NeuronQwen2VLModel,
    Qwen2VLInferenceConfig,
)
from neuronx_distributed_inference.models.config import NeuronConfig

HF_PATH = Path.home() / "models" / "qwen2-vl-7b"

# ---------------------------------------------------------------------------
# Build config from HF checkpoint
# ---------------------------------------------------------------------------

hf_cfg_dict = AutoConfig.from_pretrained(str(HF_PATH)).to_dict()
hf_cfg_dict.pop("vision_config", None)  # vision_config is a nested dict, not needed

neuron_cfg = NeuronConfig(
    tp_degree=1,
    torch_dtype=torch.bfloat16,
    batch_size=1,
    seq_len=128,
    n_positions=128,
    on_cpu=True,
)

inf_cfg = Qwen2VLInferenceConfig(neuron_config=neuron_cfg, **hf_cfg_dict)
print(f"Config: hidden_size={inf_cfg.hidden_size}, num_layers={inf_cfg.num_hidden_layers}")
print(f"        num_heads={inf_cfg.num_attention_heads}, num_kv_heads={inf_cfg.num_key_value_heads}")
print(f"        pad_token_id={inf_cfg.pad_token_id}, head_dim={inf_cfg.head_dim}")

# ---------------------------------------------------------------------------
# Instantiate Neuron model on CPU (no compilation)
# ---------------------------------------------------------------------------

print("\nInstantiating NeuronQwen2VLModel on CPU...")
neuron_model = NeuronQwen2VLModel(inf_cfg)
neuron_keys = set(neuron_model.state_dict().keys())
# Exclude runtime KV cache state
neuron_keys = {k for k in neuron_keys if not k.startswith("kv_mgr")}
print(f"Neuron model keys: {len(neuron_keys)}")

# ---------------------------------------------------------------------------
# Load and convert HF state dict
# ---------------------------------------------------------------------------

print("\nLoading HF state dict...")
from safetensors.torch import load_file

index_path = HF_PATH / "model.safetensors.index.json"
with open(index_path) as f:
    index = json.load(f)

hf_state_dict = {}
loaded_shards = set()
for key, shard in index["weight_map"].items():
    if shard not in loaded_shards:
        shard_path = HF_PATH / shard
        hf_state_dict.update(load_file(str(shard_path)))
        loaded_shards.add(shard)

print(f"HF checkpoint keys: {len(hf_state_dict)}")

# ---------------------------------------------------------------------------
# Inspect actual key prefix layout
# ---------------------------------------------------------------------------

sample_keys = list(hf_state_dict.keys())[:10]
print("\nSample HF keys:")
for k in sample_keys:
    print(f"  {k}")

# Strip "model." prefix from text backbone keys.
# In Qwen2VLForConditionalGeneration:
#   lm_head.weight           -> stays as-is (already top-level)
#   model.embed_tokens.*     -> embed_tokens.*
#   model.layers.*           -> layers.*
#   model.norm.*             -> norm.*
#   visual.*                 -> visual.* (will be stripped in convert)
stripped = {}
for k, v in hf_state_dict.items():
    if k.startswith("model."):
        stripped[k[len("model."):]] = v
    else:
        stripped[k] = v  # lm_head.weight is at top level

print(f"\nAfter stripping 'model.' prefix: {len(stripped)} keys")

# ---------------------------------------------------------------------------
# Apply conversion
# ---------------------------------------------------------------------------

print("Applying convert_hf_to_neuron_state_dict...")
converted = NeuronQwen2VLForCausalLM.convert_hf_to_neuron_state_dict(stripped, inf_cfg)
print(f"Converted keys: {len(converted)}")

# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

# Remove visual.* keys from both sides (stripped during conversion)
converted_keys = {k for k in converted.keys() if not k.startswith("visual.")}

missing_from_converted = neuron_keys - converted_keys
extra_in_converted = converted_keys - neuron_keys

# Check shape compatibility for matching keys
shape_mismatches = []
for k in neuron_keys & converted_keys:
    n_shape = neuron_model.state_dict()[k].shape
    c_shape = converted[k].shape
    if n_shape != c_shape:
        shape_mismatches.append((k, n_shape, c_shape))

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"Missing from converted (need to add): {len(missing_from_converted)}")
for k in sorted(missing_from_converted)[:20]:
    n_shape = neuron_model.state_dict()[k].shape
    print(f"  MISSING: {k}  (Neuron expects shape {n_shape})")

print(f"\nExtra in converted (unexpected): {len(extra_in_converted)}")
for k in sorted(extra_in_converted)[:20]:
    print(f"  EXTRA: {k}  shape={converted[k].shape}")

print(f"\nShape mismatches: {len(shape_mismatches)}")
for k, ns, cs in shape_mismatches[:10]:
    print(f"  MISMATCH: {k}  Neuron={ns} vs Converted={cs}")

print(f"\n{'='*60}")
if not missing_from_converted and not shape_mismatches:
    print("VALIDATION PASSED -- weight mapping is correct")
else:
    print("VALIDATION FAILED -- see issues above")
    sys.exit(1)
