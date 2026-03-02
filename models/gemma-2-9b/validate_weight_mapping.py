"""
Validation script for Gemma-2-9B NxDI weight mapping.

Steps:
  1. Diff HF vs Neuron state dict key patterns.
  2. Run convert_hf_to_neuron_state_dict on a real (1-layer) HF state dict.
  3. Assert no missing keys.
"""

import sys, json, re, os
import torch
import torch.distributed as dist

sys.path.insert(0, "/home/ubuntu/model-translation/gemma-2-9b")

# Initialize distributed process group (required by SPMDRank / parallel_state)
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12399")
if not dist.is_initialized():
    dist.init_process_group("gloo", rank=0, world_size=1)

from neuronx_distributed.parallel_layers import parallel_state
if not parallel_state.model_parallel_is_initialized():
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)

from transformers import AutoConfig
from neuronx_distributed_inference.models.config import NeuronConfig
from modeling_gemma2_neuron import (
    Gemma2InferenceConfig, NeuronGemma2ForCausalLM, NeuronGemma2Model
)

model_path = "/home/ubuntu/models/gemma-2-9b"

# ---------------------------------------------------------------------------
# Step 1: Diff state dict key patterns
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Diff HF vs Neuron key patterns")
print("=" * 70)

with open(f"{model_path}/model.safetensors.index.json") as f:
    index_data = json.load(f)
    weight_map = index_data["weight_map"]

hf_keys_raw = set(weight_map.keys())
# Strip "model." prefix (as NxDI framework does)
hf_keys = {k[len("model."):] if k.startswith("model.") else k for k in hf_keys_raw}

print(f"Total HF keys (after prefix strip): {len(hf_keys)}")

# Build config (pass HF config attributes as **kwargs per InferenceConfig signature)
hf_cfg = AutoConfig.from_pretrained(model_path)
neuron_cfg = NeuronConfig(
    tp_degree=1,
    torch_dtype="bfloat16",
    batch_size=1,
    seq_len=128,
    n_active_tokens=1,
    max_batch_size=1,
    buckets=[128],
    on_cpu=True,
)
hf_cfg_dict = hf_cfg.to_dict()
hf_cfg_dict["num_hidden_layers"] = 1   # key structure is layer-invariant
inf_cfg = Gemma2InferenceConfig(neuron_config=neuron_cfg, **hf_cfg_dict)

print("Instantiating 1-layer NeuronGemma2Model...")
neuron_model = NeuronGemma2Model(inf_cfg)
neuron_keys = set(neuron_model.state_dict().keys())
print(f"Total Neuron keys (1 layer): {len(neuron_keys)}")

print("\nAll Neuron state dict keys:")
for k in sorted(neuron_keys):
    print(f"  {k}")

def strip_layer(k):
    return re.sub(r"layers\.\d+\.", "layers.N.", k)

hf_patterns    = {strip_layer(k) for k in hf_keys}
neuron_patterns = {strip_layer(k) for k in neuron_keys}

only_in_hf     = hf_patterns - neuron_patterns
only_in_neuron = neuron_patterns - hf_patterns

print("\nIn HF, not Neuron (needs rename/drop/injection):")
for k in sorted(only_in_hf):
    print(f"  {k}")

print("\nIn Neuron, not HF (needs injection/creation by converter):")
for k in sorted(only_in_neuron):
    print(f"  {k}")

# ---------------------------------------------------------------------------
# Step 2: Run conversion on real layer-0 weights
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Run convert_hf_to_neuron_state_dict")
print("=" * 70)

from safetensors import safe_open

fake_hf_sd = {}

# Load shard 1 (contains embed_tokens and first few layers)
shard1 = f"{model_path}/model-00001-of-00008.safetensors"
print(f"Loading shard 1: {shard1}")
with safe_open(shard1, framework="pt", device="cpu") as f:
    for key in f.keys():
        stripped = key[len("model."):] if key.startswith("model.") else key
        fake_hf_sd[stripped] = f.get_tensor(key)

print(f"Keys loaded from shard 1: {len(fake_hf_sd)}")

# For remaining HF keys not loaded, create zero tensors with shape (1,)
for key in hf_keys:
    if key not in fake_hf_sd:
        fake_hf_sd[key] = torch.zeros(1)

# Restrict to layer 0 + global keys (no layers.1+)
layer0_keys = {
    k: v for k, v in fake_hf_sd.items()
    if not k.startswith("layers.") or k.startswith("layers.0.")
}

print(f"\nKeys passed to converter (layer 0 + globals): {len(layer0_keys)}")
for k in sorted(layer0_keys.keys()):
    v = layer0_keys[k]
    shape = v.shape if hasattr(v, "shape") else "?"
    print(f"  {k}: {shape}")

print("\nRunning convert_hf_to_neuron_state_dict...")
converted = NeuronGemma2ForCausalLM.convert_hf_to_neuron_state_dict(
    dict(layer0_keys), inf_cfg
)

# Also run update_state_dict_for_tied_weights (simulates framework behaviour)
NeuronGemma2ForCausalLM.update_state_dict_for_tied_weights(converted)

print("\nConverted keys (layer 0 + globals):")
for k in sorted(converted.keys()):
    v = converted[k]
    shape = v.shape if hasattr(v, "shape") else type(v)
    print(f"  {k}: {shape}")

# ---------------------------------------------------------------------------
# Step 3: Assert no missing keys
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Assert no missing keys")
print("=" * 70)

# Exclude kv_mgr keys - those are runtime state, not loaded from checkpoint
neuron_checkpoint_keys = {k for k in neuron_keys if not k.startswith("kv_mgr.")}

missing = neuron_checkpoint_keys - set(converted.keys())
extra   = set(converted.keys()) - neuron_checkpoint_keys

print(f"Neuron model keys (excl kv_mgr): {len(neuron_checkpoint_keys)}")
print(f"Converted keys                  : {len(converted.keys())}")
print(f"Missing from converted (ERRORS) : {len(missing)}")
print(f"Extra in converted (OK to ignore): {len(extra)}")

if missing:
    print("\nMissing keys:")
    for k in sorted(missing):
        print(f"  {k}")

if extra:
    print("\nExtra keys (not in Neuron model, harmless):")
    for k in sorted(extra):
        print(f"  {k}")

if missing:
    print("\nFAILED: Missing keys in converted state dict!")
    sys.exit(1)
else:
    print("\nPASSED: All Neuron checkpoint keys present in converted state dict!")
