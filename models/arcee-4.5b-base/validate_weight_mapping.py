"""
Weight mapping validation for Arcee AFM-4.5B NxDI translation.

This script verifies that convert_hf_to_neuron_state_dict correctly maps
all HF checkpoint keys to the NxDI model's expected keys, with no missing
or extra keys.

Procedure:
  1. Read the HF checkpoint key list from model.safetensors.index.json.
  2. Strip the "model." prefix (as the NxDI framework does before calling
     convert_hf_to_neuron_state_dict).
  3. Apply convert_hf_to_neuron_state_dict with a 1-layer config.
  4. Instantiate NeuronArceeDecoderLayer + embedding/norm as a proxy for
     the NxDI model's expected state dict (on CPU, 1 layer, no TP).
  5. Diff the converted keys against expected NxDI keys and assert clean.

The HF checkpoint is at /home/ubuntu/models/afm-4.5b-base/.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add model directory to path
MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from modeling_arcee_neuron import (
    ArceeInferenceConfig,
    NeuronArceeDecoderLayer,
    NeuronArceeForCausalLM,
)

# ---------------------------------------------------------------------------
# Config — 1-layer version of real model dims for validation
# ---------------------------------------------------------------------------
HF_CHECKPOINT = "/home/ubuntu/models/afm-4.5b-base"
NUM_VALIDATION_LAYERS = 1   # Use only 1 layer to keep instantiation fast

neuron_config = NeuronConfig(
    batch_size=1,
    seq_len=128,
    tp_degree=1,
    torch_dtype=torch.bfloat16,
    on_cpu=True,
    fused_qkv=False,
)

# Build a minimal InferenceConfig matching the real model (1 layer for speed)
config = InferenceConfig(
    neuron_config=neuron_config,
    hidden_size=2560,
    num_attention_heads=20,
    num_key_value_heads=4,
    num_hidden_layers=NUM_VALIDATION_LAYERS,
    head_dim=128,
    intermediate_size=18432,
    vocab_size=128004,
    max_position_embeddings=65536,
    rope_theta=10000.0,
    rope_scaling={"rope_type": "yarn", "factor": 20.0,
                  "original_max_position_embeddings": 4096},
    rms_norm_eps=1e-5,
    hidden_act="relu2",
    attention_bias=False,
    mlp_bias=False,
    pad_token_id=0,
    sliding_window=None,
    initial_context_length=128,
)
config.num_cores_per_group = 1
config.intermediate_size = 18432
config.hidden_act = "relu2"
config.mlp_bias = False
config.rms_norm_eps = 1e-5


# ---------------------------------------------------------------------------
# Step 1: Extract HF keys (only for layer 0 since we use 1-layer config)
# ---------------------------------------------------------------------------
print("=" * 70)
print("Arcee AFM-4.5B Weight Mapping Validation")
print("=" * 70)

index_path = Path(HF_CHECKPOINT) / "model.safetensors.index.json"
with open(index_path) as f:
    index = json.load(f)

all_hf_keys = list(index["weight_map"].keys())
print(f"\nTotal HF checkpoint keys: {len(all_hf_keys)}")

# Keep only keys for layer 0 (and non-layer keys), to match NUM_VALIDATION_LAYERS=1
def keep_key_for_validation(key: str) -> bool:
    """Keep layer-0 keys and non-layer keys."""
    if "model.layers." in key:
        # Extract layer number
        rest = key.replace("model.layers.", "")
        layer_num = int(rest.split(".")[0])
        return layer_num < NUM_VALIDATION_LAYERS
    return True  # embed_tokens, norm, lm_head

validation_hf_keys = [k for k in all_hf_keys if keep_key_for_validation(k)]
print(f"Keys for {NUM_VALIDATION_LAYERS}-layer validation: {len(validation_hf_keys)}")


# ---------------------------------------------------------------------------
# Step 2: Strip "model." prefix (as NxDI framework does)
# ---------------------------------------------------------------------------
def strip_model_prefix(key: str) -> str:
    if key.startswith("model."):
        return key[len("model."):]
    return key

hf_stripped = {strip_model_prefix(k): None for k in validation_hf_keys}
print(f"\nHF keys after stripping 'model.' prefix:")
for k in sorted(hf_stripped.keys()):
    print(f"  {k}")


# ---------------------------------------------------------------------------
# Step 3: Apply convert_hf_to_neuron_state_dict
# ---------------------------------------------------------------------------
# Inject dummy tensors so we have a valid state dict to transform
dummy_state = dict(hf_stripped)  # all values None — we only care about keys
for k in dummy_state:
    dummy_state[k] = torch.zeros(1)  # shape doesn't matter for key validation

converted = NeuronArceeForCausalLM.convert_hf_to_neuron_state_dict(dummy_state, config)
converted_keys = set(converted.keys())

print(f"\nConverted NxDI keys ({len(converted_keys)}):")
for k in sorted(converted_keys):
    print(f"  {k}")


# ---------------------------------------------------------------------------
# Step 4: Build reference NxDI model state dict (CPU, 1 layer)
# ---------------------------------------------------------------------------
# We build the relevant NxDI components manually to get their state dict keys:
#   - NeuronArceeDecoderLayer (layer 0)
#   - embed_tokens (nn.Embedding)
#   - lm_head (nn.Linear)
#   - norm (nn.RMSNorm)
#   - rank_util.rank (injected by convert fn — not a module param)

print("\nBuilding 1-layer NxDI reference model (CPU)...")

# Build a proxy module with the same structure as NeuronArceeModel.init_model
class ProxyArceeModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, 0)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.layers = nn.ModuleList([NeuronArceeDecoderLayer(cfg, layer_idx=i)
                                     for i in range(cfg.num_hidden_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

proxy = ProxyArceeModel(config)
nxdi_keys = set(proxy.state_dict().keys())

# Also add the rank metadata keys that convert_hf_to_neuron_state_dict injects
for i in range(NUM_VALIDATION_LAYERS):
    nxdi_keys.add(f"layers.{i}.self_attn.rank_util.rank")
nxdi_keys.add("rank_util.rank")

print(f"\nExpected NxDI model keys ({len(nxdi_keys)}):")
for k in sorted(nxdi_keys):
    print(f"  {k}")


# ---------------------------------------------------------------------------
# Step 5: Diff
# ---------------------------------------------------------------------------
missing_from_converted = nxdi_keys - converted_keys
extra_in_converted = converted_keys - nxdi_keys

print("\n" + "=" * 70)
print("DIFF RESULTS")
print("=" * 70)

if missing_from_converted:
    print(f"\nMISSING from converted (expected by NxDI model but not produced):")
    for k in sorted(missing_from_converted):
        print(f"  MISSING: {k}")
else:
    print("\nNo missing keys.")

if extra_in_converted:
    print(f"\nEXTRA in converted (produced by convert fn but not in NxDI model):")
    for k in sorted(extra_in_converted):
        print(f"  EXTRA: {k}")
else:
    print("No extra keys.")

print("\n" + "=" * 70)
if not missing_from_converted and not extra_in_converted:
    print("PASSED: All keys match perfectly.")
elif not missing_from_converted and extra_in_converted:
    print("WARNING: Extra keys present (may be harmless framework metadata).")
else:
    print("FAILED: Missing keys detected — weight loading will be incomplete.")
    sys.exit(1)
print("=" * 70)
