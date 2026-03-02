"""
Unit test: NeuronQwen2VLMLP vs PyTorch Qwen2MLP.

Validates that NeuronQwen2VLMLP produces identical outputs to the
reference PyTorch MLP when given the same weights and inputs.

Weight mapping is 1:1 — no key renames needed for MLP.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

MODEL_DIR = ROOT_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from block_testing_utils import _create_default_config, test_block_correctness
from mlp_block import NeuronQwen2VLMLP


# ---------------------------------------------------------------------------
# Test dimensions
# ---------------------------------------------------------------------------
bs, sl, hs = 2, 128, 64
intermediate_size = 256
dtype = torch.bfloat16


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
config = _create_default_config(hidden_size=hs)
config.intermediate_size = intermediate_size
config.hidden_act = "silu"
config.rms_norm_eps = 1e-6


# ---------------------------------------------------------------------------
# PyTorch reference — Qwen2MLP (identical to Qwen2VL MLP)
# ---------------------------------------------------------------------------
class PyTorchQwen2VLMLP(nn.Module):
    """Standalone Qwen2-VL SwiGLU MLP for correctness testing."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Weight mapping: PyTorch key → NxDI key (block. prefix added automatically)
# Keys are identical — no renames needed.
# ---------------------------------------------------------------------------
weight_mapping = {
    "gate_proj.weight": "gate_proj.weight",
    "up_proj.weight": "up_proj.weight",
    "down_proj.weight": "down_proj.weight",
}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
torch.manual_seed(123)
sample = torch.rand(bs, sl, hs, dtype=dtype)

example_inputs = [(torch.zeros(bs, sl, hs, dtype=dtype),)]
test_inputs = [(sample,)]
reference_inputs = [(sample,)]


# ---------------------------------------------------------------------------
# Run test
# ---------------------------------------------------------------------------
test_block_correctness(
    neuron_block_class=NeuronQwen2VLMLP,
    pytorch_block_class=PyTorchQwen2VLMLP,
    weight_mapping=weight_mapping,
    example_inputs=example_inputs,
    test_inputs=test_inputs,
    reference_inputs=reference_inputs,
    checkpoint_name="mlp.pt",
    seed=42,
    neuron_init_kwargs={"config": config},
    pytorch_init_kwargs={"config": config},
    verbose=True,
)
