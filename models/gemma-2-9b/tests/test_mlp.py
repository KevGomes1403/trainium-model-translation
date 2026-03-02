"""
Unit test: NeuronGemma2MLP vs PyTorch Gemma2MLP.

Gemma-2 uses a SwiGLU MLP with gelu_pytorch_tanh activation:
    forward(x) = down_proj(gelu(gate_proj(x), approx="tanh") * up_proj(x))

This differs from other SwiGLU models (OLMo-3, ERNIE) which use silu activation.

Validates that NeuronGemma2MLP produces identical outputs to the reference
PyTorch MLP when given the same weights and inputs.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from neuronx_distributed_inference.models.config import NeuronConfig, InferenceConfig

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

MODEL_DIR = ROOT_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from block_testing_utils import test_block_correctness
from nxdi_mlp import NeuronGemma2MLP


# ---------------------------------------------------------------------------
# Test dimensions
# ---------------------------------------------------------------------------
bs, sl, hs = 2, 128, 64
intermediate_size = 256
dtype = torch.bfloat16


# ---------------------------------------------------------------------------
# Config
# NeuronGemma2MLP reads: config.hidden_size, config.intermediate_size,
#                        config.hidden_activation,
#                        config.neuron_config.torch_dtype
#
# The test harness _create_default_config does NOT include intermediate_size
# or hidden_activation by default, so we must inject via neuron_init_kwargs.
# ---------------------------------------------------------------------------
neuron_config = NeuronConfig(
    batch_size=bs,
    seq_len=sl,
    tp_degree=1,
    torch_dtype=dtype,
    on_cpu=True,
)

config = InferenceConfig(
    neuron_config=neuron_config,
    hidden_size=hs,
    num_attention_heads=8,
    num_key_value_heads=2,
    sliding_window=32,
    rope_theta=10000.0,
    max_position_embeddings=4096,
    attention_bias=False,
)
# Inject custom attributes not in default InferenceConfig
config.intermediate_size = intermediate_size
config.hidden_activation = "gelu_pytorch_tanh"
config.num_cores_per_group = 1


# ---------------------------------------------------------------------------
# PyTorch reference (standalone, matches Gemma2MLP exactly)
# ---------------------------------------------------------------------------
class PyTorchGemma2MLP(nn.Module):
    """Standalone Gemma-2 SwiGLU MLP for correctness testing."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_activation: str):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Weight mapping: PyTorch key → NxDI key (block. prefix added automatically)
# NeuronGemma2MLP uses the same module names as PyTorchGemma2MLP.
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
    neuron_block_class=NeuronGemma2MLP,
    pytorch_block_class=PyTorchGemma2MLP,
    weight_mapping=weight_mapping,
    example_inputs=example_inputs,
    test_inputs=test_inputs,
    reference_inputs=reference_inputs,
    checkpoint_name="gemma2_mlp.pt",
    seed=42,
    # Pass config explicitly so intermediate_size and hidden_activation are available
    neuron_init_kwargs={"config": config},
    pytorch_init_kwargs={
        "hidden_size": hs,
        "intermediate_size": intermediate_size,
        "hidden_activation": "gelu_pytorch_tanh",
    },
    verbose=True,
)
