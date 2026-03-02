"""
Unit test: NeuronVisionMlp vs PyTorch VisionMlp.

Validates that NeuronVisionMlp produces identical outputs to the reference
PyTorch VisionMlp when given the same weights and inputs.

Weight mapping (PyTorch key -> NxDI key, block. prefix added automatically):
  fc1.weight -> mlp_fc1.weight
  fc1.bias   -> mlp_fc1.bias
  fc2.weight -> mlp_fc2.weight
  fc2.bias   -> mlp_fc2.bias
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

MODEL_DIR = ROOT_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from neuronx_distributed_inference.models.config import NeuronConfig, InferenceConfig
from block_testing_utils import test_block_correctness
from vision_mlp_block import NeuronVisionMlp
from qwen2_vl_pytorch import VisionMlp as _VisionMlp


# ---------------------------------------------------------------------------
# Test dimensions (small for fast compilation)
# ---------------------------------------------------------------------------
bs, sl = 2, 128
embed_dim = 64
mlp_hidden_dim = 256
dtype = torch.bfloat16


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
neuron_config = NeuronConfig(
    batch_size=bs,
    seq_len=sl,
    tp_degree=1,
    torch_dtype=dtype,
    on_cpu=True,
    fused_qkv=False,
)

config = InferenceConfig(
    neuron_config=neuron_config,
    hidden_size=embed_dim,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=16,
    max_position_embeddings=4096,
    rope_theta=10000.0,
    initial_context_length=4096,
)
config.mlp_hidden_dim = mlp_hidden_dim
config.hidden_act = "quick_gelu"
config.num_cores_per_group = 1


# ---------------------------------------------------------------------------
# PyTorch reference wrapper
# ---------------------------------------------------------------------------
class WrappedVisionMlp(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.block = _VisionMlp(
            dim=config.hidden_size,
            hidden_dim=config.mlp_hidden_dim,
            hidden_act=config.hidden_act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    def state_dict(self, **kwargs):
        return self.block.state_dict(**kwargs)

    def named_parameters(self, **kwargs):
        return self.block.named_parameters(**kwargs)


# ---------------------------------------------------------------------------
# Weight mapping: PyTorch key -> NxDI key
# ---------------------------------------------------------------------------
weight_mapping = {
    "fc1.weight": "fc1.weight",
    "fc1.bias":   "fc1.bias",
    "fc2.weight": "fc2.weight",
    "fc2.bias":   "fc2.bias",
}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
torch.manual_seed(123)
sample = torch.rand(bs, sl, embed_dim, dtype=dtype)

example_inputs  = [(torch.zeros(bs, sl, embed_dim, dtype=dtype),)]
test_inputs     = [(sample,)]
reference_inputs = [(sample,)]


# ---------------------------------------------------------------------------
# Run test
# ---------------------------------------------------------------------------
test_block_correctness(
    neuron_block_class=NeuronVisionMlp,
    pytorch_block_class=WrappedVisionMlp,
    weight_mapping=weight_mapping,
    neuron_init_kwargs={"config": config},
    pytorch_init_kwargs={"config": config},
    example_inputs=example_inputs,
    test_inputs=test_inputs,
    reference_inputs=reference_inputs,
    checkpoint_name="vision_mlp.pt",
    seed=42,
    use_moe=False,
    verbose=True,
)
