"""
NeuronVisionMlp — Qwen2-VL Vision Transformer MLP block translated for NxDI.

Architecture: fc2(act(fc1(x)))   [simple 2-layer MLP, NOT SwiGLU]

Key differences from the text-model MLP (NeuronQwen2VLMLP):
  - No gate projection — only fc1 -> act -> fc2 (standard bottleneck MLP)
  - Both fc1 and fc2 have bias=True (nn.Linear default)
  - Activation is "quick_gelu" (QuickGELU ~ x * sigmoid(1.702 * x))

Weight naming matches HF checkpoint keys after bulk rename:
  visual.blocks.{i}.mlp.fc1.* → blocks.{i}.mlp.fc1.*
  visual.blocks.{i}.mlp.fc2.* → blocks.{i}.mlp.fc2.*

Config attributes consumed (vision InferenceConfig):
  config.hidden_size        - embed_dim  (e.g. 1280)
  config.mlp_hidden_dim     - fc1 output / fc2 input  (e.g. 5120)
  config.hidden_act         - activation key  (e.g. "quick_gelu")
  config.neuron_config.torch_dtype - weight dtype  (e.g. torch.bfloat16)
"""

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.utils.distributed import get_tp_group


class NeuronVisionMlp(nn.Module):
    """
    Qwen2-VL Vision Transformer MLP translated for Neuron inference.

    Forward: x -> fc1 -> act -> fc2 -> output   (same shape as input)

    When tensor-parallelism is active fc1 is a ColumnParallelLinear (fan-out,
    no gather) and fc2 is a RowParallelLinear (fan-in, all-reduce).  When TP
    is not initialized (CPU unit-test mode) plain nn.Linear is used.
    """

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__()

        embed_dim: int = config.hidden_size
        mlp_hidden_dim: int = config.mlp_hidden_dim

        hidden_act: str = getattr(config, "hidden_act", "quick_gelu")
        self.act = ACT2FN[hidden_act]

        if parallel_state.model_parallel_is_initialized():
            tp_group = get_tp_group(config)
            dtype = config.neuron_config.torch_dtype

            self.fc1 = ColumnParallelLinear(
                embed_dim,
                mlp_hidden_dim,
                bias=True,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
            self.fc2 = RowParallelLinear(
                mlp_hidden_dim,
                embed_dim,
                bias=True,
                input_is_parallel=True,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
        else:
            self.fc1 = nn.Linear(embed_dim, mlp_hidden_dim, bias=True)
            self.fc2 = nn.Linear(mlp_hidden_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
