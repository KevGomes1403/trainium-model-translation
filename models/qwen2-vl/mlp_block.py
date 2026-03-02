"""
NeuronQwen2VLMLP — Qwen2-VL MLP block translated for NxDI.

SwiGLU forward: down(silu(gate(x)) * up(x))
No bias on any projection.
"""

from typing import Optional

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


class NeuronQwen2VLMLP(nn.Module):
    """
    Qwen2-VL SwiGLU MLP translated for Neuron inference.

    Replaces nn.Linear with ColumnParallelLinear (gate, up) and
    RowParallelLinear (down) when TP is enabled.  Falls back to
    nn.Linear on CPU (for unit tests / development).
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size          # 3584
        self.intermediate_size = config.intermediate_size  # 18944
        self.act_fn = ACT2FN[config.hidden_act]         # silu

        if parallel_state.model_parallel_is_initialized():
            tp_group = get_tp_group(config)
            dtype = config.neuron_config.torch_dtype
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
        else:
            # CPU fallback for unit tests
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
