"""
NxDI translation of Gemma-2 MLP for AWS Trainium.

This module translates the PyTorch Gemma2MLP (a SwiGLU MLP block with gelu_pytorch_tanh
activation) to its NxDI equivalent using tensor-parallel ColumnParallelLinear and
RowParallelLinear layers.

Forward computation:
    output = down_proj(gelu(gate_proj(x), approximate="tanh") * up_proj(x))

Tensor parallelism layout:
    gate_proj  [H -> I/tp]  ColumnParallelLinear, gather_output=False  (output stays sharded)
    up_proj    [H -> I/tp]  ColumnParallelLinear, gather_output=False  (output stays sharded)
    down_proj  [I/tp -> H]  RowParallelLinear,    input_is_parallel=True (all-reduce on output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.utils.distributed import get_tp_group


class NeuronGemma2MLP(nn.Module):
    """
    NxDI tensor-parallel SwiGLU MLP for Gemma-2.

    Replaces the three nn.Linear layers in the original Gemma2MLP with:
      - gate_proj: ColumnParallelLinear (H -> I, sharded along output dim)
      - up_proj:   ColumnParallelLinear (H -> I, sharded along output dim)
      - down_proj: RowParallelLinear    (I -> H, sharded along input dim, all-reduces)

    Uses gelu_pytorch_tanh activation (not silu), which distinguishes it from other
    SwiGLU models like OLMo-3 and ERNIE.

    Args:
        config: InferenceConfig carrying at minimum:
            - config.hidden_size          (int): input/output feature dimension H
            - config.intermediate_size    (int): feed-forward inner dimension I
            - config.hidden_activation    (str): activation function name ("gelu_pytorch_tanh")
            - config.neuron_config.torch_dtype: weight dtype (expected: torch.bfloat16)
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Use the configured activation function
        # Gemma-2 uses "gelu_pytorch_tanh" by default
        hidden_activation = getattr(config, "hidden_activation", "gelu_pytorch_tanh")
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[hidden_activation]

        dtype = config.neuron_config.torch_dtype

        if parallel_state.model_parallel_is_initialized():
            tp_group = get_tp_group(config)

            # gate_proj: H -> I (column-parallel, output sharded across TP ranks)
            # gather_output=False keeps the intermediate result sharded so down_proj
            # can consume it directly without an extra gather + scatter.
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
                # pad=True omitted: raises RuntimeError in training mode (model.eval()
                # not called during HLO generation). All test dims are divisible by
                # tp_degree=1 anyway.
            )

            # up_proj: H -> I (column-parallel, same sharding as gate_proj)
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )

            # down_proj: I -> H (row-parallel, accepts sharded input, all-reduces output)
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
        else:
            # Fallback to standard nn.Linear when TP is not initialized (for testing)
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            [batch, seq_len, hidden_size]
        """
        # SwiGLU with gelu_pytorch_tanh:
        # down_proj(gelu(gate_proj(x), approx="tanh") * up_proj(x))
        #
        # Both gate_proj and up_proj output [*, I/tp] tensors (or [*, I] if no TP).
        # The element-wise multiply is applied in the (possibly sharded) space before down_proj
        # performs the row-parallel reduction back to [*, H].
        gate = self.gate_proj(x)  # [bs, sl, I/tp]
        up = self.up_proj(x)      # [bs, sl, I/tp]
        hidden = self.act_fn(gate) * up  # [bs, sl, I/tp]
        out = self.down_proj(hidden)  # [bs, sl, H]  (all-reduce inside if TP active)
        return out
