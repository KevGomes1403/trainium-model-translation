"""
NeuronQwen2VLVisionAttention — Vision attention block for Qwen2-VL, translated to NxDI.

This is a faithful translation of `VisionAttention` from qwen2_vl_pytorch.py.

Key architectural properties of the source block:
  - Input: hidden_states [seq_len, embed_dim]  (packed, NO batch dimension)
  - cu_seqlens: [num_images+1] cumulative sequence lengths for chunked attention
  - position_embeddings: (cos, sin) each [seq_len, head_dim] — pre-computed 2D RoPE
    from VisionRotaryEmbedding(head_dim // 2) indexed by 2D (h, w) grid positions
  - Fused QKV: self.qkv = Linear(dim, dim*3, bias=True)
  - Output projection: self.proj = Linear(dim, dim, bias=False)  ← NO bias
  - MHA: num_kv_heads == num_heads (no GQA)
  - Chunked attention: each image/frame processed independently via cu_seqlens splits
  - apply_rotary_pos_emb_vision: cos/sin unsqueeze(-2) to broadcast over heads

Integration contract:
  - Input:  hidden_states [seq_len, embed_dim]
            cu_seqlens    [num_images + 1]  (int32, starts with 0)
            position_embeddings  (cos [seq_len, head_dim], sin [seq_len, head_dim])
  - Output: [seq_len, embed_dim]

Deviations from source:
  - Uses ColumnParallelLinear / RowParallelLinear instead of nn.Linear for TP support.
    With tp_degree=1 (unit test), these are numerically identical to nn.Linear.
  - apply_rotary_pos_emb_vision is re-implemented verbatim (no NxDI RoPE kernel used)
    because the vision encoder uses a different RoPE convention (2D grid, head_dim//2 dim).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed_inference.models.config import InferenceConfig


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims. Verbatim from HF qwen2_vl source."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Verbatim from HF transformers/models/qwen2_vl/modeling_qwen2_vl.py.

    Args:
        q, k : [seq_len, num_heads, head_dim]
        cos  : [seq_len, head_dim]
        sin  : [seq_len, head_dim]
    """
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    # unsqueeze(-2) broadcasts over the heads dimension: [S, 1, hD]
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class NeuronQwen2VLVisionAttention(nn.Module):
    """
    Qwen2-VL vision attention translated to NxDI.

    Matches VisionAttention from qwen2_vl_pytorch.py exactly:
      - Fused QKV via ColumnParallelLinear (gather_output=True), bias=True
      - Output proj via RowParallelLinear, bias=False
      - Chunked SDPA per cu_seqlens (one chunk per image/frame)
      - 2D RoPE applied via _apply_rotary_pos_emb_vision (pre-computed outside)

    Args:
        config: InferenceConfig with:
            - hidden_size (embed_dim)
            - num_attention_heads
            - neuron_config.torch_dtype
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        dtype = config.neuron_config.torch_dtype

        # Fused QKV: [dim, dim*3] — ColumnParallelLinear shards along output dim
        # gather_output=True so the full [seq_len, dim*3] tensor is available for reshape
        self.qkv = ColumnParallelLinear(
            input_size=self.dim,
            output_size=self.dim * 3,
            bias=True,
            gather_output=True,
            dtype=dtype,
        )

        # Output projection: bias=True (matches source self.proj = nn.Linear(dim, dim)
        # which uses PyTorch default bias=True)
        self.proj = RowParallelLinear(
            input_size=self.dim,
            output_size=self.dim,
            bias=True,
            input_is_parallel=False,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [seq_len, embed_dim]  (packed, no batch dim)
            cu_seqlens:    [num_images + 1]  cumulative sequence lengths (int32)
            cos:           [seq_len, head_dim]  pre-computed 2D RoPE cosines
            sin:           [seq_len, head_dim]  pre-computed 2D RoPE sines

        Note: cos and sin are passed as separate tensors (rather than a tuple) so
        that the XLA tracing harness (which requires all inputs to be plain tensors)
        can accept them. They correspond to position_embeddings=(cos, sin) in the
        original VisionAttention.forward() signature.

        Returns:
            [seq_len, embed_dim]
        """
        seq_length = hidden_states.shape[0]

        # Fused QKV projection and split — matches HF reshape/permute/unbind
        # qkv: [seq_len, dim*3] → reshape → [seq_len, 3, num_heads, head_dim]
        # → permute(1,0,2,3) → [3, seq_len, num_heads, head_dim] → unbind(0)
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )  # each: [seq_len, num_heads, head_dim]

        # Apply 2D vision RoPE (verbatim HF logic)
        query_states, key_states = _apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        # Reshape for SDPA: [1, num_heads, seq_len, head_dim]
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Chunked attention: process each image/frame independently
        # Matches HF "Other implementations" path (non-flash)
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=2)
            for tensor in (query_states, key_states, value_states)
        ]

        attn_outputs = [
            F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            )
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=2)  # [1, num_heads, seq_len, head_dim]

        # Flatten heads and apply output projection
        attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(seq_length, self.dim).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output
