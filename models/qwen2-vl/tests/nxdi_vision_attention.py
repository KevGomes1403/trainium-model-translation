"""
NxDI translation of Qwen2-VL VisionAttention using NeuronAttentionBase.

This module defines a NeuronX Distributed Inference (NxDI) attention block
for the Qwen2-VL vision encoder that:

- Inherits from `NeuronAttentionBase` to use the standard NxDI attention stack.
- Uses fused QKV projections via `GroupQueryAttention_QKV` (no manual linears).
- Implements multi-head self-attention (no GQA; num_attention_heads ==
  num_key_value_heads).
- Applies 2D vision RoPE using externally provided `cos` and `sin` tensors,
  matching the semantics of `apply_rotary_pos_emb_vision` in the original
  Qwen2-VL vision encoder.

Integration contract:
- Inputs:
    hidden_states: [seq_len, hidden_size] (packed sequence, no batch dim)
    cu_seqlens:    [num_images + 1] cumulative sequence lengths (int32, starts at 0)
    cos:           [seq_len, head_dim]
    sin:           [seq_len, head_dim]
- Output:
    Tensor of shape [seq_len, hidden_size]
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import apply_rotary_pos_emb


class NeuronQwen2VLVisionAttentionBase(NeuronAttentionBase):
    """
    NeuronAttentionBase-based implementation of Qwen2-VL VisionAttention.

    This block:
    - Uses fused QKV projection provided by NeuronAttentionBase / GroupQueryAttention_QKV.
    - Uses a standard output projection via GroupQueryAttention_O.
    - Applies 2D RoPE using externally provided `cos` and `sin` tensors.
    - Implements multi-head self-attention (MHA), not GQA.

    Args:
        config: An `InferenceConfig` carrying at least:
            - hidden_size
            - num_attention_heads
            - num_key_value_heads (must equal num_attention_heads for this block)
            - head_dim (optional; defaults to hidden_size // num_attention_heads)
    """

    def __init__(self, config: InferenceConfig):
        if config is None or getattr(config, "neuron_config", None) is None:
            raise ValueError("NeuronQwen2VLVisionAttentionBase requires a valid InferenceConfig with neuron_config.")

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=None,  # RoPE is supplied via external cos/sin tensors
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            qkv_bias=True,  # VisionAttention uses bias on QKV
            o_bias=True,  # VisionAttention uses bias on proj
            num_cores_per_group=getattr(config, "num_cores_per_group", 1),
            sliding_window=None,
        )

        if self.num_attention_heads != self.num_key_value_heads:
            raise ValueError(
                "NeuronQwen2VLVisionAttentionBase currently supports only MHA "
                "(num_attention_heads must equal num_key_value_heads)."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *_,
        **__,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [seq_len, hidden_size]
            cu_seqlens: [num_images + 1] cumulative sequence lengths (int32)
            cos: [seq_len, head_dim] 2D vision RoPE cosine values
            sin: [seq_len, head_dim] 2D vision RoPE sine values

        Returns:
            Tensor of shape [seq_len, hidden_size]
        """
        if hidden_states.dim() != 2:
            raise ValueError(
                f"Expected hidden_states of shape [seq_len, hidden_size], got {hidden_states.shape}"
            )

        seq_len, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"hidden_states hidden_size ({hidden_size}) does not match config.hidden_size ({self.hidden_size})"
            )

        if cos.shape != (seq_len, self.head_dim) or sin.shape != (seq_len, self.head_dim):
            raise ValueError(
                f"cos/sin must have shape [seq_len, head_dim]=[{seq_len}, {self.head_dim}], "
                f"got cos={tuple(cos.shape)}, sin={tuple(sin.shape)}"
            )

        if cu_seqlens.dim() != 1:
            raise ValueError(f"cu_seqlens must be 1D, got shape {cu_seqlens.shape}")

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
        if lengths.numel() == 0:
            raise ValueError("cu_seqlens must contain at least two elements (start and end).")
        if lengths.sum().item() != seq_len:
            raise ValueError(
                f"Sum of segment lengths from cu_seqlens ({lengths.sum().item()}) "
                f"must equal seq_len ({seq_len})."
            )

        device = hidden_states.device
        dtype_in = hidden_states.dtype

        # Add batch dimension: [1, S, H]
        x = hidden_states.unsqueeze(0).to(self.torch_dtype)

        # Prepare Q, K, V without applying any built-in RoPE.
        # Q, K, V: [B, num_heads or num_kv_heads, S, head_dim]
        Q, K, V, cos_cache, sin_cache, residual = self.prep_qkv_tensors(
            position_ids=None,
            hidden_states=x,
            past_key_value=None,
            adapter_ids=None,
            cos_cache=None,
            sin_cache=None,
            rmsnorm=None,
            skip_rope=True,
            residual=None,
            use_polar_compatible_rope=False,
        )

        # Apply 2D RoPE using externally provided cos/sin.
        # Convert cos/sin to [B, S, head_dim] for apply_rotary_pos_emb.
        cos_b = cos.to(dtype=self.torch_dtype, device=device).unsqueeze(0)  # [1, S, head_dim]
        sin_b = sin.to(dtype=self.torch_dtype, device=device).unsqueeze(0)  # [1, S, head_dim]

        Q, K = apply_rotary_pos_emb(Q, K, cos_b, sin_b)

        # Build block-diagonal attention mask from cu_seqlens.
        # Tokens attend only within their own image segment.
        num_segments = lengths.shape[0]
        segment_ids = torch.repeat_interleave(
            torch.arange(num_segments, device=device, dtype=torch.long),
            lengths,
        )  # [S]
        attn_mask = (
            segment_ids.view(1, 1, seq_len, 1) == segment_ids.view(1, 1, 1, seq_len)
        )  # [1, 1, S, S], bool keep-mask

        # For MHA (no GQA), K/V already have num_heads heads.
        K_active = K
        V_active = V

        # Compute scaled attention scores with NeuronAttentionBase helper.
        attn_scores = self.scaled_qk(Q, K_active, attn_mask)  # [1, num_heads, S, S]
        attn_weights = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(Q.dtype)

        # Attention output: [1, num_heads, S, head_dim]
        attn_output = torch.matmul(attn_weights, V_active)

        # Merge heads and project back to hidden_size using the Neuron out-projection.
        batch_size = 1
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )  # [1, S, H]

        attn_output = self.get_o_proj()(attn_output, adapter_ids=None)  # [1, S, H]

        # Remove batch dimension and cast back to the input dtype.
        return attn_output.squeeze(0).to(dtype_in)


__all__ = ["NeuronQwen2VLVisionAttentionBase"]

