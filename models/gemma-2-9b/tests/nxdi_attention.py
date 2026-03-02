"""
NxDI translation of Gemma-2 Attention for AWS Trainium.

This module translates the PyTorch Gemma2Attention block to its NxDI equivalent,
inheriting from NeuronAttentionBase for the core attention mechanics.

Key features of Gemma-2 attention:
  1. GQA: 16 Q heads, 8 KV heads, head_dim=256
  2. Scaling factor: query_pre_attn_scalar**-0.5 = 256**-0.5 ≈ 0.0625 (NOT 1/sqrt(head_dim))
  3. Attn logit softcapping: 50.0 (applied as tanh(scores/50)*50 before softmax)
     - NOT directly supported by NeuronAttentionBase.scaled_qk()
     - Can be overridden in subclass for custom implementations
  4. Sliding window: configurable per layer based on layer_types
     - Even-indexed layers: "sliding_attention"
     - Odd-indexed layers: "full_attention"

This translation focuses on the core attention mechanics:
  - Q/K/V projections (GQA-ready)
  - RoPE positional embeddings
  - Sliding window support (per layer_idx)
  - GQA compute

The softcapping (tanh non-linearity) is deferred for now since NxDI's NKI kernel
does not expose a softcap parameter. Custom scaled_qk override is provided as a hook.
"""

import math
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase


class NeuronGemma2Attention(NeuronAttentionBase):
    """
    Gemma-2 attention adapted for Neuron (NxDI).

    Inherits from NeuronAttentionBase to leverage:
      - GQA support
      - RoPE application
      - Sliding window attention
      - NKI attention kernels

    Overrides scaled_qk() to apply custom scaling (query_pre_attn_scalar instead of head_dim)
    and optionally softcapping (tanh).

    Args:
        config: InferenceConfig carrying:
            - hidden_size: total hidden dimension
            - num_attention_heads: number of Q heads (e.g., 16)
            - num_key_value_heads: number of KV heads (e.g., 8)
            - head_dim: per-head dimension (default: hidden_size // num_attention_heads)
            - query_pre_attn_scalar: scaling numerator (e.g., 256)
            - attn_logit_softcapping: softcap value (e.g., 50.0)
            - max_position_embeddings: sequence length (e.g., 4096)
            - rope_theta: RoPE base (e.g., 10000.0)
            - rms_norm_eps: layer norm epsilon (e.g., 1e-6)
            - layer_types: list of layer types; "sliding_attention" → use sliding_window
            - sliding_window: window size for sliding attention (e.g., 4096)
            - num_cores_per_group: (e.g., 1)
        layer_idx: layer index to determine attention type from config.layer_types
    """

    def __init__(self, config: InferenceConfig, layer_idx: int):
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        # Determine layer type (sliding vs full attention) from config.layer_types
        layer_types = getattr(config, "layer_types", None)
        layer_type = None
        if layer_types is not None and layer_idx < len(layer_types):
            layer_type = layer_types[layer_idx]
        else:
            layer_type = "full_attention"  # default

        # Set sliding_window if this is a sliding_attention layer
        sliding_window = None
        if layer_type == "sliding_attention":
            configured_window = getattr(config, "sliding_window", None)
            if configured_window is not None:
                sliding_window = configured_window

        # Store Gemma-2 custom scaling and softcap
        self.query_pre_attn_scalar = getattr(config, "query_pre_attn_scalar", 256)
        self.attn_logit_softcapping = getattr(config, "attn_logit_softcapping", None)
        self.layer_idx = layer_idx
        self.layer_type = layer_type

        # Create RoPE embedding
        # LlamaRotaryEmbedding expects a config object with specific attributes
        # If the config already has them, pass it directly
        rotary_emb = LlamaRotaryEmbedding(config)

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            qkv_bias=False,  # Gemma-2 has no bias on Q/K/V/O projections
            o_bias=False,
            num_cores_per_group=getattr(config, "num_cores_per_group", 1),
            sliding_window=sliding_window,
        )

    def scaled_qk(self, Q, K, attention_mask):
        """
        Compute scaled + softcapped attention scores.

        1. Scale: query_pre_attn_scalar**-0.5 (not head_dim**-0.5)
        2. Softcap: tanh(scores / softcap) * softcap
        3. Attention mask (NxDI boolean keep-mask)

        Args:
            Q: Query tensor [batch, num_heads, seq_len, head_dim]
            K: Key tensor [batch, num_kv_heads, seq_len, head_dim]
            attention_mask: Boolean keep-mask [batch, 1, seq_len, seq_len] or None
                1 (True) → attend, 0 (False) → mask

        Returns:
            QK: Scaled, softcapped attention scores [batch, num_heads, seq_len, seq_len]
        """
        scale = self.query_pre_attn_scalar ** -0.5
        QK = torch.matmul(Q, K.transpose(2, 3)) * scale

        if self.attn_logit_softcapping is not None:
            QK = torch.tanh(QK / self.attn_logit_softcapping) * self.attn_logit_softcapping

        if attention_mask is not None:
            QK = torch.where(
                attention_mask.to(torch.bool),
                QK,
                torch.finfo(QK.dtype).min,
            )

        return QK
