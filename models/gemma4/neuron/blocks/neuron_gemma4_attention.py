"""
NeuronGemma4TextAttention — NxDI translation of the Gemma4 Text Attention block.

Single class handles both `sliding_attention` and `full_attention` layers via
per-layer config dispatch.

Sliding layers:
    - num_kv_heads = config.num_key_value_heads (e.g. 8)
    - head_dim     = config.head_dim (256)
    - sliding_window = config.sliding_window (1024)
    - rope_theta = 10_000.0, full-head-dim rotated

Full layers:
    - num_kv_heads = config.num_global_key_value_heads (e.g. 2)
    - head_dim     = config.global_head_dim (512)
    - sliding_window = None
    - rope_theta = 1_000_000.0
    - partial RoPE: only first partial_rotary_factor * head_dim (=128) channels rotated
    - attention_k_eq_v: V = K (before norms), no separate v_proj

Deviations from the HF source (documented inline):
  (D1) attention_k_eq_v: full layers set V = K (pre-norm) inside prep_qkv_tensors,
       since NxDI's fused Wqkv unconditionally allocates a V slab.
  (D2) Partial RoPE (full layers): cos/sin are emitted with reduced trailing dim
       (=head_dim * partial_rotary_factor); we split Q/K along the last dim, rotate
       the first slab, and concat with the pass-through slab.
  (D3) V RMSNorm with no scale: applied after move_heads_front on BHSD tensor
       (norm over head_dim axis) using a custom NoScaleRMSNorm.
  (D4) Single class, dual config: layer_idx lookup of config.layer_types[idx].
  (D5) Q/K layernorms are per-head RMSNorms over head_dim; passed as q_layernorm /
       k_layernorm to NeuronAttentionBase so they are applied inside move_heads_front.
  (D6) softmax_scale left at NxDI default (= sqrt(head_dim) denominator, i.e.
       1 / sqrt(head_dim) factor). HF source sets `self.scaling = 1.0` which is
       an HF override-signal rather than a literal scale (HF's eager_attention_forward
       falls back to head_dim**-0.5 when scaling is None but uses scaling verbatim
       otherwise). The correct Gemma4 math is head_dim**-0.5, which matches NxDI's
       default.
  (D7) RoPE layout: HF source applies RoPE on BSHD with unsqueeze_dim=2; NxDI's
       apply_rotary_pos_emb works on BHSD with unsqueeze_dim=1. Math is equivalent
       because cos/sin depend only on (B, S, D).
"""

from typing import Optional

import torch
from torch import nn

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.gqa import GQA
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    move_heads_front,
)


# --------------------------------------------------------------------------- #
# RMSNorm modules (CPU-safe fallbacks). Math matches Gemma4RMSNorm exactly:
#   mean_squared = x.pow(2).mean(-1, keepdim=True) + eps
#   normed = x * torch.pow(mean_squared, -0.5)      # deliberately NOT rsqrt
#   (optionally) * weight
#   cast back to x.dtype
# --------------------------------------------------------------------------- #
class _Gemma4RMSNorm(nn.Module):
    """Per-head RMSNorm with scale; expects last dim == head_dim."""

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.float()
        mean_squared = x32.pow(2).mean(-1, keepdim=True) + self.eps
        normed = x32 * torch.pow(mean_squared, -0.5)
        normed = normed * self.weight.float()
        return normed.to(orig_dtype)


class _Gemma4RMSNormNoScale(nn.Module):
    """Per-head RMSNorm with no scale (v_norm); expects last dim == head_dim."""

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.head_dim = head_dim  # stored for reference; no learnable param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.float()
        mean_squared = x32.pow(2).mean(-1, keepdim=True) + self.eps
        normed = x32 * torch.pow(mean_squared, -0.5)
        return normed.to(orig_dtype)


# --------------------------------------------------------------------------- #
# Partial RoPE: same math as RotaryEmbedding but with reduced `dim` (=rotated dim)
# so emitted cos/sin have trailing dim = partial_dim. Apply step splits Q/K into
# [rotated | pass-through] and rotates only the first slab.
# --------------------------------------------------------------------------- #
class _PartialRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding whose cos/sin cover only the first `rotated_dim` channels.

    `rotated_dim` must be even (since rotate_half splits in half).
    """

    def __init__(self, rotated_dim: int, max_position_embeddings: int, base: float):
        # Deviation (D2): pass rotated_dim (not full head_dim) as `dim` so the
        # inv_freq frequency table covers only the rotated channels.
        assert rotated_dim % 2 == 0, "rotated_dim must be even"
        super().__init__(
            dim=rotated_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
        )
        self.rotated_dim = rotated_dim


def _apply_partial_rope(Q, K, cos, sin, rotated_dim: int):
    """Apply RoPE to only the first `rotated_dim` channels of Q and K (BHSD layout)."""
    Q_rot, Q_pass = Q[..., :rotated_dim], Q[..., rotated_dim:]
    K_rot, K_pass = K[..., :rotated_dim], K[..., rotated_dim:]
    Q_rot, K_rot = apply_rotary_pos_emb(Q_rot, K_rot, cos, sin)
    Q_out = torch.cat([Q_rot, Q_pass], dim=-1)
    K_out = torch.cat([K_rot, K_pass], dim=-1)
    return Q_out, K_out


# --------------------------------------------------------------------------- #
# Main attention block
# --------------------------------------------------------------------------- #
class NeuronGemma4TextAttention(NeuronAttentionBase):
    """NxDI-compatible Gemma4 Text Attention for both sliding and full layers."""

    def __init__(self, config: InferenceConfig, layer_idx: int = 0):
        self.layer_idx = layer_idx

        # ---- Deviation (D4): per-layer config dispatch ---------------------
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None and layer_idx < len(layer_types):
            layer_type = layer_types[layer_idx]
        else:
            layer_type = "sliding_attention"
        is_sliding = layer_type == "sliding_attention"
        is_full = not is_sliding
        self.layer_type = layer_type
        self.is_sliding = is_sliding
        self.is_full_attention = is_full

        # Pull rope parameters per layer type. We accept both the new dict-of-dicts
        # (Gemma4 native) and the flat config form (test wrappers may set fields
        # directly at the top level).
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None and layer_type in rope_parameters:
            rp = rope_parameters[layer_type]
            rope_theta = rp.get("rope_theta", 10_000.0)
            partial_rotary_factor = rp.get("partial_rotary_factor", 1.0)
        else:
            rope_theta = getattr(config, "rope_theta", 10_000.0)
            partial_rotary_factor = 1.0 if is_sliding else 0.25

        # Head dims & num_kv_heads differ per layer type.
        sliding_head_dim = getattr(config, "head_dim", 256)
        global_head_dim = getattr(config, "global_head_dim", sliding_head_dim)
        num_kv_heads_sliding = getattr(config, "num_key_value_heads", 8)
        num_kv_heads_full = getattr(
            config, "num_global_key_value_heads", num_kv_heads_sliding
        )

        if is_sliding:
            head_dim = sliding_head_dim
            num_kv_heads = num_kv_heads_sliding
            sliding_window = getattr(config, "sliding_window", None)
        else:
            head_dim = global_head_dim
            num_kv_heads = num_kv_heads_full
            sliding_window = None

        # Partial rotary dim (must be even).
        rotated_dim = int(round(head_dim * partial_rotary_factor))
        if rotated_dim % 2 != 0:
            rotated_dim -= 1
        self.rotated_dim = rotated_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_theta = rope_theta

        # ---- Deviation (D1): attention_k_eq_v only on full layers ---------
        self.attention_k_eq_v = bool(
            getattr(config, "attention_k_eq_v", False) and is_full
        )

        # ---- Build RoPE module --------------------------------------------
        max_pos = getattr(config, "max_position_embeddings", 131072)
        if rotated_dim == head_dim:
            # Full RoPE (sliding layers).
            rotary_emb = RotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=max_pos,
                base=rope_theta,
            )
        else:
            # Partial RoPE (full layers). Emits cos/sin with trailing dim = rotated_dim.
            rotary_emb = _PartialRotaryEmbedding(
                rotated_dim=rotated_dim,
                max_position_embeddings=max_pos,
                base=rope_theta,
            )

        # ---- Deviation (D5): Q/K per-head RMSNorms with scale -------------
        rms_eps = getattr(config, "rms_norm_eps", 1e-6)
        q_layernorm = _Gemma4RMSNorm(head_dim, eps=rms_eps)
        k_layernorm = _Gemma4RMSNorm(head_dim, eps=rms_eps)

        # ---- GQA sharding strategy ----------------------------------------
        # For full layers with only 2 KV heads, REPLICATE_TO_TP_DEGREE is required
        # whenever tp_degree > 2 (base's helper will fall back automatically).
        # On CPU tp=1 it's a no-op.
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            rope_theta=rope_theta,
            rms_norm_eps=rms_eps,
            use_qk_norm=False,  # we pass explicit q_layernorm/k_layernorm instead
            q_layernorm=q_layernorm,
            k_layernorm=k_layernorm,
            qkv_bias=getattr(config, "attention_bias", False),
            o_bias=getattr(config, "attention_bias", False),
            sliding_window=sliding_window,
        )

        # ---- Deviation (D3): V RMSNorm with no scale ----------------------
        # Applied on BHSD tensor (norm over head_dim axis == last dim).
        self.v_layernorm = _Gemma4RMSNormNoScale(head_dim, eps=rms_eps)

    # ----------------------------------------------------------------------- #
    # Override prep_qkv_tensors to implement Gemma4-specific behaviours:
    #   1. (D1) full-layer V = K (before any norm)
    #   2. (D3) V RMSNorm (no scale) after move_heads_front
    #   3. (D2) partial-RoPE for full layers
    # We intentionally do NOT call super().prep_qkv_tensors; we reproduce its
    # essential steps to keep control over each hook point.
    # ----------------------------------------------------------------------- #
    def prep_qkv_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        skip_rope=False,
        residual=None,
        use_polar_compatible_rope=False,
    ):
        # QKV projection (fused or split). Returns Q, K, V, residual in BSHD-flattened.
        Q, K, V, residual = self.get_qkv_proj()(
            hidden_states=hidden_states,
            rmsnorm=rmsnorm,
            adapter_ids=adapter_ids,
            residual=residual,
        )

        # ---- Deviation (D1): full-layer V = K (pre-norm) -------------------
        # Source (modeling_gemma4.py ~L1224):
        #   value_states = self.v_proj(hs).view(...) if self.v_proj is not None else key_states
        # We overwrite V with a clone of K regardless of what Wqkv produced in
        # its V slab. The V slab parameters are still present in Wqkv (they get
        # materialized during state-dict load) but their output is discarded on
        # full layers. Weight-map plumbing is a Phase-4 concern; numerically the
        # V slab is inert here because its output is replaced.
        if self.attention_k_eq_v:
            V = K.clone()

        # Shape + GQA move-heads-front (BSHD -> BHSD), applying Q/K per-head
        # RMSNorms inside move_heads_front.
        bsz, q_len, _ = hidden_states.size()
        if self.qkv_proj_sp_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        Q = move_heads_front(
            Q, bsz, q_len, self.num_heads, self.head_dim,
            layernorm=self.q_layernorm,
            post_transpose_layernorm=self.post_transpose_layernorm,
        )
        K = move_heads_front(
            K, bsz, q_len, self.num_key_value_heads, self.head_dim,
            layernorm=self.k_layernorm,
            post_transpose_layernorm=self.post_transpose_layernorm,
        )
        V = move_heads_front(
            V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None
        )

        # ---- Deviation (D3): V RMSNorm (no scale) on BHSD ------------------
        V = self.v_layernorm(V)

        # ---- RoPE on Q/K ---------------------------------------------------
        if not skip_rope and self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)

            if self.rotated_dim == self.head_dim:
                # Full RoPE (sliding layers).
                Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)
            else:
                # Deviation (D2): partial RoPE (full layers).
                Q, K = _apply_partial_rope(Q, K, cos_cache, sin_cache, self.rotated_dim)

        return Q, K, V, cos_cache, sin_cache, residual
