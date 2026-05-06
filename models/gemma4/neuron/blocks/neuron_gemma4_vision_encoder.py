"""
NeuronGemma4VisionEncoderLayer — NxDI translation of Gemma4 Vision ViT block.

Implements the non-causal ViT-style transformer inside Gemma4's vision tower.
Scope: attention + MLP + 4-RMSNorm sandwich + multi-dim RoPE. Excludes patch
embedder / pooler / projector (those are Block F).

HF sources (transformers/models/gemma4/modeling_gemma4.py):
    Gemma4VisionAttention, Gemma4VisionMLP, Gemma4VisionEncoderLayer,
    Gemma4VisionRotaryEmbedding, apply_multidimensional_rope.

Deviations from the HF source (documented inline):
  (D1) Multi-dim RoPE: we re-implement Gemma4VisionRotaryEmbedding.forward as a
       standalone nn.Module whose behaviour exactly matches the HF reference —
       inv_freq over `spatial_dim = head_dim // 2` (= 36 for real ckpt), forward
       takes `position_ids: [B, P, 2]` and returns (cos, sin) of shape
       [B, P, head_dim]. Apply-RoPE is a pure tensor op: split Q/K along head_dim
       into ndim=2 halves of size head_dim/2, rotate each half with the matching
       cos/sin half, concat back.
  (D2) V-RMSNorm with NO scale: Gemma4's v_norm has with_scale=False. We apply
       it on the BHSD tensor (after move-heads-front) using NoScaleRMSNorm.
  (D3) GQA is trivial (num_heads == num_kv_heads == 16). We route through plain
       ColumnParallelLinear / RowParallelLinear rather than NeuronAttentionBase,
       because vision is non-causal with no KV-cache / generation — going through
       NeuronAttentionBase would require working around its cache machinery and
       [B, P, 2] position_ids would break its rotary interface. This matches the
       Pixtral pattern of a minimal vision attention (see modeling_pixtral_vision)
       but further simplified because we don't reuse NeuronAttentionBase here.
  (D4) Non-causal + no sliding window. Attention mask comes from the caller
       (encoder builds it from padding_positions). We add the mask as an additive
       bias (0 for keep, -inf for mask) before softmax, which is what HF's
       eager_attention_forward does when the caller passes an additive mask.
       We accept either bool/int (1=keep) or float (additive) masks.
  (D5) Gemma4ClippableLinear strip: for this checkpoint use_clipped_linears=False,
       so the HF path is a plain nn.Linear. We replace with ColumnParallelLinear
       / RowParallelLinear. The Phase-4 state-dict converter must ignore
       input_min / input_max / output_min / output_max buffers (they only exist
       when use_clipped_linears=True, which is not the case here).
  (D6) 4-norm sandwich (like Gemma3/Gemma4 text): input_layernorm,
       post_attention_layernorm, pre_feedforward_layernorm,
       post_feedforward_layernorm. Pixtral has only 2 norms — we emulate the
       Gemma 4-norm shape.
  (D7) Gemma4VisionAttention sets `self.scaling = 1.0` in the HF source. HF's
       eager_attention_forward uses that scaling verbatim (not the conventional
       head_dim**-0.5). We faithfully reproduce `scaling = 1.0` so that QK^T is
       multiplied by 1.0 (no division by sqrt(head_dim)). This is a Gemma4
       vision-specific quirk — see modeling_gemma4.Gemma4VisionAttention L880.
  (D8) RMSNorm math: Gemma4RMSNorm uses `x * pow(mean_sq + eps, -0.5)` (not
       rsqrt) and upcasts to fp32 mid-norm; we mirror this exactly in both
       _Gemma4VisionRMSNorm (with scale) and _NoScaleRMSNorm (v_norm).
"""

from typing import Optional

import torch
from torch import nn

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed_inference.models.config import InferenceConfig

try:
    # Prefer ACT2FN from transformers when available (matches HF semantics).
    from transformers.activations import ACT2FN
    _ACT2FN = ACT2FN
except Exception:  # pragma: no cover
    _ACT2FN = None


def _gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    # Matches ACT2FN["gelu_pytorch_tanh"]. Provided as fallback so this block
    # does not hard-depend on transformers at import time.
    return torch.nn.functional.gelu(x, approximate="tanh")


def _get_activation(name: str):
    if _ACT2FN is not None and name in _ACT2FN:
        return _ACT2FN[name]
    if name == "gelu_pytorch_tanh":
        return _gelu_pytorch_tanh
    raise ValueError(f"Unsupported activation: {name}")


# --------------------------------------------------------------------------- #
# RMSNorm modules — math mirrors Gemma4RMSNorm exactly (Deviation D8):
#   mean_sq = x.pow(2).mean(-1, keepdim=True) + eps
#   normed  = x * torch.pow(mean_sq, -0.5)       # deliberately NOT rsqrt
#   (optionally) * weight
#   cast back to x.dtype
# --------------------------------------------------------------------------- #
class _Gemma4VisionRMSNorm(nn.Module):
    """RMSNorm with learned scale; used for input/post_attn/pre_ff/post_ff norms
    (over hidden_size) and for q_norm / k_norm (over head_dim)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.float()
        mean_sq = x32.pow(2).mean(-1, keepdim=True) + self.eps
        normed = x32 * torch.pow(mean_sq, -0.5)
        normed = normed * self.weight.float()
        return normed.to(orig_dtype)


class _NoScaleRMSNorm(nn.Module):
    """RMSNorm with NO scale parameter (Deviation D2: v_norm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim  # kept for reference; no learnable param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.float()
        mean_sq = x32.pow(2).mean(-1, keepdim=True) + self.eps
        normed = x32 * torch.pow(mean_sq, -0.5)
        return normed.to(orig_dtype)


# --------------------------------------------------------------------------- #
# Multi-dimensional rotary embedding (Deviation D1)
# --------------------------------------------------------------------------- #
class Gemma4VisionRotaryEmbedding(nn.Module):
    """Mirrors HF Gemma4VisionRotaryEmbedding exactly.

    inv_freq is computed over spatial_dim = head_dim // ndim (ndim=2),
    i.e. head_dim / 2. Forward takes `position_ids: [B, P, 2]` (2D patch coords)
    and returns (cos, sin) of shape [B, P, head_dim]. The trailing head_dim
    comes from concatenating [cos(h_freqs) | cos(w_freqs)] where each half has
    shape [B, P, head_dim / 2].
    """

    def __init__(self, head_dim: int, rope_theta: float = 100.0, ndim: int = 2):
        super().__init__()
        self.head_dim = head_dim
        self.ndim = ndim
        assert head_dim % (2 * ndim) == 0, (
            f"head_dim ({head_dim}) must be divisible by 2*ndim ({2 * ndim})"
        )
        # spatial_dim matches HF: head_dim // ndim. Each spatial dim gets its
        # own inv_freq table of size spatial_dim // 2.
        spatial_dim = head_dim // ndim
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, spatial_dim, 2, dtype=torch.int64).float()
                / spatial_dim
            )
        )  # shape: [spatial_dim // 2] = [head_dim // (2*ndim)]
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # position_ids: [B, P, ndim] (int64). x used only for dtype/device.
        B = position_ids.shape[0]
        inv_freq_exp = (
            self.inv_freq[None, :, None]
            .float()
            .expand(B, -1, 1)
            .to(position_ids.device)
        )  # [B, spatial_dim//2, 1]

        all_cos, all_sin = [], []
        for i in range(self.ndim):
            dim_pos = position_ids[:, :, i].float()  # [B, P]
            dim_pos_exp = dim_pos[:, None, :]  # [B, 1, P]
            # Force fp32 for the outer product & trig, matching HF.
            freqs = (inv_freq_exp.float() @ dim_pos_exp.float()).transpose(1, 2)
            # freqs: [B, P, spatial_dim//2]
            emb = torch.cat((freqs, freqs), dim=-1)  # [B, P, spatial_dim]
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
            all_cos.append(cos)
            all_sin.append(sin)

        cos = torch.cat(all_cos, dim=-1).to(dtype=x.dtype)  # [B, P, head_dim]
        sin = torch.cat(all_sin, dim=-1).to(dtype=x.dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, P, N, D_per_dim] (BSHD layout, HF's unsqueeze_dim=2),
    # cos/sin: [B, P, D_per_dim] -> unsqueeze head dim.
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (x * cos) + (_rotate_half(x) * sin)


def apply_multidimensional_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    ndim: int = 2,
) -> torch.Tensor:
    """Pure-tensor multi-dim RoPE (no dynamic control flow on tensor shapes).

    x: [B, P, N, head_dim]
    cos/sin: [B, P, head_dim]
    Splits x, cos, sin along the last dim into `ndim` equal halves and applies
    the standard RoPE rotation to each, then concatenates back. This exactly
    matches HF's apply_multidimensional_rope for num_rotated_channels_per_dim
    == head_dim / ndim (the case for Gemma4 vision: head_dim=72, ndim=2 -> 36).
    """
    D = x.shape[-1]
    per = D // ndim
    x_parts = torch.split(x, [per] * ndim, dim=-1)
    cos_parts = torch.split(cos, [per] * ndim, dim=-1)
    sin_parts = torch.split(sin, [per] * ndim, dim=-1)
    y_parts = [
        _apply_rotary(x_parts[k], cos_parts[k], sin_parts[k]) for k in range(ndim)
    ]
    return torch.cat(y_parts, dim=-1)


# --------------------------------------------------------------------------- #
# Parallel-linear helpers that fall back to plain nn.Linear when NxD parallel
# state is not initialized (happens on CPU tests run outside tracer).
# --------------------------------------------------------------------------- #
def _col_linear(in_f: int, out_f: int, dtype: torch.dtype) -> nn.Module:
    try:
        return ColumnParallelLinear(
            in_f, out_f, bias=False, gather_output=True, dtype=dtype
        )
    except AssertionError:
        # parallel state not initialized (e.g. pure CPU eval) -> plain Linear.
        lin = nn.Linear(in_f, out_f, bias=False)
        return lin.to(dtype=dtype)


def _row_linear(in_f: int, out_f: int, dtype: torch.dtype) -> nn.Module:
    try:
        return RowParallelLinear(
            in_f, out_f, bias=False, input_is_parallel=False, dtype=dtype
        )
    except AssertionError:
        lin = nn.Linear(in_f, out_f, bias=False)
        return lin.to(dtype=dtype)


# --------------------------------------------------------------------------- #
# Vision MLP (Deviation D5: clippable linears stripped)
# --------------------------------------------------------------------------- #
class NeuronGemma4VisionMLP(nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        dtype = config.neuron_config.torch_dtype
        hidden_activation = getattr(config, "hidden_activation", "gelu_pytorch_tanh")

        self.gate_proj = _col_linear(hidden_size, intermediate_size, dtype)
        self.up_proj = _col_linear(hidden_size, intermediate_size, dtype)
        self.down_proj = _row_linear(intermediate_size, hidden_size, dtype)
        self.act_fn = _get_activation(hidden_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------- #
# Vision Attention (Deviations D3, D4, D7)
# --------------------------------------------------------------------------- #
class NeuronGemma4VisionAttention(nn.Module):
    """Non-causal multi-head attention for Gemma4 vision tower.

    Trivially-MHA (num_heads == num_kv_heads == 16 for real ckpt). Written as
    a standalone module rather than inheriting NeuronAttentionBase because:
      * vision is non-causal with NO KV-cache / generation; NeuronAttentionBase
        would require bypassing its cache machinery.
      * HF's multi-dim RoPE takes position_ids of shape [B, P, 2], which does
        not fit NeuronAttentionBase's rotary_emb signature.
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(
            config, "num_key_value_heads", config.num_attention_heads
        )
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.num_heads
        )
        self.rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        # Deviation D7: Gemma4 vision uses scaling = 1.0 (not head_dim ** -0.5).
        # HF's eager_attention_forward uses the value verbatim when provided.
        self.scaling = 1.0
        dtype = config.neuron_config.torch_dtype

        # Deviation D5: plain parallel linears (no clipping).
        self.q_proj = _col_linear(
            self.hidden_size, self.num_heads * self.head_dim, dtype
        )
        self.k_proj = _col_linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, dtype
        )
        self.v_proj = _col_linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, dtype
        )
        self.o_proj = _row_linear(
            self.num_heads * self.head_dim, self.hidden_size, dtype
        )

        # Q/K RMSNorms (with scale) over head_dim.
        self.q_norm = _Gemma4VisionRMSNorm(self.head_dim, eps=self.rms_norm_eps)
        self.k_norm = _Gemma4VisionRMSNorm(self.head_dim, eps=self.rms_norm_eps)
        # Deviation D2: v_norm has NO scale.
        self.v_norm = _NoScaleRMSNorm(self.head_dim, eps=self.rms_norm_eps)

    def _reshape_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        # [B, P, num_heads * head_dim] -> [B, P, num_heads, head_dim] (BSHD).
        B, P, _ = x.shape
        return x.view(B, P, num_heads, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # hidden_states: [B, P, hidden_size]
        # position_embeddings: (cos, sin), each [B, P, head_dim]
        # attention_mask: [B, 1, P, P] — either bool/int (1=keep) or additive float.
        B, P, _ = hidden_states.shape
        cos, sin = position_embeddings

        # QKV projections + per-head reshape (BSHD).
        q = self._reshape_heads(self.q_proj(hidden_states), self.num_heads)
        k = self._reshape_heads(self.k_proj(hidden_states), self.num_kv_heads)
        v = self._reshape_heads(self.v_proj(hidden_states), self.num_kv_heads)

        # Per-head norms (on last dim = head_dim). HF applies q_norm/k_norm on
        # BSHD before RoPE; v_norm on BSHD (same result as BHSD since norm is
        # over the last axis).
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # Multi-dim RoPE on Q/K only (D1).
        q = apply_multidimensional_rope(q, cos, sin, ndim=2)
        k = apply_multidimensional_rope(k, cos, sin, ndim=2)

        # BSHD -> BHSD for SDPA.
        q = q.transpose(1, 2)  # [B, H, P, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention. Deviation D7: scaling = 1.0 (no /sqrt(d)).
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling

        # Deviation D4: non-causal mask comes from caller. Accept either
        # additive-float mask (HF-style: 0 for keep, -inf for pad) or bool/int
        # mask (1 for keep). We convert the latter on-the-fly.
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                mask_add = torch.zeros_like(attn_weights)
                mask_add = mask_add.masked_fill(
                    ~attention_mask, torch.finfo(attn_weights.dtype).min
                )
                attn_weights = attn_weights + mask_add
            elif attention_mask.dtype in (torch.int32, torch.int64, torch.uint8):
                # treat nonzero as keep
                keep = attention_mask.to(torch.bool)
                mask_add = torch.zeros_like(attn_weights)
                mask_add = mask_add.masked_fill(
                    ~keep, torch.finfo(attn_weights.dtype).min
                )
                attn_weights = attn_weights + mask_add
            else:
                # additive float mask: already 0 / -inf
                attn_weights = attn_weights + attention_mask.to(attn_weights.dtype)

        # Softmax upcast to fp32 then back, matching HF.
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)

        attn_output = torch.matmul(attn_weights, v)  # [B, H, P, D]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, P, H, D]
        attn_output = attn_output.reshape(B, P, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


# --------------------------------------------------------------------------- #
# Encoder Layer (Deviation D6: 4-norm sandwich)
# --------------------------------------------------------------------------- #
class NeuronGemma4VisionEncoderLayer(nn.Module):
    def __init__(self, config: InferenceConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        rms_eps = getattr(config, "rms_norm_eps", 1e-6)

        self.self_attn = NeuronGemma4VisionAttention(config)
        self.mlp = NeuronGemma4VisionMLP(config)

        self.input_layernorm = _Gemma4VisionRMSNorm(self.hidden_size, eps=rms_eps)
        self.post_attention_layernorm = _Gemma4VisionRMSNorm(
            self.hidden_size, eps=rms_eps
        )
        self.pre_feedforward_layernorm = _Gemma4VisionRMSNorm(
            self.hidden_size, eps=rms_eps
        )
        self.post_feedforward_layernorm = _Gemma4VisionRMSNorm(
            self.hidden_size, eps=rms_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings=None,
    ) -> torch.Tensor:
        # Deviation D6: 4-norm sandwich, mirroring HF Gemma4VisionEncoderLayer.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# --------------------------------------------------------------------------- #
# Full Encoder (27 layers + rotary ownership)
# --------------------------------------------------------------------------- #
class NeuronGemma4VisionEncoder(nn.Module):
    """Stack of `num_hidden_layers` vision encoder layers plus owned rotary_emb.

    Matches HF Gemma4VisionEncoder: builds cos/sin once per forward pass from
    `pixel_position_ids: [B, P, 2]` and feeds them (along with attention_mask)
    to each encoder layer.
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rope_theta = 100.0
        rope_parameters = getattr(config, "rope_parameters", None)
        if isinstance(rope_parameters, dict):
            rope_theta = rope_parameters.get("rope_theta", 100.0)
        self.rotary_emb = Gemma4VisionRotaryEmbedding(
            head_dim=head_dim, rope_theta=rope_theta, ndim=2
        )
        self.layers = nn.ModuleList(
            [
                NeuronGemma4VisionEncoderLayer(config=config, layer_idx=i)
                for i in range(self.num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        position_embeddings = self.rotary_emb(hidden_states, pixel_position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
        return hidden_states
