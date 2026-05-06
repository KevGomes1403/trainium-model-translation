"""
NxDI translation of Gemma4 BLOCK C: Text embeddings + Per-Layer Input (PLE) system +
RMSNorm + final logit softcap.

This file packages the non-attention / non-MoE structural components of the
Gemma4 text path into Neuron-friendly modules:

    * NeuronGemma4ScaledWordEmbedding  -- ParallelEmbedding * sqrt(hidden_size)
    * NeuronGemma4RMSNorm              -- CPU-safe bf16/fp32-mixed RMSNorm with `with_scale`
                                          matching Gemma4RMSNorm numerics (pow(-0.5), JAX parity).
                                          Falls back to CustomRMSNorm on device via `get_rmsnorm_cls()`.
    * NeuronGemma4PLE                  -- Packaged token-identity + context-projection PLE path.
                                          No-ops when `hidden_size_per_layer_input == 0`.
    * apply_logit_softcap              -- Functional helper: tanh(logits/sc) * sc  (fp32 recommended).

Source HF references (/home/ubuntu/trainium-model-translation/models/gemma4/hf/modeling_gemma4.py):
    - Gemma4RMSNorm                                  L168-186
    - Gemma4TextScaledWordEmbedding                  L1424-1435
    - Gemma4TextModel.__init__ PLE construction      L1576-1591
    - Gemma4TextModel.get_per_layer_inputs           L1692-1734
    - Gemma4TextModel.project_per_layer_inputs       L1736-1769
    - Gemma4ForCausalLM.forward softcap              L1839-1842

-------------------------------------------------------------------------------
DEVIATIONS (each cited inline below):
  (1) Scaled word embedding: we multiply by sqrt(H) at the module output
      (matching HF L1434-1435). We do NOT fold the scale into the weight.
  (2) PLE optional: hidden_size_per_layer_input==0 => no-op (HF gates at L1577,
      L1632 and L1703/L1752). Returns None so downstream ignores PLE.
  (3) RMSNorm fp32 compute: forward casts to float32, uses `pow(-0.5)`
      (NOT rsqrt) per HF comment at modeling_gemma4.py L179 -- JAX parity.
  (4) RMSNorm +1.0 offset at load time: NOT APPLIED for Gemma4.
      HF Gemma3RMSNorm stores (1+w) so NxDI Gemma3 adds +1.0 at load
      (neuronx_distributed_inference/models/gemma3/modeling_gemma3.py L374).
      HF Gemma4RMSNorm (modeling_gemma4.py L182-186) is simply `normed * weight`
      with no +1 shift, and convert_gemma4_weights.py stores the norm weight
      verbatim (see paths at L671-681, L797-810, L952-954, L986-988 -- no
      addition). Hence `RMSNORM_OFFSET = 0.0`. See the flag below.
  (5) Logit softcap: apply_logit_softcap() runs in fp32 (call site upcasts).
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

# NxDI primitives
from neuronx_distributed.parallel_layers.layers import (  # type: ignore
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode  # type: ignore
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm  # type: ignore


# -----------------------------------------------------------------------------
# Deviation (4): RMSNorm +1.0 offset.
# Gemma4 checkpoint stores final scale directly (NO +1 shift).
# Phase 4 converter should NOT add 1.0 to any Gemma4 RMSNorm weight. Exposed
# as a module-level constant so downstream code can reference it unambiguously.
# -----------------------------------------------------------------------------
RMSNORM_OFFSET: float = 0.0


# =============================================================================
# RMSNorm
# =============================================================================
class NeuronGemma4RMSNorm(nn.Module):
    """
    CPU-safe Gemma4 RMSNorm mirroring HF `Gemma4RMSNorm` (modeling_gemma4.py L168-186).

    Deviation (3): fp32 compute with `pow(-0.5)` (NOT rsqrt) for JAX parity,
    then cast back to input dtype. Matches HF formula:

        y = x * pow(mean(x^2) + eps, -0.5)     # fp32
        if with_scale:  y = y * weight         # fp32
        return y.type_as(x)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            # Initialize to zeros; weights are loaded from checkpoint. No +1 offset (see RMSNORM_OFFSET).
            self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=torch.bfloat16))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # HF modeling_gemma4.py L178-180: pow(-0.5), NOT rsqrt.
        mean_squared = x.pow(2).mean(-1, keepdim=True) + self.eps
        return x * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        out = self._norm(hidden_states.float())
        if self.with_scale:
            out = out * self.weight.float()
        return out.to(original_dtype)


class NoScaleRMSNorm(NeuronGemma4RMSNorm):
    """Alias for `Gemma4RMSNorm(with_scale=False)` used by v_norm etc."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__(hidden_size=hidden_size, eps=eps, with_scale=False)


def get_rmsnorm_cls(with_scale: bool = True):
    """
    Select RMSNorm implementation:
      * CPU (including block-correctness tests):  NeuronGemma4RMSNorm -- matches HF numerics exactly.
      * Device:                                   CustomRMSNorm       -- calls AwsNeuronRmsNorm kernel.

    NOTE: CustomRMSNorm uses `rsqrt` rather than `pow(-0.5)`. Empirically these are
    numerically equivalent within bf16 tolerance (5e-3). If strict JAX parity is
    required on device, continue using NeuronGemma4RMSNorm (pure torch) instead.
    CustomRMSNorm does not support `with_scale=False`; for that case we always use
    NeuronGemma4RMSNorm.
    """
    if not with_scale:
        return NoScaleRMSNorm
    return NeuronGemma4RMSNorm if cpu_mode() else CustomRMSNorm


# =============================================================================
# Scaled word embedding
# =============================================================================
class NeuronGemma4ScaledWordEmbedding(nn.Module):
    """
    ParallelEmbedding * sqrt(embedding_dim) with the scale applied explicitly
    (Deviation 1: matches HF L1434-1435, which multiplies at forward-time in
    the embedding's dtype; scale is NOT folded into the weight).

    For `embed_tokens_per_layer` (PLE token-identity embedding), use
    embed_scale = sqrt(hidden_size_per_layer_input) and
    embedding_dim   = num_hidden_layers * hidden_size_per_layer_input.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        embed_scale: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        shard_across_embedding: bool = True,
        sequence_parallel_enabled: bool = False,
    ):
        super().__init__()
        self.embed_scale = float(embed_scale)
        self.embed_tokens = ParallelEmbedding(
            num_embeddings,
            embedding_dim,
            padding_idx,
            dtype=dtype,
            shard_across_embedding=shard_across_embedding,
            sequence_parallel_enabled=sequence_parallel_enabled,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        # Deviation (1): multiply in tensor dtype (bf16) -- matches HF.
        return x * torch.tensor(self.embed_scale, dtype=x.dtype, device=x.device)


# =============================================================================
# Per-Layer Input (PLE) module
# =============================================================================
class NeuronGemma4PLE(nn.Module):
    """
    Per-Layer Input (PLE) pipeline from Gemma4TextModel. Bundles:

        * embed_tokens_per_layer        (ParallelEmbedding * sqrt(ple_dim))  -- HF L1578-1583
        * per_layer_model_projection    (ColumnParallelLinear, bias=False)   -- HF L1585-1589
        * per_layer_projection_norm     (Gemma4RMSNorm)                      -- HF L1591
        * scalars: per_layer_input_scale = 1/sqrt(2)                         -- HF L1584
                   per_layer_model_projection_scale = 1/sqrt(hidden_size)    -- HF L1590

    Forward produces `per_layer_inputs` of shape [B, S, L, ple_dim].
    This is then indexed by decoder layer index `i` (HF L1672) and consumed
    by `Gemma4TextDecoderLayer` (HF L1411-1418).

    Deviation (2): if `hidden_size_per_layer_input == 0` the whole path is
    disabled (HF gates at L1577 / L1632 / L1703 / L1752) and `forward()`
    returns None -- downstream code must gate on `is_enabled`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        vocab_size_per_layer_input: int,
        hidden_size_per_layer_input: int,
        rms_norm_eps: float = 1e-6,
        padding_idx: Optional[int] = 0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.vocab_size_per_layer_input = vocab_size_per_layer_input
        self.is_enabled = hidden_size_per_layer_input > 0

        if not self.is_enabled:
            # Deviation (2): no parameters at all when disabled.
            return

        # HF L1584 + L1590.
        self.per_layer_input_scale = 2.0 ** -0.5
        self.per_layer_model_projection_scale = hidden_size ** -0.5

        # Token-identity PLE lookup -- HF L1578-1583.
        self.embed_tokens_per_layer = NeuronGemma4ScaledWordEmbedding(
            num_embeddings=vocab_size_per_layer_input,
            embedding_dim=num_hidden_layers * hidden_size_per_layer_input,
            padding_idx=padding_idx,
            embed_scale=math.sqrt(hidden_size_per_layer_input),
            dtype=dtype,
            shard_across_embedding=True,
        )

        # Context projection -- HF L1585-1589. Column-parallel since output dim is large.
        # `gather_output=True` so every rank has the full projection (we reshape it).
        self.per_layer_model_projection = ColumnParallelLinear(
            hidden_size,
            num_hidden_layers * hidden_size_per_layer_input,
            bias=False,
            gather_output=True,
            dtype=dtype,
        )

        # Projection norm -- HF L1591.
        norm_cls = get_rmsnorm_cls(with_scale=True)
        self.per_layer_projection_norm = norm_cls(
            hidden_size_per_layer_input, eps=rms_norm_eps
        )

    # -------------------------------------------------------------------------
    # Token-identity path. Mirrors `Gemma4TextModel.get_per_layer_inputs`
    # (HF L1692-1734). We only support the `input_ids is not None` path --
    # the input_embed-reverse path (HF L1711-1728) is unused for a plain
    # text-only Neuron forward.
    # -------------------------------------------------------------------------
    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.is_enabled:
            return None
        emb = self.embed_tokens_per_layer(input_ids)  # [B, S, L*ple]
        return emb.reshape(
            *input_ids.shape,
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    # -------------------------------------------------------------------------
    # Context-aware path + token-identity combination. Mirrors
    # `Gemma4TextModel.project_per_layer_inputs` (HF L1736-1769).
    # -------------------------------------------------------------------------
    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self.is_enabled:
            # Deviation (2): identity when PLE is disabled. Return None so the
            # caller skips PLE injection in decoder layers.
            return None

        proj = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        proj = proj.reshape(
            *inputs_embeds.shape[:-1],
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        proj = self.per_layer_projection_norm(proj)

        if per_layer_inputs is None:
            return proj
        return (proj + per_layer_inputs) * self.per_layer_input_scale

    # -------------------------------------------------------------------------
    # Convenience: `input_ids` -> [B, S, L, ple]
    # -------------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.is_enabled:
            return None
        ple = self.get_per_layer_inputs(input_ids)
        return self.project_per_layer_inputs(inputs_embeds, ple)


# =============================================================================
# Final logit softcap
# =============================================================================
def apply_logit_softcap(logits: torch.Tensor, softcap: float = 30.0) -> torch.Tensor:
    """
    Final logit softcap -- mirrors HF `Gemma4ForCausalLM.forward` (L1839-1842):

        logits = tanh(logits / softcap) * softcap

    Deviation (5): run in fp32 regardless of input dtype (call site often has fp32 lm_head
    output but we defensively upcast). Return dtype matches input.
    """
    if softcap is None:
        return logits
    original_dtype = logits.dtype
    x = logits.to(torch.float32)
    x = torch.tanh(x / softcap) * softcap
    return x.to(original_dtype)
