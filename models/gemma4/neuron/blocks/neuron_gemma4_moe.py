"""
NeuronGemma4FFN: NxDI port of Gemma4's feed-forward block.

Gemma4 text decoder runs BOTH a dense MLP and an MoE branch in parallel on the
same pre-FFN input, then sums their post-norm outputs. This is unusual vs. other
MoE LLMs (e.g. Qwen3-MoE) where MoE *replaces* the dense MLP.

Source reference: `Gemma4TextDecoderLayer.forward` (FFN block) in
    models/gemma4/hf/modeling_gemma4.py  (approx L1390-L1409):

        residual = hidden_states                                   # after self-attn residual add
        hidden_states = self.pre_feedforward_layernorm(hidden_states)  # shared norm input
        hidden_states = self.mlp(hidden_states)                    # dense GLU
        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

            # Take hidden states *before MLP* here (i.e. the already-post-attn residual,
            # NOT the pre_feedforward_layernorm output). NOTE carefully: residual is
            # `hidden_states` prior to pre_feedforward_layernorm, so the MoE branch
            # sees the un-normed post-attn output.
            hidden_states_flat = residual.reshape(-1, residual.shape[-1])
            _, top_k_weights, top_k_index = self.router(hidden_states_flat)
            hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
            hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
            hidden_states_2 = hidden_states_2.reshape(residual.shape)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

            hidden_states = hidden_states_1 + hidden_states_2
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return hidden_states   # residual add happens outside

This module (`NeuronGemma4FFN`) owns everything inside `residual = ...` down to
`post_feedforward_layernorm` application. Its input is the *post-attn residual*
(i.e. the decoder's `residual`) and its output is the post_feedforward_layernorm
result that the decoder will subsequently add to that residual.

NOTE on router semantics (see source `Gemma4TextRouter`):
    - pre-RMSNorm(no_scale)
    - multiply by learnable `scale[H]` and by scalar 1/sqrt(H)
    - Linear(H -> E)
    - softmax over E
    - topk (weights + indices)
    - sum-normalize weights to 1 per token
    - multiply weights by per_expert_scale[index]
Stock NxDI RouterTopK only supports: Linear + softmax + topk (+ optional norm).
The pre-norm + scale × 1/sqrt(H) is non-linearly dependent on input norm and
CANNOT be absorbed into the router weight as a constant fold because RMSNorm is
input-dependent.

DEVIATIONS IMPLEMENTED (see README of task):
  (1) Parallel dense + MoE sum topology, with NO pre_feedforward_layernorm_2
      between the dense and MoE branches' inputs — they start from different
      inputs (MoE branch takes `residual` before pre_feedforward_layernorm).
  (2) Custom router: subclass RouterTopK so NxDI MoE plumbing still works, but
      override `forward()` to perform the full Gemma4 router compute. Returns
      `(router_logits, expert_affinities_full, expert_index)` where
      `expert_affinities_full: (T, E)` has final top-k weights (post normalize
      & per-expert-scale) scattered at `expert_index` and 0 elsewhere. We set
      `neuron_config.normalize_top_k_affinities = False` so ExpertMLPsV2 will
      NOT re-normalize these weights.
  (3) Router math runs in fp32 via `neuron_config.router_config.dtype=float32`.
  (4) Expert gate_up_proj shape: HF=(E, 2I, H), NxDI=(E, H, 2I). Transpose
      (1,2) at checkpoint-sync time. Similarly down_proj HF=(E, H, I),
      NxDI=(E, I, H).
  (5) GLU activation is `gelu_pytorch_tanh` — passed via `config.hidden_act`.
"""

import os
import math
from typing import Optional

import torch
from torch import nn

from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.utils import cpu_mode
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from transformers.activations import ACT2FN


# -----------------------------------------------------------------------------
# RMSNorm choice
# -----------------------------------------------------------------------------
# CustomRMSNorm uses an XLA custom call (AwsNeuronRmsNorm) that is unavailable on
# pure CPU runs. However, during NxDI compilation the `on_cpu` flag is set and
# build_module executes the module on torch_xla's CPU device, where the custom
# call IS resolved. So we default to CustomRMSNorm. For testing, we also provide
# `_Gemma4CPURMSNorm` for the PyTorch reference (matches HF math exactly — note
# Gemma4 has *two* RMSNorm variants: with_scale=True and with_scale=False; the
# router uses with_scale=False).


class _Gemma4CPURMSNorm(nn.Module):
    """CPU-fallback RMSNorm matching HF Gemma4RMSNorm exactly (fp32 internally)."""

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        x = hidden_states.float()
        ms = x.pow(2).mean(-1, keepdim=True) + self.eps
        x = x * torch.pow(ms, -0.5)
        if self.with_scale:
            x = x * self.weight.float()
        return x.to(orig_dtype)


def _rmsnorm(dim: int, eps: float) -> nn.Module:
    """RMSNorm with scale=True: use CustomRMSNorm on device, CPU fallback on CPU."""
    if cpu_mode():
        return _Gemma4CPURMSNorm(dim, eps=eps, with_scale=True)
    return CustomRMSNorm(hidden_size=dim, eps=eps)


# -----------------------------------------------------------------------------
# Custom Gemma4 Router
# -----------------------------------------------------------------------------


class Gemma4Router(RouterTopK):
    """
    Subclass RouterTopK to preserve the `MoE`-expected interface
    `forward(hidden_states) -> (router_logits, expert_affinities, expert_index)`
    while implementing full Gemma4 router semantics:

        x = rmsnorm_noscale(x)
        x = x * scale[H] * (1 / sqrt(H))
        logits = linear(x)
        probs = softmax(logits)
        top_w, top_i = topk(probs, k)
        top_w = top_w / top_w.sum(-1, keepdim=True)
        top_w = top_w * per_expert_scale[top_i]
        -> scatter top_w into expert_affinities_full at top_i

    The scatter allows ExpertMLPsV2.get_expert_affinities_masked() to pick up
    the correct combine weight for each chosen expert. `normalize_top_k_affinities`
    MUST be False on the enclosing config so those scattered weights are used
    verbatim without re-normalization.

    Note: `RouterTopK.__init__` creates `self.linear_router = nn.Linear(H, E)`,
    which is exactly what Gemma4's `Gemma4TextRouter.proj` is. We reuse it.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int,
                 rms_norm_eps: float, dtype: torch.dtype,
                 sequence_parallel_enabled: bool = False,
                 tensor_model_parallel_group=None):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            sequence_parallel_enabled=sequence_parallel_enabled,
            sequence_dimension=1,
            dtype=dtype,
            device=torch.device("cpu"),
            bias=False,
            tensor_model_parallel_group=tensor_model_parallel_group,
            act_fn="softmax",
            apply_act_fn_over_topk=False,
            jitter_eps=0.0,
            store_transposed_weights=False,
        )
        self.rms_norm_eps = rms_norm_eps
        self.scalar_root_size = hidden_size ** -0.5
        # Gemma4-specific extras (no nonlinear fold into linear_router.weight is
        # possible because the rmsnorm normalizer is input-dependent).
        # `norm` has NO scale parameter (with_scale=False).
        self.norm = _Gemma4CPURMSNorm(hidden_size, eps=rms_norm_eps, with_scale=False)
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(num_experts))

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: (B, S, H) — NOT flattened; MoE.forward passes the
                full tensor. We flatten to (T, H) internally.
        Returns:
            router_logits:   (T, E)        in float (fp32 if configured)
            expert_affinities: (T, E)      combine weights, 0 except at top-k
            expert_index:    (T, top_k)    long
        """
        # Flatten to (T, H). MoE._forward_compute_bound will later
        # reshape expert outputs back via the hidden_states shape.
        orig_shape = hidden_states.shape
        T = orig_shape[0] * orig_shape[1] if hidden_states.dim() == 3 else orig_shape[0]
        x = hidden_states.reshape(T, -1)

        # Do all router math in configured router dtype (fp32 for accuracy),
        # matching Gemma4 HF path (norm/linear are effectively fp32 inside norm).
        router_dtype = self.linear_router.weight.dtype
        x = x.to(router_dtype)

        # 1. RMSNorm (no scale)
        x = self.norm(x)
        # 2. * scale[H] * (1/sqrt(H))
        x = x * self.scale.to(router_dtype) * self.scalar_root_size
        # 3. Linear projection to expert scores
        router_logits = self.linear_router(x)                 # (T, E)
        # 4. Softmax over experts
        probs = torch.softmax(router_logits, dim=-1)          # (T, E)
        # 5. Top-k
        top_w, top_i = torch.topk(probs, k=self.top_k, dim=-1)
        # 6. Sum-normalize
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        # 7. Multiply by per-expert scale
        top_w = top_w * self.per_expert_scale[top_i].to(router_dtype)

        # Scatter back into a full (T, E) tensor of combine weights.
        # This lets ExpertMLPsV2.get_expert_affinities_masked use them verbatim
        # (provided normalize_top_k_affinities=False).
        expert_affinities_full = torch.zeros_like(probs)
        expert_affinities_full.scatter_(1, top_i, top_w)

        # Cast to input dtype as RouterBase does.
        out_dtype = hidden_states.dtype
        expert_affinities_full = expert_affinities_full.to(out_dtype)
        expert_index = top_i.detach().to(torch.long)

        return router_logits, expert_affinities_full, expert_index


# -----------------------------------------------------------------------------
# Dense MLP branch (GLU with gate_proj/up_proj/down_proj, gelu_pytorch_tanh)
# -----------------------------------------------------------------------------


class _Gemma4DenseMLP(nn.Module):
    """
    Dense GLU MLP matching HF `Gemma4TextMLP`: down_proj(act(gate_proj(x)) * up_proj(x)).

    Mirrors NeuronLlamaMLP's structure (ColumnParallel gate/up, RowParallel down)
    when model-parallel is initialized; falls back to plain nn.Linear on CPU test
    path (matches NeuronLlamaMLP lines 410-413 in modeling_llama.py).
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 hidden_activation: str, dtype: torch.dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.act_fn = ACT2FN[hidden_activation]

        if parallel_state.model_parallel_is_initialized():
            self.gate_proj = ColumnParallelLinear(
                hidden_size, intermediate_size, bias=False,
                gather_output=False, dtype=dtype, pad=True,
            )
            self.up_proj = ColumnParallelLinear(
                hidden_size, intermediate_size, bias=False,
                gather_output=False, dtype=dtype, pad=True,
            )
            self.down_proj = RowParallelLinear(
                intermediate_size, hidden_size, bias=False,
                input_is_parallel=True, dtype=dtype, pad=True,
            )
        else:
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# -----------------------------------------------------------------------------
# Main FFN module
# -----------------------------------------------------------------------------


class NeuronGemma4FFN(nn.Module):
    """
    Neuron port of Gemma4's parallel Dense-MLP + MoE FFN.

    Required config attributes:
        hidden_size, intermediate_size        (dense MLP width)
        moe_intermediate_size                 (per-expert MLP width)
        num_experts, num_local_experts, num_experts_per_tok
        rms_norm_eps
        hidden_act                            (= "gelu_pytorch_tanh" for Gemma4)
        n_shared_experts = 0                  (Gemma4 has NO shared experts)
        neuron_config.torch_dtype
        neuron_config.router_config.dtype     (fp32)
        neuron_config.normalize_top_k_affinities = False  (Gemma4 normalizes in router)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        nc = config.neuron_config

        H = config.hidden_size
        eps = config.rms_norm_eps
        dtype = nc.torch_dtype

        # --- 4 RMSNorms in the FFN block ---
        # Deviation (1): Gemma4 has *4* FFN-related RMSNorms (vs. 2 normal).
        # All are with_scale=True.
        self.pre_feedforward_layernorm = _rmsnorm(H, eps)
        self.pre_feedforward_layernorm_2 = _rmsnorm(H, eps)
        self.post_feedforward_layernorm_1 = _rmsnorm(H, eps)
        self.post_feedforward_layernorm_2 = _rmsnorm(H, eps)
        # Final post-FFN norm applied to the sum of the two branches, before
        # residual add in the caller.
        self.post_feedforward_layernorm = _rmsnorm(H, eps)

        # --- Dense branch ---
        self.mlp = _Gemma4DenseMLP(
            hidden_size=H,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_act,
            dtype=dtype,
        )

        # --- MoE branch ---
        # Deviation (3): router fp32 via config.neuron_config.router_config.dtype.
        # Deviation (2) setup: override router after module creation.
        # We *also* force config.intermediate_size used by initialize_moe_module
        # to refer to the PER-EXPERT intermediate size (see Qwen3MoE pattern
        # `self.intermediate_size = self.moe_intermediate_size`). The caller
        # must set `config.intermediate_size == config.moe_intermediate_size`
        # by the time this is called, OR provide both and we temporarily swap.
        # Here we just temporarily swap so the dense branch can use its own I.
        dense_I = config.intermediate_size
        moe_I = config.moe_intermediate_size
        config.intermediate_size = moe_I
        try:
            self.moe = initialize_moe_module(config=config)
        finally:
            config.intermediate_size = dense_I

        # Deviation (2): Swap in our custom Gemma4Router (keeps RouterTopK API).
        self.moe.router = Gemma4Router(
            hidden_size=H,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            rms_norm_eps=eps,
            dtype=nc.router_config.dtype,
            sequence_parallel_enabled=nc.sequence_parallel_enabled,
            tensor_model_parallel_group=None,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, S, H) — the *post-attn residual* (i.e. the decoder's
                `residual` value BEFORE pre_feedforward_layernorm).
        Returns:
            (B, S, H) — the post_feedforward_layernorm output. The caller adds
            the residual.
        """
        residual_input = hidden_states  # alias for clarity; this is HF's `residual`
        B, S, H = residual_input.shape

        # Dense branch: pre_feedforward_layernorm -> MLP -> post_feedforward_layernorm_1
        dense_in = self.pre_feedforward_layernorm(residual_input)
        dense_out = self.mlp(dense_in)
        dense_out = self.post_feedforward_layernorm_1(dense_out)

        # MoE branch -- DEVIATION:
        # HF runs the router on the UN-normed residual, but the experts on
        # pre_feedforward_layernorm_2(residual). Stock NxDI MoE.forward shares
        # one input between router and experts. We therefore bypass MoE.forward
        # and drive its submodules (router, expert_mlps) directly, matching
        # lines 1398-1403 of Gemma4TextDecoderLayer.forward.
        moe_expert_input = self.pre_feedforward_layernorm_2(residual_input)

        # Router sees un-normed residual (shape (B, S, H); our Gemma4Router
        # internally flattens and does its own no-scale RMSNorm + scale + 1/sqrt(H)).
        router_logits, expert_affinities, expert_index = self.moe.router(residual_input)
        # Experts receive pre_feedforward_layernorm_2-normed input flattened to (T, H).
        moe_flat_in = moe_expert_input.reshape(-1, H)
        moe_flat_out = self.moe.expert_mlps(
            hidden_states=moe_flat_in,
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            seq_len=S,
            padding_mask=None,
        )
        moe_out = moe_flat_out.reshape(B, S, H)
        moe_out = self.post_feedforward_layernorm_2(moe_out)

        # Combine and final norm.
        combined = dense_out + moe_out
        return self.post_feedforward_layernorm(combined)
