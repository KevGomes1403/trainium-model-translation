"""
BLOCK D — Gemma4 Text Decoder Layer + Text Model.

Composes Block A (attention), Block B (parallel MLP + MoE FFN), and Block C
(embeddings / PLE / norms / logit softcap) into a complete NxDI text backbone.

Source references (models/gemma4/hf/modeling_gemma4.py):
  - Gemma4TextDecoderLayer            L1335-1421
  - Gemma4TextModel                   L1547-1769
  - Gemma4ForCausalLM                 L1773-1854

Deviations from the HF source (inline):
  (D1) Mixed sliding / full attention per layer via config.layer_types.
       `NeuronGemma4TextAttention` dispatches on layer_idx; decoder layer only
       needs to pick between the sliding attention mask and the full mask when
       calling the attention module.
  (D2) Parallel dense-MLP + MoE FFN: delegated to `NeuronGemma4FFN` (Block B).
       FFN is the sum of two branches (each with its own pre/post norms) — the
       module returns the post_feedforward_layernorm output; decoder adds the
       residual around it.
  (D3) Per-layer input (PLE) gate: when `hidden_size_per_layer_input > 0`, the
       source applies `merge_per_layer_input_states(...)` after the FFN residual.
       Source (modeling_gemma4.py L1411-1418):
           residual = hidden_states
           hidden_states = self.per_layer_input_gate(residual)
           hidden_states = self.hidden_activation(hidden_states) * per_layer_input
           hidden_states = self.per_layer_projection(hidden_states)
           hidden_states = self.post_per_layer_input_norm(residual + hidden_states)
       When disabled (hidden_size_per_layer_input == 0), this branch is entirely
       absent. We mirror this with an `enable_ple` flag.
  (D4) `layer_scalar` buffer (registered as ones; source L1347) multiplies the
       final decoder output. Preserved here.
  (D5) Scaled word embedding: matches Gemma3 pattern — at layer 0 entry we
       multiply hidden_states by sqrt(hidden_size) so the base-model forward
       can keep using the standard `inputs_embeds` produced by ParallelEmbedding
       (without embedding-time scaling).
  (D6) Final logit softcap applied in the application head, not the text model.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseModel

from blocks.neuron_gemma4_attention import NeuronGemma4TextAttention
from blocks.neuron_gemma4_embeddings import (
    NeuronGemma4PLE,
    NeuronGemma4RMSNorm,
    apply_logit_softcap,
    get_rmsnorm_cls,
)
from blocks.neuron_gemma4_moe import NeuronGemma4FFN


# --------------------------------------------------------------------------- #
# Per-layer config list. Attention dims/window swap between sliding and full.
# --------------------------------------------------------------------------- #
def get_updated_configs(config: InferenceConfig) -> List[InferenceConfig]:
    """
    Return a list of per-layer InferenceConfigs where attention-related fields
    reflect that layer's type (sliding vs full).

    Deviation (D1) implementation: layer_type is read from config.layer_types.
    Attention module itself inspects config.layer_types[layer_idx] when built,
    so the returned per-layer config only needs `sliding_window` nulled out on
    full-attention layers — sufficient for mask handling in the decoder.
    """
    import copy

    layer_types = getattr(config, "layer_types", None)
    num_layers = config.num_hidden_layers
    updated = []
    for i in range(num_layers):
        per_layer = copy.copy(config)
        if layer_types is not None and i < len(layer_types):
            if layer_types[i] != "sliding_attention":
                per_layer.sliding_window = None
        updated.append(per_layer)
    return updated


# --------------------------------------------------------------------------- #
# Decoder Layer.
# --------------------------------------------------------------------------- #
class NeuronGemma4DecoderLayer(nn.Module):
    def __init__(self, config: InferenceConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None and layer_idx < len(layer_types):
            self.layer_type = layer_types[layer_idx]
        else:
            self.layer_type = "sliding_attention"
        self.is_sliding_window_attention = self.layer_type == "sliding_attention"

        # --- Attention (Block A) ---
        self.self_attn = NeuronGemma4TextAttention(config, layer_idx=layer_idx)

        # --- 4 RMSNorms around attention ---
        rms_eps = getattr(config, "rms_norm_eps", 1e-6)
        norm_cls = get_rmsnorm_cls(with_scale=True)
        self.input_layernorm = norm_cls(self.hidden_size, eps=rms_eps)
        self.post_attention_layernorm = norm_cls(self.hidden_size, eps=rms_eps)

        # --- FFN (Block B): parallel dense-MLP + MoE with its own 5 RMSNorms ---
        # D2: FFN owns pre_feedforward_layernorm, pre_feedforward_layernorm_2,
        # post_feedforward_layernorm_1, post_feedforward_layernorm_2, and the
        # final post_feedforward_layernorm that the decoder would otherwise own.
        self.ffn = NeuronGemma4FFN(config)

        # --- Per-Layer Input gate (D3) ---
        ple_dim = getattr(config, "hidden_size_per_layer_input", 0) or 0
        self.enable_ple = ple_dim > 0
        if self.enable_ple:
            # HF source L1370-1378:
            #   per_layer_input_gate = Linear(H, ple_dim, bias=False)
            #   per_layer_projection = Linear(ple_dim, H, bias=False)
            #   post_per_layer_input_norm = Gemma4RMSNorm(H, eps=rms_norm_eps)
            self.per_layer_input_gate = _column_or_linear(
                self.hidden_size, ple_dim, config
            )
            self.per_layer_projection = _row_or_linear(
                ple_dim, self.hidden_size, config
            )
            self.post_per_layer_input_norm = norm_cls(self.hidden_size, eps=rms_eps)
            # Activation: source uses self.hidden_activation which is the same
            # gelu_pytorch_tanh as MLP activation.
            from transformers.activations import ACT2FN
            self.hidden_activation = ACT2FN[
                getattr(config, "hidden_act",
                        getattr(config, "hidden_activation", "gelu_pytorch_tanh"))
            ]

        # --- D4: layer_scalar buffer (ones). Source L1347. ---
        self.register_buffer(
            "layer_scalar",
            torch.ones(1, dtype=config.neuron_config.torch_dtype),
            persistent=False,
        )

    def _merge_per_layer_input_states(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor,
    ) -> torch.Tensor:
        # D3: HF source modeling_gemma4.py L1411-1418.
        residual = hidden_states
        x = self.per_layer_input_gate(residual)
        x = self.hidden_activation(x) * per_layer_input
        x = self.per_layer_projection(x)
        return self.post_per_layer_input_norm(residual + x)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        local_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        per_layer_input: Optional[torch.Tensor] = None,
        adapter_ids=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        # D1: pick the right mask (sliding vs full).
        mask = local_mask if (self.is_sliding_window_attention and local_mask is not None) else attention_mask

        # D5: scaled word embedding applied at layer 0 entry.
        if self.layer_idx == 0:
            hidden_states = hidden_states * (self.hidden_size ** 0.5)

        # --- Self-attention block (pre-norm + attn + post-norm + residual) ---
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_kv, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # --- FFN block (parallel MLP + MoE + final post_ff_norm) ---
        residual = hidden_states
        hidden_states = self.ffn(hidden_states)  # FFN returns post-norm output
        hidden_states = residual + hidden_states

        # --- D3: per-layer input gate / merge ---
        if self.enable_ple and per_layer_input is not None:
            hidden_states = self._merge_per_layer_input_states(hidden_states, per_layer_input)

        # --- D4: layer_scalar multiply ---
        hidden_states = hidden_states * self.layer_scalar

        return (hidden_states, present_kv, cos_cache, sin_cache, None)


# --------------------------------------------------------------------------- #
# Helpers: parallel layer with plain nn.Linear fallback on CPU.
# --------------------------------------------------------------------------- #
def _column_or_linear(in_f: int, out_f: int, config: InferenceConfig) -> nn.Module:
    dtype = config.neuron_config.torch_dtype
    if parallel_state.model_parallel_is_initialized():
        return ColumnParallelLinear(
            in_f, out_f, bias=False, gather_output=False, dtype=dtype, pad=True,
        )
    return nn.Linear(in_f, out_f, bias=False).to(dtype=dtype)


def _row_or_linear(in_f: int, out_f: int, config: InferenceConfig) -> nn.Module:
    dtype = config.neuron_config.torch_dtype
    if parallel_state.model_parallel_is_initialized():
        return RowParallelLinear(
            in_f, out_f, bias=False, input_is_parallel=True, dtype=dtype, pad=True,
        )
    return nn.Linear(in_f, out_f, bias=False).to(dtype=dtype)


# --------------------------------------------------------------------------- #
# Text Model (NeuronBaseModel subclass).
# --------------------------------------------------------------------------- #
class NeuronGemma4TextModel(NeuronBaseModel):
    """Gemma4 text decoder stack wrapped as a NeuronBaseModel.

    Owns: embed_tokens, layers[L], norm, PLE (optional), lm_head, logit softcap.
    """

    def setup_attr_for_model(self, config: InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: InferenceConfig):
        self.padding_idx = getattr(config, "pad_token_id", 0)
        self.vocab_size = config.vocab_size
        self.final_logit_softcapping = getattr(config, "final_logit_softcapping", None)

        dtype = config.neuron_config.torch_dtype

        # --- Token embedding (no explicit scale — applied at layer 0) ---
        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=dtype,
                shard_across_embedding=True,
                sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
            )
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                pad=True,
                gather_output=not self.on_device_sampling,
                dtype=dtype,
            )
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size, config.hidden_size, self.padding_idx
            ).to(dtype=dtype)
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            ).to(dtype=dtype)

        # --- Decoder stack ---
        per_layer_cfgs = get_updated_configs(config)
        self.layers = nn.ModuleList(
            [NeuronGemma4DecoderLayer(cfg, layer_idx=i) for i, cfg in enumerate(per_layer_cfgs)]
        )

        # --- Final norm ---
        norm_cls = get_rmsnorm_cls(with_scale=True)
        self.norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)

        # --- PLE (optional) ---
        ple_dim = getattr(config, "hidden_size_per_layer_input", 0) or 0
        if ple_dim > 0:
            self.per_layer_embed = NeuronGemma4PLE(
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                vocab_size_per_layer_input=getattr(
                    config, "vocab_size_per_layer_input", config.vocab_size
                ),
                hidden_size_per_layer_input=ple_dim,
                rms_norm_eps=config.rms_norm_eps,
                padding_idx=self.padding_idx,
                dtype=dtype,
            )
        else:
            self.per_layer_embed = None

    # ----------------------------------------------------------------------- #
    # Scatter vision embeddings into text inputs_embeds (VLM integration).
    # Called by the application head before the decoder forward when prefilling
    # with images. For text-only requests / token generation, the caller passes
    # a no-op vision_mask (all-zeros) which makes scatter a no-op.
    # ----------------------------------------------------------------------- #
    def encode_vision_to_input(
        self,
        inputs_embeds: torch.Tensor,
        vision_embeddings: torch.Tensor,
        vision_mask: torch.Tensor,
    ) -> torch.Tensor:
        from blocks.neuron_gemma4_vision_merge import encode_vision_to_input as _scatter
        return _scatter(inputs_embeds, vision_embeddings, vision_mask)
