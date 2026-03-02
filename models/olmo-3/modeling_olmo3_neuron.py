"""
NeuronX Distributed Inference model for OLMo-3.

OLMo-3 architectural differences from Llama:
  1. Post-norm residual: norm applied to sub-layer OUTPUT then added to residual,
     instead of Llama's pre-norm (norm before attention/MLP).
  2. QK normalization: separate RMSNorms applied to the full concatenated Q and K
     (shape num_heads*head_dim and num_kv_heads*head_dim respectively) before RoPE.
  3. Mixed sliding + full attention: layer_types list controls whether each layer
     uses a sliding window (4096) or global causal attention.
  4. YaRN RoPE with attention_factor scaling.
  5. No attention bias; tied_embeddings=False.

Deviation note (QK norms):
  OLMo-3 trains with a single RMSNorm over the full num_heads*head_dim vector.
  For TP compatibility we use per-head RMSNorm(head_dim) instead.  The HF weight
  [num_heads*head_dim] is averaged across heads to initialise the [head_dim]
  scale.  This is an approximation; run accuracy eval to confirm it is acceptable.
"""

from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.utils import cpu_mode
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.utils.distributed import get_tp_group


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rms_norm(hidden_size: int, eps: float) -> nn.Module:
    """Return CustomRMSNorm on device, nn.RMSNorm on CPU (for unit tests)."""
    if cpu_mode():
        return nn.RMSNorm(hidden_size, eps=eps)
    return CustomRMSNorm(hidden_size, eps=eps)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class NeuronOlmo3Attention(NeuronAttentionBase):
    """
    OLMo-3 attention adapted for Neuron.

    Differences from NeuronLlamaAttention:
    * q_layernorm / k_layernorm use per-head RMSNorm(head_dim) for TP compatibility.
    * sliding_window set per-layer from config.layer_types[layer_idx].
    * No attention bias.
    * YaRN RoPE via LlamaRotaryEmbedding.

    Note: OLMo-3 trains with a single RMSNorm over the full num_heads*head_dim
    vector.  We use per-head RMSNorm(head_dim) here so the norm works with any
    TP degree.  The HF q_norm/k_norm weights (shape num_heads*head_dim) are
    averaged across heads to produce the per-head weight loaded at inference time.
    """

    def __init__(self, config: InferenceConfig, layer_idx: int):
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        # Per-layer attention type and sliding window
        layer_types: Optional[List[str]] = getattr(config, "layer_types", None)
        if layer_types is not None and layer_idx < len(layer_types):
            layer_type = layer_types[layer_idx]
        else:
            layer_type = "full_attention"
        sliding_window = None
        if layer_type == "sliding_attention":
            configured_window = getattr(config, "sliding_window", None)
            max_context_tokens = getattr(config.neuron_config, "n_positions", None)
            if configured_window is not None and max_context_tokens is not None:
                # NxDI's get_last_kv_window path expects window <= traced context length.
                # OLMo-3 has 4096 SWA, but many inference runs compile at smaller seq_len.
                sliding_window = min(configured_window, max_context_tokens)
            else:
                sliding_window = configured_window
        # Store so the decoder layer can route masks
        self.attention_type = layer_type

        # Per-head RMSNorm — TP-safe, weight shape (head_dim,)
        q_norm = _rms_norm(head_dim, config.rms_norm_eps)
        k_norm = _rms_norm(head_dim, config.rms_norm_eps)

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=self._build_rope(config),
            rms_norm_eps=config.rms_norm_eps,
            q_layernorm=q_norm,
            k_layernorm=k_norm,
            qkv_bias=getattr(config, "attention_bias", False),
            o_bias=getattr(config, "attention_bias", False),
            num_cores_per_group=getattr(config, "num_cores_per_group", 1),
            sliding_window=sliding_window,
        )

    @staticmethod
    def _build_rope(config: InferenceConfig) -> nn.Module:
        """
        Build rotary embeddings.  OLMo-3 uses YaRN (rope_scaling.rope_type='yarn').
        LlamaRotaryEmbedding detects the type from config.rope_scaling automatically.
        """
        return LlamaRotaryEmbedding(config)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class NeuronOlmo3MLP(nn.Module):
    """
    OLMo-3 SwiGLU MLP with ColumnParallel / RowParallel linear layers.

    forward(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
    No bias; no fused kernels (kept simple for correctness).
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

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
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class NeuronOlmo3DecoderLayer(nn.Module):
    """
    OLMo-3 transformer layer with post-norm residual connections.

    Forward pattern (differs from Llama's pre-norm):
        residual = hidden_states                    # no pre-norm
        attn_out = self_attn(hidden_states, ...)
        hidden_states = residual + post_attn_norm(attn_out)

        residual = hidden_states
        mlp_out = mlp(hidden_states)               # no pre-norm
        hidden_states = residual + post_ffn_norm(mlp_out)

    Mask routing:
        sliding_attention layers → use local_mask (windowed causal mask)
        full_attention layers    → use attention_mask (global causal mask)
    """

    def __init__(self, config: InferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = NeuronOlmo3Attention(config=config, layer_idx=layer_idx)
        # Mirror the attention type for mask routing
        self.attention_type = self.self_attn.attention_type
        self.mlp = NeuronOlmo3MLP(config)
        self.post_attention_layernorm = _rms_norm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = _rms_norm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        local_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple:
        # Route to sliding-window mask or global causal mask
        if self.attention_type == "sliding_attention" and local_mask is not None:
            attn_mask = local_mask
        else:
            attn_mask = attention_mask

        # ── Attention sub-layer (no pre-norm) ──────────────────────────────
        residual = hidden_states
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        # Post-attention norm then residual add
        hidden_states = self.post_attention_layernorm(attn_output.hidden_states)
        hidden_states = residual + hidden_states

        # ── MLP sub-layer (no pre-norm) ────────────────────────────────────
        residual = hidden_states
        mlp_out = self.mlp(hidden_states)
        # Post-feedforward norm then residual add
        hidden_states = self.post_feedforward_layernorm(mlp_out)
        hidden_states = residual + hidden_states

        # Return format consumed by NeuronBaseModel.get_model_output:
        # (hidden_states, kv, cos_cache, sin_cache, next_layer_residual)
        return (
            hidden_states,
            attn_output.present_key_value,
            attn_output.cos_cache,
            attn_output.sin_cache,
            None,  # no fused-residual forwarding
        )


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

class Olmo3InferenceConfig(InferenceConfig):
    """InferenceConfig for OLMo-3."""

    def add_derived_config(self):
        # Required by NeuronBaseModel / NeuronAttentionBase
        self.num_cores_per_group = 1

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
            "sliding_window",
            "layer_types",
            "rope_scaling",
            "attention_bias",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


# ---------------------------------------------------------------------------
# Model backbone
# ---------------------------------------------------------------------------

class NeuronOlmo3Model(NeuronBaseModel):
    """Neuron backbone for OLMo-3."""

    def setup_attr_for_model(self, config: InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        # Signal the base model to create the windowed local_attn_mask for sliding layers
        sliding_window = getattr(config, "sliding_window", None)
        max_context_tokens = getattr(config.neuron_config, "n_positions", None)
        if sliding_window is not None and max_context_tokens is not None:
            self.sliding_window = min(sliding_window, max_context_tokens)
        else:
            self.sliding_window = sliding_window

    def init_model(self, config: InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if parallel_state.model_parallel_is_initialized():
            tp_group = get_tp_group(config)
            dtype = config.neuron_config.torch_dtype
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                tensor_model_parallel_group=tp_group,
            )
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                dtype=dtype,
                bias=False,
                pad=True,
                tensor_model_parallel_group=tp_group,
            )
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size, config.hidden_size, self.padding_idx
            )
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.layers = nn.ModuleList(
            [
                NeuronOlmo3DecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = _rms_norm(config.hidden_size, config.rms_norm_eps)


# ---------------------------------------------------------------------------
# Application head
# ---------------------------------------------------------------------------

class NeuronOlmo3ForCausalLM(NeuronBaseForCausalLM):
    """
    OLMo-3 causal LM application head for Neuron inference.

    Usage (compile):
        from transformers import AutoConfig
        from neuronx_distributed_inference.models.config import NeuronConfig

        hf_cfg   = AutoConfig.from_pretrained("/path/to/olmo-3-32b")
        n_cfg    = NeuronConfig(tp_degree=32, torch_dtype="bfloat16",
                                batch_size=1, seq_len=4096)
        inf_cfg  = Olmo3InferenceConfig(hf_cfg, neuron_config=n_cfg)
        model    = NeuronOlmo3ForCausalLM(inf_cfg)
        model.compile("/tmp/olmo3_compiled")

    Usage (generate):
        model.load("/tmp/olmo3_compiled")
        output = model.generate(input_ids, ...)
    """

    _model_cls = NeuronOlmo3Model

    @classmethod
    def get_config_cls(cls):
        return Olmo3InferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: InferenceConfig
    ) -> dict:
        """
        Convert HF OLMo-3 state dict to NxDI Neuron key layout.

        Called after the framework strips the 'model.' prefix, so incoming keys
        are e.g. 'layers.0.self_attn.q_proj.weight'.

        Transformations applied per layer:
          • q/k/v projections moved under the qkv_proj sub-module
            (GroupQueryAttention_QKV naming, fused_qkv=False)
          • o_proj moved under o_proj.o_proj
            (GroupQueryAttention_O naming, layer_name='o_proj')
          • q_norm / k_norm → q_layernorm / k_layernorm (per-head RMSNorm)
            HF weight [num_heads*head_dim] averaged across heads → [head_dim]
          • rank metadata tensors injected for SPMDRank
        """
        neuron_config = config.neuron_config
        tp_degree = neuron_config.tp_degree
        num_layers = config.num_hidden_layers
        head_dim = getattr(config, "head_dim",
                           config.hidden_size // config.num_attention_heads)

        for i in range(num_layers):
            # ── Rank metadata for attention ────────────────────────────────
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

            # ── QKV projections: move under qkv_proj sub-module ───────────
            for proj in ("q_proj", "k_proj", "v_proj"):
                old = f"layers.{i}.self_attn.{proj}.weight"
                new = f"layers.{i}.self_attn.qkv_proj.{proj}.weight"
                if old in state_dict:
                    state_dict[new] = state_dict.pop(old)

            # ── Output projection: GroupQueryAttention_O naming ───────────
            old_o = f"layers.{i}.self_attn.o_proj.weight"
            new_o = f"layers.{i}.self_attn.o_proj.o_proj.weight"
            if old_o in state_dict:
                state_dict[new_o] = state_dict.pop(old_o)

            # ── QK norms → per-head RMSNorm weight [head_dim] ─────────────
            # HF stores [num_heads * head_dim]; average across heads to get
            # a single [head_dim] scale vector for the per-head norm.
            old_qn = f"layers.{i}.self_attn.q_norm.weight"
            new_qn = f"layers.{i}.self_attn.q_layernorm.weight"
            if old_qn in state_dict:
                w = state_dict.pop(old_qn)
                state_dict[new_qn] = w.view(-1, head_dim).mean(dim=0)

            old_kn = f"layers.{i}.self_attn.k_norm.weight"
            new_kn = f"layers.{i}.self_attn.k_layernorm.weight"
            if old_kn in state_dict:
                w = state_dict.pop(old_kn)
                state_dict[new_kn] = w.view(-1, head_dim).mean(dim=0)

        # ── Rank metadata for base model and (optionally) embedding ────────
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size, dtype=torch.int32
            )

        return state_dict
