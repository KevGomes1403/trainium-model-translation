"""
NeuronX Distributed Inference model for Gemma-2-9B.

Architectural notes:
  1. SwiGLU MLP with gelu_pytorch_tanh activation (not silu).
  2. GQA attention: 16 Q heads, 8 KV heads, head_dim=256.
     Custom QK scale: query_pre_attn_scalar**-0.5 = 256**-0.5 (not 1/sqrt(head_dim)).
  3. Attn logit softcapping (50.0): NOT applied — NxDI NKI kernel has no softcap hook.
     Documented deviation; core attention mechanics (GQA, RoPE, sliding window) are correct.
  4. Mixed sliding + full attention, alternating per layer:
       even layers (0,2,4,...) → "sliding_attention" (window=4096)
       odd layers  (1,3,5,...) → "full_attention"
  5. Double-norm decoder layer: 4 layernorms per layer (vs Llama's 2):
       input_layernorm, post_attention_layernorm,
       pre_feedforward_layernorm, post_feedforward_layernorm.
  6. Gemma2RMSNorm uses (1 + weight) scaling (weight zero-initialized).
     Handled by baking +1 into all norm weights in convert_hf_to_neuron_state_dict;
     CustomRMSNorm on device (which computes x_normed * weight) then gives the
     correct result.
  7. Embedding scale: hidden_states * sqrt(hidden_size) after embedding lookup.
     Implemented via ScaledEmbedding wrapper around embed_tokens.
  8. Final logit softcapping: tanh(logits / 30) * 30 after lm_head.
     Implemented via SoftcappedLinear wrapper around lm_head.
  9. Tied weights: lm_head.weight = embed_tokens.weight (no lm_head in HF checkpoint).

Weight key mapping (framework strips "model." prefix before calling convert):
  embed_tokens.weight                    → embed_tokens.embed.weight
  layers.i.self_attn.q_proj.weight       → layers.i.self_attn.qkv_proj.q_proj.weight
  layers.i.self_attn.k_proj.weight       → layers.i.self_attn.qkv_proj.k_proj.weight
  layers.i.self_attn.v_proj.weight       → layers.i.self_attn.qkv_proj.v_proj.weight
  layers.i.self_attn.o_proj.weight       → layers.i.self_attn.o_proj.o_proj.weight
  layers.i.{all 4 norm}.weight           → unchanged (baked +1 applied in-place)
  norm.weight                            → unchanged (baked +1 applied in-place)
  lm_head.weight                         → lm_head.linear.weight
                                           (produced by update_state_dict_for_tied_weights)
"""

import math
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
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
    FlashAttentionStrategy,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.utils.distributed import get_tp_group


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rms_norm(hidden_size: int, eps: float) -> nn.Module:
    """Return CustomRMSNorm on device, nn.RMSNorm on CPU (for unit tests).

    Gemma-2 uses (1 + weight) scaling. This is handled at conversion time by
    baking '+1' into all norm weights, so CustomRMSNorm (which applies x * weight)
    produces the correct result without any code change here.
    """
    if cpu_mode():
        return nn.RMSNorm(hidden_size, eps=eps)
    return CustomRMSNorm(hidden_size, eps=eps)


class ScaledEmbedding(nn.Module):
    """Wraps an embedding module to apply a fixed post-lookup scale.

    Gemma-2 multiplies the embedding output by sqrt(hidden_size) before passing
    to the transformer layers. State dict key: embed_tokens.embed.weight.
    """

    def __init__(self, embed_module: nn.Module, scale: float):
        super().__init__()
        self.embed = embed_module
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x) * self.scale


class SoftcappedLinear(nn.Module):
    """Wraps a linear module to apply tanh softcapping to its output.

    Gemma-2 applies tanh(logits / softcap) * softcap to the final logits.
    State dict key: lm_head.linear.weight.
    """

    def __init__(self, linear_module: nn.Module, softcap: float):
        super().__init__()
        self.linear = linear_module
        self.softcap = softcap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return torch.tanh(logits / self.softcap) * self.softcap


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class NeuronGemma2MLP(nn.Module):
    """
    Gemma-2 SwiGLU MLP for Neuron with tensor parallelism.

    forward(x) = down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))

    Tensor parallelism layout:
      gate_proj  [H → I/tp]  ColumnParallelLinear, gather_output=False
      up_proj    [H → I/tp]  ColumnParallelLinear, gather_output=False
      down_proj  [I/tp → H]  RowParallelLinear,    input_is_parallel=True (all-reduce)
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[getattr(config, "hidden_activation", "gelu_pytorch_tanh")]

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
# Attention
# ---------------------------------------------------------------------------

class NeuronGemma2Attention(NeuronAttentionBase):
    """
    Gemma-2 attention adapted for Neuron.

    Differences from standard Llama attention:
      - Custom QK scale: query_pre_attn_scalar**-0.5 (not head_dim**-0.5).
        Overridden via scaled_qk().
      - Per-layer sliding window: derived from config.layer_types[layer_idx].
        Even layers → "sliding_attention" (window=4096), odd → "full_attention".
      - No QK norms, no attention bias.

    attn_logit_softcapping (50.0) is applied in scaled_qk() via the PyTorch
    eager path: tanh(scores / softcap) * softcap before the attention mask.
    """

    def __init__(self, config: InferenceConfig, layer_idx: int):
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        # Determine per-layer attention type and sliding window
        layer_types = getattr(config, "layer_types", None)
        layer_type = (
            layer_types[layer_idx]
            if layer_types is not None and layer_idx < len(layer_types)
            else "full_attention"
        )
        self.attention_type = layer_type

        sliding_window = None
        if layer_type == "sliding_attention":
            configured_window = getattr(config, "sliding_window", None)
            n_positions = getattr(config.neuron_config, "n_positions", None)
            if configured_window is not None and n_positions is not None:
                sliding_window = min(configured_window, n_positions)
            else:
                sliding_window = configured_window

        # Store Gemma-2 custom scaling and softcap
        self.query_pre_attn_scalar = getattr(config, "query_pre_attn_scalar", 256)
        self.attn_logit_softcapping = getattr(config, "attn_logit_softcapping", None)

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=LlamaRotaryEmbedding(config),
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            qkv_bias=False,
            o_bias=False,
            num_cores_per_group=getattr(config, "num_cores_per_group", 1),
            sliding_window=sliding_window,
        )

    def scaled_qk(
        self, Q: torch.Tensor, K: torch.Tensor, attention_mask
    ) -> torch.Tensor:
        """Apply Gemma-2's custom QK scale and attn_logit_softcapping.

        1. Scale: query_pre_attn_scalar**-0.5 (not head_dim**-0.5)
        2. Softcap: tanh(scores / softcap) * softcap
        3. Attention mask
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
    
    def get_flash_attention_strategy(self, q_len: int, has_attention_mask: bool) -> FlashAttentionStrategy:
        """
        Disable all flash attention kernels for Gemma-2.

        The NKI flash_fwd sliding-window kernel asserts head_dim <= 128, but
        Gemma-2 uses head_dim=256.  Returning NONE forces every attention path
        (including windowed_attention_forward) through the flat compiler
        implementation, which calls self.scaled_qk() and therefore picks up
        the softcapping override correctly.
        """
        return FlashAttentionStrategy.NONE


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class NeuronGemma2DecoderLayer(nn.Module):
    """
    Gemma-2 transformer decoder layer with double-norm pattern.

    Forward (4 norms, distinct from Llama's 2-norm pre-norm):
      residual = hidden_states
      hidden_states = input_layernorm(hidden_states)
      hidden_states = self_attn(hidden_states, ...)
      hidden_states = post_attention_layernorm(hidden_states)
      hidden_states = residual + hidden_states

      residual = hidden_states
      hidden_states = pre_feedforward_layernorm(hidden_states)
      hidden_states = mlp(hidden_states)
      hidden_states = post_feedforward_layernorm(hidden_states)
      hidden_states = residual + hidden_states

    Mask routing:
      "sliding_attention" layers → use local_mask (windowed causal mask)
      "full_attention" layers    → use attention_mask (global causal mask)
    """

    def __init__(self, config: InferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = NeuronGemma2Attention(config=config, layer_idx=layer_idx)
        self.attention_type = self.self_attn.attention_type
        self.mlp = NeuronGemma2MLP(config)
        self.input_layernorm = _rms_norm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _rms_norm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = _rms_norm(config.hidden_size, config.rms_norm_eps)
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
        # Route to sliding-window or global causal mask
        if self.attention_type == "sliding_attention" and local_mask is not None:
            attn_mask = local_mask
        else:
            attn_mask = attention_mask

        # ── Attention sub-layer ──────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(attn_output.hidden_states)
        hidden_states = residual + hidden_states

        # ── MLP sub-layer ────────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
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

class Gemma2InferenceConfig(InferenceConfig):
    """InferenceConfig for Gemma-2-9B."""

    def add_derived_config(self):
        # Required by NeuronBaseModel / NeuronAttentionBase
        self.num_cores_per_group = 1

        # Derive layer_types if not present in HF config.
        # Gemma-2 alternates: even layers → sliding_attention, odd → full_attention.
        if not getattr(self, "layer_types", None):
            num_layers = self.num_hidden_layers
            self.layer_types = [
                "sliding_attention" if i % 2 == 0 else "full_attention"
                for i in range(num_layers)
            ]

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "head_dim",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_activation",
            "intermediate_size",
            "attn_logit_softcapping",
            "final_logit_softcapping",
            "query_pre_attn_scalar",
            "sliding_window",
            "tie_word_embeddings",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


# ---------------------------------------------------------------------------
# Model backbone
# ---------------------------------------------------------------------------

class NeuronGemma2Model(NeuronBaseModel):
    """Neuron backbone for Gemma-2-9B."""

    def setup_attr_for_model(self, config: InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        # Signal the base model to create windowed local_attn_mask for sliding layers
        sliding_window = getattr(config, "sliding_window", None)
        n_positions = getattr(config.neuron_config, "n_positions", None)
        if sliding_window is not None and n_positions is not None:
            self.sliding_window = min(sliding_window, n_positions)
        else:
            self.sliding_window = sliding_window

    def init_model(self, config: InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        embed_scale = math.sqrt(config.hidden_size)

        if parallel_state.model_parallel_is_initialized():
            tp_group = get_tp_group(config)
            dtype = config.neuron_config.torch_dtype
            base_embed = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                tensor_model_parallel_group=tp_group,
            )
            base_lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                dtype=dtype,
                bias=False,
                pad=True,
                tensor_model_parallel_group=tp_group,
            )
        else:
            base_embed = nn.Embedding(
                config.vocab_size, config.hidden_size, self.padding_idx
            )
            base_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Wrap embedding with post-lookup scale (sqrt(hidden_size))
        self.embed_tokens = ScaledEmbedding(base_embed, embed_scale)
        # Wrap lm_head with final logit softcapping
        self.lm_head = SoftcappedLinear(
            base_lm_head, getattr(config, "final_logit_softcapping", 30.0)
        )

        self.layers = nn.ModuleList(
            [
                NeuronGemma2DecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = _rms_norm(config.hidden_size, config.rms_norm_eps)


# ---------------------------------------------------------------------------
# Application head
# ---------------------------------------------------------------------------

class NeuronGemma2ForCausalLM(NeuronBaseForCausalLM):
    """
    Gemma-2-9B causal LM application head for Neuron inference.

    Usage (compile):
        from transformers import AutoConfig
        from neuronx_distributed_inference.models.config import NeuronConfig

        hf_cfg  = AutoConfig.from_pretrained("/path/to/gemma-2-9b")
        n_cfg   = NeuronConfig(tp_degree=8, torch_dtype="bfloat16",
                               batch_size=1, seq_len=4096)
        inf_cfg = Gemma2InferenceConfig(hf_cfg, neuron_config=n_cfg)
        model   = NeuronGemma2ForCausalLM(inf_cfg)
        model.compile("/tmp/gemma2_compiled")

    Usage (generate):
        model.load("/tmp/gemma2_compiled")
        output = model.generate(input_ids, ...)
    """

    _model_cls = NeuronGemma2Model

    @classmethod
    def get_config_cls(cls):
        return Gemma2InferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: InferenceConfig
    ) -> dict:
        """
        Convert HF Gemma-2 state dict to NxDI Neuron key layout.

        Called after the framework strips the 'model.' prefix, so incoming
        keys are e.g. 'embed_tokens.weight', 'layers.0.self_attn.q_proj.weight'.

        Transformations:
          - embed_tokens.weight            → embed_tokens.embed.weight
          - q/k/v projections              → qkv_proj.{q,k,v}_proj.weight
          - o_proj                         → o_proj.o_proj.weight
          - all norm weights               → bake (1 + weight) for Gemma2RMSNorm
          - rank metadata tensors          → injected for SPMDRank
          - lm_head.weight                 → lm_head.linear.weight
                                             (produced by update_state_dict_for_tied_weights)
        """
        neuron_config = config.neuron_config
        tp_degree = neuron_config.tp_degree
        num_layers = config.num_hidden_layers

        # ── Embedding: rename to embed_tokens.embed.weight ────────────────
        if "embed_tokens.weight" in state_dict:
            state_dict["embed_tokens.embed.weight"] = state_dict.pop("embed_tokens.weight")

        # ── Final norm: bake Gemma2RMSNorm (1 + weight) ───────────────────
        if "norm.weight" in state_dict:
            state_dict["norm.weight"] = state_dict["norm.weight"] + 1.0

        # ── Per-layer transformations ──────────────────────────────────────
        for i in range(num_layers):
            # Rank metadata for attention (always required by SPMDRank)
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

            # QKV projections: move under qkv_proj sub-module
            for proj in ("q_proj", "k_proj", "v_proj"):
                old = f"layers.{i}.self_attn.{proj}.weight"
                new = f"layers.{i}.self_attn.qkv_proj.{proj}.weight"
                if old in state_dict:
                    state_dict[new] = state_dict.pop(old)

            # Output projection: GroupQueryAttention_O naming
            old_o = f"layers.{i}.self_attn.o_proj.weight"
            new_o = f"layers.{i}.self_attn.o_proj.o_proj.weight"
            if old_o in state_dict:
                state_dict[new_o] = state_dict.pop(old_o)

            # Bake Gemma2RMSNorm (1 + weight) into all 4 decoder norms
            for norm_name in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ):
                key = f"layers.{i}.{norm_name}.weight"
                if key in state_dict:
                    state_dict[key] = state_dict[key] + 1.0

        # ── Rank metadata for base model ──────────────────────────────────
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        # ── Vocab parallel embedding rank ─────────────────────────────────
        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.embed.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size, dtype=torch.int32
            )

        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict: dict) -> None:
        """
        Gemma-2 ties lm_head.weight to embed_tokens.weight.

        Because both modules are wrapped (ScaledEmbedding / SoftcappedLinear),
        the actual weight paths are:
          embed_tokens.embed.weight  →  lm_head.linear.weight
        """
        src = "embed_tokens.embed.weight"
        dst = "lm_head.linear.weight"
        if src in state_dict and dst not in state_dict:
            state_dict[dst] = state_dict[src].clone()
