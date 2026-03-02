"""
NeuronX Distributed Inference model for Arcee AFM-4.5B.

Architectural specifics:
  1. 2-layer MLP (no gate_proj): forward = down_proj(relu2(up_proj(x)))
     This is NOT SwiGLU — there is no gating path.
  2. GQA with fused_qkv=False: separate q/k/v projections.
     num_attention_heads=20, num_key_value_heads=4, head_dim=128.
  3. YaRN RoPE: rope_scaling.rope_type="yarn", factor=20.0,
     original_max_position_embeddings=4096.
     LlamaRotaryEmbedding handles this automatically from config.
  4. tie_word_embeddings=False: lm_head.weight is a separate tensor.
  5. relu2 activation: ACT2FN["relu2"].
  6. Pre-norm residual connections (standard Llama-style).
  7. TP degree constraints: num_attention_heads=20 requires tp_degree
     that divides evenly (e.g. tp_degree=4: 20/4=5 Q, 4/4=1 KV).

Weight key mapping (framework strips "model." prefix before calling
convert_hf_to_neuron_state_dict):

  layers.i.self_attn.q_proj.weight  → layers.i.self_attn.qkv_proj.q_proj.weight
  layers.i.self_attn.k_proj.weight  → layers.i.self_attn.qkv_proj.k_proj.weight
  layers.i.self_attn.v_proj.weight  → layers.i.self_attn.qkv_proj.v_proj.weight
  layers.i.self_attn.o_proj.weight  → layers.i.self_attn.o_proj.o_proj.weight
  layers.i.mlp.up_proj.weight       → layers.i.mlp.up_proj.weight   (unchanged)
  layers.i.mlp.down_proj.weight     → layers.i.mlp.down_proj.weight (unchanged)
  layers.i.input_layernorm.weight   → unchanged
  layers.i.post_attention_layernorm.weight → unchanged
  embed_tokens.weight               → unchanged (model. prefix stripped by framework)
  lm_head.weight                    → unchanged (already at top-level in HF)
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

class NeuronArceeAttention(NeuronAttentionBase):
    """
    Arcee AFM-4.5B attention adapted for Neuron.

    Uses GQA (20 Q heads, 4 KV heads, head_dim=128) with fused_qkv=False.
    YaRN RoPE is handled automatically by LlamaRotaryEmbedding when
    config.rope_scaling has rope_type="yarn".

    No QK norms, no attention bias.
    """

    def __init__(self, config: InferenceConfig, layer_idx: int):
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=LlamaRotaryEmbedding(config),
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-5),
            qkv_bias=getattr(config, "attention_bias", False),
            o_bias=getattr(config, "attention_bias", False),
            num_cores_per_group=getattr(config, "num_cores_per_group", 1),
            # No sliding window — Arcee uses standard full attention
            sliding_window=None,
        )


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class NeuronArceeMLP(nn.Module):
    """
    Arcee AFM-4.5B 2-layer MLP with relu2 activation.

    forward(x) = down_proj(relu2(up_proj(x)))

    There is NO gate projection (unlike LlamaMLP / SwiGLU).
    Uses ColumnParallelLinear for up_proj and RowParallelLinear for down_proj
    when TP is initialized, plain nn.Linear for CPU unit tests.
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[getattr(config, "hidden_act", "relu2")]
        mlp_bias = getattr(config, "mlp_bias", False)

        if parallel_state.model_parallel_is_initialized():
            tp_group = get_tp_group(config)
            dtype = config.neuron_config.torch_dtype
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=mlp_bias,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=mlp_bias,
                input_is_parallel=True,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
        else:
            # CPU fallback for unit tests
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 2-layer MLP: no gate, just up → activate → down
        return self.down_proj(self.act_fn(self.up_proj(x)))


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class NeuronArceeDecoderLayer(nn.Module):
    """
    Arcee decoder layer with standard pre-norm residual connections (Llama-style).

    Pattern:
        residual = hidden_states
        hidden_states = self_attn(input_layernorm(hidden_states), ...)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = mlp(post_attention_layernorm(hidden_states))
        hidden_states = residual + hidden_states
    """

    def __init__(self, config: InferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = NeuronArceeAttention(config=config, layer_idx=layer_idx)
        self.mlp = NeuronArceeMLP(config)
        self.input_layernorm = _rms_norm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _rms_norm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        **kwargs,
    ) -> Tuple:
        # ── Attention sub-layer ──────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + attn_output.hidden_states

        # ── MLP sub-layer ─────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
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

class ArceeInferenceConfig(InferenceConfig):
    """InferenceConfig for Arcee AFM-4.5B."""

    def add_derived_config(self):
        # Required by NeuronBaseModel / NeuronAttentionBase
        self.num_cores_per_group = 1

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
            "hidden_act",
            "intermediate_size",
            "rope_scaling",
            "attention_bias",
            "mlp_bias",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


# ---------------------------------------------------------------------------
# Model backbone
# ---------------------------------------------------------------------------

class NeuronArceeModel(NeuronBaseModel):
    """Neuron backbone for Arcee AFM-4.5B."""

    def setup_attr_for_model(self, config: InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        # No sliding window in Arcee
        self.sliding_window = None

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
                # NOTE: pad=True fails in training mode (unit tests)
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
                NeuronArceeDecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = _rms_norm(config.hidden_size, config.rms_norm_eps)


# ---------------------------------------------------------------------------
# Application head
# ---------------------------------------------------------------------------

class NeuronArceeForCausalLM(NeuronBaseForCausalLM):
    """
    Arcee AFM-4.5B causal LM application head for Neuron inference.

    Usage (compile):
        from neuronx_distributed_inference.models.config import NeuronConfig

        # Load config from HF config.json dict
        hf_cfg_dict = {
            "hidden_size": 2560,
            "num_attention_heads": 20,
            "num_key_value_heads": 4,
            "num_hidden_layers": 36,
            "intermediate_size": 18432,
            "vocab_size": 128004,
            "max_position_embeddings": 65536,
            "head_dim": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "rope_scaling": {"rope_type": "yarn", "factor": 20.0,
                             "original_max_position_embeddings": 4096},
            "hidden_act": "relu2",
            "attention_bias": False,
            "mlp_bias": False,
            "pad_token_id": 0,
            "tie_word_embeddings": False,
        }
        n_cfg    = NeuronConfig(tp_degree=4, torch_dtype="bfloat16",
                                batch_size=1, seq_len=4096,
                                fused_qkv=False)
        inf_cfg  = ArceeInferenceConfig(hf_cfg_dict, neuron_config=n_cfg)
        model    = NeuronArceeForCausalLM(inf_cfg)
        model.compile("/tmp/arcee_compiled")

    Usage (generate):
        model.load("/tmp/arcee_compiled")
        output = model.generate(input_ids, ...)
    """

    _model_cls = NeuronArceeModel

    @classmethod
    def get_config_cls(cls):
        return ArceeInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: InferenceConfig
    ) -> dict:
        """
        Convert HF Arcee state dict to NxDI Neuron key layout.

        Called after the framework strips the 'model.' prefix, so incoming
        keys are e.g. 'layers.0.self_attn.q_proj.weight'.

        Transformations applied per layer:
          - q/k/v projections moved under the qkv_proj sub-module
            (GroupQueryAttention_QKV naming, fused_qkv=False)
          - o_proj moved under o_proj.o_proj
            (GroupQueryAttention_O naming, layer_name='o_proj')
          - rank metadata tensors injected for SPMDRank
        """
        neuron_config = config.neuron_config
        tp_degree = neuron_config.tp_degree
        num_layers = config.num_hidden_layers

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

        # ── Rank metadata for base model ──────────────────────────────────
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size, dtype=torch.int32
            )

        return state_dict
