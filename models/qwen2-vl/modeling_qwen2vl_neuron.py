"""
NeuronX Distributed Inference model for Qwen2-VL (text LM backbone + vision encoder).

Qwen2-VL-7B architectural notes:
  - Text backbone is identical to Qwen2-7B: GQA (28 Q-heads, 4 KV-heads),
    SwiGLU MLP, RMSNorm pre-norm, bias on QKV projections, no bias on o_proj.
  - MRoPE is present in HF config but for text-only tokens all three spatial
    dimensions are identical, so standard 1D RoPE is equivalent.
  - use_sliding_window=False -> no sliding window logic needed.
  - Vision encoder translated in modeling_qwen2vl_vision_neuron.py.

HF checkpoint key layout (Qwen2VLForConditionalGeneration):
  lm_head.weight          (top-level, NOT under model.)
  model.embed_tokens.*
  model.layers.*
  model.norm.*
  visual.*                (vision encoder — mapped to vision_transformer.* in NxDI)

Multimodal usage:
  Use NeuronQwen2VLForConditionalGeneration with Qwen2VLMultimodalInferenceConfig.
  Text-only usage (legacy): Use NeuronQwen2VLForCausalLM with Qwen2VLInferenceConfig.
"""

import copy
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.image_to_text_model_base import (
    ImageToTextInferenceConfig,
    NeuronBaseForImageToText,
)
from neuronx_distributed_inference.models.image_to_text_model_wrapper import ImageToTextModelWrapper
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.models.model_wrapper import VISION_ENCODER_MODEL_TAG
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.utils.distributed import get_tp_group
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
    scatter_by_index_put,
    generate_positions_from_mask,
    pad_positions,
    pad_vision_embeddings,
)

# Re-export the pre-built block implementations
from attention_block import NeuronQwen2VLAttention  # noqa: F401
from mlp_block import NeuronQwen2VLMLP  # noqa: F401

# Vision encoder components
from modeling_qwen2vl_vision_neuron import (
    NeuronQwen2VLVisionModel,
    Qwen2VLVisionInferenceConfig,
    Qwen2VLVisionModelWrapper,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _rms_norm(hidden_size: int, eps: float) -> nn.Module:
    """Return CustomRMSNorm on device, nn.RMSNorm on CPU (for unit tests)."""
    if cpu_mode():
        return nn.RMSNorm(hidden_size, eps=eps)
    return CustomRMSNorm(hidden_size, eps=eps)


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class NeuronQwen2VLDecoderLayer(nn.Module):
    """
    Standard pre-norm Llama-style decoder layer for Qwen2-VL.

    Forward pattern:
        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)
        attn_out = self_attn(hidden_states, ...)
        hidden_states = residual + attn_out.hidden_states

        residual = hidden_states
        hidden_states = post_attention_layernorm(hidden_states)
        hidden_states = mlp(hidden_states)
        hidden_states = residual + hidden_states

    Returns 5-tuple: (hidden_states, kv, cos_cache, sin_cache, None)
    as consumed by NeuronBaseModel.get_model_output.
    """

    def __init__(self, config: InferenceConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = NeuronQwen2VLAttention(config, layer_idx=layer_idx)
        self.mlp = NeuronQwen2VLMLP(config)
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
        # ── Attention sub-layer ─────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # NeuronAttentionBase returns NeuronAttentionBaseOutput which supports
        # both attribute access and tuple unpacking via __iter__.
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # ── MLP sub-layer ───────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # NeuronQwen2VLMLP.forward returns a plain tensor (not a tuple).
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # 5-tuple required by NeuronBaseModel.get_model_output
        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Qwen2VLInferenceConfig(InferenceConfig):
    """InferenceConfig for Qwen2-VL text LM backbone."""

    def add_derived_config(self):
        self.num_cores_per_group = 1
        if not hasattr(self, "head_dim") or self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        # pad_token_id is absent from Qwen2-VL config.json; fall back to eos_token_id
        if not hasattr(self, "pad_token_id") or self.pad_token_id is None:
            eos = getattr(self, "eos_token_id", 151645)
            self.pad_token_id = eos if isinstance(eos, int) else eos[0]

    def get_required_attributes(self) -> List[str]:
        return [
            "head_dim",
            "hidden_act",
            "hidden_size",
            "intermediate_size",
            "max_position_embeddings",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "rms_norm_eps",
            "rope_theta",
            "vocab_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


# ---------------------------------------------------------------------------
# Model backbone
# ---------------------------------------------------------------------------

class NeuronQwen2VLModel(NeuronBaseModel):
    """Neuron backbone for Qwen2-VL text LM."""

    def setup_attr_for_model(self, config: Qwen2VLInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        # use_sliding_window=False for Qwen2-VL-7B; no local_attn_mask needed
        self.sliding_window = None

    def init_model(self, config: Qwen2VLInferenceConfig):
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
                NeuronQwen2VLDecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = _rms_norm(config.hidden_size, config.rms_norm_eps)

    def encode_vision_to_input(
        self,
        inputs_embeds: torch.Tensor,
        vision_embeddings: torch.Tensor,
        vision_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scatter pre-encoded vision embeddings into the text embedding sequence
        at positions indicated by vision_mask.

        inputs_embeds:    [batch, seq_len, hidden_size]
        vision_embeddings:[batch, num_vision_tokens, hidden_size]
        vision_mask:      [batch, num_vision_tokens] — indices into seq_len
        """
        return scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)


# ---------------------------------------------------------------------------
# Application head (text-only, legacy)
# ---------------------------------------------------------------------------

class NeuronQwen2VLForCausalLM(NeuronBaseForCausalLM):
    """
    Qwen2-VL causal LM application head for Neuron inference.

    Handles weight conversion from the HF Qwen2VLForConditionalGeneration
    checkpoint layout to the NxDI key layout.

    HF checkpoint structure:
      lm_head.weight                   (top-level)
      model.embed_tokens.weight
      model.layers.{i}.input_layernorm.weight
      model.layers.{i}.post_attention_layernorm.weight
      model.layers.{i}.mlp.{gate,up,down}_proj.weight
      model.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}
      model.layers.{i}.self_attn.o_proj.weight
      model.norm.weight
      visual.*                          (stripped)

    The framework strips the "model." prefix before calling
    convert_hf_to_neuron_state_dict, so incoming keys are:
      embed_tokens.weight
      layers.{i}.*
      norm.weight
      lm_head.weight                   (already at top-level)
      visual.*
    """

    _model_cls = NeuronQwen2VLModel

    @classmethod
    def get_config_cls(cls) -> Type[Qwen2VLInferenceConfig]:
        return Qwen2VLInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
        """
        Convert HF Qwen2-VL state dict to NxDI Neuron key layout.

        Transformations applied:
          1. Strip visual encoder keys (visual.*).
          2. Per-layer renames:
             - rank metadata injected for SPMDRank
             - q/k/v_proj.{weight,bias} moved under qkv_proj sub-module
             - o_proj.weight moved under o_proj.o_proj
          3. Global rank metadata injected for base model.
          4. Optional vocab-parallel embedding rank injected.
        """
        tp_degree = config.neuron_config.tp_degree
        num_layers = config.num_hidden_layers

        # 1. Strip vision encoder keys (not needed for text LM inference)
        vision_keys = [k for k in list(state_dict.keys()) if k.startswith("visual.")]
        for k in vision_keys:
            del state_dict[k]

        # 2. Per-layer renames
        for i in range(num_layers):
            # Rank metadata for attention SPMDRank
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

            # QKV projections: q/k/v_proj.{weight,bias} -> qkv_proj.q/k/v_proj.{weight,bias}
            for proj in ("q_proj", "k_proj", "v_proj"):
                for suffix in ("weight", "bias"):
                    old = f"layers.{i}.self_attn.{proj}.{suffix}"
                    new = f"layers.{i}.self_attn.qkv_proj.{proj}.{suffix}"
                    if old in state_dict:
                        state_dict[new] = state_dict.pop(old)

            # Output projection: o_proj.weight -> o_proj.o_proj.weight
            old_o = f"layers.{i}.self_attn.o_proj.weight"
            new_o = f"layers.{i}.self_attn.o_proj.o_proj.weight"
            if old_o in state_dict:
                state_dict[new_o] = state_dict.pop(old_o)

        # 3. Global rank metadata for base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        # 4. Optional vocab-parallel embedding rank
        if config.neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, config.neuron_config.local_ranks_size, dtype=torch.int32
            )

        return state_dict


# ---------------------------------------------------------------------------
# Multimodal Config
# ---------------------------------------------------------------------------

class Qwen2VLMultimodalInferenceConfig(ImageToTextInferenceConfig):
    """
    Top-level InferenceConfig for full Qwen2-VL (text + vision encoder).

    The HF config.json uses a nested structure:
      text_config:   { hidden_size, num_hidden_layers, ... }
      vision_config: { depth, embed_dim, num_heads, ... }
      image_token_id: 151655

    Construction pattern:
      hf_cfg_dict = AutoConfig.from_pretrained(path).to_dict()
      inf_cfg = Qwen2VLMultimodalInferenceConfig(
          text_neuron_config=text_neuron_cfg,
          vision_neuron_config=vision_neuron_cfg,
          **hf_cfg_dict,
      )
    """

    def __init__(
        self,
        text_neuron_config,
        vision_neuron_config,
        fused_spec_config=None,
        load_config=None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
            fused_spec_config=fused_spec_config,
            load_config=load_config,
            metadata=metadata,
            **kwargs,
        )
        # Re-wrap vision_config as Qwen2VLVisionInferenceConfig so that
        # add_derived_config() is called and mlp_hidden_dim/head_dim are set.
        vision_attrs = {
            k: v for k, v in vars(self.vision_config).items() if k != "neuron_config"
        }
        self.vision_config = Qwen2VLVisionInferenceConfig(
            neuron_config=vision_neuron_config, **vision_attrs
        )

    def get_required_attributes(self) -> List[str]:
        return [
            "text_config",
            "vision_config",
            "image_token_id",
            "text_config.hidden_size",
            "text_config.num_hidden_layers",
            "text_config.num_attention_heads",
            "text_config.num_key_value_heads",
            "text_config.vocab_size",
            "text_config.rope_theta",
            "text_config.rms_norm_eps",
            "text_config.max_position_embeddings",
            "vision_config.depth",
            "vision_config.embed_dim",
            "vision_config.num_heads",
            "vision_config.patch_size",
            "vision_config.temporal_patch_size",
            "vision_config.spatial_merge_size",
            "vision_config.hidden_size",
            "vision_config.mlp_ratio",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


# ---------------------------------------------------------------------------
# Multimodal Application Head
# ---------------------------------------------------------------------------

class NeuronQwen2VLForConditionalGeneration(NeuronBaseForImageToText):
    """
    Full Qwen2-VL multimodal model for Neuron inference.

    Combines:
      - text_model_cls: NeuronQwen2VLModel (with encode_vision_to_input)
      - vision_model_cls: NeuronQwen2VLVisionModel (32-layer ViT + PatchMerger)
      - Qwen2VLVisionModelWrapper for CPU patchification and bucketing
    """

    text_model_cls = NeuronQwen2VLModel
    vision_model_cls = NeuronQwen2VLVisionModel

    text_model_wrapper = ImageToTextModelWrapper
    vision_model_wrapper = Qwen2VLVisionModelWrapper

    def __init__(self, *args, **kwargs):
        super().__init__(
            self.text_model_cls,
            self.vision_model_cls,
            self.text_model_wrapper,
            self.vision_model_wrapper,
            *args,
            **kwargs,
        )

    @classmethod
    def get_config_cls(cls) -> Type[Qwen2VLMultimodalInferenceConfig]:
        return Qwen2VLMultimodalInferenceConfig

    def get_compiler_args(self) -> str:
        cc_pipeline_tiling_factor = self.text_config.neuron_config.cc_pipeline_tiling_factor
        return (
            f"--enable-saturate-infinity --auto-cast=none --model-type=transformer "
            f"--tensorizer-options='--enable-ccop-compute-overlap "
            f"--cc-pipeline-tiling-factor={cc_pipeline_tiling_factor}' -O1 "
            f"--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    def get_vision_compiler_args(self) -> str:
        return (
            "--enable-saturate-infinity --auto-cast=none --model-type=transformer "
            "--tensorizer-options='--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2' -O1 "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    def enable_vision_encoder(self, enable_wlt_optimization: bool = True, **model_init_kwargs):
        new_config = copy.deepcopy(self.config)
        # Ensure vision neuron_config is used for the wrapper
        new_config.neuron_config = copy.deepcopy(new_config.vision_config.neuron_config)

        self.vision_encoder_model = self.vision_model_wrapper(
            config=new_config,
            model_cls=self.vision_model_cls,
            tag=VISION_ENCODER_MODEL_TAG,
            compiler_args=self.get_vision_compiler_args(),
            model_init_kwargs=model_init_kwargs,
            priority_model_idx=(0 if enable_wlt_optimization else None),
            pipeline_execution=True,
            return_ranked_to_cpu=True,
        )
        self.vision_models.append(self.vision_encoder_model)

    def get_required_kwargs(self) -> List[str]:
        return ["pixel_values", "grid_thw", "vision_mask"]

    def get_padding_length(self, input_ids: torch.Tensor) -> int:
        buckets = self.context_encoding_model.config.neuron_config.buckets
        for val in buckets:
            if val >= input_ids.shape[1]:
                return val
        raise RuntimeError(f"No bucket found for input_ids of length {input_ids.shape[1]}")

    def forward_atomic_prefill(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_thw: Optional[torch.LongTensor] = None,
        vision_mask: Optional[torch.FloatTensor] = None,
    ):
        """Run vision encoder and inject embeddings into context encoding pass."""
        # Build vision mask (positions of vision tokens in the sequence)
        if vision_mask is None:
            vision_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            vision_mask = vision_mask.to(torch.bool)
        assert vision_mask.dtype == torch.bool
        vision_mask = generate_positions_from_mask(vision_mask.squeeze())

        # Run vision encoder: returns [N_merged, 3584]
        vision_embeddings = self.vision_encoder_model(
            pixel_values.to(self.vision_config.neuron_config.torch_dtype), grid_thw
        ).to(self.text_config.neuron_config.torch_dtype)

        # Pad to text context bucket
        pad_limit = self.get_padding_length(input_ids)
        vision_mask = pad_positions(vision_mask, pad_limit, (pad_limit - 1))
        # Add batch dim if needed
        if vision_embeddings.dim() == 2:
            vision_embeddings = vision_embeddings.unsqueeze(0)
        vision_embeddings = pad_vision_embeddings(vision_embeddings, pad_limit)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            vision_embeddings=vision_embeddings,
            vision_mask=vision_mask,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_thw: Optional[torch.LongTensor] = None,
        vision_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        has_images = (
            pixel_values is not None
            and pixel_values.sum() != 0
            and input_ids.shape[-1] > 1
        )

        if has_images:
            outputs = []
            for i in range(input_ids.shape[0]):
                pv_i = pixel_values[i].unsqueeze(0) if pixel_values.dim() > 2 else pixel_values
                gt_i = grid_thw[i].unsqueeze(0) if grid_thw is not None else None
                vm_i = vision_mask[i].unsqueeze(0) if vision_mask is not None else None
                outputs.append(
                    self.forward_atomic_prefill(
                        input_ids[i].unsqueeze(0),
                        attention_mask[i].unsqueeze(0) if attention_mask is not None else None,
                        position_ids[i].unsqueeze(0) if position_ids is not None else None,
                        seq_ids[i].unsqueeze(0) if seq_ids is not None else None,
                        sampling_params[i].unsqueeze(0) if sampling_params is not None else None,
                        pv_i, gt_i, vm_i,
                    )
                )
            return outputs[0] if len(outputs) == 1 else outputs
        else:
            pad_limit = self.get_padding_length(input_ids)
            vision_embeddings, vision_mask_dummy = self.text_model_wrapper.get_dummy_vision_inputs(
                config=self.text_config,
                input_ids=input_ids,
                n_active_tokens=pad_limit,
                fill_value=(pad_limit - 1),
            )
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                seq_ids=seq_ids,
                sampling_params=sampling_params,
                vision_embeddings=vision_embeddings,
                vision_mask=vision_mask_dummy,
            )

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config) -> dict:
        """
        Convert HF Qwen2-VL state dict to NxDI Neuron key layout.

        The framework strips the "model." prefix before calling this, so:
          - Text keys arrive as: embed_tokens.*, layers.*, norm.*, lm_head.*
          - Vision keys arrive as: visual.*

        Transformations:
          Text:
            1. layers.{i}.self_attn.{q,k,v}_proj.* → qkv_proj.{q,k,v}_proj.*
            2. layers.{i}.self_attn.o_proj.weight → o_proj.o_proj.weight
            3. Inject rank metadata tensors

          Vision:
            4. visual.patch_embed.proj.weight → patch_embed.proj.weight
               (reshape [1280,3,2,14,14] → [1280,1176])
            5. visual.merger.mlp.0.* → merger.mlp_fc1.*
            6. visual.merger.mlp.2.* → merger.mlp_fc2.*
            7. visual.blocks.{i}.* → blocks.{i}.*  (bulk prefix strip)
            8. blocks.{i}.attn.qkv.{weight,bias}
               → blocks.{i}.attn.qkv_proj.{q,k,v}_proj.{weight,bias}  (split fused QKV)
            9. blocks.{i}.attn.proj.{weight,bias}
               → blocks.{i}.attn.o_proj.o_proj.{weight,bias}
        """
        # Get text config (for text model parameters)
        text_cfg = getattr(config, "text_config", config)
        tp_degree = text_cfg.neuron_config.tp_degree
        num_layers = text_cfg.num_hidden_layers

        # ── Text model renames ────────────────────────────────────────────────
        for i in range(num_layers):
            # Rank metadata for attention SPMDRank
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
            # QKV projections
            for proj in ("q_proj", "k_proj", "v_proj"):
                for suffix in ("weight", "bias"):
                    old = f"layers.{i}.self_attn.{proj}.{suffix}"
                    new = f"layers.{i}.self_attn.qkv_proj.{proj}.{suffix}"
                    if old in state_dict:
                        state_dict[new] = state_dict.pop(old)
            # Output projection
            old_o = f"layers.{i}.self_attn.o_proj.weight"
            new_o = f"layers.{i}.self_attn.o_proj.o_proj.weight"
            if old_o in state_dict:
                state_dict[new_o] = state_dict.pop(old_o)

        # Global rank metadata
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        if text_cfg.neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, text_cfg.neuron_config.local_ranks_size, dtype=torch.int32
            )

        # ── Vision model renames ──────────────────────────────────────────────
        # Vision model is loaded standalone (no parent prefix), so output keys
        # should NOT have a "vision_transformer." prefix. They should match
        # the NeuronQwen2VLVisionModel state dict directly:
        #   patch_embed.*, blocks.{i}.*, merger.*
        PREFIX_IN = "visual."
        vision_cfg = getattr(config, "vision_config", None)

        # patch_embed: reshape Conv3d weight [1280,3,2,14,14] → [1280,1176]
        pe_key_in = "visual.patch_embed.proj.weight"
        pe_key_out = "patch_embed.proj.weight"
        if pe_key_in in state_dict:
            embed_dim = getattr(vision_cfg, "embed_dim", 1280) if vision_cfg is not None else 1280
            state_dict[pe_key_out] = state_dict.pop(pe_key_in).reshape(embed_dim, -1)

        # Collect remaining visual.* keys for bulk rename
        vision_keys = [k for k in list(state_dict.keys()) if k.startswith(PREFIX_IN)]
        for key in vision_keys:
            # Strip "visual." prefix (no replacement prefix)
            new_key = key[len(PREFIX_IN):]
            # merger.mlp.0.* → merger.mlp_fc1.*
            # merger.mlp.2.* → merger.mlp_fc2.*
            new_key = new_key.replace("merger.mlp.0.", "merger.mlp_fc1.")
            new_key = new_key.replace("merger.mlp.2.", "merger.mlp_fc2.")
            state_dict[new_key] = state_dict.pop(key)

        # Vision block-level renames: split fused QKV and rename output projection.
        num_vision_layers = getattr(vision_cfg, "depth", 32) if vision_cfg is not None else 32

        for i in range(num_vision_layers):
            # Split fused blocks.{i}.attn.qkv.{weight,bias}
            # → blocks.{i}.attn.qkv_proj.{q,k,v}_proj.{weight,bias}
            for suffix in ("weight", "bias"):
                fused_key = f"blocks.{i}.attn.qkv.{suffix}"
                if fused_key in state_dict:
                    fused = state_dict.pop(fused_key)
                    q, k, v = fused.chunk(3, dim=0)
                    state_dict[f"blocks.{i}.attn.qkv_proj.q_proj.{suffix}"] = q
                    state_dict[f"blocks.{i}.attn.qkv_proj.k_proj.{suffix}"] = k
                    state_dict[f"blocks.{i}.attn.qkv_proj.v_proj.{suffix}"] = v

            # Rename output projection: blocks.{i}.attn.proj.* → blocks.{i}.attn.o_proj.o_proj.*
            for suffix in ("weight", "bias"):
                old_proj = f"blocks.{i}.attn.proj.{suffix}"
                new_proj = f"blocks.{i}.attn.o_proj.o_proj.{suffix}"
                if old_proj in state_dict:
                    state_dict[new_proj] = state_dict.pop(old_proj)

        return state_dict

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration.from_pretrained(model_path, **kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Text-only (legacy)
    "NeuronQwen2VLForCausalLM",
    "NeuronQwen2VLModel",
    "Qwen2VLInferenceConfig",
    # Multimodal
    "NeuronQwen2VLForConditionalGeneration",
    "Qwen2VLMultimodalInferenceConfig",
]
