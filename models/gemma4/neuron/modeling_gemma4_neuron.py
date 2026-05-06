"""
Top-level Gemma4 NxDI model: config classes + application head.

Scope:
  - Gemma4NeuronConfig      : NeuronConfig subclass with MoE + Gemma4-specific fields
  - Gemma4InferenceConfig   : InferenceConfig subclass for the TEXT backbone
  - Gemma4VisionInferenceConfig : InferenceConfig subclass for the VISION tower
  - Gemma4MultimodalInferenceConfig : ImageToTextInferenceConfig wrapping both
  - NeuronGemma4ForCausalLM : text-only causal LM (Gemma4ForCausalLM equivalent)
  - NeuronGemma4ForConditionalGeneration : multimodal head (Gemma4ForConditionalGeneration)

Weight conversion (convert_hf_to_neuron_state_dict) is a placeholder — the
Phase 4 subagent replaces it with a real implementation.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import List, Optional, Type

import torch
import torch.nn as nn

# Make `blocks.*` importable both when this file is imported from the
# `models/gemma4/neuron` directory AND when imported via a package path.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
    NeuronConfig,
)
from neuronx_distributed_inference.models.image_to_text_model_base import (
    ImageToTextInferenceConfig,
    NeuronBaseForImageToText,
)
from neuronx_distributed_inference.models.image_to_text_model_wrapper import (
    ImageToTextModelWrapper,
)
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM

from blocks.neuron_gemma4_decoder import NeuronGemma4TextModel
from blocks.neuron_gemma4_embeddings import apply_logit_softcap
from blocks.neuron_gemma4_vision_model import NeuronGemma4VisionTower


# =============================================================================
# NeuronConfig — extends MoENeuronConfig because Gemma4 is MoE.
# =============================================================================
class Gemma4NeuronConfig(MoENeuronConfig):
    """
    Neuron-specific config. Defaults mirror Gemma4 assumptions:
      * router dtype = fp32 (see `neuron_gemma4_moe.Gemma4Router`)
      * normalize_top_k_affinities = False (Gemma4 normalizes INSIDE the router
        and then multiplies by per_expert_scale — stock post-normalization
        would clobber those factors)
    """

    def __init__(self, **kwargs):
        # Force Gemma4-specific defaults before delegating to parent.
        kwargs.setdefault("torch_dtype", torch.bfloat16)
        # normalize_top_k_affinities is a field on MoENeuronConfig; setting it
        # here after super().__init__ because it may be read by sub-configs.
        super().__init__(**kwargs)
        # Router compute must be fp32 (affinities + softmax + topk).
        if hasattr(self, "router_config") and self.router_config is not None:
            self.router_config.dtype = torch.float32
        # Gemma4 normalizes inside the router; disable stock re-normalization.
        if hasattr(self, "normalize_top_k_affinities"):
            self.normalize_top_k_affinities = False


# =============================================================================
# Text InferenceConfig.
# =============================================================================
_TEXT_REQUIRED_ATTRIBUTES = [
    # core Llama-ish fields
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "max_position_embeddings",
    "rms_norm_eps",
    "pad_token_id",
    "vocab_size",
    # Gemma4-specific
    "sliding_window",
    "layer_types",
    "final_logit_softcapping",
    "global_head_dim",
    "num_global_key_value_heads",
    "attention_k_eq_v",
    "enable_moe_block",
    "num_experts",
    "num_local_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "hidden_size_per_layer_input",
    "vocab_size_per_layer_input",
]


class Gemma4InferenceConfig(InferenceConfig):
    """Text-backbone inference config. Surfaces Gemma4 text_config fields."""

    def get_required_attributes(self) -> List[str]:
        return _TEXT_REQUIRED_ATTRIBUTES

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return Gemma4NeuronConfig

    def add_derived_config(self):
        """
        Populate attributes that the HF Gemma4TextConfig computes at Python
        construction time but that don't round-trip through config.json.
        """
        # Map HF `hidden_activation` -> NxDI-standard `hidden_act`.
        if not hasattr(self, "hidden_act") or self.hidden_act is None:
            ha = getattr(self, "hidden_activation", "gelu_pytorch_tanh")
            self.hidden_act = ha

        # MoE primitive uses num_local_experts / num_experts_per_tok.
        if not hasattr(self, "num_local_experts") or self.num_local_experts is None:
            self.num_local_experts = getattr(self, "num_experts", None)
        if (
            not hasattr(self, "num_experts_per_tok")
            or self.num_experts_per_tok is None
        ):
            self.num_experts_per_tok = getattr(self, "top_k_experts", None)

        # Gemma4 has no shared experts.
        if not hasattr(self, "n_shared_experts") or self.n_shared_experts is None:
            self.n_shared_experts = 0

        # layer_types default (5 sliding + 1 full, last forced to full). When
        # the JSON already provides the list we keep it verbatim.
        layer_types = getattr(self, "layer_types", None)
        num_layers = getattr(self, "num_hidden_layers", 0)
        if layer_types is None and num_layers:
            pattern = ["sliding_attention"] * 5 + ["full_attention"]
            lt = [pattern[i % 6] for i in range(num_layers)]
            lt[-1] = "full_attention"  # last layer always full
            self.layer_types = lt

        # Per-layer RoPE theta fallbacks (if not using the dict-of-dicts).
        rope_params = getattr(self, "rope_parameters", None)
        if rope_params is not None:
            if "sliding_attention" in rope_params:
                self.local_rope_theta = rope_params["sliding_attention"].get(
                    "rope_theta", 10000.0
                )
            if "full_attention" in rope_params:
                self.global_rope_theta = rope_params["full_attention"].get(
                    "rope_theta", 1_000_000.0
                )
                self.global_partial_rotary_factor = rope_params[
                    "full_attention"
                ].get("partial_rotary_factor", 0.25)

        # num_cores_per_group required by flash-decoding path; harmless default.
        if not hasattr(self, "num_cores_per_group") or self.num_cores_per_group is None:
            self.num_cores_per_group = 1


# =============================================================================
# Vision InferenceConfig.
# =============================================================================
_VISION_REQUIRED_ATTRIBUTES = [
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "patch_size",
    "position_embedding_size",
    "pooling_kernel_size",
    "standardize",
    "rms_norm_eps",
]


class Gemma4VisionInferenceConfig(InferenceConfig):
    def get_required_attributes(self) -> List[str]:
        return _VISION_REQUIRED_ATTRIBUTES

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        # Vision tower is dense (no MoE), so plain NeuronConfig is fine.
        return NeuronConfig

    def add_derived_config(self):
        if not hasattr(self, "hidden_act") or self.hidden_act is None:
            self.hidden_act = getattr(self, "hidden_activation", "gelu_pytorch_tanh")
        if not hasattr(self, "num_channels") or self.num_channels is None:
            self.num_channels = 3
        rope_params = getattr(self, "rope_parameters", None)
        if rope_params is not None and not hasattr(self, "rope_theta"):
            self.rope_theta = rope_params.get("rope_theta", 100.0)


# =============================================================================
# Multimodal InferenceConfig (top level, wraps text + vision).
# =============================================================================
_MULTIMODAL_REQUIRED_ATTRIBUTES = [
    "text_config",
    "vision_config",
    "image_token_id",
    "vision_soft_tokens_per_image",
    # Nested — validated recursively via hasattr_nested.
    "text_config.hidden_size",
    "text_config.num_hidden_layers",
    "text_config.vocab_size",
    "text_config.pad_token_id",
    "vision_config.hidden_size",
    "vision_config.num_hidden_layers",
    "vision_config.patch_size",
]


class Gemma4MultimodalInferenceConfig(ImageToTextInferenceConfig):
    """Top-level Gemma4 VLM config. Wraps text + vision InferenceConfigs."""

    def get_required_attributes(self) -> List[str]:
        return _MULTIMODAL_REQUIRED_ATTRIBUTES

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return Gemma4NeuronConfig


# =============================================================================
# Text-only causal LM (useful for testing the text backbone in isolation).
# =============================================================================
class NeuronGemma4ForCausalLM(NeuronBaseForCausalLM):
    """Gemma4 text-only causal LM. Applies `final_logit_softcapping` post-LM-head."""

    _model_cls = NeuronGemma4TextModel
    _STATE_DICT_MODEL_PREFIX = "language_model.model."

    @classmethod
    def get_config_cls(cls):
        return Gemma4InferenceConfig

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        # Phase 4 will either import the HF class directly or supply a loader;
        # for scaffolding we defer to a transformers.AutoModel call.
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    def get_compiler_args(self) -> str:
        return (
            "--enable-saturate-infinity --enable-mixed-precision-accumulation "
            "--model-type transformer -O1 "
            "--tensorizer-options='--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2' "
            "--auto-cast=none "
            "--internal-enable-dge-levels vector_dynamic_offsets "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: InferenceConfig
    ) -> dict:
        """Placeholder — implemented in Phase 4."""
        return state_dict


# =============================================================================
# Multimodal head — drives the full image-to-text flow.
# =============================================================================
class NeuronGemma4ForConditionalGeneration(NeuronBaseForImageToText):
    """Gemma4 multimodal head.

    Equivalent to HF `Gemma4ForConditionalGeneration`: scatters vision
    embeddings into text `inputs_embeds` at image-token positions.
    """

    text_model_cls = NeuronGemma4TextModel
    vision_model_cls = NeuronGemma4VisionTower

    # Reuse NxDI's image-to-text wrapper. For Gemma4 we will likely need a
    # small vision-specific wrapper later (patch layout, bucket routing), but
    # the default ImageToTextModelWrapper is sufficient for scaffolding.
    text_model_wrapper = ImageToTextModelWrapper
    vision_model_wrapper = ImageToTextModelWrapper

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
    def get_config_cls(cls):
        return Gemma4MultimodalInferenceConfig

    def get_required_kwargs(self) -> List[str]:
        return [
            "pixel_values",
            "pixel_position_ids",
            "vision_mask",
        ]

    def get_compiler_args(self) -> str:
        return (
            "--enable-saturate-infinity --auto-cast=none --model-type=transformer "
            "--tensorizer-options='--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2 --vectorize-strided-dma' -O1 "
            "--hbm-scratchpad-page-size=1024 "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    def get_vision_compiler_args(self) -> str:
        return self.get_compiler_args()

    def enable_vision_encoder(self, enable_wlt_optimization: bool = True, **model_init_kwargs):
        """Register the vision tower model wrapper.

        Minimal stub: defers bucket-plan tuning to a follow-up pass. Mirrors the
        Pixtral pattern closely — uses vision_config.neuron_config as the
        wrapper's neuron_config and appends to self.vision_models.
        """
        from neuronx_distributed_inference.models.model_wrapper import (
            VISION_ENCODER_MODEL_TAG,
        )

        new_config = copy.deepcopy(self.config)
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

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        # Gemma4 has tie_word_embeddings=True (text_config).
        # Phase 4 converter handles the actual tying; keep as no-op here.
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: InferenceConfig
    ) -> dict:
        """Placeholder — implemented in Phase 4."""
        return state_dict
