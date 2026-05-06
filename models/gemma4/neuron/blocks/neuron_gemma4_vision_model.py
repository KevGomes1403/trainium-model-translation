"""
Full Gemma4 vision tower composition: patch embedder (Block F) + encoder
(Block E) + pooler (Block F) + standardize + multimodal projector (Block F).

Produces `vision_embeddings` ready to scatter into the text sequence.

Source ref (models/gemma4/hf/modeling_gemma4.py):
  Gemma4VisionModel (L1958-2020) + Gemma4MultimodalEmbedder (L2023-2047)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from neuronx_distributed_inference.models.config import InferenceConfig

from blocks.neuron_gemma4_vision_encoder import NeuronGemma4VisionEncoder
from blocks.neuron_gemma4_vision_merge import (
    NeuronGemma4MultimodalEmbedder,
    NeuronGemma4VisionPatchEmbedder,
    NeuronGemma4VisionPooler,
)


class NeuronGemma4VisionTower(nn.Module):
    """
    patch_embedder -> encoder -> pooler -> (optional standardize) -> multimodal_embedder

    Inputs:
        pixel_values:       [B, P_max, 3*patch_size*patch_size] bf16
        pixel_position_ids: [B, P_max, 2] int64 (padding patches use (-1,-1))
        output_length:      int — target number of soft tokens per image
                            (typically `vision_soft_tokens_per_image`, e.g. 280)

    Outputs:
        vision_embeddings: [B, S_soft, H_text] bf16
        pooler_mask:       [B, S_soft] bool (True where a patch mapped there)
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config

        # Resolve vision sub-config — accept either a nested `vision_config` or
        # a flat config that already carries vision fields.
        vcfg = getattr(config, "vision_config", config)
        self.standardize = getattr(vcfg, "standardize", False)

        # Build a child config that the sub-blocks can read as top-level fields.
        # Each block checks for `config.vision_config` first and falls back to
        # the passed config, so we just pass the parent directly.
        self.patch_embedder = NeuronGemma4VisionPatchEmbedder(config)
        self.encoder = NeuronGemma4VisionEncoder(vcfg)
        self.pooler = NeuronGemma4VisionPooler(config)
        self.multimodal_embedder = NeuronGemma4MultimodalEmbedder(config)

        if self.standardize:
            # Per-channel standardization parameters (source: Gemma4VisionModel
            # L1997-2012; scale/bias shape = [H_vis]).
            dtype = config.neuron_config.torch_dtype
            self.register_buffer(
                "std_bias", torch.zeros(vcfg.hidden_size, dtype=dtype)
            )
            self.register_buffer(
                "std_scale", torch.ones(vcfg.hidden_size, dtype=dtype)
            )

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        output_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Derive padding_positions from the (-1, -1) sentinel used by the
        # image processor for non-valid patches.
        padding_positions = (pixel_position_ids == -1).all(dim=-1)  # [B, P]

        # Patch embed: [B, P, H_vis]
        hidden_states = self.patch_embedder(
            pixel_values, pixel_position_ids, padding_positions
        )

        # Non-causal attention mask: True where we attend (i.e. not padding).
        attention_mask = ~padding_positions

        # Encoder: [B, P, H_vis]
        hidden_states = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            pixel_position_ids=pixel_position_ids,
        )

        # Pooler: [B, S_soft, H_vis], pooler_mask [B, S_soft]
        hidden_states, pooler_mask = self.pooler(
            hidden_states=hidden_states,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )

        # Standardize in vision space (before projection) when enabled.
        if self.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale

        # Project to text hidden size: [B, S_soft, H_text]
        vision_embeddings = self.multimodal_embedder(hidden_states)
        return vision_embeddings, pooler_mask
