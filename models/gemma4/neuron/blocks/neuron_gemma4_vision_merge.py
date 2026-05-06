"""
BLOCK F — Gemma4 Vision Patch Embedder + 2D PosEmbed + Pooler + Multimodal Projector + Scatter.

Covers everything on the vision path that is NOT the transformer stack:
  1) NeuronGemma4VisionPatchEmbedder
  2) NeuronGemma4VisionPooler
  3) NeuronGemma4VisionMergeModel (composes patch_embedder -> [encoder] -> pooler -> standardize)
  4) NeuronGemma4MultimodalEmbedder (RMSNorm-no-scale + projection to text hidden)
  5) encode_vision_to_input helper (scatter_by_index_put)

Deviations from the HF source (documented inline):
  (D1) 2D position embedding: HF uses one_hot(position_embedding_size=10240) @ table
       which materializes a 10240-wide one-hot matrix. We replace with a direct
       `index_select` / gather along dim=1 of the [2, 10240, H_vis] table:
           pos = table[0][x_ids] + table[1][y_ids]
       Mathematically identical, no huge one-hot tensor.
  (D2) Pooler dynamic padding strip (`hidden_states[pooler_mask]`) is not traceable.
       We keep a fixed [B, S_soft_max, H] output and a companion `pooler_mask`.
       Downstream scatter is position-based (vision_mask in text tokens), so no
       strip is needed — padded rows are simply never scattered into the text.
  (D3) Standardize: registered as buffers `std_scale`/`std_bias` of shape [H_vis]
       (per-channel), applied after pooling when config.standardize=True.
  (D4) Multimodal embedder RMSNorm has no scale (Gemma4RMSNorm with_scale=False)
       — implemented as NoScaleRMSNorm (just 1/rms, no weight multiply).
  (D5) Pre-scale `2 * (pixel - 0.5)` in patch embedder forward — preserved verbatim.
  (D6) Padding patches mask: source zeros positions where `(position_ids == -1).all(dim=-1)`.
       We mirror this via clamp(min=0) + `where(padding_positions, 0.0, pos_emb)`.
  (D7) Scatter integration: `encode_vision_to_input` wraps
       `scatter_by_index_put` from llama4.utils.encoder_utils.
  (D8) `input_proj` uses ColumnParallelLinear with gather_output=True. The 2D
       position-embedding TABLE is a replicated `nn.Parameter` — NOT TP-sharded
       (10240 rows is small, and we index into it so sharding would require
       all-gather anyway).
  (D9) Pooler does NOT fold `root_hidden_size * padding_zero` into one matmul; we
       mirror the source order: zero padding rows, (optionally) pool, then scale
       by sqrt(H_vis). Identical math, simpler trace.
  (D10) Source pooler `_avg_pool_by_positions` uses `F.one_hot(kernel_idxs, length)`
       where length is the number of output soft tokens (small, e.g. 280) — this
       one-hot is kept as-is (small enough to trace efficiently). We compute
       max_x from the position tensor as `clamped_positions[..., 0].max(...) + 1`
       exactly as the source does.
  (D11) `pooler_mask` in source is derived from `(weights == 0).all(dim=1)` negated.
       We compute identically on `weights: [B, P, S_soft]` (zero columns mean no
       patch mapped there). Produces boolean mask [B, S_soft].
"""

from typing import Optional, Tuple

import torch
from torch import nn

from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear
from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.models.llama4.utils.encoder_utils import (
    scatter_by_index_put,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm


# --------------------------------------------------------------------------- #
# NoScaleRMSNorm — Gemma4RMSNorm(with_scale=False) equivalent.
# --------------------------------------------------------------------------- #
class NoScaleRMSNorm(nn.Module):
    """RMSNorm without a learnable scale parameter. Matches Gemma4RMSNorm(with_scale=False)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        mean_squared = x.pow(2).mean(-1, keepdim=True) + self.eps
        # Use torch.pow (over torch.sqrt/rsqrt) to match source (addresses Torch/JAX compiler differences).
        x = x * torch.pow(mean_squared, -0.5)
        return x.to(original_dtype)


# --------------------------------------------------------------------------- #
# Patch embedder.
# --------------------------------------------------------------------------- #
class NeuronGemma4VisionPatchEmbedder(nn.Module):
    """
    Neuron translation of Gemma4VisionPatchEmbedder.

    Inputs:
        pixel_values:       [B, P_max, 3*patch_size*patch_size]   float
        pixel_position_ids: [B, P_max, 2]                         int64
                            (padding patches have (-1, -1))
        padding_positions:  [B, P_max] bool (True = padding)

    Output:
        hidden_states:      [B, P_max, hidden_size] bf16

    NOTE: Source `forward` takes raw `pixel_values: [B, C, H_img, W_img]` and relies
    on an external unfolder. We take the post-unfold flattened layout
    `[B, P_max, 3*patch*patch]` because NxDI prefers pre-flattened inputs (matches
    how Pixtral's `vision_patch_conv_linear` is fed). Caller must unfold upstream
    (the Gemma4 image processor already produces this layout).
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        vcfg = config.vision_config if hasattr(config, "vision_config") else config
        self.hidden_size = vcfg.hidden_size
        self.patch_size = vcfg.patch_size
        self.position_embedding_size = vcfg.position_embedding_size

        patch_in = 3 * self.patch_size * self.patch_size
        # D8: ColumnParallelLinear with gather_output=True so downstream is replicated.
        self.input_proj = ColumnParallelLinear(
            patch_in,
            self.hidden_size,
            bias=False,
            gather_output=True,
            dtype=config.neuron_config.torch_dtype,
        )
        # D8: Replicated table — NOT TP-sharded (indexed lookup, small inner dim).
        # Shape [2, position_embedding_size, hidden_size] matches source.
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, self.position_embedding_size, self.hidden_size,
                       dtype=config.neuron_config.torch_dtype)
        )

    def _position_embeddings(
        self,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        # D6: clamp (-1) padding sentinels to 0 so we can index safely.
        clamped = pixel_position_ids.clamp(min=0).long()
        x_ids = clamped[..., 0]  # [B, P_max]
        y_ids = clamped[..., 1]  # [B, P_max]

        # D1: gather replaces one_hot@table.
        # table: [2, N_pos, H_vis]
        # table[0][x_ids] -> [B, P_max, H_vis]; same for [1][y_ids]
        pos_emb = self.position_embedding_table[0][x_ids] + self.position_embedding_table[1][y_ids]

        # D6: Zero out padding rows.
        pos_emb = torch.where(padding_positions.unsqueeze(-1), torch.zeros_like(pos_emb), pos_emb)
        return pos_emb

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        # D5: source pre-scale.
        pixel_values = 2.0 * (pixel_values - 0.5)
        hidden_states = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        pos_emb = self._position_embeddings(pixel_position_ids, padding_positions)
        return hidden_states + pos_emb


# --------------------------------------------------------------------------- #
# Vision pooler.
# --------------------------------------------------------------------------- #
class NeuronGemma4VisionPooler(nn.Module):
    """
    Neuron translation of Gemma4VisionPooler.

    Order of ops mirrors the source:
       1) zero out padding rows: hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0)
       2) if input seq_len != output_length: 2D average pool via position-indexed one-hot
       3) multiply by sqrt(hidden_size)
    Returns (pooled: [B, S_soft, H], pooler_mask: [B, S_soft]).
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        vcfg = config.vision_config if hasattr(config, "vision_config") else config
        self.hidden_size = vcfg.hidden_size
        self.root_hidden_size = self.hidden_size ** 0.5

    def _avg_pool_by_positions(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k * k
        if k_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool {hidden_states.shape} to {length}: k={k}^2 * length={length} "
                f"must be {input_seq_len}."
            )

        # D10: identical to source. Padding sentinels clamped to 0; zero hidden_states rows
        # contribute nothing to the average.
        clamped_positions = pixel_position_ids.clamp(min=0).long()
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1  # [B, 1]
        kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]  # [B, P]

        # one_hot over `length` (small: = vision_soft_tokens_per_image, e.g. 280). Safe to trace.
        weights = torch.nn.functional.one_hot(kernel_idxs.long(), length).float() / k_squared
        # weights: [B, P, S_soft]
        output = weights.transpose(1, 2) @ hidden_states.float()  # [B, S_soft, H]
        # D11: mask[b, s] = True if any patch maps to slot s.
        mask = torch.logical_not((weights == 0).all(dim=1))  # [B, S_soft]
        return output.to(hidden_states.dtype), mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Step 1: zero padding rows.
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)

        pooler_mask = ~padding_positions  # identity pass-through mask if no pooling
        if hidden_states.shape[1] != output_length:
            hidden_states, pooler_mask = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )

        # D9: scale last, matching source line order.
        hidden_states = hidden_states * self.root_hidden_size
        return hidden_states, pooler_mask


# --------------------------------------------------------------------------- #
# Multimodal embedder (vision hidden -> text hidden).
# --------------------------------------------------------------------------- #
class NeuronGemma4MultimodalEmbedder(nn.Module):
    """
    Neuron translation of Gemma4MultimodalEmbedder.

    Pipeline:
        x -> NoScaleRMSNorm(H_vis) -> ColumnParallelLinear(H_vis -> H_text) -> y
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        vcfg = config.vision_config if hasattr(config, "vision_config") else config
        tcfg = config.text_config if hasattr(config, "text_config") else config

        self.multimodal_hidden_size = getattr(vcfg, "output_proj_dims", vcfg.hidden_size)
        self.text_hidden_size = tcfg.hidden_size
        self.eps = getattr(vcfg, "rms_norm_eps", 1e-6)

        # D4: no-scale RMSNorm.
        self.embedding_pre_projection_norm = NoScaleRMSNorm(eps=self.eps)
        # D8: ColumnParallelLinear with gather_output=True (downstream needs replicated).
        self.embedding_projection = ColumnParallelLinear(
            self.multimodal_hidden_size,
            self.text_hidden_size,
            bias=False,
            gather_output=True,
            dtype=config.neuron_config.torch_dtype,
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        x = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(x)


# --------------------------------------------------------------------------- #
# Vision merge model (patch embedder -> [encoder] -> pooler -> standardize).
# --------------------------------------------------------------------------- #
class NeuronGemma4VisionMergeModel(nn.Module):
    """
    Composes the non-transformer vision path:
        patch_embedder -> (external encoder passed via encoder_forward_fn) -> pooler -> standardize

    This block does NOT include the transformer stack (Block E is a separate block).
    For testing we accept an optional `encoder_forward_fn` that, given patch embeddings,
    returns post-encoder hidden states. In production this is wired to the Block-E
    transformer.
    """

    def __init__(
        self,
        config: InferenceConfig,
        encoder_forward_fn=None,
    ):
        super().__init__()
        self.config = config
        vcfg = config.vision_config if hasattr(config, "vision_config") else config
        self.vcfg = vcfg
        self.patch_embedder = NeuronGemma4VisionPatchEmbedder(config)
        self.pooler = NeuronGemma4VisionPooler(config)
        self.encoder_forward_fn = encoder_forward_fn
        self.standardize = getattr(vcfg, "standardize", False)

        if self.standardize:
            # D3: per-channel standardize parameters, matching source shape [H_vis].
            self.register_buffer("std_bias", torch.zeros(vcfg.hidden_size,
                                                         dtype=config.neuron_config.torch_dtype))
            self.register_buffer("std_scale", torch.ones(vcfg.hidden_size,
                                                         dtype=config.neuron_config.torch_dtype))

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        output_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        padding_positions = (pixel_position_ids == -1).all(dim=-1)  # [B, P]
        hidden_states = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)

        if self.encoder_forward_fn is not None:
            hidden_states = self.encoder_forward_fn(
                hidden_states=hidden_states,
                attention_mask=~padding_positions,
                pixel_position_ids=pixel_position_ids,
            )

        hidden_states, pooler_mask = self.pooler(
            hidden_states=hidden_states,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )

        # D2: NO dynamic strip. Keep full [B, S_soft, H]; downstream scatter is position-based.
        if self.standardize:
            # D3: (x - std_bias) * std_scale, broadcast over [B, S_soft].
            hidden_states = (hidden_states - self.std_bias) * self.std_scale

        return hidden_states, pooler_mask


# --------------------------------------------------------------------------- #
# Scatter glue.
# --------------------------------------------------------------------------- #
def encode_vision_to_input(
    inputs_embeds: torch.Tensor,
    vision_embeddings: torch.Tensor,
    vision_mask: torch.Tensor,
) -> torch.Tensor:
    """
    D7: wrap scatter_by_index_put from llama4 encoder_utils.

    Args:
        inputs_embeds:     [B, S_text, H_text] bf16 — base text embeddings.
        vision_embeddings: [B, N_vision, H_text] bf16 — projected soft tokens.
        vision_mask:       [B, N_vision, 1] int/bool — positions (in flat view) where
                            vision tokens should land. Matches scatter_by_index_put's
                            positions contract.

    Returns:
        [B, S_text, H_text] bf16 — inputs_embeds with vision tokens scattered in.
    """
    return scatter_by_index_put(inputs_embeds, vision_embeddings, vision_mask)
