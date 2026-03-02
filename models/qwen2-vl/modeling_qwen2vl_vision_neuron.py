"""
NeuronX Distributed Inference model for Qwen2-VL vision encoder.

Qwen2-VL-7B vision encoder architecture:
  - PatchEmbed: Conv3d temporal+spatial patching → [num_tokens, embed_dim]
  - 32× VisionBlock: LayerNorm + MHA (16 heads) + LayerNorm + VisionMlp
  - PatchMerger: LayerNorm + MLP that projects and merges patches to text dim

Vision config (from HF config.json → vision_config):
  depth=32, embed_dim=1280, num_heads=16, mlp_ratio=4,
  patch_size=14, temporal_patch_size=2, spatial_merge_size=2,
  in_chans=3, hidden_size=3584 (text model hidden_size for projection)

This file provides:
  - Qwen2VLVisionInferenceConfig
  - NeuronQwen2VLVisionModel  (full vision encoder)
  - Qwen2VLVisionModelWrapper (CPU patchification + bucketing)
"""

import copy
import logging
import math
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.utils import cpu_mode
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_wrapper import EncoderModelInstance, ModelWrapper
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.padding import pad_tensor, unpad_tensor
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.utils.distributed import get_tp_group

from transformers.activations import ACT2FN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _layer_norm(hidden_size: int, eps: float) -> nn.Module:
    return nn.LayerNorm(hidden_size, eps=eps)


def _rms_norm(hidden_size: int, eps: float) -> nn.Module:
    if cpu_mode():
        return nn.RMSNorm(hidden_size, eps=eps)
    return CustomRMSNorm(hidden_size, eps=eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D rotary position embeddings to Q and K in vision attention.

    Args:
        q, k: [B, H, S, Hd]
        cos, sin: [S, Hd] — pre-computed on CPU, padded to bucket size
    Returns:
        Rotated (q, k) in original dtype.
    """
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q = q.float()
    k = k.float()
    cos = cos[None, None, :, :].float()  # [1, 1, S, Hd]
    sin = sin[None, None, :, :].float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


# ---------------------------------------------------------------------------
# Vision InferenceConfig
# ---------------------------------------------------------------------------

class Qwen2VLVisionInferenceConfig(InferenceConfig):
    """
    InferenceConfig for the Qwen2-VL vision encoder.

    Derives computed attributes from HF vision_config fields:
      - mlp_hidden_dim = embed_dim * mlp_ratio
      - head_dim = embed_dim // num_heads
      - hidden_size = embed_dim  (alias for NeuronAttentionBase)
      - num_attention_heads = num_heads
      - num_key_value_heads = num_heads  (MHA)
    """

    def add_derived_config(self):
        self.num_cores_per_group = 1

        # Map HF vision_config names to NxDI expected names
        embed_dim = getattr(self, "embed_dim", 1280)
        num_heads = getattr(self, "num_heads", 16)
        mlp_ratio = getattr(self, "mlp_ratio", 4)

        self.hidden_size = embed_dim
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_heads  # MHA
        self.head_dim = embed_dim // num_heads
        self.mlp_hidden_dim = embed_dim * mlp_ratio
        self.num_hidden_layers = getattr(self, "depth", 32)
        self.hidden_act = getattr(self, "hidden_act", "quick_gelu")

        # Vision RoPE uses theta=10000.0 (NOT text's 1e6)
        if not hasattr(self, "rope_theta") or self.rope_theta is None:
            self.rope_theta = 10000.0

        # Vision encoder doesn't need these but InferenceConfig may require them
        if not hasattr(self, "max_position_embeddings") or self.max_position_embeddings is None:
            self.max_position_embeddings = 4096
        if not hasattr(self, "vocab_size"):
            self.vocab_size = 1  # unused
        if not hasattr(self, "pad_token_id"):
            self.pad_token_id = 0  # unused
        if not hasattr(self, "rms_norm_eps"):
            self.rms_norm_eps = 1e-6

    def get_required_attributes(self) -> List[str]:
        return [
            "embed_dim",
            "depth",
            "num_heads",
            "mlp_ratio",
            "patch_size",
            "temporal_patch_size",
            "spatial_merge_size",
            "in_chans",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfig


class NeuronQwen2VLVisionAttention(NeuronAttentionBase):
    """
    Qwen2-VL vision attention for NxDI using NeuronAttentionBase.

    This implementation:
      - Uses fused QKV projection via GroupQueryAttention_QKV (no manual linears).
      - Uses a standard output projection via GroupQueryAttention_O.
      - Applies 2D RoPE using externally provided `cos` and `sin` tensors.
      - Implements multi-head self-attention (MHA), not GQA.

    Args:
        config: InferenceConfig carrying at least:
            - hidden_size
            - num_attention_heads
            - num_key_value_heads (must equal num_attention_heads for this block)
            - head_dim (optional; defaults to hidden_size // num_attention_heads)
    """

    def __init__(self, config: InferenceConfig):
        if config is None or getattr(config, "neuron_config", None) is None:
            raise ValueError("NeuronQwen2VLVisionAttention requires InferenceConfig with neuron_config.")

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=None,  # RoPE is supplied via external cos/sin tensors
            rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            qkv_bias=True,  # VisionAttention uses bias on QKV
            o_bias=True,  # VisionAttention uses bias on proj
            num_cores_per_group=getattr(config, "num_cores_per_group", 1),
            sliding_window=None,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *_,
        **__,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:  [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, seq_len]  (1=attend, 0=mask)
            cos:            [seq_len, head_dim] 2D vision RoPE cosine values
            sin:            [seq_len, head_dim] 2D vision RoPE sine values

        Returns:
            Tensor of shape [batch, seq_len, hidden_size]
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                f"Expected hidden_states of shape [batch, seq_len, hidden_size], got {hidden_states.shape}"
            )

        batch_size, seq_len, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"hidden_states hidden_size ({hidden_size}) does not match config.hidden_size ({self.hidden_size})"
            )

        if cos.shape != (seq_len, self.head_dim) or sin.shape != (seq_len, self.head_dim):
            raise ValueError(
                f"cos/sin must have shape [seq_len, head_dim]=[{seq_len}, {self.head_dim}], "
                f"got cos={tuple(cos.shape)}, sin={tuple(sin.shape)}"
            )

        if attention_mask is not None and attention_mask.shape[-2:] != (seq_len, seq_len):
            raise ValueError(
                f"attention_mask must have trailing shape [seq_len, seq_len]=[{seq_len}, {seq_len}], "
                f"got {tuple(attention_mask.shape)}"
            )

        device = hidden_states.device
        input_dtype = hidden_states.dtype

        # Convert to Neuron dtype and prepare Q, K, V without built-in RoPE.
        x = hidden_states.to(self.torch_dtype)

        # Q, K, V: [B, num_heads or num_kv_heads, S, head_dim]
        Q, K, V, cos_cache, sin_cache, residual = self.prep_qkv_tensors(
            position_ids=None,
            hidden_states=x,
            past_key_value=None,
            adapter_ids=None,
            cos_cache=None,
            sin_cache=None,
            rmsnorm=None,
            skip_rope=True,
            residual=None,
            use_polar_compatible_rope=False,
        )

        # Apply 2D RoPE using externally provided cos/sin.
        cos_2d = cos.to(dtype=self.torch_dtype, device=device)
        sin_2d = sin.to(dtype=self.torch_dtype, device=device)
        Q, K = apply_rotary_pos_emb_vision(Q, K, cos_2d, sin_2d)

        # NxDI attention mask convention: boolean keep-mask.
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.to(torch.bool)

        # Compute scaled attention scores with NeuronAttentionBase helper.
        attn_scores = self.scaled_qk(Q, K, attn_mask)  # [B, num_heads, S, S]
        attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(Q.dtype)

        # Attention output: [B, num_heads, S, head_dim]
        attn_output = torch.matmul(attn_weights, V)

        # Merge heads and project back to hidden_size using the Neuron out-projection.
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )  # [B, S, H]

        attn_output = self.get_o_proj()(attn_output, adapter_ids=None)  # [B, S, H]

        # Cast back to the input dtype.
        return attn_output.to(input_dtype)

# ---------------------------------------------------------------------------
# Vision MLP
# ---------------------------------------------------------------------------

class NeuronVisionMlp(nn.Module):
    """
    Vision MLP: fc1 → act → fc2 (simple 2-layer, NOT SwiGLU).
    Both layers have bias=True. Activation is quick_gelu.

    Layer names are fc1/fc2 to match HF checkpoint keys after bulk rename
    (visual.blocks.{i}.mlp.fc1.* → blocks.{i}.mlp.fc1.*).
    """

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__()
        embed_dim = config.hidden_size
        mlp_hidden_dim = config.mlp_hidden_dim
        hidden_act = getattr(config, "hidden_act", "quick_gelu")
        self.act = ACT2FN[hidden_act]

        if parallel_state.model_parallel_is_initialized():
            tp_group = get_tp_group(config)
            dtype = config.neuron_config.torch_dtype
            self.fc1 = ColumnParallelLinear(
                embed_dim, mlp_hidden_dim, bias=True,
                gather_output=False, dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
            self.fc2 = RowParallelLinear(
                mlp_hidden_dim, embed_dim, bias=True,
                input_is_parallel=True, dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
        else:
            self.fc1 = nn.Linear(embed_dim, mlp_hidden_dim, bias=True)
            self.fc2 = nn.Linear(mlp_hidden_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------------------
# Vision Transformer Block
# ---------------------------------------------------------------------------

class NeuronQwen2VLVisionBlock(nn.Module):
    """
    Single vision transformer block: pre-norm attention + pre-norm MLP.

    Forward:
      residual = x
      x = norm1(x)
      x = residual + attn(x, mask, pos_ids)[0]
      residual = x
      x = norm2(x)
      x = residual + mlp(x)
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.norm1 = _layer_norm(config.hidden_size, eps=1e-6)
        self.norm2 = _layer_norm(config.hidden_size, eps=1e-6)
        self.attn = NeuronQwen2VLVisionAttention(config)
        self.mlp = NeuronVisionMlp(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # Attention sub-layer
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cos=cos,
            sin=sin,
        )
        hidden_states = residual + hidden_states

        # MLP sub-layer
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Patch Embedding (device-side linear, not Conv3d)
# ---------------------------------------------------------------------------

class NeuronQwen2VLPatchEmbed(nn.Module):
    """
    Patch embedding implemented as a linear projection.

    The HF model uses Conv3d(in_channels, embed_dim, kernel_size=[T,P,P], stride=[T,P,P]).
    For NxDI, the conv is done on CPU (unfold) and the linear projection runs on device.

    Weight shape: [embed_dim, in_channels * temporal_patch_size * patch_size * patch_size]
                = [1280, 3*2*14*14] = [1280, 1176]

    Forward: flattened patch pixels [batch, seq_len, 1176] → [batch, seq_len, 1280]
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        in_chans = getattr(config, "in_chans", 3)
        patch_size = config.patch_size
        temporal_patch_size = getattr(config, "temporal_patch_size", 2)
        embed_dim = config.hidden_size
        input_dim = in_chans * temporal_patch_size * patch_size * patch_size  # 1176

        if parallel_state.model_parallel_is_initialized():
            tp_group = get_tp_group(config)
            dtype = config.neuron_config.torch_dtype
            self.proj = ColumnParallelLinear(
                input_dim, embed_dim, bias=False,
                gather_output=True, dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
        else:
            self.proj = nn.Linear(input_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, num_patches, input_dim] → [batch, num_patches, embed_dim]"""
        return self.proj(x)


# ---------------------------------------------------------------------------
# Patch Merger (vision → text projection)
# ---------------------------------------------------------------------------

class NeuronQwen2VLPatchMerger(nn.Module):
    """
    Merges spatial patches and projects from vision embed_dim to text hidden_size.

    Architecture:
      ln_q:     LayerNorm(embed_dim)
      mlp_fc1:  Linear(embed_dim * spatial_merge_size^2, embed_dim * spatial_merge_size^2)
      act:      GELU
      mlp_fc2:  Linear(embed_dim * spatial_merge_size^2, text_hidden_size)

    The spatial merge groups spatial_merge_size^2 adjacent patches,
    so the input to mlp_fc1 is flattened from [N, embed_dim] to [N/4, embed_dim*4].
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        embed_dim = config.hidden_size
        spatial_merge_size = getattr(config, "spatial_merge_size", 2)
        # text_hidden_size is stored as config.hidden_size in the parent config,
        # but for the vision config we store it separately
        text_hidden_size = getattr(config, "text_hidden_size", 3584)

        self.hidden_size = embed_dim * (spatial_merge_size ** 2)
        self.spatial_merge_size = spatial_merge_size

        self.ln_q = _layer_norm(embed_dim, eps=1e-6)

        if parallel_state.model_parallel_is_initialized():
            tp_group = get_tp_group(config)
            dtype = config.neuron_config.torch_dtype
            self.mlp_fc1 = ColumnParallelLinear(
                self.hidden_size, self.hidden_size, bias=True,
                gather_output=True, dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
            self.mlp_fc2 = ColumnParallelLinear(
                self.hidden_size, text_hidden_size, bias=True,
                gather_output=True, dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
        else:
            self.mlp_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
            self.mlp_fc2 = nn.Linear(self.hidden_size, text_hidden_size, bias=True)

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, num_patches, embed_dim]
        Returns: [batch, num_patches / spatial_merge_size^2, text_hidden_size]
        """
        x = self.ln_q(x)
        # Merge spatial patches: reshape [B, N, D] → [B, N/merge^2, D*merge^2]
        x = x.view(x.shape[0], -1, self.hidden_size)
        x = self.mlp_fc2(self.act(self.mlp_fc1(x)))
        return x


# ---------------------------------------------------------------------------
# Full Vision Encoder
# ---------------------------------------------------------------------------

class NeuronQwen2VLVisionModel(nn.Module):
    """
    Full Qwen2-VL vision encoder for NxDI.

    Combines:
      - patch_embed:  NeuronQwen2VLPatchEmbed (linear projection)
      - blocks:       nn.ModuleList of NeuronQwen2VLVisionBlock (32 layers)
      - merger:       NeuronQwen2VLPatchMerger (project to text dim)

    State dict key structure (must match convert_hf_to_neuron_state_dict output):
      patch_embed.proj.weight
      blocks.{i}.norm1.weight / .norm2.weight
      blocks.{i}.attn.qkv_proj.{q,k,v}_proj.{weight,bias}
      blocks.{i}.attn.o_proj.o_proj.{weight,bias}
      blocks.{i}.mlp.fc1.{weight,bias} / .mlp.fc2.{weight,bias}
      merger.ln_q.{weight,bias}
      merger.mlp_fc1.{weight,bias}
      merger.mlp_fc2.{weight,bias}
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        vision_config = getattr(config, "vision_config", config)

        self.patch_embed = NeuronQwen2VLPatchEmbed(vision_config)
        self.blocks = nn.ModuleList(
            [NeuronQwen2VLVisionBlock(vision_config)
             for _ in range(vision_config.num_hidden_layers)]
        )
        self.merger = NeuronQwen2VLPatchMerger(vision_config)

    def forward(
        self,
        patch_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            patch_embeds:    [batch, seq_len, input_dim]  (flattened patch pixels)
            attention_mask:  [batch, 1, seq_len, seq_len] (bool keep-mask)
            cos:             [seq_len, head_dim] — pre-computed rotary cosines
            sin:             [seq_len, head_dim] — pre-computed rotary sines
        Returns:
            [batch, merged_seq_len, text_hidden_size]
        """
        # Patch embedding: linear projection
        hidden_states = self.patch_embed(patch_embeds)

        # Vision transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask, cos=cos, sin=sin)

        # Merge patches and project to text dimension
        hidden_states = self.merger(hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# Vision Model Wrapper (CPU patchification + bucketing)
# ---------------------------------------------------------------------------

class Qwen2VLVisionModelWrapper(ModelWrapper):
    """
    Wraps NeuronQwen2VLVisionModel with CPU-side patchification and bucketing.

    Handles:
      1. Conv3d-style patch extraction on CPU (unfold)
      2. Position ID generation for vision RoPE
      3. Block attention mask generation
      4. Padding to bucket boundaries
      5. Routing to compiled model
      6. Unpadding output
    """

    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        pipeline_execution: bool = True,
        return_ranked_to_cpu: bool = True,
        model_init_kwargs={},
    ) -> None:
        super().__init__(
            config, model_cls, tag, compiler_args, priority_model_idx,
            pipeline_execution, return_ranked_to_cpu, model_init_kwargs,
        )

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        """Generate sample inputs for each vision bucket."""
        vision_config = self.config.vision_config
        inputs = []
        batch_size = vision_config.neuron_config.batch_size
        in_chans = getattr(vision_config, "in_chans", 3)
        patch_size = vision_config.patch_size
        temporal_patch_size = getattr(vision_config, "temporal_patch_size", 2)
        input_dim = in_chans * temporal_patch_size * patch_size * patch_size

        embed_dim = getattr(vision_config, "embed_dim", vision_config.hidden_size)
        num_heads = getattr(vision_config, "num_heads", vision_config.num_attention_heads)
        head_dim = embed_dim // num_heads
        rope_dim = head_dim // 2
        rope_theta = getattr(vision_config, "rope_theta", 10000.0)

        for bucket in vision_config.neuron_config.buckets:
            patch_embeds = torch.ones(
                [batch_size, bucket, input_dim],
                dtype=vision_config.neuron_config.torch_dtype,
            )
            attention_mask = torch.ones(
                [batch_size, 1, bucket, bucket],
                dtype=torch.int32,
            )
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float) / rope_dim))
            positions = torch.arange(bucket, dtype=torch.float)
            freqs = torch.outer(positions, inv_freq)  # [bucket, rope_dim/2=20]
            rotary = torch.cat((freqs, freqs), dim=-1)  # [bucket, 40]
            emb = torch.cat((rotary, rotary), dim=-1)  # [bucket, 80=head_dim]
            cos = emb.cos().to(vision_config.neuron_config.torch_dtype)
            sin = emb.sin().to(vision_config.neuron_config.torch_dtype)
            inputs.append((patch_embeds, attention_mask, cos, sin))

        return inputs

    def get_model_instance(self):
        return EncoderModelInstance(model_cls=self.model_cls, config=self.config)

    def patchify(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        """
        Extract patches from pixel_values using Conv3d-style unfold on CPU,
        and pre-compute 2D rotary position embeddings (cos/sin).

        Args:
            pixel_values: [total_pixels, channel] — flattened pixel tensor from processor
            grid_thw:     [num_images, 3] — (temporal, height_patches, width_patches) per image

        Returns:
            patch_embeds:    [1, total_patches, input_dim]
            attention_mask:  [1, 1, total_patches, total_patches]  (block-diagonal)
            cos:             [total_patches, head_dim]
            sin:             [total_patches, head_dim]
        """
        vision_config = self.config.vision_config
        in_chans = getattr(vision_config, "in_chans", 3)
        patch_size = vision_config.patch_size
        temporal_patch_size = getattr(vision_config, "temporal_patch_size", 2)
        spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)

        input_dim = in_chans * temporal_patch_size * patch_size * patch_size
        if pixel_values.dim() == 2 and pixel_values.shape[-1] == in_chans:
            patches_list = []
            pixel_offset = 0
            for t, h, w in grid_thw:
                t, h, w = t.item(), h.item(), w.item()
                num_pixels = t * h * w * temporal_patch_size * patch_size * patch_size * in_chans // in_chans
                pass

        if pixel_values.dim() == 2 and pixel_values.shape[-1] == input_dim:
            patch_embeds = pixel_values
        else:
            patch_embeds = pixel_values.reshape(-1, input_dim)

        patch_embeds = patch_embeds.unsqueeze(0)  # [1, total_patches, input_dim]
        total_patches = patch_embeds.shape[1]

        # Generate block-diagonal attention mask (each image attends only to itself)
        attention_mask = self._generate_block_attention_mask(grid_thw, total_patches)

        # --- 2D rotary position embeddings (matches reference rot_pos_emb) ---
        pos_ids = []
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // spatial_merge_size, spatial_merge_size,
                w // spatial_merge_size, spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // spatial_merge_size, spatial_merge_size,
                w // spatial_merge_size, spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)  # [total_patches, 2]

        # Frequency table: VisionRotaryEmbedding with dim = head_dim // 2
        embed_dim = getattr(vision_config, "embed_dim", vision_config.hidden_size)
        num_heads = getattr(vision_config, "num_heads", vision_config.num_attention_heads)
        head_dim = embed_dim // num_heads  # 80
        rope_dim = head_dim // 2  # 40
        rope_theta = getattr(vision_config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float) / rope_dim))

        max_grid_size = grid_thw[:, 1:].max().item()
        seq = torch.arange(max_grid_size, dtype=torch.float)
        freqs = torch.outer(seq, inv_freq)  # [max_grid_size, rope_dim/2=20]

        # Index into frequency table: pos_ids is [total_patches, 2], freqs is [max_grid_size, 20]
        # freqs[pos_ids] → [total_patches, 2, 20], flatten → [total_patches, 40]
        rotary_pos_emb = freqs[pos_ids].flatten(1)  # [total_patches, 40]
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)  # [total_patches, 80]
        cos = emb.cos()  # [total_patches, head_dim]
        sin = emb.sin()  # [total_patches, head_dim]

        dtype = vision_config.neuron_config.torch_dtype

        return (
            patch_embeds.to(dtype),
            attention_mask.to(torch.int32),
            cos.to(dtype),
            sin.to(dtype),
        )

    def _generate_block_attention_mask(
        self, grid_thw: torch.Tensor, total_patches: int
    ) -> torch.Tensor:
        """
        Generate block-diagonal attention mask where patches from the same
        image attend to each other but not to patches from other images.

        Returns: [1, 1, total_patches, total_patches]
        """
        mask = torch.zeros(total_patches, total_patches, dtype=torch.int32)
        offset = 0
        for t, h, w in grid_thw:
            num_patches = t.item() * h.item() * w.item()
            mask[offset:offset + num_patches, offset:offset + num_patches] = 1
            offset += num_patches
        return mask.unsqueeze(0).unsqueeze(0)

    def pad_inputs(self, patch_embeds, attention_mask, cos, sin):
        """Pad inputs to the nearest bucket boundary."""
        target_len = self.get_target_bucket(patch_embeds)

        # Pad patch_embeds
        target_size = [patch_embeds.shape[0], target_len, patch_embeds.shape[2]]
        padded_embeds, self.original_slices = pad_tensor(patch_embeds, target_size)

        # Update slices for output dim (text_hidden_size after merger, not vision embed_dim)
        text_hidden_size = getattr(self.config, "text_config", self.config)
        if hasattr(text_hidden_size, "hidden_size"):
            text_hidden_size = text_hidden_size.hidden_size
        else:
            text_hidden_size = getattr(self.config.vision_config, "text_hidden_size", 3584)
        spatial_merge_size = getattr(self.config.vision_config, "spatial_merge_size", 2)
        merge_factor = spatial_merge_size ** 2
        self.original_slices[-2][-1] = self.original_slices[-2][-1] // merge_factor
        self.original_slices[-1][-1] = text_hidden_size

        # Pad attention mask (0 = mask padding positions)
        target_mask_size = [
            attention_mask.shape[0], attention_mask.shape[1],
            target_len, target_len,
        ]
        padded_mask, _ = pad_tensor(attention_mask, target_mask_size, pad_value=0)

        # Pad cos/sin along sequence dimension (pad with 0; masked positions are ignored)
        target_rope_size = [target_len, cos.shape[-1]]
        padded_cos, _ = pad_tensor(cos, target_rope_size, pad_value=0.0)
        padded_sin, _ = pad_tensor(sin, target_rope_size, pad_value=0.0)

        return padded_embeds, padded_mask, padded_cos, padded_sin

    def get_target_bucket(self, patch_embeds: torch.Tensor) -> int:
        """Find the smallest bucket that fits the patch sequence."""
        seq_len = patch_embeds.shape[1]
        for bucket in self.config.vision_config.neuron_config.buckets:
            if seq_len <= bucket:
                return bucket
        raise RuntimeError(
            f"Patch sequence length {seq_len} exceeds largest bucket "
            f"({self.config.vision_config.neuron_config.buckets[-1]})"
        )

    def forward(self, pixel_values, grid_thw):
        """
        Full vision encoder forward: patchify → pad → model → unpad.

        Args:
            pixel_values: Raw or pre-processed pixel tensor
            grid_thw:     [num_images, 3] — (T, H, W) patch grid per image
        Returns:
            vision_embeddings: [1, merged_patches, text_hidden_size]
        """
        if self.model is None:
            raise RuntimeError("Forward called before load.")

        patch_embeds, attention_mask, cos, sin = self.patchify(pixel_values, grid_thw)
        padded_embeds, padded_mask, padded_cos, padded_sin = self.pad_inputs(
            patch_embeds, attention_mask, cos, sin
        )

        print(f"[VIS DEBUG] padded_embeds: shape={padded_embeds.shape}, dtype={padded_embeds.dtype}, "
              f"mean={padded_embeds.float().mean():.6f}, std={padded_embeds.float().std():.6f}", flush=True)
        print(f"[VIS DEBUG] padded_mask: shape={padded_mask.shape}, dtype={padded_mask.dtype}, "
              f"sum={padded_mask.sum()}", flush=True)
        print(f"[VIS DEBUG] padded_cos: shape={padded_cos.shape}, dtype={padded_cos.dtype}, "
              f"range=[{padded_cos.float().min():.4f},{padded_cos.float().max():.4f}]", flush=True)
        print(f"[VIS DEBUG] padded_sin: shape={padded_sin.shape}, dtype={padded_sin.dtype}, "
              f"range=[{padded_sin.float().min():.4f},{padded_sin.float().max():.4f}]", flush=True)
        print(f"[VIS DEBUG] original_slices={self.original_slices}", flush=True)

        import os
        if os.environ.get("DISABLE_VISION_ROPE", "0") == "1":
            print("[VIS DEBUG] *** DISABLING ROPE: cos=1, sin=0 ***", flush=True)
            padded_cos = torch.ones_like(padded_cos)
            padded_sin = torch.zeros_like(padded_sin)
        if os.environ.get("USE_FULL_MASK", "0") == "1":
            print("[VIS DEBUG] *** USING ALL-ONES ATTENTION MASK ***", flush=True)
            padded_mask = torch.ones_like(padded_mask)

        if not self.neuron_config.on_cpu:
            args = self.convert_int64_to_int32(padded_embeds, padded_mask, padded_cos, padded_sin)
        else:
            args = (padded_embeds, padded_mask, padded_cos, padded_sin)

        print(f"[VIS DEBUG] args to _forward: {[f'shape={a.shape}, dtype={a.dtype}' for a in args]}", flush=True)

        vision_emb = self._forward(*args)
        print(f"[VIS DEBUG] raw vision_emb: shape={vision_emb.shape}, dtype={vision_emb.dtype}, "
              f"mean={vision_emb.float().mean():.6f}, std={vision_emb.float().std():.6f}", flush=True)

        vision_emb = unpad_tensor(vision_emb, self.original_slices)
        print(f"[VIS DEBUG] unpadded vision_emb: shape={vision_emb.shape}, "
              f"mean={vision_emb.float().mean():.6f}, std={vision_emb.float().std():.6f}", flush=True)

        return vision_emb


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "Qwen2VLVisionInferenceConfig",
    "NeuronQwen2VLVisionModel",
    "Qwen2VLVisionModelWrapper",
]
