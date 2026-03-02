"""
Unit test: NeuronQwen2VLVisionAttentionBase vs HF VisionAttention.

This test validates that the NeuronAttentionBase-based vision attention block
(`NeuronQwen2VLVisionAttentionBase`) produces outputs numerically consistent
with the real Qwen2-VL `VisionAttention` implementation from the installed
`transformers` package, using the exact same 2D positional embedding logic
as the real vision encoder.

Key design:
  - PyTorch reference imports `VisionAttention` and `VisionRotaryEmbedding`
    DIRECTLY from `transformers.models.qwen2_vl.modeling_qwen2_vl`.
    A thin wrapper adapts the interface to `(hidden_states,)` so it can be
    used with `test_block_correctness`.
  - 2D grid positional embeddings: `VisionRotaryEmbedding(head_dim // 2)`
    indexed by (h, w) grid positions, exactly as in
    `Qwen2VisionTransformerPretrainedModel.rot_pos_emb()`.
  - NxDI block:
      * Inherits from `NeuronAttentionBase`.
      * Uses fused QKV projection via `GroupQueryAttention_QKV` (fused_qkv=True).
      * Uses MHA (num_attention_heads == num_key_value_heads).
      * Applies 2D RoPE using externally provided cos/sin tensors.
  - Weight mapping:
      * HF fused qkv.weight / qkv.bias map to
        `qkv_proj.Wqkv.weight` / `qkv_proj.Wqkv.bias`.
      * HF proj.weight / proj.bias map to
        `o_proj.o_proj.weight` / `o_proj.o_proj.bias`.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

MODEL_DIR = ROOT_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from block_testing_utils import test_block_correctness
from nxdi_vision_attention import NeuronQwen2VLVisionAttentionBase

# ---------------------------------------------------------------------------
# Import VisionAttention and VisionRotaryEmbedding from the installed HF
# transformers package (original source implementation).
# ---------------------------------------------------------------------------
from transformers.models.qwen2_vl.modeling_qwen2_vl import (  # type: ignore
    VisionAttention,
    VisionRotaryEmbedding,
)


# ---------------------------------------------------------------------------
# Test dimensions
#
# We use a 2D grid of H_GRID x W_GRID patches = seq_len total tokens.
# spatial_merge_size=2 (the default in the real model).
# head_dim // 2 is the dim passed to VisionRotaryEmbedding (matches real model).
# ---------------------------------------------------------------------------
H_GRID = 4  # height in patches (must be divisible by spatial_merge_size=2)
W_GRID = 4  # width in patches  (must be divisible by spatial_merge_size=2)
SPATIAL_MERGE_SIZE = 2
sl = H_GRID * W_GRID  # 16 tokens per image
hs = 64  # embed_dim
num_attention_heads = 4
head_dim = hs // num_attention_heads  # 16
rope_theta = 10000.0
dtype = torch.bfloat16


# ---------------------------------------------------------------------------
# Minimal config object that VisionAttention expects.
# VisionAttention reads: config.embed_dim, config.num_heads, config._attn_implementation.
# ---------------------------------------------------------------------------
class _VisionConfig:
    embed_dim = hs
    num_heads = num_attention_heads
    _attn_implementation = "eager"


# ---------------------------------------------------------------------------
# 2D positional embedding computation
#
# Replicates Qwen2VisionTransformerPretrainedModel.rot_pos_emb() exactly.
# Returns (cos, sin) each [seq_len, head_dim].
# ---------------------------------------------------------------------------
def _compute_2d_position_embeddings(
    h_grid: int,
    w_grid: int,
    spatial_merge_size: int,
    head_dim: int,
    rope_theta: float,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute 2D vision RoPE embeddings for a single image of shape (h_grid, w_grid).

    Replicates rot_pos_emb() from Qwen2VisionTransformerPretrainedModel.

    Returns:
        (cos, sin) each [h_grid * w_grid, head_dim]
    """

    rotary_emb = VisionRotaryEmbedding(head_dim // 2, theta=rope_theta).to(device)

    # Height position ids (spatially merged, same as rot_pos_emb)
    hpos_ids = torch.arange(h_grid).unsqueeze(1).expand(-1, w_grid)
    hpos_ids = hpos_ids.reshape(
        h_grid // spatial_merge_size,
        spatial_merge_size,
        w_grid // spatial_merge_size,
        spatial_merge_size,
    )
    hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

    # Width position ids (spatially merged)
    wpos_ids = torch.arange(w_grid).unsqueeze(0).expand(h_grid, -1)
    wpos_ids = wpos_ids.reshape(
        h_grid // spatial_merge_size,
        spatial_merge_size,
        w_grid // spatial_merge_size,
        spatial_merge_size,
    )
    wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()

    # pos_ids: [S, 2]  (h and w indices stacked)
    pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1)  # [S, 2]

    max_grid_size = max(h_grid, w_grid)
    rotary_pos_emb_full = rotary_emb(max_grid_size)  # [max_grid, head_dim // 4]
    rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)  # [S, head_dim // 2]

    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)  # [S, head_dim]
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin


# ---------------------------------------------------------------------------
# Pre-compute position embeddings and cu_seqlens (shared by reference and NxDI)
# ---------------------------------------------------------------------------
cos_2d, sin_2d = _compute_2d_position_embeddings(
    H_GRID,
    W_GRID,
    SPATIAL_MERGE_SIZE,
    head_dim,
    rope_theta,
)
cos_2d = cos_2d.to(dtype)
sin_2d = sin_2d.to(dtype)

# cu_seqlens for a single image: [0, sl]
cu_seqlens = torch.tensor([0, sl], dtype=torch.int32)


# ---------------------------------------------------------------------------
# PyTorch reference wrapper
#
# Wraps VisionAttention from transformers and adapts it to (hidden_states,) so
# it can be used with test_block_correctness.
# ---------------------------------------------------------------------------
class WrappedVisionAttention(nn.Module):
    """
    Thin wrapper around VisionAttention (from transformers.models.qwen2_vl).

    Adapts the call signature for test_block_correctness:
      forward(hidden_states [S, H]) → [S, H]

    Internal computation is entirely from the unmodified VisionAttention class.
    """

    def __init__(self):
        super().__init__()
        self.attn = VisionAttention(config=_VisionConfig())
        # Store embeddings and cu_seqlens as buffers so they move with the module
        self.register_buffer("_cos", cos_2d, persistent=False)
        self.register_buffer("_sin", sin_2d, persistent=False)
        self.register_buffer("_cu_seqlens", cu_seqlens, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [seq_len, embed_dim]
        Returns:
            [seq_len, embed_dim]
        """
        return self.attn(
            hidden_states,
            cu_seqlens=self._cu_seqlens,
            position_embeddings=(self._cos, self._sin),
        )

    def state_dict(self, **kwargs):
        # Expose only the inner VisionAttention weights (not the buffers)
        return self.attn.state_dict(**kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.attn.named_parameters(*args, **kwargs)


# ---------------------------------------------------------------------------
# NxDI config
# ---------------------------------------------------------------------------
from neuronx_distributed_inference.models.config import NeuronConfig, InferenceConfig  # type: ignore

neuron_config = NeuronConfig(
    batch_size=1,  # vision encoder has no batch dim; use 1 for config
    seq_len=sl,
    tp_degree=1,
    torch_dtype=dtype,
    on_cpu=True,
    fused_qkv=True,  # use fused QKV projection
)

config = InferenceConfig(
    neuron_config=neuron_config,
    hidden_size=hs,
    head_dim=head_dim,
    num_attention_heads=num_attention_heads,
    num_key_value_heads=num_attention_heads,  # MHA
    sliding_window=None,
    rope_theta=rope_theta,
    rope_scaling=None,
    max_position_embeddings=4096,
    rms_norm_eps=1e-6,
    attention_bias=True,
    initial_context_length=4096,
)
config.num_cores_per_group = 1


# ---------------------------------------------------------------------------
# Weight mapping
#
# HF VisionAttention state dict (via WrappedVisionAttention.state_dict()):
#   qkv.weight  [3*H, H]
#   qkv.bias    [3*H]
#   proj.weight [H, H]
#   proj.bias   [H]
#
# NeuronQwen2VLVisionAttentionBase (via NeuronAttentionBase / GroupQueryAttention_QKV / GroupQueryAttention_O):
#   block.qkv_proj.Wqkv.weight  [3*H, H]
#   block.qkv_proj.Wqkv.bias    [3*H]
#   block.o_proj.o_proj.weight  [H, H]
#   block.o_proj.o_proj.bias    [H]
# ---------------------------------------------------------------------------
weight_mapping = {
    "qkv.weight": "qkv_proj.Wqkv.weight",
    "qkv.bias": "qkv_proj.Wqkv.bias",
    "proj.weight": "o_proj.o_proj.weight",
    "proj.bias": "o_proj.o_proj.bias",
}


# ---------------------------------------------------------------------------
# Inputs
#
# NxDI block receives: (hidden_states, cu_seqlens, cos, sin)
# Reference block (WrappedVisionAttention) receives: (hidden_states,)
# ---------------------------------------------------------------------------
torch.manual_seed(123)
sample = torch.rand(sl, hs, dtype=dtype)

example_inputs = [(torch.zeros(sl, hs, dtype=dtype), cu_seqlens, cos_2d, sin_2d)]
test_inputs = [(sample, cu_seqlens, cos_2d, sin_2d)]
reference_inputs = [(sample,)]


# ---------------------------------------------------------------------------
# Sanity check: verify 2D RoPE produces position-dependent outputs
# ---------------------------------------------------------------------------
def _sanity_check_2d_rope():
    torch.manual_seed(42)
    ref = WrappedVisionAttention().to(dtype=dtype)
    ref.eval()
    with torch.no_grad():
        x = torch.rand(sl, hs, dtype=dtype)
        out = ref(x)  # [sl, hs]
    # Different spatial positions should produce different outputs
    assert not torch.allclose(out[0], out[sl // 2], atol=1e-3), (
        "Sanity check FAILED: positions 0 and sl//2 produce identical output — "
        "2D RoPE may not be working"
    )
    print("Sanity check PASSED: 2D RoPE produces position-dependent outputs")
    print(f"  pos[0]   mean: {out[0].float().mean().item():.4f}")
    print(f"  pos[S/2] mean: {out[sl // 2].float().mean().item():.4f}")


_sanity_check_2d_rope()


# ---------------------------------------------------------------------------
# Run correctness test
# ---------------------------------------------------------------------------
print("=" * 80)
print("Test: NeuronQwen2VLVisionAttentionBase vs HF VisionAttention")
print(f"Dimensions: sl={sl} ({H_GRID}x{W_GRID} grid), hs={hs}, nH={num_attention_heads}, hD={head_dim}")
print(f"RoPE: VisionRotaryEmbedding(dim={head_dim // 2}, theta={rope_theta}) with 2D grid pos_ids")
print("PyTorch reference: VisionAttention imported directly from transformers.models.qwen2_vl.modeling_qwen2_vl")
print("=" * 80)

test_block_correctness(
    neuron_block_class=NeuronQwen2VLVisionAttentionBase,
    pytorch_block_class=WrappedVisionAttention,
    weight_mapping=weight_mapping,
    example_inputs=example_inputs,
    test_inputs=test_inputs,
    reference_inputs=reference_inputs,
    checkpoint_name="vision_attention_neuron_base.pt",
    seed=42,
    neuron_init_kwargs={"config": config},
    pytorch_init_kwargs={},
    verbose=True,
)

