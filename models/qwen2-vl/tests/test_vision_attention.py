"""
Unit test: NeuronQwen2VLVisionAttention vs the actual VisionAttention from qwen2_vl_pytorch.py.

This test validates that the NxDI translation produces outputs numerically consistent
with the real Qwen2-VL VisionAttention implementation, using the exact same 2D positional
embedding logic as the real vision encoder.

Key design:
  - PyTorch reference imports VisionAttention DIRECTLY from qwen2_vl_pytorch.py.
    A thin wrapper adapts the interface (provides a minimal Qwen2VLVisionConfig-like
    config object and bundles cu_seqlens + position_embeddings computation).
  - 2D grid positional embeddings: VisionRotaryEmbedding(head_dim // 2) indexed by
    (h, w) grid positions, exactly as in Qwen2VisionTransformerPretrainedModel.rot_pos_emb().
  - Weight mapping: HF fused qkv.weight [3H, H] → split into q/k/v_proj.weight [H, H]
    via custom sync function. proj.weight [H, H] → o_proj.weight [H, H] (no bias).

HF VisionAttention forward logic (from qwen2_vl_pytorch.py):
  Input: hidden_states [seq_len, embed_dim]  (packed, no batch dim)
  1. qkv = self.qkv(hidden_states)                # [S, 3*H]
  2. q,k,v = reshape(S,3,nH,hD).permute(1,0,2,3).unbind(0)  # each [S, nH, hD]
  3. cos, sin = position_embeddings               # each [S, hD]
  4. apply_rotary_pos_emb_vision(q, k, cos, sin)  # cos.unsqueeze(-2) → [S,1,hD]
  5. q/k/v: transpose(0,1).unsqueeze(0)           → [1, nH, S, hD]
  6. chunked SDPA per cu_seqlens                  → [1, nH, S, hD]
  7. reshape → [S, H], proj → [S, H]

2D RoPE computation (from rot_pos_emb in the vision encoder):
  - VisionRotaryEmbedding(head_dim // 2) → freqs [max_grid, head_dim // 4]
  - For a grid of shape (H_grid, W_grid):
      hpos_ids: [H_grid*W_grid] (height indices, spatially merged)
      wpos_ids: [H_grid*W_grid] (width indices, spatially merged)
      pos_ids:  [S, 2]  (stacked h and w ids)
  - rotary_pos_emb = freqs[pos_ids].flatten(1)   → [S, head_dim // 2]
  - emb = cat(rotary_pos_emb, rotary_pos_emb)    → [S, head_dim]
  - cos, sin = emb.cos(), emb.sin()              → each [S, head_dim]
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
from vision_attention_block import NeuronQwen2VLVisionAttention

# ---------------------------------------------------------------------------
# Import VisionAttention and VisionRotaryEmbedding from the installed HF
# transformers package. qwen2_vl_pytorch.py uses relative imports (from ...)
# and cannot be imported as a standalone module — the installed package IS
# the original source (same code, same file on disk at:
#   /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/
#    site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py
# ---------------------------------------------------------------------------
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    VisionAttention,
    VisionRotaryEmbedding,
)


# ---------------------------------------------------------------------------
# Test dimensions
#
# We use a 2D grid of H_grid x W_grid patches = seq_len total tokens.
# spatial_merge_size=2 (the default in the real model).
# head_dim // 2 is the dim passed to VisionRotaryEmbedding (matches real model).
# ---------------------------------------------------------------------------
H_GRID = 4          # height in patches (must be divisible by spatial_merge_size=2)
W_GRID = 4          # width in patches  (must be divisible by spatial_merge_size=2)
SPATIAL_MERGE_SIZE = 2
sl = H_GRID * W_GRID  # 16 tokens per image
hs = 64               # embed_dim
num_attention_heads = 4
head_dim = hs // num_attention_heads  # 16
rope_theta = 10000.0
dtype = torch.bfloat16


# ---------------------------------------------------------------------------
# Minimal config object that VisionAttention expects.
# VisionAttention reads: config.embed_dim, config.num_heads
# It also reads config._attn_implementation (for attention backend selection).
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
) -> tuple:
    """
    Compute 2D vision RoPE embeddings for a single image of shape (h_grid, w_grid).

    Replicates rot_pos_emb() from Qwen2VisionTransformerPretrainedModel verbatim.

    Returns:
        (cos, sin) each [h_grid * w_grid, head_dim]
    """
    rotary_emb = VisionRotaryEmbedding(head_dim // 2, theta=rope_theta)
    rotary_emb = rotary_emb.to(device)

    # Build height position ids (spatially merged, same as rot_pos_emb)
    hpos_ids = torch.arange(h_grid).unsqueeze(1).expand(-1, w_grid)
    hpos_ids = hpos_ids.reshape(
        h_grid // spatial_merge_size,
        spatial_merge_size,
        w_grid // spatial_merge_size,
        spatial_merge_size,
    )
    hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

    # Build width position ids (spatially merged)
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
    H_GRID, W_GRID, SPATIAL_MERGE_SIZE, head_dim, rope_theta
)
cos_2d = cos_2d.to(dtype)
sin_2d = sin_2d.to(dtype)

# cu_seqlens for a single image: [0, sl]
cu_seqlens = torch.tensor([0, sl], dtype=torch.int32)


# ---------------------------------------------------------------------------
# PyTorch reference wrapper
#
# Wraps VisionAttention (imported directly from qwen2_vl_pytorch.py).
# Adapts the interface so test_block_correctness can call it with a single tensor.
#
# test_block_correctness calls: reference_block(reference_inputs[0][0])
# So forward(hidden_states) must accept [seq_len, embed_dim] and return [seq_len, embed_dim].
#
# The wrapper bundles cu_seqlens and position_embeddings internally.
# ---------------------------------------------------------------------------
class WrappedVisionAttention(nn.Module):
    """
    Thin wrapper around VisionAttention (from qwen2_vl_pytorch.py).

    Adapts the call signature for test_block_correctness:
      forward(hidden_states [S, H]) → [S, H]

    Internal computation is entirely from the unmodified VisionAttention class.
    """

    def __init__(self):
        super().__init__()
        self.attn = VisionAttention(config=_VisionConfig())
        # Store embeddings as buffers so they move with the module
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
from neuronx_distributed_inference.models.config import NeuronConfig, InferenceConfig

neuron_config = NeuronConfig(
    batch_size=1,    # vision encoder has no batch dim; use 1 for config
    seq_len=sl,
    tp_degree=1,
    torch_dtype=dtype,
    on_cpu=True,
    fused_qkv=False,
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
#   proj.bias   [H]     (nn.Linear defaults to bias=True)
#
# NxDI NeuronQwen2VLVisionAttention state dict (under block. wrapper prefix):
#   block.qkv.weight  [3*H, H]   (ColumnParallelLinear)
#   block.qkv.bias    [3*H]
#   block.proj.weight [H, H]     (RowParallelLinear)
#   block.proj.bias   [H]
#
# The fused QKV weight is the same shape on both sides, so no splitting needed.
# ---------------------------------------------------------------------------
weight_mapping = {
    "qkv.weight":  "qkv.weight",
    "qkv.bias":    "qkv.bias",
    "proj.weight": "proj.weight",
    "proj.bias":   "proj.bias",
}


# ---------------------------------------------------------------------------
# Inputs
#
# NxDI block receives: (hidden_states, cu_seqlens, cos, sin)
#   cos and sin are passed as flat tensors (not a tuple) because the XLA
#   tracing harness requires all inputs to be plain torch.Tensor instances.
#
# Reference block (WrappedVisionAttention) receives: (hidden_states,)
#   It bundles cu_seqlens and position_embeddings=(cos, sin) internally.
# ---------------------------------------------------------------------------
torch.manual_seed(123)
sample = torch.rand(sl, hs, dtype=dtype)

example_inputs  = [(torch.zeros(sl, hs, dtype=dtype), cu_seqlens, cos_2d, sin_2d)]
test_inputs     = [(sample, cu_seqlens, cos_2d, sin_2d)]
reference_inputs = [(sample,)]


# ---------------------------------------------------------------------------
# Sanity check: verify 2D RoPE produces position-dependent outputs
# ---------------------------------------------------------------------------
def _sanity_check_2d_rope():
    torch.manual_seed(42)
    ref = WrappedVisionAttention()
    ref = ref.to(dtype=dtype)
    ref.eval()
    with torch.no_grad():
        x = torch.rand(sl, hs, dtype=dtype)
        out = ref(x)  # [sl, hs]
    # Different spatial positions should produce different outputs
    assert not torch.allclose(out[0], out[sl // 2], atol=1e-3), (
        "Sanity check FAILED: positions 0 and sl//2 produce identical output — 2D RoPE may not be working"
    )
    print("Sanity check PASSED: 2D RoPE produces position-dependent outputs")
    print(f"  pos[0]   mean: {out[0].float().mean().item():.4f}")
    print(f"  pos[S/2] mean: {out[sl // 2].float().mean().item():.4f}")

_sanity_check_2d_rope()


# ---------------------------------------------------------------------------
# Run correctness test
# ---------------------------------------------------------------------------
print("=" * 80)
print("Test: NeuronQwen2VLVisionAttention vs VisionAttention (from qwen2_vl_pytorch.py)")
print(f"Dimensions: sl={sl} ({H_GRID}x{W_GRID} grid), hs={hs}, nH={num_attention_heads}, hD={head_dim}")
print(f"RoPE: VisionRotaryEmbedding(dim={head_dim // 2}, theta={rope_theta}) with 2D grid pos_ids")
print(f"PyTorch reference: VisionAttention imported directly from qwen2_vl_pytorch.py")
print("=" * 80)

test_block_correctness(
    neuron_block_class=NeuronQwen2VLVisionAttention,
    pytorch_block_class=WrappedVisionAttention,
    weight_mapping=weight_mapping,
    example_inputs=example_inputs,
    test_inputs=test_inputs,
    reference_inputs=reference_inputs,
    checkpoint_name="vision_attention_2d.pt",
    seed=42,
    neuron_init_kwargs={"config": config},
    pytorch_init_kwargs={},
    verbose=True,
)
