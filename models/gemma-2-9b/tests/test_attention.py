"""
Unit test: NeuronGemma2Attention vs PyTorch Gemma2Attention.

Gemma-2 attention is GQA with:
  - num_attention_heads=16, num_key_value_heads=8, head_dim=256 (actual Gemma-2)
  - For unit test: hidden_size=64, so head_dim=8, to fit in memory
  - Scaling: query_pre_attn_scalar**-0.5 = 256**-0.5 (NOT 1/sqrt(head_dim))
  - Attn logit softcapping: 50.0 (tanh(scores/50)*50 before softmax)
    - Applied in scaled_qk() override (PyTorch eager path)
  - Sliding window: configurable per layer_idx based on config.layer_types
    - "sliding_attention" → use sliding_window value
    - "full_attention" → no window

NxDI attention mask convention:
  NeuronAttentionBase.scaled_qk() uses a boolean KEEP-mask:
    1 (True)  → attend (keep QK score)
    0 (False) → mask   (set to finfo.min → uniform attention)
  HuggingFace uses additive convention (opposite).

  For full unmasked attention in tests: use torch.ones(bs,1,sl,sl,bfloat16).
  Do NOT use all-zeros — that masks all positions.

Weight key mapping (fused_qkv=False, on_cpu=True, tp_degree=1):
  q_proj.weight → qkv_proj.q_proj.weight
  k_proj.weight → qkv_proj.k_proj.weight
  v_proj.weight → qkv_proj.v_proj.weight
  o_proj.weight → o_proj.o_proj.weight
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from neuronx_distributed_inference.models.config import NeuronConfig, InferenceConfig

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

MODEL_DIR = ROOT_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from block_testing_utils import test_block_correctness
from nxdi_attention import NeuronGemma2Attention


# ---------------------------------------------------------------------------
# Test dimensions
#   For memory efficiency, use smaller head_dim than real Gemma-2
#   Real Gemma-2: head_dim=256, num_attention_heads=16, num_key_value_heads=8
#   Test: hidden_size=64, head_dim=8, num_attention_heads=8, num_key_value_heads=2
# ---------------------------------------------------------------------------
bs, sl, hs = 2, 128, 64
num_attention_heads = 8
num_key_value_heads = 2
head_dim = hs // num_attention_heads  # 8
dtype = torch.bfloat16

# Gemma-2 specific config
# Real Gemma-2 uses query_pre_attn_scalar=256, but for test we'll use head_dim**2
# to match the scaling mathematically (since we scaled down head_dim)
query_pre_attn_scalar = head_dim ** 2  # 64, so 64**-0.5 ≈ 0.125


# ---------------------------------------------------------------------------
# Config
# fused_qkv=False is required for Gemma-2's separate Q/K/V projections.
# layer_types: alternating pattern for even/odd layers
# ---------------------------------------------------------------------------
neuron_config = NeuronConfig(
    batch_size=bs,
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
    num_key_value_heads=num_key_value_heads,
    sliding_window=32,  # small window for testing (must be <= seq_len)
    rope_theta=10000.0,
    max_position_embeddings=4096,
    attention_bias=False,
    initial_context_length=4096,
)
# Inject Gemma-2 specific attributes
config.rms_norm_eps = 1e-6
config.num_cores_per_group = 1
config.query_pre_attn_scalar = query_pre_attn_scalar
config.attn_logit_softcapping = 50.0
# layer_types: alternating pattern; test with layer_idx=0 (sliding_attention)
#              and layer_idx=1 (full_attention)
config.layer_types = ["sliding_attention", "full_attention"] * 21  # 42 layers


# ---------------------------------------------------------------------------
# PyTorch reference (standalone, no HF dependencies beyond LlamaRotaryEmbedding)
#
# Matches NeuronGemma2Attention numerics:
#   1. Project H → Q (nH*hD), K (nKvH*hD), V (nKvH*hD)
#   2. Reshape to per-head layout
#   3. Apply RoPE (same LlamaRotaryEmbedding as Neuron block)
#   4. GQA: expand K/V for all Q heads
#   5. Full (unmasked) attention via scaled_dot_product_attention(is_causal=False)
#      with custom scaling (query_pre_attn_scalar**-0.5)
#   6. Project output
#
# NOTE: Softcapping is NOT applied in the reference either, to match the NxDI behavior
#       and focus on core attention correctness. Real Gemma-2 would apply:
#       scores = tanh(scores / 50.0) * 50.0
# ---------------------------------------------------------------------------
class PyTorchGemma2AttentionWrapper(nn.Module):
    """
    Standalone Gemma-2 attention for correctness testing.
    Accepts only (hidden_states,) and computes position_ids internally.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.query_pre_attn_scalar = getattr(config, "query_pre_attn_scalar", 256)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # Same RoPE as NeuronGemma2Attention for bit-identical results
        # LlamaRotaryEmbedding expects a config object
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        # Linear projections
        q = self.q_proj(hidden_states)   # (B, S, nH*hD)
        k = self.k_proj(hidden_states)   # (B, S, nKvH*hD)
        v = self.v_proj(hidden_states)   # (B, S, nKvH*hD)

        # Reshape to per-head layout
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)      # (B,nH,S,hD)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)   # (B,nKvH,S,hD)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)   # (B,nKvH,S,hD)

        # RoPE — compute from sequential position_ids
        position_ids = (
            torch.arange(S, device=hidden_states.device, dtype=torch.long)
            .unsqueeze(0).expand(B, -1)
        )
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos = cos.unsqueeze(1)  # (B, 1, S, hD)
        sin = sin.unsqueeze(1)  # (B, 1, S, hD)

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        # GQA: replicate K/V to match Q head count
        if self.num_groups > 1:
            k = k[:, :, None, :, :].expand(-1, -1, self.num_groups, -1, -1).reshape(
                B, self.num_heads, S, self.head_dim
            )
            v = v[:, :, None, :, :].expand(-1, -1, self.num_groups, -1, -1).reshape(
                B, self.num_heads, S, self.head_dim
            )

        # Custom scaling: query_pre_attn_scalar**-0.5 (NOT head_dim**-0.5)
        scale = self.query_pre_attn_scalar ** -0.5
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scale  # (B, nH, S, S)

        # Attn logit softcapping: tanh(scores / softcap) * softcap
        softcap = getattr(config, "attn_logit_softcapping", None)
        if softcap is not None:
            attn_weights = torch.tanh(attn_weights / softcap) * softcap

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn = torch.matmul(attn_weights, v)  # (B, nH, S, hD)

        # Full unmasked attention (matches ones attention_mask passed to Neuron block)
        # attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
        # (B, nH, S, hD) → (B, S, nH*hD)
        attn = attn.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)

        return self.o_proj(attn)


# ---------------------------------------------------------------------------
# Weight mapping: PyTorch key → NxDI key (block. prefix added automatically)
#
# With fused_qkv=False, on_cpu=True, tp_degree=1:
#   NeuronAttentionBase creates:
#     qkv_proj.q_proj.weight  (ColumnParallelLinear → nn.Linear on CPU)
#     qkv_proj.k_proj.weight
#     qkv_proj.v_proj.weight
#     o_proj.o_proj.weight    (GroupQueryAttention_O, layer_name='o_proj')
# ---------------------------------------------------------------------------
weight_mapping = {
    "q_proj.weight": "qkv_proj.q_proj.weight",
    "k_proj.weight": "qkv_proj.k_proj.weight",
    "v_proj.weight": "qkv_proj.v_proj.weight",
    "o_proj.weight": "o_proj.o_proj.weight",
}


# ---------------------------------------------------------------------------
# Inputs
#
# NxDI attention mask convention (boolean keep-mask):
#   1 → attend, 0 → mask (all-zero masks ALL positions → uniform attn)
# Use all-ones for unmasked full attention, matching the PyTorch reference
# which uses is_causal=False.
# ---------------------------------------------------------------------------
torch.manual_seed(123)
sample = torch.rand(bs, sl, hs, dtype=dtype)

position_ids = torch.arange(sl, dtype=torch.long).unsqueeze(0).expand(bs, -1)
# All-ones keep-mask: attend to all positions (NxDI convention)
attention_mask = torch.ones(bs, 1, sl, sl, dtype=dtype)

example_inputs = [(torch.zeros(bs, sl, hs, dtype=dtype), attention_mask, position_ids)]
test_inputs = [(sample, attention_mask, position_ids)]
reference_inputs = [(sample,)]  # wrapper handles mask/pos internally


# ---------------------------------------------------------------------------
# Run test
# TEST 1: layer_idx=0 (sliding_attention)
# ---------------------------------------------------------------------------
print("=" * 80)
print("TEST 1: layer_idx=0 (sliding_attention)")
print("=" * 80)

test_block_correctness(
    neuron_block_class=NeuronGemma2Attention,
    pytorch_block_class=PyTorchGemma2AttentionWrapper,
    weight_mapping=weight_mapping,
    example_inputs=example_inputs,
    test_inputs=test_inputs,
    reference_inputs=reference_inputs,
    checkpoint_name="gemma2_attention_layer0.pt",
    seed=42,
    neuron_init_kwargs={"config": config, "layer_idx": 0},
    pytorch_init_kwargs={"config": config},
    verbose=True,
)

print("\n")

# ---------------------------------------------------------------------------
# TEST 2: layer_idx=1 (full_attention)
# ---------------------------------------------------------------------------
print("=" * 80)
print("TEST 2: layer_idx=1 (full_attention)")
print("=" * 80)

# For the second test, create a fresh checkpoint to avoid weight carry-over
test_block_correctness(
    neuron_block_class=NeuronGemma2Attention,
    pytorch_block_class=PyTorchGemma2AttentionWrapper,
    weight_mapping=weight_mapping,
    example_inputs=example_inputs,
    test_inputs=test_inputs,
    reference_inputs=reference_inputs,
    checkpoint_name="gemma2_attention_layer1.pt",
    seed=42,
    neuron_init_kwargs={"config": config, "layer_idx": 1},
    pytorch_init_kwargs={"config": config},
    verbose=True,
)

print("\n")
print("=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
