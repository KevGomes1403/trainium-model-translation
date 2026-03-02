"""
Unit test: NeuronArceeAttention vs PyTorch ArceeAttention.

Arcee attention is GQA with:
  - num_attention_heads=8, num_key_value_heads=2, head_dim=8 for tests
  - fused_qkv=False (separate q/k/v projections)
  - No QK norms, no attention bias
  - YaRN RoPE (disabled for test via rope_scaling=None)

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
from modeling_arcee_neuron import NeuronArceeAttention


# ---------------------------------------------------------------------------
# Test dimensions
#   head_dim = hidden_size // num_attention_heads → 64 // 8 = 8
# ---------------------------------------------------------------------------
bs, sl, hs = 2, 128, 64
num_attention_heads = 8
num_key_value_heads = 2
head_dim = hs // num_attention_heads  # 8
dtype = torch.bfloat16


# ---------------------------------------------------------------------------
# Config
# fused_qkv=False is required because Arcee has separate Q/K/V projections.
# rope_scaling=None: disable YaRN for simple unit test.
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
    sliding_window=None,
    rope_theta=10000.0,
    rope_scaling=None,          # no YaRN scaling for unit test simplicity
    max_position_embeddings=4096,
    attention_bias=False,
    initial_context_length=4096,
)
config.rms_norm_eps = 1e-5
config.num_cores_per_group = 1


# ---------------------------------------------------------------------------
# PyTorch reference (standalone, no HF dependencies beyond LlamaRotaryEmbedding)
#
# Matches NeuronArceeAttention numerics:
#   1. Project H → Q (nH*hD), K (nKvH*hD), V (nKvH*hD)
#   2. Reshape to per-head layout
#   3. Apply RoPE (same LlamaRotaryEmbedding as Neuron block)
#   4. GQA: expand K/V for all Q heads
#   5. Full (unmasked) attention via scaled_dot_product_attention(is_causal=False)
#   6. Project output
# ---------------------------------------------------------------------------
class PyTorchArceeAttentionWrapper(nn.Module):
    """
    Standalone Arcee attention for correctness testing.
    Accepts only (hidden_states,) and computes position_ids internally.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # Same RoPE as NeuronArceeAttention for bit-identical results
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

        # Full unmasked attention (matches ones attention_mask passed to Neuron block)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
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
# ---------------------------------------------------------------------------
test_block_correctness(
    neuron_block_class=NeuronArceeAttention,
    pytorch_block_class=PyTorchArceeAttentionWrapper,
    weight_mapping=weight_mapping,
    example_inputs=example_inputs,
    test_inputs=test_inputs,
    reference_inputs=reference_inputs,
    checkpoint_name="arcee_attention.pt",
    seed=42,
    neuron_init_kwargs={"config": config, "layer_idx": 0},
    pytorch_init_kwargs={"config": config},
    verbose=True,
)
