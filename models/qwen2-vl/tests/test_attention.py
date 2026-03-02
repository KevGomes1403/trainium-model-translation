"""
Unit test: NeuronQwen2VLAttention vs PyTorch Qwen2VLAttention.

Validates that NeuronQwen2VLAttention produces identical outputs to the
reference PyTorch attention when given the same weights and inputs.

Key differences from OLMo-3 attention test:
  - qkv_bias=True: Q, K, V projections all have bias terms
  - No QK normalization
  - Standard 1D RotaryEmbedding (MRoPE → 1D for text-only tokens)
  - fused_qkv=False: NeuronAttentionBase creates separate q/k/v projections
    under qkv_proj.q_proj / k_proj / v_proj
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

MODEL_DIR = ROOT_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from neuronx_distributed_inference.models.config import NeuronConfig, InferenceConfig
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding

from block_testing_utils import test_block_correctness
from attention_block import NeuronQwen2VLAttention


# ---------------------------------------------------------------------------
# Test dimensions
#   head_dim = hidden_size // num_attention_heads  →  64 // 4 = 16  ✓
# ---------------------------------------------------------------------------
bs, sl, hs = 2, 128, 64
num_attention_heads = 4
num_key_value_heads = 2   # GQA ratio 2:1
head_dim = hs // num_attention_heads  # 16
dtype = torch.bfloat16


# ---------------------------------------------------------------------------
# Config
#   fused_qkv=False: NeuronAttentionBase creates separate q/k/v_proj weights
#     → state dict keys: qkv_proj.q_proj.weight/.bias etc.
#   on_cpu=True: avoid TKG module initialization during unit test
#   attention_bias=True: must match qkv_bias=True in NeuronQwen2VLAttention
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
    rope_scaling=None,
    max_position_embeddings=4096,
    rms_norm_eps=1e-6,
    attention_bias=True,   # tells default config helpers bias is present
    initial_context_length=4096,
)
config.num_cores_per_group = 1


# ---------------------------------------------------------------------------
# PyTorch reference (standalone, no HF dependencies)
#
# Implements Qwen2-VL attention:
#   1. Project H → Q (nH*hD) [+bias], K (nKvH*hD) [+bias], V (nKvH*hD) [+bias]
#   2. Reshape to (B, H, S, hD)
#   3. Apply RoPE (same RotaryEmbedding as Neuron block)
#   4. GQA: expand K/V to full head count
#   5. Scaled dot-product attention (no causal mask in test)
#   6. Project output O [no bias]
# ---------------------------------------------------------------------------
class PyTorchQwen2VLAttentionWrapper(nn.Module):
    """
    Standalone Qwen2-VL attention for correctness testing.

    Accepts only (hidden_states,) and computes position_ids internally
    so it can be used as the reference_block in test_block_correctness.
    Uses the same RotaryEmbedding as NeuronQwen2VLAttention for bit-identical RoPE.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim

        # QKV with bias, o_proj without bias — matches Qwen2-VL HF checkpoint
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # Same RotaryEmbedding implementation as NeuronQwen2VLAttention
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        # Projections (with bias on Q, K, V)
        q = self.q_proj(hidden_states)  # (B, S, nH*hD)
        k = self.k_proj(hidden_states)  # (B, S, nKvH*hD)
        v = self.v_proj(hidden_states)  # (B, S, nKvH*hD)

        # Reshape to per-head layout
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)     # (B, nH, S, hD)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, nKvH, S, hD)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, nKvH, S, hD)

        # RoPE — compute from sequential position_ids
        position_ids = torch.arange(S, device=hidden_states.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_emb(hidden_states, position_ids)  # (B, S, hD) each
        cos = cos.unsqueeze(1)  # (B, 1, S, hD)
        sin = sin.unsqueeze(1)  # (B, 1, S, hD)

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        # GQA: expand K and V to full head count
        if self.num_groups > 1:
            k = k[:, :, None, :, :].expand(-1, -1, self.num_groups, -1, -1).reshape(
                B, self.num_heads, S, self.head_dim
            )
            v = v[:, :, None, :, :].expand(-1, -1, self.num_groups, -1, -1).reshape(
                B, self.num_heads, S, self.head_dim
            )

        # Full attention (no causal mask — matches all-zeros mask passed to Neuron block)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
        # (B, nH, S, hD) → (B, S, nH*hD)
        attn = attn.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)

        return self.o_proj(attn)


# ---------------------------------------------------------------------------
# Weight mapping: PyTorch key → NxDI key (block. prefix added automatically)
#
# fused_qkv=False → NeuronAttentionBase creates:
#   qkv_proj.q_proj.weight / .bias
#   qkv_proj.k_proj.weight / .bias
#   qkv_proj.v_proj.weight / .bias
#   o_proj.o_proj.weight  (GroupQueryAttention_O, layer_name='o_proj')
# ---------------------------------------------------------------------------
weight_mapping = {
    # Q projection (weight + bias)
    "q_proj.weight": "qkv_proj.q_proj.weight",
    "q_proj.bias":   "qkv_proj.q_proj.bias",
    # K projection (weight + bias)
    "k_proj.weight": "qkv_proj.k_proj.weight",
    "k_proj.bias":   "qkv_proj.k_proj.bias",
    # V projection (weight + bias)
    "v_proj.weight": "qkv_proj.v_proj.weight",
    "v_proj.bias":   "qkv_proj.v_proj.bias",
    # Output projection (no bias)
    "o_proj.weight": "o_proj.o_proj.weight",
}


# ---------------------------------------------------------------------------
# Inputs
#
# Neuron block receives: (hidden_states, attention_mask, position_ids)
# attention_mask = zeros → full attention (no masking), matching the
#                 PyTorch wrapper's F.scaled_dot_product_attention with
#                 is_causal=False.
# ---------------------------------------------------------------------------
torch.manual_seed(123)
sample = torch.rand(bs, sl, hs, dtype=dtype)

position_ids = torch.arange(sl, dtype=torch.long).unsqueeze(0).expand(bs, -1)
# All-zero additive mask → no masking effect on attention scores
attention_mask = torch.zeros(bs, 1, sl, sl, dtype=dtype)

example_inputs = [(torch.zeros(bs, sl, hs, dtype=dtype), attention_mask, position_ids)]
test_inputs = [(sample, attention_mask, position_ids)]
reference_inputs = [(sample,)]  # wrapper handles mask/pos internally


# ---------------------------------------------------------------------------
# Run test
# ---------------------------------------------------------------------------
test_block_correctness(
    neuron_block_class=NeuronQwen2VLAttention,
    pytorch_block_class=PyTorchQwen2VLAttentionWrapper,
    weight_mapping=weight_mapping,
    example_inputs=example_inputs,
    test_inputs=test_inputs,
    reference_inputs=reference_inputs,
    checkpoint_name="attention.pt",
    seed=42,
    neuron_init_kwargs={"config": config, "layer_idx": 0},
    pytorch_init_kwargs={"config": config},
    verbose=True,
)
