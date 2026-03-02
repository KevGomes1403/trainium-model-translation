"""
Unit test: NeuronOlmo3Attention vs PyTorch Olmo3Attention.

Validates that NeuronOlmo3Attention produces identical outputs to the
reference PyTorch attention when given the same weights and inputs.

Key differences handled:
  - NeuronOlmo3Attention: forward(hidden_states, attention_mask, position_ids, ...)
  - PyTorch reference:    forward(hidden_states)  [computes mask/pos internally]
  - QK norms are full-rank (num_heads*head_dim), not per-head
  - fused_qkv=False to match OLMo-3 separate Q/K/V projections
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
from modeling_olmo3_neuron import NeuronOlmo3Attention


# ---------------------------------------------------------------------------
# Test dimensions
#   head_dim = hidden_size // num_attention_heads  →  64 // 4 = 16  ✓
# ---------------------------------------------------------------------------
bs, sl, hs = 2, 128, 64
num_attention_heads = 4
num_key_value_heads = 2
head_dim = hs // num_attention_heads  # 16
dtype = torch.bfloat16


# ---------------------------------------------------------------------------
# Config
#   fused_qkv=False: NeuronAttentionBase creates separate q/k/v_proj weights
#   layer_types=["full_attention"]: no sliding window for test layer 0
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
    sliding_window=32,
    rope_theta=10000.0,
    rope_scaling=None,          # default (no YaRN) for unit test
    max_position_embeddings=4096,
    rms_norm_eps=1e-6,
    attention_bias=False,
    layer_types=["full_attention"],
    initial_context_length=4096,
)
config.num_cores_per_group = 1


# ---------------------------------------------------------------------------
# PyTorch reference (standalone, no HF dependencies)
#
# Uses the same LlamaRotaryEmbedding as NeuronOlmo3Attention so that RoPE
# computation is bit-identical.  Both implementations:
#   1. Project H → Q (nH*hD), K (nKvH*hD), V (nKvH*hD)
#   2. Apply full-rank RMSNorm to Q and K
#   3. Reshape to (B, H, S, hD)
#   4. Apply RoPE
#   5. Expand K/V for GQA, compute full attention (no causal mask in test)
#   6. Project output O
# ---------------------------------------------------------------------------
class PyTorchOlmo3AttentionWrapper(nn.Module):
    """
    Standalone OLMo-3 attention for correctness testing.

    Accepts only (hidden_states,) and computes position_ids internally
    so it can be used as the reference_block in test_block_correctness.
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

        # Full-rank QK norms matching OLMo-3: q_norm over (nH*hD), k_norm over (nKvH*hD)
        self.q_norm = nn.RMSNorm(self.num_heads * self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.num_kv_heads * self.head_dim, eps=config.rms_norm_eps)

        # Same RoPE implementation as NeuronOlmo3Attention for bit-identical results
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        # Projections
        q = self.q_proj(hidden_states)  # (B, S, nH*hD)
        k = self.k_proj(hidden_states)  # (B, S, nKvH*hD)
        v = self.v_proj(hidden_states)  # (B, S, nKvH*hD)

        # Full-rank QK normalization (before reshape, matching OLMo-3 original)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape to per-head layout
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)    # (B, nH, S, hD)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2) # (B, nKvH, S, hD)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2) # (B, nKvH, S, hD)

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

        # Full attention (no causal mask — matches the zero-mask passed to Neuron block)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
        # (B, nH, S, hD) → (B, S, nH*hD)
        attn = attn.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)

        return self.o_proj(attn)

    def state_dict(self, **kwargs):
        """
        Expose state dict with flat keys matching what the HF OLMo-3 model uses,
        so the weight_mapping below translates correctly to NxDI keys.
        """
        sd = super().state_dict(**kwargs)
        # Rename q_norm/k_norm → q_norm/k_norm (already using those names)
        return sd


# ---------------------------------------------------------------------------
# Weight mapping: PyTorch key → NxDI key (block. prefix added automatically)
#
# CPU mode, fused_qkv=False:
#   NeuronAttentionBase creates qkv_proj.q_proj / k_proj / v_proj
#   and o_proj.o_proj under the BlockWrapper's self.block namespace.
# ---------------------------------------------------------------------------
weight_mapping = {
    # Q / K / V projections
    "q_proj.weight": "qkv_proj.q_proj.weight",
    "k_proj.weight": "qkv_proj.k_proj.weight",
    "v_proj.weight": "qkv_proj.v_proj.weight",
    # Output projection (GroupQueryAttention_O uses layer_name='o_proj')
    "o_proj.weight": "o_proj.o_proj.weight",
    # QK norms (Olmo3HeadedRMSNorm stores the weight in .norm.weight)
    "q_norm.weight": "q_layernorm.norm.weight",
    "k_norm.weight": "k_layernorm.norm.weight",
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
    neuron_block_class=NeuronOlmo3Attention,
    pytorch_block_class=PyTorchOlmo3AttentionWrapper,
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
