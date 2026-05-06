"""
Unit tests for BLOCK C (Gemma4 Text Embeddings + PLE + RMSNorm + logit softcap).

References are imported DIRECTLY from the HF source -- we never copy the HF
classes. Each sub-component has its own `test_block_correctness` invocation:

  1. NeuronGemma4ScaledWordEmbedding   vs  Gemma4TextScaledWordEmbedding (HF)
  2. NeuronGemma4RMSNorm               vs  Gemma4RMSNorm (HF)
  3. NeuronGemma4PLE                   vs  an adapter that composes HF classes
                                           (Gemma4TextScaledWordEmbedding +
                                            nn.Linear + Gemma4RMSNorm) to mirror
                                           `Gemma4TextModel.get_per_layer_inputs`
                                           and `project_per_layer_inputs`
                                           (HF L1692-1769) without constructing
                                           the entire heavy Gemma4TextModel.
  4. apply_logit_softcap               vs  HF softcap math from
                                           `Gemma4ForCausalLM.forward` L1839-1842.

Tolerance: atol=5e-3 (bf16).
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---- paths ------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
MODEL_DIR = THIS_DIR.parent  # .../gemma4/neuron
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# -----------------------------------------------------------------------------
# HF source (DO NOT copy classes; load their definitions directly from the
# HF file via AST extraction so we use the *verbatim source text* for
# Gemma4RMSNorm and Gemma4TextScaledWordEmbedding).
#
# We cannot do `from modeling_gemma4 import ...` because that file's top-level
# imports use `...` (grandparent-relative) paths referring to a newer
# `transformers` private API not installed in this env. The workaround:
# parse the file, extract just the two self-contained class AST nodes, and
# exec them in an isolated namespace with only `torch` and `nn`. This uses
# the *exact HF source text* without copying it into our own files.
# -----------------------------------------------------------------------------
HF_MODELING_PATH = "/home/ubuntu/trainium-model-translation/models/gemma4/hf/modeling_gemma4.py"


def _load_hf_classes(source_path: str, class_names):
    with open(source_path, "r") as f:
        tree = ast.parse(f.read())
    wanted = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in class_names]
    missing = set(class_names) - {n.name for n in wanted}
    if missing:
        raise ImportError(f"Classes not found in {source_path}: {missing}")
    ns = {"torch": torch, "nn": nn}
    code = compile(ast.Module(body=wanted, type_ignores=[]), source_path, "exec")
    exec(code, ns)  # noqa: S102
    return {name: ns[name] for name in class_names}


_hf = _load_hf_classes(
    HF_MODELING_PATH, ["Gemma4RMSNorm", "Gemma4TextScaledWordEmbedding"]
)
Gemma4RMSNorm = _hf["Gemma4RMSNorm"]
Gemma4TextScaledWordEmbedding = _hf["Gemma4TextScaledWordEmbedding"]

from block_testing_utils import test_block_correctness  # noqa: E402
from blocks.neuron_gemma4_embeddings import (  # noqa: E402
    NeuronGemma4PLE,
    NeuronGemma4RMSNorm,
    NeuronGemma4ScaledWordEmbedding,
    RMSNORM_OFFSET,
    apply_logit_softcap,
)


# Gemma4 padding_idx default (configuration_gemma4.py L169).
PAD_IDX = 0


# =============================================================================
# Test 1: Scaled word embedding
# =============================================================================
def test_scaled_word_embedding():
    bs, sl = 2, 16
    hidden_size = 64
    vocab = 256
    dtype = torch.bfloat16

    # The Neuron block wraps ParallelEmbedding, which only supports int32/int64
    # token ids; we use int64 to match production.
    # int64 inputs cannot be randomized via torch.nn.init.normal_, so we build
    # a simple wrapper whose 'parameters' are just the embedding weight.
    class NeuronWrapper(nn.Module):
        def __init__(self, **kw):
            super().__init__()
            self.embed = NeuronGemma4ScaledWordEmbedding(
                num_embeddings=vocab,
                embedding_dim=hidden_size,
                padding_idx=PAD_IDX,
                embed_scale=math.sqrt(hidden_size),
                dtype=dtype,
                shard_across_embedding=False,
            )

        def forward(self, input_ids):
            return self.embed(input_ids)

    class ReferenceEmbedding(nn.Module):
        """Direct use of HF Gemma4TextScaledWordEmbedding (no re-implementation)."""

        def __init__(self):
            super().__init__()
            self.embed = Gemma4TextScaledWordEmbedding(
                vocab, hidden_size, PAD_IDX, embed_scale=math.sqrt(hidden_size)
            )

        def forward(self, input_ids):
            return self.embed(input_ids)

    torch.manual_seed(0)
    # int64 ids in [1, vocab)
    ids = torch.randint(low=1, high=vocab, size=(bs, sl), dtype=torch.int64)

    example_inputs = [(torch.zeros(bs, sl, dtype=torch.int64),)]
    test_inputs = [(ids,)]
    reference_inputs = [(ids,)]

    weight_mapping = {
        # HF weight -> Neuron checkpoint key (within the block.)
        "embed.weight": "embed.embed_tokens.weight",
    }

    test_block_correctness(
        neuron_block_class=NeuronWrapper,
        pytorch_block_class=ReferenceEmbedding,
        weight_mapping=weight_mapping,
        example_inputs=example_inputs,
        test_inputs=test_inputs,
        reference_inputs=reference_inputs,
        checkpoint_name="scaled_word_embedding.pt",
        seed=42,
        batch_size=bs,
        seq_len=sl,
        hidden_size=hidden_size,
        neuron_init_kwargs={},   # our wrapper takes no required kwargs
        pytorch_init_kwargs={},
        verbose=True,
    )


# =============================================================================
# Test 2: RMSNorm (with_scale=True) matches Gemma4RMSNorm exactly.
# =============================================================================
def test_rmsnorm():
    bs, sl, hs = 2, 16, 64
    dtype = torch.bfloat16

    class NeuronRMSWrapper(nn.Module):
        def __init__(self, **kw):
            super().__init__()
            self.norm = NeuronGemma4RMSNorm(hs, eps=1e-6, with_scale=True)

        def forward(self, x):
            return self.norm(x)

    class ReferenceRMSWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            # Direct HF import -- no re-implementation.
            self.norm = Gemma4RMSNorm(hs, eps=1e-6, with_scale=True)

        def forward(self, x):
            return self.norm(x)

    torch.manual_seed(1)
    sample = torch.randn(bs, sl, hs, dtype=dtype)

    # Deviation (4) sanity: RMSNORM_OFFSET is 0 for Gemma4 -- weights are copied verbatim
    # (no +1.0 add at load time).
    assert RMSNORM_OFFSET == 0.0, "Gemma4 checkpoint does not use a +1.0 RMSNorm offset."

    test_block_correctness(
        neuron_block_class=NeuronRMSWrapper,
        pytorch_block_class=ReferenceRMSWrapper,
        weight_mapping={"norm.weight": "norm.weight"},
        example_inputs=[(torch.zeros(bs, sl, hs, dtype=dtype),)],
        test_inputs=[(sample,)],
        reference_inputs=[(sample,)],
        checkpoint_name="rmsnorm.pt",
        seed=7,
        batch_size=bs,
        seq_len=sl,
        hidden_size=hs,
        neuron_init_kwargs={},
        pytorch_init_kwargs={},
        verbose=True,
    )


# =============================================================================
# Test 3: PLE module -- full path with a tiny nonzero hidden_size_per_layer_input.
# =============================================================================
def test_ple_enabled():
    """
    Tests the full PLE pipeline against an adapter that composes *unmodified*
    HF classes to replicate `Gemma4TextModel.get_per_layer_inputs` +
    `project_per_layer_inputs`. We pass `inputs_embeds` in through a small
    trick: the adapter accepts a concatenated tensor `[input_ids_as_float, inputs_embeds]`
    but that's fragile. Instead we use a custom reference module that takes the
    same tensor `inputs_embeds` as the block, and we pre-compute `input_ids`
    internally from a fixed torch.arange lookup so both sides stay in sync.
    """
    bs, sl = 2, 16
    hidden_size = 64
    ple_dim = 8
    num_layers = 2
    vocab = 128
    dtype = torch.bfloat16

    # Fixed input_ids (not trainable). Same ids used both in reference and
    # Neuron forward. We embed them into both modules by registering as buffer.
    torch.manual_seed(3)
    fixed_ids = torch.randint(low=1, high=vocab, size=(bs, sl), dtype=torch.int64)

    class NeuronPLEWrapper(nn.Module):
        def __init__(self, **kw):
            super().__init__()
            self.register_buffer("ids", fixed_ids, persistent=False)
            self.ple = NeuronGemma4PLE(
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                vocab_size_per_layer_input=vocab,
                hidden_size_per_layer_input=ple_dim,
                rms_norm_eps=1e-6,
                padding_idx=PAD_IDX,
                dtype=dtype,
            )

        def forward(self, inputs_embeds):
            return self.ple(self.ids, inputs_embeds)

    class ReferencePLE(nn.Module):
        """
        Composes HF classes (NO re-implementation) to mirror HF
        `get_per_layer_inputs` (L1692-1734) + `project_per_layer_inputs` (L1736-1769).
        """

        def __init__(self):
            super().__init__()
            self.register_buffer("ids", fixed_ids, persistent=False)
            # Token-identity embedding -- HF L1578-1583.
            self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
                vocab,
                num_layers * ple_dim,
                PAD_IDX,
                embed_scale=math.sqrt(ple_dim),
            )
            # Context projection -- HF L1585-1589.
            self.per_layer_model_projection = nn.Linear(
                hidden_size, num_layers * ple_dim, bias=False
            )
            # Projection norm -- HF L1591.
            self.per_layer_projection_norm = Gemma4RMSNorm(ple_dim, eps=1e-6)

            # Scalars -- HF L1584 + L1590.
            self.per_layer_input_scale = 2.0 ** -0.5
            self.per_layer_model_projection_scale = hidden_size ** -0.5

        def forward(self, inputs_embeds):
            # get_per_layer_inputs (HF L1730-1734)
            per_layer_inputs = self.embed_tokens_per_layer(self.ids).reshape(
                *self.ids.shape, num_layers, ple_dim
            )
            # project_per_layer_inputs (HF L1758-1769)
            proj = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
            proj = proj.reshape(*inputs_embeds.shape[:-1], num_layers, ple_dim)
            proj = self.per_layer_projection_norm(proj)
            return (proj + per_layer_inputs) * self.per_layer_input_scale

    torch.manual_seed(5)
    inputs_embeds = torch.randn(bs, sl, hidden_size, dtype=dtype)

    # Weight mapping. Note: the PLE's `per_layer_model_projection` is a
    # ColumnParallelLinear whose checkpoint key matches the HF Linear key.
    weight_mapping = {
        "embed_tokens_per_layer.weight": "ple.embed_tokens_per_layer.embed_tokens.weight",
        "per_layer_model_projection.weight": "ple.per_layer_model_projection.weight",
        "per_layer_projection_norm.weight": "ple.per_layer_projection_norm.weight",
    }

    test_block_correctness(
        neuron_block_class=NeuronPLEWrapper,
        pytorch_block_class=ReferencePLE,
        weight_mapping=weight_mapping,
        example_inputs=[(torch.zeros(bs, sl, hidden_size, dtype=dtype),)],
        test_inputs=[(inputs_embeds,)],
        reference_inputs=[(inputs_embeds,)],
        checkpoint_name="ple.pt",
        seed=11,
        batch_size=bs,
        seq_len=sl,
        hidden_size=hidden_size,
        neuron_init_kwargs={},
        pytorch_init_kwargs={},
        verbose=True,
    )


# =============================================================================
# Test 4: apply_logit_softcap -- pure functional, CPU-only check.
# =============================================================================
def test_logit_softcap():
    """
    Functional unit test: the softcap is a scalar math op, no weights and no
    XLA compilation needed. We validate against the same formula written
    inline in HF (L1839-1842). Tolerance atol=5e-3 (bf16).
    """
    torch.manual_seed(9)
    softcap = 30.0

    # fp32 path: exact math -- HF L1839-1842 verbatim on the same input.
    # Deviation (5): apply_logit_softcap runs in fp32, which is the lm_head's
    # native dtype at the HF call site; match atol = 1e-6.
    logits_fp32 = torch.randn(3, 7, 11, dtype=torch.float32) * 50.0
    got = apply_logit_softcap(logits_fp32, softcap=softcap)
    ref = logits_fp32 / softcap
    ref = torch.tanh(ref)
    ref = ref * softcap
    assert torch.allclose(got, ref, atol=1e-6), (
        f"Softcap fp32 mismatch: max|diff|={(got-ref).abs().max().item()}"
    )
    assert got.abs().max().item() <= softcap + 1e-3

    # bf16 input: our helper upcasts internally (Deviation 5). Compare against
    # the same upcast-fp32-tanh pipeline cast back to bf16 -- this is the exact
    # numerical contract of apply_logit_softcap(bf16). Tolerance 5e-3 (bf16).
    logits_bf16 = (torch.randn(3, 7, 11) * 5.0).to(torch.bfloat16)
    got_bf16 = apply_logit_softcap(logits_bf16, softcap=softcap)
    ref_bf16 = (torch.tanh(logits_bf16.float() / softcap) * softcap).to(torch.bfloat16)
    diff = (got_bf16.float() - ref_bf16.float()).abs().max().item()
    assert diff <= 5e-3, f"Softcap bf16 mismatch: max|diff|={diff}"

    # None softcap => identity.
    x = torch.randn(4, 5)
    assert torch.equal(apply_logit_softcap(x, softcap=None), x)
    print("apply_logit_softcap: OK")


# =============================================================================
# Script entry-point (runs all tests sequentially).
# =============================================================================
if __name__ == "__main__":
    print("\n--- test_logit_softcap ---")
    test_logit_softcap()

    print("\n--- test_rmsnorm ---")
    test_rmsnorm()

    print("\n--- test_scaled_word_embedding ---")
    test_scaled_word_embedding()

    print("\n--- test_ple_enabled ---")
    test_ple_enabled()

    print("\nAll tests PASSED.")
