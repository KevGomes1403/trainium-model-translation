"""
Unit tests for BLOCK F — Gemma4 Vision Merge components:
  (a) NeuronGemma4VisionPatchEmbedder vs Gemma4VisionPatchEmbedder
  (b) NeuronGemma4VisionPooler        vs Gemma4VisionPooler
  (c) NeuronGemma4MultimodalEmbedder  vs Gemma4MultimodalEmbedder

Reference PyTorch classes are imported DIRECTLY from the HF source.
Tolerance: bf16 atol ~5e-3.
"""

import contextlib
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# HF shims — identical set to test_attention.py. The Gemma4 source relies on
# several transformers internals not present in transformers 4.57.x. These
# shims are pure infrastructure plumbing; they do NOT reimplement the reference
# classes tested below (those remain the unmodified HF classes).
# --------------------------------------------------------------------------- #
def _install_hf_shims():
    import transformers  # noqa: F401
    import transformers.utils as _tu

    if "transformers.initialization" not in sys.modules:
        init_mod = types.ModuleType("transformers.initialization")
        init_mod.ones_ = torch.nn.init.ones_
        init_mod.zeros_ = torch.nn.init.zeros_
        init_mod.normal_ = torch.nn.init.normal_
        init_mod.copy_ = lambda dst, src: dst.data.copy_(src.data if hasattr(src, "data") else src)
        init_mod.constant_ = torch.nn.init.constant_
        sys.modules["transformers.initialization"] = init_mod
        setattr(transformers, "initialization", init_mod)

    if "transformers.utils.output_capturing" not in sys.modules:
        oc_mod = types.ModuleType("transformers.utils.output_capturing")

        class _OutputRecorder:
            def __init__(self, *a, **kw):
                pass

        def _capture_outputs(*a, **kw):
            def _decorator(fn):
                return fn
            return _decorator

        oc_mod.OutputRecorder = _OutputRecorder
        oc_mod.capture_outputs = _capture_outputs
        sys.modules["transformers.utils.output_capturing"] = oc_mod
        setattr(_tu, "output_capturing", oc_mod)

    if "transformers.utils.type_validators" not in sys.modules:
        tv_mod = types.ModuleType("transformers.utils.type_validators")

        def _interval(*a, **kw):
            def _identity(x):
                return x
            return _identity

        tv_mod.interval = _interval
        sys.modules["transformers.utils.type_validators"] = tv_mod
        setattr(_tu, "type_validators", tv_mod)

    import transformers.utils.generic as _gen
    if not hasattr(_gen, "maybe_autocast"):
        @contextlib.contextmanager
        def maybe_autocast(device_type="cpu", enabled=False, dtype=None):
            yield

        _gen.maybe_autocast = maybe_autocast
    if not hasattr(_gen, "merge_with_config_defaults"):
        def merge_with_config_defaults(*a, **kw):
            def _decorator(fn):
                return fn
            return _decorator

        _gen.merge_with_config_defaults = merge_with_config_defaults

    import transformers.integrations as _integ
    if not hasattr(_integ, "use_kernelized_func"):
        def use_kernelized_func(*a, **kw):
            def _decorator(cls):
                return cls
            return _decorator

        _integ.use_kernelized_func = use_kernelized_func
    if not hasattr(_integ, "use_experts_implementation"):
        def use_experts_implementation(cls):
            return cls

        _integ.use_experts_implementation = use_experts_implementation

    import transformers.masking_utils as _mu
    for name in (
        "create_bidirectional_mask",
        "create_causal_mask",
        "create_masks_for_generate",
        "create_sliding_window_causal_mask",
    ):
        if not hasattr(_mu, name):
            setattr(_mu, name, lambda *a, **kw: None)

    import transformers.configuration_utils as _cu
    if not hasattr(_cu, "PreTrainedConfig"):
        _cu.PreTrainedConfig = _cu.PretrainedConfig

    if not hasattr(_tu, "torch_compilable_check"):
        def torch_compilable_check(*a, **kw):
            if a and callable(a[0]):
                return a[0]
            def _decorator(fn):
                return fn
            return _decorator

        _tu.torch_compilable_check = torch_compilable_check


_install_hf_shims()


# --------------------------------------------------------------------------- #
# Path setup
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
MODEL_DIR = ROOT_DIR.parent  # .../gemma4/neuron
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))


from neuronx_distributed_inference.models.config import NeuronConfig, InferenceConfig

from block_testing_utils import test_block_correctness
from blocks.neuron_gemma4_vision_merge import (
    NeuronGemma4VisionPatchEmbedder,
    NeuronGemma4VisionPooler,
    NeuronGemma4MultimodalEmbedder,
)

# HF source imports — via the symlinked transformers.models.gemma4 package.
# This preserves relative imports in modeling_gemma4.py.
from transformers.models.gemma4.modeling_gemma4 import (  # noqa: E402
    Gemma4VisionPatchEmbedder,
    Gemma4VisionPooler,
    Gemma4MultimodalEmbedder,
)
from transformers.models.gemma4.configuration_gemma4 import (  # noqa: E402
    Gemma4VisionConfig,
    Gemma4TextConfig,
)


DTYPE = torch.bfloat16


# ===========================================================================
# Helpers
# ===========================================================================
def _make_vision_config(hidden_size, patch_size, position_embedding_size,
                        pooling_kernel_size=3, standardize=False):
    return Gemma4VisionConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=hidden_size // 4,
        patch_size=patch_size,
        position_embedding_size=position_embedding_size,
        pooling_kernel_size=pooling_kernel_size,
        standardize=standardize,
        rms_norm_eps=1e-6,
    )


def _make_text_config(hidden_size):
    return Gemma4TextConfig(hidden_size=hidden_size, rms_norm_eps=1e-6,
                            num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=4, head_dim=16,
                            intermediate_size=hidden_size * 2)


def _build_inference_config(vision_config, text_config=None, tp_degree=1,
                            batch_size=1, seq_len=16):
    neuron_config = NeuronConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        tp_degree=tp_degree,
        torch_dtype=DTYPE,
        on_cpu=True,
        fused_qkv=False,
    )
    ic = InferenceConfig(
        neuron_config=neuron_config,
        hidden_size=vision_config.hidden_size,
        num_attention_heads=vision_config.num_attention_heads,
        num_key_value_heads=vision_config.num_key_value_heads,
        head_dim=vision_config.head_dim,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        initial_context_length=4096,
    )
    ic.vision_config = vision_config
    if text_config is not None:
        ic.text_config = text_config
    ic.num_cores_per_group = 1
    return ic


# ===========================================================================
# Test (a): NeuronGemma4VisionPatchEmbedder vs Gemma4VisionPatchEmbedder
# ===========================================================================
def test_patch_embedder():
    H_VIS = 32
    PATCH = 4
    N_POS = 64
    P_MAX = 16
    BS = 1

    vcfg = _make_vision_config(hidden_size=H_VIS, patch_size=PATCH,
                               position_embedding_size=N_POS)
    ic = _build_inference_config(vcfg, batch_size=BS, seq_len=P_MAX)

    patch_in = 3 * PATCH * PATCH  # 48
    torch.manual_seed(123)
    pixel_values = torch.rand(BS, P_MAX, patch_in, dtype=DTYPE)
    # 4x4 grid of patch positions; last 4 patches are padding (-1, -1).
    positions = torch.full((BS, P_MAX, 2), -1, dtype=torch.long)
    n_valid = 12
    for i in range(n_valid):
        positions[0, i, 0] = i % 4
        positions[0, i, 1] = i // 4
    padding_positions = (positions == -1).all(dim=-1)

    class RefPE(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Gemma4VisionPatchEmbedder(vcfg)
            self.register_buffer("_positions", positions, persistent=False)
            self.register_buffer("_padding", padding_positions, persistent=False)

        def forward(self, pv):
            return self.block(pv, self._positions, self._padding)

        def state_dict(self, **kwargs):
            return self.block.state_dict(**kwargs)

        def named_parameters(self, *args, **kwargs):
            return self.block.named_parameters(*args, **kwargs)

    example_inputs = [(
        torch.zeros(BS, P_MAX, patch_in, dtype=DTYPE),
        positions,
        padding_positions,
    )]
    test_inputs = [(pixel_values, positions, padding_positions)]
    reference_inputs = [(pixel_values,)]

    weight_mapping = {
        "input_proj.weight": "input_proj.weight",
        "position_embedding_table": "position_embedding_table",
    }

    test_block_correctness(
        neuron_block_class=NeuronGemma4VisionPatchEmbedder,
        pytorch_block_class=RefPE,
        weight_mapping=weight_mapping,
        neuron_init_kwargs={"config": ic},
        pytorch_init_kwargs={},
        example_inputs=example_inputs,
        test_inputs=test_inputs,
        reference_inputs=reference_inputs,
        checkpoint_name="gemma4_patch_embedder.pt",
        seed=42,
        verbose=True,
    )


# ===========================================================================
# Test (b): NeuronGemma4VisionPooler vs Gemma4VisionPooler
# ===========================================================================
def test_pooler():
    H_VIS = 32
    K = 3
    P_MAX = 9      # 3x3 grid
    OUT_LEN = 1    # (3/3)*(3/3)=1
    BS = 1

    vcfg = _make_vision_config(hidden_size=H_VIS, patch_size=4,
                               position_embedding_size=64,
                               pooling_kernel_size=K)
    ic = _build_inference_config(vcfg, batch_size=BS, seq_len=P_MAX)

    torch.manual_seed(456)
    hidden_states = torch.randn(BS, P_MAX, H_VIS, dtype=DTYPE)
    positions = torch.zeros(BS, P_MAX, 2, dtype=torch.long)
    for i in range(P_MAX):
        positions[0, i, 0] = i % 3
        positions[0, i, 1] = i // 3
    padding_positions = torch.zeros(BS, P_MAX, dtype=torch.bool)

    class RefPooler(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Gemma4VisionPooler(vcfg)
            self.register_buffer("_positions", positions, persistent=False)
            self.register_buffer("_padding", padding_positions, persistent=False)

        def forward(self, hs):
            out, _ = self.block(hs, self._positions, self._padding,
                                output_length=OUT_LEN)
            return out

        def state_dict(self, **kwargs):
            return self.block.state_dict(**kwargs)

        def named_parameters(self, *args, **kwargs):
            return self.block.named_parameters(*args, **kwargs)

    class NeuronPoolerWrapped(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.block = NeuronGemma4VisionPooler(config)

        def forward(self, hs, pos, pad):
            out, _mask = self.block(hs, pos, pad, output_length=OUT_LEN)
            return out

    example_inputs = [(
        torch.zeros(BS, P_MAX, H_VIS, dtype=DTYPE),
        positions,
        padding_positions,
    )]
    test_inputs = [(hidden_states, positions, padding_positions)]
    reference_inputs = [(hidden_states,)]

    weight_mapping = {}  # Pooler has no learnable weights.

    test_block_correctness(
        neuron_block_class=NeuronPoolerWrapped,
        pytorch_block_class=RefPooler,
        weight_mapping=weight_mapping,
        neuron_init_kwargs={"config": ic},
        pytorch_init_kwargs={},
        example_inputs=example_inputs,
        test_inputs=test_inputs,
        reference_inputs=reference_inputs,
        checkpoint_name="gemma4_pooler.pt",
        seed=42,
        verbose=True,
    )


# ===========================================================================
# Test (c): NeuronGemma4MultimodalEmbedder vs Gemma4MultimodalEmbedder
# ===========================================================================
def test_multimodal_embedder():
    H_VIS = 32
    H_TEXT = 48
    SEQ = 8
    BS = 1

    vcfg = _make_vision_config(hidden_size=H_VIS, patch_size=4,
                               position_embedding_size=64)
    tcfg = _make_text_config(hidden_size=H_TEXT)
    ic = _build_inference_config(vcfg, text_config=tcfg, batch_size=BS, seq_len=SEQ)

    torch.manual_seed(789)
    inputs_embeds = torch.randn(BS, SEQ, H_VIS, dtype=DTYPE)

    class RefMME(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Gemma4MultimodalEmbedder(vcfg, tcfg)

        def forward(self, x):
            return self.block(x)

        def state_dict(self, **kwargs):
            return self.block.state_dict(**kwargs)

        def named_parameters(self, *args, **kwargs):
            return self.block.named_parameters(*args, **kwargs)

    example_inputs = [(torch.zeros(BS, SEQ, H_VIS, dtype=DTYPE),)]
    test_inputs = [(inputs_embeds,)]
    reference_inputs = [(inputs_embeds,)]

    weight_mapping = {
        "embedding_projection.weight": "embedding_projection.weight",
    }

    test_block_correctness(
        neuron_block_class=NeuronGemma4MultimodalEmbedder,
        pytorch_block_class=RefMME,
        weight_mapping=weight_mapping,
        neuron_init_kwargs={"config": ic},
        pytorch_init_kwargs={},
        example_inputs=example_inputs,
        test_inputs=test_inputs,
        reference_inputs=reference_inputs,
        checkpoint_name="gemma4_multimodal_embedder.pt",
        seed=42,
        verbose=True,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n##### (a) Patch Embedder #####")
    test_patch_embedder()
    print("\n##### (b) Pooler #####")
    test_pooler()
    print("\n##### (c) Multimodal Embedder #####")
    test_multimodal_embedder()
    print("\nALL TESTS PASSED.")
