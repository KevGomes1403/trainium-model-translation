"""
Unit test: NeuronGemma4TextAttention vs HF Gemma4TextAttention.

We import the PyTorch reference directly from the HF source file — no
copy/paste/paraphrase.  An adapter module fixes kwargs (position_ids /
position_embeddings / mask / shared_kv_states) so the block presents a single
`forward(hidden_states) -> hidden_states` signature suitable for block testing.

Two flavours are tested with separate checkpoints:
  TEST 1 — sliding_attention (exercises Q/K/V norms, sliding window, full RoPE)
  TEST 2 — full_attention    (adds partial RoPE + attention_k_eq_v)
"""

import contextlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Shim missing transformers internals that the unreleased Gemma4 HF source
# imports.  The installed transformers==4.57.x does not yet ship Gemma4.
# These shims are pure infrastructure plumbing — they do NOT reimplement
# Gemma4TextAttention (which remains the unmodified class imported below).
# --------------------------------------------------------------------------- #
def _install_hf_shims():
    import transformers  # noqa: F401
    import transformers.utils as _tu

    # 1. transformers.initialization — minimal stubs used only in weight init.
    if "transformers.initialization" not in sys.modules:
        init_mod = types.ModuleType("transformers.initialization")
        init_mod.ones_ = torch.nn.init.ones_
        init_mod.zeros_ = torch.nn.init.zeros_
        init_mod.normal_ = torch.nn.init.normal_
        init_mod.copy_ = lambda dst, src: dst.data.copy_(src.data if hasattr(src, "data") else src)
        init_mod.constant_ = torch.nn.init.constant_
        sys.modules["transformers.initialization"] = init_mod
        setattr(transformers, "initialization", init_mod)

    # 2. transformers.utils.output_capturing — OutputRecorder, capture_outputs
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

    # 3. transformers.utils.type_validators — interval() decorator used by config.
    if "transformers.utils.type_validators" not in sys.modules:
        tv_mod = types.ModuleType("transformers.utils.type_validators")

        def _interval(*a, **kw):
            # No-op validator.  Usage pattern in config file:
            #   x: float = interval(min=0.0, max=1.0)(default=0.02)
            # So returning a function that accepts `default=` and returns that value.
            def _setter(default=None, **kw2):
                return default
            return _setter

        tv_mod.interval = _interval
        sys.modules["transformers.utils.type_validators"] = tv_mod
        setattr(_tu, "type_validators", tv_mod)

    # 4. maybe_autocast + merge_with_config_defaults on transformers.utils.generic
    import transformers.utils.generic as _gen
    if not hasattr(_gen, "maybe_autocast"):
        @contextlib.contextmanager
        def maybe_autocast(device_type="cpu", enabled=False, dtype=None):
            # Null context — float32 is fine for CPU test.
            yield

        _gen.maybe_autocast = maybe_autocast
    if not hasattr(_gen, "merge_with_config_defaults"):
        def merge_with_config_defaults(*a, **kw):
            def _decorator(fn):
                return fn
            return _decorator

        _gen.merge_with_config_defaults = merge_with_config_defaults

    # 5. transformers.integrations — use_experts_implementation, use_kernelized_func
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

    # 6. transformers.masking_utils — create_bidirectional_mask etc. (unused in
    #    the attention class itself; referenced elsewhere in modeling_gemma4)
    import transformers.masking_utils as _mu
    for name in (
        "create_bidirectional_mask",
        "create_causal_mask",
        "create_masks_for_generate",
        "create_sliding_window_causal_mask",
    ):
        if not hasattr(_mu, name):
            setattr(_mu, name, lambda *a, **kw: None)

    # 7. PreTrainedConfig alias (new name in newer transformers; old name PretrainedConfig)
    import transformers.configuration_utils as _cu
    if not hasattr(_cu, "PreTrainedConfig"):
        _cu.PreTrainedConfig = _cu.PretrainedConfig

    # 8. transformers.utils.torch_compilable_check — decorator; no-op fallback.
    if not hasattr(_tu, "torch_compilable_check"):
        def torch_compilable_check(*a, **kw):
            if a and callable(a[0]):
                return a[0]
            def _decorator(fn):
                return fn
            return _decorator

        _tu.torch_compilable_check = torch_compilable_check

    # 8b. Patch auto_docstring to be a no-op — auto_method_docstring in the
    #     installed transformers doesn't handle union type annotations used by
    #     the unreleased Gemma4 source.
    import transformers.utils.auto_docstring as _ad
    def _noop_auto_docstring(obj=None, *a, **kw):
        if obj is None:
            def _decorator(fn):
                return fn
            return _decorator
        return obj
    _ad.auto_docstring = _noop_auto_docstring
    _tu.auto_docstring = _noop_auto_docstring
    # Also patch the imported reference inside already-loaded transformers.
    import transformers as _tf
    _tf.utils.auto_docstring = _noop_auto_docstring
    # And patch it in the root transformers namespace if already used.
    if hasattr(_tf, "auto_docstring"):
        _tf.auto_docstring = _noop_auto_docstring

    # 9. transformers.models.gemma4.configuration_gemma4 — the real config file
    #    uses @strict without @dataclass, which fails on the installed
    #    huggingface_hub.  We don't actually need the config classes for the
    #    test (we use SimpleNamespace).  Provide a stub module with placeholder
    #    classes so the relative import in modeling_gemma4 succeeds.
    cfg_mod_name = "transformers.models.gemma4.configuration_gemma4"
    if cfg_mod_name not in sys.modules:
        cfg_mod = types.ModuleType(cfg_mod_name)
        from transformers.configuration_utils import PretrainedConfig as _PTC

        class _StubCfg(_PTC):
            pass

        cfg_mod.Gemma4AudioConfig = _StubCfg
        cfg_mod.Gemma4Config = _StubCfg
        cfg_mod.Gemma4TextConfig = _StubCfg
        cfg_mod.Gemma4VisionConfig = _StubCfg
        sys.modules[cfg_mod_name] = cfg_mod


_install_hf_shims()


# Make block_testing_utils + neuron block importable.
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
NEURON_DIR = ROOT_DIR.parent
if str(NEURON_DIR) not in sys.path:
    sys.path.insert(0, str(NEURON_DIR))

from block_testing_utils import test_block_correctness
from blocks.neuron_gemma4_attention import NeuronGemma4TextAttention

# Import HF reference classes DIRECTLY from the original HF source file (UNMODIFIED).
# The HF source uses relative imports (from ... import ...) so it must be loaded
# as a submodule of transformers.models.  We expose it via a symlink:
#   /opt/.../site-packages/transformers/models/gemma4 ->
#   /home/ubuntu/trainium-model-translation/models/gemma4/hf
from transformers.models.gemma4.modeling_gemma4 import (  # noqa: E402
    Gemma4TextAttention,
    Gemma4TextRotaryEmbedding,
)

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig


# --------------------------------------------------------------------------- #
# Tiny config for testing.
#
# Gemma4TextAttention reads these fields from `config`:
#   layer_types, sliding_window, head_dim, global_head_dim, attention_k_eq_v,
#   num_global_key_value_heads, num_key_value_heads, num_attention_heads,
#   attention_dropout, use_bidirectional_attention, num_hidden_layers,
#   num_kv_shared_layers (optional), hidden_size, attention_bias,
#   rms_norm_eps, rope_parameters, _attn_implementation.
# Gemma4TextRotaryEmbedding reads: layer_types, rope_parameters,
#   max_position_embeddings, head_dim / global_head_dim.
# --------------------------------------------------------------------------- #
def _make_test_configs(is_sliding: bool):
    """Return (neuron_inference_config, hf_config_namespace) for a test flavour."""
    bs, sl = 2, 32
    hidden_size = 64
    num_attention_heads = 8
    # Sliding: kv_heads=2, head_dim=16 (hidden_size//nh).
    # Full:    kv_heads=2, global_head_dim=32 (bigger than head_dim to exercise
    #          the separate full-layer dim path), partial_rotary_factor=0.25
    #          so rotated_dim=8 channels.
    num_kv_heads_sliding = 2
    num_kv_heads_full = 2
    head_dim = 16
    global_head_dim = 32
    sliding_window = 16

    # HF layer_types list — two layers, one of each type (last must be full).
    layer_types = ["sliding_attention", "full_attention"]
    layer_idx = 0 if is_sliding else 1

    rope_parameters = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        "full_attention": {
            "rope_type": "default",  # avoid the HF "proportional" ROPE_INIT_FUNCTIONS path
            "rope_theta": 1_000_000.0,
            "partial_rotary_factor": 0.25,
        },
    }

    # --- HF-side config (SimpleNamespace mimicking Gemma4TextConfig) ---------
    hf_cfg = SimpleNamespace(
        layer_types=layer_types,
        sliding_window=sliding_window,
        head_dim=head_dim,
        global_head_dim=global_head_dim,
        attention_k_eq_v=True,
        num_global_key_value_heads=num_kv_heads_full,
        num_key_value_heads=num_kv_heads_sliding,
        num_attention_heads=num_attention_heads,
        attention_dropout=0.0,
        use_bidirectional_attention=None,
        num_hidden_layers=len(layer_types),
        num_kv_shared_layers=0,
        hidden_size=hidden_size,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_parameters=rope_parameters,
        _attn_implementation="eager",
        max_position_embeddings=128,
    )

    # --- NxDI-side config ---------------------------------------------------
    neuron_config = NeuronConfig(
        batch_size=bs,
        seq_len=sl,
        tp_degree=1,
        torch_dtype=torch.bfloat16,
        on_cpu=True,
        fused_qkv=False,  # separate q/k/v projections for clean weight mapping
    )
    effective_head_dim = head_dim if is_sliding else global_head_dim
    effective_num_kv_heads = num_kv_heads_sliding if is_sliding else num_kv_heads_full

    config = InferenceConfig(
        neuron_config=neuron_config,
        hidden_size=hidden_size,
        head_dim=effective_head_dim,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=effective_num_kv_heads,
        sliding_window=sliding_window,
        rope_theta=rope_parameters[layer_types[layer_idx]]["rope_theta"],
        max_position_embeddings=hf_cfg.max_position_embeddings,
        attention_bias=False,
        initial_context_length=hf_cfg.max_position_embeddings,
    )
    # Inject Gemma4-specific fields the neuron block reads.
    config.rms_norm_eps = 1e-6
    config.num_cores_per_group = 1
    config.layer_types = layer_types
    config.global_head_dim = global_head_dim
    config.num_global_key_value_heads = num_kv_heads_full
    config.attention_k_eq_v = True
    config.rope_parameters = rope_parameters

    return bs, sl, hidden_size, config, hf_cfg, layer_idx, effective_head_dim


# --------------------------------------------------------------------------- #
# Adapter wrapping the HF Gemma4TextAttention so it looks like
# `forward(hidden_states) -> hidden_states`.  The inner `.block` is UNMODIFIED.
# --------------------------------------------------------------------------- #
class HFGemma4AttentionAdapter(nn.Module):
    def __init__(self, hf_config, layer_idx: int):
        super().__init__()
        self.hf_config = hf_config
        self.layer_idx = layer_idx
        # Unmodified HF class — imported from the original file.
        self.block = Gemma4TextAttention(hf_config, layer_idx=layer_idx)
        # Gemma4's RoPE lives OUTSIDE the attention block in HF; it is called
        # by the decoder and the (cos, sin) pair is passed via position_embeddings.
        self.rotary_emb = Gemma4TextRotaryEmbedding(hf_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        device = hidden_states.device
        position_ids = (
            torch.arange(S, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        layer_type = self.hf_config.layer_types[self.layer_idx]
        cos, sin = self.rotary_emb(hidden_states, position_ids, layer_type=layer_type)
        # Unmasked: pass attention_mask=None so HF's eager path treats all
        # positions as attendable (matches the NxDI all-ones keep-mask used below).
        shared_kv_states: dict = {}
        attn_out, _ = self.block(
            hidden_states=hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None,
            shared_kv_states=shared_kv_states,
            past_key_values=None,
        )
        return attn_out


# --------------------------------------------------------------------------- #
# Weight mapping.
# HF Gemma4TextAttention params (layer 0, sliding, has v_proj):
#   block.q_proj.weight, block.k_proj.weight, block.v_proj.weight, block.o_proj.weight
#   block.q_norm.weight, block.k_norm.weight
#   (block.v_norm has no weight because with_scale=False)
# HF Gemma4TextAttention params (layer 1, full, attention_k_eq_v=True):
#   same but NO block.v_proj.weight
#
# NxDI side (fused_qkv=False, on_cpu=True, tp_degree=1):
#   qkv_proj.q_proj.weight, qkv_proj.k_proj.weight, qkv_proj.v_proj.weight
#   o_proj.o_proj.weight
#   q_layernorm.weight, k_layernorm.weight
#   (v_layernorm has no weight — NoScaleRMSNorm)
# --------------------------------------------------------------------------- #
def _weight_mapping(is_sliding: bool) -> dict:
    mapping = {
        # HF adapter stores the attention under `.block.*`
        "block.q_proj.weight": "qkv_proj.q_proj.weight",
        "block.k_proj.weight": "qkv_proj.k_proj.weight",
        "block.o_proj.weight": "o_proj.o_proj.weight",
        "block.q_norm.weight": "q_layernorm.weight",
        "block.k_norm.weight": "k_layernorm.weight",
    }
    if is_sliding:
        # Sliding layer has a real v_proj.
        mapping["block.v_proj.weight"] = "qkv_proj.v_proj.weight"
    # Full layer (attention_k_eq_v=True): HF has no v_proj; NxDI's v_proj slab
    # exists but its output is discarded in prep_qkv_tensors (V = K.clone()), so
    # leaving NxDI's v_proj weight at its compile-time init value is harmless.
    return mapping


# --------------------------------------------------------------------------- #
# Test runner
# --------------------------------------------------------------------------- #
def _run_one(is_sliding: bool, checkpoint_name: str, label: str):
    bs, sl, hs, config, hf_cfg, layer_idx, head_dim = _make_test_configs(is_sliding)
    dtype = torch.bfloat16

    torch.manual_seed(123)
    sample = torch.rand(bs, sl, hs, dtype=dtype)
    position_ids = torch.arange(sl, dtype=torch.long).unsqueeze(0).expand(bs, -1)
    # NxDI all-ones keep-mask → unmasked (matches HF attention_mask=None).
    attention_mask = torch.ones(bs, 1, sl, sl, dtype=dtype)

    example_inputs = [(torch.zeros(bs, sl, hs, dtype=dtype), attention_mask, position_ids)]
    test_inputs = [(sample, attention_mask, position_ids)]
    reference_inputs = [(sample,)]  # adapter's forward takes only hidden_states

    print("=" * 80)
    print(f"TEST: {label} (layer_idx={layer_idx}, head_dim={head_dim})")
    print("=" * 80)

    test_block_correctness(
        neuron_block_class=NeuronGemma4TextAttention,
        pytorch_block_class=HFGemma4AttentionAdapter,
        weight_mapping=_weight_mapping(is_sliding),
        example_inputs=example_inputs,
        test_inputs=test_inputs,
        reference_inputs=reference_inputs,
        checkpoint_name=checkpoint_name,
        seed=42,
        batch_size=bs,
        seq_len=sl,
        hidden_size=hs,
        neuron_init_kwargs={"config": config, "layer_idx": layer_idx},
        pytorch_init_kwargs={"hf_config": hf_cfg, "layer_idx": layer_idx},
        verbose=True,
    )


if __name__ == "__main__":
    _run_one(is_sliding=True, checkpoint_name="gemma4_attn_sliding.pt", label="sliding_attention")
    print()
    _run_one(is_sliding=False, checkpoint_name="gemma4_attn_full.pt", label="full_attention (k_eq_v + partial RoPE)")
    print()
    print("=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
