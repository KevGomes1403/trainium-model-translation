"""
Unit test: NeuronGemma4VisionEncoderLayer vs HF Gemma4VisionEncoderLayer.

This test directly imports the HF reference from the vendored source
(models/gemma4/hf/modeling_gemma4.py). The HF source uses transformers-native
relative imports against a newer transformers revision than is installed in
this venv, so we install a small compatibility shim BEFORE importing — patching
only the symbols actually referenced by the classes we load. We do NOT
paraphrase / reimplement the HF reference.

Covers in one shot:
  * Multi-dim RoPE (HF Gemma4VisionRotaryEmbedding + apply_multidimensional_rope)
  * Q/K RMSNorm (with scale), V RMSNorm (no scale)
  * Non-causal attention with additive mask
  * GLU MLP (gate_proj * act(up_proj) -> down_proj) stripped of ClippableLinear
  * 4-norm sandwich (input / post_attn / pre_ff / post_ff)
"""

import os
import sys
from pathlib import Path
import types

import torch


# -------- Path setup ---------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_NEURON_DIR = _THIS_FILE.parent.parent  # .../models/gemma4/neuron
_BLOCKS_DIR = _NEURON_DIR / "blocks"
_TESTS_DIR = _NEURON_DIR / "tests"
_HF_DIR = _NEURON_DIR.parent / "hf"  # .../models/gemma4/hf

for p in (str(_NEURON_DIR), str(_BLOCKS_DIR), str(_TESTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


# -------- Compatibility shim for transformers --------------------------------
# The vendored modeling_gemma4.py was generated against a newer transformers
# release and references several symbols missing from the installed 4.57.x.
# We inject minimal stubs — enough to construct Gemma4VisionAttention /
# Gemma4VisionMLP / Gemma4VisionEncoderLayer / Gemma4VisionRotaryEmbedding.
def _install_transformers_shim():
    import transformers
    import transformers.configuration_utils as _cu
    import transformers.masking_utils as _mu
    import transformers.utils as _tu
    import transformers.utils.generic as _tug
    import transformers.integrations as _ti

    # PreTrainedConfig alias (newer API name).
    if not hasattr(_cu, "PreTrainedConfig"):
        _cu.PreTrainedConfig = _cu.PretrainedConfig
    if not hasattr(transformers, "PreTrainedConfig"):
        transformers.PreTrainedConfig = _cu.PretrainedConfig

    # masking_utils.create_bidirectional_mask — fallback: identity pass-through.
    if not hasattr(_mu, "create_bidirectional_mask"):
        def create_bidirectional_mask(config=None, inputs_embeds=None,
                                      attention_mask=None, **kw):
            return attention_mask
        _mu.create_bidirectional_mask = create_bidirectional_mask

    # transformers.utils.generic.maybe_autocast + merge_with_config_defaults
    if not hasattr(_tug, "maybe_autocast"):
        from contextlib import contextmanager

        @contextmanager
        def maybe_autocast(device_type="cpu", enabled=False, **kw):
            # Mirror torch.autocast's context-manager signature but be a no-op
            # when enabled=False (which is how modeling_gemma4 invokes it).
            if enabled:
                with torch.autocast(device_type=device_type, **kw):
                    yield
            else:
                yield
        _tug.maybe_autocast = maybe_autocast
    if not hasattr(_tug, "merge_with_config_defaults"):
        def merge_with_config_defaults(config, **defaults):
            return config
        _tug.merge_with_config_defaults = merge_with_config_defaults

    # transformers.utils.torch_compilable_check
    if not hasattr(_tu, "torch_compilable_check"):
        def torch_compilable_check(fn=None, **kw):
            if fn is None:
                return lambda f: f
            return fn
        _tu.torch_compilable_check = torch_compilable_check

    # transformers.utils.output_capturing (not importable on this version).
    if "transformers.utils.output_capturing" not in sys.modules:
        oc = types.ModuleType("transformers.utils.output_capturing")

        class OutputRecorder:
            def __init__(self, *a, **kw):
                pass

            def __call__(self, *a, **kw):
                return self

        def capture_outputs(*a, **kw):
            def _decorator(fn):
                return fn
            return _decorator

        oc.OutputRecorder = OutputRecorder
        oc.capture_outputs = capture_outputs
        sys.modules["transformers.utils.output_capturing"] = oc
        _tu.output_capturing = oc

    # transformers.integrations.use_experts_implementation / use_kernelized_func
    if not hasattr(_ti, "use_experts_implementation"):
        def use_experts_implementation(*a, **kw):
            def _decorator(fn):
                return fn
            return _decorator
        _ti.use_experts_implementation = use_experts_implementation
    if not hasattr(_ti, "use_kernelized_func"):
        def use_kernelized_func(*a, **kw):
            # Works whether used as @use_kernelized_func or @use_kernelized_func(fn)
            if len(a) == 1 and callable(a[0]) and not kw:
                # used without parentheses OR with a fn arg that returns a decorator
                arg = a[0]

                def _decorator(cls_or_fn):
                    return cls_or_fn
                return _decorator
            def _decorator(cls_or_fn):
                return cls_or_fn
            return _decorator
        _ti.use_kernelized_func = use_kernelized_func

    # masking_utils.create_sliding_window_causal_mask may be missing in some
    # installed versions; if so, patch with identity fallback.
    if not hasattr(_mu, "create_sliding_window_causal_mask"):
        def create_sliding_window_causal_mask(*a, **kw):
            return None
        _mu.create_sliding_window_causal_mask = create_sliding_window_causal_mask

    # transformers.integrations.strict (sometimes referenced as a decorator)
    # The HF source imports @strict from transformers.configuration_utils.
    if not hasattr(_cu, "strict"):
        def strict(cls):
            return cls
        _cu.strict = strict


_install_transformers_shim()


# -------- Load HF reference as a pseudo-package ------------------------------
# modeling_gemma4.py uses relative imports (`from ... import ACT2FN` etc),
# so we must register it under a package with `__path__`.
def _load_hf_reference():
    import importlib.util

    # Create the pseudo-package under transformers.models.gemma4 so relative
    # imports (`from ...activations import ACT2FN`) resolve through the real
    # transformers package.
    import transformers.models  # noqa: F401

    pkg_name = "transformers.models.gemma4"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(_HF_DIR)]
        sys.modules[pkg_name] = pkg

    def _load_submodule(basename):
        fullname = f"{pkg_name}.{basename}"
        if fullname in sys.modules:
            return sys.modules[fullname]
        spec = importlib.util.spec_from_file_location(
            fullname, os.path.join(str(_HF_DIR), basename + ".py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = mod
        spec.loader.exec_module(mod)
        return mod

    cfg = _load_submodule("configuration_gemma4")
    mdl = _load_submodule("modeling_gemma4")
    return cfg, mdl


_cfg_mod, _hf_mod = _load_hf_reference()

Gemma4VisionConfig = _cfg_mod.Gemma4VisionConfig
HFGemma4VisionEncoderLayer = _hf_mod.Gemma4VisionEncoderLayer
HFGemma4VisionRotaryEmbedding = _hf_mod.Gemma4VisionRotaryEmbedding


# -------- Import Neuron block & test utility --------------------------------
from block_testing_utils import test_block_correctness, _create_default_config
from neuron_gemma4_vision_encoder import (
    NeuronGemma4VisionEncoderLayer,
    Gemma4VisionRotaryEmbedding as NeuronVisionRotaryEmbedding,
)

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig


# ---------------------------------------------------------------------------- #
# Tiny test dims (spec: hidden=64, heads=4, head_dim=16, inter=128, P=16)
# ---------------------------------------------------------------------------- #
TINY_HIDDEN = 64
TINY_HEADS = 4
TINY_HEAD_DIM = 16
TINY_INTER = 128
TINY_P = 16  # 4x4 patch grid
TINY_BS = 1
RMS_EPS = 1e-6
ROPE_THETA = 100.0


def _make_position_ids_2d(P_side: int, batch: int) -> torch.Tensor:
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(P_side), torch.arange(P_side), indexing="ij"
        ),
        dim=-1,
    ).reshape(-1, 2)  # [P, 2]
    return coords.unsqueeze(0).expand(batch, -1, -1).to(torch.int64).contiguous()


def _make_hf_vision_config() -> Gemma4VisionConfig:
    # PreTrainedConfig subclasses from HF gemma4 use __post_init__ so we need
    # to construct explicitly.
    cfg = Gemma4VisionConfig()
    cfg.hidden_size = TINY_HIDDEN
    cfg.intermediate_size = TINY_INTER
    cfg.num_hidden_layers = 1
    cfg.num_attention_heads = TINY_HEADS
    cfg.num_key_value_heads = TINY_HEADS
    cfg.head_dim = TINY_HEAD_DIM
    cfg.hidden_activation = "gelu_pytorch_tanh"
    cfg.rms_norm_eps = RMS_EPS
    cfg.max_position_embeddings = 4096
    cfg.attention_bias = False
    cfg.attention_dropout = 0.0
    cfg.rope_parameters = {"rope_type": "default", "rope_theta": ROPE_THETA}
    cfg.use_clipped_linears = False
    cfg.patch_size = 16
    cfg.position_embedding_size = 256
    cfg._attn_implementation = "eager"  # ensure we use eager_attention_forward
    return cfg


class _HFLayerWithRope(torch.nn.Module):
    """Wraps HFGemma4VisionEncoderLayer + its rotary module so that
    `forward(hidden_states)` takes the same signature as the Neuron wrapper
    expects for reference_inputs.
    """

    def __init__(self):
        super().__init__()
        self.hf_config = _make_hf_vision_config()
        self.rotary_emb = HFGemma4VisionRotaryEmbedding(self.hf_config)
        self.layer = HFGemma4VisionEncoderLayer(self.hf_config, layer_idx=0)

    def set_context(self, attention_mask, position_ids):
        self._attn_mask = attention_mask
        self._pos_ids = position_ids

    def forward(self, hidden_states):
        cos, sin = self.rotary_emb(hidden_states, self._pos_ids)
        out = self.layer(
            hidden_states=hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=self._attn_mask,
            position_ids=self._pos_ids,
        )
        # HF layer returns either a tensor or a tuple; coerce.
        if isinstance(out, tuple):
            out = out[0]
        return out


class _NeuronLayerWithRope(torch.nn.Module):
    """Neuron block class passed to test_block_correctness. Owns its own
    rotary module so that its forward takes (hidden_states, attention_mask,
    position_ids) — all tensor inputs (required for XLA tracing).
    """

    def __init__(self, config):
        super().__init__()
        head_dim = getattr(config, "head_dim", TINY_HEAD_DIM)
        self.rotary_emb = NeuronVisionRotaryEmbedding(
            head_dim=head_dim, rope_theta=ROPE_THETA, ndim=2
        )
        self.layer = NeuronGemma4VisionEncoderLayer(config=config, layer_idx=0)

    def forward(self, hidden_states, attention_mask, position_ids):
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        return self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=(cos, sin),
        )


def _custom_sync_weights(reference_block, checkpoint_path, weight_mapping, verbose):
    """Custom sync: our _HFLayerWithRope has `layer.*` and `rotary_emb.*` keys;
    the Neuron wrapper similarly has `layer.*` and `rotary_emb.*` keys. rotary
    has no learnable params (inv_freq is a buffer, persistent=False in HF but
    may still appear in state_dict). We sync learnable weights only.
    """
    neuron_state_dict = torch.load(checkpoint_path)
    ref_state_dict = reference_block.state_dict()
    if verbose:
        print("Neuron keys:")
        for k in sorted(neuron_state_dict.keys()):
            print(f"  {k}: {neuron_state_dict[k].shape}")
        print("Ref keys:")
        for k in sorted(ref_state_dict.keys()):
            print(f"  {k}: {ref_state_dict[k].shape}")

    count = 0
    for ref_key, neuron_key in weight_mapping.items():
        if not neuron_key.startswith("block."):
            neuron_key = f"block.{neuron_key}"
        if ref_key not in ref_state_dict:
            if verbose:
                print(f"  WARN missing ref key: {ref_key}")
            continue
        if neuron_key not in neuron_state_dict:
            if verbose:
                print(f"  WARN missing neuron key: {neuron_key}")
            continue
        t = ref_state_dict[ref_key]
        t = t.to(dtype=neuron_state_dict[neuron_key].dtype).contiguous().clone()
        if t.shape != neuron_state_dict[neuron_key].shape:
            if verbose:
                print(
                    f"  SHAPE MISMATCH {ref_key} -> {neuron_key}: "
                    f"{t.shape} vs {neuron_state_dict[neuron_key].shape}"
                )
            continue
        neuron_state_dict[neuron_key] = t
        count += 1
        if verbose:
            print(f"  synced {ref_key} -> {neuron_key} {t.shape}")
    torch.save(neuron_state_dict, checkpoint_path)
    return count


def _build_weight_mapping():
    """
    Map PyTorch reference keys -> Neuron checkpoint keys.

    Reference side (HFLayerWithRope):
      layer.self_attn.{q,k,v,o}_proj.linear.weight   (ClippableLinear -> .linear)
      layer.self_attn.{q,k,v}_norm.weight  (v_norm has no weight since with_scale=False)
      layer.mlp.{gate,up,down}_proj.linear.weight
      layer.input_layernorm.weight
      layer.post_attention_layernorm.weight
      layer.pre_feedforward_layernorm.weight
      layer.post_feedforward_layernorm.weight

    Neuron side (NeuronLayerWithRope wrapped in _BlockWrapper):
      block.layer.self_attn.{q,k,v,o}_proj.weight
      block.layer.self_attn.{q,k}_norm.weight
      block.layer.mlp.{gate,up,down}_proj.weight
      block.layer.{input,post_attention,pre_feedforward,post_feedforward}_layernorm.weight
    """
    mapping = {}
    # Attention projections (HF wraps nn.Linear inside ClippableLinear).
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        mapping[f"layer.self_attn.{proj}.linear.weight"] = (
            f"layer.self_attn.{proj}.weight"
        )
    # Q/K RMSNorms (v_norm has no learnable weight when with_scale=False).
    for norm in ("q_norm", "k_norm"):
        mapping[f"layer.self_attn.{norm}.weight"] = f"layer.self_attn.{norm}.weight"
    # MLP projections.
    for proj in ("gate_proj", "up_proj", "down_proj"):
        mapping[f"layer.mlp.{proj}.linear.weight"] = f"layer.mlp.{proj}.weight"
    # 4-norm sandwich.
    for norm in (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ):
        mapping[f"layer.{norm}.weight"] = f"layer.{norm}.weight"
    return mapping


def _build_neuron_config() -> InferenceConfig:
    """Build an InferenceConfig carrying vision params. test_block_correctness
    will merge these over its defaults."""
    neuron_cfg = NeuronConfig(
        batch_size=TINY_BS,
        seq_len=TINY_P,
        tp_degree=1,
        torch_dtype=torch.bfloat16,
        on_cpu=True,
        fused_qkv=False,
    )
    cfg = InferenceConfig(
        neuron_config=neuron_cfg,
        hidden_size=TINY_HIDDEN,
        intermediate_size=TINY_INTER,
        num_hidden_layers=1,
        num_attention_heads=TINY_HEADS,
        num_key_value_heads=TINY_HEADS,
        head_dim=TINY_HEAD_DIM,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=RMS_EPS,
        max_position_embeddings=4096,
        attention_bias=False,
        attention_dropout=0.0,
        rope_parameters={"rope_type": "default", "rope_theta": ROPE_THETA},
        use_clipped_linears=False,
        patch_size=16,
    )
    cfg.num_cores_per_group = 1
    return cfg


def test_vision_encoder_layer_matches_hf():
    torch.manual_seed(0)

    # Inputs.
    # hidden_states: [B, P, H] in bf16.
    hidden_states = torch.randn(TINY_BS, TINY_P, TINY_HIDDEN, dtype=torch.bfloat16)
    # position_ids: 2D patch coords [0,0]...[3,3].
    position_ids = _make_position_ids_2d(int(TINY_P ** 0.5), TINY_BS)
    # Non-causal mask: all positions attend to each other (additive float form,
    # HF convention: 0 = keep, -inf = mask). We leave it all-zeros (all keep).
    attention_mask = torch.zeros(
        TINY_BS, 1, TINY_P, TINY_P, dtype=torch.float32
    )

    # Zero-valued example inputs for tracing (must match runtime shapes / dtypes).
    example_hidden = torch.zeros_like(hidden_states)
    example_mask = torch.zeros_like(attention_mask)
    example_pos = position_ids.clone()

    # Build reference and cache the context (mask + pos_ids) so its forward
    # signature is (hidden_states,) only.
    ref = _HFLayerWithRope()
    ref.set_context(attention_mask=attention_mask, position_ids=position_ids)

    config = _build_neuron_config()

    test_block_correctness(
        neuron_block_class=_NeuronLayerWithRope,
        pytorch_block_class=_HFLayerWithRope,
        weight_mapping=_build_weight_mapping(),
        config=config,
        neuron_init_kwargs={"config": config},
        pytorch_init_kwargs={},
        example_inputs=[(example_hidden, example_mask, example_pos)],
        test_inputs=[(hidden_states, attention_mask, position_ids)],
        reference_inputs=[(hidden_states,)],  # reference_block(hidden_states)
        checkpoint_name="gemma4_vision_encoder_layer.pt",
        seed=0,
        batch_size=TINY_BS,
        seq_len=TINY_P,
        hidden_size=TINY_HIDDEN,
        verbose=True,
        sync_weights_fn=_custom_sync_weights,
    )


if __name__ == "__main__":
    # The test_block_correctness helper re-creates the PyTorch reference with
    # its own internal seed to match the checkpoint-init seed. We need to
    # pre-bind the attention_mask / position_ids onto ref *after* it's been
    # reinstantiated inside test_block_correctness. The helper runs the
    # reference as `reference_block(reference_inputs[0][0])`, so we monkeypatch
    # _HFLayerWithRope.__init__ to attach a default context automatically.
    _orig_init = _HFLayerWithRope.__init__

    def _patched_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        dummy_mask = torch.zeros(TINY_BS, 1, TINY_P, TINY_P, dtype=torch.float32)
        dummy_pos = _make_position_ids_2d(int(TINY_P ** 0.5), TINY_BS)
        self.set_context(attention_mask=dummy_mask, position_ids=dummy_pos)

    _HFLayerWithRope.__init__ = _patched_init
    test_vision_encoder_layer_matches_hf()
    print("TEST PASSED")
