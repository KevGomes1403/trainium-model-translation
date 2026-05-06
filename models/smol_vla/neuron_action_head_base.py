"""
neuron_action_head_base.py

Base classes for porting VLA action heads to AWS Trainium.

Architecture:
    NeuronDenoisingConfig
        — minimal config shim satisfying ModelWrapper.__init__()
        — without LLM-specific fields (KV cache, sequence buckets, etc.)
        — use this when constructing NeuronDenoisingWrapper

    NeuronDenoisingWrapper(ModelWrapper)
        — owns the single compiled-step forward (one NEFF)
        — ModelWrapper inheritance gives compile/load/shard lifecycle
        — subclass and implement forward() per model
        — requires NeuronDenoisingConfig or compatible config

    NeuronActionHeadBase(nn.Module)
        — owns the denoising loop and lifecycle coordination
        — does NOT inherit NeuronApplicationBase (LLM-oriented, incompatible)
        — scalpels compile/load/warmup/checkpoint from NeuronApplicationBase
        — everything LLM-specific dropped (KV cache, LoRA, medusa, speculation,
          continuous batching, snapshot hooks, quantization, CPU inference mode)

    ConditioningContract
        — shape agreement dataclass between VLM subgraph and action head

Model-specific subclasses (written by the port agent, not here):
    NeuronGrootActionHead(NeuronActionHeadBase)
    NeuronSmolVLAActionHead(NeuronActionHeadBase)

NOTE: Verify all import paths against the installed NxDI version on your
Trainium instance before running. Paths match NxDI venv layout as of
neuronx_distributed_inference 2.28.

Before using, read ModelWrapper.__init__() source on the Trainium instance
to confirm NeuronDenoisingConfig satisfies its requirements:
    python3 -c "
    import inspect
    from neuronx_distributed_inference.models.model_wrapper import ModelWrapper
    print(inspect.getsource(ModelWrapper.__init__))
    "
"""

from __future__ import annotations

import abc
import logging
import os
import time
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
# from cross_attention_kernel import cross_attention_kernel, get_tile_size

import torch
import torch.nn as nn

logger = logging.getLogger("NeuronActionHead")

COMPILED_MODEL_FILE_NAME = "model.pt"

# ---------------------------------------------------------------------------
# NxDI imports with CPU fallback
# ---------------------------------------------------------------------------

try:
    from neuronx_distributed_inference.models.model_wrapper import ModelWrapper
    from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
    from neuronx_distributed.parallel_layers.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from neuronx_distributed.trace.model_builder import ModelBuilder
    from neuronx_distributed_inference.modules.checkpoint import load_state_dict
    from safetensors.torch import load_file
    _NEURON_AVAILABLE = True
except ImportError:
    # CPU fallback — allows unit testing without Neuron hardware
    _NEURON_AVAILABLE = False

    class ModelWrapper(nn.Module):
        """Shim so NeuronDenoisingWrapper.__init__ can call super().__init__(config=..., model_cls=...)."""
        def __init__(self, config=None, model_cls=None, **kwargs):
            super().__init__()
            self.model = None

    InferenceConfig = object
    NeuronConfig = object
    ModelBuilder = None
    load_state_dict = None
    load_file = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_path(path: str) -> str:
    """Normalize path separators and ensure trailing slash."""
    return os.path.join(os.path.normpath(path), "")


def _is_compiled(model_path: str) -> bool:
    return os.path.isfile(model_path + COMPILED_MODEL_FILE_NAME)


# ---------------------------------------------------------------------------
# ConditioningContract
# ---------------------------------------------------------------------------

class ConditioningContract:
    """
    Defines the static shape agreement between the VLM subgraph and the
    action head subgraph. Must be identical on both sides at compile time.

    Attributes:
        num_conditioning_tokens: Number of VLM output tokens passed to the
            denoiser as cross-attention KV. Derived as:
                num_text_tokens + num_vision_tokens_for_bucket
            Must match the VLM subgraph output sequence length exactly.
        conditioning_hidden_size: Hidden dimension of conditioning tokens.
            Must match VLM output hidden size.
        dtype: Expected dtype of conditioning tokens (typically torch.bfloat16).
    """

    def __init__(
        self,
        num_conditioning_tokens: int,
        conditioning_hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_conditioning_tokens = num_conditioning_tokens
        self.conditioning_hidden_size = conditioning_hidden_size
        self.dtype = dtype

    def validate(self, conditioning_tokens: torch.Tensor) -> None:
        """
        Assert that a conditioning tensor matches this contract.
        Call this at the start of generate_actions() to catch shape
        mismatches before they produce cryptic errors inside the loop.
        """
        assert conditioning_tokens.shape[1] == self.num_conditioning_tokens, (
            f"Conditioning token count mismatch: expected "
            f"{self.num_conditioning_tokens}, got {conditioning_tokens.shape[1]}. "
            f"Check that the VLM vision bucket matches the denoiser bucket."
        )
        assert conditioning_tokens.shape[2] == self.conditioning_hidden_size, (
            f"Conditioning hidden size mismatch: expected "
            f"{self.conditioning_hidden_size}, got {conditioning_tokens.shape[2]}."
        )
        assert conditioning_tokens.dtype == self.dtype, (
            f"Conditioning dtype mismatch: expected {self.dtype}, "
            f"got {conditioning_tokens.dtype}."
        )


# ---------------------------------------------------------------------------
# NeuronDenoisingConfig — minimal config shim for ModelWrapper.__init__()
# ---------------------------------------------------------------------------

class NeuronDenoisingConfig:
    """
    Minimal config that satisfies ModelWrapper.__init__() without requiring
    LLM-specific fields (KV cache size, sequence length buckets, vocab size, etc.)

    Read ModelWrapper.__init__() source on the Trainium instance to verify
    these fields are sufficient. Add any missing fields as None/False stubs.

    Usage:
        config = NeuronDenoisingConfig(
            batch_size=1,
            tp_degree=8,
            action_chunk_size=50,
            action_dim=6,
            num_conditioning_tokens=512,
            conditioning_hidden_size=960,
            timestep_embed_dim=256,
        )
        wrapper = MyDenoisingWrapper(config)
    """

    def __init__(
        self,
        batch_size: int,
        tp_degree: int,
        action_chunk_size: int,
        action_dim: int,
        num_conditioning_tokens: int,
        conditioning_hidden_size: int,
        timestep_embed_dim: int,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        # Fields ModelWrapper.__init__() reads from config.neuron_config
        # LLM-specific fields stubbed as None/False/0
        self.neuron_config = SimpleNamespace(
            # Core — required
            torch_dtype=torch_dtype,
            tp_degree=tp_degree,
            batch_size=batch_size,
            # Compiler tuning
            cc_pipeline_tiling_factor=1,
            logical_nc_config=1,
            # LLM features — all disabled
            is_block_kv_layout=False,
            is_prefix_caching=False,
            is_medusa=False,
            token_generation_batches=None,
            async_mode=False,
            scratchpad_page_size=None,
            attn_block_tkg_nki_kernel_enabled=False,
            enable_long_context_mode=False,
            layer_boundary_markers=False,
            dma_order_config=None,
            enable_spill_reload_dge=False,
            target=None,
            quantized=False,
            quantization_dtype=None,
            kv_cache_quant=False,
            quantized_mlp_kernel_enabled=False,
            activation_quantization_type=None,
            enable_output_completion_notifications=False,
            # Weight loading
            save_sharded_checkpoint=True,
            start_rank_id=0,
            local_ranks_size=tp_degree,
            cast_type="config",
            # Parallelism
            pp_degree=1,
            ep_degree=1,
            world_size=tp_degree,
        )
        # ModelWrapper also reads config.pad_token_id
        self.pad_token_id = 0

        # Action head specific — exposed as top-level config attributes
        # so NeuronDenoisingWrapper.input_generator() can read them
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        self.num_conditioning_tokens = num_conditioning_tokens
        self.conditioning_hidden_size = conditioning_hidden_size
        self.timestep_embed_dim = timestep_embed_dim


# ---------------------------------------------------------------------------
# NeuronDenoisingWrapper — base for the compiled single-step forward
# ---------------------------------------------------------------------------

class NeuronDenoisingWrapper(ModelWrapper):
    """
    Base wrapper for the single-step denoiser forward. This is what gets
    compiled to a NEFF. The N-step denoising loop lives in
    NeuronActionHeadBase.generate_actions(), NOT here.

    Inherits ModelWrapper for compile/load/shard lifecycle. Pass a
    NeuronDenoisingConfig instance as config — it satisfies ModelWrapper.__init__()
    without requiring LLM-specific fields.

    Subclass this and implement forward() for each model.

    Compiled graph inputs (all shapes must be STATIC at compile time):
        noisy_actions:        [B, action_chunk_size, action_dim]         BF16
        conditioning_tokens:  [B, num_conditioning_tokens, hidden_size]  BF16
        timestep_embedding:   [B, timestep_embed_dim]                    BF16
        attention_mask:       [B, 1, action_chunk_size,
                               num_conditioning_tokens]                  INT32

    Compiled graph output:
        denoised_actions:     [B, action_chunk_size, action_dim]         BF16

    IMPORTANT: Do NOT pass raw timestep scalars or noise schedule arrays into
    this wrapper. The caller computes timestep_embedding on CPU (sinusoidal
    encoding + linear projection) and passes only the projected embedding.
    This avoids dynamic scalar-to-embedding computation inside the traced graph.

    IMPORTANT: Do NOT include the denoising loop here. Any Python-level
    iteration over timesteps must live in generate_actions().

    IMPORTANT: Any subclass that overrides load_state_dict must accept and
    forward **kwargs. torch_neuronx internally calls load_state_dict(...,
    assign=True) during tracing. Not forwarding **kwargs causes a TypeError
    that, if caught by a bare except, produces a silent missing-NEFF failure.

    IMPORTANT: Never call torch.arange(), torch.linspace(), torch.ones(),
    torch.zeros(), or math over these inside forward(). Pre-compute all
    fixed tensors in __init__() as register_buffer(). Dynamic constant
    creation in forward() causes [Errno 36] File name too long on deep
    models and silent performance regression on shallow ones.
    """

    def __init__(self, config: NeuronDenoisingConfig):
        super().__init__(config=config, model_cls=type(self))
        self.config = config

    @abc.abstractmethod
    def forward(
        self,
        noisy_actions: torch.Tensor,
        conditioning_tokens: torch.Tensor,
        timestep_embedding: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single denoising step forward pass. Implement in model-specific subclass.

        Args:
            noisy_actions:       [B, action_chunk_size, action_dim]        BF16
            conditioning_tokens: [B, num_conditioning_tokens, hidden_size] BF16
            timestep_embedding:  [B, timestep_embed_dim]                   BF16
            attention_mask:      [B, 1, action_chunk_size,
                                  num_conditioning_tokens]                  INT32

        Returns:
            denoised_actions:    [B, action_chunk_size, action_dim]        BF16
        """
        raise NotImplementedError

    def input_generator(self) -> List[Tuple[torch.Tensor, ...]]:
        """
        Returns example inputs for XLA tracing. Override if defaults are wrong.
        Shapes must exactly match forward() signature.

        Reads action_chunk_size, action_dim, num_conditioning_tokens,
        conditioning_hidden_size, and timestep_embed_dim from config,
        and batch_size from config.neuron_config.batch_size.
        """
        cfg = self.config
        B = cfg.neuron_config.batch_size
        return [(
            torch.zeros(B, cfg.action_chunk_size, cfg.action_dim,
                        dtype=torch.bfloat16),
            torch.zeros(B, cfg.num_conditioning_tokens,
                        cfg.conditioning_hidden_size, dtype=torch.bfloat16),
            torch.zeros(B, cfg.timestep_embed_dim, dtype=torch.bfloat16),
            torch.zeros(B, 1, cfg.action_chunk_size,
                        cfg.num_conditioning_tokens, dtype=torch.int32),
        )]

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """
        Forwards **kwargs to nn.Module.load_state_dict.

        REQUIRED: torch_neuronx internally calls load_state_dict(..., assign=True)
        during tracing. Not forwarding **kwargs causes TypeError. Any subclass
        override must also accept and forward **kwargs.
        """
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def is_neuron(self) -> bool:
        """True if the compiled NEFF is loaded on Neuron hardware."""
        return self.model is not None and isinstance(self.model, torch.jit.ScriptModule)


# ---------------------------------------------------------------------------
# NeuronActionHeadBase — owns the lifecycle and denoising loop
# ---------------------------------------------------------------------------

class NeuronActionHeadBase(nn.Module):
    """
    Abstract base class for VLA action heads on Trainium.

    Inherits nn.Module, NOT NeuronApplicationBase. NeuronApplicationBase is
    LLM-oriented (KV cache, token generation, continuous batching, LoRA,
    medusa, speculative decoding) — none of which applies to denoising
    inference. The useful parts are scalpeled directly into this class:
        compile_denoiser() — compiles via ModelBuilder (proper TP sharding)
        load()             — loads NEFF + weights to Neuron device
        load_weights()     — handles pre-sharded and on-load sharding
        warmup()           — runs example inputs once after load
        checkpoint_loader_fn() — loads HF state dict with dtype casting
        get_state_dict()   — loads and converts HF checkpoint

    Owns:
        compile_denoiser() — compiles the single-step forward to a NEFF
        generate_actions() — runs the N-step denoising loop on CPU
        get_conditioning_contract() — shape contract with VLM subgraph

    Does NOT own:
        The VLM subgraph (separate compiled graph)
        The noise schedule (computed on CPU in generate_actions)
        Weight loading for the VLM backbone
    """

    _STATE_DICT_MODEL_PREFIX = "model."
    _NEW_STATE_DICT_MODEL_PREFIX = ""

    def __init__(self, model_path: str, config: NeuronDenoisingConfig):
        """
        Args:
            model_path: Path to HF checkpoint directory. Used for weight loading.
            config: NeuronDenoisingConfig instance. Must expose neuron_config
                    with tp_degree, batch_size, torch_dtype, save_sharded_checkpoint,
                    start_rank_id, local_ranks_size. Also must expose top-level:
                    action_chunk_size, action_dim, num_conditioning_tokens,
                    conditioning_hidden_size, timestep_embed_dim.
        """
        super().__init__()
        self.model_path = _normalize_path(model_path)
        self.config = config
        self.neuron_config = config.neuron_config
        self.denoising_wrapper: Optional[NeuronDenoisingWrapper] = None
        self.attention_mask: Optional[torch.Tensor] = None
        self.traced_model = None
        self.is_compiled = _is_compiled(self.model_path)
        self.is_loaded_to_neuron = False
        self._builder = None

    # ------------------------------------------------------------------
    # Abstract methods — implement in model-specific subclass
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_conditioning_contract(self) -> ConditioningContract:
        """
        Return the ConditioningContract this action head expects from the VLM.

        Example:
            return ConditioningContract(
                num_conditioning_tokens=512,
                conditioning_hidden_size=2048,
            )
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _build_denoising_wrapper(self) -> NeuronDenoisingWrapper:
        """Instantiate and return the model-specific NeuronDenoisingWrapper."""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_timestep_sequence(self, num_steps: int) -> List[float]:
        """
        Return timestep values for the denoising loop as Python floats.
        Flow matching: return [1.0 - i / num_steps for i in range(num_steps)]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _embed_timestep(self, t: float) -> torch.Tensor:
        """
        Compute timestep embedding on CPU for a single float timestep.
        Returns [B, timestep_embed_dim] BF16.
        Pre-compute frequency tables in __init__ as register_buffer().
        Never call torch.arange() or torch.linspace() here.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _build_attention_mask(self) -> torch.Tensor:
        """
        Build static cross-attention mask once at compile time.
        Returns [B, 1, action_chunk_size, num_conditioning_tokens] INT32.
        Never build this inside forward() — dynamic mask creation degrades NEFF.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # NxDI infrastructure scalpeled from NeuronApplicationBase
    # Dropped: KV cache, token generation, continuous batching, LoRA,
    #          medusa, speculative decoding, snapshot hooks, quantization,
    #          CPU inference mode, modular flow optimization.
    # ------------------------------------------------------------------

    def is_neuron(self) -> bool:
        """True if the compiled NEFF is loaded on Neuron hardware."""
        return (
            self.traced_model is not None
            and isinstance(self.traced_model, torch.jit.ScriptModule)
        )

    def get_builder(self):
        """
        Build and cache a ModelBuilder for this action head.
        Scalpeled from NeuronApplicationBase.get_builder() — LLM options removed.

        ModelBuilder initializes parallel_state so ColumnParallelLinear and
        RowParallelLinear use the parallel path, not the nn.Linear fallback.
        This is the critical difference from raw torch_neuronx.trace().
        """
        if not _NEURON_AVAILABLE:
            raise RuntimeError(
                "ModelBuilder unavailable on CPU. "
                "get_builder() can only be called on a Trainium instance."
            )
        if self._builder is None:
            assert self.denoising_wrapper is not None, (
                "Call _build_denoising_wrapper() before get_builder()."
            )
            base_compile_work_dir = os.environ.get(
                "BASE_COMPILE_WORK_DIR", "/tmp/nxd_action_head/"
            )
            self._builder = ModelBuilder(
                router=None,
                tp_degree=self.neuron_config.tp_degree,
                pp_degree=getattr(self.neuron_config, "pp_degree", 1),
                ep_degree=getattr(self.neuron_config, "ep_degree", 1),
                world_size=getattr(
                    self.neuron_config, "world_size", self.neuron_config.tp_degree
                ),
                start_rank_id=getattr(self.neuron_config, "start_rank_id", 0),
                local_ranks_size=getattr(
                    self.neuron_config, "local_ranks_size", self.neuron_config.tp_degree
                ),
                checkpoint_loader=self.checkpoint_loader_fn,
                compiler_workdir=base_compile_work_dir,
            )
            self.denoising_wrapper.tag = getattr(
                self.denoising_wrapper, "tag", "action_head"
            )
            self._builder.add(
                key=self.denoising_wrapper.tag,
                model_instance=self.denoising_wrapper,
                example_inputs=self.denoising_wrapper.input_generator(),
                compiler_args=self.get_compiler_args(),
            )
        return self._builder

    def get_compiler_args(self) -> str:
        """
        Base Neuron compiler arguments for the action head.
        Scalpeled from ModelWrapper.__init__() default compiler_args.
        Override in subclass to add model-specific flags.

        Note: --enable-mixed-precision-accumulation forces float32 accumulators
        and adds latency. Do not add it without benchmarking the cost.
        """
        return (
            "--auto-cast=none "
            "-O1 "
            "--tensorizer-options='"
            "--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=1'"
        )

    def compile_denoiser(self, save_path: str) -> None:
        """
        Build and compile the denoising wrapper to a NEFF.

        Uses ModelBuilder — parallel_state is initialized by the builder so
        ColumnParallelLinear/RowParallelLinear use the parallel path, not
        the nn.Linear fallback. Do NOT use raw torch_neuronx.trace() here.

        IMPORTANT: Do NOT wrap this in try/except Exception: pass.
        A swallowed exception produces a missing NEFF with no error message.

        On CPU (_NEURON_AVAILABLE=False): builds wrapper and mask but does
        not compile. Allows unit testing the interface without hardware.
        """
        save_path = _normalize_path(save_path)
        self.denoising_wrapper = self._build_denoising_wrapper()
        self.attention_mask = self._build_attention_mask()

        if not _NEURON_AVAILABLE:
            logger.info("CPU mode: wrapper built but not compiled.")
            return

        self.pre_compile_validate()
        os.makedirs(save_path, exist_ok=True)

        # Compile via ModelBuilder — no try/except, failures must propagate
        traced = self.get_builder().trace(initialize_model_weights=False)
        torch.jit.save(traced, save_path + COMPILED_MODEL_FILE_NAME)
        del traced

        self._shard_weights(save_path)
        self.is_compiled = True
        logger.info(f"Compiled to {save_path}")

    def _shard_weights(self, compiled_model_path: str) -> None:
        """
        Shard and save weights after compilation.
        Scalpeled from NeuronApplicationBase.shard_weights() — LoRA removed.
        """
        compiled_model_path = _normalize_path(compiled_model_path)
        if getattr(self.neuron_config, "save_sharded_checkpoint", True):
            sharded_dir = os.path.join(compiled_model_path, "weights/")
            logger.info(f"Pre-sharding to {sharded_dir}")
            self.get_builder().shard_checkpoint(serialize_path=sharded_dir)
        else:
            logger.info("Skipping pre-sharding. Will shard at load time.")

    def load(self, compiled_model_path: str, skip_warmup: bool = False) -> None:
        """
        Load the compiled NEFF and weights to Neuron device.
        Scalpeled from NeuronApplicationBase.load() — LLM options removed.
        """
        compiled_model_path = _normalize_path(compiled_model_path)
        self.traced_model = torch.jit.load(
            compiled_model_path + COMPILED_MODEL_FILE_NAME
        )
        self.load_weights(compiled_model_path)

        if self.neuron_config.torch_dtype != torch.float32:
            self.to(self.neuron_config.torch_dtype)

        if self.denoising_wrapper is not None:
            self.denoising_wrapper.model = self.traced_model

        self.is_loaded_to_neuron = True

        if not skip_warmup:
            self.warmup()

    def load_weights(self, compiled_model_path: str) -> None:
        """
        Load sharded or unsharded weights to Neuron device.
        Scalpeled from NeuronApplicationBase.load_weights() — LoRA removed.
        """
        compiled_model_path = _normalize_path(compiled_model_path)
        if self.traced_model is None:
            raise ValueError("Call load() before load_weights().")

        start_rank_id = getattr(self.neuron_config, "start_rank_id", 0)
        local_ranks_size = getattr(
            self.neuron_config, "local_ranks_size", self.neuron_config.tp_degree
        )

        weights = []
        start = time.monotonic()

        if getattr(self.neuron_config, "save_sharded_checkpoint", True):
            for rank in range(start_rank_id, start_rank_id + local_ranks_size):
                ckpt = load_file(
                    os.path.join(
                        compiled_model_path,
                        f"weights/tp{rank}_sharded_checkpoint.safetensors",
                    )
                )
                weights.append(ckpt)
        else:
            weights = self.get_builder().shard_checkpoint()

        start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32)
        self.traced_model.nxd_model.initialize(weights, start_rank_tensor)
        logger.info(f"Weights loaded in {time.monotonic() - start:.2f}s")

    def warmup(self) -> None:
        """
        Run one forward pass to trigger lazy initialization.
        Scalpeled from NeuronApplicationBase.warmup() — async mode removed.
        """
        logger.info("Warming up...")
        start = time.time()
        if self.denoising_wrapper is not None and self.denoising_wrapper.is_neuron():
            for example in self.denoising_wrapper.input_generator():
                try:
                    self.denoising_wrapper.model.nxd_model.forward(example)
                except RuntimeError as e:
                    logger.warning(f"Warmup RuntimeError (safe to ignore): {e}")
        logger.info(f"Warmup done in {time.time() - start:.2f}s")

    def checkpoint_loader_fn(self, mmap: bool = False) -> dict:
        """
        Load HF state dict with dtype casting.
        Scalpeled from NeuronApplicationBase.checkpoint_loader_fn() —
        fused speculation, medusa, quantization removed.
        """
        model_sd = self.get_state_dict(self.model_path, self.config)
        if (
            self.neuron_config.torch_dtype != torch.float32
            and getattr(self.neuron_config, "cast_type", "config") == "config"
        ):
            for name, param in model_sd.items():
                if (
                    torch.is_floating_point(param)
                    and param.dtype not in [torch.float8_e4m3fn]
                    and not name.endswith("scale")
                    and param.dtype != self.neuron_config.torch_dtype
                ):
                    model_sd[name] = param.to(self.neuron_config.torch_dtype)
        return model_sd

    @classmethod
    def get_state_dict(cls, model_path: str, config) -> dict:
        """
        Load and convert the HF checkpoint state dict.
        Scalpeled from NeuronApplicationBase.get_state_dict() —
        medusa, fused spec, tied weights removed.
        """
        if os.path.isdir(model_path):
            if _NEURON_AVAILABLE and load_state_dict is not None:
                model_sd = load_state_dict(model_path)
            else:
                import glob
                sf_files = glob.glob(os.path.join(model_path, "*.safetensors"))
                pt_files = glob.glob(os.path.join(model_path, "*.bin"))
                model_sd = {}
                if sf_files:
                    from safetensors.torch import load_file as _lf
                    for f in sf_files:
                        model_sd.update(_lf(f))
                elif pt_files:
                    for f in pt_files:
                        model_sd.update(torch.load(f, map_location="cpu"))
                else:
                    raise FileNotFoundError(f"No checkpoint files in {model_path}")
        elif os.path.isfile(model_path):
            model_sd = torch.load(model_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"model_path does not exist: {model_path}")

        # Apply prefix renaming
        for param_name in list(model_sd.keys()):
            updated = param_name
            if param_name.startswith(cls._STATE_DICT_MODEL_PREFIX):
                updated = param_name.replace(
                    cls._STATE_DICT_MODEL_PREFIX,
                    cls._NEW_STATE_DICT_MODEL_PREFIX, 1
                )
            if updated != param_name:
                model_sd[updated] = model_sd.pop(param_name)

        return cls.convert_hf_to_neuron_state_dict(model_sd, config)

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: Dict[str, torch.Tensor],
        config,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert HF checkpoint keys to Neuron model keys.
        Override in model-specific subclass. Implemented in Phase 4.
        """
        return state_dict  # placeholder — implemented in Phase 4

    # ------------------------------------------------------------------
    # Denoising loop
    # ------------------------------------------------------------------

    def pre_compile_validate(self) -> None:
        """
        CPU forward pass before neuronx-cc to catch errors cheaply.
        compile_denoiser() calls this automatically.
        """
        assert self.denoising_wrapper is not None, (
            "Call _build_denoising_wrapper() before pre_compile_validate()"
        )
        example = self.denoising_wrapper.input_generator()[0]
        self.denoising_wrapper.eval()
        with torch.no_grad():
            try:
                output = self.denoising_wrapper(*example)
            except Exception as e:
                raise RuntimeError(
                    f"pre_compile_validate() failed. Fix before compiling.\n{e}"
                ) from e
        assert output is not None, "Denoising wrapper forward() returned None."
        assert not torch.isnan(output).any(), "NaN in CPU forward pass."
        logger.info("pre_compile_validate() passed.")

    def generate_actions(
        self,
        conditioning_tokens: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """
        Run the N-step denoising loop on CPU, calling the compiled NEFF each step.

        Args:
            conditioning_tokens: [B, num_conditioning_tokens, hidden_size] BF16
            num_steps: Python int — never a tensor.

        Returns:
            action_chunk: [B, action_chunk_size, action_dim] BF16
        """
        assert self.denoising_wrapper is not None, (
            "compile_denoiser() must be called before generate_actions()"
        )
        assert isinstance(num_steps, int), (
            f"num_steps must be a Python int, got {type(num_steps)}."
        )

        self.get_conditioning_contract().validate(conditioning_tokens)

        B = conditioning_tokens.shape[0]
        cfg = self.config

        noisy_actions = torch.randn(
            B, cfg.action_chunk_size, cfg.action_dim, dtype=torch.bfloat16
        )
        timesteps = self._get_timestep_sequence(num_steps)

        for t in timesteps:
            timestep_emb = self._embed_timestep(t)
            noisy_actions = self.denoising_wrapper(
                noisy_actions,
                conditioning_tokens,
                timestep_emb,
                self.attention_mask,
            )

        return noisy_actions

    def verify_contract_against_vlm(self, vlm_output: torch.Tensor) -> None:
        """Verify VLM output satisfies conditioning contract. Call in smoke test."""
        self.get_conditioning_contract().validate(vlm_output)