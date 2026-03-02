#!/usr/bin/env python3
"""
Diagnose vision model weight loading/sharding for Qwen2-VL on Trainium.

This script simulates the exact weight loading path the NxDI framework uses,
then checks for key mismatches, sharding correctness, and reconstruction.

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    cd ~/model-translation/qwen2-vl
    python diagnose_sharding.py
"""

import copy
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

MODEL_PATH = os.path.expanduser("~/models/qwen2-vl-7b")
TP_DEGREE = 8


def load_and_convert_state_dict():
    """Replicate exactly what checkpoint_loader_fn + get_state_dict does."""
    from neuronx_distributed_inference.modules.checkpoint import load_state_dict as nxdi_load_state_dict
    from transformers import AutoConfig

    from modeling_qwen2vl_neuron import (
        NeuronQwen2VLForConditionalGeneration,
        Qwen2VLMultimodalInferenceConfig,
    )
    from neuronx_distributed_inference.models.config import NeuronConfig

    print("=" * 70)
    print("STEP 1: Load HF state dict and convert keys")
    print("=" * 70)

    model_sd = nxdi_load_state_dict(MODEL_PATH)
    print(f"  Loaded {len(model_sd)} keys from HF checkpoint")

    # Strip "model." prefix (same as get_state_dict)
    _PREFIX = "model."
    _NEW_PREFIX = ""
    param_name_list = list(model_sd.keys())
    for param_name in param_name_list:
        updated_param_name = param_name
        if param_name.startswith(_PREFIX):
            updated_param_name = param_name.replace(_PREFIX, _NEW_PREFIX, 1)
        if updated_param_name != param_name:
            model_sd[updated_param_name] = model_sd[param_name]
            del model_sd[param_name]

    print(f"  After prefix strip: {len(model_sd)} keys")
    vision_keys_before = sorted(k for k in model_sd if k.startswith("visual."))
    print(f"  Vision keys (before convert): {len(vision_keys_before)}")
    for k in vision_keys_before[:5]:
        print(f"    {k}: shape={model_sd[k].shape}, dtype={model_sd[k].dtype}")
    if len(vision_keys_before) > 5:
        print(f"    ... and {len(vision_keys_before) - 5} more")

    # Build config (same as build_config in run_multimodal_inference.py)
    text_neuron_cfg = NeuronConfig(
        tp_degree=TP_DEGREE, batch_size=1, max_batch_size=1,
        seq_len=1024, n_positions=1024, n_active_tokens=1024,
        max_context_length=1024, max_length=1024, buckets=[1024],
        torch_dtype=torch.bfloat16, on_device_sampling_config=None,
        cc_pipeline_tiling_factor=2, padding_side="right",
    )
    vision_neuron_cfg = NeuronConfig(
        tp_degree=TP_DEGREE, batch_size=1, max_batch_size=1,
        seq_len=1024, n_positions=1024, n_active_tokens=1024,
        max_context_length=1024, max_length=1024, buckets=[256, 1024],
        torch_dtype=torch.bfloat16, on_device_sampling_config=None,
        cc_pipeline_tiling_factor=2, padding_side="right",
    )
    hf_cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    hf_cfg_dict = hf_cfg.to_dict()
    hf_cfg_dict.pop("torch_dtype", None)
    hf_cfg_dict.pop("dtype", None)
    inf_cfg = Qwen2VLMultimodalInferenceConfig(
        text_neuron_config=text_neuron_cfg,
        vision_neuron_config=vision_neuron_cfg,
        **hf_cfg_dict,
    )

    # Run convert_hf_to_neuron_state_dict
    model_sd = NeuronQwen2VLForConditionalGeneration.convert_hf_to_neuron_state_dict(
        model_sd, inf_cfg
    )
    print(f"  After convert_hf_to_neuron_state_dict: {len(model_sd)} keys")

    vision_keys_after = sorted(k for k in model_sd if not k.startswith("layers.") and
                               not k.startswith("embed_tokens") and
                               not k.startswith("norm.") and
                               not k.startswith("lm_head") and
                               not k.startswith("rank_util"))
    print(f"  Potential vision keys (after convert): {len(vision_keys_after)}")
    for k in vision_keys_after[:10]:
        print(f"    {k}: shape={model_sd[k].shape}, dtype={model_sd[k].dtype}")
    if len(vision_keys_after) > 10:
        print(f"    ... and {len(vision_keys_after) - 10} more")

    # Cast to bfloat16 (same as _cast_helper in checkpoint_loader_fn)
    for name, param in model_sd.items():
        if torch.is_floating_point(param) and param.dtype not in [torch.float8_e4m3fn]:
            if param.dtype != torch.bfloat16:
                model_sd[name] = param.to(torch.bfloat16)

    return model_sd, inf_cfg


def create_vision_model_and_check_keys(full_state_dict, inf_cfg):
    """Create vision model in mock distributed mode and check key matching."""
    from neuronx_distributed.parallel_layers import parallel_state
    from neuronx_distributed.trace.mock_torchdist import mock_distributed

    from modeling_qwen2vl_vision_neuron import NeuronQwen2VLVisionModel

    print("\n" + "=" * 70)
    print("STEP 2: Create vision model (mock TP=8) and check key matching")
    print("=" * 70)

    # Build the config that the vision encoder model instance receives
    new_config = copy.deepcopy(inf_cfg)
    new_config.neuron_config = copy.deepcopy(new_config.vision_config.neuron_config)

    with mock_distributed(world_size=TP_DEGREE * 1):
        torch.distributed.init_process_group(backend="xla", rank=0, world_size=TP_DEGREE)
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=TP_DEGREE,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            skip_collective_init=True,
        )

        try:
            model = NeuronQwen2VLVisionModel(new_config)
            model.eval()
            model_keys = set(model.state_dict().keys())
            print(f"  Vision model has {len(model_keys)} parameters")

            # Check which checkpoint keys match
            ckpt_vision_keys = set()
            ckpt_text_keys = set()
            ckpt_other_keys = set()
            for k in full_state_dict:
                if k in model_keys:
                    ckpt_vision_keys.add(k)
                elif k.startswith("layers.") or k.startswith("embed_tokens") or \
                     k.startswith("norm.") or k.startswith("lm_head") or \
                     "rank_util" in k:
                    ckpt_text_keys.add(k)
                else:
                    ckpt_other_keys.add(k)

            missing_in_ckpt = model_keys - set(full_state_dict.keys())
            extra_in_ckpt = ckpt_vision_keys - model_keys

            print(f"  Checkpoint vision keys matched: {len(ckpt_vision_keys)}")
            print(f"  Checkpoint text keys (will be removed): {len(ckpt_text_keys)}")
            print(f"  Checkpoint other keys: {len(ckpt_other_keys)}")
            if ckpt_other_keys:
                for k in sorted(ckpt_other_keys)[:10]:
                    print(f"    {k}: shape={full_state_dict[k].shape}")

            if missing_in_ckpt:
                print(f"\n  *** MISSING in checkpoint ({len(missing_in_ckpt)} keys) ***")
                for k in sorted(missing_in_ckpt):
                    p = dict(model.named_parameters()).get(k)
                    shape = p.shape if p is not None else "?"
                    print(f"    {k}: expected shape={shape}")
            else:
                print(f"\n  All {len(model_keys)} model keys found in checkpoint")

            if extra_in_ckpt:
                print(f"\n  *** EXTRA vision keys in checkpoint ({len(extra_in_ckpt)}) ***")
                for k in sorted(extra_in_ckpt):
                    print(f"    {k}")

            # Check shapes
            shape_mismatches = []
            for k in sorted(ckpt_vision_keys):
                ckpt_shape = full_state_dict[k].shape
                model_param = dict(model.named_parameters())[k]
                model_shape = model_param.shape
                # Model param shape is per-rank (sharded), ckpt shape is full
                # For ColumnParallelLinear: ckpt dim0 = model dim0 * tp_degree
                # For RowParallelLinear: ckpt dim1 = model dim1 * tp_degree
                has_tp = hasattr(model_param, "tensor_model_parallel") and model_param.tensor_model_parallel
                if has_tp:
                    pdim = model_param.partition_dim
                    expected_ckpt_dim = model_shape[pdim] * TP_DEGREE
                    if ckpt_shape[pdim] != expected_ckpt_dim:
                        shape_mismatches.append((k, ckpt_shape, model_shape, pdim, expected_ckpt_dim))
                        print(f"  *** SHAPE MISMATCH: {k}: ckpt={ckpt_shape}, model={model_shape}, "
                              f"partition_dim={pdim}, expected ckpt dim[{pdim}]={expected_ckpt_dim}")

            if not shape_mismatches:
                print(f"  All shape checks passed for {len(ckpt_vision_keys)} keys")

            # Print parameter attributes for attention layers
            print("\n  Parameter attributes for blocks.0.attn:")
            for name, param in model.named_parameters():
                if name.startswith("blocks.0.attn."):
                    attrs = []
                    for a in ["tensor_model_parallel", "partition_dim", "partition_stride",
                              "num_partitions", "fused_qkv"]:
                        if hasattr(param, a):
                            attrs.append(f"{a}={getattr(param, a)}")
                    print(f"    {name}: shape={param.shape}, {', '.join(attrs)}")

            return model, model_keys
        finally:
            parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()


def shard_and_verify(full_state_dict, inf_cfg):
    """Shard checkpoint for all ranks and verify reconstruction."""
    from neuronx_distributed.parallel_layers import parallel_state
    from neuronx_distributed.trace.mock_torchdist import mock_distributed
    from neuronx_distributed.utils.model_utils import init_on_device
    from neuronx_distributed.trace.trace import (
        preprocess_checkpoint,
        shard_children,
    )

    from modeling_qwen2vl_vision_neuron import NeuronQwen2VLVisionModel

    print("\n" + "=" * 70)
    print("STEP 3: Shard checkpoint and verify reconstruction")
    print("=" * 70)

    new_config = copy.deepcopy(inf_cfg)
    new_config.neuron_config = copy.deepcopy(new_config.vision_config.neuron_config)

    with mock_distributed(world_size=TP_DEGREE), init_on_device(
        torch.device("meta"), force_custom_init_on_device=True
    ):
        torch.distributed.init_process_group(backend="xla", rank=0, world_size=TP_DEGREE)
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=TP_DEGREE,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            skip_collective_init=True,
        )

        try:
            model = NeuronQwen2VLVisionModel(new_config)
            model.eval()
            model_keys = set(model.state_dict().keys())

            # Make a working copy of the checkpoint
            base_ckpt = {k: v.clone() for k, v in full_state_dict.items()}

            # Run preprocess_checkpoint (same as framework)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                preprocess_checkpoint(model, base_ckpt)
                for warning in w:
                    msg = str(warning.message)
                    if "Removing redundant keys" in msg:
                        print(f"  {msg[:200]}...")

            print(f"  After preprocess: {len(base_ckpt)} keys remain")

            # Save original (pre-shard) values for comparison
            originals = {k: v.clone() for k, v in base_ckpt.items()}

            # Shard for each rank
            sharded = []
            for rank in range(TP_DEGREE):
                ckpt_copy = {k: v.clone() for k, v in originals.items()}
                shard_children(model, ckpt_copy, "", torch.bfloat16, rank, TP_DEGREE)
                sharded.append(ckpt_copy)

            print(f"  Sharded into {TP_DEGREE} rank checkpoints")

            # Verify: for each parameter, reconstruct from shards and compare
            print("\n  Reconstruction check:")
            max_errors = {}
            for key in sorted(originals.keys()):
                original = originals[key]
                param = dict(model.named_parameters())[key]
                has_tp = hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel

                if has_tp:
                    pdim = param.partition_dim
                    # Gather shards along partition dimension
                    shard_list = [sharded[r][key] for r in range(TP_DEGREE)]
                    reconstructed = torch.cat(shard_list, dim=pdim)
                else:
                    reconstructed = sharded[0][key]

                if reconstructed.shape != original.shape:
                    print(f"    *** {key}: SHAPE MISMATCH after reconstruction: "
                          f"reconstructed={reconstructed.shape} vs original={original.shape}")
                    continue

                diff = (reconstructed.float() - original.float()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                max_errors[key] = max_diff

                if max_diff > 1e-6:
                    print(f"    *** {key}: RECONSTRUCTION ERROR max={max_diff:.6e}, mean={mean_diff:.6e}")

            all_good = all(v < 1e-6 for v in max_errors.values())
            if all_good:
                print(f"    All {len(max_errors)} sharded parameters reconstruct perfectly (max_diff < 1e-6)")
            else:
                bad = sum(1 for v in max_errors.values() if v >= 1e-6)
                print(f"    {bad}/{len(max_errors)} parameters have reconstruction errors!")

            return sharded, originals, model

        finally:
            parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()


def test_forward_cpu_with_loaded_weights(full_state_dict, inf_cfg):
    """Load weights into vision model (TP=1 CPU mode) and run forward."""
    from modeling_qwen2vl_vision_neuron import NeuronQwen2VLVisionModel
    from modeling_qwen2vl_vision_neuron import Qwen2VLVisionModelWrapper

    print("\n" + "=" * 70)
    print("STEP 4: Test forward pass on CPU with loaded weights")
    print("=" * 70)

    # Create a TP=1 config for CPU testing
    from neuronx_distributed_inference.models.config import NeuronConfig
    from transformers import AutoConfig

    cpu_neuron_cfg = NeuronConfig(
        tp_degree=1, batch_size=1, max_batch_size=1,
        seq_len=1024, n_positions=1024, n_active_tokens=1024,
        max_context_length=1024, max_length=1024, buckets=[256, 1024],
        torch_dtype=torch.bfloat16, on_device_sampling_config=None,
        cc_pipeline_tiling_factor=2, padding_side="right",
        on_cpu=True,
    )

    hf_cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    hf_cfg_dict = hf_cfg.to_dict()
    hf_cfg_dict.pop("torch_dtype", None)
    hf_cfg_dict.pop("dtype", None)

    # Build a config with TP=1
    from modeling_qwen2vl_neuron import Qwen2VLMultimodalInferenceConfig
    cpu_inf_cfg = Qwen2VLMultimodalInferenceConfig(
        text_neuron_config=cpu_neuron_cfg,
        vision_neuron_config=cpu_neuron_cfg,
        **hf_cfg_dict,
    )

    # Create model in non-parallel mode (no mock_distributed)
    model = NeuronQwen2VLVisionModel(cpu_inf_cfg)
    model.eval()

    # Filter to vision keys only
    vision_sd = {}
    model_keys = set(model.state_dict().keys())
    for k in model_keys:
        if k in full_state_dict:
            vision_sd[k] = full_state_dict[k].clone()
        else:
            print(f"  *** Key {k} not in checkpoint!")

    missing = model_keys - set(vision_sd.keys())
    if missing:
        print(f"  *** {len(missing)} keys missing from checkpoint")
    else:
        print(f"  All {len(model_keys)} keys found, loading state dict...")

    result = model.load_state_dict(vision_sd, strict=True)
    print(f"  load_state_dict result: {result}")

    # Prepare a test input
    from transformers import Qwen2VLProcessor
    from qwen_vl_utils import process_vision_info

    processor = Qwen2VLProcessor.from_pretrained(MODEL_PATH)
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = 200704
        processor.image_processor.min_pixels = 3136

    image_path = str(SCRIPT_DIR / "puppy.jpg")
    if not os.path.exists(image_path):
        image_path = str(SCRIPT_DIR / "smol_puppy.jpg")

    messages = [{"role": "user", "content": [
        {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
        {"type": "text", "text": "Describe this image."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt")

    pixel_values = inputs["pixel_values"]
    grid_thw = inputs["image_grid_thw"]
    print(f"  pixel_values: shape={pixel_values.shape}, dtype={pixel_values.dtype}")
    print(f"  grid_thw: {grid_thw}")

    # Wrap in the model wrapper to get patchified inputs
    wrapper = Qwen2VLVisionModelWrapper.__new__(Qwen2VLVisionModelWrapper)
    wrapper.config = cpu_inf_cfg
    patch_embeds, attention_mask, cos, sin = wrapper.patchify(pixel_values, grid_thw)
    print(f"  patch_embeds: shape={patch_embeds.shape}")
    print(f"  attention_mask: shape={attention_mask.shape}")
    print(f"  cos: shape={cos.shape}")
    print(f"  sin: shape={sin.shape}")

    with torch.no_grad():
        output = model(patch_embeds.to(torch.bfloat16), attention_mask.to(torch.int32),
                       cos.to(torch.bfloat16), sin.to(torch.bfloat16))

    print(f"\n  Output: shape={output.shape}, dtype={output.dtype}")
    print(f"  Output stats: mean={output.float().mean():.6f}, std={output.float().std():.6f}")
    print(f"  Expected stats: mean~=-0.04, std~=1.37 (from HF reference)")

    return output


def compare_with_hf_reference(inf_cfg):
    """Run the HF reference model and compare output."""
    print("\n" + "=" * 70)
    print("STEP 5: Compare with HF reference model")
    print("=" * 70)

    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    from qwen_vl_utils import process_vision_info

    processor = Qwen2VLProcessor.from_pretrained(MODEL_PATH)
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = 200704
        processor.image_processor.min_pixels = 3136

    image_path = str(SCRIPT_DIR / "puppy.jpg")
    if not os.path.exists(image_path):
        image_path = str(SCRIPT_DIR / "smol_puppy.jpg")

    messages = [{"role": "user", "content": [
        {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
        {"type": "text", "text": "Describe this image."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt")

    hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    hf_model.eval()

    with torch.no_grad():
        pixel_values = inputs["pixel_values"].to(torch.bfloat16)
        grid_thw = inputs["image_grid_thw"]
        hf_vision_out = hf_model.visual(pixel_values, grid_thw=grid_thw)

    print(f"  HF vision output: shape={hf_vision_out.shape}, dtype={hf_vision_out.dtype}")
    print(f"  HF vision stats: mean={hf_vision_out.float().mean():.6f}, "
          f"std={hf_vision_out.float().std():.6f}")

    return hf_vision_out


def main():
    print("Qwen2-VL Vision Model Weight Sharding Diagnostic")
    print("=" * 70)
    print(f"  Model: {MODEL_PATH}")
    print(f"  TP degree: {TP_DEGREE}")
    print()

    # Step 1: Load and convert
    full_sd, inf_cfg = load_and_convert_state_dict()

    # Step 2: Check key matching
    model, model_keys = create_vision_model_and_check_keys(full_sd, inf_cfg)

    # Step 3: Shard and verify reconstruction
    sharded, originals, shard_model = shard_and_verify(full_sd, inf_cfg)

    # Step 4: Forward on CPU with loaded weights
    try:
        cpu_output = test_forward_cpu_with_loaded_weights(full_sd, inf_cfg)
    except Exception as e:
        print(f"  *** CPU forward test failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 5: HF reference comparison
    try:
        hf_output = compare_with_hf_reference(inf_cfg)
    except Exception as e:
        print(f"  *** HF reference test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
