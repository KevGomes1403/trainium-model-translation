#!/usr/bin/env python3
"""
End-to-end multimodal inference for Qwen2-VL on AWS Trainium using NxDI.

Usage:
  # Compile + run (first time):
  python run_multimodal_inference.py \
      --model_path ~/models/qwen2-vl-7b \
      --compiled_model_path ~/models/qwen2-vl-multimodal-compiled \
      --image_path /path/to/image.jpg \
      --prompt "Describe this image."

  # Run with pre-compiled model (skips compilation):
  python run_multimodal_inference.py \
      --model_path ~/models/qwen2-vl-7b \
      --compiled_model_path ~/models/qwen2-vl-multimodal-compiled \
      --image_path /path/to/image.jpg \
      --prompt "What do you see?" \
      --skip_compile

Environment:
  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from modeling_qwen2vl_neuron import (
    NeuronQwen2VLForConditionalGeneration,
    Qwen2VLMultimodalInferenceConfig,
)

logger = logging.getLogger("Neuron")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen2-VL multimodal inference on Trainium"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.expanduser("~/models/qwen2-vl-7b"),
        help="Path to HuggingFace Qwen2-VL checkpoint",
    )
    parser.add_argument(
        "--compiled_model_path",
        type=str,
        default=os.path.expanduser("~/models/qwen2-vl-multimodal-compiled"),
        help="Path for compiled Neuron model artifacts",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to input image (omit for text-only generation)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Text prompt",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--tp_degree", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument(
        "--text_buckets",
        type=int,
        nargs="+",
        default=[1024],
        help="Text context encoding bucket sizes",
    )
    parser.add_argument(
        "--vision_buckets",
        type=int,
        nargs="+",
        default=[256, 1024],
        help="Vision encoder patch sequence bucket sizes",
    )
    parser.add_argument("--skip_compile", action="store_true", default=False)
    parser.add_argument("--compile_only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def build_config(args) -> Qwen2VLMultimodalInferenceConfig:
    """Build the multimodal inference config from HF checkpoint + Neuron params."""
    text_neuron_cfg = NeuronConfig(
        tp_degree=args.tp_degree,
        batch_size=1,
        max_batch_size=1,
        seq_len=args.seq_len,
        n_positions=args.seq_len,
        n_active_tokens=args.seq_len,
        max_context_length=args.seq_len,
        max_length=args.seq_len,
        buckets=args.text_buckets,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=None,
        cc_pipeline_tiling_factor=2,
        padding_side="right",
    )

    vision_neuron_cfg = NeuronConfig(
        tp_degree=args.tp_degree,
        batch_size=1,
        max_batch_size=1,
        seq_len=max(args.vision_buckets),
        n_positions=max(args.vision_buckets),
        n_active_tokens=max(args.vision_buckets),
        max_context_length=max(args.vision_buckets),
        max_length=max(args.vision_buckets),
        buckets=args.vision_buckets,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=None,
        cc_pipeline_tiling_factor=2,
        padding_side="right",
    )

    hf_cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    hf_cfg_dict = hf_cfg.to_dict()
    hf_cfg_dict.pop("torch_dtype", None)
    hf_cfg_dict.pop("dtype", None)

    inf_cfg = Qwen2VLMultimodalInferenceConfig(
        text_neuron_config=text_neuron_cfg,
        vision_neuron_config=vision_neuron_cfg,
        **hf_cfg_dict,
    )
    return inf_cfg


def compile_model(model_path, inf_cfg, compiled_model_path, debug=False):
    """Trace, compile, and shard weights for both text and vision models."""
    logger.info("Instantiating NeuronQwen2VLForConditionalGeneration for compilation...")
    model = NeuronQwen2VLForConditionalGeneration(model_path, inf_cfg)

    logger.info(f"Compiling to {compiled_model_path} ...")
    start = time.monotonic()
    model.compile(compiled_model_path, debug=debug)
    elapsed = time.monotonic() - start
    logger.info(f"Compilation finished in {elapsed:.1f}s")
    return model


def load_model(model_path, inf_cfg, compiled_model_path):
    """Load a pre-compiled model onto Trainium."""
    logger.info("Instantiating NeuronQwen2VLForConditionalGeneration for loading...")
    model = NeuronQwen2VLForConditionalGeneration(model_path, inf_cfg)

    logger.info(f"Loading compiled model from {compiled_model_path} ...")
    start = time.monotonic()
    model.load(compiled_model_path)
    elapsed = time.monotonic() - start
    logger.info(f"Model loaded in {elapsed:.1f}s")
    return model


def prepare_inputs(model_path, image_path, prompt):
    """
    Use Qwen2VLProcessor to prepare multimodal inputs.

    Returns dict with: input_ids, attention_mask, pixel_values, image_grid_thw
    """
    from qwen_vl_utils import process_vision_info
    from transformers import Qwen2VLProcessor

    processor = Qwen2VLProcessor.from_pretrained(model_path)
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = 200704
        processor.image_processor.min_pixels = 3136

    content = []
    if image_path is not None:
        content.append({"type": "image", "image": f"file://{os.path.abspath(image_path)}"})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if image_path is not None:
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[text],
            return_tensors="pt",
        )

    logger.info(f"input_ids shape: {inputs['input_ids'].shape}")
    if "pixel_values" in inputs:
        logger.info(f"pixel_values shape: {inputs['pixel_values'].shape}")
    if "image_grid_thw" in inputs:
        logger.info(f"image_grid_thw: {inputs['image_grid_thw']}")

    return inputs, processor.tokenizer


def generate_tokens(model, inputs, tokenizer, max_new_tokens=128):
    """
    Run autoregressive generation using the HuggingFace generation adapter.

    The adapter handles:
      - Prefill (context encoding) with vision embeddings on the first forward pass
      - Autoregressive token generation with KV cache on subsequent passes
    """
    adapter = HuggingFaceGenerationAdapter(model)

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "generation_config": gen_config,
    }

    if "pixel_values" in inputs:
        gen_kwargs["pixel_values"] = inputs["pixel_values"]
    if "image_grid_thw" in inputs:
        gen_kwargs["grid_thw"] = inputs["image_grid_thw"]

    logger.info("Starting generation...")
    start = time.monotonic()
    output_ids = adapter.generate(**gen_kwargs)
    elapsed = time.monotonic() - start

    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, input_len:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    num_new = new_tokens.shape[0]
    logger.info(
        f"Generated {num_new} tokens in {elapsed:.2f}s "
        f"({num_new / elapsed:.1f} tok/s)"
    )
    return generated_text


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    inf_cfg = build_config(args)

    compiled_model_path = args.compiled_model_path
    compiled_exists = (
        os.path.isdir(compiled_model_path)
        and os.path.isdir(os.path.join(compiled_model_path, "text_model"))
        and os.path.isdir(os.path.join(compiled_model_path, "vision_model"))
    )

    if not args.skip_compile and not compiled_exists:
        model = compile_model(args.model_path, inf_cfg, compiled_model_path, debug=args.debug)
        if args.compile_only:
            logger.info("Compile-only mode; exiting.")
            return
    elif not compiled_exists:
        raise FileNotFoundError(
            f"No compiled model found at {compiled_model_path}. "
            "Run without --skip_compile to compile first."
        )

    model = load_model(args.model_path, inf_cfg, compiled_model_path)

    inputs, tokenizer = prepare_inputs(args.model_path, args.image_path, args.prompt)

    generated_text = generate_tokens(
        model, inputs, tokenizer, max_new_tokens=args.max_new_tokens
    )

    print("\n" + "=" * 60)
    print("PROMPT:", args.prompt)
    if args.image_path:
        print("IMAGE:", args.image_path)
    print("-" * 60)
    print("RESPONSE:", generated_text)
    print("=" * 60)


if __name__ == "__main__":
    main()
