# Vision Encoder Compilation Pattern

## When to use

Vision encoders (SigLIP2, ViT variants, Qwen3-VL vision tower) at TP=1. These do not use NxD Inference ModelBuilder — compile with raw `torch_neuronx.trace()` on a single core.

## Model variant

Always use the FixRes `opt-` variant with a fixed resolution (e.g. `google/siglip2-giant-opt-patch16-256`). Fixed-resolution models produce static input shapes and compile without bucketing.

## Compiler flags

```
--auto-cast matmult    # BF16 matmuls, FP32 accumulators — ~50% NEFF size reduction, 99.999% accuracy
--optlevel 3           # Maximum optimization
--model-type unet-inference   # +6% over transformer for vision encoders
```

**Do NOT use `--model-type=transformer`** for vision encoders. The `unet-inference` type is correct for both ViT and DiT/denoising models.

## Compilation call

```python
import os
import torch
import torch_neuronx
from transformers import SiglipVisionModel

model = SiglipVisionModel.from_pretrained(model_name).eval()
traced = torch_neuronx.trace(
    model,
    torch.randn(1, 3, 256, 256),
    compiler_workdir=save_path + "workdir/",
    compiler_args=[
        "--auto-cast", "matmult",
        "--optlevel", "3",
        "--model-type", "unet-inference",
    ],
)
os.makedirs(save_path, exist_ok=True)
torch.jit.save(traced, save_path + "model.pt")
```

The helper `scripts/vision_encoder_utils.py:compile_siglip2_vit()` wraps this pattern.

## Expected performance

70× speedup over CPU for SigLIP2-giant at batch=1 on a single NeuronCore.

## TP

TP=1 only. `torch_neuronx.trace()` does not initialize `parallel_state` — vision encoders compiled this way run as single-core. This is correct and expected. Do not attempt TP>1 via NxDI ModelBuilder unless the vision encoder is part of a full VLM port using NxDI's VLM pipeline.

## Qwen3-VL specific: packed attention removal

The Qwen3-VL vision encoder uses `cu_seqlens`-based packed attention with `torch.split(..., lengths.tolist())` — data-dependent shapes that are incompatible with neuronx-cc. Fix before tracing: replace the packed attention with standard full-sequence attention. For single fixed-size images this is numerically identical.

## Deepstack intermediate features

Some ViT variants (e.g. deepstack configurations) expose intermediate layer features (layers 5, 11, 17) in the forward signature. Verify whether these are actually consumed at inference time before including them in the compiled graph. Unused outputs add NEFF size and latency for no benefit.
