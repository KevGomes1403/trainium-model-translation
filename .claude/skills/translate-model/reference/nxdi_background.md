---
name: nxdi-background
description: Technical reference for NxDI model compilation on Trainium — TP initialization, compiler flags, error codes, and dynamic constant patterns.
---

# NxDI Technical Reference

This document is a reference for issues that arise during translation and compilation. Consult it as needed — it is not a workflow to follow sequentially.

---

## TP Initialization: load_module() vs __init__()

`ColumnParallelLinear` and `RowParallelLinear` check whether `parallel_state` is initialized at construction time. `parallel_state` is only initialized by `ModelBuilder`, which calls `init_model()` (for NeuronBaseModel subclasses) or `load_module()` (for ModelWrapper subclasses) after setup.

**Correct pattern — layers in `init_model()` / `load_module()`:**
```python
class MyNeuronWrapper(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.model = None        # do not construct here

    def load_module(self):
        # parallel_state is active here — ColumnParallelLinear uses real TP
        self.model = MyActualModel()
        self.model = self.model.bfloat16().eval()
```

**Wrong pattern — silent TP=1, broken weight sharding:**
```python
class MyNeuronWrapper(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.model = MyActualModel()  # parallel_state not active — WRONG
```

If using an existing NxDI port (e.g. `NeuronQwen3VLTextForCausalLM`), layer construction already happens inside `NeuronBaseModel.init_model()`, which ModelBuilder calls correctly. Subclassing and overriding only `convert_hf_to_neuron_state_dict` is safe.

**TP verification check** — run before compiling any subgraph targeting `tp_degree > 1`:
```python
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear
from neuronx_distributed.parallel_layers import parallel_state

parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_degree)
block = MyNeuronWrapper()
block.load_module()

has_parallel = any(isinstance(m, ColumnParallelLinear) for m in block.model.modules())
assert has_parallel, "TP FAILED: layers constructed before parallel_state was active"
print("TP verification PASSED")
```

---

## Dynamic Constants

Any tensor created in `forward()` from Python operations (`torch.arange`, `torch.linspace`, `torch.ones`, `torch.zeros`, sinusoidal embeddings, RoPE frequency bases) becomes a dynamic compiler node. In deep models (16+ layers), identical nodes accumulate and their merged debug filenames exceed the 255-character limit → `[Errno 36] File name too long`.

**Fix:** pre-compute all such tensors in `__init__()` as `register_buffer()`:
```python
def __init__(self, ...):
    ...
    self.register_buffer("position_ids", torch.arange(max_seq_len).unsqueeze(0))
    self.register_buffer("attention_mask", torch.ones(1, 1, seq_len, seq_len))
```

---

## Compiler Flags

| Flag | Effect | When to use |
|------|--------|-------------|
| `-O1` | Standard optimization | **Default for all subgraphs** |
| `--model-type=transformer` | Replaces softmax with custom NKI kernel | **Never on DiT/denoising models** — causes cos_sim≈0.916, 37% error per step. Safe only for standard causal LM backbones. |
| `--model-type=unet-inference` | Optimized for ViT/conv patterns | Vision encoders. +6% over transformer. |
| `--auto-cast=matmult` | BF16 matmuls, FP32 accumulators | Vision encoders. ~50% NEFF size reduction, 99.999% accuracy maintained. |
| `--auto-cast=none` | No dtype casting | DiT and action head subgraphs — preserve BF16 throughout. |
| `--optlevel 3` | Maximum optimization | Vision encoders with `--auto-cast=matmult`. |

---
