---
name: translate-model
description: End-to-end VLA/VLM/LLM port to AWS Trainium. Give it a model, it produces a compiled, validated deployment. No separate steps required.
---

# Translate Model to AWS Trainium

Port a PyTorch model to run on AWS Trainium using NeuronX Distributed Inference (NxDI).

---

## Background

### Hardware

AWS Trainium instances run NeuronCore-v2 (trn1) or NeuronCore-v3 (trn2) accelerators. Execution is graph-compiled — models are traced ahead-of-time by `neuronx-cc` to a NEFF (Neuron Executable File Format). Input shapes must be static at compile time; variable-length inputs are handled through bucketing (one compiled NEFF per bucket, selected at runtime).

Instance reference:
- `trn1.2xlarge` — 2 NeuronCores, 32 GiB HBM
- `trn1.32xlarge` — 32 NeuronCores, 512 GiB HBM
- `trn2.48xlarge` — 64 NeuronCores, 1.5 TiB HBM

Tensor parallelism (TP) shards weights across NeuronCores. `tp_degree` must divide the number of attention heads evenly.

### NxDI Model Structure

Every NxDI model is assembled from four classes:

**NeuronConfig** — Neuron-specific runtime parameters: `tp_degree`, batch size, buckets, dtype, attention kernel flags. Subclass only when custom Neuron-only parameters are needed; otherwise use `NeuronConfig` directly.

**InferenceConfig** — Wraps the HuggingFace `PretrainedConfig` and surfaces required attributes. Implement `get_required_attributes()` listing every field the model reads, and wire `get_neuron_config_cls()` to return the `NeuronConfig` class.

**NeuronBaseModel** — The `nn.Module`. Implement `setup_attr_for_model()` (assigns `tp_degree`, `hidden_size`, `buckets`, etc. from config) and `init_model()` (constructs layers using NxDI parallel primitives).

**ApplicationHead** — Subclasses `NeuronBaseForCausalLM` (text), `NeuronBaseForImageToText` (VLM), or `NeuronApplicationHead` (other). Sets `_model_cls`, implements `get_config_cls()`, and overrides `convert_hf_to_neuron_state_dict()` for weight key mapping.

**Critical: layer construction must happen in `init_model()` / `load_module()`, not in `__init__()`.**
`parallel_state` is only active after `ModelBuilder` calls `init_model()`. Layers constructed in `__init__()` silently fall back to `nn.Linear` (TP=1), weight sharding fails, and loading raises "Expected weight tensors for N ranks. Received 1." See [reference/nxdi_background.md](reference/nxdi_background.md) for the correct pattern.

### NxDI Parallel Primitives

| PyTorch layer | NxDI replacement |
|---|---|
| `nn.Linear` (Q/K/V, gate, up) | `ColumnParallelLinear` |
| `nn.Linear` (O, down, out) | `RowParallelLinear` |
| `nn.Embedding` | `ParallelEmbedding` |
| Attention | `NeuronAttentionBase` subclass |
| `nn.RMSNorm` | `CustomRMSNorm` (with CPU fallback for unit testing) |

### Installed NxDI Examples

Read relevant examples before writing any code. If the target architecture is already ported, subclass it — do not rewrite working NxDI code.

```bash
ls /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/neuronx_distributed_inference/models/
```

Key installed ports:
- `qwen3_vl/` — Qwen3-VL vision encoder + text backbone
- `llama/`, `qwen3/` — standard dense text models
- `qwen3_moe/` — MoE text model
- `pixtral/` — VLM with vision embeddings scattered into text sequence
- `mllama/` — VLM with cross-attention in text decoder
- `llama4/` — VLM with chunked image flow
- `diffusers/` — diffusion models (Flux)

---

## Workflow

**Activate the environment before running any code:**
```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
```

**Routing — read the relevant reference doc before writing code:**
- Text LLM → proceed directly below
- VLM (accepts image inputs: `pixel_values`, aspect ratios, vision masks) → read [reference/vlm_translation.md](reference/vlm_translation.md) first
- VLA (VLM + action head) → read [reference/vlm_translation.md](reference/vlm_translation.md) then [reference/action_head_translation.md](reference/action_head_translation.md)

### 1. Inspect the model

Read the model source and HuggingFace config. Identify:
- Architecture: attention type, MLP, positional encoding, any non-standard ops
- Closest installed NxDI port — read that port's source to understand the target class structure
- Architecture constants from the checkpoint

Write all architecture constants to `config_constants.py`. All subsequent code imports from there — no hardcoded numbers anywhere else.

```python
# Read checkpoint key/shape info without loading all weights
import json
with open(f"{model_path}/model.safetensors.index.json") as f:
    keys = json.load(f)["weight_map"]
```

### 2. Write the translation

**If a matching NxDI port exists:** subclass it, override only `convert_hf_to_neuron_state_dict` for checkpoint key differences. This is always preferred over rewriting.

**If writing custom blocks:** replace PyTorch layers with NxDI primitives per the table above. Follow [reference/scaffolding_integration.md](reference/scaffolding_integration.md) for the full class structure.

Invariants that must hold across all translated code:
- All architecture constants imported from `config_constants.py`
- Layer construction in `init_model()` or `load_module()`, never `__init__()`
- Dynamic constants (`torch.arange`, `torch.linspace`, `torch.ones`) pre-computed as `register_buffer()` in `__init__()`
- `load_state_dict` overrides accept `**kwargs` — NxDI internally passes `assign=True`
- No bare `except` around compile calls — failures must propagate

For weight mapping, see [reference/weight_mapping.md](reference/weight_mapping.md).
For compiler flags, error codes, and TP verification, see [reference/nxdi_background.md](reference/nxdi_background.md).

### 3. Compile

```python
model.compile(save_path)           # NxDI NeuronBaseModel subclasses
model.compile_denoiser(save_path)  # action heads using NeuronActionHeadBase
# torch_neuronx.trace() for standalone vision encoders — see reference/patterns/vit_compilation.md
```

Run a CPU forward pass before compiling to catch shape errors cheaply before multi-minute neuronx-cc runs.

### 4. Validate

For every compiled NEFF, compare against HF CPU reference:

```python
hf_out = hf_model(*inputs).float()
neff_out = neff_model(*inputs).float()
mean_diff = (hf_out - neff_out).abs().mean().item()
cos_sim = torch.nn.functional.cosine_similarity(
    hf_out.flatten(), neff_out.flatten(), dim=0
).item()
print(f"mean_diff={mean_diff:.4f}  cos_sim={cos_sim:.6f}")
```

Thresholds: `mean_diff < 0.1`, `cos_sim > 0.99`. If `mean_diff > 1.0`, weights were not loaded before trace — fix `load_module()`.

### 5. Unit test blocks (on demand)

If a NEFF fails validation or a block's correctness is uncertain, test it in isolation using `scripts/block_testing_utils.py:test_block_correctness`. Instantiate both the original PyTorch block and the translated block with identical weights and assert numerical equivalence (`atol=1e-3` for BF16). Do this after the full translation is written, not as a gate between blocks.

---

## Deliverables

- `config_constants.py` — all architecture constants extracted from the checkpoint
- Translated model file(s)
- `run_inference.py` — loads compiled NEFFs and checkpoint weights, exposes a clean inference API, runnable on Trainium
