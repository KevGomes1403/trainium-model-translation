# NxDI Library: Capabilities and VLM Translation Guide

## Library Structure Overview

The library lives at `neuronx_distributed_inference` and is organized into two main namespaces:

- **`models/`** — High-level model classes, configs, and application heads
- **`modules/`** — Low-level building blocks (attention, KV cache, MoE, custom ops)

---

## Core Building Blocks

### 1. Parallel Layers (from `neuronx_distributed`)

These replace all standard PyTorch layers for distributed inference:

| PyTorch | NxDI Replacement | Use case |
|---|---|---|
| `nn.Linear` (column-wise) | `ColumnParallelLinear` | Q, K, V, gate, up projections |
| `nn.Linear` (row-wise) | `RowParallelLinear` | O, down projections |
| `nn.Embedding` | `ParallelEmbedding` | Token embeddings |

### 2. Attention (`modules/attention/attention_base.py`)

`NeuronAttentionBase` is the core attention primitive. It provides:
- GQA/MHA/MQA support
- KV cache management (standard and block/paged)
- Flash attention kernels (NKI)
- Sliding window attention
- RoPE integration via a `rotary_emb` argument
- Customizable `scaled_qk()` hook for non-standard attention scaling (used by Gemma-2 for softcapping)
- `FlashAttentionStrategy` enum to disable NKI kernels when needed (e.g., Gemma-2's head_dim=256)

### 3. Custom Ops (`modules/custom_calls.py`)

- `CustomRMSNorm` — Neuron-optimized RMSNorm. Must be swapped for `nn.RMSNorm` when running on CPU (unit tests). Pattern seen in all models:

```python
def _rms_norm(hidden_size, eps):
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm
```

### 4. Autobucketing (`modules/autobucketing.py`)

- `autobucketing.generate_buckets(min, max)` — Generates a list of bucket sizes for variable-length inputs. Used for vision encoders to handle variable image patch counts.

### 5. Padding Utilities (`modules/padding.py`)

- `pad_tensor(tensor, target_size, pad_value)` / `unpad_tensor(tensor, slices)` — Used by vision model wrappers to pad inputs to bucket boundaries before tracing.

---

## Config System

### `NeuronConfig`
Neuron-specific runtime parameters: `tp_degree`, `batch_size`, `seq_len`, `torch_dtype`, `on_device_sampling_config`, `buckets`, `vocab_parallel`, etc.

### `InferenceConfig`
Model architecture parameters (equivalent to HuggingFace `PretrainedConfig`). Subclass this and implement:
- `get_required_attributes()` — list of attribute names that must be present
- `add_derived_config()` — compute derived attributes (e.g., `num_cores_per_group`, `head_dim`, `layer_types`)
- `get_neuron_config_cls()` — return the `NeuronConfig` subclass

### `ImageToTextInferenceConfig` (for VLMs)
Extends `InferenceConfig` for multimodal models. Holds **two** sub-configs:
- `self.text_config` — `InferenceConfig` for the text decoder
- `self.vision_config` — `InferenceConfig` for the vision encoder

Both are initialized from the HF config's nested `text_config`/`vision_config` dicts, each with their own `NeuronConfig`.

---

## Model Class Hierarchy

### Text Models

```
NeuronBaseModel (nn.Module)
  └── setup_attr_for_model(config)  ← set tp_degree, hidden_size, buckets, etc.
  └── init_model(config)            ← instantiate layers, embed, lm_head

NeuronBaseForCausalLM (application head)
  └── _model_cls = YourNeuronModel
  └── get_config_cls()
  └── convert_hf_to_neuron_state_dict(state_dict, config)
  └── update_state_dict_for_tied_weights(state_dict)
```

### Multimodal (VLM) Models

```
NeuronBaseForImageToText (extends NeuronBaseForCausalLM)
  ├── text_model_cls           ← NeuronBaseModel subclass for text decoder
  ├── vision_model_cls         ← nn.Module for vision encoder
  ├── text_model_wrapper       ← ImageToTextModelWrapper
  ├── vision_model_wrapper     ← Custom ModelWrapper subclass
  └── enable_vision_encoder()  ← Must be implemented; registers vision_models list
```

The base class handles:
- Separate `compile()` / `load()` / `shard_weights()` for text and vision sub-models
- Separate `ModelBuilder` instances for text and vision (different TP degrees possible)
- Routing `vision_embeddings` and `vision_mask` through the text model's `forward()` call
- Snapshot hooks for debugging

---

## ModelWrapper and Vision Wrappers

`ModelWrapper` handles tracing, bucketing, and input generation for a compiled sub-model. For VLMs:

### `ImageToTextModelWrapper`
Wraps the text decoder. Extends `ModelWrapper` with:
- Fixed 24-argument positional input signature (NxD tracing doesn't support kwargs)
- Positions 22–23 are `vision_embeddings` and `vision_mask`
- `get_dummy_vision_inputs()` — generates zero-filled vision inputs for tracing
- Batch padding/sorting for continuous batching

### Custom vision wrapper (e.g., `PixtralVisionModelWrapper`)
Wraps the vision encoder. Must implement:
- `input_generator()` — generates example inputs for each bucket (patch_embeds, attention_mask, position_ids)
- `forward(pixel_values, image_sizes)` — preprocesses raw pixels (patchify, pad to bucket, call model, unpad)
- `get_model_instance()` — returns `EncoderModelInstance(model_cls, config)`
- `get_target_bucket(patch_embeds)` — routes to closest bucket size

---

## Vision Encoder Pattern (Pixtral as Reference)

The Pixtral vision encoder (`models/pixtral/modeling_pixtral_vision.py`) demonstrates the canonical pattern for a VLM vision tower:

1. **Patch embedding**: Conv2d replaced by `unfold + ColumnParallelLinear` (conv2d not supported on Neuron)
2. **Vision transformer**: Stack of attention layers using `NeuronAttentionBase` with custom `rotary_emb` (2D RoPE for image patches)
3. **Multimodal projector**: `ColumnParallelLinear → activation → RowParallelLinear` mapping vision hidden dim to text hidden dim
4. **Block attention mask**: Vision patches only attend within the same image (not causal)

### Key utilities from `llama4/utils/encoder_utils.py`

| Function | Purpose |
|---|---|
| `scatter_by_index_put(h_image, vision_embeddings, vision_mask)` | Injects vision embeddings into text embedding positions using index_put |
| `generate_positions_from_mask(mask)` | Converts a boolean vision mask to position indices |
| `pad_positions(positions, target_size, fill_value)` | Pads position indices to text bucket size |
| `pad_vision_embeddings(vision_embeddings, pad_limit)` | Pads vision embeddings to text bucket size |

---

## How Vision Embeddings Flow Through the Text Model

The text model's `forward()` receives two extra tensors:
- `vision_embeddings: [BS, seq_len, hidden_size]` — pre-projected vision features
- `vision_mask: [BS, seq_len, 1]` — integer positions of vision tokens

The text model calls `encode_vision_to_input(inputs_embeds, vision_embeddings, vision_mask)` to scatter vision embeddings into the text embedding sequence before passing to the transformer layers. `NeuronPixtralTextModel` overrides this from `NeuronLlamaModel` using `scatter_by_index_put`.

---

## Weight Mapping for VLMs

`convert_hf_to_neuron_state_dict` must handle both sub-models:

1. **Text model**: Standard renames (q/k/v → qkv_proj, o_proj → o_proj.o_proj, rank metadata injection)
2. **Vision model**: Key prefix remapping (e.g., `vision_tower.` → `vision_`), conv2d weight reshape (flatten kernel dims for the linear replacement), dtype casting
3. **Tied weights**: `update_state_dict_for_tied_weights` handles lm_head ↔ embed_tokens ties

---

## Existing VLM Models in NxDI

| Model | Files | Notes |
|---|---|---|
| **Pixtral** | `pixtral/modeling_pixtral.py`, `modeling_pixtral_vision.py` | LLaVA-style with Mistral text backbone; best reference for new VLMs |
| **MLlama** (Llama 3.2 Vision) | `mllama/` | Cross-attention vision integration; more complex |
| **Llama 4** | `llama4/modeling_llama4.py`, `modeling_llama4_vision.py` | Multi-chunk image tiling with complex padding/depadding utilities |
| **Flux** | `diffusers/flux/` | Diffusion model; different paradigm (encoder-only, no KV cache) |

---

## What is Required to Translate a Multimodal Model

To translate a VLM like Qwen2-VL to Trainium, you need to implement six components:

### 1. `ImageToTextInferenceConfig` subclass
With nested text/vision configs and `get_required_attributes()` covering both `text_config.*` and `vision_config.*` attribute paths.

### 2. Vision encoder model (`nn.Module`)
Replace:
- Conv2d → unfold + `ColumnParallelLinear`
- All attention → `NeuronAttentionBase` with appropriate `rotary_emb` (2D RoPE for vision if needed)
- All linear layers → `ColumnParallelLinear` / `RowParallelLinear`
- Norms → `CustomRMSNorm` (with `cpu_mode()` guard)
- Include the multimodal projector in the vision model (projects to text hidden dim)

### 3. Vision model wrapper
Subclass `ModelWrapper` with:
- `input_generator()` for each vision bucket
- `forward(pixel_values, image_sizes)` with patchify + pad + call + unpad
- `get_model_instance()` returning `EncoderModelInstance`

### 4. Text model
Subclass `NeuronBaseModel` (or reuse `NeuronLlamaModel` if text backbone is Llama-compatible). Override `encode_vision_to_input()` to scatter vision embeddings using `scatter_by_index_put`.

### 5. Application head
Subclass `NeuronBaseForImageToText` with:
- `text_model_cls`, `vision_model_cls`, `text_model_wrapper`, `vision_model_wrapper`
- `enable_vision_encoder()` — instantiate the vision wrapper and append to `self.vision_models`
- `forward()` — handle prefill (call vision encoder, pad, scatter) vs. token generation (dummy vision inputs)
- `convert_hf_to_neuron_state_dict()` — handle both text and vision key remapping
- `get_required_kwargs()` — return `["pixel_values", "vision_mask", "image_sizes"]`

### 6. Input processor (optional but recommended)
Preprocessing logic for raw images to `pixel_values` tensors, handling variable image sizes and multi-image batches.

---

## Qwen2-VL Specific Notes

**What's already done** in this workspace: the text LM backbone (`NeuronQwen2VLForCausalLM`) is complete and handles the Qwen2-7B-style transformer with GQA, SwiGLU MLP, and QKV bias. The `visual.*` keys are stripped in weight conversion.

**What remains for full VLM support**:

1. **Vision encoder** — Qwen2-VL uses a ViT with 2D RoPE (MRoPE). The patch embedding is a Conv2d that must be replaced with unfold + `ColumnParallelLinear`. The attention uses 2D positional embeddings (separate height/width frequencies), similar to Pixtral's `PixtralRotaryEmbedding`.

2. **Multimodal projector** — Qwen2-VL uses a simple MLP projector (similar to `NeuronLlavaMultiModalProjector`) mapping vision hidden dim → text hidden dim.

3. **`ImageToTextInferenceConfig`** — Wrapping both the existing `Qwen2VLInferenceConfig` (text) and a new vision config.

4. **`NeuronQwen2VLForConditionalGeneration`** — The full VLM application head subclassing `NeuronBaseForImageToText`, with `enable_vision_encoder()`, `forward_atomic_prefill()`, and the complete `convert_hf_to_neuron_state_dict()` that also handles vision keys.

The Pixtral implementation is the closest architectural analog and the best template to follow for Qwen2-VL's vision tower.
