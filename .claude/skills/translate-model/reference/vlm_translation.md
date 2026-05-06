---
name: vlm-translation
description: Reference for translating vision-language models (VLMs) to NxDI — architecture patterns, runtime flow, and critical gotchas.
---

# VLM Translation Reference

Use this when the target model consumes image inputs (`pixel_values`, image chunks, aspect ratios, vision masks) in addition to text. For text-only models, use the main workflow in [SKILL.md](../SKILL.md).

---

## 1. NxDI Mental Model for VLMs

1. Trainium compilation is shape-sensitive. Variable input lengths require **bucketing**.
2. NxDI splits into `models/` (model/application abstractions) and `modules/` (reusable primitives). Both matter.
3. Multimodal image-to-text models compile **two subgraphs**: a text graph (context encoding + token generation) and a vision graph (image encoder + projector).
4. The base class `image_to_text_model_base.py` handles separate compile/load/shard for text and vision builders.
5. Tracing does not support kwargs on the image-to-text path. `ImageToTextModelWrapper` enforces a fixed 24-argument ordered signature (vision args at positions 22 and 23).

---

## 2. Pick the Correct Reference Pattern

Read the matching installed port before writing any code:

**Pixtral-style** — vision embeddings scattered into the text sequence (LLaVA-like, Qwen2-VL-like):
- `models/pixtral/modeling_pixtral.py`
- `models/pixtral/modeling_pixtral_vision.py`

**Llama4-style** — chunked images + vision adapter + scatter:
- `models/llama4/modeling_llama4.py`
- `models/llama4/modeling_llama4_vision.py`

**MLlama-style** — cross-attention in the text decoder, not simple scatter:
- `models/mllama/modeling_mllama.py`
- `models/mllama/modeling_mllama_vision.py`

**Diffusion/generation (Flux-style)** — separate pipeline architecture:
- `models/diffusers/flux/application.py` — do not force image-to-text base classes here.

**Qwen3-VL text backbone** — if the target backbone is Qwen3-VL, subclass `NeuronQwen3VLTextForCausalLM` from `qwen3_vl/modeling_qwen3_vl_text.py` and override only `convert_hf_to_neuron_state_dict`. Do not rewrite it.

---

## 3. Canonical Runtime Flow (Image-to-Text)

1. `NeuronBaseForImageToText.__init__` builds text context/token models, then calls `enable_vision_encoder()`.
2. `compile()` traces and saves text model and vision model separately.
3. **Prefill with image:**
   - Vision wrapper preprocesses image → vision model returns projected embeddings.
   - Vision mask is converted to integer positions and padded to the selected text bucket.
   - Text model receives `vision_embeddings` and `vision_mask`.
4. **Token generation / text-only:**
   - Use dummy vision tensors from `ImageToTextModelWrapper.get_dummy_vision_inputs`.

---

## 4. Translation Steps

### Preflight

- Identify model family (Pixtral / Llama4 / MLlama / Flux-like).
- Confirm tp_degree, bucket strategy, and any unsupported features for the selected reference.

### Config Classes

For image-to-text, create an `ImageToTextInferenceConfig` subclass with nested `text_config` and `vision_config`. Include required attributes from both. Use `NeuronConfig` or a custom subclass if extra Neuron-only fields are needed.

### Vision Encoder

1. Replace unsupported ops with NxDI-compatible primitives:
   - `Conv2d` patch embedding → unfold/patchify + `ColumnParallelLinear` (or approved parallel conv wrapper)
   - Attention → `NeuronAttentionBase` subclass
   - Dense layers → `ColumnParallelLinear` / `RowParallelLinear`
   - Norm → `CustomRMSNorm` with CPU fallback for tests
2. Implement vision positional encoding compatible with the source (2D RoPE, chunk-specific, etc.).
3. Include multimodal projector: maps vision hidden size → text hidden size.

### Vision Wrapper

Subclass `ModelWrapper`. Implement `input_generator()` per bucket and a `forward()` that: patchifies/chunks the image, builds attention mask/position IDs, routes to the correct bucket, pads, calls the compiled model, and unpads.

### Text Model Integration

- **Scatter-based:** override `encode_vision_to_input()` and scatter embeddings using the positional mask.
- **Cross-attention-based:** implement the multimodal cross-attention path and cache behavior (MLlama pattern).

In both cases, ensure the text forward accepts vision args in prefill and uses dummy inputs in token generation.

### Application Head

Subclass `NeuronBaseForImageToText`. Set `text_model_cls`, `vision_model_cls`, `text_model_wrapper`, `vision_model_wrapper`. Implement `enable_vision_encoder()` and `get_required_kwargs()` for generation (e.g. `pixel_values`, `vision_mask`, `image_sizes`).

### Weight Mapping

Start from the text weight mapping. Add vision mapping: rename prefixes, convert/fuse QKV layouts, reshape patch conv weights when moving from conv to unfold+linear form. See [weight_mapping.md](weight_mapping.md).

---

## 5. Primitive Replacement Map

| Source op | NxDI replacement |
|---|---|
| `nn.Linear` (Q/K/V, gate, up) | `ColumnParallelLinear` |
| `nn.Linear` (O, down) | `RowParallelLinear` |
| `nn.Embedding` | `ParallelEmbedding` |
| Attention | `NeuronAttentionBase` subclass |
| `nn.RMSNorm` | `CustomRMSNorm` (CPU fallback in tests) |
| Vision conv patch embedding | unfold/patchify + parallel linear |

---

## 6. Critical Gotchas

1. Image-to-text tracing uses ordered positional args — missing placeholder args break runtime.
2. Vision mask is often converted from a boolean mask to a **positions tensor** for tracing compatibility.
3. Bucket mismatch between text and vision paths is a common failure mode — verify both are aligned.
4. Always use vision-specific NeuronConfig inside vision wrappers to avoid cross-config contamination.
5. CPU tests require fallback norms/operators where Neuron custom calls are device-specific.
6. **Qwen3-VL packed attention:** replace `cu_seqlens`-based packed attention with standard full-sequence attention before tracing. See [nxdi_background.md](nxdi_background.md).
7. **Image preprocessing:** client must resize images to align with `patch_size × spatial_merge_size` before calling the vision encoder.
