# SmolVLA on Trainium (NxDI port)

A `HuggingFaceVLA/smolvla_libero` port to AWS Trainium 3 using **NeuronX
Distributed Inference (NxDI)**. Three compiled
subgraphs, maximally on-Neuron, written in the production NxDI structure.
Driven end-to-end in a closed-loop LIBERO simulation.

[![Demo of SmolVLA running on Trainium 3 in a closed-loop LIBERO object simulation](./demo_libero_object_t2_s42.gif)]

*Demo of SmolVLA running on Trainium 3 in a closed-loop `libero_object`
simulation.*

## Layout

```
smol_vla/
├── config_constants.py          # All architecture constants from the checkpoint
├── modeling_smolvla.py          # SmolVLAPolicy: orchestrator (compile / load / generate)
├── modeling_smolvla_vision.py   # SigLIP-12L + connector  (NEFF #1)
├── modeling_smolvla_text.py     # VLM 32L prefix + Action expert 32L denoiser  (NEFF #2 + #3)
├── weight_mapping.py            # HF safetensors -> 3 per-subgraph state dicts
├── run_inference.py             # CLI: compile / run / benchmark (synthetic inputs)
├── demo_libero.py               # Closed-loop LIBERO simulation demo
├── run_demo.sh                  # Convenience wrapper for the demo
├── neuron_action_head_base.py   # Skill-provided base (ModelWrapper-compatible config)
└── README.md
```

This mirrors the per-model layout in
`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/.../models/<model>/` (e.g.
`pixtral/` has `modeling_pixtral.py` + `modeling_pixtral_vision.py`); we add
`_text` because SmolVLA has separate text-prefix and action-expert subgraphs.

## What runs where

| Component                                              | Where      | Subgraph |
|--------------------------------------------------------|------------|----------|
| SigLIP vision encoder (12 layers, 768 hidden)          | **Neuron** | #1       |
| Pixel-shuffle 4× + connector + scale by sqrt(960)      | **Neuron** | #1       |
| Lang token embed + scale by sqrt(960)                  | **Neuron** | #2       |
| State projection (32 → 960)                            | **Neuron** | #2       |
| VLM 32-layer text decoder (eager GQA, RoPE, RMSNorm)   | **Neuron** | #2       |
| Pad-aware position_ids + 2D attention mask             | **Neuron** | #2/#3    |
| Action expert: 16× self-attn (concat past KV) layers   | **Neuron** | #3       |
| Action expert: 16× cross-attn (Q from suffix) layers   | **Neuron** | #3       |
| Sinusoidal timestep embedding                          | **Neuron** | #3       |
| Action in/out projections + time MLP                   | **Neuron** | #3       |
| Image preprocessing (flip, resize-with-pad, normalize) | CPU        | —        |
| Tokenization                                           | CPU        | —        |
| 10-step Euler denoising loop                           | CPU        | —        |
| LIBERO mujoco simulation                               | CPU        | —        |

**Deviations from "everything on Neuron":**
1. The 10-step Euler loop runs on CPU. Static-shape compilation cannot host
   `for step in range(10)` as a single graph; the loop body is the compiled
   subgraph. Each step calls NEFF #3 with the updated `noisy_actions`.
2. Tokenization, image flip / resize-with-pad, and state-vector composition
   run on CPU because they are data-loading, not model compute.
3. The mujoco physics step in the LIBERO demo is CPU only (it is the
   environment, not the policy).

## Hardware constraints flagged

`tp_degree = 1` because `num_attention_heads = 15` and `num_kv_heads = 5`
— neither divides cleanly into the 4 Neuron cores on `trn3pd98.3xlarge`.

The NxDI parallel primitives (`ColumnParallelLinear`, `RowParallelLinear`,
`ParallelEmbedding`) are still used throughout, so this code is portable to
instances with divisor-friendly head counts. On this instance, 3 of 4 cores
idle; the model fits comfortably in one core's HBM partition with vast headroom.

## Inference flow

```
images [2 cams x [B, 3, 512, 512]]   lang_token_ids [B, 48]   lang_mask [B, 48]   state [B, 32]
                |                              |                    |                 |
        [Neuron NEFF #1]                       |                    |                 |
        Vision (per camera)                    |                    |                 |
                |                              |                    |                 |
        [B, 128, 960] vision_features          |                    |                 |
                |______________________________|____________________|_________________|
                                                |
                                       [Neuron NEFF #2]
                                       VLM Prefix (32 layers, pad-aware)
                                                |
                                       prefix_keys, prefix_values
                                       [32, B, 177, 5, 64] each
                                                |
                          ┌─────────────────────┴─────────────────────┐
                          │  CPU Euler loop (10 steps)                 │
                          │     for t in [1.0, 0.9, ..., 0.1]:          │
                          │         v_t = NEFF#3(x_t, t, K, V, pad)    │
                          │         x_t += dt * v_t                     │
                          └─────────────────────┬─────────────────────┘
                                                |
                                       action_chunk [B, 50, 32]
                                       (first 7 dims used by env)
```

## Validation

| Check                                            | Result                              |
|--------------------------------------------------|-------------------------------------|
| Vision NEFF vs HF SmolVLM2 vision                | cos_sim = 0.99990 (single image)    |
| Prefix KV layer 0..31 vs lerobot CPU             | max abs diff ≤ 0.4 (BF16)           |
| Full action chunk vs lerobot CPU (matched noise) | cos_sim = 0.9999, mean abs diff = 0.007 |
| Closed-loop LIBERO `libero_object` task 0        | success                             |
| End-to-end inference latency (one chunk)         | ~65 ms warm on trn3pd98.3xlarge     |

The numerical match is achieved by replicating four lerobot-specific quirks
that aren't in the SmolVLM2 HF config:

1. **`resize_with_pad` pads top+left only** (image lands in the bottom-right
   corner of the 512×512 frame), not centered.
2. **Pad-aware attention**: dynamic 2D mask + cumsum-based position_ids that
   skip padding tokens. A static prefix-LM mask leaks attention into pad-token
   positions.
3. **RoPE max_wavelength = 10000** (lerobot hardcodes this in
   `apply_rope`); the SmolVLM2 HF config says 100000, but lerobot trained
   the model with 10000.
4. **Image flip** in the LIBERO env (180° rotate, both H and W) per the
   `libero_processor` step in `lerobot.processor.env_processor`.

## Compile

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Download checkpoint (one-time)
python -c "from huggingface_hub import snapshot_download; \
    print(snapshot_download(repo_id='HuggingFaceVLA/smolvla_libero'))"

# Compile (one-time, ~90s wall clock for 3 NEFFs)
python -m smol_vla.run_inference --action compile \
    --hf-checkpoint /home/ubuntu/.cache/huggingface/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/<hash>/ \
    --neff-dir      /home/ubuntu/vla/smol_vla_neff_libero_hfvla
```

## Synthetic-input run

```bash
python -m smol_vla.run_inference --action run \
    --hf-checkpoint <ckpt> \
    --neff-dir      /home/ubuntu/vla/smol_vla_neff_libero_hfvla
# -> p50 latency, output stats, NaN check on a random batch
```

## Closed-loop LIBERO demo

```bash
# Defaults: libero_object task 0 seed 7, replan every 1 step
./smol_vla/run_demo.sh

# Pick a different task / seed:
./smol_vla/run_demo.sh --task 2 --seed 42
./smol_vla/run_demo.sh --suite libero_spatial --task 0 --seed 0
./smol_vla/run_demo.sh --task 1 --seed 17 --output mydemo.mp4

# All flags:
#   --suite     libero_object | libero_spatial | libero_goal | libero_10 | libero_90
#   --task      task index (0..N-1 in suite)
#   --seed      initial-state seed (changes object positions and policy noise)
#   --steps     max env steps (default 250)
#   --replan    chunk actions to execute before replanning (default 1)
#   --output    output mp4 path
```

The script auto-resolves the checkpoint and NEFF dir; override with
`SMOLVLA_CKPT` / `SMOLVLA_NEFF` environment variables if needed.

The output mp4 shows:
- main pane: agentview (closed-loop sim driven by Neuron SmolVLA)
- top-right inset: wrist camera
- bottom HUD: step counter, per-step infer latency, status badge, gripper bar

## Programmatic use

```python
from smol_vla import SmolVLAPolicy

policy = SmolVLAPolicy(hf_checkpoint_dir="<path>", tp_degree=1)
policy.load("/home/ubuntu/vla/smol_vla_neff_libero_hfvla")

# images: list of NUM_CAMERAS tensors, each [B, 3, 512, 512] BF16
# lang_token_ids: [B, 48] INT32
# lang_mask: [B, 48] BOOL  (True = real token, False = pad)
# state: [B, 32] FP32  (already normalized, zero-padded)
action_chunk = policy.generate(images, lang_token_ids, state, lang_mask=lang_mask)
# action_chunk: [B, 50, 32] FP32  (first 7 dims used by LIBERO)
```
