---
name: action-head-translation
description: Reference for porting VLA action heads to Trainium — classification, NeuronDenoisingWrapper pattern, CPU denoising loop, compiler args, and gotchas.
---

# Action Head Translation Reference

The action head is the portion of a VLA that takes VLM output and predicts the next action. Read [vlm_translation.md](vlm_translation.md) first, then this doc.

---

## Part 1: Classify the Action Head

Read the action head class definition, its forward pass, and the noise/sampling config. Look for:

**Discrete token indicators:**
- Action values binned into vocabulary tokens (OpenVLA-style)
- Forward pass appends action tokens to the LM sequence and predicts them autoregressively
- Config has fields like `num_action_bins`, `action_token_begin_idx`

**Continuous indicators:**
- Real-valued action vectors
- Forward takes noisy action chunk + timestep/noise level, outputs denoised version
- Imports reference a noise scheduler, flow matching, DDPM, or diffusion process

Continuous subtypes:
- **Flow matching** (GR00T N1.6, SmolVLA, π0) — linear interpolation, typically fewer steps
- **DDPM-style** — fixed Markov chain, discrete timesteps, longer denoising chains
- **Regression MLP** — no denoising loop, single forward pass from VLM features to actions
- **ACT/CVAE** — encoder-decoder, latent sampling, no iterative denoising

**Branching:**
1. **Discrete tokens** → return to SKILL.md, follow the standard causal LM path.
2. **Regression MLP** → port as a standard MLP with `ColumnParallelLinear` + `RowParallelLinear`. No denoising loop needed. Skip to Part 4.
3. **Flow matching / DDPM-style** → continue to Part 2.
4. **ACT/CVAE** → compile encoder and decoder as separate subgraphs; latent sampling stays on CPU. Flag dynamic control flow in the encoder.
5. **Unknown** → fall back to `torch_neuronx.trace()` on the full action head as a single subgraph. Document the fallback.

---

## Part 2: NeuronDenoisingWrapper

`NeuronDenoisingWrapper` and `NeuronActionHeadBase` are defined in `scripts/neuron_action_head_base.py`. Import them — do not redefine:

```python
from scripts.neuron_action_head_base import (
    NeuronDenoisingWrapper,
    NeuronActionHeadBase,
    NeuronDenoisingConfig,
    ConditioningContract,
)
from scripts.cross_attention_nki import cross_attention_kernel, get_tile_size
```

**Mandatory: DiT model construction must happen in `load_module()`, not `__init__()`.**

```python
# CORRECT
class NeuronGrootDenoisingWrapper(NeuronDenoisingWrapper):
    def __init__(self, config):
        nn.Module.__init__(self)   # bypass ModelWrapper.__init__ — it's LLM-oriented
        self.config = config
        self.model = None          # do not construct DiT here

    def load_module(self):
        # parallel_state is active here
        self.model = GR00TDiTModel(self.config)
        if self._preload_sd is not None:
            self.model.load_state_dict(self._preload_sd, strict=False)
        self.model = self.model.bfloat16().eval()

    def forward(self, noisy_actions, conditioning_tokens, timestep_embedding, attention_mask):
        return self.model(noisy_actions, conditioning_tokens, timestep_embedding, attention_mask)
```

### Compiled Graph Boundary

**Inside the compiled graph** (`NeuronDenoisingWrapper.forward`):
- Single forward pass of the denoiser (one denoising step)
- Self-attention over action tokens
- Cross-attention from action tokens into VLM conditioning tokens
- Timestep embedding MLP (sinusoidal frequency stays on CPU — only the projected embedding enters)
- AdaLN / FiLM conditioning
- Output projection to action space

**Outside the compiled graph** (CPU):
- The N-step denoising loop
- Noise schedule computation (timestep sequence, sigma values)
- Python control flow over steps
- VLM subgraph execution (runs once before the loop)

### Interface

```python
def forward(
    self,
    noisy_actions,        # [B, action_chunk_size, action_dim]        BF16
    conditioning_tokens,  # [B, num_conditioning_tokens, hidden_size] BF16
    timestep_embedding,   # [B, timestep_embed_dim]                   BF16
    attention_mask,       # [B, 1, action_chunk_size,
                          #    num_conditioning_tokens]                INT32
) -> torch.Tensor:        # [B, action_chunk_size, action_dim]        BF16
```

All four input shapes must be static at compile time.

### Static Shape Contract

**`action_chunk_size`** — read from model config (`chunk_size`, `pred_horizon`, `action_horizon`). Typical: 16 for GR00T, 100 for Diffusion Policy.

**`num_conditioning_tokens`** — number of VLM output tokens passed as cross-attention KV. Must exactly match the sequence length produced by the VLM for the chosen vision bucket:
```
num_conditioning_tokens = num_text_tokens + num_vision_tokens_for_bucket
```
If the VLM uses multiple vision buckets, either compile a separate denoiser bucket per VLM bucket, or pad VLM output to a fixed maximum.

**`timestep_embed_dim`** — output dimension of the timestep embedding projection. Read from config.

### Bucketing

For most VLAs (GR00T, SmolVLA) action_chunk_size and num_conditioning_tokens are fixed — a single bucket suffices. Bucket only if these vary.

---

## Part 3: CPU-Side Denoising Loop

```python
def generate_actions(self, conditioning_tokens, num_steps: int) -> torch.Tensor:
    # 1. Compute noise schedule on CPU
    timesteps = self._get_timestep_sequence(num_steps)  # list of Python floats

    # 2. Sample initial noise on CPU
    noisy_actions = torch.randn(B, self.action_chunk_size, self.action_dim, dtype=torch.bfloat16)

    # 3. Denoising loop — calls compiled graph N times
    for t in timesteps:
        timestep_emb = self._embed_timestep(t)  # CPU, returns [B, embed_dim]
        noisy_actions = self.denoising_wrapper(
            noisy_actions, conditioning_tokens, timestep_emb, self.attention_mask
        )

    return noisy_actions
```

- `num_steps` must be a Python int — never a tensor
- `timesteps` is computed entirely on CPU; only the projected `timestep_emb` crosses into the compiled graph
- `conditioning_tokens` is computed once before the loop (VLM output) and passed unchanged on every step
- For flow matching, `timesteps` is a linear sequence from 1.0 to 0.0

---

## Part 4: Primitive Replacement Map

| Source op | NxDI replacement | Notes |
|---|---|---|
| Self-attention (action tokens) | `NeuronAttentionBase` subclass | Standard |
| Cross-attention (action → conditioning) | `NeuronAttentionBase` subclass | Fixed KV shapes |
| `nn.Linear` gate/up | `ColumnParallelLinear` | |
| `nn.Linear` down/out | `RowParallelLinear` | |
| Timestep sinusoidal encoding | Keep on CPU | Do not trace |
| Timestep projection MLP | `ColumnParallelLinear` + `RowParallelLinear` | Input is projected embedding |
| AdaLN / FiLM conditioning | Custom implementation | No NxDI primitive |

---

## Part 4b: Compiler Args for DiT Subgraphs

Use `-O1` only. **Never use `--model-type=transformer` on a DiT** — it replaces softmax with a custom NKI kernel that is numerically inaccurate for DiT/flow-matching models (cos_sim≈0.916, 37% error per step).

The correct default is set in `NeuronActionHeadBase.get_compiler_args()`:
```python
def get_compiler_args(self) -> str:
    return (
        "--auto-cast=none "
        "-O1 "
        "--tensorizer-options='"
        "--enable-ccop-compute-overlap "
        "--cc-pipeline-tiling-factor=1'"
    )
```

---

## Part 5: Gotchas

**Dynamic iteration count** — `num_steps` must never enter the compiled graph as a tensor. Convert to Python int before the loop.

**Dynamic constants in denoising forward** — `torch.arange` for position IDs, `torch.ones` for masks, RoPE frequencies — pre-compute ALL as `register_buffer()` in `__init__()`. With 32 DiT layers this causes `[Errno 36] File name too long` at compile time.

**TP=1 from wrong construction order** — if the DiT model is constructed in `__init__()`, the NEFF is silently TP=1, weight sharding produces N identical copies instead of N shards, and NEFF loading fails with "Expected weight tensors for N ranks. Received 1."

**`load_state_dict` must accept `**kwargs`** — signature: `(self, state_dict, strict=True, **kwargs)`. NxDI internally passes `assign=True`. Missing `**kwargs` causes a TypeError that, if swallowed, produces a silent missing-NEFF failure.

**Never wrap `compile_denoiser()` in a bare exception handler** — failures must propagate.

**Noise schedule on CPU** — the full schedule array must be computed on CPU before the loop. Only the scalar or projected embedding for the current step crosses into the compiled graph. Passing a schedule tensor into the graph will fail tracing or cause FP32 casting.

**Shape contract violations** — the most common runtime failure. Assert `conditioning_tokens.shape[1] == self.num_conditioning_tokens` before the loop.

**AdaLN inside the compiled graph** — adaptive layer norm is traceable as long as scale/shift vectors are derived from the input tensor. If the original model uses Python conditionals over timestep values, replace them with tensor ops.

**BF16 accumulation across steps** — small errors can compound across N denoising steps. If error grows monotonically, look for a dtype cast inside the compiled graph converting BF16 to FP32 and back.

---

## Part 6: Validation

After compiling, validate against HF CPU reference:

1. CPU unit test for the cross-attention block — `test_block_correctness` with synced weights, `atol=1e-3`
2. CPU unit test for the DiT MLP / AdaLN block — `test_block_correctness`, `atol=1e-3`
3. Full N-step denoising loop on CPU matches HF reference output within `atol=1e-2`
4. Denoiser NEFF compiles with zero CPU fallback ops — verify with `torch_neuronx.analyze`
5. Integration smoke test — VLM output fed into action head, denoising loop completes, output is non-degenerate (no NaN, no all-zeros, physically plausible values)
