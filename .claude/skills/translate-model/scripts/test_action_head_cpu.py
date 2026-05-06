# test_action_head_cpu.py
import torch
import torch.nn as nn
from neuron_action_head_base import (
    NeuronDenoisingConfig,
    NeuronDenoisingWrapper,
    NeuronActionHeadBase,
    ConditioningContract,
)

# --- Minimal concrete wrapper ---
class DummyDenoisingWrapper(NeuronDenoisingWrapper):
    def __init__(self, config):
        super().__init__(config)
        # Simple linear just to have weights
        self.proj = nn.Linear(config.action_dim, config.action_dim)

    def forward(self, noisy_actions, conditioning_tokens,
                timestep_embedding, attention_mask):
        # Dummy: just return noisy_actions unchanged
        return noisy_actions

# --- Minimal concrete action head ---
class DummyActionHead(NeuronActionHeadBase):
    def get_conditioning_contract(self):
        return ConditioningContract(
            num_conditioning_tokens=128,
            conditioning_hidden_size=64,
        )

    def _build_denoising_wrapper(self):
        return DummyDenoisingWrapper(self.config)

    def _get_timestep_sequence(self, num_steps):
        return [1.0 - i / num_steps for i in range(num_steps)]

    def _embed_timestep(self, t):
        B = self.config.neuron_config.batch_size
        return torch.zeros(B, self.config.timestep_embed_dim,
                           dtype=torch.bfloat16)

    def _build_attention_mask(self):
        B = self.config.neuron_config.batch_size
        return torch.ones(
            B, 1,
            self.config.action_chunk_size,
            self.config.num_conditioning_tokens,
            dtype=torch.int32,
        )

# --- Tests ---
cfg = NeuronDenoisingConfig(
    batch_size=1, tp_degree=1,
    action_chunk_size=16, action_dim=7,
    num_conditioning_tokens=128, conditioning_hidden_size=64,
    timestep_embed_dim=64,
)

head = DummyActionHead(model_path="/tmp/dummy", config=cfg)

# Test compile_denoiser on CPU (no hardware — should build but not compile)
head.compile_denoiser("/tmp/dummy_compiled/")
assert head.denoising_wrapper is not None
assert head.attention_mask is not None
print("compile_denoiser CPU mode: OK")

# Test pre_compile_validate
head.pre_compile_validate()
print("pre_compile_validate: OK")

# Test generate_actions on CPU
conditioning = torch.zeros(1, 128, 64, dtype=torch.bfloat16)
actions = head.generate_actions(conditioning, num_steps=3)
assert actions.shape == (1, 16, 7)
assert not torch.isnan(actions).any()
print("generate_actions CPU: OK")

# Test contract validation catches wrong shape
try:
    bad_tokens = torch.zeros(1, 64, 64, dtype=torch.bfloat16)  # wrong token count
    head.verify_contract_against_vlm(bad_tokens)
    assert False, "Should have raised"
except AssertionError:
    print("ConditioningContract shape check: OK")

print("\nAll CPU tests passed.")