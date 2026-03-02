#!/usr/bin/env python3
"""Run NxDI inference_demo with local OLMo-3 model registration."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Ensure local "src" package imports resolve when launched from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuronx_distributed_inference import inference_demo
from neuronx_distributed_inference.models.config import to_torch_dtype
from modeling_qwen2vl_neuron import NeuronQwen2VLForCausalLM
from transformers import AutoConfig, PretrainedConfig

# os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
# os.environ["XLA_IR_DEBUG"]= "1"
# os.environ["XLA_HLO_DEBUG"]= "1"
# os.environ["NEURON_RT_INSPECT_ENABLE"]= "1"
# os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"]= "1"
# os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"]= "./output"


def _load_pretrained_config_compat(
    model_path_or_name=None,
    hf_config=None,
):
    """Compatibility loader for checkpoints with newer HF model_type values."""

    def load_config(self):
        if (model_path_or_name is None and hf_config is None) or (
            model_path_or_name is not None and hf_config is not None
        ):
            raise ValueError('Please provide only one of "model_path_or_name" or "hf_config"')

        if model_path_or_name is not None:
            try:
                config = AutoConfig.from_pretrained(model_path_or_name, trust_remote_code=True)
            except Exception:
                # Fallback for older transformers that do not recognize `model_type`.
                config_path = Path(model_path_or_name).expanduser() / "config.json"
                with open(config_path, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)
                config = PretrainedConfig.from_dict(config_dict)
                if "transformers_version" in config_dict:
                    config.transformers_version = config_dict["transformers_version"]
        else:
            config = hf_config

        config_dict = config.to_dict()

        if config.transformers_version is not None:
            config_dict["transformers_version"] = config.transformers_version

        hf_dtype = config_dict.get("dtype", config_dict.get("torch_dtype", None))
        if hf_dtype is not None:
            if self.neuron_config is not None and not self.neuron_config.overrides_torch_dtype:
                self.neuron_config.torch_dtype = hf_dtype
                if isinstance(self.neuron_config.torch_dtype, str):
                    self.neuron_config.torch_dtype = to_torch_dtype(self.neuron_config.torch_dtype)
            config_dict.pop("dtype", None)
            config_dict.pop("torch_dtype", None)

        for k, v in config_dict.items():
            if isinstance(getattr(config, k, None), PretrainedConfig):
                config_dict[k] = SimpleNamespace(**v)
                if config.transformers_version is not None:
                    config_dict[k].transformers_version = config.transformers_version

        self.__dict__.update(config_dict)
        if hasattr(config, "attribute_map"):
            self.attribute_map = config.attribute_map

    return load_config


def main():
    # Register local model implementation for the standard demo CLI.
    inference_demo.MODEL_TYPES["qwen2_vl"] = {"causal-lm": NeuronQwen2VLForCausalLM}
    inference_demo.load_pretrained_config = _load_pretrained_config_compat

    inference_demo.main()


if __name__ == "__main__":
    main()
