# Neuron Model Translation

This repository provides reusable agent skills for translating Hugging Face models to AWS Trainium chips with minimal human-in-the-loop effort. The skills are designed to plug into existing agent tools (Claude Code, Codex, Cursor, and similar workflows) so model porting can be repeated consistently.

Porting HF models to Trainium is usually time-consuming and easy to get wrong when done manually across many model architectures. The goal here is to package the process into practical, reusable skills that agents can execute end-to-end: block translation, Neuron implementation, validation, and inference bring-up.

In short, this repo turns model translation into an agent workflow:

- Plug skills into your agent tool
- Start from an HF/PyTorch model
- Translate it into Neuron/NxDI-compatible code
- Validate correctness (weights + outputs)
- Run inference on Trainium with minimal manual intervention

## Trainium and Neuron

`Trainium` is AWS's purpose-built ML accelerator hardware. `Neuron` is the SDK/runtime/tooling stack used to compile and run models on Trainium (and Inferentia).

In this repo, translations are implemented with `NxDI` (Neuronx Distributed Inference), which provides:

- Neuron model primitives (parallel linear/embedding layers, attention bases, etc.)
- Distributed inference patterns such as tensor parallelism
- A model framework for adapting HF architectures to Neuron execution

## Project Goal

The core goal is autonomously translating HF model implementations to Neuron hardware with reliable, repeatable engineering workflows. That includes both text-only and multimodal models.

## Repository Layout

- `models/` - Example translations and runnable inference scripts
- `agents/` - Specialized agent definitions/prompts, including the NxDI block translation agent
- `docs/` - Notes and technical references
- `skills/` - Reusable agent skills that can be used with existing coding-agent tools (for example Claude Code, Codex, Cursor)

## Current Example Translations

Under `models/`:

- `arcee-4.5b-base`
- `gemma-2-9b`
- `olmo-3`
- `qwen2-vl`
- `qwen2.5-vl`
- `smol_vla` - SmolVLA LIBERO policy port to Trainium 3 with a closed-loop demo

All model ports currently in this repository were translated using the skill workflow with Claude Code.

Model folders typically include:

- A PyTorch/HF reference implementation
- A Neuron/NxDI implementation (`modeling_*_neuron.py`)
- Inference entrypoints (`inference_*.py` or `run_*.py`)
- Optional weight-mapping validation scripts

For a multimodal example walkthrough, see:

- `models/qwen2-vl/README.md`
- `models/smol_vla/README.md`

## Status

Active development workspace. Existing model directories are working examples and templates for future autonomous HF-to-Neuron ports.
