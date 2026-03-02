## Qwen2-VL on AWS Trainium (NxDI)

This directory contains an end-to-end multimodal inference script for running **Qwen2-VL** on **AWS Trainium** using **Neuronx Distributed Inference (NxDI)**.

### Environment

- **Activate Neuron PyTorch inference environment** (example):

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
cd ~/model-translation/qwen2-vl
```

Make sure your environment has:

- `transformers` with Qwen2-VL support
- `torch` built for Neuron
- `neuronx_distributed_inference`

### Model checkpoints

Download or place a Qwen2-VL checkpoint (for example `Qwen/Qwen2-VL-7B-Instruct`) in a local directory, e.g.:

- Default expected path (can be overridden with `--model_path`):

```bash
~/models/qwen2-vl-7b
```

### One-time compile + run

The first run will **compile** the model to Neuron, then perform inference:

```bash
cd ~/model-translation/qwen2-vl

python run_multimodal_inference.py \
    --model_path ~/models/qwen2-vl-7b \
    --compiled_model_path ~/models/qwen2-vl-multimodal-compiled \
    --image_path IMG_4292.jpg \
    --prompt "Describe this image."
```

- **`--model_path`**: HF Qwen2-VL checkpoint (defaults to `~/models/qwen2-vl-7b`)
- **`--compiled_model_path`**: where Neuron artifacts will be written (defaults to `~/models/qwen2-vl-multimodal-compiled`)
- **`--image_path`**: path to an input image (omit for text-only prompts)
- **`--prompt`**: text prompt for the model

### Re-using a compiled model (`--skip_compile`)

If you have already compiled the model and just want to run inference:

```bash
python run_multimodal_inference.py \
    --model_path ~/models/qwen2-vl-7b \
    --compiled_model_path ~/models/qwen2-vl-multimodal-compiled \
    --image_path IMG_4292.jpg \
    --prompt "Describe this image." \
    --skip_compile
```

If `--skip_compile` is set and the compiled artifacts are missing, the script will fail with a clear error asking you to compile first.

### Text-only example

You can also run Qwen2-VL in text-only mode by omitting `--image_path`:

```bash
python run_multimodal_inference.py \
    --model_path ~/models/qwen2-vl-7b \
    --compiled_model_path ~/models/qwen2-vl-multimodal-compiled \
    --prompt "Explain what Qwen2-VL is and what it can do."
```

### Useful arguments

- `--max_new_tokens`: maximum tokens to generate (default: `128`)
- `--tp_degree`: tensor parallel degree (default: `8`)
- `--seq_len`: text sequence length (default: `1024`)
- `--text_buckets`: text context bucket sizes (default: `1024`)
- `--vision_buckets`: vision patch sequence bucket sizes (default: `256 1024`)
- `--skip_compile`: skip compilation and only load a pre-compiled model
- `--compile_only`: compile and exit without running inference
- `--debug`: enable verbose logging during compile and run

