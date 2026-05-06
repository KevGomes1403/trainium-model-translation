#!/usr/bin/env bash
# Closed-loop LIBERO demo runner for SmolVLA on Trainium.
#
# Usage:
#   ./run_demo.sh                         # defaults: libero_object task 0 seed 7
#   ./run_demo.sh --task 1 --seed 42
#   ./run_demo.sh --suite libero_spatial --task 2 --seed 0 --output mydemo.mp4
#   ./run_demo.sh --steps 250 --replan 1
#
# Flags:
#   --suite     libero_object | libero_spatial | libero_goal | libero_10 | libero_90
#   --task      task index (0..N-1 in suite)
#   --seed      initial-state seed (changes object positions / scene)
#   --steps     max env steps (default 250)
#   --replan    chunk actions to execute before replanning (default 1 — finetune n_action_steps=1)
#   --output    output mp4 path (default ./demo_<suite>_t<task>_s<seed>.mp4)

set -euo pipefail

SUITE="libero_object"
TASK=0
SEED=7
STEPS=250
REPLAN=1
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite)  SUITE="$2"; shift 2 ;;
        --task)   TASK="$2"; shift 2 ;;
        --seed)   SEED="$2"; shift 2 ;;
        --steps)  STEPS="$2"; shift 2 ;;
        --replan) REPLAN="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,/^set/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="demo_${SUITE}_t${TASK}_s${SEED}.mp4"
fi

# Resolve checkpoint + NEFF dir from environment, with sane defaults.
CKPT="${SMOLVLA_CKPT:-$(ls -d /home/ubuntu/.cache/huggingface/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/*/ 2>/dev/null | head -1)}"
NEFF="${SMOLVLA_NEFF:-/home/ubuntu/vla/smol_vla_neff_libero_hfvla}"

if [[ -z "$CKPT" || ! -d "$CKPT" ]]; then
    echo "ERROR: SmolVLA checkpoint not found. Set SMOLVLA_CKPT or download HuggingFaceVLA/smolvla_libero." >&2
    echo "  python -c \"from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='HuggingFaceVLA/smolvla_libero'))\"" >&2
    exit 1
fi
if [[ ! -d "$NEFF" ]]; then
    echo "ERROR: NEFF dir not found at $NEFF. Compile first:" >&2
    echo "  python -m smol_vla.run_inference --action compile --hf-checkpoint \"$CKPT\" --neff-dir \"$NEFF\"" >&2
    exit 1
fi

# Activate the Neuron venv if not already on PATH
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
fi

cd "$(dirname "$0")/.."

echo "============================================================"
echo "  SmolVLA on Trainium — closed-loop LIBERO demo"
echo "  suite : $SUITE   task : $TASK   seed : $SEED"
echo "  steps : $STEPS   replan : every $REPLAN   output : $OUTPUT"
echo "============================================================"

# `yes 'N'` answers LIBERO's first-time setup prompt non-interactively.
yes 'N' | python -m smol_vla.demo_libero \
    --hf-checkpoint "$CKPT" \
    --neff-dir      "$NEFF" \
    --suite         "$SUITE" \
    --task-id       "$TASK" \
    --seed          "$SEED" \
    --max-steps     "$STEPS" \
    --replan-every  "$REPLAN" \
    --output        "$OUTPUT"

echo
echo "Saved $OUTPUT"
