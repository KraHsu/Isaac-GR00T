#!/usr/bin/env bash
set -euo pipefail

export BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/zch/workspace/GR00T-N1.7-3B}"
export OUTPUT_DIR="${OUTPUT_DIR:-./outputs/pi05_finetune}"
export NUM_GPUS="${NUM_GPUS:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export USE_WANDB="${USE_WANDB:-0}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"  # use ffmpeg on dev machine (no torchcodec)

SCRIPT="examples/PI05/launch_finetune_pi05.py"

if [ "$NUM_GPUS" = "1" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    exec python "$SCRIPT"
else
    exec torchrun --nproc_per_node="$NUM_GPUS" --master_port="${MASTER_PORT:-29500}" "$SCRIPT"
fi
