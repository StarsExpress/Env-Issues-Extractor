#!/usr/bin/env bash
set -euo pipefail
set -x  # echo every command for debugging

ENV_NAME="env-extract-code"
module load anaconda3
conda activate "$ENV_NAME"

echo "Using python: $(which python)"
python -V
nvidia-smi || true

# portable background GPU util logger (every 30 s)
( while true; do
    echo "$(date +%F,%T),$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits)" >> gpu_util.log
    sleep 10
  done ) &
MON_PID=$!

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR"

export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1

python main.py "$@"

# stop monitor
kill $MON_PID || true
