#!/bin/bash
set -euo pipefail

# Full 8-card training after selecting best params from min-cost matrix.
#
# Usage:
#   bash tenrec_adapter/scripts/run_ctr1m_ddp8_full_train.sh 127 12 0.05
#   # args: <num_negatives> <ranking_layers> <label_smoothing>

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <num_negatives> <ranking_layers> <label_smoothing>"
  exit 1
fi

NEG="$1"
LAYERS="$2"
LS="$3"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CNCL_TIMEOUT="${CNCL_TIMEOUT:-1800}"
export TORCH_DIST_TIMEOUT="${TORCH_DIST_TIMEOUT:-3600}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_LAUNCH_MODULE="${TORCH_LAUNCH_MODULE:-torch.distributed.run}"
NPROC="${NPROC:-8}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/tenrec/Tenrec}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-0}"

RUN_NAME="${RUN_NAME:-ctr1m_ddp8_final_neg${NEG}_L${LAYERS}_ls${LS}}"

"${PYTHON_BIN}" -u -m "${TORCH_LAUNCH_MODULE}" \
  --nproc_per_node="${NPROC}" \
  --master_port=29701 \
  tenrec_adapter/run_ddp_train.py \
  --stage "ranking" \
  --scenario "ctr_data_1M" \
  --data_dir "${DATA_DIR}" \
  --run_name "${RUN_NAME}" \
  --encoder_type "fastformer" \
  --ranking_num_layers "${LAYERS}" \
  --num_negatives "${NEG}" \
  --batch_size 2048 \
  --ranking_batch_size 2048 \
  --grad_accumulation 1 \
  --epochs 3 \
  --lr 0.0008 \
  --warmup_steps 800 \
  --weight_decay 0.01 \
  --embed_dim 512 \
  --hidden_dim 1024 \
  --num_heads 8 \
  --label_smoothing "${LS}" \
  --patience 5 \
  --eval_interval 100000 \
  --history_seq_len 10 \
  --max_eval_samples 500000 \
  --max_steps_per_epoch "${MAX_STEPS_PER_EPOCH}" \
  --seed 2026 \
  --num_workers "${NUM_WORKERS}" \
  --gradient_checkpointing
