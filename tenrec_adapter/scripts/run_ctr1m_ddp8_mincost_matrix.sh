#!/bin/bash
set -euo pipefail

# Low-compute 8-card experiment matrix for ctr_data_1M ranking.
# Stage A only: quick screening before full training.
#
# Usage:
#   bash tenrec_adapter/scripts/run_ctr1m_ddp8_mincost_matrix.sh
#
# Default matrix:
#   A1: neg127, L12, ls0.10
#   A2: neg63,  L12, ls0.10
#   A3: neg127, L16, ls0.10
#   A4: neg127, L12, ls0.05

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

MAX_USERS="${MAX_USERS:-200000}"
EPOCHS="${EPOCHS:-1}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-100000}"
MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-1200}"

BATCH_SIZE="${BATCH_SIZE:-2048}"
RANKING_BATCH_SIZE="${RANKING_BATCH_SIZE:-2048}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs/ctr1m_ddp8_mincost_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

run_exp() {
  local tag="$1"
  local neg="$2"
  local layers="$3"
  local ls="$4"
  local ls_tag="$5"
  local port="$6"
  local run_name="ctr1m_ddp8_${tag}_neg${neg}_L${layers}_${ls_tag}"
  local log_file="${LOG_DIR}/${run_name}.log"

  echo "===================================================="
  echo "START ${run_name}"
  echo "===================================================="

  "${PYTHON_BIN}" -u -m "${TORCH_LAUNCH_MODULE}" \
    --nproc_per_node="${NPROC}" \
    --master_port="${port}" \
    tenrec_adapter/run_ddp_train.py \
    --stage "ranking" \
    --scenario "ctr_data_1M" \
    --data_dir "${DATA_DIR}" \
    --run_name "${run_name}" \
    --encoder_type "fastformer" \
    --ranking_num_layers "${layers}" \
    --num_negatives "${neg}" \
    --batch_size "${BATCH_SIZE}" \
    --ranking_batch_size "${RANKING_BATCH_SIZE}" \
    --grad_accumulation "${GRAD_ACCUM}" \
    --epochs "${EPOCHS}" \
    --lr 0.0008 \
    --warmup_steps 800 \
    --weight_decay 0.01 \
    --embed_dim 512 \
    --hidden_dim 1024 \
    --num_heads 8 \
    --label_smoothing "${ls}" \
    --patience 2 \
    --eval_interval 100000 \
    --history_seq_len 10 \
    --max_users "${MAX_USERS}" \
    --max_eval_samples "${MAX_EVAL_SAMPLES}" \
    --max_steps_per_epoch "${MAX_STEPS_PER_EPOCH}" \
    --seed 2026 \
    --num_workers "${NUM_WORKERS}" \
    --gradient_checkpointing \
    2>&1 | tee "${log_file}"

  echo "END ${run_name}"
}

run_exp "A1" 127 12 0.10 "ls010" 29601
run_exp "A2" 63  12 0.10 "ls010" 29602
run_exp "A3" 127 16 0.10 "ls010" 29603
run_exp "A4" 127 12 0.05 "ls005" 29604

echo "All matrix experiments done."
echo "Logs: ${LOG_DIR}"
