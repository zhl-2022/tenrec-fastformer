#!/bin/bash
set -euo pipefail

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
PHASES="${PHASES:-P0,P1,P2,P3}"

BEST_NEG="${BEST_NEG:-127}"
BEST_LAYERS="${BEST_LAYERS:-12}"
BEST_LS="${BEST_LS:-0.10}"

contains_phase() {
  local needle="$1"
  [[ ",${PHASES}," == *",${needle},"* ]]
}

run_p0() {
  local run_name="ctr1m_ddp8_p0_smoke_$(date +%Y%m%d_%H%M%S)"
  echo "[P0] Smoke check: ${run_name}"
  "${PYTHON_BIN}" -u -m "${TORCH_LAUNCH_MODULE}" \
    --nproc_per_node="${NPROC}" \
    --master_port=29801 \
    tenrec_adapter/run_ddp_train.py \
    --stage ranking \
    --scenario ctr_data_1M \
    --data_dir "${DATA_DIR}" \
    --run_name "${run_name}" \
    --encoder_type fastformer \
    --ranking_num_layers 12 \
    --num_negatives 63 \
    --batch_size 1024 \
    --ranking_batch_size 1024 \
    --epochs 1 \
    --max_users 50000 \
    --max_eval_samples 20000 \
    --max_steps_per_epoch 200 \
    --num_workers 2 \
    --seed 2026
}

run_p1() {
  echo "[P1] Min-cost matrix"
  MAX_USERS="${MAX_USERS:-200000}" \
  MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-100000}" \
  MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-1200}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" \
  bash tenrec_adapter/scripts/run_ctr1m_ddp8_mincost_matrix.sh
}

run_p2() {
  echo "[P2] Full training with best params: neg=${BEST_NEG}, layers=${BEST_LAYERS}, ls=${BEST_LS}"
  NUM_WORKERS="${NUM_WORKERS:-8}" \
  MAX_STEPS_PER_EPOCH="${MAX_STEPS_PER_EPOCH:-0}" \
  bash tenrec_adapter/scripts/run_ctr1m_ddp8_full_train.sh "${BEST_NEG}" "${BEST_LAYERS}" "${BEST_LS}"
}

run_p3() {
  echo "[P3] Rust pipeline (optional)"
  if [[ "${ENABLE_RUST_PIPELINE:-0}" != "1" ]]; then
    echo "[P3] Skip. Set ENABLE_RUST_PIPELINE=1 to build rust_pipeline."
    return 0
  fi

  (cd rust_pipeline && bash build.sh release)
  "${PYTHON_BIN}" - <<'PY'
from tenrec_adapter.rust_components import is_rust_available
print(f"rust_components_available={is_rust_available()}")
PY
}

contains_phase "P0" && run_p0
contains_phase "P1" && run_p1
contains_phase "P2" && run_p2
contains_phase "P3" && run_p3

echo "All requested phases finished."
