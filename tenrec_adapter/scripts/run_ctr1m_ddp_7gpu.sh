#!/bin/bash
# =============================================================================
# ctr_data_1M DDP Training Script - 7 MLU cards
# =============================================================================

set -e

# ===== Configuration =====
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="ctr1m_ddp7_ranking_sideinfo"
RUN_NAME="${EXP_NAME}"
LOG_DIR="${PROJECT_ROOT}/logs/${EXP_NAME}_${TIMESTAMP}"
TMUX_SESSION="ctr1m_ddp7_${TIMESTAMP}"

DATA_DIR="${PROJECT_ROOT}/data/tenrec/Tenrec"

NUM_GPUS=7
MASTER_PORT=29500
export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6

PYTHON="python3"

mkdir -p "${LOG_DIR}"

echo "=============================================="
echo "ctr_data_1M DDP Training - 7 MLU"
echo "=============================================="
echo "Log Directory: ${LOG_DIR}"
echo "Tmux Session:  ${TMUX_SESSION}"
echo "Dataset:       ctr_data_1M"
echo "MLU Visible:   ${MLU_VISIBLE_DEVICES}"
echo "Stage:         ranking"
echo "Run Name:      ${RUN_NAME}"
echo "=============================================="

TRAIN_SCRIPT="${LOG_DIR}/run_command.sh"
cat > "${TRAIN_SCRIPT}" <<EOF
#!/bin/bash
export PYTHONPATH=\$PYTHONPATH:${PROJECT_ROOT}
export MLU_VISIBLE_DEVICES=${MLU_VISIBLE_DEVICES}
TORCH_LAUNCH_MODULE="\${TORCH_LAUNCH_MODULE:-torch.distributed.run}"
cd ${PROJECT_ROOT}

echo "Starting DDP training at \$(date)"
echo "World Size: ${NUM_GPUS}"
echo "MLU_VISIBLE_DEVICES: \${MLU_VISIBLE_DEVICES}"
echo "Logs being written to ${LOG_DIR}/train.log"

${PYTHON} -u -m \${TORCH_LAUNCH_MODULE} \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    tenrec_adapter/run_ddp_train.py \
    --stage "ranking" \
    --scenario "ctr_data_1M" \
    --run_name "${RUN_NAME}" \
    --data_dir "${DATA_DIR}" \
    --encoder_type "fastformer" \
    --ranking_num_layers 12 \
    --num_negatives 127 \
    --batch_size 2048 \
    --ranking_batch_size 2048 \
    --grad_accumulation 4 \
    --epochs 4 \
    --lr 0.001 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --embed_dim 512 \
    --hidden_dim 1024 \
    --num_heads 8 \
    --temperature 0.07 \
    --label_smoothing 0.1 \
    --patience 5 \
    --history_seq_len 10 \
    --max_eval_samples 500000 \
    --seed 2026 \
    --num_workers 8 \
    --gradient_checkpointing \
    2>&1 | tee "${LOG_DIR}/train.log"

echo "Training finished at \$(date)"
EOF

chmod +x "${TRAIN_SCRIPT}"

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "Session '${TMUX_SESSION}' already exists."
    exit 1
fi

tmux new-session -d -s "${TMUX_SESSION}" "bash ${TRAIN_SCRIPT}; read"

echo ""
echo "DDP training started in tmux session."
echo "Attach:    tmux attach -t ${TMUX_SESSION}"
echo "View log:  tail -f ${LOG_DIR}/train.log"
echo "Stop:      tmux kill-session -t ${TMUX_SESSION}"
