#!/bin/bash
# =============================================================================
# ctr_data_1M Retrieval (粗排/召回) Training Script
#
# Dataset: ctr_data_1M (Tenrec CTR Leaderboard, 1M users, ~1.55 亿行)
#   - Pre-built history (hist_1~hist_10)
#
# Hardware: Cambrian MLU590-H8 (85GB)
#   - TwoTowerModel lighter than Fastformer
#   - history_seq_len=16 (only 10 hist items) → much less VRAM
#   - batch 8192 safe with short history + lighter model
#
# Training Time Estimate (full dataset):
#   - Steps/epoch: ~124M / 8192 ≈ 15,137
#   - ~0.5s/step → ~2.1 hours/epoch → 4 epochs ≈ 8.4 hours
# =============================================================================

set -e

# ===== Configuration =====
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="ctr1m_retrieval_neg127_dim512"
LOG_DIR="${PROJECT_ROOT}/logs/${EXP_NAME}_${TIMESTAMP}"
TMUX_SESSION="ctr1m_retr_${TIMESTAMP}"
DATA_DIR="${PROJECT_ROOT}/data/tenrec/Tenrec"

# ===== Create Directory =====
mkdir -p "${LOG_DIR}"

echo "=============================================="
echo "🚀 ctr_data_1M Retrieval (粗排/召回) Training"
echo "=============================================="
echo "Log Directory: ${LOG_DIR}"
echo "Tmux Session:  ${TMUX_SESSION}"
echo "=============================================="

# ===== Create Inner Training Script =====
TRAIN_SCRIPT="${LOG_DIR}/run_command.sh"
cat > "${TRAIN_SCRIPT}" <<EOF
#!/bin/bash
export PYTHONPATH=\$PYTHONPATH:${PROJECT_ROOT}
cd ${PROJECT_ROOT}

echo "Starting ctr_data_1M Retrieval training at \$(date)"
echo "Logs being written to ${LOG_DIR}/train.log"

python -u tenrec_adapter/run_two_stage_train.py \\
    --stage "retrieval" \\
    --scenario "ctr_data_1M" \\
    --data_dir "${DATA_DIR}" \\
    --num_negatives 127 \\
    --batch_size 8192 \\
    --epochs 4 \\
    --lr 0.0005 \\
    --warmup_steps 300 \\
    --weight_decay 0.01 \\
    --embed_dim 512 \\
    --hidden_dim 1024 \\
    --num_heads 8 \\
    --label_smoothing 0.1 \\
    --patience 5 \\
    --eval_interval 5000 \\
    --history_seq_len 10 \\
    --max_eval_samples 500000 \\
    --seed 2026 \\
    2>&1 | tee "${LOG_DIR}/train.log"

echo "Training finished at \$(date)"
EOF

chmod +x "${TRAIN_SCRIPT}"

# ===== Start Tmux Session =====
if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "⚠️  Session '${TMUX_SESSION}' already exists."
    exit 1
fi

tmux new-session -d -s "${TMUX_SESSION}" "bash ${TRAIN_SCRIPT}; read"

echo ""
echo "✅ Retrieval training started in background tmux session!"
echo ""
echo "📋 Commands:"
echo "  Attach:      tmux attach -t ${TMUX_SESSION}"
echo "  View Logs:   tail -f ${LOG_DIR}/train.log"
echo "  Kill:        tmux kill-session -t ${TMUX_SESSION}"
echo ""
echo "📌 After retrieval training, run ranking:"
echo "  bash tenrec_adapter/scripts/run_ctr1m_ranking.sh"
echo ""
