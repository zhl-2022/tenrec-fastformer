#!/bin/bash
# =============================================================================
# ctr_data_1M Ranking (精排) Training Script — Plan C (均衡方案)
#
# Dataset: ctr_data_1M (Tenrec CTR Leaderboard, 1M users, ~1.55 亿行)
#   - 20 columns: basic 10 + hist_1~hist_10 (pre-built user history)
#   - Pre-built history eliminates dynamic history construction overhead
#
# Hardware: Cambrian MLU590-H8 (85GB)
#   - gradient_checkpointing: 以~30%训练时间换取~10x激活显存节省
#   - 实测基准: batch=3072, neg255, seq=267 → 53.8GB (64%)
#   - 本轮: batch=8192, neg127, seq=139 → 预估 ~71GB (84%)
#   - grad_accumulation=1: 有效 batch 8192 (等效之前 4096×2)
#   - LR 保持 0.0008 (有效 batch 不变)
#
# Training Time Estimate (基于实测 ~4.2s/step @ batch=3072, seq=267):
#   - train samples ≈ 96.3M
#   - Steps/epoch: 96,300,000 / 8192 ≈ 11,755
#   - Time/step: ~4.2 × (8192×139)/(3072×267) ≈ 5.8s
#   - Per epoch: 11,755 × 5.8s ≈ 19 hours
#   - 3 epochs ≈ 57 hours (~2.4 days)  ← 比 neg255 快近一倍
#   - Eval every 15000 steps
# =============================================================================

set -e

# ===== Configuration =====
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="ctr1m_ranking_neg127_dim512_L12_b8192"
LOG_DIR="${PROJECT_ROOT}/logs/${EXP_NAME}_${TIMESTAMP}"
TMUX_SESSION="ctr1m_rank_${TIMESTAMP}"
DATA_DIR="${PROJECT_ROOT}/data/tenrec/Tenrec"

# ===== Create Directory =====
mkdir -p "${LOG_DIR}"

echo "=============================================="
echo "🚀 ctr_data_1M Ranking (精排) Training — Speed Optimized"
echo "=============================================="
echo "Log Directory: ${LOG_DIR}"
echo "Tmux Session:  ${TMUX_SESSION}"
echo "Dataset:       ctr_data_1M (1M users, pre-built history)"
echo "=============================================="

# ===== Create Inner Training Script =====
TRAIN_SCRIPT="${LOG_DIR}/run_command.sh"
cat > "${TRAIN_SCRIPT}" <<EOF
#!/bin/bash
export PYTHONPATH=\$PYTHONPATH:${PROJECT_ROOT}
cd ${PROJECT_ROOT}

echo "Starting ctr_data_1M Ranking training at \$(date)"
echo "Logs being written to ${LOG_DIR}/train.log"

python -u tenrec_adapter/run_two_stage_train.py \\
    --stage "ranking" \\
    --scenario "ctr_data_1M" \\
    --data_dir "${DATA_DIR}" \\
    --encoder_type "fastformer" \\
    --ranking_num_layers 12 \\
    --num_negatives 127 \\
    --batch_size 3584 \\
    --ranking_batch_size 8192 \\
    --grad_accumulation 1 \\
    --epochs 3 \\
    --lr 0.0008 \\
    --warmup_steps 800 \\
    --weight_decay 0.01 \\
    --embed_dim 512 \\
    --hidden_dim 1024 \\
    --num_heads 8 \\
    --label_smoothing 0.1 \\
    --patience 5 \\
    --eval_interval 15000 \\
    --history_seq_len 10 \\
    --max_eval_samples 500000 \\
    --seed 2026 \\
    --num_workers 8 \\
    --gradient_checkpointing \\
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
echo "✅ Ranking training started in background tmux session!"
echo ""
echo "📋 Commands:"
echo "  Attach:      tmux attach -t ${TMUX_SESSION}"
echo "  View Logs:   tail -f ${LOG_DIR}/train.log"
echo "  Kill:        tmux kill-session -t ${TMUX_SESSION}"
echo ""
