# Training Guide

## Quick Start

```bash
# 1. Setup environment
bash tenrec_adapter/scripts/setup_server.sh

# 2. Train retrieval model (Two-Tower)
bash tenrec_adapter/scripts/run_ctr1m_retrieval.sh

# 3. Train ranking model (Fastformer)
bash tenrec_adapter/scripts/run_ctr1m_ranking.sh
```

## Monitor Training

```bash
# Attach to tmux session
tmux attach -t <session_name>

# Detach without stopping: Ctrl+B then D

# View logs
tail -f logs/<experiment>/train.log

# TensorBoard
tensorboard --logdir=logs --host=0.0.0.0 --port=6006

# List all sessions
tmux ls

# Kill a session
tmux kill-session -t <session_name>
```

## Custom Training

```bash
python tenrec_adapter/run_two_stage_train.py \
    --stage ranking \
    --scenario ctr_data_1M \
    --data_dir data/tenrec/Tenrec \
    --encoder_type fastformer \
    --ranking_num_layers 12 \
    --num_negatives 127 \
    --batch_size 1024 \
    --ranking_batch_size 8192 \
    --epochs 3 \
    --lr 0.0008 \
    --gradient_checkpointing
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--stage` | `both` | `retrieval`, `ranking`, or `both` |
| `--scenario` | `QB-video` | Dataset: `QB-video`, `QK-video`, `ctr_data_1M` |
| `--encoder_type` | `fastformer` | `fastformer` or `transformer` |
| `--ranking_num_layers` | 12 | Encoder depth |
| `--num_negatives` | 127 | Negatives per positive |
| `--batch_size` | 64 | Training batch size |
| `--ranking_batch_size` | — | Override batch size for ranking |
| `--gradient_checkpointing` | off | Trade compute for memory |
| `--grad_accumulation` | 1 | Gradient accumulation steps |
| `--eval_interval` | 5000 | Steps between evaluations |
| `--patience` | 5 | Early stopping patience |

## Evaluation Metrics

| Metric | Description | Leaderboard |
|--------|-------------|-------------|
| **AUC** | Click prediction quality | CTR |
| **like-AUC** | Like prediction quality | Multi-Task |
| **NDCG@20** | Top-20 ranking quality | Top-N |
| Hit@1 | Precision at rank 1 | — |
| MRR | Mean reciprocal rank | — |
