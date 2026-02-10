# Tenrec-Fastformer

A two-stage recommendation system (Retrieval → Ranking) built with **Fastformer** encoder, evaluated on the [Tenrec](https://github.com/yuangh-x/2022-NIPS-Tenrec) benchmark dataset.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Two-Stage Pipeline                  │
├──────────────────────┬──────────────────────────────┤
│   Stage 1: Retrieval │   Stage 2: Ranking           │
│   (Two-Tower Model)  │   (Fastformer Encoder)       │
│                      │                              │
│   User Tower ─┐      │   [User History]             │
│               ├→ ANN │         ↓                    │
│   Item Tower ─┘      │   Fastformer × 12 layers     │
│                      │         ↓                    │
│   Top-K Candidates   │   Click / Like Prediction    │
└──────────────────────┴──────────────────────────────┘
```

**Key Features:**
- **Fastformer Encoder** — additive attention mechanism, O(n) complexity vs O(n²) for standard Transformer
- **Gradient Checkpointing** — enables large batch training (8192+) on limited GPU memory
- **Vectorized Evaluation Cache** — pre-builds evaluation batches for fast validation
- **Multi-device** — supports NVIDIA CUDA, Cambricon MLU, and CPU fallback

## Results on Tenrec

| Metric | QB-video (val) | QB-video (test) |
|--------|---------------|-----------------|
| **AUC** (click) | **0.9335** | 0.8442 |
| like-AUC | — | 0.6990 |
| NDCG@20 | — | 0.1467 |
| Hit@1 | — | 0.1861 |
| MRR | — | 0.3117 |

*Configuration: `embed_dim=512`, `num_layers=12`, `num_negatives=127`, `batch_size=512`*

## Quick Start

### Install

```bash
pip install -e .
```

### Data Preparation

Download the [Tenrec dataset](https://drive.google.com/drive/folders/1GQWB6X2RQOV4wgX4kE6tldTO-K2bOWH) and place CSV files in:
```
data/tenrec/Tenrec/
├── QB-video.csv
├── QK-video.csv
└── ctr_data_1M.csv    # generated via official script
```

### Training

```bash
# Stage 1: Retrieval (Two-Tower)
bash tenrec_adapter/scripts/run_ctr1m_retrieval.sh

# Stage 2: Ranking (Fastformer)
bash tenrec_adapter/scripts/run_ctr1m_ranking.sh
```

Training runs inside a `tmux` session. Monitor with:
```bash
tmux attach -t <session_name>
tail -f logs/<experiment>/train.log
```

## Project Structure

```
tenrec_adapter/
├── fastformer.py           # Fastformer encoder (additive attention)
├── ranking_model.py        # Ranking model (Fastformer + scoring head)
├── models.py               # Two-Tower retrieval model
├── run_two_stage_train.py   # Main training script (retrieval + ranking)
├── data_loader.py          # Tenrec CSV loader with caching
├── eval_cache.py           # Vectorized evaluation cache
├── metrics.py              # AUC, NDCG@K, HitRate@K, MRR
├── device_utils.py         # Multi-device abstraction (CUDA/MLU/CPU)
├── checkpoint_manager.py   # Checkpoint save/load/cleanup
├── config_manager.py       # YAML config management
├── tensorboard_logger.py   # TensorBoard integration
├── phoenix_adapter.py      # Data format adapter
├── scripts/
│   ├── run_ctr1m_ranking.sh    # Ranking training script
│   ├── run_ctr1m_retrieval.sh  # Retrieval training script
│   └── setup_server.sh        # Server environment setup
└── tests/
    └── test_eval_cache_v4.py   # Evaluation cache tests
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--embed_dim` | 512 | Embedding dimension |
| `--hidden_dim` | 1024 | FFN hidden dimension |
| `--ranking_num_layers` | 12 | Fastformer encoder depth |
| `--num_negatives` | 127 | Negative samples per positive |
| `--batch_size` | 1024 | Training batch size |
| `--ranking_batch_size` | 8192 | Ranking-specific batch size |
| `--gradient_checkpointing` | flag | Enable memory-efficient training |
| `--encoder_type` | fastformer | Encoder architecture |

## References

- [Tenrec: A Large-scale Multipurpose Benchmark Dataset for Recommender Systems](https://arxiv.org/abs/2210.10629) (NeurIPS 2022)
- [Fastformer: Additive Attention Can Be All You Need](https://arxiv.org/abs/2108.09084) (arXiv 2021)

## License

[MIT](LICENSE)
