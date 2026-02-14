# Training Guide

## 1) Baseline (Single Process)

```bash
# Setup env
bash tenrec_adapter/scripts/setup_server.sh

# Retrieval
bash tenrec_adapter/scripts/run_ctr1m_retrieval.sh

# Ranking
bash tenrec_adapter/scripts/run_ctr1m_ranking.sh
```

## 2) DDP8 Workflow (Recommended on 8 MLU)

```bash
# P0: smoke
PHASES=P0 bash tenrec_adapter/scripts/run_ctr1m_ddp8_p0_p3.sh

# P1: min-cost matrix
PHASES=P1 MAX_USERS=200000 MAX_STEPS_PER_EPOCH=1200 \
bash tenrec_adapter/scripts/run_ctr1m_ddp8_p0_p3.sh

# P2: full train with best params from P1
PHASES=P2 BEST_NEG=127 BEST_LAYERS=12 BEST_LS=0.10 \
bash tenrec_adapter/scripts/run_ctr1m_ddp8_p0_p3.sh

# P3: optional Rust pipeline build
PHASES=P3 ENABLE_RUST_PIPELINE=1 \
bash tenrec_adapter/scripts/run_ctr1m_ddp8_p0_p3.sh
```

Direct execution:

```bash
bash tenrec_adapter/scripts/run_ctr1m_ddp8_mincost_matrix.sh
bash tenrec_adapter/scripts/run_ctr1m_ddp8_full_train.sh 127 12 0.10
```

## 3) Docker + Streamlit Ops

```bash
# Recreate container (8 cards + ports)
bash tenrec_adapter/scripts/recreate_zhl_container_8mlu.sh

# Start Streamlit in container
bash tenrec_adapter/scripts/start_streamlit_in_docker.sh

# Debug 8501/host mapping
bash tenrec_adapter/scripts/debug_docker_streamlit_8501.sh
```

## 4) Data Loading Optimization

```bash
# Convert large CSV to Parquet once
python3 tenrec_adapter/scripts/convert_tenrec_csv_to_parquet.py \
  --data_dir data/tenrec/Tenrec \
  --scenario ctr_data_1M \
  --chunk_size 1000000 \
  --compression zstd
```

## 5) Monitoring

```bash
tail -f logs/<run>/train.log
cnmon
```

## 6) Notes

- DDP launcher uses `python -m torch.distributed.run` by default.
- If `.sh` is copied from Windows and fails on Linux with shebang errors:
  `sed -i 's/\r$//' tenrec_adapter/scripts/*.sh`.
