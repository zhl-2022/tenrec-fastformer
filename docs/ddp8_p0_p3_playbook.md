# DDP8 P0-P3 Playbook

## Short answer: do we need Rust pipeline?
No for training. Your current `run_ddp_train.py` training path does not require Rust.
Rust pipeline is optional in `P3` for inference/re-ranking/filtering acceleration.

## What was fixed in code (this patch)
- `tenrec_adapter/run_ddp_train.py`
  - Fixed `use_ddp` scope bug in `main`.
  - Removed duplicated `scheduler.step()`.
  - Added `--max_steps_per_epoch` for low-cost matrix experiments.
  - Added throughput log (`samples/s`) every `log_interval`.
- `tenrec_adapter/data_loader.py`
  - Cache key now includes `max_users` (e.g., `..._full...` vs `..._u200000...`).
  - Split cache key also includes user scope.
  - Cache metadata now stores and validates `max_users` to avoid cache pollution.
- Launch scripts now default to `torch.distributed.run`:
  - `tenrec_adapter/scripts/run_ctr1m_ddp8_mincost_matrix.sh`
  - `tenrec_adapter/scripts/run_ctr1m_ddp8_full_train.sh`
  - `tenrec_adapter/scripts/run_ctr1m_ddp_7gpu.sh`

## Recommended phase execution

### P0: smoke check (stability, 10-30 min)
```bash
cd /root/zhl/x-algorithm
PHASES=P0 bash tenrec_adapter/scripts/run_ctr1m_ddp8_p0_p3.sh
```

### P1: min-cost matrix (ablation screening)
Default is 4 experiments, each limited by `MAX_STEPS_PER_EPOCH=1200`.
```bash
cd /root/zhl/x-algorithm
PHASES=P1 MAX_USERS=200000 MAX_STEPS_PER_EPOCH=1200 \
bash tenrec_adapter/scripts/run_ctr1m_ddp8_p0_p3.sh
```

### P2: full training with best params
Set the best config from P1.
```bash
cd /root/zhl/x-algorithm
PHASES=P2 BEST_NEG=127 BEST_LAYERS=12 BEST_LS=0.10 \
bash tenrec_adapter/scripts/run_ctr1m_ddp8_p0_p3.sh
```

### P3: optional Rust pipeline build
Only if you want inference-side acceleration.
```bash
cd /root/zhl/x-algorithm
PHASES=P3 ENABLE_RUST_PIPELINE=1 \
bash tenrec_adapter/scripts/run_ctr1m_ddp8_p0_p3.sh
```

## Direct script usage (without orchestrator)
```bash
# matrix
bash tenrec_adapter/scripts/run_ctr1m_ddp8_mincost_matrix.sh

# full train
bash tenrec_adapter/scripts/run_ctr1m_ddp8_full_train.sh 127 12 0.10
```

## Notes
- If old cache exists, new cache naming by `max_users` will avoid full/subset collision.
- For 8-card DDP, start with `NUM_WORKERS=4` in matrix stage, then increase to 8 in full training if CPU headroom is enough.

## Repo Upload Scope
- See `docs/repo_upload_scope.md` for:
  - Files to push
  - Local-only files
  - Cleanup result
  - Commit command template
