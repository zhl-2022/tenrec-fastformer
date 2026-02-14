# Repo Upload Scope (2026-02-14)

This file defines what should be pushed to remote, and what should stay local.

## A. Deleted as Obsolete/Temporary

The following files were removed locally:

- `_tmp_head_data_loader.py`
- `fix_server.py`
- `route_table.txt`
- `route_table_utf8.txt`
- `run_ddp_train.ps1`

Reason: temporary debug scripts, local network dumps, or outdated launch entry points.

## B. Keep Local Only (Not for Remote)

The following are now ignored by `.gitignore`:

- `scripts/fix_vpn_split.ps1`
- `docs/project_plan.md`
- `docs/连接服务器.md`
- `docs/Windows SSH免密登录远程服务器配置指南.md`
- `_tmp_*.py`
- `route_table*.txt`

Reason: machine-specific, network-specific, or personal notes.

## C. Recommended to Push

Core training/runtime changes:

- `tenrec_adapter/config_manager.py`
- `tenrec_adapter/data_loader.py`
- `tenrec_adapter/models.py`
- `tenrec_adapter/ranking_model.py`
- `tenrec_adapter/run_two_stage_train.py`
- `tenrec_adapter/run_ddp_train.py`

New scripts:

- `tenrec_adapter/scripts/check_server_env.sh`
- `tenrec_adapter/scripts/convert_tenrec_csv_to_parquet.py`
- `tenrec_adapter/scripts/debug_docker_streamlit_8501.sh`
- `tenrec_adapter/scripts/recreate_zhl_container_8mlu.sh`
- `tenrec_adapter/scripts/run_ctr1m_ddp8_full_train.sh`
- `tenrec_adapter/scripts/run_ctr1m_ddp8_mincost_matrix.sh`
- `tenrec_adapter/scripts/run_ctr1m_ddp8_p0_p3.sh`
- `tenrec_adapter/scripts/run_ctr1m_ddp_7gpu.sh`
- `tenrec_adapter/scripts/start_streamlit_in_docker.sh`
- `tenrec_adapter/scripts/TRAINING_GUIDE.md`

Inference/Demo/Pipeline additions:

- `tenrec_adapter/app.py`
- `tenrec_adapter/inference_demo.py`
- `tenrec_adapter/recommendation_pipeline.py`
- `tenrec_adapter/rust_components.py`
- `rust_pipeline/`

Project docs/config:

- `docs/ddp8_p0_p3_playbook.md`
- `docs/repo_upload_scope.md`
- `.gitignore`
- `.gitattributes`

## D. Suggested Commit Steps

```bash
git status

git add .gitattributes .gitignore
git add docs/ddp8_p0_p3_playbook.md docs/repo_upload_scope.md
git add rust_pipeline
git add tenrec_adapter/app.py tenrec_adapter/inference_demo.py
git add tenrec_adapter/recommendation_pipeline.py tenrec_adapter/rust_components.py
git add tenrec_adapter/run_ddp_train.py
git add tenrec_adapter/config_manager.py tenrec_adapter/data_loader.py
git add tenrec_adapter/models.py tenrec_adapter/ranking_model.py
git add tenrec_adapter/run_two_stage_train.py
git add tenrec_adapter/scripts/TRAINING_GUIDE.md
git add tenrec_adapter/scripts/check_server_env.sh
git add tenrec_adapter/scripts/convert_tenrec_csv_to_parquet.py
git add tenrec_adapter/scripts/debug_docker_streamlit_8501.sh
git add tenrec_adapter/scripts/recreate_zhl_container_8mlu.sh
git add tenrec_adapter/scripts/run_ctr1m_ddp8_full_train.sh
git add tenrec_adapter/scripts/run_ctr1m_ddp8_mincost_matrix.sh
git add tenrec_adapter/scripts/run_ctr1m_ddp8_p0_p3.sh
git add tenrec_adapter/scripts/run_ctr1m_ddp_7gpu.sh
git add tenrec_adapter/scripts/start_streamlit_in_docker.sh

# Optional but recommended once after adding .gitattributes
git add --renormalize tenrec_adapter/scripts/*.sh

git commit -m "ddp8 training pipeline, data loader optimization, streamlit docker ops, and repo cleanup"
```

## E. Line Ending Safety

`.gitattributes` is added to keep `*.sh` as LF to avoid `/bin/bash^M` errors after upload to Linux.
