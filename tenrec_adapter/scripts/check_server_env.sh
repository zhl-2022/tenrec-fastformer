#!/bin/bash
# =============================================================================
# 服务器环境诊断脚本 V2 — 使用 python3
# 用法: bash tenrec_adapter/scripts/check_server_env.sh 2>&1 | tee server_env.log
# =============================================================================

echo "=========================================="
echo "1. 硬件信息 (GPU/MLU)"
echo "=========================================="
if command -v cnmon &> /dev/null; then
    echo "[MLU detected]"
    cnmon
else
    echo "[No cnmon found]"
fi

echo ""
echo "=========================================="
echo "2. Python3 & PyTorch 环境"
echo "=========================================="
# 检测 python3 路径
PYTHON3=$(which python3 2>/dev/null)
if [ -z "$PYTHON3" ]; then
    # 尝试 conda 环境
    PYTHON3=$(which conda 2>/dev/null && conda run python --version 2>/dev/null)
    echo "❌ python3 not found in PATH"
    echo "Looking for python in common locations..."
    ls -la /usr/bin/python* 2>/dev/null
    ls -la /usr/local/bin/python* 2>/dev/null
    # Check conda
    conda env list 2>/dev/null
else
    echo "Python3: $($PYTHON3 --version 2>&1)"
    echo "Which Python3: $PYTHON3"
fi

# Use python3 for all checks
$PYTHON3 << 'PYEOF'
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
except ImportError:
    print("❌ PyTorch not installed!")
    sys.exit(1)

# MLU check
try:
    import torch_mlu
    print(f"torch_mlu: {torch_mlu.__version__}")
    print(f"MLU available: {torch.mlu.is_available()}")
    if torch.mlu.is_available():
        n = torch.mlu.device_count()
        print(f"MLU device count: {n}")
        for i in range(n):
            try:
                props = torch.mlu.get_device_properties(i)
                print(f"  MLU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")
            except Exception as e:
                print(f"  MLU {i}: error getting props: {e}")
except ImportError:
    print("torch_mlu: not installed")
except Exception as e:
    print(f"torch_mlu error: {e}")

# CUDA check
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")

# DDP backend
try:
    print(f"NCCL available: {torch.distributed.is_nccl_available()}")
except: pass
try:
    print(f"GLOO available: {torch.distributed.is_gloo_available()}")
except: pass
try:
    print(f"CNCL available: {torch.distributed.is_cncl_available()}")
except:
    print("CNCL: N/A")

# torchrun / torch.distributed.launch
import shutil
torchrun_path = shutil.which("torchrun")
print(f"torchrun: {torchrun_path}")

# Dependencies
for dep in ['numpy', 'pandas', 'tensorboard', 'sklearn', 'scipy']:
    try:
        mod = __import__(dep)
        ver = getattr(mod, '__version__', 'unknown')
        print(f"  ✅ {dep}: {ver}")
    except ImportError:
        print(f"  ❌ {dep}: NOT INSTALLED")

# Quick DDP smoke test
print("\n--- DDP Smoke Test ---")
try:
    import os
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    try:
        import torch_mlu
        backend = "cncl"
    except:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    torch.distributed.init_process_group(backend=backend, rank=0, world_size=1)
    print(f"  ✅ DDP init_process_group({backend}) OK!")
    torch.distributed.destroy_process_group()

    # Cleanup env
    for k in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"]:
        del os.environ[k]
except Exception as e:
    print(f"  ❌ DDP init failed: {e}")
PYEOF

echo ""
echo "=========================================="
echo "3. 数据集路径检查"
echo "=========================================="
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_DIR="${PROJECT_ROOT}/data/tenrec/Tenrec"
echo "Project Root: ${PROJECT_ROOT}"
echo "Data Dir: ${DATA_DIR}"

if [ -f "${DATA_DIR}/ctr_data_1M.csv" ]; then
    size=$(du -h "${DATA_DIR}/ctr_data_1M.csv" | cut -f1)
    lines=$(wc -l < "${DATA_DIR}/ctr_data_1M.csv")
    echo "  ✅ ctr_data_1M.csv: ${size}, ${lines} lines"
    echo "  Header:"
    head -1 "${DATA_DIR}/ctr_data_1M.csv"
else
    echo "  ❌ ctr_data_1M.csv NOT found"
fi

echo ""
echo "=========================================="
echo "4. Card 7 占用情况"
echo "=========================================="
echo "Card 7 进程:"
cnmon 2>/dev/null | grep -A2 "Processes"
ps aux | grep "python" | grep -v grep | head -10

echo ""
echo "=========================================="
echo "5. 端口 & 磁盘"
echo "=========================================="
if command -v ss &> /dev/null; then
    occupied=$(ss -tlnp | grep ":29500 " | head -1)
    if [ -z "$occupied" ]; then echo "  ✅ Port 29500 available"
    else echo "  ❌ Port 29500 occupied: ${occupied}"; fi
fi
df -h "${PROJECT_ROOT}"

echo ""
echo "✅ 诊断完成！"
