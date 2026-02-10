#!/bin/bash
# =============================================================================
# Tenrec 训练服务器部署脚本
# 
# 用法: bash tenrec_adapter/scripts/setup_server.sh
# =============================================================================

set -e

echo "=============================================="
echo "Tenrec 训练环境部署"
echo "=============================================="

# 检查当前目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "[1/5] 检查 Python 环境..."
python3 --version || { echo "Python3 未安装"; exit 1; }

echo "[2/5] 安装 Python 依赖..."
pip install --upgrade pip
pip install numpy pandas scikit-learn torch pyyaml tensorboard

# 可选依赖
echo "[2.1] 安装可选依赖..."
pip install pytest pytest-cov || echo "pytest 安装失败，跳过"

echo "[3/5] 检查 MLU 环境..."
if python3 -c "import torch_mlu" 2>/dev/null; then
    echo "✓ torch_mlu 已安装"
    python3 -c "import torch_mlu; print(f'MLU 设备数: {torch_mlu.device_count() if hasattr(torch_mlu, \"device_count\") else \"未知\"}')"
else
    echo "⚠ torch_mlu 未安装（将使用 CPU）"
fi

echo "[4/5] 检查数据目录..."
DATA_DIR="data/tenrec/Tenrec"
if [ -d "$DATA_DIR" ]; then
    echo "✓ 数据目录存在: $DATA_DIR"
    ls -la "$DATA_DIR"/*.csv 2>/dev/null | head -5 || echo "  (无 CSV 文件)"
else
    echo "⚠ 数据目录不存在: $DATA_DIR"
    echo "  请确保数据已放置在正确位置"
fi

echo "[5/5] 验证模块导入..."
cd "$PROJECT_ROOT"
python3 -c "
import sys
sys.path.insert(0, '.')

# 核心模块
from tenrec_adapter import TenrecDataLoader, TwoTowerModel, RankingModel, get_device
print('✓ 核心模块导入成功')

# 配置和监控
from tenrec_adapter import Config, load_config, TensorBoardLogger, CheckpointManager
print('✓ 配置和监控模块导入成功')

# 设备检测
from tenrec_adapter import print_device_info
print_device_info()
"

echo ""
echo "=============================================="
echo "部署完成！"
echo "=============================================="
echo ""
echo "启动训练命令:"
echo "  cd $PROJECT_ROOT"
echo "  bash tenrec_adapter/scripts/run_ctr1m_retrieval.sh   # 召回"
echo "  bash tenrec_adapter/scripts/run_ctr1m_ranking.sh     # 精排"
echo ""
echo "查看 TensorBoard:"
echo "  tensorboard --logdir=logs --host=0.0.0.0 --port=6006"
echo ""
