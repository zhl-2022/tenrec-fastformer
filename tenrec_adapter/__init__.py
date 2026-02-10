# Tenrec Adapter Module

"""
Tenrec 数据集到 Phoenix 推荐系统的适配模块。

提供:
- TenrecDataLoader: 加载 Tenrec CSV 数据（支持缓存）
- TenrecToPhoenixAdapter: 转换为 Phoenix RecsysBatch 格式
- 评测指标: AUC, NDCG@K, HitRate@K, MRR
- TwoTowerModel: Two-Tower 召回模型
- RankingModel: Transformer 精排模型
- Config: YAML 配置管理
- TensorBoardLogger: 训练监控
- CheckpointManager: 断点续训
"""

from .data_loader import TenrecDataLoader, TenrecInteraction
from .phoenix_adapter import TenrecToPhoenixAdapter
from .metrics import compute_auc, compute_ndcg_at_k, compute_hit_rate_at_k, compute_mrr
from .device_utils import (
    get_device,
    get_device_count,
    is_mlu_available,
    is_cuda_available,
    to_device,
    print_device_info,
)
from .models import TwoTowerModel, TenrecDataset
from .ranking_model import RankingModel, TwoStageModel
from .config_manager import Config, load_config, get_default_config
from .tensorboard_logger import TensorBoardLogger, MetricsTracker
from .checkpoint_manager import CheckpointManager

__all__ = [
    # 数据加载
    "TenrecDataLoader",
    "TenrecInteraction",
    "TenrecToPhoenixAdapter",
    # 评测指标
    "compute_auc",
    "compute_ndcg_at_k",
    "compute_hit_rate_at_k",
    "compute_mrr",
    # 设备工具
    "get_device",
    "get_device_count",
    "is_mlu_available",
    "is_cuda_available",
    "to_device",
    "print_device_info",
    # 模型
    "TwoTowerModel",
    "TenrecDataset",
    "RankingModel",
    "TwoStageModel",
    # 配置管理
    "Config",
    "load_config",
    "get_default_config",
    # 监控
    "TensorBoardLogger",
    "MetricsTracker",
    # 检查点
    "CheckpointManager",
]

