# MIT License - see LICENSE for details

"""
TensorBoard 监控模块。

提供训练过程的可视化监控。
"""

import logging
import os
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """
    TensorBoard 日志记录器。
    
    封装 TensorBoard 写入操作，支持：
    - 标量记录（Loss, AUC, LR）
    - 直方图记录（权重分布）
    - 文本记录（配置信息）
    """
    
    def __init__(
        self,
        log_dir: str = "runs",
        experiment_name: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        初始化 TensorBoard 日志记录器。
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称（默认使用时间戳）
            enabled: 是否启用
        """
        self.enabled = enabled
        self.writer = None
        
        if not enabled:
            return
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            if experiment_name is None:
                experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            self.log_path = os.path.join(log_dir, experiment_name)
            os.makedirs(self.log_path, exist_ok=True)
            
            self.writer = SummaryWriter(self.log_path)
            logger.info(f"TensorBoard 日志目录: {self.log_path}")
            
        except ImportError:
            logger.warning("TensorBoard 未安装，监控已禁用")
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量值。"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """记录多个标量值。"""
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """记录直方图。"""
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: int = 0):
        """记录文本。"""
        if self.writer:
            self.writer.add_text(tag, text, step)
    
    def log_hparams(self, hparams: Dict, metrics: Dict):
        """记录超参数和指标。"""
        if self.writer:
            self.writer.add_hparams(hparams, metrics)
    
    def log_config(self, config: Dict):
        """记录配置信息。"""
        if self.writer:
            import json
            config_str = json.dumps(config, indent=2, ensure_ascii=False)
            self.writer.add_text("config", f"```json\n{config_str}\n```", 0)
    
    def log_model_params(self, model, step: int):
        """记录模型参数分布。"""
        if self.writer:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f"params/{name}", param.data, step)
                    if param.grad is not None:
                        self.writer.add_histogram(f"grads/{name}", param.grad, step)
    
    def flush(self):
        """刷新缓冲区。"""
        if self.writer:
            self.writer.flush()
    
    def close(self):
        """关闭写入器。"""
        if self.writer:
            self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class MetricsTracker:
    """
    指标追踪器。
    
    追踪训练过程中的指标并支持 TensorBoard 记录。
    """
    
    def __init__(self, tb_logger: Optional[TensorBoardLogger] = None):
        self.tb_logger = tb_logger
        self.metrics_history = {}
        self.best_metrics = {}
    
    def update(self, metrics: Dict[str, float], step: int, prefix: str = "train"):
        """
        更新指标。
        
        Args:
            metrics: 指标字典
            step: 步数
            prefix: 前缀（train/val/test）
        """
        for name, value in metrics.items():
            full_name = f"{prefix}/{name}"
            
            # 记录到 TensorBoard
            if self.tb_logger:
                self.tb_logger.log_scalar(full_name, value, step)
            
            # 记录到历史
            if full_name not in self.metrics_history:
                self.metrics_history[full_name] = []
            self.metrics_history[full_name].append((step, value))
            
            # 更新最佳值
            if name in ['auc', 'ndcg', 'hit_rate', 'mrr']:
                if full_name not in self.best_metrics or value > self.best_metrics[full_name]:
                    self.best_metrics[full_name] = value
            elif name in ['loss']:
                if full_name not in self.best_metrics or value < self.best_metrics[full_name]:
                    self.best_metrics[full_name] = value
    
    def get_best(self, metric_name: str) -> Optional[float]:
        """获取最佳指标值。"""
        return self.best_metrics.get(metric_name)
    
    def get_history(self, metric_name: str):
        """获取指标历史。"""
        return self.metrics_history.get(metric_name, [])
