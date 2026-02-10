# MIT License - see LICENSE for details

"""
检查点管理模块。

提供训练检查点的保存、加载和管理功能。
"""

import glob
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    检查点管理器。
    
    支持：
    - 检查点保存和加载
    - 断点续训
    - 最佳模型追踪
    - 旧检查点清理
    """
    
    def __init__(
        self,
        save_dir: str,
        keep_last_n: int = 3,
        keep_best: bool = True,
    ):
        """
        初始化检查点管理器。
        
        Args:
            save_dir: 保存目录
            keep_last_n: 保留最近的 N 个检查点
            keep_best: 是否保留最佳模型
        """
        self.save_dir = Path(save_dir)
        self.keep_last_n = keep_last_n
        self.keep_best = keep_best
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_metric = float('-inf')
        self.best_checkpoint = None
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Optional[Dict] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[Dict] = None,
        is_best: bool = False,
    ) -> str:
        """
        保存检查点。
        
        Args:
            model: 模型
            optimizer: 优化器
            epoch: 当前 epoch
            step: 当前步数
            metrics: 指标字典
            scheduler: 学习率调度器
            config: 配置
            is_best: 是否为最佳模型
            
        Returns:
            保存路径
        """
        # 处理 DDP 模型
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        checkpoint = {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
        }
        
        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        if config:
            checkpoint["config"] = config
        
        # 保存常规检查点
        filename = f"checkpoint_epoch{epoch}_step{step}.pt"
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        
        # 确保文件写入完成（防止文件系统缓存延迟）
        import time
        max_retries = 3
        for retry in range(max_retries):
            if save_path.exists():
                break
            time.sleep(0.2)  # 等待文件系统刷新
        else:
            logger.warning(f"检查点文件写入可能未完成: {save_path}")
        
        logger.info(f"检查点已保存: {save_path}")
        
        # 保存最新检查点链接（在清理之前）
        # 添加异常处理防止竞态条件
        latest_path = self.save_dir / "latest_checkpoint.pt"
        try:
            if latest_path.exists():
                latest_path.unlink()
            if save_path.exists():
                shutil.copy(save_path, latest_path)
        except FileNotFoundError as e:
            logger.warning(f"复制到 latest_checkpoint 失败（可能被清理）: {e}")
        except Exception as e:
            logger.warning(f"保存 latest_checkpoint 时出错: {e}")
        
        # 保存最佳模型
        if is_best and self.keep_best:
            best_path = self.save_dir / "best_model.pt"
            try:
                if save_path.exists():
                    shutil.copy(save_path, best_path)
                    self.best_checkpoint = best_path
                    logger.info(f"最佳模型已更新: {best_path}")
                else:
                    logger.warning(f"无法保存最佳模型，源文件不存在: {save_path}")
            except Exception as e:
                logger.warning(f"保存最佳模型时出错: {e}")
        
        # 清理旧检查点（在所有复制完成后）
        self._cleanup_old_checkpoints()
        
        return str(save_path)
    
    def load(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cpu",
    ) -> Tuple[Dict, int, int]:
        """
        加载检查点。
        
        Args:
            checkpoint_path: 检查点路径（默认加载最新）
            model: 模型（可选）
            optimizer: 优化器（可选）
            scheduler: 调度器（可选）
            device: 目标设备
            
        Returns:
            (checkpoint_dict, epoch, step)
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
        
        if checkpoint_path is None:
            logger.info("没有找到可用的检查点")
            return None, 0, 0
        
        logger.info(f"加载检查点: {checkpoint_path}")
        # weights_only=False 兼容 PyTorch 2.6+（默认值从 False 改为 True）
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 恢复模型
        if model is not None:
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint["model_state_dict"])
        
        # 恢复优化器
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # 恢复调度器
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        epoch = checkpoint.get("epoch", 0)
        step = checkpoint.get("step", 0)
        
        logger.info(f"恢复训练: epoch={epoch}, step={step}")
        
        return checkpoint, epoch, step
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新检查点路径。"""
        latest_path = self.save_dir / "latest_checkpoint.pt"
        if latest_path.exists():
            return str(latest_path)
        
        # 查找最新的检查点文件
        checkpoints = self._list_checkpoints()
        if checkpoints:
            return checkpoints[-1][0]
        
        return None
    
    def get_best_checkpoint(self) -> Optional[str]:
        """获取最佳模型路径。"""
        best_path = self.save_dir / "best_model.pt"
        if best_path.exists():
            return str(best_path)
        return None
    
    def _list_checkpoints(self) -> List[Tuple[str, int]]:
        """列出所有检查点（按步数排序）。"""
        pattern = str(self.save_dir / "checkpoint_epoch*_step*.pt")
        checkpoints = []
        
        for path in glob.glob(pattern):
            # 提取步数
            match = re.search(r"step(\d+)", path)
            if match:
                step = int(match.group(1))
                checkpoints.append((path, step))
        
        checkpoints.sort(key=lambda x: x[1])
        return checkpoints
    
    def _cleanup_old_checkpoints(self):
        """清理旧检查点，只保留最近的 N 个。"""
        if self.keep_last_n <= 0:
            return
        
        checkpoints = self._list_checkpoints()
        
        # 保留最后 N 个
        to_delete = checkpoints[:-self.keep_last_n] if len(checkpoints) > self.keep_last_n else []
        
        for path, _ in to_delete:
            try:
                os.remove(path)
                logger.debug(f"删除旧检查点: {path}")
            except Exception as e:
                logger.warning(f"删除检查点失败: {path}, {e}")
    
    def exists(self) -> bool:
        """检查是否存在可恢复的检查点。"""
        return self.get_latest_checkpoint() is not None
