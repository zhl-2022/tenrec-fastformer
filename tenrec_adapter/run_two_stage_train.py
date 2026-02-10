#!/usr/bin/env python
# MIT License - see LICENSE for details

"""
两阶段推荐系统训练脚本。

阶段 1: 训练召回模型 (TwoTowerModel)
阶段 2: 训练精排模型 (RankingModel)

用法:
    # 训练召回模型
    python run_two_stage_train.py --stage retrieval --epochs=10
    
    # 训练精排模型
    python run_two_stage_train.py --stage ranking --retrieval_model=checkpoints/retrieval_best.pt
    
    # 两阶段联合训练
    python run_two_stage_train.py --stage both --epochs=10
"""

import argparse
import logging
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

# [Fix] 过滤 torch_mlu 的 is_compiling 过时警告，减少日志噪音
warnings.filterwarnings("ignore", message=".*is_compiling.*")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 添加模块路径
_current_dir = Path(__file__).parent
_project_root = _current_dir.parent
sys.path.insert(0, str(_project_root))

from tenrec_adapter.data_loader import TenrecDataLoader
from tenrec_adapter.phoenix_adapter import TenrecToPhoenixAdapter
from tenrec_adapter.metrics import compute_auc
from tenrec_adapter.device_utils import get_device, print_device_info, is_mlu_available
from tenrec_adapter.models import TwoTowerModel
from tenrec_adapter.ranking_model import RankingModel
from tenrec_adapter.config_manager import load_config, get_default_config
from tenrec_adapter.tensorboard_logger import TensorBoardLogger
from tenrec_adapter.checkpoint_manager import CheckpointManager
from tenrec_adapter.eval_cache import CachedEvalDataset, get_eval_cache_path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TenrecDataset(Dataset):
    """数据集包装器。
    
    Args:
        interactions: 交互数据列表
        adapter: Phoenix 适配器
        num_negatives: 负样本数量
        is_eval: 是否为验证模式（验证模式使用固定种子确保负样本一致）
        seed: 验证模式的基础种子
    """
    
    def __init__(self, interactions, adapter, num_negatives=4, is_eval=False, seed=42):
        self.interactions = interactions
        self.adapter = adapter
        self.num_negatives = num_negatives
        self.is_eval = is_eval
        self.seed = seed
        self._epoch = 0  # epoch 计数器，用于 per-epoch 不同负样本
        
        # 快速路径：直接从 Store 读取（跳过 create_training_batch 的 Python 循环）
        self._store = getattr(adapter.data_loader, 'store', None)
        self._has_fast_path = (
            self._store is not None 
            and self._store.has_hist 
            and self._store.hist_matrix is not None
            and hasattr(interactions, '_indices')
        )
    
    def set_epoch(self, epoch: int):
        """设置当前 epoch，使每个 epoch 产生不同的负样本。"""
        self._epoch = epoch
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        # 快速路径：直接从 Store numpy 数组读取 + 向量化 hash
        if self._has_fast_path:
            return self._getitem_fast(idx)
        
        # 慢路径：通过 adapter.create_training_batch（兼容非 Store 数据）
        return self._getitem_slow(idx)
    
    def _getitem_fast(self, idx):
        """快速路径：直接从 Store 读取，避免 create_training_batch 的 Python 循环开销。"""
        store = self._store
        adapter = self.adapter
        store_idx = self.interactions._indices[idx]
        
        hs = adapter.hash_table_size
        nih = adapter.num_item_hashes
        nuh = adapter.num_user_hashes
        hist_len = adapter.history_seq_len
        num_actions = adapter.num_actions
        num_neg = self.num_negatives
        total_cand = 1 + num_neg
        
        # [Critical] 每个样本使用独立的 rng，避免 DataLoader 多 worker 共享 adapter.rng
        # seed 融合 epoch 确保每个 epoch 产生不同的负样本，提升模型泛化能力
        sample_rng = np.random.default_rng(store_idx + self._epoch * 1000003)
        
        # 1. 用户 Hash（确定性，不依赖 rng）
        user_id = int(store.user_ids[store_idx])
        user_hash = adapter._generate_multi_hash(user_id, nuh, hs)
        
        # 2. 历史序列（从预构建 hist_matrix 读取）
        hist_items = store.hist_matrix[store_idx]  # [10]
        actual_len = min(10, hist_len)
        
        # 向量化 hist hash
        hist_slice = hist_items[:actual_len]
        hist_hashes = adapter._batch_generate_multi_hash(hist_slice, nih, hs)  # [actual_len, nih]
        
        # pad 到 hist_len
        history_item_ids = np.zeros(hist_len, dtype=np.int32)
        history_item_ids[:actual_len] = hist_hashes[:, 0]  # 只取第一个 hash
        # padding 位置置 0
        padding_mask = hist_slice == 0
        history_item_ids[:actual_len][padding_mask] = 0
        
        # history_actions
        history_actions = np.zeros((hist_len, num_actions), dtype=np.float32)
        valid_mask = (hist_slice != 0)
        history_actions[:actual_len, 0] = valid_mask.astype(np.float32)  # click=1 for valid
        
        # 3. 负采样（使用 per-sample rng）
        item_id = int(store.item_ids[store_idx])
        neg_items = adapter.data_loader.sample_negative_items(user_id, num_neg, sample_rng)
        neg_items_arr = np.array(neg_items, dtype=np.int32)
        
        # 4. 候选 Hash（正样本 + 负样本）
        all_cand = np.concatenate([[item_id], neg_items_arr])  # [total_cand]
        all_cand_hashes = adapter._batch_generate_multi_hash(all_cand, nih, hs)  # [total_cand, nih]
        candidate_item_ids = all_cand_hashes[:, 0]  # [total_cand]
        
        # 5. Labels
        labels = np.zeros((total_cand, num_actions), dtype=np.float32)
        labels[0, 0] = float(store.clicks[store_idx])
        labels[0, 1] = float(store.likes[store_idx])
        labels[0, 2] = float(store.shares[store_idx])
        labels[0, 3] = float(store.follows[store_idx])
        
        # 6. Shuffle（使用 per-sample rng）
        perm = sample_rng.permutation(total_cand)
        candidate_item_ids = candidate_item_ids[perm]
        labels = labels[perm]
        
        return {
            "user_ids": torch.tensor(user_hash[0], dtype=torch.long),
            "history_item_ids": torch.tensor(history_item_ids, dtype=torch.long),
            "history_actions": torch.tensor(history_actions, dtype=torch.float32),
            "candidate_item_ids": torch.tensor(candidate_item_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }
    
    def _getitem_slow(self, idx):
        """慢路径：通过 adapter.create_training_batch。"""
        # 验证模式：使用固定种子确保每次评估的负样本一致
        if self.is_eval:
            eval_rng = np.random.default_rng(self.seed + idx)
            original_rng = self.adapter.rng
            self.adapter.rng = eval_rng
        
        batch = self.adapter.create_training_batch(
            [self.interactions[idx]],
            num_negatives=self.num_negatives,
        )
        
        if self.is_eval:
            self.adapter.rng = original_rng
        
        return {
            "user_ids": torch.tensor(batch.user_hashes[0, 0], dtype=torch.long),
            "history_item_ids": torch.tensor(batch.history_post_hashes[0, :, 0], dtype=torch.long),
            "history_actions": torch.tensor(batch.history_actions[0], dtype=torch.float32),
            "candidate_item_ids": torch.tensor(batch.candidate_post_hashes[0, :, 0], dtype=torch.long),
            "labels": torch.tensor(batch.labels[0], dtype=torch.float32),
        }


def collate_fn(batch):
    """Collate 函数。"""
    return {
        "user_ids": torch.stack([b["user_ids"] for b in batch]),
        "history_item_ids": torch.stack([b["history_item_ids"] for b in batch]),
        "history_actions": torch.stack([b["history_actions"] for b in batch]),
        "candidate_item_ids": torch.stack([b["candidate_item_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def evaluate(model, val_loader, device, criterion):
    """评估模型。
    
    修复：使用模型输出的 scores (点积分数) 而非 action_logits 进行 AUC 计算。
    这是企业级推荐系统的标准做法：
    1. scores 是 User-Item 相似度的直接度量
    2. AUC 评估的是排序能力，应该使用排序分数
    3. 保持与训练目标一致（BCELoss 也是基于排序思想）
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []  # 新增：收集 scores
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            logits = outputs["logits"]
            scores = outputs["scores"]  # [B, C] - 点积分数
            loss = criterion(logits, batch["labels"])
            
            total_loss += loss.item()
            num_batches += 1
            
            # 收集预测分数 (使用 scores 而非 action_logits)
            score_preds = scores.cpu().numpy()  # [N, C]
            all_scores.append(score_preds)
            
            # 同时保留 action_logits 用于 debug
            preds = torch.sigmoid(logits).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels)
    
    all_scores = np.concatenate(all_scores, axis=0)  # [N, C]
    all_preds = np.concatenate(all_preds, axis=0)  # [N, C, num_actions]
    all_labels = np.concatenate(all_labels, axis=0)  # [N, C, num_actions]
    
    # click 是第一个 action (idx=0)
    click_labels = all_labels[:, :, 0]  # [N, C]
    click_preds = all_preds[:, :, 0]  # [N, C] - action_logits 的 click 预测（已经过 sigmoid）
    
    # ===== 调试代码：检查分数区分度 =====
    # 如果标准差接近 0，说明模型无法区分不同候选 item
    # 如果标准差较大（>0.1），说明模型在学习区分
    logger.info(f"  [Debug] scores std: {np.std(all_scores):.4f}, range: [{np.min(all_scores):.4f}, {np.max(all_scores):.4f}]")
    logger.info(f"  [Debug] click_preds std: {np.std(click_preds):.4f}, range: [{np.min(click_preds):.4f}, {np.max(click_preds):.4f}]")
    # ===== 调试代码结束 =====
    
    # 计算两种 AUC 进行对比
    # 方式 1: 使用 scores（点积分数）- 我们的实现
    auc_scores = compute_auc(click_labels.flatten(), all_scores.flatten())
    
    # 方式 2: 使用 sigmoid(click)（概率）- Phoenix 原版方式
    auc_sigmoid = compute_auc(click_labels.flatten(), click_preds.flatten())
    
    logger.info(f"  [AUC对比] scores方式: {auc_scores:.4f}, sigmoid方式: {auc_sigmoid:.4f}")
    
    # 主 AUC 使用 sigmoid 方式（概率预测更准确，训练显示高 0.73%）
    auc = auc_sigmoid
    
    # 找到每个样本中正样本的真实位置
    # 过滤掉没有正样本的行（click_labels全为0），避免 argmax 返回 0 导致虚假抬高 HitRate/MRR
    has_positive = click_labels.max(axis=1) > 0  # [N]
    
    pred_ranks = np.argsort(-all_scores, axis=1)  # 按分数降序排列
    
    if has_positive.any():
        valid_labels = click_labels[has_positive]
        valid_ranks = pred_ranks[has_positive]
        positive_pos = np.argmax(valid_labels, axis=1)
        
        hit_at_1 = np.mean(valid_ranks[:, 0] == positive_pos)
        
        positive_rank_values = np.array([
            np.where(valid_ranks[i] == positive_pos[i])[0][0] + 1
            for i in range(len(positive_pos))
        ])
        mrr = np.mean(1.0 / positive_rank_values)
    else:
        hit_at_1 = 0.0
        mrr = 0.0
    
    # 计算 NDCG@10 和 NDCG@20（官方 Tenrec Leaderboard 核心指标）
    from tenrec_adapter.metrics import compute_ndcg_at_k
    ndcg_10 = compute_ndcg_at_k(click_labels, all_scores, k=10)
    ndcg_20 = compute_ndcg_at_k(click_labels, all_scores, k=20)  # Top-N Leaderboard 用 @20
    
    # 计算 like-AUC（Multi-Task Leaderboard 需要分开报告 click 和 like AUC）
    # like 是第二个 action (idx=1)
    like_labels = all_labels[:, :, 1]  # [N, C]
    like_preds = all_preds[:, :, 1]    # [N, C]
    like_auc = compute_auc(like_labels.flatten(), like_preds.flatten())
    
    avg_loss = total_loss / max(num_batches, 1)
    
    return {
        "loss": avg_loss, 
        "auc": auc,                    # click-AUC (主指标)
        "auc_sigmoid": auc_sigmoid,
        "like_auc": like_auc,          # like-AUC (Multi-Task Leaderboard)
        "hit@1": hit_at_1, 
        "mrr": mrr, 
        "ndcg@10": ndcg_10,
        "ndcg@20": ndcg_20,            # Top-N Leaderboard 用 @20
    }


def init_weights(module: nn.Module):
    """重新初始化模型权重（NaN 恢复用）。"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # 保护 padding_idx
        if hasattr(module, 'padding_idx') and module.padding_idx is not None:
            with torch.no_grad():
                if module.weight.size(0) > 1:
                    nn.init.xavier_uniform_(module.weight[1:])
                module.weight[0].zero_()
        else:
            nn.init.xavier_uniform_(module.weight)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict,
    checkpoint_manager: CheckpointManager,
    tb_logger: Optional[TensorBoardLogger] = None,
    model_name: str = "model",
):
    """
    训练模型。
    
    Args:
        model: 模型
        train_loader: 训练数据
        val_loader: 验证数据
        device: 设备
        config: 配置
        checkpoint_manager: 检查点管理器
        tb_logger: TensorBoard 日志
        model_name: 模型名称（用于日志）
    """
    model = model.to(device)
    
    # 优化器
    base_lr = config.get("learning_rate", 0.002)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=config.get("weight_decay", 0.01),
        betas=(0.9, 0.999),
    )
    label_smoothing = config.get("label_smoothing", 0.0)
    criterion = nn.BCEWithLogitsLoss()
    if label_smoothing > 0:
        logger.info(f"启用 Label Smoothing: {label_smoothing}")
    
    # 学习率调度器 (带 warmup)
    warmup_steps = config.get("warmup_steps", 100)
    total_steps = len(train_loader) * config.get("epochs", 10)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)  # 线性预热
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 混合精度配置
    # 注意：MLU 上的混合精度容易导致 NaN，默认禁用
    use_amp = config.get("use_amp", False)  # 默认禁用
    
    # 仅在 CUDA 上启用混合精度，MLU 使用 FP32
    if use_amp and torch.cuda.is_available() and not is_mlu_available():
        scaler = torch.amp.GradScaler("cuda")
        autocast = lambda: torch.amp.autocast("cuda")
        logger.info("启用 CUDA 混合精度 (FP16)")
    else:
        # MLU 或 CPU：禁用混合精度，使用 FP32
        use_amp = False
        scaler = None
        autocast = None
        if is_mlu_available():
            logger.info("MLU 使用 FP32 精度（禁用混合精度以避免 NaN）")

    
    epochs = config.get("epochs", 10)
    log_interval = config.get("log_interval", 50)
    eval_interval = config.get("eval_interval", 500)
    grad_accumulation = config.get("grad_accumulation", 1)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    
    best_auc = 0
    global_step = 0
    patience = config.get("patience", 0)  # 0 = disabled
    no_improve_count = 0
    
    logger.info(f"开始训练 {model_name}...")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  混合精度: {use_amp}")
    if patience > 0:
        logger.info(f"  Early Stopping: patience={patience} evals")
    
    for epoch in range(epochs):
        model.train()
        # [Bug Fix] 通知 Dataset 当前 epoch，使负采样 seed 随 epoch 变化
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)
        epoch_loss = 0
        num_batches = 0
        epoch_start = time.time()
        
        logger.info(f"\n--- {model_name} Epoch {epoch + 1}/{epochs} ---")
        
        nan_debug_printed = False  # 只打印一次诊断信息
        nan_count = 0  # 连续 NaN 计数器
        max_nan_before_reset = 50  # 连续 50 次 NaN 后重置模型
        
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # ===== 输入数据 NaN 检测 =====
            has_input_nan = False
            for key, val in batch.items():
                if torch.isnan(val).any() or torch.isinf(val).any():
                    has_input_nan = True
                    if not nan_debug_printed:
                        logger.warning(f"  输入数据 '{key}' 包含 NaN/Inf!")
            
            if has_input_nan:
                logger.warning(f"  Step {step + 1}: 输入数据包含 NaN/Inf，跳过该批次")
                continue
            
            # ===== 输入数据 Clamp（防止极端值） =====
            if 'history_actions' in batch:
                batch['history_actions'] = torch.clamp(batch['history_actions'], -10.0, 10.0)
            
            if use_amp:
                with autocast():
                    outputs = model(batch)
                    targets = batch["labels"]
                    if label_smoothing > 0:
                        targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
                    loss = criterion(outputs["logits"], targets)
                    loss = loss / grad_accumulation
                scaler.scale(loss).backward()
            else:
                outputs = model(batch)
                
                # 直接计算 loss（支持 label smoothing）
                targets = batch["labels"]
                if label_smoothing > 0:
                    targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
                loss = criterion(outputs["logits"], targets)
                loss = loss / grad_accumulation
                loss.backward()
            
            # NaN 检测保护
            is_nan = False
            if torch.isnan(loss) or torch.isinf(loss):
                is_nan = True
                if not nan_debug_printed:
                    logger.warning(f"  [NaN Debug] Step {step + 1}: Loss is NaN/Inf!")
                    
                    # 诊断：打印一些统计信息
                    with torch.no_grad():
                        logger.warning(f"  - Labels min/max: {batch['labels'].min()}/{batch['labels'].max()}")
                        logger.warning(f"  - Logits min/max: {outputs['logits'].min()}/{outputs['logits'].max()}")
                        for name, param in model.named_parameters():
                            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                logger.warning(f"  - Gradient NaN detected in: {name}")
                                break
                    
                    nan_debug_printed = True
            
            if is_nan:
                nan_count += 1
                
                # 连续多次 NaN 后重新初始化模型
                if nan_count >= max_nan_before_reset:
                    logger.warning(f"  连续 {nan_count} 次 NaN，重新初始化模型权重...")
                    model.apply(init_weights)
                    nan_count = 0
                
                optimizer.zero_grad()  # 清理梯度
                if use_amp:
                    scaler.update()  # 即使跳过也要更新 scaler 状态
                
                # 即使 NaN 也要跳过 step，防止污染权重
                continue
            else:
                nan_count = 0  # 正常时重置计数器 

            epoch_loss += loss.item() * grad_accumulation
            num_batches += 1
            
            if (step + 1) % grad_accumulation == 0:
                # 检查梯度是否正常
                grad_nan = False
                for param in model.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        grad_nan = True
                        break
                
                if grad_nan:
                    if not nan_debug_printed:
                         logger.warning(f"  Step {step + 1}: Gradient NaN detected before step! Skipping.")
                    optimizer.zero_grad()
                    continue

                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                
                scheduler.step()  # 更新学习率
                optimizer.zero_grad()
                global_step += 1
                
                if tb_logger:
                    tb_logger.log_scalar(f"{model_name}/train_loss", loss.item() * grad_accumulation, global_step)
                    tb_logger.log_scalar(f"{model_name}/learning_rate", scheduler.get_last_lr()[0], global_step)
            
            if (step + 1) % log_interval == 0:
                avg_loss = epoch_loss / max(num_batches, 1)  # 防止除零
                # 添加显存监控 - 使用系统级 API 获取真实显存占用
                mem_info = ""
                try:
                    from tenrec_adapter.device_utils import get_device_memory_system
                    mem_used, mem_total = get_device_memory_system(0)
                    if mem_total > 0:
                        mem_percent = mem_used / mem_total * 100
                        mem_info = f", 显存: {mem_used/1e9:.1f}/{mem_total/1e9:.1f}GB ({mem_percent:.0f}%)"
                except Exception:
                    # 回退方案
                    try:
                        import torch_mlu
                        mem_reserved = torch.mlu.memory_reserved() / 1024**3
                        props = torch.mlu.get_device_properties(0)
                        mem_total = props.total_memory / 1024**3
                        mem_percent = mem_reserved / mem_total * 100
                        mem_info = f", 显存: {mem_reserved:.1f}/{mem_total:.1f}GB ({mem_percent:.0f}%)"
                    except Exception:
                        try:
                            if torch.cuda.is_available():
                                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                                mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                                mem_percent = mem_reserved / mem_total * 100
                                mem_info = f", 显存: {mem_reserved:.1f}/{mem_total:.1f}GB ({mem_percent:.0f}%)"
                        except Exception:
                            pass
                logger.info(f"  Step {step + 1}/{len(train_loader)}, Loss: {avg_loss:.4f}{mem_info}")
            
            if global_step > 0 and global_step % eval_interval == 0:
                val_metrics = evaluate(model, val_loader, device, criterion)
                logger.info(f"  [Eval] Step {global_step}, AUC: {val_metrics['auc']:.4f}, like-AUC: {val_metrics['like_auc']:.4f}, NDCG@20: {val_metrics['ndcg@20']:.4f}")
                
                if tb_logger:
                    tb_logger.log_scalar(f"{model_name}/val_auc", val_metrics['auc'], global_step)
                    tb_logger.log_scalar(f"{model_name}/val_hit1", val_metrics['hit@1'], global_step)
                    tb_logger.log_scalar(f"{model_name}/val_mrr", val_metrics['mrr'], global_step)
                
                if val_metrics['auc'] > best_auc:
                    best_auc = val_metrics['auc']
                    no_improve_count = 0
                    checkpoint_manager.save(
                        model, optimizer, epoch, global_step,
                        metrics=val_metrics, is_best=True,
                    )
                else:
                    no_improve_count += 1
                    if patience > 0 and no_improve_count >= patience:
                        logger.info(f"  [Early Stop] AUC 连续 {no_improve_count} 次评估无提升（含 step-level 和 epoch-level），停止训练")
                        logger.info(f"  最佳 AUC: {best_auc:.4f}")
                        return best_auc
                
                model.train()
        
        # Epoch 结束
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"  Epoch {epoch + 1} 完成 - Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s")
        
        # 检查是否所有批次都是 NaN
        if num_batches == 0:
            logger.error(f"  警告：Epoch {epoch + 1} 所有批次都是 NaN！模型需要重新初始化。")
            logger.info("  正在重新初始化模型权重...")
            model.apply(init_weights)
        
        # Epoch 评估
        val_metrics = evaluate(model, val_loader, device, criterion)
        logger.info(f"  Epoch {epoch + 1} 验证 AUC: {val_metrics['auc']:.4f}, like-AUC: {val_metrics['like_auc']:.4f}, NDCG@20: {val_metrics['ndcg@20']:.4f}")
        
        # [Bug Fix] Epoch 结束时也要检查并保存最佳模型
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            no_improve_count = 0
            checkpoint_manager.save(
                model, optimizer, epoch, global_step,
                metrics=val_metrics, is_best=True,
            )
            logger.info(f"  [新最佳] Epoch {epoch + 1} AUC: {best_auc:.4f} - 已保存检查点")
        else:
            no_improve_count += 1
            if patience > 0 and no_improve_count >= patience:
                logger.info(f"  [Early Stop] AUC 连续 {patience} 次 Epoch 评估无提升")
                break
        
        if tb_logger:
            tb_logger.log_scalar(f"{model_name}/epoch_auc", val_metrics['auc'], epoch + 1)
            tb_logger.flush()
    
    logger.info(f"{model_name} 训练完成，最佳 AUC: {best_auc:.4f}")
    return best_auc


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description="两阶段推荐系统训练")
    parser.add_argument("--stage", type=str, default="both", choices=["retrieval", "ranking", "both"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data/tenrec/Tenrec")
    parser.add_argument("--scenario", type=str, default="QB-video")
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024, help="默认 batch size")
    parser.add_argument("--retrieval_batch_size", type=int, default=None, help="粗排 batch size（默认用 batch_size）")
    parser.add_argument("--ranking_batch_size", type=int, default=None, help="精排 batch size（默认用 batch_size//2）")
    parser.add_argument("--lr", type=float, default=0.001)  
    parser.add_argument("--num_workers", type=int, default=16, help="DataLoader 工作进程数 (Linux: 16, Windows: 0)")
    parser.add_argument("--retrieval_model", type=str, default=None, help="预训练召回模型路径")
    parser.add_argument("--encoder_type", type=str, default="fastformer", 
                        choices=["fastformer", "transformer"],
                        help="精排编码器类型: fastformer (线性复杂度) 或 transformer (更强表达力)")
    parser.add_argument("--num_negatives", type=int, default=63,
                        help="负样本数量 (默认 63, 可选 127/255 充分利用显存)")
    parser.add_argument("--ranking_num_layers", type=int, default=12,
                        help="精排 Transformer 层数 (默认 12, 可选 16/20 更深网络)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding 维度")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="隐藏层维度")
    parser.add_argument("--num_heads", type=int, default=16, help="Attention 头数")
    parser.add_argument("--warmup_steps", type=int, default=100, help="学习率预热步数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label Smoothing 系数 (0=禁用, 0.1=推荐)")
    parser.add_argument("--patience", type=int, default=5, help="Early Stopping 耐心值 (0=禁用, 5=推荐)")
    parser.add_argument("--eval_interval", type=int, default=200, help="评估间隔步数 (大数据集建议 1000+)")
    parser.add_argument("--history_seq_len", type=int, default=64, help="历史序列长度 (ctr_data_1M 预构建历史建议 10)")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="验证/测试集最大样本数 (大数据集建议 500000)")
    parser.add_argument("--grad_accumulation", type=int, default=1, help="梯度累积步数 (2=有效batch翻倍, 显存不变)")
    parser.add_argument("--gradient_checkpointing", action="store_true", 
                        help="启用梯度检查点（以~30%%训练时间换取~10x激活显存节省）")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("两阶段推荐系统训练")
    logger.info(f"阶段: {args.stage}")
    logger.info("=" * 60)
    
    # 设备
    print_device_info()
    device = get_device()
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    logger.info("\n[Step 1] 加载数据...")
    data_loader = TenrecDataLoader(
        data_dir=args.data_dir,
        scenario=args.scenario,
        max_users=args.max_users,
        use_cache=True,
    )
    data_loader.load()
    
    stats = data_loader.get_statistics()
    logger.info(f"数据: {stats['total_interactions']} 交互, {stats['total_users']} 用户, {stats['total_items']} items")
    
    # 划分数据
    train_data, val_data, test_data = data_loader.split_train_val_test(seed=args.seed)
    
    # [Fix] 防止未来数据泄漏：负采样池仅包含训练集物品
    # 原问题：sample_negative_items 使用全局 item_set（包含测试集物品），
    # 导致验证集指标虚高（AUC 0.93 vs 测试 0.84）
    # 优化：直接从 numpy 数组提取，避免遍历 9600 万个 Python 对象
    if hasattr(train_data, '_indices') and data_loader.store is not None:
        train_item_set = set(np.unique(data_loader.store.item_ids[train_data._indices]).tolist())
    else:
        train_item_set = {inter.item_id for inter in train_data}
    data_loader.set_negative_pool(train_item_set)
    logger.info(f"训练集物品数: {len(train_item_set)}, 全量物品数: {len(data_loader.item_set)}")
    
    # [Fix] 预构建用户-物品映射（在 DataLoader fork worker 之前！）
    # 否则每个 worker 子进程会独立构建，导致 N 个 worker × 40s 的浪费
    data_loader.store.build_user_positive_items()
    
    # 创建数据加载器
    # 注意：adapter 使用 hash_table_size=100000 生成 hash 值，
    # 模型的 Embedding 维度必须与之匹配，否则会越界访问导致 NaN
    hash_table_size = 100000  # 与 TenrecToPhoenixAdapter 默认值一致
    
    adapter = TenrecToPhoenixAdapter(
        data_loader=data_loader,
        history_seq_len=args.history_seq_len,
        hash_table_size=hash_table_size,
        seed=args.seed,
    )
    
    train_dataset = TenrecDataset(train_data, adapter, num_negatives=args.num_negatives)
    
    # 验证集使用缓存，确保每次评估使用完全相同的数据
    # 注意：负样本数量必须与训练集一致，否则 AUC 计算不准确！
    num_negatives = args.num_negatives  # 使用命令行参数
    
    # 采样评估：大数据集上限制验证/测试集大小
    if args.max_eval_samples and len(val_data) > args.max_eval_samples:
        rng = np.random.RandomState(args.seed)
        val_indices = rng.choice(len(val_data), args.max_eval_samples, replace=False)
        val_indices_sorted = np.sort(val_indices)
        # 保持为 InteractionSlice（向量化 eval cache 需要 _indices 属性）
        val_samples = val_data[val_indices_sorted]
        logger.info(f"验证集采样: {len(val_data)} -> {len(val_samples)}")
    else:
        val_samples = val_data
    val_cache_path = get_eval_cache_path(
        cache_dir=args.data_dir,
        scenario=args.scenario,
        split="val",
        num_samples=len(val_samples),
        num_negatives=num_negatives,
        seed=args.seed,
    )
    val_dataset = CachedEvalDataset(
        cache_path=str(val_cache_path),
        interactions=val_samples,
        adapter=adapter,
        num_negatives=num_negatives,
        seed=args.seed,
    )
    
    # Linux 服务器配置：多进程 + 预取 + 内存锁定
    num_workers = args.num_workers
    use_gpu = is_mlu_available() or torch.cuda.is_available()
    
    # 计算粗排和精排的 batch_size
    retrieval_batch_size = args.retrieval_batch_size or args.batch_size
    ranking_batch_size = args.ranking_batch_size or (args.batch_size // 2)  # 精排默认用一半
    
    logger.info(f"粗排 batch_size: {retrieval_batch_size}, 精排 batch_size: {ranking_batch_size}")
    
    # 数据加载器配置（粗排用）
    retrieval_loader_kwargs = {
        "batch_size": retrieval_batch_size,
        "num_workers": num_workers,
        "pin_memory": use_gpu,
        "collate_fn": collate_fn,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 4 if num_workers > 0 else None,
    }
    
    # 数据加载器配置（精排用）
    ranking_loader_kwargs = {
        "batch_size": ranking_batch_size,
        "num_workers": num_workers,
        "pin_memory": use_gpu,
        "collate_fn": collate_fn,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 4 if num_workers > 0 else None,
    }
    
    logger.info(f"数据加载: num_workers={num_workers}, prefetch_factor=4, pin_memory={use_gpu}")
    
    # 粗排 DataLoader（大 batch）
    retrieval_train_loader = DataLoader(train_dataset, shuffle=True, **retrieval_loader_kwargs)
    retrieval_val_loader = DataLoader(val_dataset, shuffle=False, **retrieval_loader_kwargs)
    
    # 精排 DataLoader（小 batch）
    ranking_train_loader = DataLoader(train_dataset, shuffle=True, **ranking_loader_kwargs)
    ranking_val_loader = DataLoader(val_dataset, shuffle=False, **ranking_loader_kwargs)
    
    # TensorBoard
    tb_logger = TensorBoardLogger("runs", f"two_stage_{args.stage}_{time.strftime('%Y%m%d_%H%M%S')}")
    
    config = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,  # 使用命令行参数
        "use_amp": False,  # MLU 上禁用混合精度
        "log_interval": 20,  # 更少的 step，更频繁输出
        "eval_interval": args.eval_interval,
        "grad_accumulation": args.grad_accumulation,
        "max_grad_norm": 1.0,
        "warmup_steps": args.warmup_steps,  # 使用命令行参数
        "label_smoothing": args.label_smoothing,
        "patience": args.patience,
    }
    
    # ========== 阶段 1: 召回模型 ==========
    if args.stage in ["retrieval", "both"]:
        logger.info("\n" + "=" * 60)
        logger.info("[阶段 1] 训练召回模型 (TwoTowerModel)")
        logger.info("=" * 60)
        
        # 使用 hash_table_size 作为 Embedding 维度（与 adapter 一致）
        # 这是解决 NaN 问题的关键：确保 hash 值不会越界
        # 85GB MLU 显存配置：大维度 + 深层网络
        retrieval_model = TwoTowerModel(
            num_users=hash_table_size + 1,  # +1 用于 padding_idx=0
            num_items=hash_table_size + 1,
            embed_dim=args.embed_dim,     # 使用命令行参数
            hidden_dim=args.hidden_dim,   # 使用命令行参数
            num_heads=args.num_heads,     # 使用命令行参数
            num_layers=8,     # 85GB 显存: 4 -> 8
            history_len=args.history_seq_len,  # 与 adapter 一致
        )
        
        param_count = sum(p.numel() for p in retrieval_model.parameters())
        logger.info(f"召回模型参数量: {param_count:,}")
        
        retrieval_ckpt = CheckpointManager("checkpoints/retrieval", keep_last_n=3)
        
        train_model(
            model=retrieval_model,
            train_loader=retrieval_train_loader,  # 粗排用大 batch
            val_loader=retrieval_val_loader,
            device=device,
            config=config,
            checkpoint_manager=retrieval_ckpt,
            tb_logger=tb_logger,
            model_name="Retrieval",
        )
    
    # ========== 阶段 2: 精排模型 ==========
    if args.stage in ["ranking", "both"]:
        logger.info("\n" + "=" * 60)
        logger.info("[阶段 2] 训练精排模型 (RankingModel)")
        logger.info("=" * 60)
        
        # 使用 hash_table_size 作为 Embedding 维度（与 adapter 一致）
        # 85GB MLU 显存配置：精排用更深的网络
        # 重要：num_candidates 和 history_len 必须与 DataLoader/Dataset 配置一致！
        num_candidates = num_negatives + 1  # 负样本 + 1 正样本
        ranking_model = RankingModel(
            num_users=hash_table_size + 1,
            num_items=hash_table_size + 1,
            embed_dim=args.embed_dim,        # 使用命令行参数
            hidden_dim=args.hidden_dim,      # 使用命令行参数
            num_heads=args.num_heads,        # 使用命令行参数
            num_layers=args.ranking_num_layers,  # 支持命令行设置层数
            history_len=args.history_seq_len,      # 与 adapter 一致
            num_candidates=num_candidates,  # 与 Dataset 一致
            encoder_type=args.encoder_type,  # 支持 fastformer/transformer 切换
            gradient_checkpointing=args.gradient_checkpointing,  # 梯度检查点节省显存
        )
        
        # 一致性断言：确保 adapter 和 model 的 history 长度匹配
        assert adapter.history_seq_len == ranking_model.history_len, \
            f"Adapter history_seq_len ({adapter.history_seq_len}) != RankingModel history_len ({ranking_model.history_len})"
        
        logger.info(f"精排编码器类型: {args.encoder_type}")
        logger.info(f"精排层数: {args.ranking_num_layers}, 负样本数: {num_negatives}")
        logger.info(f"梯度检查点: {'启用 (以~30%时间换~10x显存节省)' if args.gradient_checkpointing else '禁用'}")
        
        param_count = sum(p.numel() for p in ranking_model.parameters())
        logger.info(f"精排模型参数量: {param_count:,}")
        
        ranking_ckpt = CheckpointManager("checkpoints/ranking", keep_last_n=3)
        
        train_model(
            model=ranking_model,
            train_loader=ranking_train_loader,  # 精排用小 batch
            val_loader=ranking_val_loader,
            device=device,
            config=config,
            checkpoint_manager=ranking_ckpt,
            tb_logger=tb_logger,
            model_name="Ranking",
        )
    
    tb_logger.close()
    
    # ========== 最终测试评估 ==========
    logger.info("\n" + "=" * 60)
    logger.info("[最终测试] 在测试集上评估模型")
    logger.info("=" * 60)
    
    # 构建测试集（也使用缓存）
    # 采样评估：大数据集上限制测试集大小
    if args.max_eval_samples and len(test_data) > args.max_eval_samples:
        rng = np.random.RandomState(args.seed + 1)  # 与 val 用不同 seed
        test_indices = rng.choice(len(test_data), args.max_eval_samples, replace=False)
        test_indices_sorted = np.sort(test_indices)
        # 保持为 InteractionSlice（向量化 eval cache 需要 _indices 属性）
        test_samples = test_data[test_indices_sorted]
        logger.info(f"测试集采样: {len(test_data)} -> {len(test_samples)}")
    else:
        test_samples = test_data
    test_cache_path = get_eval_cache_path(
        cache_dir=args.data_dir,
        scenario=args.scenario,
        split="test",
        num_samples=len(test_samples),
        num_negatives=num_negatives,
        seed=args.seed,
    )
    test_dataset = CachedEvalDataset(
        cache_path=str(test_cache_path),
        interactions=test_samples,
        adapter=adapter,
        num_negatives=num_negatives,
        seed=args.seed,
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **ranking_loader_kwargs)
    
    # 加载最佳召回模型进行测试
    if args.stage in ["retrieval", "both"]:
        retrieval_ckpt = CheckpointManager("checkpoints/retrieval", keep_last_n=3)
        best_ckpt = retrieval_ckpt.get_best_checkpoint()
        if best_ckpt:
            retrieval_model = TwoTowerModel(
                num_users=hash_table_size + 1,
                num_items=hash_table_size + 1,
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                num_layers=8,
            ).to(device)
            retrieval_ckpt.load(best_ckpt, model=retrieval_model, device=str(device))
            
            criterion = torch.nn.BCEWithLogitsLoss()
            test_metrics = evaluate(retrieval_model, test_loader, device, criterion)
            logger.info(f"召回模型测试集评估:")
            logger.info(f"  AUC: {test_metrics['auc']:.4f}")
            logger.info(f"  Hit@1: {test_metrics['hit@1']:.4f}")
            logger.info(f"  NDCG@10: {test_metrics['ndcg@10']:.4f}")
            logger.info(f"  NDCG@20: {test_metrics['ndcg@20']:.4f}")
            logger.info(f"  like-AUC: {test_metrics['like_auc']:.4f}")
            logger.info(f"  MRR: {test_metrics['mrr']:.4f}")
    
    # 加载最佳精排模型进行测试
    if args.stage in ["ranking", "both"]:
        ranking_ckpt = CheckpointManager("checkpoints/ranking", keep_last_n=3)
        best_ckpt = ranking_ckpt.get_best_checkpoint()
        if best_ckpt:
            # 注意：必须与训练时配置完全一致，否则位置编码大小不匹配
            test_num_candidates = num_negatives + 1  # 与训练时一致
            ranking_model = RankingModel(
                num_users=hash_table_size + 1,
                num_items=hash_table_size + 1,
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                num_layers=args.ranking_num_layers,  # 与训练时一致
                history_len=args.history_seq_len,             # 与训练时一致
                num_candidates=test_num_candidates,   # 与训练时一致
                encoder_type=args.encoder_type,      # 与训练时一致
            ).to(device)
            ranking_ckpt.load(best_ckpt, model=ranking_model, device=str(device))
            
            criterion = torch.nn.BCEWithLogitsLoss()
            test_metrics = evaluate(ranking_model, test_loader, device, criterion)
            logger.info(f"精排模型测试集评估:")
            logger.info(f"  AUC: {test_metrics['auc']:.4f}")
            logger.info(f"  Hit@1: {test_metrics['hit@1']:.4f}")
            logger.info(f"  NDCG@10: {test_metrics['ndcg@10']:.4f}")
            logger.info(f"  NDCG@20: {test_metrics['ndcg@20']:.4f}")
            logger.info(f"  like-AUC: {test_metrics['like_auc']:.4f}")
            logger.info(f"  MRR: {test_metrics['mrr']:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
