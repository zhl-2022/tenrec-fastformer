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
from datetime import timedelta
from typing import Dict, Optional, List

# [Fix] 过滤 torch_mlu 的 is_compiling 过时警告，减少日志噪音
warnings.filterwarnings("ignore", message=".*is_compiling.*")

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# 添加模块路径
_current_dir = Path(__file__).parent
_project_root = _current_dir.parent
sys.path.insert(0, str(_project_root))

from tenrec_adapter.data_loader import TenrecDataLoader, TenrecInteraction, TenrecUserHistory
from tenrec_adapter.phoenix_adapter import TenrecToPhoenixAdapter
from tenrec_adapter.metrics import compute_auc
from tenrec_adapter.device_utils import get_device, print_device_info, is_mlu_available
from tenrec_adapter.models import TwoTowerModel
from tenrec_adapter.ranking_model import RankingModel
from tenrec_adapter.config_manager import load_config, get_default_config
from tenrec_adapter.tensorboard_logger import TensorBoardLogger
from tenrec_adapter.checkpoint_manager import CheckpointManager
from tenrec_adapter.eval_cache import CachedEvalDataset, get_eval_cache_path


def setup_ddp():
    """初始化分布式环境。"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        # ?? 10 ?????????????????????????????
        dist_timeout_sec = int(os.environ.get("TORCH_DIST_TIMEOUT", "3600"))
        dist_timeout = timedelta(seconds=dist_timeout_sec)

        if is_mlu_available():
            import torch_mlu
            torch.mlu.set_device(local_rank)
            dist.init_process_group(backend="cncl", init_method="env://", timeout=dist_timeout)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl", init_method="env://", timeout=dist_timeout)
        else:
            dist.init_process_group(backend="gloo", init_method="env://", timeout=dist_timeout)

        return rank, local_rank, world_size
    else:
        # 非分布式环境
        return 0, 0, 1

def cleanup_ddp():
    """清理分布式环境。"""
    if dist.is_initialized():
        dist.destroy_process_group()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TenrecDataset(Dataset):
    """
    Tenrec 数据集 (Map-style)。
    字段名统一为模型 forward() 期望的格式。
    """
    def __init__(
        self,
        interactions: List[TenrecInteraction],
        user_histories: Dict[int, TenrecUserHistory],
        num_negatives: int = 1,
        is_training: bool = True,
        data_loader: Optional[TenrecDataLoader] = None,
        item_category_map: Optional[np.ndarray] = None,
        history_seq_len: int = 10,
        num_actions: int = 4,
    ):
        self.interactions = interactions
        self.user_histories = user_histories
        self.num_negatives = num_negatives
        self.is_training = is_training
        self.data_loader = data_loader
        self.item_category_map = item_category_map
        self.history_seq_len = history_seq_len
        self.num_actions = num_actions

        self.len = len(self.interactions)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        inter = self.interactions[idx]
        user_id = inter.user_id
        item_id = inter.item_id

        # 1. 用户特征
        gender = inter.gender
        age = inter.age

        # 2. 历史序列 — padding 到固定长度
        raw_history = inter.hist_items  # List[int]
        hist_len = len(raw_history)
        S = self.history_seq_len

        # 截断或 padding
        if hist_len >= S:
            history_item_ids = raw_history[:S]
        else:
            history_item_ids = raw_history + [0] * (S - hist_len)

        # history_actions: TwoTowerModel 需要，Ranking 不需要
        # 用全 1 的 click 占位 (已交互的物品 action=click=1)
        history_actions = []
        for i in range(S):
            if i < hist_len:
                actions = [1.0] + [0.0] * (self.num_actions - 1)  # click=1, rest=0
            else:
                actions = [0.0] * self.num_actions  # padding
            history_actions.append(actions)

        # 3. 候选物品 (1 pos + N neg)
        candidate_item_ids = [item_id]
        labels_click = [1.0]

        if self.is_training:
            if self.data_loader:
                rng = np.random.default_rng(seed=idx + self.epoch * self.len + os.getpid())
                neg_items = self.data_loader.sample_negative_items(
                    user_id=user_id,
                    num_negatives=self.num_negatives,
                    rng=rng
                )
            else:
                neg_items = [0] * self.num_negatives

            candidate_item_ids.extend(neg_items)
            labels_click.extend([0.0] * self.num_negatives)

        # Multi-action labels: [C, num_actions]
        num_cand = len(candidate_item_ids)
        labels = []
        for i in range(num_cand):
            if i == 0:  # positive
                labels.append([1.0] + [0.0] * (self.num_actions - 1))
            else:
                labels.append([0.0] * self.num_actions)

        # 4. 候选物品 Category
        if self.item_category_map is not None:
            ids = np.array(candidate_item_ids, dtype=int)
            valid_mask = ids < len(self.item_category_map)
            cats = np.zeros_like(ids)
            cats[valid_mask] = self.item_category_map[ids[valid_mask]]
            candidate_categories = cats.tolist()
        else:
            candidate_categories = [0] * num_cand

        return {
            # === 统一字段名 (匹配 TwoTowerModel + RankingModel forward) ===
            "user_ids": user_id,                               # scalar
            "user_id": user_id,                                # alias for RankingModel
            "user_gender": gender,                             # scalar
            "user_age": age,                                   # scalar
            "gender": gender,                                  # alias for RankingModel
            "age": age,                                        # alias for RankingModel
            "history_item_ids": history_item_ids,              # [S]
            "history_actions": history_actions,                 # [S, num_actions]
            "candidate_item_ids": candidate_item_ids,          # [1+N]
            "candidate_categories": candidate_categories,      # [1+N]
            "labels": labels,                                  # [1+N, num_actions]
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
    """Collate 函数 — numpy stacking + 单次 tensor 转换 (高性能版)。"""
    return {
        "user_ids": torch.tensor(np.array([b["user_ids"] for b in batch]), dtype=torch.long),
        "user_id": torch.tensor(np.array([b["user_id"] for b in batch]), dtype=torch.long),
        "user_gender": torch.tensor(np.array([b["user_gender"] for b in batch]), dtype=torch.long),
        "user_age": torch.tensor(np.array([b["user_age"] for b in batch]), dtype=torch.long),
        "gender": torch.tensor(np.array([b["gender"] for b in batch]), dtype=torch.long),
        "age": torch.tensor(np.array([b["age"] for b in batch]), dtype=torch.long),
        "history_item_ids": torch.tensor(np.array([b["history_item_ids"] for b in batch]), dtype=torch.long),
        "history_actions": torch.tensor(np.array([b["history_actions"] for b in batch]), dtype=torch.float32),
        "candidate_item_ids": torch.tensor(np.array([b["candidate_item_ids"] for b in batch]), dtype=torch.long),
        "candidate_categories": torch.tensor(np.array([b["candidate_categories"] for b in batch]), dtype=torch.long),
        "labels": torch.tensor(np.array([b["labels"] for b in batch]), dtype=torch.float32),
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
            logits_preds = torch.sigmoid(logits).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_preds.append(logits_preds)
            all_labels.append(labels)

    # 局部聚合
    local_scores = np.concatenate(all_scores, axis=0)
    local_preds = np.concatenate(all_preds, axis=0)
    local_labels = np.concatenate(all_labels, axis=0)

    # [DDP] 全局聚合 (如果使用多卡)
    if dist.is_initialized():
        # 1. 收集各卡数据大小
        local_size = torch.tensor([local_scores.shape[0]], device=device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        dist.all_gather(all_sizes, local_size)
        max_size = max([x.item() for x in all_sizes])

        # 2. Pad 到最大长度 (all_gather 要求 tensor 大小一致)
        def pad_and_gather(data_np, name):
            local_tensor = torch.from_numpy(data_np).to(device)
            pad_len = max_size - local_tensor.shape[0]
            if pad_len > 0:
                padding = torch.zeros((pad_len, *local_tensor.shape[1:]), dtype=local_tensor.dtype, device=device)
                local_tensor = torch.cat([local_tensor, padding], dim=0)

            gathered_list = [torch.zeros_like(local_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_list, local_tensor)

            # 去除 padding 并拼接
            data_list = []
            for size_tensor, tensor in zip(all_sizes, gathered_list):
                valid_len = size_tensor.item()
                data_list.append(tensor[:valid_len].cpu().numpy())
            return np.concatenate(data_list, axis=0)

        all_scores = pad_and_gather(local_scores, "scores")
        all_preds = pad_and_gather(local_preds, "preds")
        all_labels = pad_and_gather(local_labels, "labels")
    else:
        all_scores = local_scores
        all_preds = local_preds
        all_labels = local_labels

    # 仅 Master 计算指标 (数据已全量聚合)
    if not (dist.is_initialized() and dist.get_rank() != 0):
        # click 是第一个 action (idx=0)
        click_labels = all_labels[:, :, 0]  # [N, C]
        click_preds = all_preds[:, :, 0]  # [N, C]

        # 计算两种 AUC 进行对比
        # 方式 1: 使用 scores（点积分数）- 我们的实现
        auc_scores = compute_auc(click_labels.flatten(), all_scores.flatten())

        # 方式 2: 使用 sigmoid(click)（概率）- Phoenix 原版方式
        auc_sigmoid = compute_auc(click_labels.flatten(), click_preds.flatten())

        logger.info(f"  [AUC对比] scores方式: {auc_scores:.4f}, sigmoid方式: {auc_sigmoid:.4f}")
        auc = auc_sigmoid

        # 找到每个样本中正样本的真实位置
        # 过滤掉没有正样本的行
        has_positive = click_labels.max(axis=1) > 0  # [N]

        pred_ranks = np.argsort(-all_scores, axis=1)  # 按分数降序排列 [N, C]

        if has_positive.any():
            valid_labels = click_labels[has_positive]
            valid_ranks = pred_ranks[has_positive]

            # Vectorized MRR / Hit@1 Calculation
            # 找到正样本的索引 (argmax 返回第一个最大值索引，既然是 binary 0/1，就是 1 的位置)
            positive_pos = np.argmax(valid_labels, axis=1) # [M]

            # 检查 rank 0 是否是正样本
            hit_at_1 = np.mean(valid_ranks[:, 0] == positive_pos)

            # 找到正样本在预测排序中的位置 (1-based)
            # numpy broadcasting: valid_ranks [M, C] == positive_pos [M, 1]
            rows = np.arange(len(positive_pos))
            # argmax on boolean gives index of first True
            output_rank_indices = np.argmax(valid_ranks == positive_pos[:, None], axis=1)
            positive_rank_values = output_rank_indices + 1

            mrr = np.mean(1.0 / positive_rank_values)
        else:
            hit_at_1 = 0.0
            mrr = 0.0

        # 计算 NDCG@10 和 NDCG@20
        from tenrec_adapter.metrics import compute_ndcg_at_k
        ndcg_10 = compute_ndcg_at_k(click_labels, all_scores, k=10)
        ndcg_20 = compute_ndcg_at_k(click_labels, all_scores, k=20)

        # 计算 like-AUC
        like_labels = all_labels[:, :, 1]
        like_preds = all_preds[:, :, 1]
        like_auc = compute_auc(like_labels.flatten(), like_preds.flatten())

        avg_loss = total_loss / max(num_batches, 1)

        return {
            "loss": avg_loss,
            "auc": auc,
            "auc_sigmoid": auc_sigmoid,
            "like_auc": like_auc,
            "hit@1": hit_at_1,
            "mrr": mrr,
            "ndcg@10": ndcg_10,
            "ndcg@20": ndcg_20,
        }

    # 非 Master 节点返回空字典 (或本地 loss)
    return {"loss": total_loss / max(num_batches, 1), "auc": 0.0}


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
    is_master: bool = True,
):
    """
    训练模型（支持 DDP）。
    """
    # model 已经在外部 to(device) 和 DDP wrap

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
    if label_smoothing > 0 and is_master:
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
    use_amp = config.get("use_amp", False)

    if use_amp and torch.cuda.is_available() and not is_mlu_available():
        scaler = torch.amp.GradScaler("cuda")
        autocast = lambda: torch.amp.autocast("cuda")
        if is_master: logger.info("启用 CUDA 混合精度 (FP16)")
    else:
        use_amp = False
        scaler = None
        autocast = None
        if is_master and is_mlu_available():
            logger.info("MLU 使用 FP32 精度（禁用混合精度以避免 NaN）")

    epochs = config.get("epochs", 10)
    log_interval = config.get("log_interval", 50)
    eval_interval = config.get("eval_interval", 500)
    grad_accumulation = config.get("grad_accumulation", 1)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    max_steps_per_epoch = config.get("max_steps_per_epoch", 0)

    best_auc = 0
    global_step = 0
    patience = config.get("patience", 0)
    no_improve_count = 0
    use_ddp = dist.is_initialized() and dist.get_world_size() > 1
    world_size = dist.get_world_size() if use_ddp else 1

    if is_master:
        logger.info(f"开始训练 {model_name}...")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  混合精度: {use_amp}")
        if patience > 0:
            logger.info(f"  Early Stopping: patience={patience} evals")

    for epoch in range(epochs):
        model.train()
        # [DDP Critical] 设置 DistributedSampler 的 epoch 以确保 shuffle 不同
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # [Bug Fix] 通知 Dataset 当前 epoch，使负采样 seed 随 epoch 变化 (Store 级)
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)

        epoch_loss = 0
        num_batches = 0
        epoch_start = time.time()
        log_window_start = time.time()
        log_window_steps = 0

        if is_master:
            logger.info(f"\n--- {model_name} Epoch {epoch + 1}/{epochs} ---")

        nan_debug_printed = False
        nan_count = 0
        max_nan_before_reset = 50

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # ===== 输入数据 NaN 检测 =====
            has_input_nan = False
            for key, val in batch.items():
                if torch.isnan(val).any() or torch.isinf(val).any():
                    has_input_nan = True
                    if not nan_debug_printed and is_master:
                        logger.warning(f"  输入数据 '{key}' 包含 NaN/Inf!")

            if has_input_nan:
                if is_master and not nan_debug_printed:
                    logger.warning(f"  Step {step + 1}: 输入数据包含 NaN/Inf，已替换为 0 (避免 DDP 死锁)")
                    nan_debug_printed = True

                # [DDP Fix] 不能 continue，否则也是 rank 步数不一致导致的死锁
                # 修复方案：替换为 0，继续 forward/backward
                for key, val in batch.items():
                    if torch.is_tensor(val) and torch.is_floating_point(val):
                        batch[key] = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)

            # ===== 输入数据 Clamp =====
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
                if not nan_debug_printed and is_master:
                    logger.warning(f"  [NaN Debug] Step {step + 1}: Loss is NaN/Inf!")
                    nan_debug_printed = True

            if is_nan:
                nan_count += 1
                if nan_count >= max_nan_before_reset:
                    if is_master: logger.warning(f"  连续 {nan_count} 次 NaN，重新初始化模型权重...")
                    # DDP 模型取 .module
                    if hasattr(model, 'module'):
                        model.module.apply(init_weights)
                    else:
                        model.apply(init_weights)
                    nan_count = 0

                optimizer.zero_grad()
                if use_amp: scaler.update()
                continue
            else:
                nan_count = 0

            # 聚合多卡 Loss用于显示 (可选，为性能通常只打 log master 的 loss 或不做 all_reduce)
            # 这里简单起见，只打印 master 卡的 loss
            epoch_loss += loss.item() * grad_accumulation
            num_batches += 1
            log_window_steps += 1

            if (step + 1) % grad_accumulation == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if tb_logger and is_master:
                    tb_logger.log_scalar(f"{model_name}/train_loss", loss.item() * grad_accumulation, global_step)
                    tb_logger.log_scalar(f"{model_name}/learning_rate", scheduler.get_last_lr()[0], global_step)

            if (step + 1) % log_interval == 0 and is_master:
                avg_loss = epoch_loss / max(num_batches, 1)
                elapsed = max(time.time() - log_window_start, 1e-6)
                local_batch = train_loader.batch_size or 0
                samples_per_sec = 0.0
                if local_batch > 0:
                    samples_per_sec = (log_window_steps * local_batch * world_size) / elapsed
                logger.info(
                    f"  Step {step + 1}/{len(train_loader)}, Loss: {avg_loss:.4f}, "
                    f"Throughput: {samples_per_sec:.1f} samples/s"
                )
                log_window_start = time.time()
                log_window_steps = 0

            if max_steps_per_epoch > 0 and (step + 1) >= max_steps_per_epoch:
                if is_master:
                    logger.info(f"  Reached max_steps_per_epoch={max_steps_per_epoch}, stop current epoch early")
                break

            # [注: step-level 评估已移除, 仅在 Epoch 结束时评估]

        # Epoch 结束
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(num_batches, 1)
        if is_master:
            logger.info(f"  Epoch {epoch + 1} ?? - Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s")

        if num_batches == 0:
            if is_master:
                logger.error(f"  ???Epoch {epoch + 1} ?????? NaN???????????")
            if hasattr(model, 'module'):
                model.module.apply(init_weights)
            else:
                model.apply(init_weights)

        val_metrics = None
        if val_loader is not None:
            # evaluate() ??? all_gather????? rank ????
            val_metrics = evaluate(model, val_loader, device, criterion)

        should_stop = False
        if is_master and val_metrics is not None:
            logger.info(
                f"  Epoch {epoch + 1} ?? AUC: {val_metrics['auc']:.4f}, "
                f"like-AUC: {val_metrics['like_auc']:.4f}, NDCG@20: {val_metrics['ndcg@20']:.4f}"
            )

            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                no_improve_count = 0
                if checkpoint_manager:
                    checkpoint_manager.save(
                        model.module if hasattr(model, 'module') else model,
                        optimizer, epoch, global_step,
                        metrics=val_metrics, is_best=True,
                    )
                logger.info(f"  [???] Epoch {epoch + 1} AUC: {best_auc:.4f} - ??????")
            else:
                no_improve_count += 1
                if patience > 0 and no_improve_count >= patience:
                    logger.info(f"  [Early Stop] AUC ?? {patience} ? Epoch ?????")
                    should_stop = True

            if tb_logger:
                tb_logger.log_scalar(f"{model_name}/epoch_auc", val_metrics['auc'], epoch + 1)
                tb_logger.flush()

        if use_ddp:
            stop_tensor = torch.tensor(1 if should_stop else 0, device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())

        # [DDP Fix] Epoch ????? rank ?? (??????)
        if use_ddp:
            dist.barrier()

        if should_stop:
            break
    logger.info(f"{model_name} 训练完成，最佳 AUC: {best_auc:.4f}")
    return best_auc


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description="两阶段推荐系统训练")
    parser.add_argument("--stage", type=str, default="both", choices=["retrieval", "ranking", "both"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data/tenrec/Tenrec")
    parser.add_argument("--scenario", type=str, default="QB-video")
    parser.add_argument("--run_name", type=str, default=None,
                        help="运行名称，用于 checkpoint 和 TensorBoard 子目录隔离 (默认=scenario)")
    parser.add_argument("--user_attr_file", type=str, default=None, help="用户属性文件 (默认 user_attr.csv)")
    parser.add_argument("--item_attr_file", type=str, default=None, help="物品属性文件 (默认 video_attr.csv)")
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
    parser.add_argument("--max_steps_per_epoch", type=int, default=0,
                        help="Maximum training steps per epoch (0 means no limit)")
    parser.add_argument("--ranking_num_layers", type=int, default=12,
                        help="精排 Transformer 层数 (默认 12, 可选 16/20 更深网络)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding 维度")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="隐藏层维度")
    parser.add_argument("--num_heads", type=int, default=16, help="Attention 头数")
    parser.add_argument("--warmup_steps", type=int, default=100, help="学习率预热步数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")

    # Temperature Scaling
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for Softmax/Contrastive Loss")

    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label Smoothing 系数 (0=禁用, 0.1=推荐)")
    parser.add_argument("--patience", type=int, default=5, help="Early Stopping 耐心值 (0=禁用, 5=推荐)")
    parser.add_argument("--eval_interval", type=int, default=200, help="评估间隔步数 (大数据集建议 1000+)")
    parser.add_argument("--history_seq_len", type=int, default=64, help="历史序列长度 (ctr_data_1M 预构建历史建议 10)")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="验证/测试集最大样本数 (大数据集建议 500000)")
    parser.add_argument("--grad_accumulation", type=int, default=1, help="梯度累积步数 (2=有效batch翻倍, 显存不变)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="启用梯度检查点（以~30%%训练时间换取~10x激活显存节省）")
    parser.add_argument("--use_amp", action="store_true",
                        help="启用混合精度训练 (MLU 设备自动禁用)")

    args = parser.parse_args()

    # run_name 默认与 scenario 相同
    if args.run_name is None:
        args.run_name = args.scenario

    # DDP 初始化
    rank, local_rank, world_size = setup_ddp()
    is_master = (rank == 0)
    use_ddp = world_size > 1

    # 设置随机种子 (每个 rank 使用不同种子，或者相同种子但通过 sampler 切分)
    # 通常 DDP 中 base seed 相同，DistributedSampler 会根据 rank 切分
    seed = args.seed + rank # 避免 pytorch 默认行为导致的随机性重叠（虽然 DistributedSampler 会处理）
    torch.manual_seed(seed)
    np.random.seed(seed)

    if is_master:
        logger.info("=" * 60)
        logger.info(f"两阶段推荐系统训练 (DDP Mode, World Size: {world_size})")
        logger.info(f"阶段: {args.stage}")
        logger.info("=" * 60)

    # 设备
    if is_master:
        print_device_info()

    # DDP 模式下，device 由 local_rank 决定
    if is_mlu_available():
        device = torch.device(f"mlu:{local_rank}")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_master:
        logger.info(f"使用设备: {device}")

    # 加载数据 (只在 Master 节点加载，或者每个节点加载但由于 mmap/ZeroCopy 不会占用多倍内存)
    # TenrecDataLoader 使用 mmap，多进程安全且节省内存
    if is_master:
        logger.info("\n[Step 1] 加载数据...")

    data_loader = TenrecDataLoader(
        data_dir=args.data_dir,
        scenario=args.scenario,
        max_users=args.max_users,
        use_cache=True,
    )

    # [DDP Memory Fix] 避免 cache miss 时 8 个 rank 同时读 9.9GB CSV 导致主机 OOM
    # 按 rank 串行加载：rank0 先构建/命中缓存，其他 rank 再依次从缓存加载
    if world_size > 1:
        for r in range(world_size):
            if rank == r:
                if is_master and r == 0:
                    logger.info("[DDP] 启用按 rank 串行加载数据，降低内存峰值")
                data_loader.load()
            dist.barrier()
    else:
        data_loader.load()

    if is_master:
        stats = data_loader.get_statistics()
        logger.info(f"数据: {stats['total_interactions']} 交互, {stats['total_users']} 用户, {stats['total_items']} items")

    # 划分数据
    train_data, val_data, test_data = data_loader.split_train_val_test(seed=args.seed)

    # [Fix] 防止未来数据泄漏：负采样池仅包含训练集物品
    if hasattr(train_data, '_indices') and data_loader.store is not None:
        train_item_set = set(np.unique(data_loader.store.item_ids[train_data._indices]).tolist())
    else:
        train_item_set = {inter.item_id for inter in train_data}
    data_loader.set_negative_pool(train_item_set)
    if is_master:
        logger.info(f"训练集物品数: {len(train_item_set)}, 全量物品数: {len(data_loader.item_set)}")

    # [Fix] 预构建用户-物品映射
    data_loader.store.build_user_positive_items()

    # 获取最大 ID 用于 Embedding 大小
    max_user_id = data_loader.store.user_ids.max()
    max_item_id = data_loader.store.item_ids.max()
    if data_loader.item_category_map is not None:
        max_item_id = max(max_item_id, len(data_loader.item_category_map) - 1)

    if is_master:
        logger.info(f"Max User ID: {max_user_id}, Max Item ID: {max_item_id}")

    # 创建训练集
    train_dataset = TenrecDataset(
        interactions=train_data,
        user_histories=data_loader.user_histories,
        num_negatives=args.num_negatives,
        is_training=True,
        data_loader=data_loader,
        item_category_map=data_loader.item_category_map,
        history_seq_len=args.history_seq_len,
    )

    # DDP Sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    # 验证集
    num_negatives = args.num_negatives

    if args.max_eval_samples and len(val_data) > args.max_eval_samples:
        rng = np.random.RandomState(args.seed)
        val_indices = rng.choice(len(val_data), args.max_eval_samples, replace=False)
        val_indices_sorted = np.sort(val_indices)
        val_samples = val_data[val_indices_sorted]
        if is_master:
            logger.info(f"验证集采样: {len(val_data)} -> {len(val_samples)}")
    else:
        val_samples = val_data

    # 验证 Dataset
    val_dataset = TenrecDataset(
        interactions=val_samples,
        user_histories=data_loader.user_histories,
        num_negatives=num_negatives,
        is_training=True, # 验证集也需要 negative sampling 来计算 AUC
        data_loader=data_loader,
        item_category_map=data_loader.item_category_map,
        history_seq_len=args.history_seq_len,
    )

    # Val Loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=DistributedSampler(val_dataset, shuffle=False) if use_ddp else None,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 模型初始化
    config = {
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "use_amp": args.use_amp,
        "log_interval": 50,
        "eval_interval": args.eval_interval,  # Use args
        "patience": args.patience,
        "warmup_steps": args.warmup_steps,
        "grad_accumulation": args.grad_accumulation,
        "label_smoothing": args.label_smoothing,
        "max_steps_per_epoch": args.max_steps_per_epoch,
    }

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


    # 粗排模型
    if args.stage in ["retrieval", "both"]:
        # 1. 召回模型

        # 计算类别数量
        num_categories = 0
        if data_loader.item_category_map is not None:
            num_categories = len(data_loader.item_category_map) # 0 is padding, max_cat < len

        retrieval_model = TwoTowerModel(
            num_users=int(max_user_id) + 1,
            num_items=int(max_item_id) + 1,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=2, # TwoTower 默认2层
            history_len=args.history_seq_len,
            dropout=0.1,
            temperature=args.temperature,
            num_categories=num_categories,
        )

        # 加载预训练（如果指定）
        if args.retrieval_model and os.path.exists(args.retrieval_model):
            if is_master:
                logger.info(f"加载预训练召回模型: {args.retrieval_model}")
            # map_location 到指定 device
            checkpoint = torch.load(args.retrieval_model, map_location=device)
            # 处理 DDP state_dict key 前缀 "module."
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            retrieval_model.load_state_dict(new_state_dict, strict=False)

        # DDP Wrap
        retrieval_model.to(device)
        # MLU DDP 不需要 device_ids
        if is_mlu_available():
             retrieval_model = DDP(retrieval_model, broadcast_buffers=False, find_unused_parameters=True)
        else:
             retrieval_model = DDP(retrieval_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        # 检查点管理器 (只在 master 保存)
        retrieval_ckpt_manager = CheckpointManager(
            save_dir=os.path.join("checkpoints", args.run_name, "retrieval"),
            keep_last_n=3,
        ) if is_master else None

        # TensorBoard (只在 master 记录)
        tb_logger = TensorBoardLogger(
            log_dir=os.path.join("logs", args.run_name, "retrieval_ddp"),
        ) if is_master else None

        # 训练配置
        config = {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "eval_interval": args.eval_interval, # DDP 模式下通常按 epoch 评估，或者减少 eval 频率
            "patience": args.patience,
            "grad_accumulation": args.grad_accumulation,
            "use_amp": False, # MLU 禁用 AMP
            "label_smoothing": args.label_smoothing,
        }

        best_auc = train_model(
            model=retrieval_model,
            train_loader=train_loader,
            val_loader=val_loader, # 所有 rank 都传 loader，用于分布式评估
            device=device,
            config=config,
            checkpoint_manager=retrieval_ckpt_manager, # master only
            tb_logger=tb_logger, # master only
            model_name="retrieval",
            is_master=is_master, # 传递给 train_model
        )

        if is_master:
            logger.info(f"召回模型训练完成，最佳 AUC: {best_auc:.4f}")

    # 粗排 (Retrieval) 结束

    # 精排 (Ranking)
    if args.stage in ["ranking", "both"]:
        if is_master:
            logger.info("\n" + "=" * 60)
            logger.info("阶段 2: 训练精排模型 (Ranking)")
            logger.info("=" * 60)

        # 释放召回模型显存 (如果只跑 Ranking 或者 Both 且显存紧张)
        # if 'retrieval_model' in locals():
        #     del retrieval_model
        #     torch.cuda.empty_cache()

        # 数据加载器配置（精排用）
        ranking_loader_kwargs = {
            "batch_size": ranking_batch_size,
            "num_workers": num_workers,
            "pin_memory": use_gpu,
            "collate_fn": collate_fn,
            "persistent_workers": num_workers > 0,
            "prefetch_factor": 4 if num_workers > 0 else None,
        }

        # 这里的 Dataset 和 Loader 代码似乎缺失了？
        # 通常需要重新创建 Dataset (因为 Ranking 需要负采样或者不同的采样策略)
        # 但 TenrecDataset 支持 negative sampling.
        # 精排通常是对 Retrieval 召回的结果进行排序 (训练时则是对 Pos + Neg 进行排序)
        # 下面的代码直接用了 num_candidates 和 adapter?
        # 注意: 上下文里 ranking_train_loader 还没定义!

        # 定义 Ranking Train Loader
        ranking_train_dataset = TenrecDataset(
            interactions=train_data,
            user_histories=data_loader.user_histories,
            num_negatives=args.num_negatives,
            is_training=True,
            data_loader=data_loader,
            item_category_map=data_loader.item_category_map,
            history_seq_len=args.history_seq_len,
        )
        ranking_train_sampler = DistributedSampler(ranking_train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        ranking_train_loader = DataLoader(
            ranking_train_dataset,
            batch_size=ranking_batch_size,
            sampler=ranking_train_sampler,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=use_gpu,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
            drop_last=True,
        )

        # Ranking Val Loader
        ranking_val_loader = DataLoader(
            val_dataset, # 复用 val_dataset
            batch_size=ranking_batch_size,
            sampler=DistributedSampler(val_dataset, shuffle=False) if use_ddp else None,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=use_gpu
        )

        # 重要：num_candidates 和 history_len 必须与 DataLoader/Dataset 配置一致！
        num_candidates = args.num_negatives + 1  # 负样本 + 1 正样本

        # 计算类别数量
        num_categories = 0
        if data_loader.item_category_map is not None:
            num_categories = len(data_loader.item_category_map)

        ranking_model = RankingModel(
            num_users=int(max_user_id) + 1,
            num_items=int(max_item_id) + 1,
            embed_dim=args.embed_dim,        # 使用命令行参数
            hidden_dim=args.hidden_dim,      # 使用命令行参数
            num_heads=args.num_heads,        # 使用命令行参数
            num_layers=args.ranking_num_layers,  # 支持命令行设置层数
            history_len=args.history_seq_len,      # 与 adapter 一致
            num_candidates=num_candidates,  # 与 Dataset 一致
            encoder_type=args.encoder_type,  # 支持 fastformer/transformer 切换
            gradient_checkpointing=args.gradient_checkpointing,  # 梯度检查点节省显存
            num_categories=num_categories,
        )

        # DDP Wrap Ranking Model
        ranking_model.to(device)
        if is_mlu_available():
             ranking_model = DDP(ranking_model, broadcast_buffers=False, find_unused_parameters=True)
        else:
             ranking_model = DDP(ranking_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        if is_master:
            logger.info(f"精排编码器类型: {args.encoder_type}")
            logger.info(f"精排层数: {args.ranking_num_layers}, 负样本数: {num_negatives}")
            logger.info(f"梯度检查点: {'启用 (以~30%时间换~10x显存节省)' if args.gradient_checkpointing else '禁用'}")

            # Count params of original model (inside DDP wrapper)
            original_model = ranking_model.module if hasattr(ranking_model, 'module') else ranking_model
            param_count = sum(p.numel() for p in original_model.parameters())
            logger.info(f"精排模型参数量: {param_count:,}")

        ranking_ckpt = CheckpointManager(
            save_dir=os.path.join("checkpoints", args.run_name, "ranking"),
            keep_last_n=3,
        ) if is_master else None

        # TensorBoard for Ranking
        stage_tb_logger = TensorBoardLogger(
            log_dir=os.path.join("logs", args.run_name, "ranking_ddp"),
        ) if is_master else None

        # 精排专用配置（不复用粗排的 config，避免被覆盖）
        ranking_config = {
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "use_amp": False,  # MLU 禁用 AMP
            "log_interval": 50,
            "eval_interval": args.eval_interval,
            "patience": args.patience,
            "warmup_steps": args.warmup_steps,
            "grad_accumulation": args.grad_accumulation,
            "label_smoothing": args.label_smoothing,
            "max_steps_per_epoch": args.max_steps_per_epoch,
        }

        train_model(
            model=ranking_model,
            train_loader=ranking_train_loader,  # 精排用小 batch
            val_loader=ranking_val_loader,      # 所有 rank 都传 loader
            device=device,
            config=ranking_config,
            checkpoint_manager=ranking_ckpt,
            tb_logger=stage_tb_logger,
            model_name="Ranking",
            is_master=is_master,
        )

        if is_master and stage_tb_logger:
            stage_tb_logger.close()

    # 构建测试集
    # 采样评估：大数据集上限制测试集大小
    if args.max_eval_samples and len(test_data) > args.max_eval_samples:
        rng = np.random.RandomState(args.seed + 1)
        test_indices = rng.choice(len(test_data), args.max_eval_samples, replace=False)
        test_indices_sorted = np.sort(test_indices)
        test_samples = test_data[test_indices_sorted]
        if is_master:
            logger.info(f"测试集采样: {len(test_data)} -> {len(test_samples)}")
    else:
        test_samples = test_data

    # 使用 TenrecDataset (不再支持 CachedEvalDataset)
    test_dataset = TenrecDataset(
        interactions=test_samples,
        user_histories=data_loader.user_histories,
        num_negatives=args.num_negatives,
        is_training=True, # 测试时也需要负采样算指标
        data_loader=data_loader,
        item_category_map=data_loader.item_category_map,
        history_seq_len=args.history_seq_len,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=ranking_batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_gpu
    )

    # 仅 Master 节点执行测试评估
    if is_master:
        # 加载最佳召回模型进行测试
        if args.stage in ["retrieval", "both"]:
            retrieval_ckpt = CheckpointManager(
                save_dir=os.path.join("checkpoints", args.run_name, "retrieval"),
                keep_last_n=3,
            )
            best_ckpt = retrieval_ckpt.get_best_checkpoint()
            if best_ckpt:
                # 使用 max_user_id/max_item_id 而非 hash_table_size
                retrieval_model = TwoTowerModel(
                    num_users=int(max_user_id) + 1,
                    num_items=int(max_item_id) + 1,
                    embed_dim=args.embed_dim,
                    hidden_dim=args.hidden_dim,
                    num_heads=args.num_heads,
                    num_layers=2, # TwoTower default
                    history_len=args.history_seq_len,
                    num_categories=num_categories,
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
            ranking_ckpt = CheckpointManager(
                save_dir=os.path.join("checkpoints", args.run_name, "ranking"),
                keep_last_n=3,
            )
            best_ckpt = ranking_ckpt.get_best_checkpoint()
            if best_ckpt:
                test_num_candidates = args.num_negatives + 1  # 负样本 + 1 正样本

                # 计算类别数量
                num_categories = 0
                if data_loader.item_category_map is not None:
                    num_categories = len(data_loader.item_category_map)

                ranking_model = RankingModel(
                    num_users=int(max_user_id) + 1,
                    num_items=int(max_item_id) + 1,
                    embed_dim=args.embed_dim,
                    hidden_dim=args.hidden_dim,
                    num_heads=args.num_heads,
                    num_layers=args.ranking_num_layers,  # 与训练时一致
                    history_len=args.history_seq_len,             # 与训练时一致
                    num_candidates=test_num_candidates,   # 与训练时一致
                    encoder_type=args.encoder_type,      # 与训练时一致
                    num_categories=num_categories,
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

    if is_master:
        logger.info("\n" + "=" * 60)
        logger.info("训练完成！")
        logger.info("=" * 60)

    # 清理
    cleanup_ddp()

if __name__ == "__main__":
    main()
