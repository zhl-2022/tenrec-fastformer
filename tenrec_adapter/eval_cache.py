#!/usr/bin/env python
# MIT License - see LICENSE for details

"""
验证集缓存管理器。

提供验证集的持久化缓存，确保每次评估使用完全相同的数据。
优化说明：
v3 -> v4: 存储格式从 List[Dict] 改为 Dict[str, Tensor]（列式存储），
将 2.2GB Pickle 加载时间从 400s+ 降低到 <5s。
"""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CachedEvalDataset(Dataset):
    """
    缓存的评估数据集。
    
    预先构建完整的验证/测试数据并缓存到磁盘，确保：
    1. 每次评估使用完全相同的数据
    2. 加速评估过程（无需重复采样）
    3. 支持可复现的实验
    
    优化：使用列式 Tensor 存储，极大加速加载速度。
    
    Args:
        cache_path: 缓存文件路径
        interactions: 交互数据列表
        adapter: Phoenix 适配器
        num_negatives: 负样本数量
        seed: 随机种子
        force_rebuild: 强制重建缓存
    """
    
    CACHE_VERSION = 4  # v4: 列式 Tensor 存储
    
    def __init__(
        self,
        cache_path: str,
        interactions: List,
        adapter,
        num_negatives: int = 4,
        seed: int = 42,
        force_rebuild: bool = False,
    ):
        self.cache_path = Path(cache_path)
        self.interactions = interactions
        self.adapter = adapter
        self.num_negatives = num_negatives
        self.seed = seed
        
        # 缓存的数据 (列式存储)
        self.data: Dict[str, torch.Tensor] = {}
        self.length = 0
        
        # 尝试加载缓存，或构建新缓存
        if not force_rebuild and self._load_cache():
            pass
        else:
            self._build_cache()
            self._save_cache()
    
    def _get_cache_key(self) -> str:
        """生成缓存 key。"""
        key_parts = [
            f"v{self.CACHE_VERSION}",
            f"n{len(self.interactions)}",
            f"neg{self.num_negatives}",
            f"seed{self.seed}",
        ]
        return "_".join(key_parts)
    
    def _load_cache(self) -> bool:
        """从磁盘加载缓存。"""
        if not self.cache_path.exists():
            return False
        
        try:
            start_time = time.time()
            # 使用 torch.load 加载 tensor 字典，速度极快
            cache_content = torch.load(self.cache_path, map_location="cpu", weights_only=False)
            
            # 兼容性检查：如果是旧版 pickle 格式 (dict with 'data' key)，则认为无效
            if not isinstance(cache_content, dict) or "meta" not in cache_content:
                logger.info("缓存格式不匹配，重建缓存")
                return False

            meta = cache_content["meta"]
            
            # 验证缓存元数据
            if meta.get("version") != self.CACHE_VERSION:
                logger.info(f"缓存版本不匹配 (期望 v{self.CACHE_VERSION}, 实际 v{meta.get('version')})，重建缓存")
                return False
            
            if meta.get("cache_key") != self._get_cache_key():
                logger.info("缓存 key 不匹配，重建缓存")
                return False
            
            self.data = cache_content["data"]
            self.length = meta["num_samples"]
            
            elapsed = time.time() - start_time
            logger.info(f"从缓存加载验证集: {self.length} 样本, 耗时 {elapsed:.2f}s")
            return True
            
        except Exception as e:
            logger.warning(f"缓存加载失败: {e}")
            return False
    
    def _build_cache(self):
        """构建缓存数据（向量化实现）。"""
        n = len(self.interactions)
        num_neg = self.num_negatives
        logger.info(f"构建验证集缓存: {n} 样本, {num_neg} 负样本...")
        start_time = time.time()
        
        # 尝试使用向量化快速路径
        if self._try_build_cache_vectorized():
            elapsed = time.time() - start_time
            logger.info(f"验证集缓存构建完成（向量化），耗时 {elapsed:.2f}s")
            return
        
        # 回退到逐样本构建（兼容非 Store 数据）
        logger.info("回退到逐样本缓存构建...")
        self._build_cache_sequential()
        elapsed = time.time() - start_time
        logger.info(f"验证集缓存构建完成，耗时 {elapsed:.2f}s")
    
    def _try_build_cache_vectorized(self) -> bool:
        """
        向量化批量构建缓存（核心优化）。
        
        将 500K 样本 × 127 负样本的构建从 ~10 小时缩短到 ~3 分钟。
        并且直接输出列式 Tensor，避免 list(dict) 转换开销。
        
        Returns:
            True 如果成功使用向量化路径，False 需要回退到逐样本
        """
        adapter = self.adapter
        store = adapter.data_loader.store
        
        # 需要 ColumnarInteractionStore 支持
        if store is None:
            return False
        
        n = len(self.interactions)
        num_neg = self.num_negatives
        total_cand = 1 + num_neg
        
        rng = np.random.default_rng(self.seed)
        
        # ===== Step 1: 批量提取原始数据 =====
        logger.info(f"  [向量化] Step 1: 提取原始数据...")
        t0 = time.time()
        
        # 从 InteractionSlice 或 list 获取索引
        if hasattr(self.interactions, '_indices'):
            indices = self.interactions._indices
        elif hasattr(self.interactions, '__len__') and hasattr(self.interactions[0], '_idx'):
            # List of InteractionView
            indices = np.array([v._idx for v in self.interactions])
        else:
            # 普通列表 - 无法向量化
            return False
        
        user_ids = store.user_ids[indices]       # [N]
        item_ids = store.item_ids[indices]       # [N]
        clicks = store.clicks[indices]           # [N]
        likes = store.likes[indices]             # [N]
        shares = store.shares[indices]           # [N]
        follows = store.follows[indices]         # [N]
        
        # 预构建历史矩阵
        has_hist = store.has_hist and store.hist_matrix is not None
        if has_hist:
            hist_matrix = store.hist_matrix[indices]  # [N, 10]
        else:
            logger.info("  [向量化] 无预构建历史矩阵，回退到逐样本构建")
            return False
        
        logger.info(f"  [向量化] Step 1 完成: {time.time() - t0:.2f}s")
        
        # ===== Step 2: 批量 Hash 生成 =====
        logger.info(f"  [向量化] Step 2: 批量 Hash 生成...")
        t1 = time.time()
        
        hs = adapter.hash_table_size
        nih = adapter.num_item_hashes
        nuh = adapter.num_user_hashes
        hist_len = adapter.history_seq_len
        num_actions = adapter.num_actions
        
        # 用户 Hash: [N] -> [N, nuh]
        user_hashes_all = adapter._batch_generate_multi_hash(user_ids, nuh, hs)
        # 只取第一个 hash
        user_hashes_scalar = user_hashes_all[:, 0]  # [N]
        
        # 正样本 Hash: [N] -> [N, nih]
        pos_item_hashes = adapter._batch_generate_multi_hash(item_ids, nih, hs)
        
        # 预构建历史 Hash
        # hist_matrix [N, 10] -> history_post_hashes [N, hist_len, nih]
        actual_hist_len = min(10, hist_len)
        hist_items = hist_matrix[:, :actual_hist_len]  # [N, actual_hist_len]
        
        # 批量 hash: [N, actual_hist_len] -> [N, actual_hist_len, nih]
        hist_post_hashes = adapter._batch_generate_multi_hash(hist_items, nih, hs)
        
        # 处理 padding (hist_item == 0)
        padding_mask = hist_items == 0  # [N, actual_hist_len]
        hist_post_hashes[padding_mask] = 0
        
        # 只取第一个 hash
        hist_post_hashes_0 = hist_post_hashes[:, :, 0]  # [N, actual_hist_len]
        
        # 构建完整 history 数组 (pad 到 hist_len)
        if actual_hist_len < hist_len:
            hist_ids_padded = np.zeros((n, hist_len), dtype=np.int32)
            hist_ids_padded[:, :actual_hist_len] = hist_post_hashes_0
        else:
            hist_ids_padded = hist_post_hashes_0[:, :hist_len]
        
        # history_actions: [N, hist_len, num_actions]
        hist_actions = np.zeros((n, hist_len, num_actions), dtype=np.float32)
        # 非 padding 位置的 click=1
        valid_mask = ~padding_mask  # [N, actual_hist_len]
        if actual_hist_len < hist_len:
            valid_padded = np.zeros((n, hist_len), dtype=bool)
            valid_padded[:, :actual_hist_len] = valid_mask
        else:
            valid_padded = valid_mask[:, :hist_len]
        hist_actions[:, :, 0] = valid_padded.astype(np.float32)
        
        logger.info(f"  [向量化] Step 2 完成: {time.time() - t1:.2f}s")
        
        # ===== Step 3: 批量负采样 =====
        logger.info(f"  [向量化] Step 3: 批量负采样 ({n} × {num_neg})...")
        t2 = time.time()
        
        neg_items = adapter.data_loader.batch_sample_negative_items(
            user_ids, num_neg, rng
        )  # [N, num_neg]
        
        logger.info(f"  [向量化] Step 3 完成: {time.time() - t2:.2f}s")
        
        # ===== Step 4: 构建 candidate Hash =====
        logger.info(f"  [向量化] Step 4: 构建候选 Hash...")
        t3 = time.time()
        
        neg_item_hashes = adapter._batch_generate_multi_hash(neg_items, nih, hs)
        
        # candidate_hashes: [N, total_cand, nih]
        # 合并正样本(位置0) + 负样本(位置1~num_neg)
        all_cand_hashes = np.concatenate(
            [pos_item_hashes[:, None, :], neg_item_hashes], axis=1
        )  # [N, total_cand, nih]
        # 只取第一个 hash
        cand_hashes_0 = all_cand_hashes[:, :, 0]  # [N, total_cand]
        
        # labels: [N, total_cand, num_actions]
        labels = np.zeros((n, total_cand, num_actions), dtype=np.float32)
        labels[:, 0, 0] = clicks.astype(np.float32)
        labels[:, 0, 1] = likes.astype(np.float32)
        labels[:, 0, 2] = shares.astype(np.float32)
        labels[:, 0, 3] = follows.astype(np.float32)
        
        logger.info(f"  [向量化] Step 4 完成: {time.time() - t3:.2f}s")
        
        # ===== Step 5: 向量化 Shuffle =====
        logger.info(f"  [向量化] Step 5: 向量化 Shuffle...")
        t4 = time.time()
        
        # 为每个样本生成独立的 permutation
        shuffle_rng = np.random.default_rng(self.seed + 7777)
        
        for i in range(n):
            perm = shuffle_rng.permutation(total_cand)
            cand_hashes_0[i] = cand_hashes_0[i, perm]
            labels[i] = labels[i, perm]
        
        logger.info(f"  [向量化] Step 5 完成: {time.time() - t4:.2f}s")
        
        # ===== Step 6: 存储为列式 Tensor =====
        logger.info(f"  [向量化] Step 6: 存储为列式 Tensor...")
        t5 = time.time()
        
        self.data = {
            "user_ids": torch.from_numpy(user_hashes_scalar).long(),
            "history_item_ids": torch.from_numpy(hist_ids_padded).long(),
            "history_actions": torch.from_numpy(hist_actions).float(),
            "candidate_item_ids": torch.from_numpy(cand_hashes_0).long(),
            "labels": torch.from_numpy(labels).float(),
        }
        self.length = n
        
        logger.info(f"  [向量化] Step 6 完成: {time.time() - t5:.2f}s")
        return True
    
    def _build_cache_sequential(self):
        """逐样本构建缓存（回退路径）。"""
        rng = np.random.default_rng(self.seed)
        original_rng = self.adapter.rng
        
        temp_data_list = []
        
        for idx, interaction in enumerate(self.interactions):
            # 为每个样本使用确定性种子
            sample_rng = np.random.default_rng(self.seed + idx)
            self.adapter.rng = sample_rng
            
            batch = self.adapter.create_training_batch(
                [interaction],
                num_negatives=self.num_negatives,
            )
            
            sample = {
                "user_ids": torch.tensor(batch.user_hashes[0, 0], dtype=torch.long),
                "history_item_ids": torch.tensor(batch.history_post_hashes[0, :, 0], dtype=torch.long),
                "history_actions": torch.tensor(batch.history_actions[0], dtype=torch.float32),
                "candidate_item_ids": torch.tensor(batch.candidate_post_hashes[0, :, 0], dtype=torch.long),
                "labels": torch.tensor(batch.labels[0], dtype=torch.float32),
            }
            temp_data_list.append(sample)
            
            if (idx + 1) % 500 == 0:
                logger.info(f"  已处理 {idx + 1}/{len(self.interactions)} 样本")
        
        # 恢复原始 rng
        self.adapter.rng = original_rng
        
        # 转换为列式存储 (Stack)
        logger.info("  正在将列表转换为 Tensor 存储...")
        if not temp_data_list:
            self.length = 0
            self.data = {}
            return

        self.data = {
            "user_ids": torch.stack([s["user_ids"] for s in temp_data_list]),
            "history_item_ids": torch.stack([s["history_item_ids"] for s in temp_data_list]),
            "history_actions": torch.stack([s["history_actions"] for s in temp_data_list]),
            "candidate_item_ids": torch.stack([s["candidate_item_ids"] for s in temp_data_list]),
            "labels": torch.stack([s["labels"] for s in temp_data_list]),
        }
        self.length = len(temp_data_list)
    
    def _save_cache(self):
        """保存缓存到磁盘（torch.save）。"""
        try:
            os.makedirs(self.cache_path.parent, exist_ok=True)
            
            save_content = {
                "meta": {
                    "version": self.CACHE_VERSION,
                    "cache_key": self._get_cache_key(),
                    "num_samples": self.length,
                    "num_negatives": self.num_negatives,
                    "seed": self.seed,
                },
                "data": self.data
            }
            
            # 使用 torch.save 保存，速度远快于 pickle 序列化 list[dict]
            torch.save(save_content, self.cache_path)
            
            cache_size_mb = self.cache_path.stat().st_size / (1024 * 1024)
            logger.info(f"验证集缓存已保存: {self.cache_path} ({cache_size_mb:.1f} MB)")
            
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 动态从列式 Tensor 中切片，无需存储数十万个 dict 对象
        return {
            "user_ids": self.data["user_ids"][idx],
            "history_item_ids": self.data["history_item_ids"][idx],
            "history_actions": self.data["history_actions"][idx],
            "candidate_item_ids": self.data["candidate_item_ids"][idx],
            "labels": self.data["labels"][idx],
        }


def get_eval_cache_path(
    cache_dir: str,
    scenario: str,
    split: str,
    num_samples: int,
    num_negatives: int,
    seed: int,
) -> Path:
    """
    获取评估集缓存路径。
    注意：文件名从 .pkl 改为 .pt 以区分新旧格式
    """
    cache_name = f"{scenario}_{split}_n{num_samples}_neg{num_negatives}_seed{seed}_v4.pt"
    return Path(cache_dir) / ".eval_cache" / cache_name
