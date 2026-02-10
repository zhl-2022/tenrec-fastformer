# MIT License - see LICENSE for details

"""
Tenrec 到 Phoenix 格式适配器。

将 Tenrec 数据转换为 Phoenix 系统的 RecsysBatch 和 RecsysEmbeddings 格式。
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# 兼容直接运行和模块导入
try:
    from .data_loader import TenrecDataLoader, TenrecInteraction, TenrecUserHistory
except ImportError:
    from tenrec_adapter.data_loader import TenrecDataLoader, TenrecInteraction, TenrecUserHistory

logger = logging.getLogger(__name__)


# Tenrec Action 到 Phoenix Action 的映射
# Phoenix 有 19 种 actions，Tenrec 有 4 种
TENREC_TO_PHOENIX_ACTION_MAP = {
    "click": 4,    # click_score
    "like": 0,     # favorite_score
    "share": 7,    # share_score
    "follow": 13,  # follow_author_score
}

# Phoenix 的所有 Action 名称
PHOENIX_ACTIONS = [
    "favorite_score",      # 0 - like
    "reply_score",         # 1
    "repost_score",        # 2
    "photo_expand_score",  # 3
    "click_score",         # 4 - click
    "profile_click_score", # 5
    "vqv_score",           # 6
    "share_score",         # 7 - share
    "share_via_dm_score",  # 8
    "share_via_copy_link_score",  # 9
    "dwell_score",         # 10
    "quote_score",         # 11
    "quoted_click_score",  # 12
    "follow_author_score", # 13 - follow
    "not_interested_score",# 14
    "block_author_score",  # 15
    "mute_author_score",   # 16
    "report_score",        # 17
    "dwell_time",          # 18
]


@dataclass
class PhoenixBatchData:
    """
    Phoenix 格式的批次数据。
    
    对应 phoenix/recsys_model.py 中的 RecsysBatch 和 RecsysEmbeddings。
    """
    
    # RecsysBatch 字段
    user_hashes: np.ndarray           # [B, num_user_hashes]
    history_post_hashes: np.ndarray   # [B, S, num_item_hashes]
    history_author_hashes: np.ndarray # [B, S, num_author_hashes]
    history_actions: np.ndarray       # [B, S, num_actions]
    history_product_surface: np.ndarray  # [B, S]
    candidate_post_hashes: np.ndarray    # [B, C, num_item_hashes]
    candidate_author_hashes: np.ndarray  # [B, C, num_author_hashes]
    candidate_product_surface: np.ndarray  # [B, C]
    
    # 标签（用于训练）
    labels: Optional[np.ndarray] = None  # [B, C, num_actions] 或 [B, C]
    
    @property
    def batch_size(self) -> int:
        return self.user_hashes.shape[0]
    
    @property
    def history_len(self) -> int:
        return self.history_post_hashes.shape[1]
    
    @property
    def num_candidates(self) -> int:
        return self.candidate_post_hashes.shape[1]


class TenrecToPhoenixAdapter:
    """
    将 Tenrec 数据转换为 Phoenix 推荐系统格式。
    
    主要功能：
    1. ID 到多值 Hash 的转换
    2. Action 类型映射（4种 → 19种）
    3. 用户历史序列构建
    4. 批次数据生成
    """
    
    def __init__(
        self,
        data_loader: TenrecDataLoader,
        num_user_hashes: int = 2,
        num_item_hashes: int = 2,
        num_author_hashes: int = 2,
        history_seq_len: int = 128,
        candidate_seq_len: int = 32,
        num_actions: int = 4,  # Tenrec 只有 4 种 action
        hash_table_size: int = 100000,
        product_surface_vocab_size: int = 16,
        seed: int = 42,
    ):
        """
        初始化适配器。
        
        Args:
            data_loader: Tenrec 数据加载器
            num_user_hashes: 用户 Hash 数量
            num_item_hashes: Item Hash 数量
            num_author_hashes: Author Hash 数量
            history_seq_len: 历史序列最大长度
            candidate_seq_len: 候选序列长度
            num_actions: Phoenix Action 数量
            hash_table_size: Hash 表大小
            product_surface_vocab_size: 产品表面词表大小
            seed: 随机种子
        """
        self.data_loader = data_loader
        self.num_user_hashes = num_user_hashes
        self.num_item_hashes = num_item_hashes
        self.num_author_hashes = num_author_hashes
        self.history_seq_len = history_seq_len
        self.candidate_seq_len = candidate_seq_len
        self.num_actions = num_actions
        self.hash_table_size = hash_table_size
        self.product_surface_vocab_size = product_surface_vocab_size
        
        self.rng = np.random.default_rng(seed)
        
    def _generate_multi_hash(
        self,
        id_value: int,
        num_hashes: int,
        table_size: int,
    ) -> np.ndarray:
        """
        将单个 ID 转换为多个 Hash 值。
        
        使用不同的种子生成多个独立的 hash 值，
        这与 Phoenix 的 hash-based embedding 设计一致。
        
        Args:
            id_value: 原始 ID
            num_hashes: Hash 数量
            table_size: Hash 表大小
            
        Returns:
            [num_hashes] 的 hash 值数组
        """
        hashes = np.zeros(num_hashes, dtype=np.int32)
        for i in range(num_hashes):
            # 使用不同的乘数来生成不同的 hash
            hashes[i] = ((id_value * (i + 1) * 2654435761) % table_size) + 1  # +1 避免 0（padding）
        return hashes
    
    def _batch_generate_multi_hash(
        self,
        id_values: np.ndarray,
        num_hashes: int,
        table_size: int,
    ) -> np.ndarray:
        """
        向量化批量 Hash 生成。
        
        将 N 个 ID 一次性转换为 Hash 值，使用 numpy 广播而非 Python 循环。
        对于 500K 个 ID，比逐个调用 _generate_multi_hash 快 200x+。
        
        Args:
            id_values: 原始 ID 数组 [N] 或 [N, M]（任意形状）
            num_hashes: 每个 ID 的 Hash 数量
            table_size: Hash 表大小
            
        Returns:
            Hash 值数组，形状为 input_shape + (num_hashes,)
        """
        original_shape = id_values.shape
        ids_flat = id_values.astype(np.int64).ravel()  # 展平 + int64 防止溢出
        
        # 构建乘数数组 [num_hashes]
        multipliers = np.arange(1, num_hashes + 1, dtype=np.int64) * 2654435761
        
        # 广播计算: [N, 1] * [1, num_hashes] -> [N, num_hashes]
        hashes = ((ids_flat[:, None] * multipliers[None, :]) % table_size + 1).astype(np.int32)
        
        # 恢复原始形状 + hash 维度
        return hashes.reshape(original_shape + (num_hashes,))
    
    def _map_tenrec_actions_to_phoenix(
        self,
        interaction: TenrecInteraction,
    ) -> np.ndarray:
        """
        将 Tenrec 的 4 种 action 直接映射。
        
        Args:
            interaction: Tenrec 交互记录
            
        Returns:
            [num_actions] 的 action 向量 (4 维)
        """
        actions = np.zeros(self.num_actions, dtype=np.float32)
        
        # 简化映射：click=0, like=1, share=2, follow=3
        actions[0] = float(interaction.click)
        actions[1] = float(interaction.like)
        actions[2] = float(interaction.share)
        actions[3] = float(interaction.follow)
            
        return actions
    
    def _build_history_from_user(
        self,
        user_history: TenrecUserHistory,
        before_timestamp: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        从用户历史构建 Phoenix 格式的历史序列。
        
        Args:
            user_history: 用户历史记录
            before_timestamp: 只使用该时间戳之前的交互（用于时间敏感的评估）
            
        Returns:
            (history_post_hashes, history_author_hashes, history_actions, history_product_surface)
        """
        # 过滤交互
        interactions = user_history.interactions
        if before_timestamp is not None:
            interactions = [i for i in interactions if i.timestamp < before_timestamp]
        
        # 按时间倒序排列，取最近的
        interactions = sorted(interactions, key=lambda x: x.timestamp, reverse=True)
        interactions = interactions[:self.history_seq_len]
        
        # 初始化数组
        history_post_hashes = np.zeros((self.history_seq_len, self.num_item_hashes), dtype=np.int32)
        history_author_hashes = np.zeros((self.history_seq_len, self.num_author_hashes), dtype=np.int32)
        history_actions = np.zeros((self.history_seq_len, self.num_actions), dtype=np.float32)
        history_product_surface = np.zeros(self.history_seq_len, dtype=np.int32)
        
        for i, inter in enumerate(interactions):
            history_post_hashes[i] = self._generate_multi_hash(
                inter.item_id, self.num_item_hashes, self.hash_table_size
            )
            # Tenrec 没有 author_id，使用 item_id 的变体作为代理
            history_author_hashes[i] = self._generate_multi_hash(
                inter.item_id * 31, self.num_author_hashes, self.hash_table_size
            )
            history_actions[i] = self._map_tenrec_actions_to_phoenix(inter)
            # 产品表面：随机分配（Tenrec 没有此信息）
            history_product_surface[i] = inter.item_id % self.product_surface_vocab_size
            
        return history_post_hashes, history_author_hashes, history_actions, history_product_surface
    
    def create_training_batch(
        self,
        interactions: List[TenrecInteraction],
        num_negatives: int = 4,
    ) -> PhoenixBatchData:
        """
        从交互列表创建训练批次。
        
        每个正样本配对 num_negatives 个负样本。
        
        Args:
            interactions: 交互列表
            num_negatives: 每个正样本的负样本数量
            
        Returns:
            PhoenixBatchData 训练批次
        """
        batch_size = len(interactions)
        total_candidates = 1 + num_negatives  # 1 个正样本 + num_negatives 个负样本
        
        # 初始化数组
        user_hashes = np.zeros((batch_size, self.num_user_hashes), dtype=np.int32)
        history_post_hashes = np.zeros(
            (batch_size, self.history_seq_len, self.num_item_hashes), dtype=np.int32
        )
        history_author_hashes = np.zeros(
            (batch_size, self.history_seq_len, self.num_author_hashes), dtype=np.int32
        )
        history_actions = np.zeros(
            (batch_size, self.history_seq_len, self.num_actions), dtype=np.float32
        )
        history_product_surface = np.zeros((batch_size, self.history_seq_len), dtype=np.int32)
        
        candidate_post_hashes = np.zeros(
            (batch_size, total_candidates, self.num_item_hashes), dtype=np.int32
        )
        candidate_author_hashes = np.zeros(
            (batch_size, total_candidates, self.num_author_hashes), dtype=np.int32
        )
        candidate_product_surface = np.zeros((batch_size, total_candidates), dtype=np.int32)
        
        # 标签：多目标预测
        labels = np.zeros((batch_size, total_candidates, self.num_actions), dtype=np.float32)
        
        for i, inter in enumerate(interactions):
            # 用户 Hash
            user_hashes[i] = self._generate_multi_hash(
                inter.user_id, self.num_user_hashes, self.hash_table_size
            )
            
            # 构建用户历史
            if inter.hist_items:
                # ctr_data_1M 格式：使用预构建历史（hist_1~hist_10）
                for hi, hist_item_id in enumerate(inter.hist_items):
                    if hi >= self.history_seq_len or hist_item_id == 0:
                        break
                    history_post_hashes[i, hi] = self._generate_multi_hash(
                        hist_item_id, self.num_item_hashes, self.hash_table_size
                    )
                    history_author_hashes[i, hi] = self._generate_multi_hash(
                        hist_item_id * 31, self.num_author_hashes, self.hash_table_size
                    )
                    # 预构建历史无 action 信息，默认 click=1
                    history_actions[i, hi, 0] = 1.0
                    history_product_surface[i, hi] = hist_item_id % self.product_surface_vocab_size
            else:
                # QB-video / QK-video 格式：从用户历史动态构建
                user_history = self.data_loader.get_user_history(inter.user_id)
                if user_history:
                    (
                        history_post_hashes[i],
                        history_author_hashes[i],
                        history_actions[i],
                        history_product_surface[i],
                    ) = self._build_history_from_user(user_history, before_timestamp=inter.timestamp)
            
            # 正样本（位置 0）
            candidate_post_hashes[i, 0] = self._generate_multi_hash(
                inter.item_id, self.num_item_hashes, self.hash_table_size
            )
            candidate_author_hashes[i, 0] = self._generate_multi_hash(
                inter.item_id * 31, self.num_author_hashes, self.hash_table_size
            )
            candidate_product_surface[i, 0] = inter.item_id % self.product_surface_vocab_size
            labels[i, 0] = self._map_tenrec_actions_to_phoenix(inter)
            
            # 负样本
            neg_items = self.data_loader.sample_negative_items(
                inter.user_id, num_negatives, self.rng
            )
            for j, neg_item_id in enumerate(neg_items):
                candidate_post_hashes[i, j + 1] = self._generate_multi_hash(
                    neg_item_id, self.num_item_hashes, self.hash_table_size
                )
                candidate_author_hashes[i, j + 1] = self._generate_multi_hash(
                    neg_item_id * 31, self.num_author_hashes, self.hash_table_size
                )
                candidate_product_surface[i, j + 1] = neg_item_id % self.product_surface_vocab_size
                # 负样本标签全为 0
            
            # 打乱候选项顺序，消除正样本位置固定导致的数据泄露
            shuffle_idx = self.rng.permutation(total_candidates)
            candidate_post_hashes[i] = candidate_post_hashes[i, shuffle_idx]
            candidate_author_hashes[i] = candidate_author_hashes[i, shuffle_idx]
            candidate_product_surface[i] = candidate_product_surface[i, shuffle_idx]
            labels[i] = labels[i, shuffle_idx]
        
        return PhoenixBatchData(
            user_hashes=user_hashes,
            history_post_hashes=history_post_hashes,
            history_author_hashes=history_author_hashes,
            history_actions=history_actions,
            history_product_surface=history_product_surface,
            candidate_post_hashes=candidate_post_hashes,
            candidate_author_hashes=candidate_author_hashes,
            candidate_product_surface=candidate_product_surface,
            labels=labels,
        )
    
    def batch_generator(
        self,
        interactions: List[TenrecInteraction],
        batch_size: int = 32,
        num_negatives: int = 4,
        shuffle: bool = True,
    ):
        """
        生成训练批次的生成器。
        
        Args:
            interactions: 交互列表
            batch_size: 批次大小
            num_negatives: 负样本数量
            shuffle: 是否打乱
            
        Yields:
            PhoenixBatchData 批次
        """
        indices = np.arange(len(interactions))
        if shuffle:
            self.rng.shuffle(indices)
        
        for start_idx in range(0, len(interactions), batch_size):
            end_idx = min(start_idx + batch_size, len(interactions))
            batch_indices = indices[start_idx:end_idx]
            batch_interactions = [interactions[i] for i in batch_indices]
            
            yield self.create_training_batch(batch_interactions, num_negatives)
