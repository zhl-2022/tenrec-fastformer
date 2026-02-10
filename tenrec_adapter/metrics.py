# MIT License - see LICENSE for details

"""
推荐系统评测指标。

实现 Tenrec 和 Phoenix 推荐系统的标准评测指标：
- AUC: CTR 预测的主要指标
- NDCG@K: 排序质量指标
- HitRate@K: 召回指标
- MRR: 平均倒数排名
- Recall@K: 召回率
- Precision@K: 精确率
"""

import numpy as np
from typing import List, Optional, Tuple, Union


def compute_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """
    计算 AUC (Area Under the ROC Curve)。
    
    用于 CTR 预测任务的评估。
    
    Args:
        y_true: 真实标签 [N] 或 [N, num_actions]，二值
        y_score: 预测分数 [N] 或 [N, num_actions]，连续值
        
    Returns:
        AUC 分数 (0-1)
    """
    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()
    
    # 检查是否有正负样本
    if len(np.unique(y_true)) < 2:
        return 0.5  # 只有一类样本，返回 0.5
    
    # 计算 AUC（基于 Mann-Whitney U 统计量，排序法）
    # 排序法内存复杂度 O(n)，时间复杂度 O(n·log(n))
    # 相比矩阵法 O(n_pos × n_neg)，避免了全量数据下的内存爆炸
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]
    
    if len(pos_indices) == 0 or len(neg_indices) == 0:
        return 0.5
    
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    
    # 排序法：合并所有分数并排序，利用排名计算 U 统计量
    # AUC = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    # 其中 rank_sum_pos 是正样本在所有样本中的排名之和（1-indexed）
    ordered_indices = np.argsort(y_score)
    # 构建排名数组（处理 tie 用平均排名）
    n_total = len(y_score)
    ranks = np.empty(n_total, dtype=np.float64)
    i = 0
    while i < n_total:
        # 找到相同分数的区间
        j = i + 1
        while j < n_total and y_score[ordered_indices[j]] == y_score[ordered_indices[i]]:
            j += 1
        # 相同分数的样本赋予平均排名（1-indexed）
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered_indices[k]] = avg_rank
        i = j
    
    # 正样本的排名之和
    rank_sum_pos = np.sum(ranks[pos_indices])
    
    # Mann-Whitney U 统计量
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    
    return float(np.clip(auc, 0.0, 1.0))


def compute_ndcg_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 10,
) -> float:
    """
    计算 NDCG@K (Normalized Discounted Cumulative Gain)。
    
    用于评估排序质量，考虑位置折扣。
    
    Args:
        y_true: 真实相关性分数 [N] 或 [B, N]
        y_score: 预测分数 [N] 或 [B, N]
        k: 截断位置
        
    Returns:
        NDCG@K 分数 (0-1)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # 处理单样本情况
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_score = y_score.reshape(1, -1)
    
    batch_size, n_items = y_true.shape
    k = min(k, n_items)
    
    ndcg_scores = []
    
    for i in range(batch_size):
        # 按预测分数排序
        ranked_indices = np.argsort(-y_score[i])[:k]
        ranked_relevance = y_true[i][ranked_indices]
        
        # 计算 DCG
        positions = np.arange(1, k + 1)
        discounts = 1.0 / np.log2(positions + 1)
        dcg = np.sum(ranked_relevance * discounts)
        
        # 计算理想 DCG（按真实相关性排序）
        ideal_relevance = np.sort(y_true[i])[::-1][:k]
        idcg = np.sum(ideal_relevance * discounts)
        
        # 计算 NDCG
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)
    
    return float(np.mean(ndcg_scores))


def compute_hit_rate_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 10,
) -> float:
    """
    计算 HitRate@K (Hit Rate / Recall@K for single positive)。
    
    评估 top-K 推荐列表中是否包含正样本。
    适用于每个样本只有一个正确答案的场景。
    
    Args:
        y_true: 真实标签 [B, N]，每行只有一个 1
        y_score: 预测分数 [B, N]
        k: 截断位置
        
    Returns:
        HitRate@K (0-1)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_score = y_score.reshape(1, -1)
    
    batch_size, n_items = y_true.shape
    k = min(k, n_items)
    
    hits = 0
    for i in range(batch_size):
        # 获取 top-K 预测索引
        top_k_indices = np.argsort(-y_score[i])[:k]
        # 检查是否命中正样本
        if np.any(y_true[i][top_k_indices] > 0):
            hits += 1
    
    return hits / batch_size


def compute_mrr(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """
    计算 MRR (Mean Reciprocal Rank)。
    
    评估第一个正确答案的排名位置。
    
    Args:
        y_true: 真实标签 [B, N]
        y_score: 预测分数 [B, N]
        
    Returns:
        MRR 分数 (0-1)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_score = y_score.reshape(1, -1)
    
    batch_size = y_true.shape[0]
    
    reciprocal_ranks = []
    for i in range(batch_size):
        # 按预测分数排序
        ranked_indices = np.argsort(-y_score[i])
        ranked_labels = y_true[i][ranked_indices]
        
        # 找到第一个正样本的位置
        positive_positions = np.where(ranked_labels > 0)[0]
        if len(positive_positions) > 0:
            first_positive_rank = positive_positions[0] + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / first_positive_rank)
        else:
            reciprocal_ranks.append(0.0)
    
    return float(np.mean(reciprocal_ranks))


def compute_recall_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 10,
) -> float:
    """
    计算 Recall@K。
    
    评估 top-K 推荐列表中正样本的比例。
    
    Args:
        y_true: 真实标签 [B, N]，可以有多个 1
        y_score: 预测分数 [B, N]
        k: 截断位置
        
    Returns:
        Recall@K (0-1)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_score = y_score.reshape(1, -1)
    
    batch_size, n_items = y_true.shape
    k = min(k, n_items)
    
    recalls = []
    for i in range(batch_size):
        n_positives = np.sum(y_true[i] > 0)
        if n_positives == 0:
            recalls.append(0.0)
            continue
            
        top_k_indices = np.argsort(-y_score[i])[:k]
        n_hits = np.sum(y_true[i][top_k_indices] > 0)
        recalls.append(n_hits / n_positives)
    
    return float(np.mean(recalls))


def compute_precision_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 10,
) -> float:
    """
    计算 Precision@K。
    
    评估 top-K 推荐列表中的正确率。
    
    Args:
        y_true: 真实标签 [B, N]
        y_score: 预测分数 [B, N]
        k: 截断位置
        
    Returns:
        Precision@K (0-1)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_score = y_score.reshape(1, -1)
    
    batch_size, n_items = y_true.shape
    k = min(k, n_items)
    
    precisions = []
    for i in range(batch_size):
        top_k_indices = np.argsort(-y_score[i])[:k]
        n_hits = np.sum(y_true[i][top_k_indices] > 0)
        precisions.append(n_hits / k)
    
    return float(np.mean(precisions))


class MetricsCalculator:
    """
    评测指标计算器。
    
    支持累积计算和批次更新。
    """
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """
        初始化计算器。
        
        Args:
            k_values: 用于 @K 指标的 K 值列表
        """
        self.k_values = k_values
        self.reset()
    
    def reset(self):
        """重置累积数据。"""
        self.all_y_true = []
        self.all_y_score = []
    
    def update(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
    ):
        """
        添加一个批次的数据。
        
        Args:
            y_true: 真实标签
            y_score: 预测分数
        """
        self.all_y_true.append(np.asarray(y_true))
        self.all_y_score.append(np.asarray(y_score))
    
    def compute_all(self) -> dict:
        """
        计算所有指标。
        
        Returns:
            包含所有指标的字典
        """
        y_true = np.concatenate(self.all_y_true, axis=0)
        y_score = np.concatenate(self.all_y_score, axis=0)
        
        results = {
            "auc": compute_auc(y_true, y_score),
            "mrr": compute_mrr(y_true, y_score),
        }
        
        for k in self.k_values:
            results[f"ndcg@{k}"] = compute_ndcg_at_k(y_true, y_score, k)
            results[f"hit_rate@{k}"] = compute_hit_rate_at_k(y_true, y_score, k)
            results[f"recall@{k}"] = compute_recall_at_k(y_true, y_score, k)
            results[f"precision@{k}"] = compute_precision_at_k(y_true, y_score, k)
        
        return results
    
    def format_results(self, results: dict) -> str:
        """
        格式化输出结果。
        
        Args:
            results: 指标字典
            
        Returns:
            格式化的字符串
        """
        lines = ["=" * 50, "评测结果", "=" * 50]
        
        # 基础指标
        lines.append(f"AUC: {results['auc']:.4f}")
        lines.append(f"MRR: {results['mrr']:.4f}")
        
        # @K 指标
        for k in self.k_values:
            lines.append(f"\n--- @{k} ---")
            lines.append(f"NDCG@{k}: {results[f'ndcg@{k}']:.4f}")
            lines.append(f"HitRate@{k}: {results[f'hit_rate@{k}']:.4f}")
            lines.append(f"Recall@{k}: {results[f'recall@{k}']:.4f}")
            lines.append(f"Precision@{k}: {results[f'precision@{k}']:.4f}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
