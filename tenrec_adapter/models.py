# MIT License - see LICENSE for details

"""
推荐系统 PyTorch 模型。

用于在 Tenrec 数据集上训练的 Two-Tower 推荐模型。
支持 MLU/CUDA/CPU 训练。
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入 Fastformer（MLU 兼容的高效 Transformer）
from tenrec_adapter.fastformer import FastformerEncoder


class PositionalEncoding(nn.Module):
    """位置编码。"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D] 输入张量
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class UserTower(nn.Module):
    """
    用户塔。

    编码用户特征和历史行为序列。
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        history_len: int = 128,
        num_actions: int = 4,  # Tenrec 4 种 action
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # 用户 Embedding
        self.user_embedding = nn.Embedding(num_users + 1, embed_dim, padding_idx=0)

        # Side Info Embedding (新增)
        # gender: 0=unknown, 1=male, 2=female (假设 3 种)
        self.gender_embedding = nn.Embedding(3, embed_dim, padding_idx=0)
        # age: 0=unknown, 1-8=age_groups (假设 9 种)
        self.age_embedding = nn.Embedding(10, embed_dim, padding_idx=0)

        # Item Embedding（用于历史序列）
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)

        # Action Embedding
        self.action_projection = nn.Linear(num_actions, embed_dim)

        # 位置编码
        self.pos_encoding = PositionalEncoding(embed_dim * 2, history_len, dropout)

        # 使用 Fastformer 替代 PyTorch TransformerEncoder（MLU 兼容性更好）
        self.transformer = FastformerEncoder(
            d_model=embed_dim * 2,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="relu",  # 用 relu 代替 gelu，更稳定
        )

        # 输出投影
        self.output_projection = nn.Linear(embed_dim * 2, embed_dim)

        self._init_weights()

    def _init_weights(self):
        """初始化权重（保护 padding_idx 的零向量）。"""
        # 跳过 padding_idx=0，只初始化非 padding 的 embedding
        with torch.no_grad():
            nn.init.xavier_uniform_(self.user_embedding.weight[1:])
            nn.init.xavier_uniform_(self.item_embedding.weight[1:])
            nn.init.xavier_uniform_(self.gender_embedding.weight[1:])
            nn.init.xavier_uniform_(self.age_embedding.weight[1:])
            # 确保 padding 位置为零
            self.user_embedding.weight[0].zero_()
            self.item_embedding.weight[0].zero_()
            self.gender_embedding.weight[0].zero_()
            self.age_embedding.weight[0].zero_()
        nn.init.xavier_uniform_(self.action_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)

    def forward(
        self,
        user_ids: torch.Tensor,           # [B]
        history_item_ids: torch.Tensor,   # [B, S]
        history_actions: torch.Tensor,    # [B, S, num_actions]
        user_gender: Optional[torch.Tensor] = None, # [B]
        user_age: Optional[torch.Tensor] = None,    # [B]
    ) -> torch.Tensor:
        """
        前向传播。

        Returns:
            user_repr: [B, D] 用户表示
        """
        batch_size = user_ids.size(0)

        # 用户 Embedding
        user_emb = self.user_embedding(user_ids)  # [B, D]

        # Side Info Embedding (Add)
        if user_gender is not None:
             user_emb = user_emb + self.gender_embedding(user_gender)
        if user_age is not None:
             user_emb = user_emb + self.age_embedding(user_age)

        # 历史 Item Embedding
        history_emb = self.item_embedding(history_item_ids)  # [B, S, D]

        # Action Embedding（添加归一化和 clamp 保护）
        action_emb = self.action_projection(history_actions)  # [B, S, D]
        # 使用 LayerNorm 归一化，防止数值爆炸（这是 NaN 的关键修复点）
        action_emb = F.layer_norm(action_emb, [action_emb.size(-1)], eps=1e-6)
        action_emb = torch.clamp(action_emb, -10.0, 10.0)  # 防止极端值

        # 拼接 Item 和 Action
        history_repr = torch.cat([history_emb, action_emb], dim=-1)  # [B, S, 2D]

        # 位置编码
        history_repr = self.pos_encoding(history_repr)

        # 创建 padding mask（MLU 安全版本）
        padding_mask = (history_item_ids == 0)  # [B, S], True = 需要 mask

        # 防止全 padding：如果某行全是 padding，强制保留第一个位置
        all_padding = padding_mask.all(dim=1, keepdim=True)  # [B, 1]
        if all_padding.any():
            # 对全 padding 的行，取消第一个位置的 mask
            first_col_mask = torch.zeros_like(padding_mask)
            first_col_mask[:, 0] = True
            padding_mask = padding_mask & ~(all_padding & first_col_mask)

        # Fastformer 编码（安全 mask 处理）
        history_encoded = self.transformer(
            history_repr,
            src_key_padding_mask=padding_mask,
        )  # [B, S, 2D]

        # NaN 保护：在 Transformer 输出后立即处理
        history_encoded = torch.nan_to_num(history_encoded, nan=0.0, posinf=1.0, neginf=-1.0)

        # 平均池化（忽略 padding）
        mask = (~padding_mask).unsqueeze(-1).float()  # [B, S, 1]
        mask_sum = mask.sum(dim=1).clamp(min=1.0)  # 防止除以零
        history_pooled = (history_encoded * mask).sum(dim=1) / mask_sum  # [B, 2D]

        # 投影到输出维度
        history_repr = self.output_projection(history_pooled)  # [B, D]

        # 合并用户和历史表示
        user_repr = user_emb + history_repr  # [B, D]

        # NaN 检查与保护
        if torch.isnan(user_repr).any() or torch.isinf(user_repr).any():
             # 使用 hook 或 print 调试，这里先做安全替换
             user_repr = torch.nan_to_num(user_repr, nan=0.0, posinf=1.0, neginf=-1.0)

        # L2 归一化（增加 epsilon 防止数值不稳定）
        user_repr = F.normalize(user_repr, p=2, dim=-1, eps=1e-6)

        return user_repr


class ItemTower(nn.Module):
    """
    Item 塔。

    编码候选 Item 特征。
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_categories: int = 0, # 支持类别特征
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_categories = num_categories

        # Item Embedding
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)

        # Category Embedding
        if num_categories > 0:
            self.category_embedding = nn.Embedding(num_categories + 1, embed_dim, padding_idx=0)
        else:
            self.category_embedding = None

        # MLP 层
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重（保护 padding_idx 的零向量）。"""
        # 跳过 padding_idx=0，只初始化非 padding 的 embedding
        with torch.no_grad():
            nn.init.xavier_uniform_(self.item_embedding.weight[1:])
            self.item_embedding.weight[0].zero_()
            if self.category_embedding is not None:
                nn.init.xavier_uniform_(self.category_embedding.weight[1:])
                self.category_embedding.weight[0].zero_()

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, item_ids: torch.Tensor, categories: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播。

        Args:
            item_ids: [B, C] 或 [B] Item IDs
            categories: [B, C] 或 [B] Item Categories (Optional)

        Returns:
            item_repr: [B, C, D] 或 [B, D] Item 表示
        """
        # Item Embedding
        item_emb = self.item_embedding(item_ids)  # [B, ?, D]

        # Category Embedding
        if self.category_embedding is not None and categories is not None:
            cat_emb = self.category_embedding(categories) # [B, ?, D]
            item_emb = item_emb + cat_emb

        # MLP
        item_repr = self.mlp(item_emb)  # [B, ?, D]

        # Residual
        item_repr = item_emb + item_repr

        # L2 归一化（增加 epsilon 防止数值不稳定）
        item_repr = F.normalize(item_repr, p=2, dim=-1, eps=1e-6)

        return item_repr


class TwoTowerModel(nn.Module):
    """
    Two-Tower 推荐模型。

    用于检索和排序任务。
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        history_len: int = 128,
        num_actions: int = 4,  # Tenrec 4 种 action: click, like, share, follow
        dropout: float = 0.1,
        temperature: float = 1.0,  # 默认 1.0 更稳定，0.5 可能导致 exp 后数值过大
        num_categories: int = 0, # 支持类别特征
    ):
        super().__init__()

        self.temperature = temperature

        # User Tower
        self.user_tower = UserTower(
            num_users=num_users,
            num_items=num_items,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            history_len=history_len,
            num_actions=num_actions,
            dropout=dropout,
        )

        # Item Tower
        self.item_tower = ItemTower(
            num_items=num_items,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_categories=num_categories,
        )

        # 多任务预测层：从 combined representation 预测 action logits
        # 修复：使用统一的 unembedding 层，确保不同候选 item 有差异化输出
        self.output_norm = nn.LayerNorm(embed_dim)
        self.unembedding = nn.Linear(embed_dim, num_actions)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        Args:
            batch: 包含以下键的字典:
                - user_ids: [B]
                - history_item_ids: [B, S]
                - history_actions: [B, S, num_actions]
                - candidate_item_ids: [B, C]

        Returns:
            包含 logits 的字典
        """
        # 用户编码
        user_repr = self.user_tower(
            user_ids=batch["user_ids"],
            history_item_ids=batch["history_item_ids"],
            history_actions=batch["history_actions"],
            user_gender=batch.get("user_gender"),
            user_age=batch.get("user_age"),
        )  # [B, D]

        # 候选 Item 编码
        candidate_categories = batch.get("candidate_categories")
        candidate_repr = self.item_tower(
            item_ids=batch["candidate_item_ids"],
            categories=candidate_categories,
        )  # [B, C, D]

        # ===== L2 归一化（与 Phoenix 原版保持一致）=====
        # 归一化后点积等价于余弦相似度，支持 FAISS cosine 检索
        EPS = 1e-12
        user_repr_norm = user_repr / (user_repr.norm(dim=-1, keepdim=True) + EPS)  # [B, D]
        candidate_repr_norm = candidate_repr / (candidate_repr.norm(dim=-1, keepdim=True) + EPS)  # [B, C, D]

        # 计算得分（点积 / 温度）
        user_repr_expanded = user_repr_norm.unsqueeze(1)  # [B, D] -> [B, 1, D]

        # 点积得分（归一化后范围为 [-1, 1]）
        logits = torch.sum(user_repr_expanded * candidate_repr_norm, dim=-1)  # [B, C]

        # 数值保护：限制 logits 范围
        logits = torch.clamp(logits, min=-100.0, max=100.0)

        scores = logits / self.temperature  # [B, C] - 这是主要的排序分数

        # 多任务预测：从 combined representation 生成 action logits
        # 修复：使用 unembedding 层，确保不同候选 item 有差异化输出
        combined_repr = user_repr_expanded * candidate_repr  # [B, C, D]
        combined_repr = self.output_norm(combined_repr)  # LayerNorm 归一化
        action_logits = self.unembedding(combined_repr)  # [B, C, num_actions]

        # click 是第一个 action (idx=0)
        # 这样 action_logits[:, :, 0] 不仅有 unembedding 的输出，还有点积 scores 的信息
        action_logits[:, :, 0] = action_logits[:, :, 0] + scores

        # 添加数值保护
        action_logits = torch.clamp(action_logits, min=-100.0, max=100.0)

        return {
            "logits": action_logits,
            "scores": scores,
            "user_repr": user_repr,
            "candidate_repr": candidate_repr,
        }

    def encode_user(
        self,
        user_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
        history_actions: torch.Tensor,
    ) -> torch.Tensor:
        """编码用户（用于检索）。"""
        return self.user_tower(user_ids, history_item_ids, history_actions)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """编码 Items（用于构建索引）。"""
        return self.item_tower(item_ids)


class TenrecDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset 包装器。

    将 PhoenixBatchData 转换为 PyTorch 格式。
    """

    def __init__(
        self,
        user_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
        history_actions: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        labels: torch.Tensor,
    ):
        self.user_ids = user_ids
        self.history_item_ids = history_item_ids
        self.history_actions = history_actions
        self.candidate_item_ids = candidate_item_ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "user_ids": self.user_ids[idx],
            "history_item_ids": self.history_item_ids[idx],
            "history_actions": self.history_actions[idx],
            "candidate_item_ids": self.candidate_item_ids[idx],
            "labels": self.labels[idx],
        }
