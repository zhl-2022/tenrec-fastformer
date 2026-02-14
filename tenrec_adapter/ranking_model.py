# MIT License - see LICENSE for details

"""
精排模型 (Ranking Model)。

实现与 Phoenix PhoenixModel 一致的精排架构：
- 将 User + History + Candidate 拼接后通过 Transformer 联合编码
- Cross-Attention 使 Candidate 能够关注 User 和 History
- 输出 19 种 Action 的预测 logits

用法:
    from tenrec_adapter.ranking_model import RankingModel

    model = RankingModel(num_users=10000, num_items=50000)
    outputs = model(batch)  # {"logits": [B, C, 19]}
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SegmentEmbedding(nn.Module):
    """
    分段 Embedding。

    区分 User、History、Candidate 三个部分。
    """

    def __init__(self, embed_dim: int, num_segments: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(num_segments, embed_dim)

    def forward(self, segment_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(segment_ids)


class RankingModel(nn.Module):
    """
    精排模型。

    与 Phoenix PhoenixModel 一致的架构：
    - 拼接 [User] + [History Seq] + [Candidate Seq]
    - 通过 Transformer 进行联合编码
    - Candidate 位置输出 19 种 Action 的 logits

    这与 TwoTowerModel 的关键区别：
    - TwoTower: User 和 Candidate 独立编码，仅通过点积交互
    - Ranking: User/History/Candidate 联合编码，有完整的 Cross-Attention
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        history_len: int = 64,
        num_candidates: int = 32,
        num_actions: int = 4,  # Tenrec 4 种 action
        dropout: float = 0.1,
        encoder_type: str = "fastformer",  # "fastformer" 或 "transformer"
        gradient_checkpointing: bool = False,
        num_categories: int = 0, # 支持类别特征
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.history_len = history_len
        self.num_candidates = num_candidates
        self.num_actions = num_actions
        self.encoder_type = encoder_type
        self.num_categories = num_categories

        # 1. Embedding 层
        self.user_embedding = nn.Embedding(num_users + 1, embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.action_embedding = nn.Embedding(num_actions + 1, embed_dim, padding_idx=0)

        # Side Info Embeddings
        self.gender_embedding = nn.Embedding(3, embed_dim, padding_idx=0)
        self.age_embedding = nn.Embedding(10, embed_dim, padding_idx=0)
        if num_categories > 0:
            self.category_embedding = nn.Embedding(num_categories + 1, embed_dim, padding_idx=0)
        else:
            self.category_embedding = None

        # 2. Segment Embedding (区分 User/History/Candidate)
        self.segment_embedding = SegmentEmbedding(embed_dim, num_segments=3)

        # 3. Position Embedding (用于 History 和 Candidate 序列)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=max(history_len, num_candidates) + 2, dropout=dropout)

        # 4. Encoder
        if encoder_type == "transformer":
            # 标准 PyTorch Transformer 编码器（O(n²) 但表达力更强）
            # [MLU 兼容性修复] 禁用 nested tensor 优化
            # MLU 不支持 aten::_nested_tensor_from_mask_left_aligned 操作，
            # 会导致 CPU fallback 和输出无区分度的问题
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            # enable_nested_tensor=False 禁用 nested tensor 优化，避免 MLU 兼容性问题
            # norm=None 与 Fastformer 保持一致（输出层有单独的 LayerNorm）
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers,
                enable_nested_tensor=False,
                norm=None,
            )
        else:
            # Fastformer 编码器（线性复杂度 O(n)，更快更稳定）
            from tenrec_adapter.fastformer import FastformerEncoder
            self.encoder = FastformerEncoder(
                d_model=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation="gelu",
                gradient_checkpointing=gradient_checkpointing,
            )

        # 5. Prediction Head
        self.output_norm = nn.LayerNorm(embed_dim)
        self.unembedding = nn.Linear(embed_dim, num_actions)

        # 梯度检查点
        if gradient_checkpointing:
            # FastformerEncoder handles its own gradient checkpointing internally
            # For TransformerEncoder, it's usually handled by `torch.utils.checkpoint`
            # If using PyTorch's TransformerEncoder, this line might need adjustment
            if hasattr(self.encoder, 'gradient_checkpointing_enable'):
                self.encoder.gradient_checkpointing_enable()
            else:
                # For standard TransformerEncoder, you'd wrap the forward pass
                # with torch.utils.checkpoint.checkpoint
                pass # Not directly enabling here for standard TransformerEncoder

        self._init_weights()

    def _init_weights(self):
        """初始化权重。"""
        with torch.no_grad():
            nn.init.xavier_uniform_(self.user_embedding.weight[1:])
            nn.init.xavier_uniform_(self.item_embedding.weight[1:])
            nn.init.xavier_uniform_(self.action_embedding.weight[1:])
            nn.init.xavier_uniform_(self.gender_embedding.weight[1:])
            nn.init.xavier_uniform_(self.age_embedding.weight[1:])
            if self.category_embedding is not None:
                nn.init.xavier_uniform_(self.category_embedding.weight[1:])

            # Zero padding
            self.user_embedding.weight[0].zero_()
            self.item_embedding.weight[0].zero_()
            self.action_embedding.weight[0].zero_()
            self.gender_embedding.weight[0].zero_()
            self.age_embedding.weight[0].zero_()
            if self.category_embedding is not None:
                self.category_embedding.weight[0].zero_()

        nn.init.xavier_uniform_(self.unembedding.weight)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        Args:
            batch: Data dictionary with keys:
                - user_id: [B]
                - history_item_ids: [B, S]
                - candidate_item_ids: [B, C]
                - gender: [B] (optional)
                - age: [B] (optional)
                - candidate_categories: [B, C] (optional)

        Returns:
            字典包含:
                - logits: [B, C, num_actions] 每个候选的 Action 预测
                - scores: [B, C] 综合得分 (click logit)
        """
        # Unpack inputs
        user_ids = batch["user_id"]                    # [B]
        history_item_ids = batch["history_item_ids"]   # [B, S]
        candidate_item_ids = batch["candidate_item_ids"]  # [B, C]

        # Side Info (optional)
        gender = batch.get("gender")                   # [B]
        age = batch.get("age")                         # [B]
        candidate_categories = batch.get("candidate_categories")  # [B, C]

        batch_size = user_ids.size(0)
        device = user_ids.device

        # === 1. User Feature [B, 1, D] ===
        user_emb = self.user_embedding(user_ids).unsqueeze(1)  # [B, 1, D]

        # Add side info to User Token
        if gender is not None:
            user_emb = user_emb + self.gender_embedding(gender).unsqueeze(1)
        if age is not None:
            user_emb = user_emb + self.age_embedding(age).unsqueeze(1)

        user_seg = self.segment_embedding(torch.zeros(1, 1, dtype=torch.long, device=device))
        user_feat = user_emb + user_seg

        # === 2. History Features [B, S, D] ===
        hist_emb = self.item_embedding(history_item_ids)  # [B, S, D]
        hist_feat = self.pos_encoding(hist_emb)
        hist_seg = self.segment_embedding(torch.ones(1, 1, dtype=torch.long, device=device))
        hist_feat = hist_feat + hist_seg

        # === 3. Candidate Features [B, C, D] ===
        cand_emb = self.item_embedding(candidate_item_ids)  # [B, C, D]

        # Add Category embedding
        if self.category_embedding is not None and candidate_categories is not None:
            cand_emb = cand_emb + self.category_embedding(candidate_categories)

        cand_feat = self.pos_encoding(cand_emb)
        cand_seg = self.segment_embedding(torch.full((1, 1), 2, dtype=torch.long, device=device))
        cand_feat = cand_feat + cand_seg

        # === 4. Concat [B, 1+S+C, D] ===
        input_seq = torch.cat([user_feat, hist_feat, cand_feat], dim=1)

        # Padding Mask (True = padded/ignore)
        user_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        hist_mask = (history_item_ids == 0)
        cand_mask = torch.zeros(batch_size, candidate_item_ids.size(1), dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([user_mask, hist_mask, cand_mask], dim=1)

        # === 5. Encode [B, L, D] ===
        enc_output = self.encoder(input_seq, src_key_padding_mask=key_padding_mask)

        # === 6. Extract Candidate positions ===
        num_cand = candidate_item_ids.size(1)
        cand_output = enc_output[:, -num_cand:, :]  # [B, C, D]

        # === 7. Predict Actions ===
        cand_output = self.output_norm(cand_output)
        logits = self.unembedding(cand_output)  # [B, C, num_actions]

        # 综合得分：raw click_logit 作为排序分数
        scores = logits[:, :, 0]  # [B, C]

        return {
            "logits": logits,
            "scores": scores,
        }


class TwoStageModel(nn.Module):
    """
    两阶段推荐模型。

    结合召回 (Retrieval) 和精排 (Ranking) 两个阶段。
    """

    def __init__(
        self,
        retrieval_model: nn.Module,
        ranking_model: nn.Module,
        top_k_retrieval: int = 100,
    ):
        super().__init__()
        self.retrieval_model = retrieval_model
        self.ranking_model = ranking_model
        self.top_k_retrieval = top_k_retrieval

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        corpus_embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        两阶段推理。

        Args:
            batch: 输入批次
            corpus_embeddings: 预计算的 Item Embedding 语料库

        Returns:
            最终排序结果
        """
        # 阶段 1: 召回
        if corpus_embeddings is not None:
            retrieval_output = self.retrieval_model(batch)
            user_repr = retrieval_output["user_repr"]  # [B, D]

            # 计算与语料库的相似度并取 Top-K
            scores = torch.mm(user_repr, corpus_embeddings.T)  # [B, N]
            top_k_scores, top_k_indices = torch.topk(scores, self.top_k_retrieval, dim=-1)

            # 构建精排输入
            batch["candidate_item_ids"] = top_k_indices

        # 阶段 2: 精排
        ranking_output = self.ranking_model(batch)

        return ranking_output
