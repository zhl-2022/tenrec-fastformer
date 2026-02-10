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
        num_actions: int = 4,  # Tenrec 4 种 action: click, like, share, follow
        dropout: float = 0.1,
        encoder_type: str = "fastformer",  # "fastformer" 或 "transformer"
        gradient_checkpointing: bool = False,  # 梯度检查点（以时间换显存）
    ):
        """
        初始化精排模型。
        
        Args:
            num_users: 用户数量
            num_items: Item 数量
            embed_dim: Embedding 维度
            hidden_dim: FFN 隐藏层维度
            num_heads: 注意力头数
            num_layers: Transformer 层数
            history_len: 历史序列长度
            num_candidates: 候选数量
            num_actions: Action 类型数 (默认 19)
            dropout: Dropout 比例
            encoder_type: 编码器类型 ("fastformer" 或 "transformer")
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.history_len = history_len
        self.num_candidates = num_candidates
        self.num_actions = num_actions
        self.encoder_type = encoder_type
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users + 1, embed_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.action_projection = nn.Linear(num_actions, embed_dim)
        self.segment_embedding = SegmentEmbedding(embed_dim, num_segments=3)
        
        # 位置编码（增大缓冲区，支持动态 num_candidates）
        # 注意：实际 num_candidates 可能被外部设为 64，所以需要足够大的缓冲
        max_seq_len = 1 + history_len + num_candidates * 2 + 64  # 增大缓冲区
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)
        
        # 根据 encoder_type 选择编码器
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
            self.transformer = nn.TransformerEncoder(
                encoder_layer, 
                num_layers,
                enable_nested_tensor=False,
                norm=None,
            )
        else:
            # Fastformer 编码器（线性复杂度 O(n)，更快更稳定）
            from tenrec_adapter.fastformer import FastformerEncoder
            self.transformer = FastformerEncoder(
                d_model=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation="gelu",
                gradient_checkpointing=gradient_checkpointing,
            )
        
        # 输出层
        self.output_norm = nn.LayerNorm(embed_dim)
        self.unembedding = nn.Linear(embed_dim, num_actions)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重。"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.action_projection.weight)
        nn.init.xavier_uniform_(self.unembedding.weight)
    
    def build_inputs(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> tuple:
        """
        构建模型输入。
        
        将 User + History + Candidates 拼接成统一序列。
        
        Returns:
            embeddings: [B, 1 + S + C, D]
            padding_mask: [B, 1 + S + C]
            candidate_start: int
        """
        device = batch["user_ids"].device
        batch_size = batch["user_ids"].size(0)
        
        # === 1. User Embedding [B, 1, D] ===
        user_emb = self.user_embedding(batch["user_ids"])  # [B, D]
        user_emb = user_emb.unsqueeze(1)  # [B, 1, D]
        user_segment = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        # === 2. History Embedding [B, S, D] ===
        history_item_emb = self.item_embedding(batch["history_item_ids"])  # [B, S, D]
        history_action_emb = self.action_projection(batch["history_actions"])  # [B, S, D]
        history_emb = history_item_emb + history_action_emb  # [B, S, D]
        
        history_len = history_emb.size(1)
        history_segment = torch.ones(batch_size, history_len, dtype=torch.long, device=device)
        
        # === 3. Candidate Embedding [B, C, D] ===
        candidate_emb = self.item_embedding(batch["candidate_item_ids"])  # [B, C, D]
        
        num_candidates = candidate_emb.size(1)
        candidate_segment = torch.full((batch_size, num_candidates), 2, dtype=torch.long, device=device)
        
        # === 4. 拼接 ===
        embeddings = torch.cat([user_emb, history_emb, candidate_emb], dim=1)  # [B, 1+S+C, D]
        segment_ids = torch.cat([user_segment, history_segment, candidate_segment], dim=1)
        
        # 添加分段 Embedding
        segment_emb = self.segment_embedding(segment_ids)  # [B, L, D]
        embeddings = embeddings + segment_emb
        
        # 位置编码
        embeddings = self.pos_encoding(embeddings)
        
        # === 5. Padding Mask ===
        user_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        history_mask = (batch["history_item_ids"] == 0)  # [B, S]
        candidate_mask = (batch["candidate_item_ids"] == 0)  # [B, C]
        padding_mask = torch.cat([user_mask, history_mask, candidate_mask], dim=1)
        
        candidate_start = 1 + history_len
        
        return embeddings, padding_mask, candidate_start, num_candidates
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        isolate_candidates: bool = False,  # 默认关闭隔离模式以节省显存
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播。
        
        Args:
            batch: 包含以下键:
                - user_ids: [B]
                - history_item_ids: [B, S]
                - history_actions: [B, S, num_actions]
                - candidate_item_ids: [B, C]
            isolate_candidates: 是否隔离候选（每个候选独立编码，防止互相影响）
                - True: 候选不互相影响（与 Phoenix 原版一致）
                - False: 候选可互相影响（原始 Fastformer 行为）
                
        Returns:
            字典包含:
                - logits: [B, C, num_actions] 每个候选的 Action 预测
                - scores: [B, C] 综合得分 (logits 加权求和)
        """
        if isolate_candidates:
            # ===== 候选隔离模式（与 Phoenix recsys_attn_mask 效果相同）=====
            # 每个候选独立与 User+History 拼接后编码
            # 这样候选之间不会互相影响
            return self._forward_isolated(batch)
        else:
            # ===== 原始模式（候选可互相影响）=====
            return self._forward_joint(batch)
    
    def _forward_joint(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """联合编码模式（原始行为，候选可互相影响）。"""
        # 构建输入
        embeddings, padding_mask, candidate_start, num_candidates = self.build_inputs(batch)
        
        # Transformer 编码
        encoded = self.transformer(
            embeddings,
            src_key_padding_mask=padding_mask,
        )  # [B, 1+S+C, D]
        
        # 提取 Candidate 位置的输出
        candidate_encoded = encoded[:, candidate_start:candidate_start + num_candidates, :]  # [B, C, D]
        
        # 输出层
        candidate_encoded = self.output_norm(candidate_encoded)
        logits = self.unembedding(candidate_encoded)  # [B, C, num_actions]
        
        # 综合得分：使用 raw click_logit 作为排序分数
        # [Bug Fix] sigmoid 在饱和区（logit≈-5）将所有分数压缩到 ~0.005，
        # 导致 scores std ≈ 0.0005，丧失区分能力。raw logit 保留完整区分信号。
        # AUC/NDCG/MRR 只关心相对排序，不需要概率语义。
        scores = logits[:, :, 0]  # [B, C] raw click logit
        
        return {
            "logits": logits,
            "scores": scores,
        }
    
    def _forward_isolated(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        候选隔离模式（每个候选独立编码，防止互相影响）。
        
        实现原理：
        1. 将每个候选展平为独立的 batch 样本
        2. 每个候选独立与 User+History 拼接后编码
        3. 恢复原始 batch 形状
        
        这与 Phoenix 的 recsys_attn_mask 效果相同，但适用于 Fastformer。
        """
        device = batch["user_ids"].device
        B = batch["user_ids"].size(0)
        C = batch["candidate_item_ids"].size(1)  # 候选数量
        
        # === 1. User Embedding [B, 1, D] ===
        user_emb = self.user_embedding(batch["user_ids"])  # [B, D]
        user_emb = user_emb.unsqueeze(1)  # [B, 1, D]
        
        # === 2. History Embedding [B, S, D] ===
        history_item_emb = self.item_embedding(batch["history_item_ids"])  # [B, S, D]
        history_action_emb = self.action_projection(batch["history_actions"])  # [B, S, D]
        history_emb = history_item_emb + history_action_emb  # [B, S, D]
        S = history_emb.size(1)
        
        # === 3. Candidate Embedding [B, C, D] ===
        candidate_emb = self.item_embedding(batch["candidate_item_ids"])  # [B, C, D]
        
        # === 4. 为每个候选独立编码 ===
        # 扩展 user 和 history: [B, 1+S, D] -> [B, C, 1+S, D] -> [B*C, 1+S, D]
        user_history_emb = torch.cat([user_emb, history_emb], dim=1)  # [B, 1+S, D]
        user_history_emb = user_history_emb.unsqueeze(1).expand(-1, C, -1, -1)  # [B, C, 1+S, D]
        user_history_emb = user_history_emb.reshape(B * C, 1 + S, self.embed_dim)  # [B*C, 1+S, D]
        
        # 扩展 candidate: [B, C, D] -> [B*C, 1, D]
        candidate_emb_expanded = candidate_emb.reshape(B * C, 1, self.embed_dim)  # [B*C, 1, D]
        
        # 拼接: [B*C, 1+S+1, D]
        embeddings = torch.cat([user_history_emb, candidate_emb_expanded], dim=1)  # [B*C, 1+S+1, D]
        
        # 添加分段 Embedding
        segment_ids = torch.cat([
            torch.zeros(B * C, 1, dtype=torch.long, device=device),  # user
            torch.ones(B * C, S, dtype=torch.long, device=device),   # history
            torch.full((B * C, 1), 2, dtype=torch.long, device=device),  # candidate
        ], dim=1)
        segment_emb = self.segment_embedding(segment_ids)
        embeddings = embeddings + segment_emb
        
        # 位置编码
        embeddings = self.pos_encoding(embeddings)
        
        # Padding mask
        user_mask = torch.zeros(B * C, 1, dtype=torch.bool, device=device)
        history_mask = (batch["history_item_ids"] == 0)  # [B, S]
        history_mask = history_mask.unsqueeze(1).expand(-1, C, -1).reshape(B * C, S)  # [B*C, S]
        candidate_mask = torch.zeros(B * C, 1, dtype=torch.bool, device=device)  # 候选不 mask
        padding_mask = torch.cat([user_mask, history_mask, candidate_mask], dim=1)  # [B*C, 1+S+1]
        
        # === 5. Transformer 编码 ===
        encoded = self.transformer(
            embeddings,
            src_key_padding_mask=padding_mask,
        )  # [B*C, 1+S+1, D]
        
        # === 6. 提取候选位置输出 ===
        candidate_encoded = encoded[:, -1, :]  # [B*C, D] - 最后一个位置是候选
        candidate_encoded = candidate_encoded.reshape(B, C, self.embed_dim)  # [B, C, D]
        
        # === 7. 输出层 ===
        candidate_encoded = self.output_norm(candidate_encoded)
        logits = self.unembedding(candidate_encoded)  # [B, C, num_actions]
        
        # 综合得分：使用 raw click_logit（与 _forward_joint 保持一致）
        scores = logits[:, :, 0]  # [B, C] raw click logit
        
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
