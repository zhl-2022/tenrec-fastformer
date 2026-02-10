# MIT License - see LICENSE for details

"""
Fastformer 实现。

基于论文: "Fastformer: Additive Attention Can Be All You Need"
https://arxiv.org/abs/2108.09084

特点:
- 线性复杂度 O(n) 替代 O(n²)
- 专为推荐系统用户行为序列建模设计
- 效果与标准 Transformer 相当，速度 3-5x 更快
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastformerAttention(nn.Module):
    """
    Fastformer Additive Attention。
    
    核心思想：
    1. 使用 Additive Attention 替代 Dot-Product Attention
    2. 通过 Global Query 和 Global Key 聚合信息
    3. 避免 n×n 的注意力矩阵计算
    
    复杂度: O(nd) 替代 O(n²d)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Query, Key, Value 投影
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # Global Query 权重 (用于计算全局查询向量)
        self.query_weight = nn.Linear(d_model, num_heads)
        
        # Global Key 权重 (用于计算全局键向量)  
        self.key_weight = nn.Linear(d_model, num_heads)
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 注意：不在这里做 LayerNorm，交给 FastformerLayer 的 Pre-LN 处理
        # 这样避免双重归一化导致的数值不稳定
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重。"""
        for module in [self.query, self.key, self.value, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        
        nn.init.xavier_uniform_(self.query_weight.weight)
        nn.init.zeros_(self.query_weight.bias)
        nn.init.xavier_uniform_(self.key_weight.weight)
        nn.init.zeros_(self.key_weight.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播（MLU 兼容版本）。
        
        Args:
            x: 输入 [B, S, D]
            attention_mask: 注意力掩码 [B, S]，True 表示需要掩码的位置
            
        Returns:
            输出 [B, S, D]
        """
        batch_size, seq_len, _ = x.shape
        
        # 注意：这里不做归一化，由调用者 (FastformerLayer) 的 Pre-LN 处理
        
        # 计算 Q, K, V
        Q = self.query(x)  # [B, S, D]
        K = self.key(x)    # [B, S, D]
        V = self.value(x)  # [B, S, D]
        
        # 重塑为多头: [B, S, H, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # ===== Step 1: 计算 Global Query =====
        # 对 Query 进行加权求和得到全局查询向量
        query_weight = self.query_weight(x)  # [B, S, H]
        
        if attention_mask is not None:
            # 掩码位置设为较小的负数（MLU 兼容：避免 -inf 和极大负数）
            query_weight = query_weight.masked_fill(
                attention_mask.unsqueeze(-1), -1e4
            )
        
        # 稳定 softmax：减去最大值
        query_weight = query_weight - query_weight.max(dim=1, keepdim=True)[0]
        query_weight = F.softmax(query_weight, dim=1)  # [B, S, H]
        
        # 数值稳定性：clamp 避免极端值
        query_weight = torch.clamp(query_weight, min=1e-9, max=1.0)
        query_weight = self.dropout(query_weight)
        
        # Global Query: [B, H, head_dim]
        query_weight_expanded = query_weight.unsqueeze(-1)  # [B, S, H, 1]
        global_query = (query_weight_expanded * Q).sum(dim=1)  # [B, H, head_dim]
        
        # 归一化 Global Query（防止累积导致数值爆炸）
        global_query = F.normalize(global_query, p=2, dim=-1, eps=1e-6)
        
        # ===== Step 2: Query-Key 交互 =====
        P = K * global_query.unsqueeze(1)  # [B, S, H, head_dim]
        
        # ===== Step 3: 计算 Global Key =====
        # 通过加权求和得到全局键向量
        key_weight = self.key_weight(x)  # [B, S, H]
        
        if attention_mask is not None:
            key_weight = key_weight.masked_fill(
                attention_mask.unsqueeze(-1), -1e4
            )
        
        # 稳定 softmax
        key_weight = key_weight - key_weight.max(dim=1, keepdim=True)[0]
        key_weight = F.softmax(key_weight, dim=1)  # [B, S, H]
        key_weight = torch.clamp(key_weight, min=1e-9, max=1.0)
        key_weight = self.dropout(key_weight)
        
        # Global Key: [B, H, head_dim]
        key_weight_expanded = key_weight.unsqueeze(-1)  # [B, S, H, 1]
        global_key = (key_weight_expanded * P).sum(dim=1)  # [B, H, head_dim]
        
        # 归一化 Global Key（防止累积导致数值爆炸）
        global_key = F.normalize(global_key, p=2, dim=-1, eps=1e-6)
        
        # ===== Step 4: 计算输出 =====
        R = V * global_key.unsqueeze(1)  # [B, S, H, head_dim]
        R = R + Q  # 残差连接
        
        # 重塑回 [B, S, D]
        R = R.contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.out_proj(R)
        output = self.dropout(output)
        
        # 最终数值稳定性保护
        output = torch.clamp(output, -100.0, 100.0)
        
        return output


class FastformerLayer(nn.Module):
    """
    Fastformer 编码器层。
    
    结构: FastformerAttention -> FFN
    使用 Pre-LN (Layer Normalization before attention/FFN)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        
        # 注意力层
        self.attention = FastformerAttention(d_model, num_heads, dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer Normalization (Pre-LN)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入 [B, S, D]
            attention_mask: 注意力掩码 [B, S]
            
        Returns:
            输出 [B, S, D]
        """
        # Pre-LN + Attention + Residual
        residual = x
        x = self.norm1(x)
        x = self.attention(x, attention_mask)
        x = residual + x
        
        # Pre-LN + FFN + Residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class FastformerEncoder(nn.Module):
    """
    Fastformer 编码器。
    
    多层堆叠的 FastformerLayer。
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        self.gradient_checkpointing = gradient_checkpointing
        
        self.layers = nn.ModuleList([
            FastformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ])
        
        # 最终 Layer Normalization
        self.final_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入 [B, S, D]
            src_key_padding_mask: Padding 掩码 [B, S]，True 表示 padding 位置
            
        Returns:
            输出 [B, S, D]
        """
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # 梯度检查点：前向时不保存中间激活，backward 时重算
                # use_reentrant=False 是 PyTorch 推荐的新模式，更安全
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, src_key_padding_mask, use_reentrant=False
                )
            else:
                x = layer(x, src_key_padding_mask)
        
        x = self.final_norm(x)
        
        return x


class FastformerForSequenceClassification(nn.Module):
    """
    用于序列分类的 Fastformer。
    
    适用于推荐系统的用户行为序列建模。
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        max_seq_len: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ):
        super().__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Fastformer 编码器
        self.encoder = FastformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        
        # 分类头
        self.classifier = nn.Linear(d_model, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重。"""
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            input_ids: 输入 ID [B, S]
            attention_mask: 注意力掩码 [B, S]，1 表示有效，0 表示 padding
            
        Returns:
            logits [B, num_classes]
        """
        batch_size, seq_len = input_ids.shape
        
        # 嵌入
        x = self.embedding(input_ids)
        
        # 位置编码
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # 创建 padding mask (True 表示需要掩码)
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None
        
        # 编码
        x = self.encoder(x, padding_mask)
        
        # 取 [CLS] 位置 (第一个 token) 或平均池化
        x = x[:, 0]  # 取第一个位置
        
        # 分类
        logits = self.classifier(x)
        
        return logits
