import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    """位置编码模块
    
    为输入序列添加位置信息，使模型能够理解特征的顺序关系。
    
    Args:
        d_model (int): 嵌入维度
        max_len (int, optional): 最大序列长度，默认为10
    """
    def __init__(self, d_model: int, max_len: int = 10):
        super(PositionalEncoding, self).__init__()
        
        # 预计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数位置使用正弦，奇数位置使用余弦
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 调整形状为 [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: 添加位置编码后的张量，形状为 [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]

class FeatureSequenceEncoder(nn.Module):
    """特征序列编码器
    
    将4个物理量编码为4个token的序列
    
    Args:
        hidden_dim (int, optional): 嵌入维度，默认为128
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        # 3个特征的嵌入层（顺序必须与实际特征顺序一致）
        # 顺序：log_L, log_Teff, [Fe/H]
        self.feature_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ) for _ in range(3)
        ])
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=3)
        
        # 特征类型嵌入（可选）
        self.feature_type_embedding = nn.Embedding(3, hidden_dim)
    
    def forward(self, logL: torch.Tensor, logTeff: torch.Tensor, FeH: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            logL (torch.Tensor): 恒星光度的对数 (log_L)
            logTeff (torch.Tensor): 恒星有效温度的对数 (log_Teff)
            FeH (torch.Tensor): 金属丰度 ([Fe/H])
            
        Returns:
            torch.Tensor: 编码后的特征序列，形状为 [batch, 3, hidden_dim]
        """
        batch_size = logL.shape[0]
        
        # 特征列表
        features = [logL, logTeff, FeH]
        tokens = []
        
        for i, feat in enumerate(features):
            # 特征值编码
            feat_tensor = feat.view(-1, 1)
            encoded = self.feature_embeddings[i](feat_tensor)  # [batch, hidden]
            
            # 特征类型嵌入
            type_ids = torch.full((batch_size,), i, device=logL.device)
            type_emb = self.feature_type_embedding(type_ids)
            
            # token = 特征编码 + 类型嵌入
            token = encoded + type_emb
            tokens.append(token.unsqueeze(1))
        
        # 组合成序列 [光度, 温度, 金属丰度]
        sequence = torch.cat(tokens, dim=1)  # [batch, 3, hidden]
        
        # 添加位置编码
        sequence = self.pos_encoder(sequence)
        
        return sequence

class StellarTransformer(nn.Module):
    """恒星质量预测Transformer
    
    基于Transformer架构的恒星质量预测模型，使用对数光度、对数有效温度和金属丰度作为输入。
    
    Args:
        hidden_dim (int, optional): 嵌入维度，默认为128
        num_layers (int, optional): Transformer编码器层数，默认为4
        num_heads (int, optional): 多头注意力头数，默认为4
        dropout (float, optional): Dropout概率，默认为0.1
    """
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4, 
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # 1. 编码器
        self.encoder = FeatureSequenceEncoder(hidden_dim)
        
        # 2. Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 3. 回归头（预测质量）
        # 使用[CLS] token策略：取序列第一个token（光度token）
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # 4. 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重
        
        使用Xavier均匀分布初始化所有可训练参数。
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, logL: torch.Tensor, logTeff: torch.Tensor, 
                FeH: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            logL (torch.Tensor): 恒星光度的对数 (log_L)
            logTeff (torch.Tensor): 恒星有效温度的对数 (log_Teff)
            FeH (torch.Tensor): 金属丰度 ([Fe/H])
            
        Returns:
            torch.Tensor: 预测的恒星质量，形状为 [batch]
        """
        # 1. 编码为序列
        sequence = self.encoder(logL, logTeff, FeH)  # [batch, 3, hidden]
        
        # 2. Transformer处理
        transformed = self.transformer(sequence)  # [batch, 3, hidden]
        
        # 3. 取光度token（位置0）作为代表
        # 模型会学到这个token包含了最重要的信息
        luminosity_token = transformed[:, 0, :]  # [batch, hidden]
        
        # 4. 预测质量
        mass_pred = self.regression_head(luminosity_token)  # [batch, 1]
        
        return mass_pred.squeeze(-1)  # [batch]

class StellarTransformerWithPooling(nn.Module):
    """带有全局平均池化的恒星质量预测Transformer
    
    另一种变体：使用全局平均池化聚合序列特征进行预测。
    
    Args:
        hidden_dim (int, optional): 嵌入维度，默认为128
        num_layers (int, optional): Transformer编码器层数，默认为3
        num_heads (int, optional): 多头注意力头数，默认为4
        dropout (float, optional): Dropout概率，默认为0.1
    """
    def __init__(self, hidden_dim: int = 128, num_layers: int = 4, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = FeatureSequenceEncoder(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 全局平均池化
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 回归头
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, logL: torch.Tensor, logTeff: torch.Tensor, 
                FeH: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            logL (torch.Tensor): 恒星光度的对数 (log_L)
            logTeff (torch.Tensor): 恒星有效温度的对数 (log_Teff)
            FeH (torch.Tensor): 金属丰度 ([Fe/H])
            
        Returns:
            torch.Tensor: 预测的恒星质量，形状为 [batch]
        """
        # 编码
        sequence = self.encoder(logL, logTeff, FeH)
        
        # Transformer
        transformed = self.transformer(sequence)  # [batch, 3, hidden]
        
        # 全局平均池化
        pooled = transformed.transpose(1, 2)  # [batch, hidden, 3]
        pooled = self.pool(pooled).squeeze(-1)  # [batch, hidden]
        
        # 预测
        mass = self.regression_head(pooled)
        
        return mass.squeeze(-1)


class WeightedMSELoss(nn.Module):
    """加权均方误差损失函数
    
    为极大和极小质量样本分配更高的权重，让模型更重视极端值的预测准确性。
    
    Args:
        low_threshold (float, optional): 极小质量的阈值，低于此值的样本将被赋予更高权重
        high_threshold (float, optional): 极大质量的阈值，高于此值的样本将被赋予更高权重
        low_weight (float, optional): 极小质量样本的权重
        high_weight (float, optional): 极大质量样本的权重
        mid_weight (float, optional): 中间质量样本的权重
    """
    def __init__(self, low_threshold: float = -1.0, high_threshold: float = 1.0, 
                 low_weight: float = 2.0, high_weight: float = 2.0, mid_weight: float = 1.0):
        super(WeightedMSELoss, self).__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.low_weight = low_weight
        self.high_weight = high_weight
        self.mid_weight = mid_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            pred (torch.Tensor): 预测值
            target (torch.Tensor): 真实值
            
        Returns:
            torch.Tensor: 加权均方误差损失
        """
        # 计算权重
        weights = torch.ones_like(target)
        weights = torch.where(target < self.low_threshold, self.low_weight, weights)
        weights = torch.where(target > self.high_threshold, self.high_weight, weights)
        weights = torch.where((target >= self.low_threshold) & (target <= self.high_threshold), self.mid_weight, weights)
        
        # 计算加权MSE
        mse = (pred - target) ** 2
        weighted_mse = (mse * weights).mean()
        
        return weighted_mse


class FocalMSELoss(nn.Module):
    """Focal MSE损失函数
    
    基于Focal Loss的思想，针对难分样本（预测误差大的样本）加大惩罚，
    同时降低易分样本的权重，解决模型只学中间样本的问题。
    
    Args:
        alpha (float, optional): 平衡因子，默认为0.25
        gamma (float, optional): 聚焦参数，默认为2.0
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalMSELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            pred (torch.Tensor): 预测值
            target (torch.Tensor): 真实值
            
        Returns:
            torch.Tensor: Focal MSE损失
        """
        # 计算MSE损失
        mse = (pred - target) ** 2
        
        # 计算预测难度（与真实值的绝对误差）
        pt = torch.exp(-mse)
        
        # 计算Focal损失
        focal_loss = self.alpha * (1 - pt) ** self.gamma * mse
        
        return focal_loss.mean()