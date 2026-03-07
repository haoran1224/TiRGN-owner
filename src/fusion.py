"""
局部-全局融合模块 (Local-Global Fusion Module)

该模块实现了通过门控机制将局部节点表示与全局路径表示融合的方法。
融合公式:
    g = σ(W_gate [h_s_local || h_path_global] + b)
    h_s_final = g · h_s_local + (1 - g) · h_path_global
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LocalGlobalFusion(nn.Module):
    """
    局部-全局融合模块

    该模块通过可学习的门控机制自适应地融合局部节点表示和全局路径表示，
    用于时序知识图谱推理任务中的特征融合。

    Attributes:
        h_dim: 嵌入向量的维度
        gate_weight: 门控权重矩阵 W_gate，形状为 [h_dim, 2*h_dim]
        gate_bias: 门控偏置向量 b，形状为 [h_dim]
        dropout: 可选的 Dropout 层
        layer_norm: 可选的 LayerNorm 层
    """

    def __init__(
        self,
        h_dim: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_multi_head_gate: bool = False,
        num_heads: int = 4
    ):
        """
        初始化局部-全局融合模块

        Args:
            h_dim: 嵌入维度
            dropout: Dropout 比例，默认为 0.1
            use_layer_norm: 是否使用 LayerNorm，默认为 True
            use_multi_head_gate: 是否使用多头门控，默认为 False
            num_heads: 多头门控的头数，默认为 4
        """
        super(LocalGlobalFusion, self).__init__()

        self.h_dim = h_dim
        self.use_layer_norm = use_layer_norm
        self.use_multi_head_gate = use_multi_head_gate
        self.num_heads = num_heads

        if use_multi_head_gate:
            # 多头门控机制：每个头独立学习门控权重
            # 将 h_dim 分成 num_heads 个子空间
            assert h_dim % num_heads == 0, "h_dim 必须能被 num_heads 整除"
            self.head_dim = h_dim // num_heads

            # 多头门控权重: [num_heads, head_dim, 2*head_dim]
            self.gate_weight = nn.Parameter(
                torch.Tensor(num_heads, self.head_dim, 2 * self.head_dim)
            )
            # 多头门控偏置: [num_heads, head_dim]
            self.gate_bias = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        else:
            # 单头门控权重: [h_dim, 2*h_dim]
            self.gate_weight = nn.Parameter(torch.Tensor(h_dim, 2 * h_dim))
            # 门控偏置: [h_dim]
            self.gate_bias = nn.Parameter(torch.Tensor(h_dim))

        # 初始化参数
        self._init_parameters()

        # Dropout 层
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # LayerNorm 层
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(h_dim)
        else:
            self.layer_norm = None

    def _init_parameters(self):
        """初始化模块参数"""
        if self.use_multi_head_gate:
            nn.init.xavier_uniform_(self.gate_weight, gain=nn.init.calculate_gain('sigmoid'))
            nn.init.zeros_(self.gate_bias)
        else:
            nn.init.xavier_uniform_(self.gate_weight, gain=nn.init.calculate_gain('sigmoid'))
            nn.init.zeros_(self.gate_bias)

    def forward(
        self,
        h_s_local: torch.Tensor,
        h_path_global: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播：融合局部节点表示和全局路径表示

        Args:
            h_s_local: 局部节点表示
                形状为 [batch_size, h_dim] 或 [batch_size, num_nodes, h_dim]
            h_path_global: 全局路径表示
                形状为 [batch_size, h_dim] 或 [batch_size, num_nodes, h_dim]
            mask: 可选的掩码张量，用于屏蔽某些位置的融合
                形状为 [batch_size] 或 [batch_size, num_nodes]
                值为 0 表示该位置有效，值为 1 表示该位置需要屏蔽

        Returns:
            h_s_final: 融合后的最终节点表示
                形状与输入相同
        """
        # 确保输入在相同设备上
        device = h_s_local.device
        h_path_global = h_path_global.to(device)

        # 处理不同的输入形状
        if h_s_local.dim() == 2:
            # [batch_size, h_dim] -> [batch_size, 1, h_dim]
            h_s_local = h_s_local.unsqueeze(1)
            h_path_global = h_path_global.unsqueeze(1)
            squeeze_output = True
        else:
            # [batch_size, num_nodes, h_dim]
            squeeze_output = False

        batch_size, num_nodes, h_dim = h_s_local.shape

        # 1. 拼接局部表示和全局表示
        # Shape: [batch_size, num_nodes, 2*h_dim]
        concatenated = torch.cat([h_s_local, h_path_global], dim=-1)

        if self.use_multi_head_gate:
            # 多头门控机制
            # 将输入reshape为多头形式
            # [batch_size, num_nodes, num_heads, 2*head_dim]
            concatenated = concatenated.view(
                batch_size, num_nodes, self.num_heads, 2 * self.head_dim
            )

            # 对每个头独立计算门控值
            # gate_weight: [num_heads, head_dim, 2*head_dim]
            # 我们需要对其转置: [num_heads, 2*head_dim, head_dim]
            gate_weight_t = self.gate_weight.transpose(1, 2)  # [num_heads, 2*head_dim, head_dim]

            # 使用 einsum 进行批矩阵乘法
            # 'btnh, nhm -> btnm':
            #   - b: batch_size
            #   - t: num_nodes
            #   - n: num_heads
            #   - h: 2*head_dim (输入维度)
            #   - m: head_dim (输出维度)
            gate_per_head = torch.einsum('btnh, nhm -> btnm', concatenated, gate_weight_t)
            # gate_per_head: [batch_size, num_nodes, num_heads, head_dim]

            # 加上偏置
            gate = gate_per_head + self.gate_bias.unsqueeze(0).unsqueeze(0)  # [batch_size, num_nodes, num_heads, head_dim]
            gate = torch.sigmoid(gate)

            # 重塑回原始形状
            gate = gate.reshape(batch_size, num_nodes, h_dim)  # [batch_size, num_nodes, h_dim]

        else:
            # 单头门控机制
            # 2. 计算门控值 g = σ(W_gate @ [h_local || h_global] + b)
            # concatenated @ gate_weight^T: [batch_size, num_nodes, h_dim]
            gate = torch.matmul(concatenated, self.gate_weight.t())

            # 加上偏置: [batch_size, num_nodes, h_dim]
            gate = gate + self.gate_bias.unsqueeze(0).unsqueeze(0)

            # 通过 sigmoid 激活函数得到门控值
            # Shape: [batch_size, num_nodes, h_dim]
            gate = torch.sigmoid(gate)

        # 3. 应用掩码（如果有）
        if mask is not None:
            if mask.dim() == 1:
                # [batch_size] -> [batch_size, 1, 1]
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 2:
                # [batch_size, num_nodes] -> [batch_size, num_nodes, 1]
                mask = mask.unsqueeze(-1)
            gate = gate * (1 - mask)

        # 4. 执行门控融合 h_final = g · h_local + (1 - g) · h_global
        # 逐元素相乘: [batch_size, num_nodes, h_dim]
        gated_local = gate * h_s_local
        gated_global = (1 - gate) * h_path_global

        # 相加得到最终表示: [batch_size, num_nodes, h_dim]
        h_s_final = gated_local + gated_global

        # 5. 可选的 LayerNorm
        if self.layer_norm is not None:
            h_s_final = self.layer_norm(h_s_final)

        # 6. 可选的 Dropout
        if self.dropout is not None:
            h_s_final = self.dropout(h_s_final)

        # 恢复原始形状
        if squeeze_output:
            h_s_final = h_s_final.squeeze(1)  # [batch_size, h_dim]

        return h_s_final


class HierarchicalLocalGlobalFusion(nn.Module):
    """
    分层局部-全局融合模块

    支持多层级的融合，适用于需要逐步融合多个层次特征的场景。
    例如：先融合时间特征，再融合空间特征。

    Attributes:
        h_dim: 嵌入向量的维度
        num_layers: 融合层数
        fusion_layers: 融合层列表
    """

    def __init__(
        self,
        h_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        初始化分层融合模块

        Args:
            h_dim: 嵌入维度
            num_layers: 融合层数，默认为 2
            dropout: Dropout 比例，默认为 0.1
            use_layer_norm: 是否使用 LayerNorm，默认为 True
        """
        super(HierarchicalLocalGlobalFusion, self).__init__()

        self.h_dim = h_dim
        self.num_layers = num_layers

        # 创建多层融合模块
        self.fusion_layers = nn.ModuleList([
            LocalGlobalFusion(
                h_dim=h_dim,
                dropout=dropout,
                use_layer_norm=use_layer_norm
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        h_local: torch.Tensor,
        h_global: torch.Tensor,
        intermediate_features: Optional[list] = None
    ) -> torch.Tensor:
        """
        分层融合前向传播

        Args:
            h_local: 局部节点表示，形状为 [batch_size, h_dim] 或 [batch_size, num_nodes, h_dim]
            h_global: 全局路径表示，形状与 h_local 相同
            intermediate_features: 可选的中间特征列表，用于多层融合

        Returns:
            h_final: 融合后的最终表示，形状与输入相同
        """
        # 设备兼容性
        device = h_local.device
        h_global = h_global.to(device)

        if intermediate_features is None:
            # 如果没有提供中间特征，使用标准的多层融合
            h = h_local
            for fusion_layer in self.fusion_layers:
                h = fusion_layer(h, h_global)
            return h
        else:
            # 使用中间特征进行分层融合
            h = h_local
            all_features = [h_global] + intermediate_features
            for i, fusion_layer in enumerate(self.fusion_layers):
                if i < len(all_features):
                    h = fusion_layer(h, all_features[i].to(device))
                else:
                    h = fusion_layer(h, h_global)
            return h


class AttentionBasedFusion(nn.Module):
    """
    基于注意力的局部-全局融合模块

    使用多头自注意力机制进行局部和全局表示的融合，
    相比简单的门控机制，可以捕获更复杂的特征交互。

    Attributes:
        h_dim: 嵌入向量的维度
        num_heads: 注意力头的数量
        attention: 多头注意力层
        fusion_weight: 融合权重
    """

    def __init__(
        self,
        h_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        初始化基于注意力的融合模块

        Args:
            h_dim: 嵌入维度
            num_heads: 注意力头数，默认为 8
            dropout: Dropout 比例，默认为 0.1
            use_layer_norm: 是否使用 LayerNorm，默认为 True
        """
        super(AttentionBasedFusion, self).__init__()

        self.h_dim = h_dim
        self.num_heads = num_heads

        # 多头自注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=h_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 融合权重（可学习的标量，用于加权融合）
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

        # LayerNorm
        if use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(h_dim)
            self.layer_norm2 = nn.LayerNorm(h_dim)
        else:
            self.layer_norm1 = None
            self.layer_norm2 = None

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(h_dim, h_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim * 4, h_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        h_local: torch.Tensor,
        h_global: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        基于注意力的融合前向传播

        Args:
            h_local: 局部节点表示，形状为 [batch_size, seq_len, h_dim]
            h_global: 全局路径表示，形状为 [batch_size, seq_len, h_dim]
            query_mask: 可选的查询掩码，形状为 [batch_size, seq_len]

        Returns:
            h_final: 融合后的最终表示，形状与输入相同
        """
        device = h_local.device
        h_global = h_global.to(device)

        # 处理维度
        if h_local.dim() == 2:
            h_local = h_local.unsqueeze(1)
            h_global = h_global.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        # 拼接作为 key 和 value
        # Shape: [batch_size, 2*seq_len, h_dim]
        kv = torch.cat([h_local, h_global], dim=1)

        # 自注意力计算
        # attn_output: [batch_size, seq_len, h_dim]
        attn_output, _ = self.attention(
            query=h_local,
            key=kv,
            value=kv,
            key_padding_mask=query_mask
        )

        # 残差连接和 LayerNorm
        if self.layer_norm1 is not None:
            h = self.layer_norm1(h_local + attn_output)
        else:
            h = h_local + attn_output

        # FFN
        ffn_output = self.ffn(h)

        # 残差连接和 LayerNorm
        if self.layer_norm2 is not None:
            h = self.layer_norm2(h + ffn_output)
        else:
            h = h + ffn_output

        # 可学习的权重融合
        alpha = torch.sigmoid(self.fusion_weight)
        h_final = alpha * h + (1 - alpha) * h_global

        if squeeze_output:
            h_final = h_final.squeeze(1)

        return h_final


def create_fusion_module(
    fusion_type: str = "gate",
    h_dim: int = 128,
    **kwargs
) -> nn.Module:
    """
    工厂函数：创建指定类型的融合模块

    Args:
        fusion_type: 融合模块类型
            - "gate": 标准门控融合
            - "multi_head_gate": 多头门控融合
            - "hierarchical": 分层融合
            - "attention": 基于注意力的融合
        h_dim: 嵌入维度
        **kwargs: 其他参数

    Returns:
        fusion_module: 融合模块实例
    """
    if fusion_type == "gate":
        return LocalGlobalFusion(h_dim=h_dim, **kwargs)
    elif fusion_type == "multi_head_gate":
        return LocalGlobalFusion(h_dim=h_dim, use_multi_head_gate=True, **kwargs)
    elif fusion_type == "hierarchical":
        return HierarchicalLocalGlobalFusion(h_dim=h_dim, **kwargs)
    elif fusion_type == "attention":
        return AttentionBasedFusion(h_dim=h_dim, **kwargs)
    else:
        raise ValueError(f"未知的融合类型: {fusion_type}")
