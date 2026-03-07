import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PathEncoder(nn.Module):
    """
    路径编码器模块

    该模块用于编码历史路径信息，通过 GRU 和注意力机制对路径进行建模，
    生成全局逻辑表示。

    Attributes:
        num_ents: 实体数量
        num_rels: 关系数量
        h_dim: 嵌入维度
        K: 每条查询的路径数量
        max_seq_len: 路径的最大长度
    """

    def __init__(self, num_ents, num_rels, h_dim, K=50, max_seq_len=10):
        """
        初始化路径编码器

        Args:
            num_ents: 实体数量
            num_rels: 关系数量
            h_dim: 嵌入维度
            K: 每条查询的路径数量
            max_seq_len: 路径的最大长度
        """
        super(PathEncoder, self).__init__()
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.h_dim = h_dim
        self.K = K
        self.max_seq_len = max_seq_len

        # 时间编码层
        self.time_encoder = nn.Linear(1, h_dim)

        # GRU 用于序列编码
        self.gru = nn.GRU(input_size=h_dim * 2, hidden_size=h_dim, batch_first=True)

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, 1)
        )

    def forward(self, paths, path_lengths, query_times, entity_emb, relation_emb):
        """
        前向传播函数

        Args:
            paths: 路径张量，形状为 [batch_size, K, max_seq_len, 4]，其中 4 表示 (s, r, o, t)
            path_lengths: 路径长度张量，形状为 [batch_size, K]
            query_times: 查询时间张量，形状为 [batch_size]
            entity_emb: 实体嵌入，形状为 [num_ents, h_dim]
            relation_emb: 关系嵌入，形状为 [num_rels * 2, h_dim]

        Returns:
            global_emb: 全局逻辑表示，形状为 [batch_size, h_dim]
        """
        batch_size = paths.size(0)
        device = paths.device

        # 确保嵌入在正确的设备上
        entity_emb = entity_emb.to(device)
        relation_emb = relation_emb.to(device)

        # 确保 paths 是正确的类型和形状
        if paths.dim() != 4:
            raise ValueError(f"paths 应该是 4D 张量 [batch_size, K, max_seq_len, 4]，但得到形状: {paths.shape}")
        if paths.size(-1) != 4:
            raise ValueError(f"paths 的最后一维应该是 4，但得到: {paths.size(-1)}")

        # 确保 query_times 的长度与 batch_size 匹配
        if query_times.size(0) != batch_size:
            # 如果不匹配，截断或扩展 query_times
            if query_times.size(0) > batch_size:
                query_times = query_times[:batch_size]
            else:
                # 扩展 query_times（重复最后一个值）
                last_time = query_times[-1]
                padding = last_time.unsqueeze(0).expand(batch_size - query_times.size(0))
                query_times = torch.cat([query_times, padding], dim=0)

        # 1. 特征嵌入与时间差编码
        # 关键修复：添加边界检查，防止索引越界导致 nan
        num_ents = entity_emb.size(0)
        num_rels = relation_emb.size(0)

        # 对实体 ID 进行裁剪，防止越界
        entity_ids = paths[..., 0].long()
        entity_ids = torch.clamp(entity_ids, 0, num_ents - 1)

        # 对关系 ID 进行裁剪，防止越界
        rel_ids = paths[..., 1].long()
        rel_ids = torch.clamp(rel_ids, 0, num_rels - 1)

        # 提取实体和关系的嵌入（使用裁剪后的 ID）
        s_emb = entity_emb[entity_ids]  # [batch_size, K, max_seq_len, h_dim]
        r_emb = relation_emb[rel_ids]  # [batch_size, K, max_seq_len, h_dim]

        # 计算时间差并编码
        # 确保 paths[..., 3:4] 是 float 类型
        path_times = paths[..., 3:4].float()  # [batch_size, K, max_seq_len, 1]
        time_diffs = query_times.view(batch_size, 1, 1, 1).to(device) - path_times  # [batch_size, K, max_seq_len, 1]
        time_emb = self.time_encoder(time_diffs)  # [batch_size, K, max_seq_len, h_dim]

        # 融合关系嵌入和时间编码
        rel_time_emb = r_emb + time_emb  # [batch_size, K, max_seq_len, h_dim]

        # 构建序列输入：实体嵌入 + 关系-时间嵌入
        seq_input = torch.cat([s_emb, rel_time_emb], dim=-1)  # [batch_size, K, max_seq_len, h_dim * 2]

        # 2. 单链路序列建模
        # 调整形状为 [batch_size * K, max_seq_len, h_dim * 2]
        seq_input = seq_input.view(-1, self.max_seq_len, self.h_dim * 2)
        path_lengths_flat = path_lengths.view(-1)  # [batch_size * K]

        # 处理长度为 0 的路径
        valid_mask = path_lengths_flat > 0
        if valid_mask.sum() == 0:
            # 所有路径长度都为 0，返回零向量
            return torch.zeros(batch_size, self.h_dim, device=device)

        # 只处理长度大于 0 的路径
        valid_seq_input = seq_input[valid_mask]
        valid_lengths = path_lengths_flat[valid_mask]

        # 打包变长序列，lengths 参数需要在 CPU 上
        packed_input = nn.utils.rnn.pack_padded_sequence(
            valid_seq_input,
            valid_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # GRU 前向传播
        _, h_n = self.gru(packed_input)  # h_n: [1, valid_count, h_dim]

        # 构建路径嵌入，长度为 0 的路径使用零向量
        path_emb = torch.zeros(batch_size * self.K, self.h_dim, device=device)
        path_emb[valid_mask] = h_n.squeeze(0)
        path_emb = path_emb.view(batch_size, self.K, self.h_dim)  # [batch_size, K, h_dim]

        # 3. 多链路注意力聚合
        # 使用主体实体特征作为 Query
        # 提取每条路径的起始实体嵌入作为查询
        query_emb = s_emb[:, :, 0, :]  # [batch_size, K, h_dim]

        # 计算注意力分数
        attn_input = torch.cat([path_emb, query_emb], dim=-1)  # [batch_size, K, h_dim * 2]
        attn_scores = self.attention(attn_input).squeeze(-1)  # [batch_size, K]

        # 处理填充路径的注意力分数
        mask = (path_lengths == 0).float()  # [batch_size, K]
        attn_scores = attn_scores.masked_fill(mask.bool(), -1e9)  # 填充路径的注意力分数设为极小值

        # 归一化注意力分数
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # [batch_size, K, 1]

        # 加权求和得到全局逻辑表示
        global_emb = torch.sum(path_emb * attn_weights, dim=1)  # [batch_size, h_dim]

        return global_emb


def process_paths(batch_paths, K=50, max_seq_len=10, device='cpu', num_ents=None, num_rels=None):
    """
    处理路径数据，进行 Padding 和生成 Attention Mask

    Args:
        batch_paths: 当前批次的路径列表，每个元素是一个路径列表
                     格式: batch_paths[i] = [(s1,r1,o1,t1), (s2,r2,o2,t2), ...] 表示一条路径
                     batch_paths[i] 应该是一个列表，包含 K 条路径
        K: 每条查询的路径数量
        max_seq_len: 路径的最大长度
        device: 目标设备
        num_ents: 实体数量，用于边界检查（可选）
        num_rels: 关系数量，用于边界检查（可选）

    Returns:
        paths_tensor: 路径张量，形状为 [batch_size, K, max_seq_len, 4]
        path_lengths: 路径长度张量，形状为 [batch_size, K]
    """
    # 处理 batch_paths 为 None 或空的情况
    if batch_paths is None or len(batch_paths) == 0:
        batch_size = 1
        paths_tensor = torch.zeros(batch_size, K, max_seq_len, 4, dtype=torch.long, device=device)
        path_lengths = torch.zeros(batch_size, K, dtype=torch.long, device=device)
        return paths_tensor, path_lengths

    batch_size = len(batch_paths)

    # 初始化张量
    paths_tensor = torch.zeros(batch_size, K, max_seq_len, 4, dtype=torch.long, device=device)
    path_lengths = torch.zeros(batch_size, K, dtype=torch.long, device=device)

    for i in range(batch_size):
        # 获取当前查询的路径列表
        query_paths = batch_paths[i]

        # 处理 query_paths 为 None 或空的情况
        if query_paths is None:
            continue

        # 确保 query_paths 是一个列表
        if not isinstance(query_paths, list):
            # 如果 query_paths 不是列表，可能它本身就是一条路径
            # 将其包装成一个列表
            query_paths = [query_paths]

        # 取前 K 条路径
        selected_paths = query_paths[:K]

        for j, path in enumerate(selected_paths):
            # 处理 path 为 None 的情况
            if path is None:
                continue

            # 确保 path 是一个列表
            if not isinstance(path, list):
                continue

            # 限制路径长度
            path = path[:max_seq_len]
            path_len = len(path)
            path_lengths[i, j] = path_len

            # 填充路径
            for k, edge in enumerate(path):
                # 确保 edge 是一个长度为 4 的元组/列表
                if edge is not None and len(edge) == 4:
                    s, r, o, t = edge
                    # 关键修复：添加边界检查，防止无效 ID
                    try:
                        s_int = int(s)
                        r_int = int(r)
                        o_int = int(o)
                        t_int = int(t)

                        # 如果提供了 num_ents 和 num_rels，进行边界检查
                        if num_ents is not None:
                            s_int = min(s_int, num_ents - 1)
                            o_int = min(o_int, num_ents - 1)
                        if num_rels is not None:
                            r_int = min(r_int, num_rels - 1)

                        # 确保非负
                        s_int = max(0, s_int)
                        r_int = max(0, r_int)
                        o_int = max(0, o_int)
                        t_int = max(0, t_int)

                        paths_tensor[i, j, k, 0] = s_int
                        paths_tensor[i, j, k, 1] = r_int
                        paths_tensor[i, j, k, 2] = o_int
                        paths_tensor[i, j, k, 3] = t_int
                    except (ValueError, TypeError):
                        # 如果转换失败，使用默认值 0
                        continue

    return paths_tensor, path_lengths
