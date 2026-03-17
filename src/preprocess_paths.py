"""
TiRGN 路径数据离线预处理脚本

该脚本用于将原始的 .pkl 路径文件预处理为规整的 PyTorch Tensor 格式，
包括：
1. 正向路径的 Padding 和张量化
2. 逆向路径的生成和张量化
3. 保存为 .pt 文件供训练时直接加载

目的：消除训练时的 CPU 瓶颈，提高 GPU 利用率
"""

import os
import pickle
import torch
import numpy as np
from tqdm import tqdm


def get_inverse_paths_single(path, num_rels):
    """
    为单条路径生成逆向路径

    Args:
        path: 正向路径，列表形式，每个元素为 (s, r, o, t)
        num_rels: 正向关系的数量

    Returns:
        inverse_path: 逆向路径
    """
    inverse_path = []
    for edge in reversed(path):
        s, r, o, t = edge
        # 翻转关系：正向关系 -> 逆向关系，逆向关系 -> 正向关系
        if r < num_rels:
            inverse_r = r + num_rels
        else:
            inverse_r = r - num_rels
        # 构建逆向 edge: (o, inverse_r, s, t)
        inverse_edge = (o, inverse_r, s, t)
        inverse_path.append(inverse_edge)
    return inverse_path


def process_single_path(path, max_seq_len=10, num_ents=None, num_rels=None):
    """
    处理单条路径，进行截断和边界检查

    Args:
        path: 单条路径，列表形式
        max_seq_len: 路径的最大长度
        num_ents: 实体数量，用于边界检查
        num_rels: 关系数量，用于边界检查

    Returns:
        processed_path: 处理后的路径列表
    """
    # 截断路径
    path = path[:max_seq_len]

    processed_path = []
    for edge in path:
        if edge is None or len(edge) != 4:
            continue

        s, r, o, t = edge
        try:
            s_int = int(s)
            r_int = int(r)
            o_int = int(o)
            t_int = int(t)

            # 边界检查
            if num_ents is not None:
                s_int = max(0, min(s_int, num_ents - 1))
                o_int = max(0, min(o_int, num_ents - 1))
            if num_rels is not None:
                r_int = max(0, min(r_int, num_rels * 2 - 1))

            # 确保非负
            s_int = max(0, s_int)
            r_int = max(0, r_int)
            o_int = max(0, o_int)
            t_int = max(0, t_int)

            processed_path.append((s_int, r_int, o_int, t_int))
        except (ValueError, TypeError):
            continue

    return processed_path


def preprocess_paths_to_tensor(all_paths, num_rels, K=50, max_seq_len=10, num_ents=None):
    """
    将路径列表预处理为规整的 PyTorch Tensor

    Args:
        all_paths: 所有查询的路径列表，每个元素是一个查询的路径列表
                   all_paths[i] = [(s1,r1,o1,t1), (s2,r2,o2,t2), ...] 表示一条路径
        num_rels: 正向关系的数量
        K: 每条查询的路径数量
        max_seq_len: 路径的最大长度
        num_ents: 实体数量，用于边界检查

    Returns:
        fwd_paths_tensor: 正向路径张量，形状 [N, K, max_seq_len, 4]
        fwd_lens: 正向路径长度张量，形状 [N, K]
        inv_paths_tensor: 逆向路径张量，形状 [N, K, max_seq_len, 4]
        inv_lens: 逆向路径长度张量，形状 [N, K]
    """
    N = len(all_paths)

    # 初始化张量
    fwd_paths_tensor = torch.zeros(N, K, max_seq_len, 4, dtype=torch.long)
    fwd_lens = torch.zeros(N, K, dtype=torch.long)

    inv_paths_tensor = torch.zeros(N, K, max_seq_len, 4, dtype=torch.long)
    inv_lens = torch.zeros(N, K, dtype=torch.long)

    for i in tqdm(range(N), desc="Processing paths"):
        query_paths = all_paths[i]

        # 处理空路径情况
        if query_paths is None:
            continue

        # 确保是列表
        if not isinstance(query_paths, list):
            query_paths = [query_paths]

        # 取前 K 条路径
        selected_paths = query_paths[:K]

        for j, path in enumerate(selected_paths):
            if path is None:
                continue

            # 确保是列表
            if not isinstance(path, list):
                continue

            # 处理正向路径
            processed_path = process_single_path(path, max_seq_len, num_ents, num_rels)
            path_len = len(processed_path)
            fwd_lens[i, j] = path_len

            # 填充正向路径
            for k, edge in enumerate(processed_path):
                fwd_paths_tensor[i, j, k, 0] = edge[0]  # s
                fwd_paths_tensor[i, j, k, 1] = edge[1]  # r
                fwd_paths_tensor[i, j, k, 2] = edge[2]  # o
                fwd_paths_tensor[i, j, k, 3] = edge[3]  # t

            # 生成并处理逆向路径
            inverse_path = get_inverse_paths_single(processed_path, num_rels)
            inv_path_len = len(inverse_path)
            inv_lens[i, j] = inv_path_len

            # 填充逆向路径
            for k, edge in enumerate(inverse_path):
                if k < max_seq_len:
                    inv_paths_tensor[i, j, k, 0] = edge[0]  # o -> s
                    inv_paths_tensor[i, j, k, 1] = edge[1]  # inverse_r
                    inv_paths_tensor[i, j, k, 2] = edge[2]  # s -> o
                    inv_paths_tensor[i, j, k, 3] = edge[3]  # t

    return fwd_paths_tensor, fwd_lens, inv_paths_tensor, inv_lens


def load_dataset_info(dataset_path):
    """
    从数据集中加载实体和关系数量信息

    文件格式：每行为 "名称\tID"
    例如: "Sign formal agreement\t0"

    Args:
        dataset_path: 数据集路径

    Returns:
        num_ents: 实体数量（最大ID + 1）
        num_rels: 关系数量（最大ID + 1）
    """
    # 从 entity2id.txt 读取实体数量
    num_ents = None
    entity2id_path = os.path.join(dataset_path, "entity2id.txt")
    if os.path.exists(entity2id_path):
        with open(entity2id_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            max_id = -1
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # 分割并获取ID部分
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        idx = int(parts[-1].strip())
                        max_id = max(max_id, idx)
                    except (ValueError, IndexError):
                        continue
            if max_id >= 0:
                num_ents = max_id + 1

    # 从 relation2id.txt 读取关系数量
    num_rels = None
    relation2id_path = os.path.join(dataset_path, "relation2id.txt")
    if os.path.exists(relation2id_path):
        with open(relation2id_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            max_id = -1
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # 分割并获取ID部分
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        idx = int(parts[-1].strip())
                        max_id = max(max_id, idx)
                    except (ValueError, IndexError):
                        continue
            if max_id >= 0:
                num_rels = max_id + 1

    return num_ents, num_rels


def main():
    """
    主函数：处理指定数据集的路径文件
    """
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess paths data for TiRGN')
    parser.add_argument('--dataset', type=str, default='ICEWS14',
                        help='Dataset name (e.g., ICEWS14, ICEWS14s, WIKI, YAGO)')
    parser.add_argument('--data_root', type=str,default='../data',
                        help='Root directory for data')
    parser.add_argument('--K', type=int, default=50,
                        help='Number of paths per query')
    parser.add_argument('--max_seq_len', type=int, default=3,
                        help='Maximum path sequence length')

    args = parser.parse_args()

    dataset_path = os.path.join(args.data_root, args.dataset)

    print(f"Processing dataset: {args.dataset}")
    print(f"Data path: {dataset_path}")

    # 加载数据集信息
    print("Loading dataset info...")
    num_ents, num_rels = load_dataset_info(dataset_path)
    print(f"  num_ents: {num_ents}, num_rels: {num_rels}")

    if num_rels is None:
        raise ValueError("Cannot determine num_rels from dataset. Please check relation2id.txt")

    # 处理训练集路径
    print("\nProcessing train paths...")
    train_path_file = os.path.join(dataset_path, 'train_paths_top50.pkl')
    if os.path.exists(train_path_file):
        with open(train_path_file, 'rb') as f:
            all_train_paths = pickle.load(f)

        fwd_train, lens_train, inv_train, inv_lens_train = preprocess_paths_to_tensor(
            all_train_paths, num_rels, K=args.K, max_seq_len=args.max_seq_len, num_ents=num_ents
        )

        # 保存训练集张量
        torch.save(fwd_train, os.path.join(dataset_path, 'train_fwd_paths.pt'))
        torch.save(lens_train, os.path.join(dataset_path, 'train_fwd_lens.pt'))
        torch.save(inv_train, os.path.join(dataset_path, 'train_inv_paths.pt'))
        torch.save(inv_lens_train, os.path.join(dataset_path, 'train_inv_lens.pt'))
        print(f"  Train paths shape: {fwd_train.shape}")
        print(f"  Train lens shape: {lens_train.shape}")
        print(f"  Saved train tensors to {dataset_path}")
    else:
        print(f"  Warning: {train_path_file} not found, skipping...")

    # 处理验证集路径
    print("\nProcessing valid paths...")
    valid_path_file = os.path.join(dataset_path, 'valid_paths_top50.pkl')
    if os.path.exists(valid_path_file):
        with open(valid_path_file, 'rb') as f:
            all_valid_paths = pickle.load(f)

        fwd_valid, lens_valid, inv_valid, inv_lens_valid = preprocess_paths_to_tensor(
            all_valid_paths, num_rels, K=args.K, max_seq_len=args.max_seq_len, num_ents=num_ents
        )

        # 保存验证集张量
        torch.save(fwd_valid, os.path.join(dataset_path, 'valid_fwd_paths.pt'))
        torch.save(lens_valid, os.path.join(dataset_path, 'valid_fwd_lens.pt'))
        torch.save(inv_valid, os.path.join(dataset_path, 'valid_inv_paths.pt'))
        torch.save(inv_lens_valid, os.path.join(dataset_path, 'valid_inv_lens.pt'))
        print(f"  Valid paths shape: {fwd_valid.shape}")
        print(f"  Valid lens shape: {lens_valid.shape}")
        print(f"  Saved valid tensors to {dataset_path}")
    else:
        print(f"  Warning: {valid_path_file} not found, skipping...")

    # 处理测试集路径
    print("\nProcessing test paths...")
    test_path_file = os.path.join(dataset_path, 'test_paths_top50.pkl')
    if os.path.exists(test_path_file):
        with open(test_path_file, 'rb') as f:
            all_test_paths = pickle.load(f)

        fwd_test, lens_test, inv_test, inv_lens_test = preprocess_paths_to_tensor(
            all_test_paths, num_rels, K=args.K, max_seq_len=args.max_seq_len, num_ents=num_ents
        )

        # 保存测试集张量
        torch.save(fwd_test, os.path.join(dataset_path, 'test_fwd_paths.pt'))
        torch.save(lens_test, os.path.join(dataset_path, 'test_fwd_lens.pt'))
        torch.save(inv_test, os.path.join(dataset_path, 'test_inv_paths.pt'))
        torch.save(inv_lens_test, os.path.join(dataset_path, 'test_inv_lens.pt'))
        print(f"  Test paths shape: {fwd_test.shape}")
        print(f"  Test lens shape: {lens_test.shape}")
        print(f"  Saved test tensors to {dataset_path}")
    else:
        print(f"  Warning: {test_path_file} not found, skipping...")

    print("\nPreprocessing completed!")


if __name__ == '__main__':
    main()
