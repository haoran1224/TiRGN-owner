import os

import numpy as np
import torch
import dgl
from tqdm import tqdm
import rgcn.knowledge_graph as knwlgrh
from collections import defaultdict


#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


#TODO filer by groud truth in the same time snapshot not all ground truth
def sort_and_rank_time_filter(batch_a, batch_r, score, target, total_triplets):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    for i in range(len(batch_a)):
        ground = indices[i]
    indices = indices[:, 1].view(-1)
    return indices


def sort_and_rank_filter(batch_a, batch_r, score, target, all_ans):
    for i in range(len(batch_a)):
        ans = target[i]
        b_multi = list(all_ans[batch_a[i].item()][batch_r[i].item()])
        ground = score[i][ans]
        score[i][b_multi] = 0
        score[i][ans] = ground
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def filter_score(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t, no_use = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score

def filter_score_r(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t, no_use = triple
        ans = list(all_ans[h.item()][t.item()])
        ans.remove(r.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score


def r2e(triplets, num_rels):
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_r = np.unique(rel)
    uniq_r = np.concatenate((uniq_r, uniq_r+num_rels))
    # generate r2e
    r_to_e = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel+num_rels].add(src)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx,idx+len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx


def build_sub_graph(num_nodes, num_rels, triples, use_cuda, gpu):
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm
    # print(triples.shape)
    triples = triples[:, :3]
    src, rel, dst = triples.transpose()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)

    uniq_r, r_len, r_to_e = r2e(triples, num_rels)
    g.uniq_r = uniq_r
    g.r_to_e = r_to_e
    g.r_len = r_len
    if use_cuda:
        g.to(gpu)
        g.r_to_e = torch.from_numpy(np.array(r_to_e)).long()
    return g

def get_total_rank(test_triples, score, all_ans, eval_bz, rel_predict=0):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank = []
    filter_rank = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]
        if rel_predict==1:
            target = test_triples[batch_start:batch_end, 1]
        elif rel_predict == 2:
            target = test_triples[batch_start:batch_end, 0]
        else:
            target = test_triples[batch_start:batch_end, 2]
        rank.append(sort_and_rank(score_batch, target))

        if rel_predict:
            filter_score_batch = filter_score_r(triples_batch, score_batch, all_ans)
        else:
            filter_score_batch = filter_score(triples_batch, score_batch, all_ans)
        filter_rank.append(sort_and_rank(filter_score_batch, target))

    rank = torch.cat(rank)
    filter_rank = torch.cat(filter_rank)
    rank += 1 # change to 1-indexed
    filter_rank += 1
    mrr = torch.mean(1.0 / rank.float())
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    return filter_mrr.item(), mrr.item(), rank, filter_rank


def stat_ranks(rank_list, method):
    hits = [1, 3, 10]
    total_rank = torch.cat(rank_list)

    mrr = torch.mean(1.0 / total_rank.float())
    print("MRR ({}): {:.6f}".format(method, mrr.item()))
    hit_result = []
    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float())
        hit_result.append(avg_count)
        print("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
    return mrr, hit_result


def flatten(l):
    flatten_l = []
    for c in l:
        if type(c) is list or type(c) is tuple:
            flatten_l.extend(flatten(c))
        else:
            flatten_l.append(c)
    return flatten_l

def UnionFindSet(m, edges):
    roots = [i for i in range(m)]
    rank = [0 for i in range(m)]
    count = m

    def find(member):
        tmp = []
        while member != roots[member]:
            tmp.append(member)
            member = roots[member]
        for root in tmp:
            roots[root] = member
        return member

    for i in range(m):
        roots[i] = i
    # print ufs.roots
    for edge in edges:
        print(edge)
        start, end = edge[0], edge[1]
        parentP = find(start)
        parentQ = find(end)
        if parentP != parentQ:
            if rank[parentP] > rank[parentQ]:
                roots[parentQ] = parentP
            elif rank[parentP] < rank[parentQ]:
                roots[parentP] = parentQ
            else:
                roots[parentQ] = parentP
                rank[parentP] -= 1
            count -= 1
    return count

def append_object(e1, e2, r, d):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def add_subject(e1, e2, r, d, num_rel):
    if not e2 in d:
        d[e2] = {}
    if not r+num_rel in d[e2]:
        d[e2][r+num_rel] = set()
    d[e2][r+num_rel].add(e1)


def add_object(e1, e2, r, d, num_rel):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def load_all_answers(total_data, num_rel):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    all_subjects, all_objects = {}, {}
    for line in total_data:
        s, r, o = line[: 3]
        add_subject(s, o, r, all_subjects, num_rel=num_rel)
        add_object(s, o, r, all_objects, num_rel=0)
    return all_objects, all_subjects


def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    def add_relation(e1, e2, r, d):
        if not e1 in d:
            d[e1] = {}
        if not e2 in d[e1]:
            d[e1][e2] = set()
        d[e1][e2].add(r)

    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        if rel_p:
            add_relation(s, o, r, all_ans)
            add_relation(o, s, r + num_rel, all_ans)
        else:
            add_subject(s, o, r, all_ans, num_rel=num_rel)
            add_object(s, o, r, all_ans, num_rel=0)
    return all_ans


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_list = []
    all_snap, nouse = split_by_time(total_data)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)
        all_ans_list.append(all_ans_t)
    return all_ans_list

def split_by_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        if latest_t != t:
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:])
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # relabel
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    times = set()
    for triple in data:
        times.add(triple[3])
    times = list(times)
    times.sort()
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    return snapshot_list, np.asarray(times)


def slide_list(snapshots, k=1):
    k = k
    if k > len(snapshots):
        print("ERROR: history length exceed the length of snapshot: {}>{}".format(k, len(snapshots)))
    for _ in tqdm(range(len(snapshots)-k+1)):
        yield snapshots[_: _+k]



def load_data(dataset, bfs_level=3, relabel=False):
    if dataset in ['aifb', 'mutag', 'bgs', 'am']:
        return knwlgrh.load_entity(dataset, bfs_level, relabel)
    elif dataset in ['FB15k', 'wn18', 'FB15k-237']:
        return knwlgrh.load_link(dataset)
    elif dataset in ['ICEWS18', 'ICEWS14', "GDELT", "SMALL", "ICEWS14s", "ICEWS05-15","YAGO",
                     "WIKI"]:
        return knwlgrh.load_from_local("../data", dataset)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

def construct_snap(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, r = test_triples[_][0], test_triples[_][1]
            if r < num_rels:
                predict_triples.append([test_triples[_][0], r, index, test_triples[_][3]])
            else:
                predict_triples.append([index, r-num_rels, test_triples[_][0], test_triples[_][3]])

    # 转化为numpy array
    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples

def construct_snap_r(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []

    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, t = test_triples[_][0], test_triples[_][2]
            if index < num_rels:
                predict_triples.append([h, index, t])
                #predict_triples.append([t, index+num_rels, h])
            else:
                predict_triples.append([t, index-num_rels, h])
                #predict_triples.append([t, index-num_rels, h])

    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples


def dilate_input(input_list, dilate_len):
    dilate_temp = []
    dilate_input_list = []
    for i in range(len(input_list)):
        if i % dilate_len == 0 and i:
            if len(dilate_temp):
                dilate_input_list.append(dilate_temp)
                dilate_temp = []
        if len(dilate_temp):
            dilate_temp = np.concatenate((dilate_temp, input_list[i]))
        else:
            dilate_temp = input_list[i]
    dilate_input_list.append(dilate_temp)
    dilate_input_list = [np.unique(_, axis=0) for _ in dilate_input_list]
    return dilate_input_list

def emb_norm(emb, epo=0.00001):
    x_norm = torch.sqrt(torch.sum(emb.pow(2), dim=1))+epo
    emb = emb/x_norm.view(-1,1)
    return emb

def shuffle(data, labels):
    shuffle_idx = np.arange(len(data))
    np.random.shuffle(shuffle_idx)
    relabel_output = data[shuffle_idx]
    labels = labels[shuffle_idx]
    return relabel_output, labels


def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor


def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.sum(t)
    return a

import pickle
# 1. 辅助函数：根据 Triples 的切分方式，切分 Paths
def align_paths_with_snapshots(triples_list, flat_paths):
    """
    triples_list: TiRGN split_by_time 生成的 train_list (List of numpy arrays)
    flat_paths: 你加载的那个巨大的 train_paths (List)
    """
    path_snapshots = []
    start_idx = 0

    print("Aligning paths with snapshots...")
    for snap in triples_list:
        # 获取当前时间步的样本数量
        num_samples = len(snap)
        end_idx = start_idx + num_samples

        # 切片
        current_paths = flat_paths[start_idx: end_idx]
        path_snapshots.append(current_paths)

        start_idx = end_idx

    assert start_idx == len(flat_paths) / 2, "Error: Path list length does not match Triples length!"
    return path_snapshots


def process_batch_paths(batch_path_data, num_rels, max_len=3, max_paths=50, use_cuda=True, gpu_id=0):
    """
    将 List 格式的路径转换为 Tensor。
    batch_path_data: 当前时间步及 batch 下的路径列表，长度为 Batch_Size。
                     每个元素是 [ [s,r,o,t], [s,r,o,t]... ] (即该实体的 Top K 条路径)

    Returns:
        path_rels: (Batch, TopK, Max_Len) - 存储关系 ID
        path_times: (Batch, TopK, Max_Len) - 存储时间 (或 mask)
    """
    if batch_path_data is None:
        return None, None

    batch_size = len(batch_path_data)

    # 初始化 Tensor (全部填充为 0 或特定的 Padding ID)
    # 假设 0 是 padding id (通常关系ID从0开始，建议用 num_rels 或 -1 做 padding，这里为了简单用 num_rels)
    padding_val = num_rels*2

    # path_rels: 存储路径上的关系
    path_rels = torch.full((batch_size, max_paths, max_len), padding_val, dtype=torch.long)
    # path_times: 存储路径上的时间 (用于计算衰减)
    path_times = torch.zeros((batch_size, max_paths), dtype=torch.float)
    # 注意：CRAFT 的 Attention 只需要路径的最早时间 (t_ear) 或者每一步的时间？
    # 根据之前的 CRAFT 逻辑，我们需要 Top-K 路径的 "最早发生时间" 或 "每一步时间"。
    # 假设你的数据里 [s,r,o,t] 的 t 是这一步的时间。
    # 这里我们只取每条路径的第一步时间作为 t_ear (用于时间衰减)。
    path_masks = torch.zeros((batch_size, max_paths), dtype=torch.float)

    for i, paths_list in enumerate(batch_path_data):
        # paths_list 是当前样本的所有历史路径 (最多50条)
        # 截断或填充到 max_paths
        curr_paths = paths_list[:max_paths]

        for j, path in enumerate(curr_paths):
            # path 是一个列表: [[s1, r1, o1, t1], [s2, r2, o2, t2], ...]
            # 1. 提取时间 (取路径第一跳的时间作为 t_ear)
            if len(path) > 0:
                path_times[i, j] = float(path[0][3])
                path_masks[i, j] = 1.0

            # 2. 提取关系链并填充到 max_len
            for k, quad in enumerate(path):
                if k >= max_len: break
                # quad: [s, r, o, t] -> 取 r (索引1)
                r_id = quad[1]
                path_rels[i, j, k] = r_id

    if use_cuda:
        device = torch.device(gpu_id)
        path_rels = path_rels.to(device)
        path_times = path_times.to(device)
        path_masks = path_masks.to(device)

    return path_rels, path_times, path_masks


def align_data_to_path(path_file_path, data_list):
    if os.path.exists(path_file_path):
        print(f"Loading paths from {path_file_path}...")
        with open(path_file_path, "rb") as f:
            train_paths_flat = pickle.load(f)

        # 核心步骤：切分路径数据以对齐 train_list
        train_path_snaps = align_paths_with_snapshots(data_list, train_paths_flat)
        print(f"Paths aligned. Total snapshots: {len(train_path_snaps)}")
    else:
        print("Warning: Path file not found!")
        train_path_snaps = [None] * len(data_list)

    return train_path_snaps


import pickle
import dgl
import torch


def build_global_graph_from_paths(batch_paths, num_ents, num_rels=None, add_inverse=True, use_cuda=False):
    """
    将 TLogic 检索到的当前 batch 的历史路径转换为 DGL 全局子图。

    参数:
    batch_paths (list): 当前 batch 对应的路径列表。
                        格式: [[path1, path2], [], [path3], ...]
                        其中 path = [[s, r, o, t], ...]
    num_ents (int):     知识图谱中的实体总数。
    num_rels (int):     知识图谱中的关系总数（仅在 add_inverse=True 时需要）。
    add_inverse (bool): 是否为每条边添加反向关系边 (o, r + num_rels, s)。
                        TiRGN 的 R-GCN 层通常期望输入包含反向边的双向图。
    use_cuda (bool):    是否将图放到 GPU 上。

    返回:
    g (dgl.DGLGraph):   构建好的全局子图。
    """
    src_list = []
    rel_list = []
    dst_list = []

    # 使用 set 去重，避免同一个事实 (s, r, o) 在图中出现多次导致权重翻倍
    unique_edges = set()

    # 1. 解析 batch 内的所有路径
    for query_paths in batch_paths:
        if not query_paths:  # 如果该 query 列表为空（没有检索到历史路径）
            continue

        for path in query_paths:
            for edge in path:
                # 解析单条边 [s, r, o, t]
                s, r, o, t = edge
                unique_edges.add((s, r, o))

                # 如果 TiRGN 的 RGCN 层(num_rels * 2)需要反向边，在此处统一添加
                if add_inverse and num_rels is not None:
                    unique_edges.add((o, r + num_rels, s))

    # 2. 处理 batch 内完全没有历史路径的极端情况
    if len(unique_edges) == 0:
        # 构造一个无边的空图，避免 DGL 在前向传播时报错
        g = dgl.graph(([], []), num_nodes=num_ents)
        g.edata['type'] = torch.tensor([], dtype=torch.long)
        g.ndata['norm'] = torch.ones(num_ents, 1)
        if use_cuda:
            g = g.to('cuda')
        return g

    # 3. 将去重后的边解包
    for s, r, o in unique_edges:
        src_list.append(s)
        rel_list.append(r)
        dst_list.append(o)

    # 转换为 Tensor
    src_tensor = torch.tensor(src_list, dtype=torch.long)
    dst_tensor = torch.tensor(dst_list, dtype=torch.long)
    rel_tensor = torch.tensor(rel_list, dtype=torch.long)

    # 4. 构建 DGL 图
    g = dgl.graph((src_tensor, dst_tensor), num_nodes=num_ents)
    g.edata['type'] = rel_tensor

    # 5. 计算 R-GCN 需要的归一化常数 (norm = 1 / in_degree)
    # clamp(min=1) 防止孤立节点的入度为 0 导致除零错误
    in_deg = g.in_degrees().float().clamp(min=1)
    norm = 1.0 / in_deg
    g.ndata['norm'] = norm.unsqueeze(1)  # shape: (num_ents, 1)

    if use_cuda:
        g = g.to('cuda')

    return g


# ==========================================
# 测试与使用示例 (Dummy Example)
# ==========================================
if __name__ == "__main__":
    # 假设你已经通过 TLogic 生成了 train_paths_top50.pkl
    # 模拟读取 pickle 文件的过程
    """
    with open('train_paths_top50.pkl', 'rb') as f:
        all_train_paths = pickle.load(f)
    """

    # 模拟 all_train_paths 中的一个 Batch (假设 batch_size = 3)
    # 路径格式: [s, r, o, t]
    mock_batch_paths = [
        # Query 0 检索到的两条路径
        [
            [[10, 1, 20, 100], [20, 2, 30, 101]],
            [[10, 3, 30, 100]]
        ],
        # Query 1 没有检索到任何路径
        [],
        # Query 2 检索到的一条路径 (注意里面有一条边与 Query 0 重复了)
        [
            [[40, 4, 50, 102], [10, 1, 20, 100]]
        ]
    ]

    num_entities = 100  # 假设数据集中有 100 个实体
    num_relations = 10  # 假设数据集中有 10 个正向关系

    # 生成 DGL Global Graph
    global_g = build_global_graph_from_paths(
        batch_paths=mock_batch_paths,
        num_ents=num_entities,
        num_rels=num_relations,
        add_inverse=True,  # 为 TiRGN 添加反向边
        use_cuda=False
    )

    print(f"全局子图节点数: {global_g.num_nodes()}")
    print(f"全局子图边数 (包含反向边): {global_g.num_edges()}")
    print(f"节点归一化常数 Shape: {global_g.ndata['norm'].shape}")
    print(f"边类型 Shape: {global_g.edata['type'].shape}")