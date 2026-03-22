import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder import *
from src.fusion import LocalGlobalFusion
from src.path_encoder import PathEncoder, process_paths


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "convgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "convgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')



class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, num_times, time_interval, h_dim, opn, history_rate, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0, analysis=False, K=50, max_seq_len=3):
        """
        RecurrentRGCN 模型，集成路径编码器和局部-全局融合模块

        Args:
            K: 每条查询的路径数量，默认为 50
            max_seq_len: 路径的最大长度，默认为 3
        """
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.history_rate = history_rate
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.num_times = num_times
        self.time_interval = time_interval
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu
        self.sin = torch.sin
        self.linear_0 = nn.Linear(num_times, 1)
        self.linear_1 = nn.Linear(num_times, self.h_dim - 1)
        self.tanh = nn.Tanh()
        self.use_cuda = None

        # ==================== 新增：路径编码器和融合模块参数 ====================
        self.K = K  # 每条查询的路径数量
        self.max_seq_len = max_seq_len  # 路径的最大长度
        # ====================================================================

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)
        

        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))


        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        # add
        self.global_weight = nn.Parameter(torch.Tensor(self.num_ents, 1))
        nn.init.xavier_uniform_(self.global_weight , gain=nn.init.calculate_gain('relu'))
        self.global_bias = nn.Parameter(torch.Tensor(1))
        nn.init.zeros_(self.global_bias)

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)
        self.entity_cell_1 = nn.GRUCell(self.h_dim, self.h_dim)

        # decoder
        if decoder_name == "timeconvtranse":
            self.decoder_ob1 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob2 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re1 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re2 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob3 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout,feat_dropout)  # <--【新增 1-hop 解码器】
        else:
            raise NotImplementedError

        # SRM-LLM Section 4.3: Rule-Based Historical Relation Retrieval Module
        # =================================================================
        # 1. 融合权重 \lambda (论文中表示为 \lambda，可学习或固定，论文默认设为0.9)
        self.lamb = nn.Parameter(torch.tensor(0.9), requires_grad=True)

        # 2. 全局逻辑子图的 R-GCN 聚合器 (对应论文公式5)
        # 这里复用 TiRGN 已有的 RGCNBlockLayer 或 UnionRGCNLayer
        self.global_rgcn_layer = RGCNBlockLayer(
            self.h_dim, self.h_dim, self.num_rels * 2, num_bases,
            activation=F.rrelu, dropout=dropout, self_loop=True, skip_connect=False
        )

        # ==================== 新增：路径编码器模块 ====================
        # 路径编码器：将历史路径编码为全局逻辑表示
        self.path_encoder = PathEncoder(
            num_ents=num_ents,
            num_rels=num_rels,
            h_dim=h_dim,
            K=K,
            max_seq_len=max_seq_len
        )
        # =============================================================

        # ==================== 新增：局部-全局融合模块 ====================
        # 局部-全局融合模块：融合局部演化表示和全局路径表示
        self.local_global_fusion = LocalGlobalFusion(
            h_dim=h_dim,
            dropout=dropout,
            use_layer_norm=layer_norm
        )
        # =============================================================

    def forward(self, g_list, static_graph, use_cuda):
        gate_list = []
        degree_list = []

        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_embs = []

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            self.h = self.entity_cell_1(current_h, self.h)
            self.h = F.normalize(self.h) if self.layer_norm else self.h
            history_embs.append(self.h)
        return history_embs, static_emb, self.h_0, gate_list, degree_list


    def predict(self, test_graph, num_rels, static_graph, test_triplets, entity_history_vocabulary, rel_history_vocabulary, entity_local_vocabulary, fwd_paths_tensor, fwd_lens, inv_paths_tensor, inv_lens, use_cuda):
        """
        预测函数，集成路径编码器和局部-全局融合模块（离线预处理版本）

        Args:
            fwd_paths_tensor: 正向路径张量，形状 [batch_size, K, max_seq_len, 4]
            fwd_lens: 正向路径长度张量，形状 [batch_size, K]
            inv_paths_tensor: 逆向路径张量，形状 [batch_size, K, max_seq_len, 4]
            inv_lens: 逆向路径长度张量，形状 [batch_size, K]
        """
        self.use_cuda = use_cuda
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0, 3]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets))

            evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda)
            # 提取局部演化表示 E_{t_q}^L
            local_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

            # ==================== 离线预处理版本：直接使用预处理的张量 ====================
            device = self.gpu if use_cuda else 'cpu'

            # 检查是否有路径数据
            if fwd_paths_tensor is None or len(fwd_paths_tensor) == 0:
                # 如果没有路径数据，直接使用局部嵌入
                embedding = local_emb
            else:
                batch_size = all_triples.size(0)

                # 直接在 batch 维度拼接正向和逆向张量
                paths_tensor = torch.cat([fwd_paths_tensor, inv_paths_tensor], dim=0).to(device)
                path_lengths = torch.cat([fwd_lens, inv_lens], dim=0).to(device)

                # 拼接后的维度应该对应 all_triples 的 [batch_size * 2] 维度
                query_times = all_triples[:, 3].float()  # [batch_size * 2]
                query_entity_ids = all_triples[:, 0]     # [batch_size * 2]
                local_emb_per_query = local_emb[query_entity_ids]  # [batch_size * 2, h_dim]

                try:
                    # 使用路径编码器
                    path_global_emb = self.path_encoder(
                        paths_tensor, path_lengths, query_times,
                        local_emb, self.emb_rel
                    )  # [batch_size * 2, h_dim]

                    # 使用局部-全局融合模块融合表示
                    final_emb = self.local_global_fusion(
                        local_emb_per_query,  # [batch_size * 2, h_dim]
                        path_global_emb       # [batch_size * 2, h_dim]
                    )

                    # 将融合后的表示放回完整实体嵌入中
                    embedding = local_emb.clone()
                    embedding[query_entity_ids] = final_emb
                except Exception as e:
                    # 如果路径编码失败，回退到使用局部嵌入
                    print(f"路径编码失败，使用局部嵌入: {str(e)}")
                    embedding = local_emb
            # ===========================================================
            start_embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
            time_embs = self.get_init_time(all_triples)

            score_rel_r = self.rel_raw_mode(embedding, r_emb, time_embs, all_triples)
            score_rel_h = self.rel_history_mode(embedding, r_emb, time_embs, all_triples, rel_history_vocabulary)
            score_r = self.raw_mode(embedding, r_emb, time_embs, all_triples)
            score_h = self.history_mode(start_embedding, r_emb, time_embs, all_triples, entity_history_vocabulary)
            # ======== 新增：计算局部 1-hop 历史打分 ========
            score_l = self.history_local_mode(start_embedding, r_emb, time_embs, all_triples, entity_local_vocabulary) # p_local

            # 关键修复：添加数值稳定性保护
            # 添加小量 epsilon 防止 log(0) = nan
            epsilon = 1e-8

            score_rel = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r
            score_rel = torch.clamp(score_rel, min=epsilon, max=1.0)
            score_rel = torch.log(score_rel)

            score = self.history_rate * score_h + (1 - self.history_rate -0.2) * score_r + 0.2 * score_l
            score = torch.clamp(score, min=epsilon, max=1.0)
            score = torch.log(score)

            return all_triples, score, score_rel


    def get_loss(self, glist, triples, static_graph, entity_history_vocabulary, rel_history_vocabulary, entity_local_vocabulary, fwd_paths_tensor, fwd_lens, inv_paths_tensor, inv_lens, use_cuda):
        """
        计算损失函数，集成路径编码器和局部-全局融合模块（离线预处理版本）

        Args:
            fwd_paths_tensor: 正向路径张量，形状 [batch_size, K, max_seq_len, 4]
            fwd_lens: 正向路径长度张量，形状 [batch_size, K]
            inv_paths_tensor: 逆向路径张量，形状 [batch_size, K, max_seq_len, 4]
            inv_lens: 逆向路径长度张量，形状 [batch_size, K]
        """
        self.use_cuda = use_cuda
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0, 3]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)

        evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, use_cuda)
        # 提取局部演化表示 E_{t_q}^L
        local_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

        # ==================== 离线预处理版本：直接使用预处理的张量 ====================
        device = self.gpu if use_cuda else 'cpu'

        # 检查是否有路径数据
        if fwd_paths_tensor is None or len(fwd_paths_tensor) == 0:
            # 如果没有路径数据，直接使用局部嵌入
            pre_emb = local_emb
        else:
            batch_size = all_triples.size(0)

            # 直接在 batch 维度拼接正向和逆向张量
            paths_tensor = torch.cat([fwd_paths_tensor, inv_paths_tensor], dim=0).to(device)
            path_lengths = torch.cat([fwd_lens, inv_lens], dim=0).to(device)

            # 拼接后的维度应该对应 all_triples 的 [batch_size * 2] 维度
            query_times = all_triples[:, 3].float()  # [batch_size * 2]
            query_entity_ids = all_triples[:, 0]     # [batch_size * 2]
            local_emb_per_query = local_emb[query_entity_ids]  # [batch_size * 2, h_dim]

            try:
                # 使用路径编码器
                path_global_emb = self.path_encoder(
                    paths_tensor, path_lengths, query_times,
                    local_emb, self.emb_rel
                )  # [batch_size * 2, h_dim]

                # 使用局部-全局融合模块融合表示
                final_emb = self.local_global_fusion(
                    local_emb_per_query,  # [batch_size * 2, h_dim]
                    path_global_emb       # [batch_size * 2, h_dim]
                )

                # 将融合后的表示放回完整实体嵌入中
                pre_emb = local_emb.clone()
                pre_emb[query_entity_ids] = final_emb
            except Exception as e:
                # 如果路径编码失败，回退到使用局部嵌入
                print(f"路径编码失败，使用局部嵌入: {str(e)}")
                pre_emb = local_emb
        # ===========================================================

        start_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
        time_embs = self.get_init_time(all_triples)

        # 关键修复：添加数值稳定性保护
        epsilon = 1e-8

        if self.entity_prediction:
            score_r = self.raw_mode(pre_emb, r_emb, time_embs, all_triples)
            score_h = self.history_mode(start_emb, r_emb, time_embs, all_triples, entity_history_vocabulary)
            # ======== 新增：计算局部 1-hop 历史打分 ========
            score_l = self.history_local_mode(start_emb, r_emb, time_embs, all_triples, entity_local_vocabulary)
            score_en = self.history_rate * score_h + (1 - self.history_rate-0.2) * score_r + 0.2 * score_l
            score_en = torch.clamp(score_en, min=epsilon, max=1.0)
            scores_en = torch.log(score_en)
            loss_ent += F.nll_loss(scores_en, all_triples[:, 2])

        if self.relation_prediction:
            score_rel_r = self.rel_raw_mode(pre_emb, r_emb, time_embs, all_triples)
            score_rel_h = self.rel_history_mode(pre_emb, r_emb, time_embs, all_triples, rel_history_vocabulary)
            score_re = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r
            score_re = torch.clamp(score_re, min=epsilon, max=1.0)
            scores_re = torch.log(score_re)
            loss_rel += F.nll_loss(scores_re, all_triples[:, 1])

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    angle = 90 // len(evolve_embs)
                    # step = (self.angle * math.pi / 180) * (time_step + 1)
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        # 关键修复：防止除以零
                        c = torch.clamp(c, min=1e-8)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        # 关键修复：防止除以零
                        c = torch.clamp(c, min=1e-8)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_rel, loss_static

    def get_init_time(self, quadrupleList):
        T_idx = quadrupleList[:, 3] // self.time_interval
        T_idx = T_idx.unsqueeze(1).float()
        t1 = self.weight_t1 * T_idx + self.bias_t1
        t2 = self.sin(self.weight_t2 * T_idx + self.bias_t2)
        return t1, t2

    def raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        scores_ob = self.decoder_ob1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, self.num_ents)
        score = F.softmax(scores_ob, dim=1)
        return score

    def history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        if self.use_cuda:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
            global_index = global_index.to('cuda')
        else:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        score_global = self.decoder_ob2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding = global_index)
        score_h = score_global
        score_h = F.softmax(score_h, dim=1)
        return score_h

    def history_local_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        if self.use_cuda:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
            global_index = global_index.to('cuda')
        else:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        # 这里使用新增的 decoder_ob3
        score_global = self.decoder_ob3.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding=global_index)
        score_h = F.softmax(score_global, dim=1)
        return score_h

    def rel_raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        scores_re = self.rdecoder_re1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, 2 * self.num_rels)
        score = F.softmax(scores_re, dim=1)
        return score

    def rel_history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        if self.use_cuda:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
            global_index = global_index.to('cuda')
        else:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        score_global = self.rdecoder_re2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding=global_index)
        score_h = score_global
        score_h = F.softmax(score_h, dim=1)
        return score_h






