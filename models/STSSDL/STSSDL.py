"""
ST-SSDL: 原型增强的时空对比学习
- 自适应图卷积DCRNN编码器-解码器架构
- 原型查询机制：原型驱动的时空表示学习
- 三元组对比损失：query-pos-neg对比学习
- 自适应时空嵌入：节点嵌入 + 时间嵌入
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel


# =============================================================================
# 图卷积模块
# =============================================================================

class AGCN(nn.Module):
    """自适应图卷积层"""
    
    def __init__(self, dim_in, dim_out, cheb_k, num_support):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(num_support * cheb_k * dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, supports):
        """图卷积前向传播
        
        Args:
            x: (B, N, C_in) 输入特征
            supports: list of (N, N) 邻接矩阵
        """
        x_g = []
        for support in supports:
            if len(support.shape) == 2:
                # 单个图
                support_ks = [torch.eye(support.shape[0]).to(support.device), support]
                for k in range(2, self.cheb_k):
                    support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
                for graph in support_ks:
                    x_g.append(torch.einsum("nm,bmc->bnc", graph, x))
            else:
                # batch图
                support_ks = [torch.eye(support.shape[1]).repeat(support.shape[0], 1, 1).to(support.device), support]
                for k in range(2, self.cheb_k):
                    support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
                for graph in support_ks:
                    x_g.append(torch.einsum("bnm,bmc->bnc", graph, x))
        
        x_g = torch.cat(x_g, dim=-1)
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias
        return x_gconv


class AGCRNCell(nn.Module):
    """自适应图卷积循环单元"""
    
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_support):
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, num_support)
        self.update = AGCN(dim_in + self.hidden_dim, dim_out, cheb_k, num_support)

    def forward(self, x, state, supports):
        """前向传播
        
        Args:
            x: (B, N, D_in) 输入
            state: (B, N, hidden_dim) 隐藏状态
            supports: list of 邻接矩阵
        """
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class ADCRNN_Encoder(nn.Module):
    """自适应DCRNN编码器"""
    
    def __init__(self, node_num, dim_in, dim_out, cheb_k, rnn_layers, num_support):
        super().__init__()
        assert rnn_layers >= 1
        self.node_num = node_num
        self.input_dim = dim_in
        self.rnn_layers = rnn_layers
        
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, num_support))
        for _ in range(1, rnn_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, num_support))

    def forward(self, x, init_state, supports):
        """编码器前向传播
        
        Args:
            x: (B, T, N, D) 输入序列
            init_state: list of (B, N, hidden_dim) 初始隐藏状态
            supports: list of 邻接矩阵
        Returns:
            current_inputs: (B, T, N, hidden_dim) 所有时间步的输出
            output_hidden: list of (B, N, hidden_dim) 每层的最终隐藏状态
        """
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        
        for i in range(self.rnn_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, supports)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = [self.dcrnn_cells[i].init_hidden_state(batch_size) for i in range(self.rnn_layers)]
        return init_states


class ADCRNN_Decoder(nn.Module):
    """自适应DCRNN解码器"""
    
    def __init__(self, node_num, dim_in, dim_out, cheb_k, rnn_layers, num_support):
        super().__init__()
        assert rnn_layers >= 1
        self.node_num = node_num
        self.input_dim = dim_in
        self.rnn_layers = rnn_layers
        
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, num_support))
        for _ in range(1, rnn_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, num_support))

    def forward(self, xt, init_state, supports):
        """解码器前向传播（单步）
        
        Args:
            xt: (B, N, D) 当前输入
            init_state: list of (B, N, hidden_dim) 初始隐藏状态
            supports: list of 邻接矩阵
        """
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        
        for i in range(self.rnn_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        
        return current_inputs, output_hidden


# =============================================================================
# 原型模块
# =============================================================================

class PrototypeModule(nn.Module):
    """原型查询模块"""
    
    def __init__(self, hidden_dim, prototype_num, prototype_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prototype_num = prototype_num
        self.prototype_dim = prototype_dim
        
        # 原型参数
        self.prototypes = nn.Parameter(torch.randn(prototype_num, prototype_dim))
        nn.init.xavier_normal_(self.prototypes)
        
        # 查询投影
        self.Wq = nn.Parameter(torch.randn(hidden_dim, prototype_dim))
        nn.init.xavier_normal_(self.Wq)
    
    def forward(self, h_t):
        """查询原型
        
        Args:
            h_t: (B, N, hidden_dim) 隐藏状态
        Returns:
            value: (B, N, prototype_dim) 加权原型值
            query: (B, N, prototype_dim) 查询向量
            pos: (B, N, prototype_dim) 正原型
            neg: (B, N, prototype_dim) 负原型
            mask: (B, N, 2) 原型索引
        """
        # 查询投影
        query = torch.matmul(h_t, self.Wq)  # (B, N, prototype_dim)
        
        # 注意力分数
        att_score = torch.softmax(torch.matmul(query, self.prototypes.t()), dim=-1)  # (B, N, M)
        
        # 加权原型值
        value = torch.matmul(att_score, self.prototypes)  # (B, N, prototype_dim)
        
        # 正负原型
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.prototypes[ind[:, :, 0]]  # (B, N, prototype_dim)
        neg = self.prototypes[ind[:, :, 1]]  # (B, N, prototype_dim)
        mask = torch.stack([ind[:, :, 0], ind[:, :, 1]], dim=-1)  # (B, N, 2)
        
        return value, query, pos, neg, mask


# =============================================================================
# 主模型
# =============================================================================

class STSSDL(BaseModel):
    """原型增强的时空对比学习模型
    
    数据流: 编码器 -> 原型查询 -> 解码器 -> 预测
    支持3维输入: [时序值, 时刻(day-of-time), 历史值(day-of-week)]
    """
    
    model_name = "stssdl"

    def __init__(self, num_nodes: int, model_dim: int, output_dim: int = 1,
                 in_window: int = 12, out_window: int = 12,
                 rnn_units: int = 128, rnn_layers: int = 1, cheb_k: int = 3,
                 prototype_num: int = 20, prototype_dim: int = 64,
                 tod_embed_dim: int = 10, node_embed_dim: int = 20,
                 adaptive_embed_dim: int = 48,
                 lambda_contrastive: float = 0.1, lambda_deviation: float = 1.0,
                 cl_decay_steps: int = 2000, use_curriculum_learning: bool = True,
                 adj_mx: np.ndarray = None,
                 device: torch.device = None,
                 time_intervals: int = 1800):
        """初始化STSSDL模型
        
        Args:
            num_nodes: 节点数量
            model_dim: 模型维度
            output_dim: 输出维度
            in_window: 输入时间窗口
            out_window: 输出时间窗口
            rnn_units: RNN单元数
            rnn_layers: RNN层数
            cheb_k: Chebyshev多项式阶数
            prototype_num: 原型数量
            prototype_dim: 原型维度
            tod_embed_dim: 时间嵌入维度
            node_embed_dim: 节点嵌入维度
            adaptive_embed_dim: 自适应嵌入维度
            lambda_contrastive: 对比损失权重
            lambda_deviation: 偏差损失权重
            cl_decay_steps: 课程学习衰减步数
            use_curriculum_learning: 是否使用课程学习
            adj_mx: 邻接矩阵
            time_intervals: 时间间隔（秒）
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.in_window = in_window
        self.out_window = out_window
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers
        self.cheb_k = cheb_k
        self.prototype_num = prototype_num
        self.prototype_dim = prototype_dim
        self.tod_embed_dim = tod_embed_dim
        self.node_embed_dim = node_embed_dim
        self.adaptive_embed_dim = adaptive_embed_dim
        self.input_embedding_dim = model_dim
        self.lambda_contrastive = lambda_contrastive
        self.lambda_deviation = lambda_deviation
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        self.device = device or torch.device('cpu')
        
        # 时间嵌入配置
        self.time_of_day_size = 24 * 60 * 60 // time_intervals
        
        # 输入投影层
        self.input_proj = nn.Linear(1, model_dim)
        
        # 时间嵌入
        self.time_embedding = nn.Embedding(self.time_of_day_size, tod_embed_dim)
        
        # 节点嵌入
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, node_embed_dim))
        nn.init.xavier_normal_(self.node_embedding)
        
        # 自适应嵌入
        self.adaptive_embedding = nn.Parameter(torch.randn(in_window, num_nodes, adaptive_embed_dim))
        nn.init.xavier_uniform_(self.adaptive_embedding)
        
        # 总嵌入维度
        total_embed_dim = model_dim + tod_embed_dim + adaptive_embed_dim + node_embed_dim
        
        # 编码器
        self.encoder = ADCRNN_Encoder(
            num_nodes, total_embed_dim, rnn_units, cheb_k, rnn_layers, 1
        )
        
        # 解码器
        decoder_dim = rnn_units + prototype_dim
        self.decoder = ADCRNN_Decoder(
            num_nodes, total_embed_dim - adaptive_embed_dim, decoder_dim, cheb_k, rnn_layers, 1
        )
        
        # 原型模块
        self.prototype_module = PrototypeModule(rnn_units, prototype_num, prototype_dim)
        
        # 超网络生成自适应图
        self.hypernet = nn.Sequential(
            nn.Linear(decoder_dim * 2, tod_embed_dim, bias=True)
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(decoder_dim, output_dim, bias=True)
        )
        
        # 邻接矩阵处理
        if adj_mx is not None:
            if isinstance(adj_mx, np.ndarray):
                adj = torch.from_numpy(adj_mx).float().to(self.device)
            else:
                adj = adj_mx.to(self.device)
            adj = torch.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
            self.register_buffer('adj_mx', adj)
        else:
            self.register_buffer('adj_mx', torch.eye(num_nodes, device=self.device))
        
        self.batches_seen = 0
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """Xavier均匀初始化"""
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def from_args(args, num_nodes, adj_mx, device):
        """从命令行参数创建模型实例"""
        return STSSDL(
            num_nodes=num_nodes,
            model_dim=args.input_embedding_dim,
            output_dim=1,
            in_window=args.input_window,
            out_window=args.output_window,
            rnn_units=getattr(args, 'rnn_units', 128),
            rnn_layers=getattr(args, 'rnn_layers', 1),
            cheb_k=getattr(args, 'cheb_k', 3),
            prototype_num=getattr(args, 'prototype_num', 20),
            prototype_dim=getattr(args, 'prototype_dim', 64),
            tod_embed_dim=getattr(args, 'tod_embed_dim', 10),
            node_embed_dim=getattr(args, 'node_embed_dim', 20),
            adaptive_embed_dim=getattr(args, 'adaptive_embed_dim', 48),
            lambda_contrastive=getattr(args, 'lambda_contrastive', 0.1),
            lambda_deviation=getattr(args, 'lambda_deviation', 1.0),
            cl_decay_steps=getattr(args, 'cl_decay_steps', 2000),
            use_curriculum_learning=getattr(args, 'use_curriculum_learning', True),
            adj_mx=adj_mx,
            device=device,
            time_intervals=args.time_intervals
        ).to(device)

    def compute_sampling_threshold(self):
        """计算课程学习采样阈值"""
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(self.batches_seen / self.cl_decay_steps))

    def _build_features(self, x, x_cov):
        """构建时空嵌入特征
        
        Args:
            x: (B, T, N, 1) 时序值（当前时刻或历史时刻）
            x_cov: (B, T, N, 1) 时刻特征
        Returns:
            features: (B, T, N, total_embed_dim) 拼接后的特征
        """
        B, T, N, _ = x.shape
        
        # 时序值投影
        x_proj = self.input_proj(x)  # (B, T, N, model_dim)
        features = [x_proj]
        
        # 时间嵌入
        if self.tod_embed_dim > 0:
            tod_ids = (x_cov.squeeze(-1) * self.time_of_day_size).long()
            tod_ids = tod_ids.clamp(0, self.time_of_day_size - 1)
            time_emb = self.time_embedding(tod_ids)  # (B, T, N, tod_embed_dim)
            features.append(time_emb)
        
        # 自适应嵌入
        if self.adaptive_embed_dim > 0:
            adp_emb = self.adaptive_embedding.unsqueeze(0).expand(B, -1, -1, -1)
            features.append(adp_emb)
        
        # 节点嵌入
        if self.node_embed_dim > 0:
            node_emb = self.node_embedding.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
            features.append(node_emb)
        
        return torch.cat(features, dim=-1)  # (B, T, N, total_embed_dim)

    def forward(self, batch: dict, return_repr=False) -> torch.Tensor:
        """模型前向传播（双流时空编码）
        
        Args:
            batch['X']: (B, T, N, 3) - [时序值, 时刻, 历史值]
            return_repr: 是否返回中间表示用于对比学习
        Returns:
            (B, out_window, N, 1) 预测结果，或 (output, query, pos, neg, mask, latent_dis, prototype_dis)
        """
        X = batch['X']
        B, T, N, _ = X.shape
        
        # 解析输入
        x = X[..., :1]  # (B, T, N, 1) 时序值
        x_cov = X[..., 1:2]  # (B, T, N, 1) 时刻
        x_his = X[..., 2:3]  # (B, T, N, 1) 历史值
        
        supports = [self.adj_mx]
        init_state = self.encoder.init_hidden(B)
        
        # ========== 当前时刻编码流 ==========
        features = self._build_features(x, x_cov)
        h_en, _ = self.encoder(features, init_state, supports)
        h_t = h_en[:, -1, :, :]  # (B, N, rnn_units)
        
        # 原型查询
        v_t, q_t, p_t, n_t, mask = self.prototype_module(h_t)
        
        # ========== 历史时刻编码流（关键！原版ST-SSDL的双流设计）==========
        features_his = self._build_features(x_his, x_cov)
        h_his_en, _ = self.encoder(features_his, init_state, supports)
        h_a = h_his_en[:, -1, :, :]  # (B, N, rnn_units)
        
        # 历史原型查询
        v_a, q_a, p_a, n_a, mask_his = self.prototype_module(h_a)
        
        # ========== 距离约束（原版核心损失）==========
        latent_dis = torch.sum(torch.abs(q_t - q_a), dim=-1)  # (B, N)
        prototype_dis = torch.sum(torch.abs(p_t - p_a), dim=-1)  # (B, N)
        
        # 解码器准备
        h_de = torch.cat([h_t, v_t], dim=-1)  # (B, N, decoder_dim)
        h_aug = torch.cat([h_t, v_t, h_a, v_a], dim=-1)
        
        # 生成自适应图（与原版一致）
        node_embeddings = self.hypernet(h_aug)
        support = F.softmax(F.relu(torch.einsum('bnc,bmc->bnm', node_embeddings, node_embeddings)), dim=-1)
        supports_de = [support]
        
        ht_list = [h_de] * self.rnn_layers
        go = torch.zeros((B, N, self.output_dim), device=x.device)
        
        # 解码
        outputs = []
        for t in range(self.out_window):
            # 构建解码输入
            if self.input_embedding_dim > 0:
                go_proj = self.input_proj(go.unsqueeze(-1))  # (B, N, 1, model_dim)
                if go_proj.shape[2] == 1:
                    go_proj = go_proj.squeeze(2)  # (B, N, model_dim)
            else:
                go_proj = go  # (B, N, output_dim)
            
            features_list = [go_proj]
            
            # 时间嵌入
            if self.tod_embed_dim > 0:
                tod_ids = (x_cov[:, -1, :, 0] * self.time_of_day_size).long()
                tod_ids = tod_ids.clamp(0, self.time_of_day_size - 1)
                time_emb = self.time_embedding(tod_ids)  # (B, N, tod_embed_dim)
                features_list.append(time_emb)
            
            # 节点嵌入
            if self.node_embed_dim > 0:
                node_emb = self.node_embedding.unsqueeze(0).expand(B, -1, -1)  # (B, N, node_embed_dim)
                features_list.append(node_emb)
            
            dec_input = torch.cat(features_list, dim=-1)  # (B, N, dec_dim)
            
            # 使用自适应图
            h_de, ht_list = self.decoder(dec_input, ht_list, supports_de)
            go = self.output_proj(h_de)
            outputs.append(go)
        
        output = torch.stack(outputs, dim=1)  # (B, out_window, N, output_dim)
        
        if return_repr:
            return output, q_t, p_t, n_t, mask, latent_dis, prototype_dis, q_a, p_a, n_a
        return output

    def predict(self, batch: dict) -> torch.Tensor:
        """预测接口"""
        return self.forward(batch)

    def calculate_loss(self, batch: dict, labels=None) -> tuple:
        """计算STSSDL的组合损失
        
        Args:
            batch: 包含 'X' 和 'y' 的字典
            labels: 可选的标签（用于课程学习）
        Returns:
            total_loss: 总损失
            sep_losses: 分离的损失字典
        """
        X = batch['X']
        y = batch['y']
        B, T, N, _ = X.shape
        
        # 解析输入
        x = X[..., :1]  # (B, T, N, 1)
        x_cov = X[..., 1:2]  # (B, T, N, 1)
        x_his = X[..., 2:3]  # (B, T, N, 1)
        
        # 构建当前时刻特征和历史特征（双流编码）
        features = self._build_features(x, x_cov)
        features_his = self._build_features(x_his, x_cov)
        
        # 编码
        supports = [self.adj_mx]
        init_state = self.encoder.init_hidden(B)
        
        # 当前时刻编码
        h_en, _ = self.encoder(features, init_state, supports)
        h_t = h_en[:, -1, :, :]  # (B, N, rnn_units)
        
        # 原型查询
        v_t, q_t, p_t, n_t, mask = self.prototype_module(h_t)
        
        # 历史编码
        h_his_en, _ = self.encoder(features_his, init_state, supports)
        h_a = h_his_en[:, -1, :, :]  # (B, N, rnn_units)
        
        # 原型查询
        v_a, q_a, p_a, n_a, _ = self.prototype_module(h_a)
        
        # 计算距离
        latent_dis = torch.sum(torch.abs(q_t - q_a), dim=-1)  # (B, N)
        prototype_dis = torch.sum(torch.abs(p_t - p_a), dim=-1)  # (B, N)
        
        # 解码器准备
        h_de = torch.cat([h_t, v_t], dim=-1)
        h_aug = torch.cat([h_t, v_t, h_a, v_a], dim=-1)
        
        # 生成自适应图
        node_embeddings = self.hypernet(h_aug)
        support = F.softmax(F.relu(torch.einsum('bnc,bmc->bnm', node_embeddings, node_embeddings)), dim=-1)
        supports_de = [support]
        
        ht_list = [h_de] * self.rnn_layers
        go = torch.zeros((B, N, self.output_dim), device=x.device)
        
        outputs = []
        for t in range(self.out_window):
            # 构建解码输入
            if self.input_embedding_dim > 0:
                go_proj = self.input_proj(go.unsqueeze(-1))  # (B, N, 1, model_dim)
                if go_proj.shape[2] == 1:
                    go_proj = go_proj.squeeze(2)  # (B, N, model_dim)
            else:
                go_proj = go
            
            features_list = [go_proj]
            
            # 时间嵌入
            if self.tod_embed_dim > 0:
                # 使用x_cov的最后一个时间步
                tod_ids = (x_cov[:, -1, :, 0] * self.time_of_day_size).long()
                tod_ids = tod_ids.clamp(0, self.time_of_day_size - 1)
                time_emb = self.time_embedding(tod_ids)  # (B, N, tod_embed_dim)
                features_list.append(time_emb)
            
            # 节点嵌入
            if self.node_embed_dim > 0:
                node_emb = self.node_embedding.unsqueeze(0).expand(B, -1, -1)  # (B, N, node_embed_dim)
                features_list.append(node_emb)
            
            dec_input = torch.cat(features_list, dim=-1)
            
            h_de, ht_list = self.decoder(dec_input, ht_list, supports_de)
            go = self.output_proj(h_de)
            outputs.append(go)
            
            # 课程学习
            if self.training and self.use_curriculum_learning and labels is not None:
                if np.random.uniform(0, 1) < self.compute_sampling_threshold():
                    go = labels[:, t, :, :]
        
        output = torch.stack(outputs, dim=1)  # (B, out_window, N, output_dim)
        
        # 预测损失
        pred_loss = F.l1_loss(output, y)
        
        # 对比损失 (triplet loss)
        contrastive_loss = nn.TripletMarginLoss(margin=0.5)
        loss_c = contrastive_loss(q_t.detach(), p_t, n_t)
        
        # 偏差损失
        deviation_loss = F.l1_loss(latent_dis.detach(), prototype_dis)
        
        # 总损失
        total_loss = pred_loss + self.lambda_contrastive * loss_c + self.lambda_deviation * deviation_loss
        
        sep_losses = {
            'pred': pred_loss.item(),
            'contrastive': loss_c.item(),
            'deviation': deviation_loss.item()
        }
        
        self.batches_seen += 1
        
        return total_loss, sep_losses
