import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from models.base import BaseModel


class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2 * cheb_k * dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, supports):
        x_g = []
        support_set = []
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
            support_set.extend(support_ks)
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1)
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k)
        self.update = AGCN(dim_in + self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, supports):
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
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(ADCRNN_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, x, init_state, supports):
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, supports)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states


class ADCRNN_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(ADCRNN_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, xt, init_state, supports):
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class MegaCRN(BaseModel):
    """MegaCRN: Memory-Augmented Graph Convolutional Recurrent Network for Traffic Forecasting."""

    model_name = "megacrn"

    def __init__(self, num_nodes: int, rnn_units: int = 64,
                 output_dim: int = 1, input_dim: int = 1,
                 in_window: int = 12, out_window: int = 12,
                 num_layers: int = 3, cheb_k: int = 2,
                 mem_num: int = 20, mem_dim: int = 64,
                 cl_decay_steps: int = 2000, use_curriculum_learning: bool = True,
                 ycov_dim: int = 2, lamb: float = 0.01, lamb1: float = 0.01,
                 device: torch.device = None):
        super().__init__()

        self.num_nodes = num_nodes
        self.rnn_units = rnn_units
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.in_window = in_window
        self.out_window = out_window
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim

        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.lamb = lamb
        self.lamb1 = lamb1

        self.device = device or torch.device('cpu')

        # 时间编码维度
        self.time_of_day_size = 24  # 一天24个小时
        self.day_of_week_size = 7   # 一周7天

        self.memory = self.construct_memory()

        self.encoder = ADCRNN_Encoder(
            self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.num_layers
        )

        self.decoder_dim = self.rnn_units + self.mem_dim
        self.decoder = ADCRNN_Decoder(
            self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim,
            self.cheb_k, self.num_layers
        )

        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))

        self.apply(self._init_weights)

    @staticmethod
    def from_args(args, num_nodes, adj_mx, device):
        """从命令行参数创建MegaCRN模型实例"""
        return MegaCRN(
            num_nodes=num_nodes,
            rnn_units=args.input_embedding_dim,
            output_dim=1,
            input_dim=1,
            in_window=args.input_window,
            out_window=args.output_window,
            num_layers=args.num_layers,
            cheb_k=args.cheb_k,
            mem_num=args.mem_num,
            mem_dim=args.mem_dim,
            cl_decay_steps=args.cl_decay_steps,
            use_curriculum_learning=args.use_curriculum_learning,
            ycov_dim=args.ycov_dim,
            lamb=args.lamb,
            lamb1=args.lamb1,
            device=device
        ).to(device)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim), requires_grad=True)
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def query_memory(self, h_t: torch.Tensor):
        query = torch.matmul(h_t, self.memory['Wq'])
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)
        value = torch.matmul(att_score, self.memory['Memory'])
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.memory['Memory'][ind[:, :, 0]]
        neg = self.memory['Memory'][ind[:, :, 1]]
        return value, query, pos, neg

    def forward(self, batch):
        """模型前向传播

        输入:
            batch['X']: 输入张量，形状 (B, T, N, F)
                        B=batch_size, T=时间步数, N=节点数, F=特征维度
                        F 包含: [时序值, 一天内时刻(归一化), 一周中星期(归一化)]
            batch.get('y_cov'): 预测阶段的时间协变量，形状 (B, out_window, N, ycov_dim)

        返回:
            预测张量，形状 (B, out_window, N, output_dim)
        """
        x = batch['X']
        x = x.to(self.device)

        B, T, N, feat_dim = x.shape

        # MegaCRN 原始设计用于单变量预测，只使用第一个特征（流量数据）
        x = x[..., 0:1]  # Shape: (B, T, N, 1)

        # 提取时间协变量用于解码器
        y_cov = self._extract_time_encoding(x)  # Shape: (B, out_window, N, ycov_dim=2)

        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        supports = [g1, g2]

        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, state_en = self.encoder(x, init_state, supports)
        h_t = h_en[:, -1, :, :]

        h_att, query, pos, neg = self.query_memory(h_t)
        h_t = torch.cat([h_t, h_att], dim=-1)

        ht_list = [h_t] * self.num_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        out = []
        for t in range(self.out_window):
            h_de, ht_list = self.decoder(torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, supports)
            go = self.proj(h_de)
            out.append(go)
        output = torch.stack(out, dim=1)

        return output

    def _extract_time_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """生成时间协变量用于解码器

        返回:
            y_cov: 时间协变量，形状 (B, out_window, N, ycov_dim=2)
                   [day_of_week, time_of_day]
        """
        B = x.shape[0]
        device = x.device

        # 生成未来时间步的时间编码
        # 使用随机生成作为占位符
        dow = torch.randint(0, self.day_of_week_size, (B, self.out_window, self.num_nodes), device=device).float()
        tod = torch.randint(0, self.time_of_day_size, (B, self.out_window, self.num_nodes), device=device).float()

        # 归一化到 [0, 1]
        dow = dow / self.day_of_week_size
        tod = tod / self.time_of_day_size

        y_cov = torch.stack([dow, tod], dim=-1)  # (B, out_window, N, 2)
        return y_cov

    def predict(self, batch):
        return self.forward(batch)
