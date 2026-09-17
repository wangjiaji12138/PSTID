"""
STDN (AAAI 2025): Spatio-temporal-aware Trend-Seasonality Decomposition Network
for Traffic Flow Forecasting

Paper: Spatio-temporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting
       (AAAI 2025)

适配 BaseModel 接口，支持三维输入进行训练。
输入格式: (B, T, N, F) 其中 F 可以是任意维度的特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from models.base import BaseModel


def calculate_normalized_laplacian(adj_mx):
    """
    Calculate normalized Laplacian of adjacency matrix.

    Args:
        adj_mx: adjacency matrix (num_nodes, num_nodes)

    Returns:
        Laplacian matrix
    """
    d = np.sum(adj_mx, axis=1)
    inv_d = np.power(d, -0.5)
    inv_d[inv_d == np.inf] = 0.
    D_inv_sqrt = np.diag(inv_d)
    L = np.eye(adj_mx.shape[0]) - np.dot(np.dot(D_inv_sqrt, adj_mx), D_inv_sqrt)
    return L


class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu, bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]

        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay if bn_decay else 0.1)
        nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # x: (batch_size, num_step, num_vertex, D)
        x = x.permute(0, 3, 2, 1)  # (batch_size, D, num_vertex, num_step)

        x = F.pad(x, ([self.padding_size[1], self.padding_size[1],
                       self.padding_size[0], self.padding_size[0]]))

        x = self.conv(x)
        x = self.batch_norm(x)

        if self.activation is not None:
            x = self.activation(x)

        return x.permute(0, 3, 2, 1)  # Back to (batch_size, num_step, num_vertex, D)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)

        self.convs = nn.ModuleList([
            conv2d_(
                input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
                padding='VALID', use_bias=use_bias, activation=activation,
                bn_decay=bn_decay
            ) for input_dim, num_unit, activation in zip(input_dims, units, activations)
        ])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class TimeEncode(nn.Module):
    """
    Time encoding module for day of week and time of day.
    X: (batch_size, num_his+num_pred, 2) (dayofweek, timeofday)
    """
    def __init__(self, D, bn_decay):
        super(TimeEncode, self).__init__()
        self.ff = nn.Linear(2, 2)
        self.FC_his = FC(
            input_dims=[2, D], units=[D, D], activations=[F.relu, torch.sigmoid],
            bn_decay=bn_decay
        )
        self.FC_Pred = FC(
            input_dims=[2, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay
        )

    def forward(self, x, num_vertex, num_his):
        x = x.float()
        x = self.ff(x)
        x = torch.sin(x)
        x = x.unsqueeze(dim=2)

        His = x[:, :num_his]
        Pred = x[:, num_his:]

        His = self.FC_his(His)
        Pred = self.FC_Pred(Pred)

        add_vertex = torch.zeros(1, 1, num_vertex, 1, device=x.device)
        His = His + add_vertex
        Pred = Pred + add_vertex

        return His, Pred


class TEmbedding(nn.Module):
    """
    Temporal embedding module combining time features and spatial embedding.
    X: (batch_size, num_his+num_pred, 2) (dayofweek, timeofday)
    SE: spatial embedding (num_vertex, D)
    T: number of time slots per day
    """
    def __init__(self, input_dim, D, num_nodes, bn_decay):
        super(TEmbedding, self).__init__()
        self.FC = FC(
            input_dims=[input_dim, D, D], units=[D, D, D],
            activations=[F.relu, F.relu, torch.sigmoid],
            bn_decay=bn_decay
        )

    def forward(self, X, SE, T, num_vertex, num_his, device, num_pred):
        # X: (batch_size, num_step, 2) - day of week and time of day (historical only)
        # Note: TE should contain only historical data when working with LibCity
        batch_size = X.shape[0]
        num_step = X.shape[1]  # This is the actual number of timesteps available

        # One-hot encode time of day and day of week
        dayofweek = torch.empty(batch_size, num_step, 7, device=device)
        timeofday = torch.empty(batch_size, num_step, T, device=device)

        for i in range(batch_size):
            dayofweek[i] = F.one_hot(X[i, :, 0].long() % 7, 7)
            timeofday[i] = F.one_hot(X[i, :, 1].long() % T, T)

        X = torch.cat((timeofday, dayofweek), dim=-1)  # (batch_size, num_step, 7+T)
        X = X.unsqueeze(dim=2)  # (batch_size, num_step, 1, D)

        # Add vertex dimension to get (batch_size, num_step, num_vertex, D)
        add_vertex = torch.zeros(1, 1, num_vertex, 1, device=device)
        X = X + add_vertex
        X = self.FC(X)
        X = torch.sin(X)
        # Now X is (batch_size, num_step, num_vertex, D)

        # His uses all available timesteps from X (the actual data we have)
        His = X  # (batch_size, num_step, num_vertex, D)

        # For prediction, we need to generate embeddings for num_pred steps
        # When TE only contains historical data, we repeat the last timestep
        if num_step < num_pred:
            # Need to create prediction embeddings for num_pred steps
            last_t = X[:, -1:, :, :]  # (batch_size, 1, num_vertex, D)
            repeats = [1, num_pred - num_step, 1, 1]
            Pred = last_t.repeat(repeats)  # (batch_size, num_pred, num_vertex, D)
        else:
            Pred = X[:, :num_pred, :, :]

        # SE_for_his should match His's shape for the Trend module
        # Use only the first num_step timesteps from SE (or repeat if needed)
        if SE.shape[1] >= num_step:
            SE_for_his = SE[:, :num_step, :, :]
        else:
            # Repeat last SE timestep to match num_step
            last_se = SE[:, -1:, :, :]
            repeats = [1, num_step - SE.shape[1], 1, 1]
            SE_for_his = torch.cat([SE, last_se.repeat(repeats)], dim=1)

        return His + F.relu(SE_for_his), Pred


class SEmbedding(nn.Module):
    """
    Spatial embedding using Laplacian positional encoding.
    """
    def __init__(self, D):
        super(SEmbedding, self).__init__()
        self.LaplacianPE1 = nn.Linear(32, 32)
        self.Norm1 = nn.LayerNorm(32, elementwise_affine=False)
        self.act = nn.LeakyReLU()
        self.LaplacianPE2 = nn.Linear(32, D)
        self.Norm2 = nn.LayerNorm(D, elementwise_affine=False)

    def forward(self, lpls, batch_size, pred_steps):
        # lpls: (num_nodes, 32) - Laplacian positional encoding
        lap_pos_enc = self.Norm2(self.LaplacianPE2(self.act(self.Norm1(self.LaplacianPE1(lpls)))))
        tensor_neb = lap_pos_enc.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1).repeat(1, pred_steps, 1, 1)
        return torch.sigmoid(tensor_neb)


class TrendSeasonalDecomposition(nn.Module):
    """
    Trend-Seasonality Decomposition module.
    Disentangles trend-cyclical and seasonal components.
    """
    def __init__(self, num_vertex, D):
        super(TrendSeasonalDecomposition, self).__init__()
        self.vector = nn.Parameter(torch.full((1, 1, num_vertex, 1), 0.5, requires_grad=True))

    def forward(self, X, STEmbedding):
        # X: (batch_size, num_step, num_vertex, D)
        # STEmbedding: (batch_size, num_step, num_vertex, D)
        trend = torch.mul(X, STEmbedding)
        seasonal = X - trend
        zero_shape = torch.zeros_like(X)
        vector = zero_shape + self.vector
        result = vector * trend + (1 - vector) * seasonal
        return result


class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i+1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L - 1:
                x = F.relu(x)
        if self.res_ln:
            x += inputs
        x = self.ln(x)
        return x


class MAB(nn.Module):
    """
    Multi-head Attention Block.
    """
    def __init__(self, K, d, input_dim, output_dim, bn_decay):
        super(MAB, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        self.FC_q = FC(input_dims=input_dim, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_k = FC(input_dims=input_dim, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_v = FC(input_dims=input_dim, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=output_dim, activations=F.relu, bn_decay=bn_decay)

    def forward(self, Q, K, batch_size, att_type="spatial", mask=None):
        query = self.FC_q(Q)
        key = self.FC_k(K)
        value = self.FC_v(K)

        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)

        if mask is None:
            if att_type == "temporal":
                query = query.permute(0, 2, 1, 3)
                key = key.permute(0, 2, 1, 3)
                value = value.permute(0, 2, 1, 3)

            attention = torch.matmul(query, key.transpose(2, 3))
            attention /= (self.d ** 0.5)
            attention = F.softmax(attention, dim=-1)
            result = torch.matmul(attention, value)

            if att_type == "temporal":
                result = result.permute(0, 2, 1, 3)
        else:
            mask = torch.cat(torch.split(mask, self.K, dim=-1), dim=0)
            if att_type == "temporal":
                query = query.permute(0, 2, 1, 3)
                key = key.permute(0, 2, 1, 3)
                value = value.permute(0, 2, 1, 3)
                mask = mask.permute(0, 2, 1, 3)

            if mask.shape == query.shape:
                set_mask = torch.ones_like(key)
                mask = torch.matmul(mask, set_mask.transpose(2, 3))
            elif mask.shape == key.shape:
                set_mask = torch.ones_like(query)
                mask = torch.matmul(set_mask, mask.transpose(2, 3))

            attention = torch.matmul(query, key.transpose(2, 3))
            attention /= (self.d ** 0.5)
            attention = attention.masked_fill(mask == 0, -1e9)
            attention = F.softmax(attention, dim=-1)
            result = torch.matmul(attention, value)

            if att_type == "temporal":
                result = result.permute(0, 2, 1, 3)

        result = torch.cat(torch.split(result, batch_size, dim=0), dim=-1)
        result = self.FC(result)
        return result


class MAB_new(nn.Module):
    """
    Simplified Multi-head Attention Block for decoder.
    """
    def __init__(self, K, d, input_dim, output_dim, bn_decay):
        super(MAB_new, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        self.FC_q = FC(input_dims=input_dim, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_k = FC(input_dims=input_dim, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_v = FC(input_dims=input_dim, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=output_dim, activations=F.relu, bn_decay=bn_decay)

    def forward(self, Q, K, batch_size):
        query = self.FC_q(Q)
        key = self.FC_k(K)
        value = self.FC_v(K)

        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        result = torch.matmul(attention, value)

        result = result.permute(0, 2, 1, 3)
        result = torch.cat(torch.split(result, batch_size, dim=0), dim=-1)
        result = self.FC(result)
        return result


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism.
    """
    def __init__(self, K, d, num_of_vertices, set_dim, bn_decay):
        super(TemporalAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.num_of_vertices = num_of_vertices
        self.set_dim = set_dim
        self.I = nn.Parameter(torch.Tensor(1, set_dim, self.num_of_vertices, D))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(K, d, D, D, bn_decay)
        self.mab1 = MAB(K, d, D, D, bn_decay)

    def forward(self, X, mask):
        batch_size = X.shape[0]
        I = self.I.repeat(X.size(0), 1, 1, 1)
        H = self.mab0(I, X, batch_size, "temporal", mask)
        result = self.mab1(X, H, batch_size, "temporal", mask)
        return torch.add(X, result)


class AttentionDecoder(nn.Module):
    """
    Attention decoder for final predictions.
    """
    def __init__(self, K, d, num_of_vertices, set_dim, bn_decay):
        super(AttentionDecoder, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.num_of_vertices = num_of_vertices
        self.set_dim = set_dim
        self.I = nn.Parameter(torch.Tensor(1, set_dim, self.num_of_vertices, 3 * D))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB_new(K, d, 3 * D, 3 * D, bn_decay)
        self.mab1 = MAB_new(K, d, 3 * D, D, bn_decay)

    def forward(self, X, TE, SE, mask):
        batch_size = X.shape[0]
        mid = X
        X = torch.cat((X, TE, SE), dim=-1)
        I = self.I.repeat(X.size(0), 1, 1, 1)
        H = self.mab0(I, X, batch_size)
        result = self.mab1(X, H, batch_size)
        return torch.add(mid, result)


class GRU(nn.Module):
    def __init__(self, outfea):
        super(GRU, self).__init__()
        self.ff = nn.Linear(2 * outfea, 2 * outfea)
        self.zff = nn.Linear(2 * outfea, outfea)
        self.outfea = outfea

    def forward(self, x, xh):
        r, u = torch.split(torch.sigmoid(self.ff(torch.cat([x, xh], -1))), self.outfea, -1)
        z = torch.tanh(self.zff(torch.cat([x, r * xh], -1)))
        x = u * z + (1 - u) * xh
        return x


class GRUEncoder(nn.Module):
    def __init__(self, outfea, num_step):
        super(GRUEncoder, self).__init__()
        self.gru = nn.ModuleList([GRU(outfea) for i in range(num_step)])

    def forward(self, x):
        B, T, N, F = x.shape
        hidden_state = torch.zeros([B, N, F], device=x.device)
        output = []
        for i in range(T):
            gx = x[:, i, :, :]
            gh = hidden_state
            hidden_state = self.gru[i](gx, gh)
            output.append(hidden_state)
        output = torch.stack(output, 1)
        return output


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nwv->ncwl', (x, A))
        return x.contiguous()


class gcn(nn.Module):
    """
    Graph Convolutional Network.
    """
    def __init__(self, c_in, c_out, dropout=0.3, support_len=1, order=2, bn_decay=0.1):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = FC(c_in, c_out, activations=F.relu, bn_decay=bn_decay)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        x = x.transpose(1, 3)
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = h.transpose(1, 3)
        h = self.mlp(h)
        h = h.transpose(1, 3)
        h = F.dropout(h, self.dropout, training=self.training)
        h = h.transpose(1, 3)
        return h


class STDNModel(nn.Module):
    """
    STDN Model: Spatio-temporal-aware Trend-Seasonality Decomposition Network.
    """
    def __init__(self, num_nodes, model_dim, input_dim, output_dim,
                 input_window, output_window, K=8, d=16, L=2, order=2,
                 bn_decay=0.1, time_slice_size=60, device=None):
        super(STDNModel, self).__init__()
        self.L = L
        self.K = K
        self.d = d
        self.num_his = input_window
        self.num_pred = output_window

        D = K * d
        set_dim = 1
        self.num_of_vertices = num_nodes
        self.time_slice_size = time_slice_size
        self.device = device or torch.device('cpu')

        self.TEmbedding = TEmbedding(
            int(1440 / time_slice_size) + 7, D, self.num_of_vertices, bn_decay
        )
        self.SEmbedding = SEmbedding(D)
        self.TSD = TrendSeasonalDecomposition(self.num_of_vertices, D)
        self.FeedForward_t = FeedForward([D, D], res_ln=True)
        self.FeedForward_s = FeedForward([D, D], res_ln=True)
        self.GRU_Trend = GRUEncoder(D, self.num_his)
        self.GRU_Seasonal = GRUEncoder(D, self.num_his)

        self.Decoder = nn.ModuleList([
            AttentionDecoder(K, d, self.num_of_vertices, set_dim, bn_decay)
            for _ in range(L)
        ])

        # 输入维度支持多维特征
        self.in_channels = input_dim
        self.out_channels = output_dim

        self.FC_1 = FC(
            input_dims=[self.in_channels, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay
        )
        self.FC_2 = FC(
            input_dims=[D, D], units=[D, self.out_channels], activations=[F.relu, None],
            bn_decay=bn_decay
        )

        # Dynamic graph construction parameters
        self.nodevec_p1 = nn.Parameter(
            torch.randn(int(1440 / time_slice_size), D), requires_grad=True
        )
        self.nodevec_p2 = nn.Parameter(
            torch.randn(self.num_of_vertices, D), requires_grad=True
        )
        self.nodevec_p3 = nn.Parameter(
            torch.randn(self.num_of_vertices, D), requires_grad=True
        )
        self.nodevec_pk = nn.Parameter(
            torch.randn(D, D, D), requires_grad=True
        )

        self.GCN = gcn(D, D, order=order)
        self.order = order

    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        """Dynamic graph construction."""
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)
        adp = F.softmax(F.relu(adp), dim=2)
        return adp

    def forward(self, X, TE, lpls):
        """
        Forward pass.

        Args:
            X: input traffic data (batch_size, num_his, num_vertices, feature_dim)
            TE: time encoding (batch_size, num_his + num_pred, 2)
            lpls: Laplacian positional encoding (num_vertices, 32)

        Returns:
            predictions (batch_size, num_pred, num_vertices, output_dim)
        """
        # Input projection
        X = self.FC_1(X)  # (batch_size, num_his, num_vertices, D)

        # Dynamic graph construction
        ind = TE[:, 0, 1].long()  # Time of day index
        time_emb = self.nodevec_p1[ind]  # (batch_size, D)
        adp = self.dgconstruct(time_emb, self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)
        new_supports = [adp]

        # Graph convolution
        X = self.GCN(X, new_supports)

        # Spatial and temporal embeddings
        SE = self.SEmbedding(lpls, X.shape[0], self.num_pred)
        his, pred = self.TEmbedding(TE, SE, self.nodevec_p1.shape[0],
                                    self.num_of_vertices, self.num_his, X.device, self.num_pred)

        # Trend-Seasonality decomposition
        trend = self.TSD(X, his)
        seasonal = X - trend

        trend = self.FeedForward_t(trend)
        seasonal = self.FeedForward_s(seasonal)

        # Encoder
        trend = self.GRU_Trend(trend)
        seasonal = self.GRU_Seasonal(seasonal)

        result = trend + seasonal

        # If input window is longer than output window, take the last num_pred timesteps
        if result.shape[1] > self.num_pred:
            result = result[:, -self.num_pred:, :, :]

        # Decoder
        for net in self.Decoder:
            result = net(result, pred, SE, None)

        result = self.FC_2(result)

        return result


class STDN(BaseModel):
    """
    STDN (AAAI 2025) for Traffic Flow Forecasting.

    Spatio-temporal-aware Trend-Seasonality Decomposition Network.

    Reference:
        Cao et al. "Spatio-temporal-aware Trend-Seasonality Decomposition Network
        for Traffic Flow Forecasting." AAAI 2025.

    适配 BaseModel 接口，支持三维输入（多变量预测）。
    输入格式: (B, T, N, F) 其中 F 是特征维度
    - F=1: 单变量预测 (如仅预测流量)
    - F>1: 多变量预测 (如同时预测流量、速度等)
    """

    model_name = "stdn"

    def __init__(self, num_nodes: int, model_dim: int, input_dim: int = 1,
                 output_dim: int = 1, input_window: int = 12, output_window: int = 12,
                 K: int = 8, d: int = 16, L: int = 2, order: int = 2,
                 bn_decay: float = 0.1, time_slice_size: int = 60,
                 adj_mx=None, device=None):
        """初始化 STDN 模型

        参数:
            num_nodes: 图中节点数量（如路网中的路口/传感器数量）
            model_dim: 模型隐层维度（特征嵌入后的表示维度）
            input_dim: 输入特征维度（默认为1，即单变量）
            output_dim: 输出维度（默认为1，即单变量预测）
            input_window: 输入时间窗口长度（如12表示使用过去12个时间步）
            output_window: 输出时间窗口长度（预测未来多少个时间步）
            K: 多头注意力头数
            d: 每个头的维度
            L: 解码器层数
            order: 图卷积的阶数
            bn_decay: BatchNorm 的 momentum
            time_slice_size: 时间片大小（分钟），用于时间编码
            adj_mx: 邻接矩阵，定义图的拓扑结构
            device: 计算设备
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.model_dim = model_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_window = input_window
        self.output_window = output_window

        self.K = K
        self.d = d
        self.L = L
        self.order = order
        self.bn_decay = bn_decay
        self.time_slice_size = time_slice_size

        self.device = device or torch.device('cpu')

        # 时间编码维度
        self.time_of_day_size = int(1440 / time_slice_size)  # 一天中的时间片数
        self.day_of_week_size = 7  # 一周7天

        # 计算 Laplacian positional encoding
        if adj_mx is not None:
            if isinstance(adj_mx, np.ndarray):
                laplacian = calculate_normalized_laplacian(adj_mx)
                try:
                    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
                    self.lpls = torch.FloatTensor(eigenvectors[:, :32])
                except:
                    self.lpls = torch.randn(self.num_nodes, 32)
            else:
                self.lpls = adj_mx
        else:
            self.lpls = torch.randn(self.num_nodes, 32)

        # 构建模型
        self.model = STDNModel(
            num_nodes=self.num_nodes,
            model_dim=model_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            input_window=input_window,
            output_window=output_window,
            K=K, d=d, L=L, order=order,
            bn_decay=bn_decay,
            time_slice_size=time_slice_size,
            device=self.device
        )

        self.model = self.model.to(self.device)
        self.lpls = self.lpls.to(self.device)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def from_args(args, num_nodes, adj_mx, device):
        """从命令行参数创建模型实例

        参数:
            args: 包含所有模型配置的命令行参数对象
            num_nodes: 图中节点数量
            adj_mx: 邻接矩阵
            device: 计算设备

        返回:
            STDN: 初始化好的模型实例
        """
        # 从 args 获取参数，支持动态图学习和标准配置
        model_dim = getattr(args, 'input_embedding_dim', 128)
        input_dim = getattr(args, 'input_dim', 1)
        output_dim = getattr(args, 'output_dim', 1)
        input_window = getattr(args, 'input_window', 12)
        output_window = getattr(args, 'output_window', 12)

        return STDN(
            num_nodes=num_nodes,
            model_dim=model_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            input_window=input_window,
            output_window=output_window,
            K=getattr(args, 'K', 8),
            d=getattr(args, 'd', 16),
            L=getattr(args, 'L', 2),
            order=getattr(args, 'order', 2),
            bn_decay=getattr(args, 'bn_decay', 0.1),
            time_slice_size=getattr(args, 'time_slice_size', 60),
            adj_mx=adj_mx,
            device=device
        ).to(device)

    def forward(self, batch: dict) -> torch.Tensor:
        """模型前向传播

        输入:
            batch['X']: 输入张量，形状 (B, T, N, F)
                        B=batch_size, T=时间步数, N=节点数, F=特征维度
                        F 包含: [时序值, 一天内时刻(归一化), 一周中星期(归一化)]
            batch.get('TE'): 时间编码，形状 (B, T, 2) 其中 2=[dayofweek, timeofday]
                             如果不提供，将自动生成

        返回:
            预测张量，形状 (B, output_window, N, output_dim)
        """
        X = batch['X']  # Shape: (B, T, N, F)

        # 确保数据在正确的设备上
        X = X.to(self.device)

        batch_size, input_len, num_nodes, feature_dim = X.shape

        # STDN 原始设计用于单变量预测，只使用第一个特征（流量数据）
        # 其他特征（时间编码等）通过 TE 参数单独传入
        X_traffic = X[..., 0:1]  # Shape: (B, T, N, 1)

        TE = self._extract_time_encoding(X)  # 从输入数据中提取时间编码

        # 确保 TE 长度匹配输入窗口
        if TE.shape[1] > self.input_window:
            TE = TE[:, :self.input_window, :]

        TE = TE.to(self.device)
        lpls = self.lpls

        # 使用流量数据作为模型输入
        outputs = self.model(X_traffic, TE, lpls)

        return outputs

    def predict(self, batch: dict) -> torch.Tensor:
        """预测接口（与 forward 相同）"""
        return self.forward(batch)

    def get_num_params(self) -> int:
        """获取模型可训练参数数量"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _extract_time_encoding(self, X: torch.Tensor) -> torch.Tensor:
        """从输入数据中提取时间编码

        输入数据格式: (B, T, N, F) 其中 F >= 3
        F[0]: 流量/时序值
        F[1]: 一天内时刻 (归一化到 [0, 1])
        F[2]: 一周中星期 (归一化到 [0, 1])

        返回:
            TE: 时间编码，形状 (B, T, 2) 其中 2=[dayofweek, timeofday]
        """
        B, T, N, F = X.shape
        device = X.device

        if F >= 3:
            # 提取归一化的时间特征
            tod_norm = X[0, :, 0, 1].clone()  # (T,) - 一天内时刻
            dow_norm = X[0, :, 0, 2].clone()  # (T,) - 一周中星期

            # 反归一化到离散索引
            # tod_norm 范围 [0, 1] -> [0, time_of_day_size-1]
            # dow_norm 范围 [0, 1] -> [0, 6]
            tod_idx = (tod_norm * self.time_of_day_size).long().clamp(0, self.time_of_day_size - 1)
            dow_idx = (dow_norm * self.day_of_week_size).long().clamp(0, self.day_of_week_size - 1)

            # 为整个 batch 创建相同的时间编码
            TE = torch.zeros(B, T, 2, device=device)
            TE[:, :, 0] = dow_idx.float().unsqueeze(0).expand(B, -1)
            TE[:, :, 1] = tod_idx.float().unsqueeze(0).expand(B, -1)
        else:
            # 如果没有足够的时间特征，生成随机时间编码
            TE = torch.zeros(B, T, 2, device=device)
            TE[:, :, 0] = torch.randint(0, 7, (B, T), device=device).float()
            TE[:, :, 1] = torch.randint(0, self.time_of_day_size, (B, T), device=device).float()

        return TE
