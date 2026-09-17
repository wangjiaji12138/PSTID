"""
ST-SSL: 时空自监督学习
- 时空编码器：时空卷积层捕获多尺度时空依赖
- 空间异质性建模：基于原型聚类的空间对比学习
- 时间异质性建模：基于对比学习的时间模式识别
- 数据增强：拓扑增强 + 流量增强
"""

from __future__ import annotations

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

from models.base import BaseModel


# =============================================================================
# 数据增强模块
# =============================================================================

def sim_global(flow_data, sim_type='cos'):
    """计算全局相似度矩阵
    
    Args:
        flow_data: (B, T, N, C) 或 (B, N, C) 或 (N, C)
        sim_type: 'cos' 或 'att'
    Returns:
        sim: (N, N) 相似度矩阵
    """
    # 确保至少3维
    if len(flow_data.shape) == 2:
        # (N, C) -> (1, 1, N, C)
        flow_data = flow_data.unsqueeze(0).unsqueeze(0)
    elif len(flow_data.shape) == 3:
        # (B, N, C) -> (B, 1, N, C)
        flow_data = flow_data.unsqueeze(1)
    
    B, T, N, C = flow_data.shape
    
    if sim_type == 'cos':
        # 余弦相似度: 先squeeze到(B, N, C)，再计算
        flow_2d = flow_data.squeeze(1)  # (B, N, C)
        # 归一化并计算余弦相似度
        flow_norm = F.normalize(flow_2d, p=2, dim=-1)  # (B, N, C)
        sim = torch.einsum('bnc,bmc->nm', flow_norm, flow_norm)  # (N, N)
    elif sim_type == 'att':
        # 缩放点积注意力
        sim = torch.einsum('btnc,btmc->nm', flow_data, flow_data)
        scaling = float(B * T * C) ** -0.5
        sim = torch.softmax(sim * scaling, dim=-1)
    else:
        raise ValueError('sim_global only supports sim_type in [cos, att]')
    
    return sim


def aug_topology(sim_mx, input_graph, percent=0.2):
    """拓扑视角的数据增强
    
    Args:
        sim_mx: (N, N) 相似度矩阵
        input_graph: (N, N) 邻接矩阵（无自环）
        percent: 增强比例
    """
    drop_percent = percent / 2
    
    # 获取下三角的边索引（排除对角线）
    lower_tri_mask = torch.ones(input_graph.size(), dtype=bool, device=input_graph.device).tril(diagonal=-1)
    edge_mask = (input_graph > 0) & lower_tri_mask
    
    index_list = edge_mask.nonzero()
    edge_num = index_list.shape[0]
    
    # 如果没有边，直接返回原始图
    if edge_num == 0:
        return input_graph.clone()
    
    add_drop_num = max(1, int(edge_num * drop_percent / 2))
    # 使用clone而不是deepcopy
    aug_graph = input_graph.clone()

    masked_sim = sim_mx[edge_mask]
    if masked_sim.numel() == 0:
        return aug_graph
    
    drop_prob = torch.softmax(masked_sim, dim=0).cpu().numpy()
    drop_prob = (1. - drop_prob)
    drop_prob /= drop_prob.sum()
    
    drop_list = np.random.choice(edge_num, size=add_drop_num, p=drop_prob)
    drop_index = index_list[drop_list]
    
    zeros = torch.zeros_like(aug_graph[0, 0])
    aug_graph[drop_index[:, 0], drop_index[:, 1]] = zeros
    aug_graph[drop_index[:, 1], drop_index[:, 0]] = zeros

    # Edge adding
    node_num = input_graph.shape[0]
    x, y = np.meshgrid(range(node_num), range(node_num), indexing='ij')
    mask = y < x
    x, y = x[mask], y[mask]

    no_edge_mask = (input_graph == 0) & lower_tri_mask
    add_prob = sim_mx[no_edge_mask].cpu().numpy()
    if len(add_prob) > 0:
        add_prob = torch.softmax(torch.from_numpy(add_prob), dim=0).numpy()
        add_prob /= add_prob.sum()
        add_list = np.random.choice(len(add_prob), size=min(add_drop_num, len(add_prob)), p=add_prob)
        ones = torch.ones_like(aug_graph[0, 0])
        aug_graph[x[add_list], y[add_list]] = ones
        aug_graph[y[add_list], x[add_list]] = ones
    
    return aug_graph


def aug_traffic(t_sim_mx, flow_data, percent=0.2):
    """流量视角的数据增强
    
    Args:
        t_sim_mx: (T, B, N) 时间相似度矩阵
        flow_data: (B, T, N, C) 流量数据
        percent: 增强比例
    """
    B, T, N, C = flow_data.shape
    mask_num = int(B * T * N * percent)
    
    # 克隆而不是deepcopy
    aug_flow = flow_data.clone()

    # t_sim_mx: (T, B, N) -> permute -> (B, T, N)
    t_sim_perm = t_sim_mx.permute(1, 0, 2)  # (B, T, N)
    
    mask_prob = (1. - t_sim_perm.reshape(-1)).cpu().numpy()
    mask_prob /= mask_prob.sum()

    x, y, z = np.meshgrid(range(B), range(T), range(N), indexing='ij')
    mask_list = np.random.choice(B * T * N, size=mask_num, p=mask_prob)

    aug_flow[x.reshape(-1)[mask_list], y.reshape(-1)[mask_list], z.reshape(-1)[mask_list]] = 0.0

    return aug_flow


# =============================================================================
# Sinkhorn算法（空间对比学习用）
# =============================================================================

def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    """Sinkhorn算法用于最优传输"""
    Q = torch.exp(out / epsilon).t()
    B = Q.shape[1]
    K = Q.shape[0]

    sum_Q = torch.sum(Q)
    Q /= sum_Q
    
    for it in range(sinkhorn_iterations):
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B
    return Q.t()


# =============================================================================
# 时空编码器
# =============================================================================

class Align(nn.Module):
    """对齐输入输出维度"""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x


class TemporalConvLayer(nn.Module):
    """时间卷积层"""
    def __init__(self, kt, c_in, c_out, act="relu"):
        super().__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        # x: (B, C, T, N)
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)


class SpatioConvLayer(nn.Module):
    """空间图卷积层（Chebyshev多项式）"""
    def __init__(self, ks, c_in, c_out):
        super().__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1))
        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        # x: (B, C, T, N), Lk: (K, N, N)
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        x_in = self.align(x)
        return torch.relu(x_gc + x_in)


class Pooler(nn.Module):
    """时间注意力池化层"""
    def __init__(self, n_query, d_model, agg='avg'):
        super().__init__()
        self.att = nn.Conv2d(d_model, n_query, 1)
        self.align = Align(d_model, d_model)
        self.softmax = nn.Softmax(dim=2)
        self.d_model = d_model
        self.n_query = n_query
        if agg == 'avg':
            self.agg = nn.AvgPool2d(kernel_size=(n_query, 1), stride=1)
        elif agg == 'max':
            self.agg = nn.MaxPool2d(kernel_size=(n_query, 1), stride=1)

    def forward(self, x):
        # x: (B, C, T, N) -> nclv格式
        x_in = self.align(x)[:, :, -self.n_query:, :]
        A = self.att(x)
        A = F.softmax(A, dim=2)
        x = torch.einsum('nclv,nqlv->ncqv', x, A)
        x_agg = self.agg(x).squeeze(2)
        x_agg = torch.einsum('ncv->nvc', x_agg)
        A = torch.einsum('nqlv->lnqv', A)
        A = self.softmax(self.agg(A).squeeze(2))
        return torch.relu(x + x_in), x_agg.detach(), A.detach()


class STEncoder(nn.Module):
    """时空编码器
    
    Args:
        Kt: 时间卷积核大小
        Ks: 空间Chebyshev阶数
        blocks: [[C_in, C_hidden, C_out], ...] 每层的通道配置
        input_length: 输入时间长度
        num_nodes: 节点数量
        droprate: dropout比例
    """
    def __init__(self, Kt, Ks, blocks, input_length, num_nodes, droprate=0.1):
        super().__init__()
        self.Ks = Ks
        
        # Block 1
        c = blocks[0]
        self.tconv11 = TemporalConvLayer(Kt, c[0], c[1], "GLU")
        self.pooler = Pooler(input_length - (Kt - 1), c[1])
        self.sconv12 = SpatioConvLayer(Ks, c[1], c[1])
        self.tconv13 = TemporalConvLayer(Kt, c[1], c[2])
        self.ln1 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout1 = nn.Dropout(droprate)

        # Block 2
        c = blocks[1]
        self.tconv21 = TemporalConvLayer(Kt, c[0], c[1], "GLU")
        self.sconv22 = SpatioConvLayer(Ks, c[1], c[1])
        self.tconv23 = TemporalConvLayer(Kt, c[1], c[2])
        self.ln2 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout2 = nn.Dropout(droprate)

        self.s_sim_mx = None
        self.t_sim_mx = None

        out_len = input_length - 2 * (Kt - 1) * len(blocks)
        self.out_conv = TemporalConvLayer(out_len, c[2], c[2], "GLU")
        self.ln3 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout3 = nn.Dropout(droprate)
        self.receptive_field = input_length + Kt - 1

    def forward(self, x, graph):
        """前向传播
        
        Args:
            x: (B, T, N, C) 输入张量
            graph: (N, N) 邻接矩阵（无自环）
        Returns:
            (B, 1, N, d_model) 编码后的表示
        """
        # 计算拉普拉斯矩阵和Chebyshev多项式
        lap_mx = self._cal_laplacian(graph)
        Lk = self._cheb_polynomial(lap_mx, self.Ks)
        
        B, T, N, C = x.shape
        
        # 填充到receptive_field
        if T < self.receptive_field:
            x = F.pad(x, (0, 0, 0, 0, self.receptive_field - T, 0))
        
        # 转换格式: (B, T, N, C) -> (B, C, T, N)
        x = x.permute(0, 3, 1, 2)
        
        # Block 1
        x = self.tconv11(x)
        x, x_agg, self.t_sim_mx = self.pooler(x)
        # 计算空间相似度: (B, 1, N, C) -> (N, N)
        self.s_sim_mx = sim_global(x_agg.unsqueeze(1), sim_type='cos')
        x = self.sconv12(x, Lk)
        x = self.tconv13(x)
        x = self.dropout1(self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        
        # Block 2
        x = self.tconv21(x)
        x = self.sconv22(x, Lk)
        x = self.tconv23(x)
        x = self.dropout2(self.ln2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

        # Output block
        x = self.out_conv(x)
        x = self.dropout3(self.ln3(x.permute(0, 2, 3, 1)))
        
        # 返回 (B, 1, N, d_model)
        return x

    def _cheb_polynomial(self, laplacian, K):
        """计算Chebyshev多项式"""
        N = laplacian.size(0)
        multi_order_laplacian = torch.zeros([K, N, N], device=laplacian.device, dtype=torch.float)
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k-2]
        return multi_order_laplacian

    def _cal_laplacian(self, graph):
        """计算图的拉普拉斯矩阵"""
        I = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
        graph = graph + I
        D = torch.diag(torch.sum(graph, dim=-1) ** (-0.5))
        L = I - torch.mm(torch.mm(D, graph), D)
        return L


# =============================================================================
# 空间异质性建模（原型对比学习）
# =============================================================================

class SpatialHeteroModel(nn.Module):
    """空间异质性建模：基于软聚类的原型对比学习"""
    
    def __init__(self, c_in, nmb_prototype, tau=0.5):
        super().__init__()
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.prototypes = nn.Linear(c_in, nmb_prototype, bias=False)
        self.tau = tau
        self.d_model = c_in
        self.nmb_prototype = nmb_prototype

        for m in self.modules():
            self._weights_init(m)

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, z1, z2):
        """计算空间对比损失
        
        Args:
            z1, z2: (B, T, N, C) 或 (B, N, C) 两个增强视图的表示
        Returns:
            loss: 标量对比损失
        """
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = self.l2norm(w)
            self.prototypes.weight.copy_(w)

        # 处理 (B, 1, N, C) 格式 -> (B, N, C)
        if z1.dim() == 4 and z1.shape[1] == 1:
            z1 = z1.squeeze(1)
        if z2.dim() == 4 and z2.shape[1] == 1:
            z2 = z2.squeeze(1)
        
        # 展平并计算原型分配
        z1_flat = z1.reshape(-1, self.d_model)
        z2_flat = z2.reshape(-1, self.d_model)
        
        zc1 = self.prototypes(self.l2norm(z1_flat))
        zc2 = self.prototypes(self.l2norm(z2_flat))
        
        with torch.no_grad():
            q1 = sinkhorn(zc1.detach())
            q2 = sinkhorn(zc2.detach())
        
        l1 = - torch.mean(torch.sum(q1 * F.log_softmax(zc2 / self.tau, dim=1), dim=1))
        l2 = - torch.mean(torch.sum(q2 * F.log_softmax(zc1 / self.tau, dim=1), dim=1))
        
        return l1 + l2


# =============================================================================
# 时间异质性建模（对比学习）
# =============================================================================

class AvgReadout(nn.Module):
    """平均池化读出函数"""
    def __init__(self):
        super().__init__()
        self.sigm = nn.Sigmoid()

    def forward(self, h):
        """对节点维度平均
        
        Args:
            h: (B, N, C) 隐藏表示
        Returns:
            s: (B, C) 总结向量
        """
        s = torch.mean(h, dim=1)
        s = self.sigm(s)
        return s


class Discriminator(nn.Module):
    """二分类判别器"""
    def __init__(self, n_h):
        super().__init__()
        self.net = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self._weights_init(m)

    def _weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, summary, h_rl, h_fk):
        """判别真实/伪造表示
        
        Args:
            summary: (B, C) 总结向量
            h_rl: (B, N, C) 真实隐藏表示
            h_fk: (B, N, C) 伪造隐藏表示
        Returns:
            logits: (B, 2*N) 判别分数
        """
        s = torch.unsqueeze(summary, dim=1)
        s = s.expand_as(h_rl).contiguous()
        sc_rl = torch.squeeze(self.net(h_rl, s), dim=2)
        sc_fk = torch.squeeze(self.net(h_fk, s), dim=2)
        logits = torch.cat((sc_rl, sc_fk), dim=1)
        return logits


class TemporalHeteroModel(nn.Module):
    """时间异质性建模：判别当前时间模式与随机打乱的时间模式"""
    
    def __init__(self, c_in, num_nodes, device):
        super().__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_nodes, c_in))
        self.W2 = nn.Parameter(torch.FloatTensor(num_nodes, c_in))
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        
        self.read = AvgReadout()
        self.disc = Discriminator(c_in)
        self.b_xent = nn.BCEWithLogitsLoss()
        self.num_nodes = num_nodes
        self.c_in = c_in
        self._device = device
        
        # 注册buffer用于标签，但batch_size会在forward时动态设置
        self.register_buffer('lbl', torch.zeros(1, 1))  # placeholder

    def _update_labels(self, batch_size):
        """动态更新标签张量"""
        lbl_rl = torch.ones(batch_size, self.num_nodes, device=self._device)
        lbl_fk = torch.zeros(batch_size, self.num_nodes, device=self._device)
        lbl = torch.cat((lbl_rl, lbl_fk), dim=1)
        return lbl

    def forward(self, z1, z2):
        """计算时间对比损失
        
        Args:
            z1, z2: (B, T, N, C) 或 (B, N, C) 两个增强视图的表示
        Returns:
            loss: 标量损失
        """
        # 处理 (B, 1, N, C) 格式 -> (B, N, C)
        if z1.dim() == 4 and z1.shape[1] == 1:
            z1 = z1.squeeze(1)
        if z2.dim() == 4 and z2.shape[1] == 1:
            z2 = z2.squeeze(1)
        
        B, N, C = z1.shape
        
        # z1, z2 现在是 (B, N, C)
        h = (z1 * self.W1 + z2 * self.W2)  # (B, N, C)
        s = self.read(h)

        # 随机打乱batch
        idx = torch.randperm(B, device=h.device)
        shuf_h = h[idx]

        # 动态创建标签
        lbl = self._update_labels(B)
        
        logits = self.disc(s, h, shuf_h)
        loss = self.b_xent(logits, lbl)
        return loss


# =============================================================================
# MLP预测头
# =============================================================================

class MLP(nn.Module):
    """简单MLP预测头
    
    输入: (B, 1, N, C) 时间维度固定为1
    输出: (B, 1, N, out_window) 或 reshape后的预测结果
    """
    def __init__(self, c_in, c_out):
        super().__init__()
        self.fc1 = nn.Conv2d(c_in, c_in // 2, 1)
        self.fc2 = nn.Conv2d(c_in // 2, c_out, 1)

    def forward(self, x):
        # x: (B, 1, N, C) -> permute -> (B, C, 1, N)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        # 输出: (B, c_out, 1, N) -> (B, 1, N, c_out)
        return x.permute(0, 2, 3, 1)


# =============================================================================
# 主模型
# =============================================================================

class STSSL(BaseModel):
    """时空自监督学习模型
    
    数据流: 输入嵌入 -> 时空编码器 -> 预测分支/空间对比/时间对比
    输入格式: (B, T, N, 3) - [时序值, 时刻(day-of-time), 星期(day-of-week)]
    """
    
    model_name = "stssl"

    def __init__(self, num_nodes: int, model_dim: int, output_dim: int = 1,
                 in_window: int = 24, out_window: int = 24,
                 num_layers: int = 2, dropout: float = 0.1,
                 Kt: int = 3, Ks: int = 3,
                 nmb_prototype: int = 32,
                 spatial_temp: float = 0.5,
                 augmentation_percent: float = 0.1,
                 adj_mx: np.ndarray = None,
                 device: torch.device = None,
                 time_intervals: int = 1800,
                 input_dim: int = 3):
        """初始化STSSL模型
        
        Args:
            num_nodes: 图中节点数量
            model_dim: 模型隐层维度
            output_dim: 输出维度
            in_window/out_window: 输入/输出时间窗口
            num_layers: STEncoder层数（目前固定为2）
            dropout: dropout比例
            Kt/Ks: 时间/空间卷积核大小
            nmb_prototype: 空间原型数量
            spatial_temp: 空间对比学习温度参数
            augmentation_percent: 数据增强比例
            adj_mx: 邻接矩阵
            time_intervals: 时间间隔（秒）
            input_dim: 输入特征维度
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.in_window = in_window
        self.out_window = out_window
        self.model_dim = model_dim
        self.dropout = dropout
        self.nmb_prototype = nmb_prototype
        self.spatial_temp = spatial_temp
        self.augmentation_percent = augmentation_percent
        self.device = device or torch.device('cpu')
        self.input_dim = input_dim
        self.st_ssl_loss_weights = [1.0, 0.1, 0.1]  # 默认loss weights
        
        # 输入嵌入层: input_dim -> model_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )
        
        # STEncoder配置
        # 输入是投影后的model_dim维特征
        blocks = [
            [model_dim, model_dim // 2, model_dim],
            [model_dim, model_dim // 2, model_dim]
        ]
        self.encoder = STEncoder(
            Kt=Kt, Ks=Ks, blocks=blocks,
            input_length=in_window, num_nodes=num_nodes, droprate=dropout
        )
        
        # 预测分支: 直接输出out_window步预测
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, out_window * output_dim)
        )
        
        # 空间异质性建模
        self.shm = SpatialHeteroModel(model_dim, nmb_prototype, tau=spatial_temp)
        
        # 时间异质性建模
        self.thm = TemporalHeteroModel(model_dim, num_nodes=num_nodes, device=self.device)
        
        # 邻接矩阵处理
        if adj_mx is not None:
            if isinstance(adj_mx, np.ndarray):
                W = torch.from_numpy(adj_mx).float().to(self.device)
            else:
                W = adj_mx.to(self.device)
            W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
            self.register_buffer('adj_mx', W)
        else:
            self.register_buffer('adj_mx', torch.eye(num_nodes, device=self.device))
        
        # MAE损失
        self.mae = nn.L1Loss()
        
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
        return STSSL(
            num_nodes=num_nodes,
            model_dim=args.input_embedding_dim,
            output_dim=1,
            in_window=args.input_window,
            out_window=args.output_window,
            num_layers=args.num_layers,
            dropout=args.dropout,
            Kt=3, Ks=3,
            nmb_prototype=getattr(args, 'nmb_prototype', 8),
            spatial_temp=getattr(args, 'spatial_temp', 0.5),
            augmentation_percent=getattr(args, 'augmentation_percent', 0.1),
            adj_mx=adj_mx,
            device=device,
            time_intervals=args.time_intervals
        ).to(device)

    def forward(self, batch: dict) -> torch.Tensor:
        """模型前向传播
        
        Args:
            batch['X']: (B, T, N, 3) - [时序值, 时刻, 星期]
        Returns:
            (B, out_window, N, 1) 预测结果
        """
        X = batch['X']
        B, T, N, _ = X.shape
        
        # 输入投影: (B, T, N, 3) -> (B, T, N, model_dim)
        x = self.input_proj(X)
        
        # 时空编码
        enc_out = self.encoder(x, self.adj_mx)  # (B, 1, N, model_dim)
        
        # 预测: (B, 1, N, model_dim) -> (B, 1, N, out_window*output_dim) -> (B, out_window, N, output_dim)
        pred = self.mlp(enc_out)
        pred = pred.reshape(B, N, self.out_window, self.output_dim).permute(0, 2, 1, 3)
        return pred

    def predict(self, batch: dict) -> torch.Tensor:
        """预测接口"""
        return self.forward(batch)

    def calculate_loss(self, batch: dict, loss_weights: list = None) -> tuple:
        """计算STSSL的组合损失
        
        Args:
            batch: 包含 'X' 和 'y' 的字典
            loss_weights: [pred_weight, temporal_weight, spatial_weight]
        Returns:
            total_loss: 总损失
            sep_losses: 分离的损失字典
        """
        if loss_weights is None:
            loss_weights = [1.0, 0.1, 0.1]
        
        X = batch['X']
        y = batch['y']
        B, T, N, _ = X.shape
        
        # 输入投影
        x = self.input_proj(X)  # (B, T, N, model_dim)
        
        # ========== 原始视图 ==========
        repr1 = self.encoder(x, self.adj_mx)  # (B, 1, N, model_dim)
        
        # ========== 拓扑增强视图 ==========
        s_sim_mx = self.encoder.s_sim_mx
        graph2 = aug_topology(s_sim_mx, self.adj_mx, percent=self.augmentation_percent * 2)
        
        # ========== 流量增强视图 ==========
        t_sim_mx = self.encoder.t_sim_mx
        view2 = aug_traffic(t_sim_mx, x, percent=self.augmentation_percent)
        
        # ========== 增强视图编码 ==========
        repr2 = self.encoder(view2, graph2)  # (B, 1, N, model_dim)
        
        # ========== 预测损失 ==========
        # repr1: (B, 1, N, model_dim) -> MLP -> (B, 1, N, out_window*output_dim)
        pred = self.mlp(repr1)  # (B, 1, N, out_window*output_dim)
        pred = pred.squeeze(1)  # (B, N, out_window*output_dim)
        
        if self.output_dim == 1:
            pred = pred.unsqueeze(-1)  # (B, N, out_window, 1)
        else:
            pred = pred.reshape(B, N, self.out_window, self.output_dim)
        pred = pred.transpose(1, 2)  # (B, out_window, N, output_dim)
        
        # y: (B, out_window, N, 1)
        pred_loss = F.l1_loss(pred, y)
        
        # ========== 时间对比损失 ==========
        temporal_loss = self.thm(repr1, repr2)
        
        # ========== 空间对比损失 ==========
        spatial_loss = self.shm(repr1, repr2)
        
        # ========== 总损失 ==========
        total_loss = (loss_weights[0] * pred_loss + 
                      loss_weights[1] * temporal_loss + 
                      loss_weights[2] * spatial_loss)
        
        sep_losses = {
            'pred': pred_loss.item(),
            'temporal': temporal_loss.item(),
            'spatial': spatial_loss.item()
        }
        
        return total_loss, sep_losses
