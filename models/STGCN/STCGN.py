"""
Classic STGCN: Spatio-Temporal Graph Convolutional Networks
From: "Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework 
       for Traffic Forecasting" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from models.base import BaseModel


class GraphConv(nn.Module):
    """First-order Graph Convolution (simplified Chebyshev with K=1)"""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        self.bias = nn.Parameter(torch.zeros(c_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        BT, N, C_in = x.shape

        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(BT, -1, -1)

        x_w = torch.matmul(x, self.weight) + self.bias
        out = torch.bmm(adj, x_w)
        return out


class TemporalConv(nn.Module):
    """Temporal Convolution with Gated Mechanism"""

    def __init__(self, cin: int, cout: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, (1, kernel_size), padding=(0, (kernel_size - 1) // 2))
        self.conv2 = nn.Conv2d(cin, cout, (1, kernel_size), padding=(0, (kernel_size - 1) // 2))
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x: Tensor) -> Tensor:
        x1 = torch.tanh(self.conv1(x))
        x2 = torch.sigmoid(self.conv2(x))
        out = x1 * x2
        out = self.bn(out)
        return out


class STConvBlock(nn.Module):
    """Spatio-Temporal Convolutional Block"""

    def __init__(self, model_dim: int, num_nodes: int, dropout: float = 0.1):
        super().__init__()
        self.model_dim = model_dim

        self.tconv1 = TemporalConv(model_dim, model_dim * 2, kernel_size=3)
        self.gconv1 = GraphConv(model_dim * 2, model_dim * 2)
        self.tconv2 = TemporalConv(model_dim * 2, model_dim, kernel_size=3)
        self.gconv2 = GraphConv(model_dim, model_dim)

        self.ln = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        residual = x

        x = self.tconv1(x)

        B, C, N, T = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B * T, N, C)
        x = F.relu(self.gconv1(x, adj))
        x = x.reshape(B, T, N, C).permute(0, 3, 2, 1)

        x = self.tconv2(x)

        B, C, N, T = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B * T, N, C)
        x = F.relu(self.gconv2(x, adj))
        x = x.reshape(B, T, N, C).permute(0, 3, 2, 1)

        x = self.dropout(x)
        x = x + residual

        return x


class STGCN(BaseModel):
    """Classic Spatio-Temporal Graph Convolutional Network"""

    model_name = "stgcn"

    def __init__(self, num_nodes: int, model_dim: int = 64, output_dim: int = 1,
                 in_window: int = 12, out_window: int = 12,
                 num_layers: int = 2,
                 dropout: float = 0.1, adj_mx: Tensor = None,
                 device: torch.device = None):
        super().__init__()

        self.num_nodes = num_nodes
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.in_window = in_window
        self.out_window = out_window

        self.device = device or torch.device('cpu')

        self.input_proj = nn.Conv2d(3, model_dim, (1, 1))

        self.st_blocks = nn.ModuleList([
            STConvBlock(model_dim, num_nodes, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Conv2d(model_dim, out_window * output_dim, (1, 1))

        if adj_mx is not None:
            if isinstance(adj_mx, np.ndarray):
                W = torch.from_numpy(adj_mx).to(self.device)
            else:
                W = adj_mx.to(self.device)
            W = torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
            A = W + torch.eye(self.num_nodes, device=self.device, dtype=W.dtype)
            deg = A.sum(dim=-1).clamp(min=1e-8)
            d_inv_sqrt = deg.pow(-0.5)
            adj_norm = d_inv_sqrt.unsqueeze(-1) * A * d_inv_sqrt.unsqueeze(-2)
            self.register_buffer('adj_norm', adj_norm)
        else:
            self.register_buffer('adj_norm', torch.eye(num_nodes, device=self.device))

        self.apply(self._init_weights)

    @staticmethod
    def from_args(args, num_nodes, adj_mx, device):
        """从命令行参数创建模型"""
        return STGCN(
            num_nodes=num_nodes,
            model_dim=args.input_embedding_dim,
            output_dim=1,
            in_window=args.input_window,
            out_window=args.output_window,
            num_layers=args.num_layers,
            dropout=args.dropout,
            adj_mx=adj_mx,
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

    def forward(self, batch: dict) -> Tensor:
        x = batch['X']
        B, T_in, N, C = x.shape

        x = x.permute(0, 3, 2, 1)

        x = torch.relu(self.input_proj(x))

        adj = self.adj_norm
        for st_block in self.st_blocks:
            x = st_block(x, adj)

        x = self.output_proj(x)

        return x
