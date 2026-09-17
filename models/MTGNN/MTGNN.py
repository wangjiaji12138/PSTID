import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from torch.nn import init

from models.base import BaseModel


class NConv(nn.Module):
    def forward(self, x, adj):
        x = torch.einsum('ncwl,vw->ncvl', (x, adj))
        return x.contiguous()


class DyNconv(nn.Module):
    def forward(self, x, adj):
        x = torch.einsum('ncvl,nvwl->ncwl', (x, adj))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class MixProp(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super().__init__()
        self.nconv = NConv()
        self.mlp = Linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super().__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        outputs = []
        for i in range(len(self.kernel_set)):
            out = self.tconv[i](input)
            outputs.append(out)
        # Find min temporal length and pad/truncate to match
        min_len = min(o.size(3) for o in outputs)
        outputs = [o[..., -min_len:] for o in outputs]
        x = torch.cat(outputs, dim=1)
        return x


class GraphConstructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super().__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0), dtype=adj.dtype, device=adj.device)
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, torch.ones_like(s1))
        adj = adj * mask
        return adj


class LayerNorm(nn.Module):
    """Channel-wise Layer Normalization for (B, C, N, T) tensors.
    Normalize over C dimension for each (node, time) position."""
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)  # (C,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, inputs, idx):
        # inputs: (B, C, N, T)
        # Reshape to (B*N*T, C) for channel-wise normalization
        B, C, N, T = inputs.shape
        inputs_flat = inputs.permute(0, 2, 3, 1).reshape(-1, C)  # (B*N*T, C)
        if self.elementwise_affine:
            out = F.layer_norm(inputs_flat, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            out = F.layer_norm(inputs_flat, self.normalized_shape, self.weight, self.bias, self.eps)
        return out.reshape(B, N, T, C).permute(0, 3, 1, 2)  # back to (B, C, N, T)


class MTGNN(BaseModel):
    """MTGNN: Multivariate Time Series Graph Neural Network."""

    model_name = "mtgnn"

    def __init__(self, num_nodes: int, feature_dim: int = 3,
                 output_dim: int = 1, in_window: int = 12, out_window: int = 12,
                 conv_channels: int = 32, residual_channels: int = 32,
                 skip_channels: int = 64, end_channels: int = 128,
                 gcn_depth: int = 2, dropout: float = 0.3,
                 subgraph_size: int = 20, node_dim: int = 40,
                 layers: int = 3, propalpha: float = 0.05,
                 tanhalpha: float = 3, dilation_exponential: int = 1,
                 layer_norm_affline: bool = True,
                 adj_mx: np.ndarray = None, device: torch.device = None):
        super().__init__()

        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.in_window = in_window
        self.out_window = out_window
        self.gcn_depth = gcn_depth
        self.dropout = dropout
        self.subgraph_size = subgraph_size
        self.node_dim = node_dim
        self.dilation_exponential = dilation_exponential
        self.layers = layers
        self.propalpha = propalpha
        self.tanhalpha = tanhalpha
        self.layer_norm_affline = layer_norm_affline

        self.device = device or torch.device('cpu')

        self.predefined_A = torch.tensor(adj_mx - np.eye(self.num_nodes)).to(self.device) if adj_mx is not None else None
        self.static_feat = None

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=self.feature_dim, out_channels=residual_channels, kernel_size=(1, 1))

        self.gc = GraphConstructor(self.num_nodes, self.subgraph_size, self.node_dim,
                                 self.device, alpha=self.tanhalpha, static_feat=self.static_feat)

        kernel_size = 7
        if self.dilation_exponential > 1:
            self.receptive_field = int(1 + (kernel_size - 1) * (self.dilation_exponential ** self.layers - 1)
                                       / (self.dilation_exponential - 1))
        else:
            self.receptive_field = self.layers * (kernel_size - 1) + 1

        for j in range(1, self.layers + 1):
            if self.dilation_exponential > 1:
                rf_size_j = 1 + (kernel_size - 1) * ((self.dilation_exponential ** j - 1) / (self.dilation_exponential - 1))
            else:
                rf_size_j = 1 + j * (kernel_size - 1)

            self.filter_convs.append(DilatedInception(residual_channels, conv_channels, dilation_factor=1))
            self.gate_convs.append(DilatedInception(residual_channels, conv_channels, dilation_factor=1))
            self.residual_convs.append(nn.Conv2d(conv_channels, residual_channels, kernel_size=(1, 1)))

            self.skip_convs.append(nn.Conv2d(conv_channels, skip_channels, kernel_size=(1, self.receptive_field - rf_size_j + 1)))
            self.gconv1.append(MixProp(conv_channels, residual_channels, self.gcn_depth, self.dropout, self.propalpha))
            self.gconv2.append(MixProp(conv_channels, residual_channels, self.gcn_depth, self.dropout, self.propalpha))

            self.norm.append(LayerNorm(residual_channels,
                                      elementwise_affine=self.layer_norm_affline))

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, self.out_window, kernel_size=(1, 1), bias=True)
        self.skip0 = nn.Conv2d(self.feature_dim, skip_channels, kernel_size=(1, self.receptive_field), bias=True)
        self.skipE = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, 1), bias=True)

        self.idx = torch.arange(self.num_nodes).to(self.device)

        self._init_weights()

    @staticmethod
    def from_args(args, num_nodes, adj_mx, device):
        """从命令行参数创建MTGNN模型实例"""
        adj_mx_np = adj_mx.cpu().numpy() if isinstance(adj_mx, torch.Tensor) else adj_mx
        return MTGNN(
            num_nodes=num_nodes,
            feature_dim=1,  # 固定为1，只使用时序值特征
            output_dim=1,
            in_window=args.input_window,
            out_window=args.output_window,
            conv_channels=args.conv_channels,
            residual_channels=args.residual_channels,
            skip_channels=args.skip_channels,
            end_channels=args.end_channels,
            gcn_depth=args.gcn_depth,
            dropout=args.dropout,
            subgraph_size=args.subgraph_size,
            node_dim=args.node_dim,
            layers=args.num_layers,
            propalpha=args.propalpha,
            tanhalpha=args.tanhalpha,
            dilation_exponential=args.dilation_exponential,
            adj_mx=adj_mx_np,
            device=device
        ).to(device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch):
        inputs = batch['X']
        inputs = inputs.transpose(1, 3)

        if self.in_window < self.receptive_field:
            inputs = nn.functional.pad(inputs, (self.receptive_field - self.in_window, 0, 0, 0))

        adp = self.gc(self.idx)

        x = self.start_conv(inputs)
        skip = self.skip0(F.dropout(inputs, self.dropout, training=self.training))

        for i in range(self.layers):
            residual = x
            filters = torch.tanh(self.filter_convs[i](x))
            gate = torch.sigmoid(self.gate_convs[i](x))
            x = filters * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x, self.idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

    def predict(self, batch):
        return self.forward(batch)
