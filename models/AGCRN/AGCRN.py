import torch
import torch.nn.functional as F
import torch.nn as nn
from models.base import BaseModel


# wangjiaji：
# 之前的_init_weights 方法只初始化了标准 PyTorch 层 (nn.Linear, nn.Conv*, nn.LayerNorm)，没有初始化这些自定义参数（AVWGCN 层使用了自定义的 nn.Parameter 权重）
# 现在修改了

class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     # b, N, dim_out
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, num_nodes: int, feature_dim: int, hidden_dim: int,
                 embed_dim: int, num_layers: int, cheb_k: int):
        super(AVWDCRNN, self).__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        assert self.num_layers >= 1, 'At least one DCRNN layer in the Encoder.'

        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(self.num_nodes, self.feature_dim,
                                          self.hidden_dim, self.cheb_k, self.embed_dim))
        for _ in range(1, self.num_layers):
            self.dcrnn_cells.append(AGCRNCell(self.num_nodes, self.hidden_dim,
                                              self.hidden_dim, self.cheb_k, self.embed_dim))

    def forward(self, x, init_state, node_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.num_nodes and x.shape[3] == self.feature_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      # (num_layers, B, N, hidden_dim)


class AGCRN(BaseModel):
    """AGCRN: Adaptive Graph Convolutional Recurrent Network"""

    model_name = "agcrn"

    def __init__(self, num_nodes: int, feature_dim: int, hidden_dim: int,
                 output_dim: int = 1, in_window: int = 12, out_window: int = 12,
                 embed_dim: int = 10, num_layers: int = 2, cheb_k: int = 2,
                 device: torch.device = None):
        super().__init__()

        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.in_window = in_window
        self.out_window = out_window
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.cheb_k = cheb_k

        self.device = device or torch.device('cpu')

        self.node_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.embed_dim), requires_grad=True
        )

        self.encoder = AVWDCRNN(
            num_nodes=self.num_nodes,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            cheb_k=self.cheb_k
        )

        self.end_conv = nn.Conv2d(
            1, self.out_window * self.output_dim,
            kernel_size=(1, self.hidden_dim), bias=True
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, AVWGCN):
            nn.init.xavier_uniform_(m.weights_pool)
            nn.init.zeros_(m.bias_pool)

    @staticmethod
    def from_args(args, num_nodes, adj_mx, device):
        return AGCRN(
            num_nodes=num_nodes,
            feature_dim=1,  # 固定为1，只使用时序值特征
            hidden_dim=args.input_embedding_dim,
            output_dim=1,
            in_window=args.input_window,
            out_window=args.output_window,
            embed_dim=args.embed_dim if hasattr(args, 'embed_dim') else 10,
            num_layers=args.num_layers,
            cheb_k=args.cheb_k if hasattr(args, 'cheb_k') else 2,
            device=device
        ).to(device)

    def forward(self, batch):
        source = batch['X']

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)
        output = output[:, -1:, :, :]

        output = self.end_conv(output)
        output = output.squeeze(-1).reshape(-1, self.out_window, self.output_dim, self.num_nodes)
        output = output.permute(0, 1, 3, 2)
        return output

    def predict(self, batch):
        return self.forward(batch)
