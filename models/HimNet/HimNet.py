"""
HimNet: Hierarchical Multi-scale Graph Network
基于时空异质性建模的交通预测模型
"""

import torch
import torch.nn as nn
import numpy as np

from models.base import BaseModel


class HimGCN(nn.Module):
    def __init__(self, input_dim, output_dim, cheb_k, embed_dim, meta_axis=None):
        super().__init__()
        self.cheb_k = cheb_k
        self.meta_axis = meta_axis.upper() if meta_axis else None

        if meta_axis:
            self.weights_pool = nn.init.xavier_normal_(
                nn.Parameter(
                    torch.FloatTensor(embed_dim, cheb_k * input_dim, output_dim)
                )
            )
            self.bias_pool = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(embed_dim, output_dim))
            )
        else:
            self.weights = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(cheb_k * input_dim, output_dim))
            )
            self.bias = nn.init.constant_(
                nn.Parameter(torch.FloatTensor(output_dim)), val=0
            )

    def forward(self, x, support, embeddings):
        x_g = []

        if support.dim() == 2:
            graph_list = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            for graph in graph_list:
                x_g.append(torch.einsum("nm,bmc->bnc", graph, x))
        elif support.dim() == 3:
            graph_list = [
                torch.eye(support.shape[1])
                .repeat(support.shape[0], 1, 1)
                .to(support.device),
                support,
            ]
            for k in range(2, self.cheb_k):
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            for graph in graph_list:
                x_g.append(torch.einsum("bnm,bmc->bnc", graph, x))
        x_g = torch.cat(x_g, dim=-1)

        if self.meta_axis:
            if self.meta_axis == "T":
                weights = torch.einsum(
                    "bd,dio->bio", embeddings, self.weights_pool
                )  # B, cheb_k*in_dim, out_dim
                bias = torch.matmul(embeddings, self.bias_pool)  # B, out_dim
                x_gconv = (
                    torch.einsum("bni,bio->bno", x_g, weights) + bias[:, None, :]
                )  # B, N, out_dim
            elif self.meta_axis == "S":
                weights = torch.einsum(
                    "nd,dio->nio", embeddings, self.weights_pool
                )  # N, cheb_k*in_dim, out_dim
                bias = torch.matmul(embeddings, self.bias_pool)
                x_gconv = (
                    torch.einsum("bni,nio->bno", x_g, weights) + bias
                )  # B, N, out_dim
            elif self.meta_axis == "ST":
                weights = torch.einsum(
                    "bnd,dio->bnio", embeddings, self.weights_pool
                )  # B, N, cheb_k*in_dim, out_dim
                bias = torch.einsum("bnd,do->bno", embeddings, self.bias_pool)
                x_gconv = (
                    torch.einsum("bni,bnio->bno", x_g, weights) + bias
                )  # B, N, out_dim

        else:
            x_gconv = torch.einsum("bni,io->bno", x_g, self.weights) + self.bias

        return x_gconv


class HimGCRU(nn.Module):
    def __init__(
        self, num_nodes, input_dim, output_dim, cheb_k, embed_dim, meta_axis="S"
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = output_dim
        self.gate = HimGCN(
            input_dim + self.hidden_dim, 2 * output_dim, cheb_k, embed_dim, meta_axis
        )
        self.update = HimGCN(
            input_dim + self.hidden_dim, output_dim, cheb_k, embed_dim, meta_axis
        )

    def forward(self, x, state, support, embeddings):
        # x: B, N, input_dim
        # state: B, N, hidden_dim
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, support, embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, support, embeddings))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.num_nodes, self.hidden_dim)


class HimEncoder(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_dim,
        output_dim,
        cheb_k,
        num_layers,
        embed_dim,
        meta_axis="S",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [HimGCRU(num_nodes, input_dim, output_dim, cheb_k, embed_dim, meta_axis)]
            + [
                HimGCRU(num_nodes, output_dim, output_dim, cheb_k, embed_dim, meta_axis)
                for _ in range(1, num_layers)
            ]
        )

    def forward(self, x, support, embeddings):
        # x: (B, T, N, C)
        batch_size = x.shape[0]
        in_steps = x.shape[1]

        current_input = x
        output_hidden = []
        for cell in self.cells:
            state = cell.init_hidden_state(batch_size).to(x.device)
            inner_states = []
            for t in range(in_steps):
                state = cell(current_input[:, t, :, :], state, support, embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_input = torch.stack(inner_states, dim=1)

        # current_input: the outputs of last layer: (B, T, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        return current_input, output_hidden


class HimDecoder(nn.Module):
    def __init__(
        self,
        num_nodes,
        input_dim,
        output_dim,
        cheb_k,
        num_layers,
        embed_dim,
        meta_axis="ST",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [HimGCRU(num_nodes, input_dim, output_dim, cheb_k, embed_dim, meta_axis)]
            + [
                HimGCRU(num_nodes, output_dim, output_dim, cheb_k, embed_dim, meta_axis)
                for _ in range(1, num_layers)
            ]
        )

    def forward(self, xt, init_state, support, embeddings):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        current_input = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.cells[i](current_input, init_state[i], support, embeddings)
            output_hidden.append(state)
            current_input = state
        return current_input, output_hidden


class HimNet(BaseModel):
    """Hierarchical Multi-scale Graph Network
    
    输入格式: (B, T, N, 3) - [时序值, 时刻(day-of-time), 星期(day-of-week)]
    输出格式: (B, out_window, N, 1) 预测结果
    """
    
    model_name = "himnet"

    def __init__(
        self,
        num_nodes: int,
        input_dim: int = 3,
        output_dim: int = 1,
        out_steps: int = 12,
        in_window: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 1,
        cheb_k: int = 2,
        ycov_dim: int = 2,
        tod_embedding_dim: int = 8,
        dow_embedding_dim: int = 8,
        node_embedding_dim: int = 16,
        st_embedding_dim: int = 16,
        tf_decay_steps: int = 4000,
        use_teacher_forcing: bool = True,
        adj_mx: np.ndarray = None,
        device: torch.device = None,
        time_intervals: int = 1800,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.out_steps = out_steps
        self.in_window = in_window
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.node_embedding_dim = node_embedding_dim
        self.st_embedding_dim = st_embedding_dim
        self.tf_decay_steps = tf_decay_steps
        self.use_teacher_forcing = use_teacher_forcing
        self.device = device or torch.device('cpu')
        self.time_intervals = time_intervals
        
        # 根据时间间隔计算 time_of_day_size
        self.time_of_day_size = int((24 * 60 * 60) / time_intervals)
        self.day_of_week_size = 7
        
        # 空间编码器
        self.encoder_s = HimEncoder(
            num_nodes,
            input_dim,
            hidden_dim,
            cheb_k,
            num_layers,
            node_embedding_dim,
            meta_axis="S",
        )
        
        # 时间编码器
        self.encoder_t = HimEncoder(
            num_nodes,
            input_dim,
            hidden_dim,
            cheb_k,
            num_layers,
            tod_embedding_dim + dow_embedding_dim,
            meta_axis="T",
        )

        # 解码器
        self.decoder = HimDecoder(
            num_nodes,
            output_dim + ycov_dim,
            hidden_dim,
            cheb_k,
            num_layers,
            st_embedding_dim,
            meta_axis="ST",
        )

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        # 时间嵌入 - 使用 nn.Embedding 保持原始设计
        self.tod_embedding = nn.Embedding(288, tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        
        # 节点嵌入
        self.node_embedding = nn.init.xavier_normal_(
            nn.Parameter(torch.empty(self.num_nodes, node_embedding_dim))
        )
        
        # 时空投影
        self.st_proj = nn.Linear(self.hidden_dim, st_embedding_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """Xavier均匀初始化"""
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def compute_sampling_threshold(self, batches_seen):
        return self.tf_decay_steps / (
            self.tf_decay_steps + np.exp(batches_seen / self.tf_decay_steps)
        )

    def forward(self, batch: dict, batches_seen: int = None) -> torch.Tensor:
        """模型前向传播
        
        Args:
            batch: 包含 'X' 和 'y' 的字典
            batches_seen: 用于 curriculum learning 的批次计数
        Returns:
            (B, out_window, N, output_dim) 预测结果
        """
        x = batch['X']  # (B, T, N, 3) - [时序值, 时刻, 星期]
        y = batch.get('y', None)  # (B, out_window, N, output_dim) 可选
        
        B, T, N, _ = x.shape
        
        # 构建节点相似度矩阵作为邻接矩阵
        support = torch.softmax(
            torch.relu(self.node_embedding @ self.node_embedding.T), dim=-1
        )
        support = support.to(x.device)

        # 提取时间特征用于时间编码器
        tod = x[:, -1, 0, 1]  # 使用最后一个时刻的 day-of-time
        dow = x[:, -1, 0, 2]  # 使用最后一个时刻的 day-of-week
        tod_emb = self.tod_embedding((tod * 288).long())
        dow_emb = self.dow_embedding(dow.long())
        time_embedding = torch.cat([tod_emb, dow_emb], dim=-1)

        # 空间编码
        h_s, _ = self.encoder_s(x, support, self.node_embedding)  # 使用完整输入
        # 时间编码
        h_t, _ = self.encoder_t(x, support, time_embedding)  # 使用完整输入
        # 融合最后一层状态
        h_last = (h_s + h_t)[:, -1, :, :]  # (B, N, hidden_dim)

        # 时空嵌入用于解码器
        st_embedding = self.st_proj(h_last)  # (B, N, st_emb_dim)
        support = torch.softmax(
            torch.relu(torch.einsum("bnc,bmc->bnm", st_embedding, st_embedding)),
            dim=-1,
        )

        # 解码器初始状态
        ht_list = [h_last] * self.num_layers
        
        # 解码器输入: 全零初始
        go = torch.zeros((B, self.num_nodes, self.output_dim), device=x.device)
        
        # 创建未来时间特征的虚拟输入 (用于兼容框架)
        # 这里我们使用与输入相同的时间特征简化处理
        y_cov = torch.zeros((B, self.out_steps, self.num_nodes, self.ycov_dim), device=x.device)
        
        out = []
        for t in range(self.out_steps):
            h_de, ht_list = self.decoder(
                torch.cat([go, y_cov[:, t, ...]], dim=-1),
                ht_list,
                support,
                st_embedding,
            )
            go = self.out_proj(h_de)
            out.append(go)
            
            # Teacher forcing (如果提供了 labels)
            if self.training and self.use_teacher_forcing and y is not None and batches_seen is not None:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = y[:, t, :, :]
        
        output = torch.stack(out, dim=1)  # (B, out_steps, N, output_dim)
        return output

    def predict(self, batch: dict) -> torch.Tensor:
        """预测接口"""
        return self.forward(batch)

    def calculate_loss(self, batch: dict, batches_seen: int = None) -> tuple:
        """计算损失
        
        Args:
            batch: 包含 'X' 和 'y' 的字典
            batches_seen: 用于 curriculum learning
        Returns:
            total_loss: 总损失
            sep_losses: 分离的损失字典
        """
        y_true = batch['y']
        y_pred = self.forward(batch, batches_seen)
        
        # MAE 损失
        loss = torch.mean(torch.abs(y_pred - y_true))
        
        sep_losses = {'mae': loss.item()}
        return loss, sep_losses

    @staticmethod
    def from_args(args, num_nodes, adj_mx, device):
        """从命令行参数创建模型实例"""
        return HimNet(
            num_nodes=num_nodes,
            input_dim=3,
            output_dim=1,
            out_steps=args.output_window,
            in_window=args.input_window,
            hidden_dim=args.input_embedding_dim,
            num_layers=args.num_layers,
            cheb_k=getattr(args, 'cheb_k', 2),
            ycov_dim=2,
            tod_embedding_dim=getattr(args, 'tod_embedding_dim', 8),
            dow_embedding_dim=getattr(args, 'dow_embedding_dim', 8),
            node_embedding_dim=getattr(args, 'node_embedding_dim', 16),
            st_embedding_dim=getattr(args, 'st_embedding_dim', 16),
            tf_decay_steps=getattr(args, 'tf_decay_steps', 4000),
            use_teacher_forcing=getattr(args, 'use_teacher_forcing', True),
            adj_mx=adj_mx,
            device=device,
        ).to(device)


if __name__ == "__main__":
    from torchinfo import summary

    model = HimNet(num_nodes=207).cpu()
    summary(model, [[64, 12, 207, 3], [64, 12, 207, 2]], device="cpu")
