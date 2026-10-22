"""PSTID model components."""

import torch
from torch import nn
import torch.nn.functional as F
from models.base import BaseModel
from .graph import _compute_adj_mx_emb_init
from .layers import MultiLayerPerceptron
from .prototype import ProtoModule


class PSTID(BaseModel):
    """PSTID: 原型时空身份网络

    在 STID 的基础上，通过原型模块增强输入嵌入，减少噪声并提升可解释性。
    """

    model_name = "pstid"

    def __init__(self, num_nodes: int, input_window: int = 24, output_window: int = 24,
                 feature_dim: int = 3, output_dim: int = 1,
                 time_intervals: int = 1800,
                 num_block: int = 2, time_series_emb_dim: int = 64,
                 spatial_emb_dim: int = 64, temp_dim_tid: int = 64, temp_dim_diw: int = 64,
                 if_spatial: bool = True, if_time_in_day: bool = True, if_day_in_week: bool = True,
                 device: torch.device = None,
                 # ProtoModule 参数
                 num_spatial_prototypes: int = 4,
                 num_temporal_prototypes: int = 16,
                 proto_temperature: float = 0.5,
                 use_proto: bool = True,
                 use_spatio: bool = True,
                 use_temporal: bool = True,
                 # 新增参数
                 proto_emb_dim: int = None,
                 # 是否使用 adj_mx 衍生的可学习节点嵌入（消融开关）
                 # True: 使用 adj_mx_emb 作为空间码本 Query，并拼入 hidden_dim
                 # False: 使用 node_emb 作为空间码本 Query，不拼入 hidden_dim
                 use_adj: bool = True,
                 # adj_mx 衍生的可学习节点嵌入维度
                 adj_mx_emb_dim: int = 64,
                 adj_mx: torch.Tensor = None,
                 use_his: bool = True, time_slots=None,
                 use_time_interaction: bool = False,
                 use_proto_change: bool = False, proto_change_context: str = "encoded"):
        super().__init__()

        # ==================== STID 基础参数 ====================
        self.num_nodes = num_nodes
        self.input_window = input_window
        self.output_window = output_window
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.time_intervals = time_intervals
        self.num_block = num_block
        self.time_series_emb_dim = time_series_emb_dim
        self.spatial_emb_dim = spatial_emb_dim
        self.temp_dim_tid = temp_dim_tid
        self.temp_dim_diw = temp_dim_diw
        self.if_spatial = if_spatial
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        self.device = device or torch.device('cpu')

        if time_intervals <= 0 or 86400 % time_intervals != 0:
            raise ValueError('time_intervals must be a positive divisor of 86400')
        self.time_of_day_size = int(86400 / time_intervals)
        slots = torch.as_tensor(
            list(range(self.time_of_day_size)) if time_slots is None else time_slots,
            dtype=torch.long,
        )
        if (slots.ndim != 1 or slots.numel() == 0
                or torch.any(slots < 0) or torch.any(slots >= self.time_of_day_size)
                or torch.any(slots[1:] <= slots[:-1])):
            raise ValueError('time_slots must be sorted, unique daily slot indices')
        self.register_buffer('time_slots', slots, persistent=False)
        self.day_of_week_size = 7

        # ==================== ProtoModule 参数 ====================
        # use_temporal 和 use_spatio 仅控制 ProtoModule 中码本的生成
        # 不影响时间/空间嵌入的使用（嵌入由 if_time_in_day, if_day_in_week, if_spatial 控制）
        self.use_temporal = use_temporal
        self.use_spatio = use_spatio
        self.use_his = use_his

        self.use_proto = use_proto
        self.num_spatial_prototypes = num_spatial_prototypes
        self.num_temporal_prototypes = num_temporal_prototypes
        self.proto_temperature = proto_temperature

        # ==================== STID 嵌入层 ====================
        # 空间嵌入
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.spatial_emb_dim))
        nn.init.xavier_uniform_(self.node_emb)

        # 时间嵌入
        self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.temp_dim_tid))
        nn.init.xavier_uniform_(self.time_in_day_emb)

        self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.temp_dim_diw))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(in_channels=self.input_window, out_channels=self.time_series_emb_dim, kernel_size=(1, 1), bias=True)


        self.use_adj = use_adj
        # ==================== 计算 hidden_dim ====================
        # STID 的 hidden_dim 是 time_series_emb + 所有启用的嵌入
        # 当 use_adj=True 时，adj_mx_emb 也参与主嵌入的残差，因此 hidden_dim 必须包含 adj_mx_emb_dim
        # 当 use_adj=False 时，不使用 adj_mx_emb，hidden_dim 不包含 adj_mx_emb_dim
        self.time_series_dim = self.time_series_emb_dim
        # 先把 adj_mx_emb_dim 绑定到 self 上，方便 hidden_dim 公式引用
        self.adj_mx_emb_dim = adj_mx_emb_dim
        self.hidden_dim = (self.time_series_emb_dim + self.spatial_emb_dim
                           + self.temp_dim_tid + self.temp_dim_diw)
        if self.use_adj:
            self.hidden_dim += self.adj_mx_emb_dim

        # ProtoModule 的 proto_dim：如果提供了 proto_emb_dim 则使用它，否则使用 hidden_dim
        self.proto_emb_dim = proto_emb_dim if proto_emb_dim is not None else self.hidden_dim
        self.proto_dim = self.proto_emb_dim

        # adj_mx 衍生的可学习节点嵌入（仅 use_adj=True 时创建）
        self.adj_mx_emb = None
        if self.use_adj:
            if adj_mx is not None:
                init_emb = _compute_adj_mx_emb_init(adj_mx, adj_mx_emb_dim).to(device)
                if init_emb.shape[1] < adj_mx_emb_dim:
                    padding = torch.randn(init_emb.shape[0], adj_mx_emb_dim - init_emb.shape[1], device=init_emb.device)
                    init_emb = torch.cat([init_emb, padding], dim=1)
                elif init_emb.shape[1] > adj_mx_emb_dim:
                    init_emb = init_emb[:, :adj_mx_emb_dim]
                self.adj_mx_emb = nn.Parameter(init_emb)
            else:
                self.adj_mx_emb = nn.Parameter(torch.empty(self.num_nodes, self.adj_mx_emb_dim))
                nn.init.xavier_uniform_(self.adj_mx_emb)

        # ==================== ProtoModule ====================
        # w/o P：不创建增强分支，拼接后的 embeddings 直接输入 MLP。
        self.proto_module = None
        if self.use_proto:
            self.proto_module = ProtoModule(
                num_nodes=self.num_nodes,
                proto_dim=self.proto_dim,
                n_spatial=self.num_spatial_prototypes,
                n_temporal=self.num_temporal_prototypes,
                use_spatio=self.use_spatio,
                use_temporal=self.use_temporal,
                use_his=self.use_his,
                use_proto=self.use_proto,
                use_time_interaction=use_time_interaction,
                proto_temp=self.proto_temperature,
                adj_mx_emb_dim=self.adj_mx_emb_dim,
                spatial_emb_dim=self.spatial_emb_dim,
                use_adj=self.use_adj,
                time_series_dim=self.time_series_emb_dim,
                time_of_day_size=self.time_of_day_size,
                day_of_week_size=self.day_of_week_size,
                temp_dim_tid=self.temp_dim_tid,
                temp_dim_diw=self.temp_dim_diw,
                proto_emb_dim=self.proto_emb_dim,
                hidden_dim=self.hidden_dim,  # 传入 hidden_dim 用于 proto_out_proj
            )

        # ==================== MLP Layers ====================
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_block)]
        )

        # 逐步时间特征只通过轻量残差修正预测，不改变多步主回归头。
        self.future_readout = None
        if self.use_proto and self.use_temporal:
            self.future_readout = nn.Linear(self.hidden_dim, 1, bias=False)
            nn.init.zeros_(self.future_readout.weight)

        # 当前窗口状态调制未来-历史原型差，提供节点相关的非线性变化修正。
        if proto_change_context not in ('raw', 'encoded'):
            raise ValueError('proto_change_context must be raw or encoded')
        self.proto_change_context = proto_change_context
        self.proto_change_query = self.proto_change_delta = self.proto_change_readout = None
        if self.use_proto and self.use_temporal and use_proto_change:
            rank = min(16, self.hidden_dim)
            with torch.random.fork_rng(devices=[]):
                self.proto_change_query = nn.Linear(self.hidden_dim, rank)
                self.proto_change_delta = nn.Linear(self.hidden_dim, rank, bias=False)
                self.proto_change_readout = nn.Linear(2 * rank, 1, bias=False)
            nn.init.zeros_(self.proto_change_readout.weight)

        # ==================== Regression Layer ====================
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.output_window,
            kernel_size=(1, 1),
            bias=True
        )

        # ==================== 权重初始化 ====================
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_time(self, calendar):
        """历史和未来共享嵌入表，每个时间步按自身 tid、diw 查表。"""
        tid_idx = torch.round(calendar[..., 0] * self.time_of_day_size).long().clamp(0, self.time_of_day_size - 1)
        diw_idx = torch.round(calendar[..., 1] * self.day_of_week_size).long().clamp(0, self.day_of_week_size - 1)
        # 专用 embedding backward 避免重复日期索引的高级索引梯度累加开销。
        return F.embedding(tid_idx, self.time_in_day_emb), F.embedding(diw_idx, self.day_in_week_emb)

    def _future_calendar(self, x):
        """从历史末步推算未来日程，返回 (B, output_window, N, 2)，不使用标签。"""
        tid = torch.round(x[:, -1, :, 1] * self.time_of_day_size).long()
        diw = torch.round(x[:, -1, :, 2] * self.day_of_week_size).long()
        position = torch.searchsorted(self.time_slots, tid.contiguous())
        # 输入时间应属于配置的日程；此处不做 Tensor -> Python 判断，避免每批 GPU 同步。
        offsets = torch.arange(1, self.output_window + 1, device=x.device)
        future_position = position[:, None, :] + offsets[None, :, None]
        future_tid = self.time_slots[future_position % self.time_slots.numel()]
        future_diw = (diw[:, None, :] + future_position // self.time_slots.numel()) % 7
        return torch.stack((future_tid.to(x.dtype) / self.time_of_day_size,
                            future_diw.to(x.dtype) / self.day_of_week_size), dim=-1)

    def _get_embeddings(self, batch):
        """获取窗口级嵌入表示 (B, 1, N, hidden_dim)

        整个历史窗口编码流量，时间身份只取历史末步
        adj_mx_emb 也参与主嵌入的残差（拼接到 hidden 中），由 ProtoModule 的空间码本同时使用
        """
        input_data = batch['X']
        batch_size, _, num_nodes, _ = input_data.shape

        # ========== 时间序列嵌入 ==========
        # 整个历史窗口的流量投影为 (B, 1, N, time_series_emb_dim)
        time_series = input_data[..., :1]  # (B, T, N, 1)
        time_series = time_series.transpose(1, 2).contiguous()
        time_series = time_series.view(
        batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(time_series).permute(0, 3, 2, 1)

        # ========== 时间嵌入 ==========
        # 历史只取末步时间身份；未来窗口在 forward 中逐步编码
        time_in_day_emb, day_in_week_emb = self._encode_time(input_data[:, -1:, :, 1:3])

        # ========== 空间嵌入 ==========
        node_emb = self.node_emb  # (N, spatial_emb_dim)

        # ========== 拼接所有嵌入 ==========
        # 收集所有需要拼接的嵌入（4D）
        emb_list = [time_series_emb]
        emb_list.append(time_in_day_emb)
        emb_list.append(day_in_week_emb)

        node_emb_4d = node_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1, -1)  # (B, 1, N, D)
        emb_list.append(node_emb_4d)

        # adj_mx_emb 作为主嵌入的残差项参与拼接（仅 use_adj=True 时）
        if self.use_adj and self.adj_mx_emb is not None:
            adj_mx_emb_4d = self.adj_mx_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1, -1)  # (B, 1, N, D_adj)
            emb_list.append(adj_mx_emb_4d)

        # (B, 1, N, hidden_dim)
        hidden = torch.cat(emb_list, dim=-1)

        # 返回拼接后的嵌入以及各分量（供原型模块使用）
        return hidden, time_series_emb, time_in_day_emb, day_in_week_emb, node_emb, self.adj_mx_emb


    def forward(self, batch):
        """Predict (B, output_window, N, 1) from a batch of historical inputs."""
        hidden, time_series_emb, tid_emb, diw_emb, spatio_emb, _ = self._get_embeddings(batch)

        # 主干维持原 PSTID 的窗口级特征和各预测步独立的回归参数。
        hidden = hidden[:, -1:]
        correction = None
        change = change_context = None
        if self.proto_module is not None:
            future_tid = future_diw = None
            if self.use_temporal:
                future_tid, future_diw = self._encode_time(self._future_calendar(batch['X']))
                series = time_series_emb[:, :1].expand(
                    -1, 1 + self.output_window, -1, -1)
            else:
                series = time_series_emb[:, :1]
            proto_enhanced = self.proto_module(
                time_series_emb=series,
                adj_mx_emb=self.adj_mx_emb,
                node_emb=spatio_emb,
                tid_emb=tid_emb,
                diw_emb=diw_emb,
                future_tid_emb=future_tid,
                future_diw_emb=future_diw,
            )
            if proto_enhanced is not None:
                if self.use_temporal:
                    history_proto = proto_enhanced[:, :1]
                    hidden = hidden + history_proto[:, -1:]
                    # 历史保持单个窗口身份；未来各步独立修正对应预测。
                    future_proto = proto_enhanced[:, 1:]
                    correction = self.future_readout(future_proto)
                    if self.proto_change_readout is not None:
                        change_context = hidden
                        change = F.gelu(self.proto_change_delta(future_proto - history_proto))
                else:
                    hidden = hidden + proto_enhanced

        hidden = self.encoder(hidden.permute(0, 3, 2, 1).contiguous())
        if change is not None:
            context = (hidden.permute(0, 3, 2, 1) if self.proto_change_context == 'encoded'
                       else change_context)
            current = torch.tanh(self.proto_change_query(context))
            correction = correction + self.proto_change_readout(
                torch.cat([change, change * current], dim=-1))
        prediction = self.regression_layer(hidden)
        return prediction if correction is None else prediction + correction


    @staticmethod
    def from_args(args, num_nodes, adj_mx, device):
        """从命令行参数创建 PSTID 模型实例

        Args:
            args: 命令行参数或参数字典
            num_nodes: 节点数量
            adj_mx: 邻接矩阵（用于谱嵌入初始化 adj_mx_emb）
            device: 计算设备

        Returns:
            PSTID 模型实例
        """
        def _get_arg(a, key, default):
            if isinstance(a, dict):
                return a.get(key, default)
            return getattr(a, key, default)

        return PSTID(
            num_nodes=num_nodes,
            time_slots=_get_arg(args, 'pstid_time_slots', None),
            input_window=_get_arg(args, 'input_window', 24),
            output_window=_get_arg(args, 'output_window', 24),
            feature_dim=3,
            output_dim=1,
            time_intervals=_get_arg(args, 'time_intervals', 1800),
            num_block=_get_arg(args, 'num_layers', 3),
            time_series_emb_dim=_get_arg(args, 'input_embedding_dim', 32),
            spatial_emb_dim=_get_arg(args, 'node_emb_dim', 16),
            temp_dim_tid=_get_arg(args, 'tid', 16),
            temp_dim_diw=_get_arg(args, 'diw', 16),
            if_spatial=_get_arg(args, 'if_spatial', True),
            if_time_in_day=_get_arg(args, 'if_time_in_day', True),
            if_day_in_week=_get_arg(args, 'if_day_in_week', True),
            device=device,
            # ProtoModule 参数
            num_spatial_prototypes=_get_arg(args, 'num_spatial_prototypes', 16),
            num_temporal_prototypes=_get_arg(args, 'num_temporal_prototypes', 8),
            proto_temperature=_get_arg(args, 'proto_temperature', 0.5),
            use_proto=_get_arg(args, 'use_proto', True),
            use_adj=_get_arg(args, 'use_adj', True),
            use_spatio=_get_arg(args, 'use_spatio', True),
            use_temporal=_get_arg(args, 'use_temporal', True),
            use_his=_get_arg(args, 'use_his', True),
            use_time_interaction=bool(_get_arg(args, 'use_time_interaction', False)),
            use_proto_change=bool(_get_arg(args, 'use_proto_change', False)),
            proto_change_context=_get_arg(args, 'proto_change_context', 'raw'),
            # 新增参数
            proto_emb_dim=_get_arg(args, 'proto_emb_dim', 64),
            # adj_mx 衍生的可学习节点嵌入维度
            adj_mx_emb_dim=_get_arg(args, 'adj_mx_emb_dim', 64),
            # 传入 adj_mx 用于谱嵌入初始化
            adj_mx=adj_mx,
        )
