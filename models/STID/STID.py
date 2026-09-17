import torch
import torch.nn as nn

from models.base import BaseModel


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        hidden = hidden + input_data
        return hidden


class STID(BaseModel):
    """STID: Spatial-Temporal Identity Network."""

    model_name = "stid"

    def __init__(self, num_nodes: int, input_window: int = 24, output_window: int = 24,
                 feature_dim: int = 3, output_dim: int = 1,
                 time_intervals: int = 1800,
                 num_block: int = 2, time_series_emb_dim: int = 64,
                 spatial_emb_dim: int = 64, temp_dim_tid: int = 64, temp_dim_diw: int = 64,
                 if_spatial: bool = True, if_time_in_day: bool = True, if_day_in_week: bool = True,
                 device: torch.device = None):
        super().__init__()

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

        self.time_of_day_size = int((24 * 60 * 60) / time_intervals)
        self.day_of_week_size = 7

        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.spatial_emb_dim))
            nn.init.xavier_uniform_(self.node_emb)

        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)

        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.output_dim * self.input_window,
            out_channels=self.time_series_emb_dim,
            kernel_size=(1, 1),
            bias=True
        )

        hidden_dim = self.time_series_emb_dim
        if self.if_spatial:
            hidden_dim += self.spatial_emb_dim
        if self.if_time_in_day:
            hidden_dim += self.temp_dim_tid
        if self.if_day_in_week:
            hidden_dim += self.temp_dim_diw

        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(hidden_dim, hidden_dim) for _ in range(self.num_block)]
        )

        self.regression_layer = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=self.output_window,
            kernel_size=(1, 1),
            bias=True
        )

        self._init_weights()

    @staticmethod
    def from_args(args, num_nodes, adj_mx, device):
        """从命令行参数创建STID模型实例"""
        return STID(
            num_nodes=num_nodes,
            input_window=args.input_window,
            output_window=args.output_window,
            feature_dim=3,
            output_dim=1,
            time_intervals=args.time_intervals,
            num_block=args.num_block,
            time_series_emb_dim=args.time_series_emb_dim,
            spatial_emb_dim=args.spatial_emb_dim,
            temp_dim_tid=args.temp_dim_tid,
            temp_dim_diw=args.temp_dim_diw,
            if_spatial=args.if_spatial,
            if_time_in_day=args.if_time_in_day,
            if_day_in_week=args.if_day_in_week,
            device=device
        ).to(device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch):
        input_data = batch['X']
        time_series = input_data[..., :1]

        if self.if_time_in_day:
            tid_data = input_data[..., 1:2]
            time_in_day_emb = self.time_in_day_emb[(tid_data[:, -1, :] * self.time_of_day_size).squeeze(-1).type(torch.LongTensor)]
        else:
            time_in_day_emb = None

        if self.if_day_in_week:
            diw_data = input_data[..., 2:3]
            day_in_week_emb = self.day_in_week_emb[(diw_data[:, -1, :].squeeze(-1)).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        batch_size, _, num_nodes, _ = time_series.shape
        time_series = time_series.transpose(1, 2).contiguous()
        time_series = time_series.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(time_series)

        node_emb = []
        if self.if_spatial:
            node_emb.append(self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        hidden = self.encoder(hidden)
        prediction = self.regression_layer(hidden)

        return prediction

    def predict(self, batch):
        return self.forward(batch)
