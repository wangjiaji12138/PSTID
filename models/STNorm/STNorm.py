import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel


class SNorm(nn.Module):
    def __init__(self, channels):
        super(SNorm, self).__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        x_norm = (x - x.mean(2, keepdims=True)) / (x.var(2, keepdims=True, unbiased=True) + 0.00001) ** 0.5
        out = x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return out


class TNorm(nn.Module):
    def __init__(self, num_nodes, channels, track_running_stats=True, momentum=0.1):
        super(TNorm, self).__init__()
        self.track_running_stats = track_running_stats
        self.beta = nn.Parameter(torch.zeros(1, channels, num_nodes, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, num_nodes, 1))
        self.register_buffer('running_mean', torch.zeros(1, channels, num_nodes, 1))
        self.register_buffer('running_var', torch.ones(1, channels, num_nodes, 1))
        self.momentum = momentum

    def forward(self, x):
        if self.track_running_stats:
            mean = x.mean((0, 3), keepdims=True)
            var = x.var((0, 3), keepdims=True, unbiased=False)
            if self.training:
                n = x.shape[3] * x.shape[0]
                with torch.no_grad():
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                    self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = x.mean((3), keepdims=True)
            var = x.var((3), keepdims=True, unbiased=True)
        x_norm = (x - mean) / (var + 0.00001) ** 0.5
        out = x_norm * self.gamma + self.beta
        return out


class STNorm(BaseModel):
    """Spatio-Temporal Normalization Network."""

    model_name = "stnorm"

    def __init__(self, num_nodes: int, feature_dim: int = 3,
                 output_dim: int = 1, in_window: int = 12, out_window: int = 24,
                 blocks: int = 1, layers: int = 4, kernel_size: int = 2,
                 channels: int = 16, snorm_bool: bool = True, tnorm_bool: bool = True,
                 dropout: float = 0.2,
                 device: torch.device = None):
        super().__init__()

        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.in_window = in_window
        self.out_window = out_window

        self.blocks = blocks
        self.layers = layers
        self.kernel_size = kernel_size
        self.channels = channels
        self.snorm_bool = snorm_bool
        self.tnorm_bool = tnorm_bool
        self.dropout_rate = dropout

        self.device = device or torch.device('cpu')

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        if self.snorm_bool:
            self.sn = nn.ModuleList()
        if self.tnorm_bool:
            self.tn = nn.ModuleList()

        num = int(self.tnorm_bool) + int(self.snorm_bool) + 1

        self.start_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.channels,
            kernel_size=(1, 1)
        )

        receptive_field = 1
        self.dropout = nn.Dropout(dropout)

        self.dilation = []

        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                self.dilation.append(new_dilation)
                if self.tnorm_bool:
                    self.tn.append(TNorm(self.num_nodes, self.channels))
                if self.snorm_bool:
                    self.sn.append(SNorm(self.channels))
                self.filter_convs.append(nn.Conv2d(
                    in_channels=num * self.channels,
                    out_channels=self.channels,
                    kernel_size=(1, self.kernel_size), dilation=new_dilation
                ))
                self.gate_convs.append(nn.Conv2d(
                    in_channels=num * self.channels,
                    out_channels=self.channels,
                    kernel_size=(1, self.kernel_size), dilation=new_dilation
                ))
                self.residual_convs.append(nn.Conv2d(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=(1, 1)
                ))
                self.skip_convs.append(nn.Conv2d(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=(1, 1)
                ))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=(1, 1),
            bias=True
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.out_window,
            kernel_size=(1, 1),
            bias=True
        )

        self.receptive_field = receptive_field
        self.apply(self._init_weights)

    @staticmethod
    def from_args(args, num_nodes, adj_mx, device):
        """从命令行参数创建STNorm模型实例"""
        return STNorm(
            num_nodes=num_nodes,
            feature_dim=3,
            output_dim=1,
            in_window=args.input_window,
            out_window=args.output_window,
            blocks=args.blocks,
            layers=args.num_layers,
            kernel_size=args.kernel_size,
            channels=args.channels,
            snorm_bool=args.snorm_bool,
            tnorm_bool=args.tnorm_bool,
            dropout=args.dropout,
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

    def forward(self, batch):
        input = batch['X'][..., :1]  # 只使用第一个特征维度（值数据）
        input = input.permute(0, 3, 2, 1)
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        for i in range(self.blocks * self.layers):
            residual = x
            x_list = [x]
            if self.tnorm_bool:
                x_tnorm = self.tn[i](x)
                x_list.append(x_tnorm)
            if self.snorm_bool:
                x_snorm = self.sn[i](x)
                x_list.append(x_snorm)

            x = torch.cat(x_list, dim=1)
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]

        x = F.relu(skip)
        rep = F.relu(self.end_conv_1(x))
        out = self.end_conv_2(rep)
        out = out[:, :, :, -1]
        return out.unsqueeze(-1)

    def predict(self, batch):
        return self.forward(batch)
