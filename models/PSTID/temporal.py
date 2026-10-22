"""PSTID temporal components."""

import torch
from torch import nn
import torch.nn.functional as F


class TemporalBasis(nn.Module):
    """时间基：编码动态时序模式

    核心思想：让时间序列特征 time_series_emb 通过主模型的时间身份（tid_emb 和 diw_emb）
    来引导的注意力来生成每个时刻不同的身份。

    注意力计算（使用 MLP 直接预测分配权重）：
        1. 输入 = 每个位置自身的 tid_emb、diw_emb（历史末步与全部未来步沿时间轴拼接），use_his=True 时额外拼接 time_series_emb
        2. MLP 预测基础分配，未来步叠加时间对关系与历史原型转移偏好
    """

    def __init__(self, num_protos: int, proto_emb_dim: int = None, temp: float = 0.1,
                 time_series_dim: int = None, time_of_day_size: int = None, day_of_week_size: int = None,
                 temp_dim_tid: int = None, temp_dim_diw: int = None,
                 use_his: bool = True, use_proto: bool = True,
                 use_time_interaction: bool = True):
        super().__init__()
        self.num_protos = num_protos
        # 如果没有提供 proto_emb_dim，则使用 time_series_dim
        self.proto_emb_dim = proto_emb_dim if proto_emb_dim is not None else time_series_dim
        self.time_series_dim = time_series_dim
        self.use_his = use_his
        self.use_proto = use_proto
        self.temp_dim_tid = temp_dim_tid or 0
        self.temp_dim_diw = temp_dim_diw or 0
        if use_proto:
            self.temp = nn.Parameter(torch.tensor(temp))
            self.prototypes = nn.Parameter(torch.randn(num_protos, self.proto_emb_dim))
            nn.init.xavier_normal_(self.prototypes)

        # ========== MLP 分配器（替代点积注意力）==========
        # use_his=False 时，分配器和 Query 投影都只使用时间身份嵌入
        in_dim = (time_series_dim if use_his else 0) + (temp_dim_tid or 0) + (temp_dim_diw or 0)

        if use_proto:
            self.allocator = nn.Sequential(
                nn.Linear(in_dim, in_dim * 2),
                nn.GELU(),
                nn.Linear(in_dim * 2, num_protos),
            )
        else:
            # 原分配器输入先映射为分配权重，再映射回增强特征。
            self.linear = nn.Sequential(
                nn.Linear(in_dim, num_protos),
                nn.Softmax(dim=-1),
                nn.Linear(num_protos, self.proto_emb_dim),
            )

        # ========== 投影网络（用于生成 proto_enhanced）==========
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.GELU(),
            nn.Linear(in_dim * 2, self.proto_emb_dim),
        )

        # 只作用于未来分配：时间对关系 + 当前原型到目标原型的转移偏好。
        # 零初始化使加载上一版权重后的初始函数保持不变。
        self.use_time_interaction = use_time_interaction and use_proto
        if self.use_time_interaction:
            calendar_dim = self.temp_dim_tid + self.temp_dim_diw
            # 不消耗主模型初始化的随机流，开关对照可使用相同种子。
            with torch.random.fork_rng(devices=[]):
                self.calendar_interaction = nn.Linear(3 * calendar_dim, num_protos, bias=False)
            nn.init.zeros_(self.calendar_interaction.weight)
            self.prototype_transition = nn.Parameter(torch.zeros(num_protos, num_protos))

    def _interaction_logits(self, history_calendar, future_calendar, history_weights):
        """返回 (B, H, N, K)，不构造历史×未来全连接注意力。"""
        current = history_calendar.expand_as(future_calendar)
        relation = torch.cat([current, future_calendar - current,
                              current * future_calendar], dim=-1)
        return (self.calendar_interaction(relation)
                + history_weights @ self.prototype_transition)

    def forward(self, time_series_emb: torch.Tensor, tid_emb: torch.Tensor = None, diw_emb: torch.Tensor = None,
                future_tid_emb: torch.Tensor = None, future_diw_emb: torch.Tensor = None) -> tuple:
        """对历史末步及全部未来步独立分配原型，返回 (B, 1+H, N, D)。

        time_series_emb 将整个历史窗口特征扩展到 1+H；历史时间身份
        仅保留末步，未来时间身份逐步保留，共享分配器与原型表。
        """
        if any(emb is None for emb in (tid_emb, diw_emb, future_tid_emb, future_diw_emb)):
            raise ValueError('TemporalBasis requires historical and future tid/diw embeddings')
        tid = torch.cat([tid_emb, future_tid_emb], dim=1)
        diw = torch.cat([diw_emb, future_diw_emb], dim=1)
        if tid.shape[:3] != time_series_emb.shape[:3] or diw.shape[:3] != tid.shape[:3]:
            raise ValueError('Time-series and calendar embeddings must align on B, 1+H, N')
        parts = [time_series_emb] if self.use_his else []
        x_alloc = torch.cat(parts + [tid, diw], dim=-1)
        proj_feat = self.proj(x_alloc)
        if self.use_proto:
            logits = self.allocator(x_alloc)
            if self.use_time_interaction:
                current_calendar = torch.cat([tid_emb, diw_emb], dim=-1)
                target_calendar = torch.cat([future_tid_emb, future_diw_emb], dim=-1)
                current_weights = F.softmax(logits[:, :1] / self.temp, dim=-1)
                adjustment = self._interaction_logits(current_calendar, target_calendar, current_weights)
                logits = torch.cat([logits[:, :1], logits[:, 1:] + adjustment], dim=1)
            weights = F.softmax(logits / self.temp, dim=-1)
            proto_enhanced = weights @ self.prototypes
        else:
            proto_enhanced = self.linear(x_alloc)
        return proto_enhanced, proj_feat
