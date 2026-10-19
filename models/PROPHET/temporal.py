"""PROPHET temporal components."""

import torch
from torch import nn
import torch.nn.functional as F


class TemporalBasis(nn.Module):
    """时间基：编码动态时序模式

    核心思想：让时间序列特征 time_series_emb 通过主模型的时间身份（tid_emb 和 diw_emb）
    来引导的注意力来生成每个时刻不同的身份。

    注意力计算（使用 MLP 直接预测分配权重）：
        1. 输入 = tid_emb + diw_emb，use_his=True 时额外拼接 time_series_emb
        2. MLP 直接预测每个原型的分配权重
    """

    def __init__(self, num_protos: int, proto_emb_dim: int = None, temp: float = 0.1,
                 time_series_dim: int = None, time_of_day_size: int = None, day_of_week_size: int = None,
                 temp_dim_tid: int = None, temp_dim_diw: int = None,
                 use_his: bool = True):
        super().__init__()
        self.num_protos = num_protos
        # 如果没有提供 proto_emb_dim，则使用 time_series_dim
        self.proto_emb_dim = proto_emb_dim if proto_emb_dim is not None else time_series_dim
        self.time_series_dim = time_series_dim
        self.use_his = use_his
        self.temp_dim_tid = temp_dim_tid or 0
        self.temp_dim_diw = temp_dim_diw or 0
        self.temp = nn.Parameter(torch.tensor(temp))

        # 原型参数（使用可配置的 proto_emb_dim）
        self.prototypes = nn.Parameter(torch.randn(num_protos, self.proto_emb_dim))
        nn.init.xavier_normal_(self.prototypes)

        # ========== MLP 分配器（替代点积注意力）==========
        # use_his=False 时，分配器和 Query 投影都只使用时间身份嵌入
        in_dim = (time_series_dim if use_his else 0) + (temp_dim_tid or 0) + (temp_dim_diw or 0)

        self.allocator = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.GELU(),
            nn.Linear(in_dim * 2, num_protos),
        )

        # ========== 投影网络（用于生成 proto_enhanced）==========
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.GELU(),
            nn.Linear(in_dim * 2, self.proto_emb_dim),
        )

    def forward(self, time_series_emb: torch.Tensor, tid_emb: torch.Tensor = None, diw_emb: torch.Tensor = None) -> tuple:
        """时间基前向传播：使用 MLP 直接预测原型分配权重

        Args:
            time_series_emb: 时间序列嵌入 (B, T, N, D_time_series)，use_his=False 时仅取形状
            tid_emb: 主模型的时间嵌入 (B, T, N, D_tid) - 代表最后一个历史时间的嵌入
            diw_emb: 主模型的星期嵌入 (B, T, N, D_diw) - 代表最后一个历史时间的嵌入

        Returns:
            proto_enhanced_feat: (B, T, N, proto_emb_dim) 原型增强特征
            proj_feat: (B, T, N, proto_emb_dim) 投影特征
        """
        B, T, N, _ = time_series_emb.shape

        # ========== 2. 提取 tid_emb 和 diw_emb 的最后一个时间步 ==========
        tid_last = tid_emb[:, -1, :, :] if tid_emb is not None else None
        diw_last = diw_emb[:, -1, :, :] if diw_emb is not None else None

        # ========== 3. MLP 直接预测分配权重 ==========
        x_list = [time_series_emb[:, -1, :, :]] if self.use_his else []
        if tid_last is not None:
            x_list.append(tid_last)
        if diw_last is not None:
            x_list.append(diw_last)
        x_alloc = torch.cat(x_list, dim=-1)  # (B, N, in_dim)
        x_alloc_flat = x_alloc.reshape(B * N, -1)  # (B*N, in_dim)
        base_attn = self.allocator(x_alloc_flat)  # (B*N, num_protos)
        base_attn = F.softmax(base_attn / self.temp, dim=-1)  # 用温度控制锐度

        # ========== 4. 投影到 proto_emb_dim ==========
        proj_feat = self.proj(x_alloc_flat)  # (B*N, proto_emb_dim)

        # ========== 5. 加权求和得到原型增强特征 ==========
        proto_enhanced_flat = base_attn @ self.prototypes  # (B*N, proto_emb_dim)

        # ========== 6. reshape 回 (B, T, N, dim) ==========
        proto_enhanced = proto_enhanced_flat.reshape(B, 1, N, -1).expand(B, T, N, -1)
        proj_feat = proj_feat.reshape(B, 1, N, -1).expand(B, T, N, -1)

        return proto_enhanced, proj_feat
