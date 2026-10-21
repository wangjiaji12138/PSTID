"""PSTID prototype components."""

import torch
from torch import nn
from .spatial import SpatialCodebook
from .temporal import TemporalBasis


class ProtoModule(nn.Module):
    """原型模块：只负责编排和融合，不再计算时空码本的内部逻辑

    职责：
    - 接收主特征 (B,T,N,D) 和 adj_mx_emb（可学习节点嵌入）
    - 调用各码本获取 proto_enhanced
    - 融合输出
    """

    def __init__(self, num_nodes: int, proto_dim: int,
                 n_spatial: int = 4, n_temporal: int = 16,
                 use_spatio: bool = True,
                 use_temporal: bool = True,
                 proto_temp: float = 0.5,
                 adj_mx_emb_dim: int = None,
                 spatial_emb_dim: int = None,
                 time_series_dim: int = None,
                 time_of_day_size: int = None,
                 day_of_week_size: int = None,
                 temp_dim_tid: int = None,
                 temp_dim_diw: int = None,
                 proto_emb_dim: int = None,
                 hidden_dim: int = None,
                 use_adj: bool = True,
                 use_his: bool = True,
                 use_proto: bool = True):
        super().__init__()

        self.n_spatial, self.n_temporal = n_spatial, n_temporal
        self.use_spatio = use_spatio
        self.use_temporal = use_temporal
        self.proto_dim = proto_dim
        self.adj_mx_emb_dim = adj_mx_emb_dim
        self.spatial_emb_dim = spatial_emb_dim
        self.proto_emb_dim = proto_emb_dim
        self.use_adj = use_adj
        self.use_his = use_his
        self.use_proto = use_proto

        self.spatial_codebook = None
        self.temporal_basis = None

        if use_spatio:
            self.spatial_codebook = SpatialCodebook(
                n_spatial, proto_emb_dim=proto_emb_dim,
                adj_mx_emb_dim=adj_mx_emb_dim if use_adj else None,
                spatial_emb_dim=spatial_emb_dim,
                time_series_dim=time_series_dim,
                temperature=proto_temp,
                use_his=use_his,
                use_proto=use_proto,
            )

        if use_temporal:
            self.temporal_basis = TemporalBasis(
                n_temporal, proto_emb_dim=proto_emb_dim, temp=proto_temp,
                time_series_dim=time_series_dim,
                time_of_day_size=time_of_day_size,
                day_of_week_size=day_of_week_size,
                temp_dim_tid=temp_dim_tid,
                temp_dim_diw=temp_dim_diw,
                use_his=use_his,
                use_proto=use_proto,
            )

        # ---- 输出投影层：proto_enhanced (proto_dim) -> hidden_dim ----
        # hidden_dim 由 PSTID 传入，用于确保 proto_enhanced 与嵌入维度匹配
        self.proto_out_proj = nn.Linear(proto_dim, hidden_dim)

    def forward(self, time_series_emb: torch.Tensor,
                adj_mx_emb: torch.Tensor = None,
                node_emb: torch.Tensor = None,
                tid_emb: torch.Tensor = None,
                diw_emb: torch.Tensor = None) -> torch.Tensor | None:
        """Fuse branch features; add query residuals only with prototypes enabled."""
        spatial_out = None
        temporal_enhanced = temporal_proj = None
        if self.use_spatio:
            spatial_out = self.spatial_codebook(time_series_emb, adj_mx_emb, node_emb)
        if self.use_temporal:
            temporal_enhanced, temporal_proj = self.temporal_basis(
                time_series_emb, tid_emb, diw_emb
            )

        if self.use_spatio and self.use_temporal:
            enhanced = spatial_out['proto_enhanced'] + temporal_enhanced
            projected = spatial_out['proj_feat'] + temporal_proj
        elif self.use_spatio:
            enhanced = spatial_out['proto_enhanced']
            projected = spatial_out['proj_feat']
        elif self.use_temporal:
            enhanced = temporal_enhanced
            projected = temporal_proj
        else:
            return None

        if self.use_proto:
            enhanced = enhanced + projected
        return self.proto_out_proj(enhanced)
