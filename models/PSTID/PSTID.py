"""
PSTID: 原型时空身份网络

继承 STID 的基础架构，在输入嵌入和 MLP 之间加入原型时空身份模块。
通过原型模块将节点嵌入归类成空间码本信息，将时间嵌入归类成时间码本信息，
减少噪声并提升模型的可解释性。

核心思想：
- STID 的 TID (Time-In-Day) 和 DIW (Day-In-Week) 嵌入以及空间节点嵌入可能产生噪声
- 通过码本模块，我们可以将节点嵌入归类成几个空间码本信息
- 将时间嵌入归类成几个时间码本信息，从而极大减少噪声并提升可解释性

架构流程：
Input → STID Embedding → ProtoModule → MLP Layers → Regression Layer → Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from models.base import BaseModel


def _compute_adj_mx_emb_init(adj_mx: torch.Tensor, emb_dim: int) -> torch.Tensor:
    """用 adj_mx 的谱嵌入（SVD）初始化节点嵌入

    Args:
        adj_mx: 邻接矩阵 (N, N)
        emb_dim: 目标嵌入维度

    Returns:
        初始化后的嵌入 (N, emb_dim)
    """
    # 确保 adj_mx 是 numpy 数组
    if torch.is_tensor(adj_mx):
        adj_np = adj_mx.cpu().numpy()
    else:
        adj_np = np.array(adj_mx)

    # SVD 分解
    # 使用淡淡的归一化拉普拉斯谱嵌入
    d = np.sum(adj_np, axis=1)
    d_inv_sqrt = np.power(d, -0.5, where=d > 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(d.shape[0]) - D_inv_sqrt @ adj_np @ D_inv_sqrt  # 对称归一化拉普拉斯

    try:
        # 取最小的 emb_dim 个特征向量（跳过第一个全1的特征向量）
        eigvals, eigvecs = np.linalg.eigh(L)
        # 按特征值从小到大排序，取第 1 到 emb_dim 个（跳过第 0 个全 1 向量）
        indices = np.argsort(eigvals)
        idx = indices[1:emb_dim + 1]  # 取 emb_dim 个非平凡特征向量
        init_emb = eigvecs[:, idx]  # (N, emb_dim)
        # 缩放到 [-1, 1] 范围
        init_emb = init_emb / (np.abs(init_emb).max(axis=0, keepdims=True) + 1e-8)
        return torch.from_numpy(init_emb).float()
    except Exception:
        # SVD 失败时降维到 emb_dim
        U, S, Vt = np.linalg.svd(adj_np, full_matrices=False)
        init_emb = U[:, :emb_dim] * np.sqrt(S[:emb_dim])
        return torch.from_numpy(init_emb).float()


class ProtoVisData:
    """可视化数据结构（从 models.PSTID.PSTID 复制，避免依赖不存在的模块）"""

    def __init__(self):
        self.use_spatio = False
        self.use_temporal = False
        self.proto_dim = 0
        self.num_layers = 1
        self.layer_idx = 0
        self.spatial_prototypes = None
        self.n_spatial = 0
        self.temporal_prototypes = None
        self.n_temporal = 0
        self.time_of_day_size = 0
        self.day_of_week_size = 0
        self.spatial_query_proj = None
        self.spatial_query_shape = None
        self.temporal_query_proj = None
        self.temporal_query_shape = None
        self.spatial_usage = None  # list of float, 每个原型的平均使用率 (对 B,T,N 求均值)
        self.temporal_usage = None  # list of float, 每个原型的平均使用率 (对 B,T,N 求均值)
        self.temporal_per_timestep = None  # (T, n_temporal) 每个时间步的使用率
        self.identity_first_batch = None
        # 每个节点的注意力权重: (N, n_spatial)，对 B,T 求均值
        self.spatial_attention_per_node = None

    def get_identity_for_layer(self, layer_idx):
        """返回指定层的节点 identity 数据"""
        return self.identity_first_batch


class MultiLayerPerceptron(nn.Module):
    """ MLP层 """
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.1)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        hidden = hidden + input_data
        return hidden


class SpatialCodebook(nn.Module):
    """空间码本：编码静态/半静态节点身份

    职责：
    - 接收主特征 (B,T,N,D) 和 adj_mx_emb（可学习节点嵌入）
    - 自行完成节点均值计算、维度对齐、投影、码本注意力融合
    - 返回 (attention, proto_enhanced, proj_feat)
    
    分配策略：
    - 使用 softmax 软分配，根据相似度加权所有原型
    """

    def __init__(self, num_protos: int, proto_emb_dim: int = None,
                 adj_mx_emb_dim: int = None, time_series_dim: int = None):
        super().__init__()
        self.num_protos = num_protos
        self.proto_emb_dim = proto_emb_dim

        # 原型参数
        self.prototypes = nn.Parameter(torch.randn(num_protos, self.proto_emb_dim))
        nn.init.orthogonal_(self.prototypes)

        # 统一投影层：拼接 (adj_mx_emb + time_series_mean) → proto_emb_dim
        combined_dim = adj_mx_emb_dim + time_series_dim
        self.proj = nn.Linear(combined_dim, proto_emb_dim)
    
    def _compute_attention(self, proj_feat: torch.Tensor) -> tuple:
        """计算原型注意力（proj_feat 形状: (N, D)）

        Args:
            proj_feat: 投影特征 (N, proto_emb_dim)

        Returns:
            attention: (N, num_protos) 软分配权重
            proto_enhanced: (N, proto_emb_dim) 原型增强特征
        """
        # 点积相似度（无归一化，无温度）
        sim = proj_feat @ self.prototypes.t()  # (N, num_protos)

        # 软分配
        soft_weights = F.softmax(sim, dim=-1)  # (N, num_protos)

        # 软加权求和
        proto_enhanced = soft_weights @ self.prototypes  # (N, D)

        return soft_weights, proto_enhanced

    def forward(self, time_series_emb: torch.Tensor, adj_mx_emb: torch.Tensor = None) -> dict:
        """完整前向传播：投影 + 码本注意力

        Args:
            time_series_emb: 时间序列嵌入 (B, T, N, D_time_series)
            adj_mx_emb: adj_mx 衍生的可学习节点嵌入 (N, D_adj_mx)

        Returns:
            dict with keys:
                attention: (B, T, N, num_protos) 原型注意力权重
                proto_enhanced: (B, T, N, proto_emb_dim) 原型增强特征
                proj_feat: (B, T, N, proto_emb_dim) 投影特征
        """
        B, T, N, _ = time_series_emb.shape

        # ===== 静态 Query：所有样本共享 =====
        # 使用时间序列均值 + adj_mx_emb 拼接后统一投影
        x_node_mean = time_series_emb.mean(dim=1).mean(dim=0)  # (N, D_time_series)
        x_combined = torch.cat([adj_mx_emb, x_node_mean], dim=-1)  # (N, D_adj_mx + D_time_series)
        proj_feat = self.proj(x_combined)  # (N, proto_emb_dim)

        attention_weights, proto_enhanced = self._compute_attention(proj_feat)
        # attention_weights: (N, num_protos), proto_enhanced: (N, proto_emb_dim)

        # reshape 回 (B, T, N, dim)
        attention = attention_weights.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)
        proto_enhanced = proto_enhanced.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)
        proj_feat = proj_feat.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)

        return {
            'attention': attention,
            'proto_enhanced': proto_enhanced,
            'proj_feat': proj_feat,
        }


class TemporalBasis(nn.Module):
    """时间基：编码动态时序模式

    核心思想：让时间序列特征 time_series_emb 通过主模型的时间身份（tid_emb 和 diw_emb）
    来引导的注意力来生成每个时刻不同的身份。

    注意力计算（使用 MLP 直接预测分配权重）：
        1. 输入 = time_series_emb 的统计特征 + tid_emb + diw_emb
        2. MLP 直接预测每个原型的分配权重
    """

    def __init__(self, num_protos: int, proto_emb_dim: int = None, temp: float = 0.1,
                 time_series_dim: int = None, time_of_day_size: int = None, day_of_week_size: int = None,
                 temp_dim_tid: int = None, temp_dim_diw: int = None):
        super().__init__()
        self.num_protos = num_protos
        # 如果没有提供 proto_emb_dim，则使用 time_series_dim
        self.proto_emb_dim = proto_emb_dim if proto_emb_dim is not None else time_series_dim
        self.time_series_dim = time_series_dim
        self.temp_dim_tid = temp_dim_tid or 0
        self.temp_dim_diw = temp_dim_diw or 0
        self.temp = nn.Parameter(torch.tensor(temp))

        # 原型参数（使用可配置的 proto_emb_dim）
        self.prototypes = nn.Parameter(torch.randn(num_protos, self.proto_emb_dim))
        nn.init.xavier_normal_(self.prototypes)
        
        # ========== MLP 分配器（替代点积注意力）==========
        # 输入特征：ts_last + tid_last + diw_last
        in_dim = time_series_dim + (temp_dim_tid or 0) + (temp_dim_diw or 0)
        
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
            time_series_emb: 时间序列嵌入 (B, T, N, D_time_series)
            tid_emb: 主模型的时间嵌入 (B, T, N, D_tid) - 代表最后一个历史时间的嵌入
            diw_emb: 主模型的星期嵌入 (B, T, N, D_diw) - 代表最后一个历史时间的嵌入

        Returns:
            base_attn: (B, T, N, num_protos) 原型分配权重
            proto_enhanced_feat: (B, T, N, proto_emb_dim) 原型增强特征
            proj_feat: (B, T, N, proto_emb_dim) 投影特征
        """
        B, T, N, D_ts = time_series_emb.shape

        # ========== 1. 提取 time_series_emb 的最后一个时间步 ==========
        ts_last = time_series_emb[:, -1, :, :]  # (B, N, D_ts)

        # ========== 2. 提取 tid_emb 和 diw_emb 的最后一个时间步 ==========
        tid_last = tid_emb[:, -1, :, :]
        diw_last = diw_emb[:, -1, :, :]

        # ========== 3. MLP 直接预测分配权重 ==========
        x_alloc = torch.cat([ts_last, tid_last, diw_last], dim=-1)  # (B, N, in_dim)
        x_alloc_flat = x_alloc.reshape(B * N, -1)  # (B*N, in_dim)
        base_attn = self.allocator(x_alloc_flat)  # (B*N, num_protos)
        base_attn = F.softmax(base_attn / self.temp, dim=-1)  # 用温度控制锐度

        # ========== 4. 投影到 proto_emb_dim ==========
        proj_feat = self.proj(x_alloc_flat)  # (B*N, proto_emb_dim)

        # ========== 5. 加权求和得到原型增强特征 ==========
        proto_enhanced_flat = base_attn @ self.prototypes  # (B*N, proto_emb_dim)

        # ========== 6. reshape 回 (B, T, N, dim) ==========
        base_attn = base_attn.reshape(B, 1, N, -1).expand(B, T, N, -1)
        proto_enhanced = proto_enhanced_flat.reshape(B, 1, N, -1).expand(B, T, N, -1)
        proj_feat = proj_feat.reshape(B, 1, N, -1).expand(B, T, N, -1)

        return base_attn, proto_enhanced, proj_feat


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
                 time_series_dim: int = None,
                 time_of_day_size: int = None,
                 day_of_week_size: int = None,
                 temp_dim_tid: int = None,
                 temp_dim_diw: int = None,
                 proto_emb_dim: int = None,
                 hidden_dim: int = None):
        super().__init__()

        self.n_spatial, self.n_temporal = n_spatial, n_temporal
        self.use_spatio = use_spatio
        self.use_temporal = use_temporal
        self.proto_dim = proto_dim
        self.adj_mx_emb_dim = adj_mx_emb_dim
        self.proto_emb_dim = proto_emb_dim

        self.spatial_codebook = None
        self.temporal_basis = None

        if use_spatio:
            self.spatial_codebook = SpatialCodebook(
                n_spatial, proto_emb_dim=proto_emb_dim,
                adj_mx_emb_dim=adj_mx_emb_dim,
                time_series_dim=time_series_dim,
            )

        if use_temporal:
            self.temporal_basis = TemporalBasis(
                n_temporal, proto_emb_dim=proto_emb_dim, temp=proto_temp,
                time_series_dim=time_series_dim,
                time_of_day_size=time_of_day_size,
                day_of_week_size=day_of_week_size,
                temp_dim_tid=temp_dim_tid,
                temp_dim_diw=temp_dim_diw,
            )

        # ---- 输出投影层：proto_enhanced (proto_dim) -> hidden_dim ----
        # hidden_dim 由 PSTID 传入，用于确保 proto_enhanced 与嵌入维度匹配
        self.proto_out_proj = nn.Linear(proto_dim, hidden_dim)

    def forward(self, time_series_emb: torch.Tensor,
                adj_mx_emb: torch.Tensor = None,
                tid_emb: torch.Tensor = None, diw_emb: torch.Tensor = None) -> tuple:
        """计算原型注意力融合（纯编排：调用码本 → 融合输出）

        Args:
            time_series_emb: 时间序列嵌入 (B, T, N, D_time_series)
            adj_mx_emb: adj_mx 衍生的可学习节点嵌入 (N, D_adj_mx)
            tid_emb: 主模型的时间嵌入 (B, T, N, D_tid)
            diw_emb: 主模型的星期嵌入 (B, T, N, D_diw)
        """
        # ---- 空间码本：使用 time_series_emb 和 adj_mx_emb ----
        if self.use_spatio:
            spatial_out = self.spatial_codebook(time_series_emb, adj_mx_emb)
        else:
            spatial_out = None

        # ---- 时间基：使用主模型的时间嵌入 ----
        if self.use_temporal:
            attn, proto_enhanced, proj_feat = self.temporal_basis(
                time_series_emb, tid_emb, diw_emb
            )
            temporal_out = {'attention': attn, 'proto_enhanced': proto_enhanced,
                           'proj_feat': proj_feat}
        else:
            temporal_out = None

        # ---- 收集各码本的输出 ----
        sp_e = spatial_out['proto_enhanced'] if spatial_out else None
        tp_e = temporal_out['proto_enhanced'] if temporal_out else None
        sp_proj = spatial_out['proj_feat'] if spatial_out else None
        tp_proj = temporal_out['proj_feat'] if temporal_out else None

        # ---- 融合 proto_enhanced + 统一残差 ----
        # 融合原型增强
        proto_enhanced = None
        if self.use_spatio and self.use_temporal:
            proto_enhanced = sp_e + tp_e
        elif self.use_spatio:
            proto_enhanced = sp_e
        elif self.use_temporal:
            proto_enhanced = tp_e

        # 收集投影特征并做残差
        total_proj = None
        if self.use_spatio:
            total_proj = sp_proj
        if self.use_temporal:
            total_proj = tp_proj if total_proj is None else total_proj + tp_proj

        # 统一残差连接
        if proto_enhanced is not None and total_proj is not None:
            proto_enhanced = proto_enhanced + total_proj

        # ---- 输出投影：proto_enhanced -> hidden_dim ----
        if proto_enhanced is not None:
            proto_enhanced = self.proto_out_proj(proto_enhanced)

        # ---- codebook_proj_feats（供可视化用）----
        codebook_proj_feats = {}
        if spatial_out:
            codebook_proj_feats['spatial'] = spatial_out['proj_feat']
        if temporal_out:
            codebook_proj_feats['temporal'] = temporal_out['proj_feat']

        spatial_proto = self.spatial_codebook.prototypes if self.spatial_codebook else None
        temporal_proto = self.temporal_basis.prototypes if self.temporal_basis else None

        return (spatial_proto,
                temporal_proto,
                {
                    'attention': {
                        'spatial': spatial_out['attention'] if spatial_out else None,
                        'temporal': temporal_out['attention'] if temporal_out else None,
                    },
                    'proto_enhanced': proto_enhanced,
                    'proto_enhanced_spatial': sp_e,
                    'proto_enhanced_temporal': tp_e,
                },
                None,
                codebook_proj_feats)

    def apply(self, time_series_emb: torch.Tensor,
              adj_mx_emb: torch.Tensor = None,
              tid_emb: torch.Tensor = None, diw_emb: torch.Tensor = None,
              collect_cache: bool = False) -> tuple:
        """应用原型模块增强嵌入

        Args:
            time_series_emb: 时间序列嵌入 (B, T, N, D_time_series)
            adj_mx_emb: adj_mx 衍生的可学习节点嵌入 (N, D_adj_mx)
            tid_emb: 主模型的时间嵌入 (B, T, N, D_tid)
            diw_emb: 主模型的星期嵌入 (B, T, N, D_diw)
            collect_cache: 是否收集可视化缓存数据

        Returns:
            proto_enhanced: (B, T, N, proto_dim) 原型增强后的嵌入
            proto_info: dict 包含注意力权重等信息
        """
        # 调用 proto_module
        spatial_proto, temporal_proto, pa, _, codebook_proj_feats = self.forward(
            time_series_emb, adj_mx_emb, tid_emb, diw_emb
        )

        # 提取 proto_enhanced
        proto_enhanced = pa.get('proto_enhanced', None)
        if proto_enhanced is None:
            return time_series_emb, None

        # 缓存原型使用信息（每次 forward 后都更新，供日志打印使用）
        self._last_proto_usage = [self._compute_proto_usage(pa)]

        # 收集 visualization cache（仅当 collect_cache=True）
        if collect_cache:
            self._cache_proto_vis_data(pa, codebook_proj_feats, proto_enhanced)

        # proto_info
        proto_info = {
            'spatial_proto': spatial_proto,
            'temporal_proto': temporal_proto,
            'attention': pa.get('attention', {}),
        }

        return proto_enhanced, proto_info

    def _cache_proto_vis_data(self, pa: dict, codebook_proj_feats: dict, proto_enhanced: torch.Tensor):
        """统一缓存原型可视化数据"""
        # temporal query projection
        temporal_q = codebook_proj_feats.get('temporal', None)
        if temporal_q is not None:
            Bq, Tq, Nq, Dq = temporal_q.shape
            pa['temporal_query_proj'] = temporal_q.reshape(Bq * Tq * Nq, Dq).detach().cpu()
            pa['temporal_shape'] = (Bq, Tq, Nq)

        # spatial query projection
        sp_q = codebook_proj_feats.get('spatial', None)
        if sp_q is not None:
            Bs, Ts, Ns, Ds = sp_q.shape
            pa['spatial_query_proj'] = sp_q.reshape(Bs * Ts * Ns, Ds).detach().cpu()
            pa['spatial_shape'] = (Bs, Ts, Ns)

        self._last_pa = pa
        self._last_proto_enhanced_first_batch = [proto_enhanced[:, 0, :, :].detach().cpu()]
        self._last_proto_assigns = [pa]
        self._last_proto_usage = [self._compute_proto_usage(pa)]

    @staticmethod
    def _compute_proto_usage(pa: dict) -> dict:
        """统计原型注意力权重分布

        Returns:
            dict with keys:
                'spatial': list of float, 平均使用率 (对 B,T,N 求均值)
                'temporal': list of float, 平均使用率 (对 B,T,N 求均值)
                'spatial_node': list of int, 主导原型索引 (对 B,T 求均值后取 argmax)
                'temporal_node': list of int, 主导原型索引 (对 B,T 求均值后取 argmax)
                'spatial_per_node': (N, n_spatial) 每个节点的注意力权重（对 B,T 求均值）
                'temporal_per_timestep': (T, n_temporal) 每个时间步的 temporal prototype 使用率
        """
        with torch.no_grad():
            result = {}
            att_dict = pa.get('attention', {})
            if 'spatial' in att_dict and att_dict['spatial'] is not None:
                # spatial_att: (B, T, N, n_spatial)，但所有样本共享同一个 query
                spatial_att = att_dict['spatial']
                spatial_usage = spatial_att.mean(dim=[0, 1, 2])  # 对 B,T,N 求均值
                result['spatial'] = spatial_usage.tolist()
                result['spatial_node'] = spatial_att.mean(dim=[0, 1]).argmax(dim=-1).tolist()
                spatial_per_node = spatial_att.mean(dim=[0, 1])  # (N, n_spatial)
                result['spatial_per_node'] = spatial_per_node.cpu().numpy()
            if 'temporal' in att_dict and att_dict['temporal'] is not None:
                temporal_att = att_dict['temporal']
                temporal_usage = temporal_att.mean(dim=[0, 1, 2])
                result['temporal'] = temporal_usage.tolist()
                result['temporal_node'] = temporal_att.mean(dim=[0, 1]).argmax(dim=-1).tolist()
                temporal_per_timestep = temporal_att.mean(dim=[0, 2])  # (T, n_temporal)
                result['temporal_per_timestep'] = temporal_per_timestep.cpu().numpy()

            return result

    def get_usage_uniformity_loss(self) -> torch.Tensor:
        """原型使用均匀性损失"""
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)

        all_attentions = getattr(self, '_all_layer_attentions', None)
        if all_attentions is None:
            all_attentions = [getattr(self, '_last_proto_attentions', None)] if getattr(self, '_last_proto_attentions', None) is not None else []

        for att in all_attentions:
            if att is None:
                continue
            spatial_att = att.get('spatial', None)
            if spatial_att is not None and self.use_spatio:
                usage = spatial_att.mean(dim=[0, 1, 2])
                expected = 1.0 / self.n_spatial
                loss = loss + ((usage - expected) ** 2).sum()

            temporal_att = att.get('temporal', None)
            if temporal_att is not None and self.use_temporal:
                usage = temporal_att.mean(dim=[0, 1, 2])
                expected = 1.0 / self.n_temporal
                loss = loss + ((usage - expected) ** 2).sum()

        return loss


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
                 proto_uniformity_weight: float = 0.0,
                 # 新增参数
                 proto_emb_dim: int = None,
                 # adj_mx 衍生的可学习节点嵌入维度
                 adj_mx_emb_dim: int = 64,
                 adj_mx: torch.Tensor = None):
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

        self.time_of_day_size = int((24 * 60 * 60) / time_intervals)
        self.day_of_week_size = 7

        # ==================== ProtoModule 参数 ====================
        # use_temporal 和 use_spatio 仅控制 ProtoModule 中码本的生成
        # 不影响时间/空间嵌入的使用（嵌入由 if_time_in_day, if_day_in_week, if_spatial 控制）
        self.use_temporal = use_temporal
        self.use_spatio = use_spatio
        
        self.use_proto = use_proto
        self.num_spatial_prototypes = num_spatial_prototypes
        self.num_temporal_prototypes = num_temporal_prototypes
        self.proto_temperature = proto_temperature
        self.proto_uniformity_weight = proto_uniformity_weight

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


        # ==================== 计算 hidden_dim ====================
        # STID 的 hidden_dim 是 time_series_emb + 所有启用的嵌入
        self.time_series_dim = self.time_series_emb_dim
        self.hidden_dim = self.time_series_emb_dim + self.spatial_emb_dim + self.temp_dim_tid + self.temp_dim_diw

        # ProtoModule 的 proto_dim：如果提供了 proto_emb_dim 则使用它，否则使用 hidden_dim
        self.proto_emb_dim = proto_emb_dim if proto_emb_dim is not None else self.hidden_dim
        self.proto_dim = self.proto_emb_dim
        
        # adj_mx 衍生的可学习节点嵌入
        self.adj_mx_emb_dim = adj_mx_emb_dim
        if adj_mx is not None:
            # 用 adj_mx 的谱嵌入（SVD）初始化
            init_emb = _compute_adj_mx_emb_init(adj_mx, adj_mx_emb_dim).to(device)
            # 如果 init_emb 的维度与目标不符，做线性插值
            if init_emb.shape[1] < adj_mx_emb_dim:
                # 填充随机噪声
                padding = torch.randn(init_emb.shape[0], adj_mx_emb_dim - init_emb.shape[1], device=init_emb.device)
                init_emb = torch.cat([init_emb, padding], dim=1)
            elif init_emb.shape[1] > adj_mx_emb_dim:
                init_emb = init_emb[:, :adj_mx_emb_dim]
            self.adj_mx_emb = nn.Parameter(init_emb)
        else:
            self.adj_mx_emb = nn.Parameter(torch.empty(self.num_nodes, self.adj_mx_emb_dim))
            nn.init.xavier_uniform_(self.adj_mx_emb)

        # ==================== ProtoModule ====================
        if self.use_proto:
            self.proto_module = ProtoModule(
                num_nodes=self.num_nodes,
                proto_dim=self.proto_dim,
                n_spatial=self.num_spatial_prototypes,
                n_temporal=self.num_temporal_prototypes,
                use_spatio=self.use_spatio,
                use_temporal=self.use_temporal,
                proto_temp=self.proto_temperature,
                adj_mx_emb_dim=self.adj_mx_emb_dim,
                time_series_dim=self.time_series_emb_dim,
                time_of_day_size=self.time_of_day_size,
                day_of_week_size=self.day_of_week_size,
                temp_dim_tid=self.temp_dim_tid,
                temp_dim_diw=self.temp_dim_diw,
                proto_emb_dim=self.proto_emb_dim,
                hidden_dim=self.hidden_dim,  # 传入 hidden_dim 用于 proto_out_proj
            )
        else:
            self.proto_module = None

        # ==================== MLP Layers ====================
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_block)]
        )

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

    def _get_embeddings(self, batch):
        """获取 STID 的嵌入表示 (B, T, N, hidden_dim)

        与 STID.forward() 中的嵌入逻辑一致，但扩展到所有 T 个时间步
        """
        input_data = batch['X']
        batch_size, T, num_nodes, _ = input_data.shape

        # ========== 时间序列嵌入 ==========
        # (B, T, N, 1) -> (B, T, N, 1) -> Conv2d -> (B, T, N, input_embedding_dim)
        time_series = input_data[..., :1]  # (B, T, N, 1)
        time_series = time_series.transpose(1, 2).contiguous()
        time_series = time_series.view(
        batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(time_series).expand(-1, -1, -1, T).permute(0, 3, 2, 1)

        # ========== 时间嵌入 ==========
        # 使用最后一个时间步的 tid/diw 作为代表，扩展到所有 T 个时间步
        tid_data = input_data[..., 1:2]  # (B, T, N, 1)
        tid_idx = (tid_data[:, -1, :, 0] * self.time_of_day_size).long().clamp(0, self.time_of_day_size - 1)  # (B, N) 只取最后一步
        time_in_day_emb = self.time_in_day_emb[tid_idx]  # (B, N, temp_dim_tid)
        time_in_day_emb = time_in_day_emb.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, N, temp_dim_tid)

        diw_data = input_data[..., 2:3]  # (B, T, N, 1)
        diw_idx = (diw_data[:, -1, :, 0] * self.day_of_week_size).long().clamp(0, self.day_of_week_size - 1)  # (B, N) 只取最后一步
        day_in_week_emb = self.day_in_week_emb[diw_idx]  # (B, N, temp_dim_diw)
        day_in_week_emb = day_in_week_emb.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, N, temp_dim_diw)

        # ========== 空间嵌入 ==========
        node_emb = self.node_emb  # (N, spatial_emb_dim)

        # ========== 拼接所有嵌入 ==========
        # 收集所有需要拼接的嵌入（4D）
        emb_list = [time_series_emb]
        emb_list.append(time_in_day_emb)
        emb_list.append(day_in_week_emb)
        
        node_emb_4d = node_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, T, -1, -1)  # (B, T, N, D)
        emb_list.append(node_emb_4d)

        # (B, 2D, N, 1)
        hidden = torch.cat(emb_list, dim=-1)

        # 返回拼接后的嵌入以及各分量（供原型模块使用）
        return hidden, time_series_emb, time_in_day_emb, day_in_week_emb, node_emb

    def collect_visualization_data(self, layer_idx: int = 0) -> ProtoVisData:
        """收集可视化所需的原型数据，生成结构化的 ProtoVisData 对象
        
        与 PSTID.collect_visualization_data 保持一致
        
        使用方式：
            vis_data = model.collect_visualization_data(layer_idx=0)
            if vis_data:
                plot_prototype_analysis(vis_data, save_path)
                plot_temporal_query_prototype(vis_data, save_path)
                plot_prototype_usage(vis_data, save_path)
        
        Args:
            layer_idx: 用于可视化的层索引，默认 0
        
        Returns:
            ProtoVisData 对象，包含所有可视化所需的数据；如无原型则返回 None
        """
        if not self.use_proto or self.proto_module is None:
            return None
        
        vis = ProtoVisData()
        vis.use_spatio = self.use_spatio
        vis.use_temporal = self.use_temporal
        vis.proto_dim = self.hidden_dim  # PSTID 使用 hidden_dim
        vis.num_layers = 1  # PSTID 只有一层 ProtoModule
        vis.layer_idx = layer_idx
        
        # ==== 1. 收集码本参数 ====
        if self.use_spatio and self.proto_module.spatial_codebook is not None:
            vis.spatial_prototypes = self.proto_module.spatial_codebook.prototypes.data.cpu().numpy()
            vis.n_spatial = vis.spatial_prototypes.shape[0]
        
        if self.use_temporal and self.proto_module.temporal_basis is not None:
            vis.temporal_prototypes = self.proto_module.temporal_basis.prototypes.data.cpu().numpy()
            vis.n_temporal = vis.temporal_prototypes.shape[0]
        
        # 保存时间维度信息（用于可视化）
        vis.time_of_day_size = self.time_of_day_size
        vis.day_of_week_size = self.day_of_week_size
        
        # ==== 2. 收集 Query 投影特征 ====
        pa = getattr(self.proto_module, '_last_pa', None)
        if pa is not None:
            # Spatial query
            sp_q = pa.get('spatial_query_proj', None)
            sp_shape = pa.get('spatial_shape', None)
            if sp_q is not None and isinstance(sp_q, (torch.Tensor, np.ndarray)):
                if isinstance(sp_q, torch.Tensor):
                    vis.spatial_query_proj = sp_q.detach().cpu().numpy()
                else:
                    vis.spatial_query_proj = sp_q
            if sp_shape is not None:
                vis.spatial_query_shape = sp_shape
            
            # Temporal query
            temp_q = pa.get('temporal_query_proj', None)
            temp_shape = pa.get('temporal_shape', None)
            if temp_q is not None and isinstance(temp_q, (torch.Tensor, np.ndarray)):
                if isinstance(temp_q, torch.Tensor):
                    vis.temporal_query_proj = temp_q.detach().cpu().numpy()
                else:
                    vis.temporal_query_proj = temp_q
            if temp_shape is not None:
                vis.temporal_query_shape = temp_shape
        
        # ==== 3. 收集原型使用率 ====
        usages = getattr(self.proto_module, '_last_proto_usage', None)
        if usages and len(usages) > 0:
            safe_idx = min(layer_idx, len(usages) - 1)
            u = usages[safe_idx]
            if 'spatial' in u:
                vis.spatial_usage = u['spatial']
            if 'temporal' in u:
                vis.temporal_usage = u['temporal']
            # 每个时间步的 temporal prototype 使用率: (T, n_temporal)
            if 'temporal_per_timestep' in u:
                vis.temporal_per_timestep = u['temporal_per_timestep']
            # 每个节点的 spatial attention: (N, n_spatial)
            if 'spatial_per_node' in u:
                vis.spatial_attention_per_node = u['spatial_per_node']
        
        # ==== 4. 收集 Proto Enhanced 特征 ====
        enhanced_list = getattr(self, '_last_proto_enhanced_first_batch', None)
        if enhanced_list is not None and len(enhanced_list) > 0:
            vis.identity_first_batch = []
            for enhanced_tensor in enhanced_list:
                if isinstance(enhanced_tensor, torch.Tensor):
                    vis.identity_first_batch.append(enhanced_tensor.numpy())
                else:
                    vis.identity_first_batch.append(enhanced_tensor)
        
        # ==== 5. 收集时间分组的原型使用率 ====
        acc_pstid = getattr(self, '_proto_vis_accumulator', None)
        assert acc_pstid is not None, \
            "collect_visualization_data: _proto_vis_accumulator not found on PSTID — did you run with collect_cache=True?"
        tod_last = np.array(acc_pstid.get('tod_last', []))
        temporal_att = acc_pstid.get('temporal_att', [])

        assert len(tod_last) > 0, \
            "collect_visualization_data: tod_last is empty — check _proto_vis_accumulator['tod_last']"
        assert len(temporal_att) > 0, \
            "collect_visualization_data: temporal_att is empty — check _proto_vis_accumulator['temporal_att']"

        all_temporal_att = np.concatenate(temporal_att, axis=0)  # (total_samples, n_temporal)
        n_tod = len(tod_last)
        n_att = len(all_temporal_att)
        assert n_tod == n_att, \
            f"collect_visualization_data: tod_last ({n_tod}) and temporal_att ({n_att}) count mismatch"

        unique_tods = np.unique(tod_last)
        logging.info(f"[Temporal by TOD] {n_tod} samples, {len(unique_tods)} unique time slots, "
                     f"tod range: [{unique_tods.min()}, {unique_tods.max()}]")

        temporal_by_tod = {}
        for tod in unique_tods:
            mask = tod_last == tod
            temporal_by_tod[int(tod)] = all_temporal_att[mask].mean(axis=0)  # (n_temporal,)

        vis.temporal_by_time_of_day = temporal_by_tod

        # ==== 6. 不在 collect_visualization_data 里清空 accumulator ====
        # 累加器由 PSTID.forward 在下次 collect_cache=True 时自动重置，避免
        # 同一份 cache 被多次 collect_visualization_data 调用读到不完整的数据。

        return vis

    def forward(self, batch, collect_cache=False):
        """PSTID 前向传播
        
        流程：
        1. 获取 STID 嵌入 (B, T, N, hidden_dim)
        2. 应用原型模块进行增强
        3. 转换为 (B, hidden_dim, N, T) 格式
        4. 通过 MLP Layers
        5. 通过 Regression Layer 输出
        
        Args:
            batch: 输入数据字典
            collect_cache: 是否收集可视化缓存数据（仅在评估/可视化时启用）
        """
        # 初始化/重置可视化数据累积器
        # 每次 collect_cache=True 都重置（覆盖上次数据），避免脏数据
        if collect_cache:
            self._proto_vis_accumulator = {
                'tod_last': [],
                'temporal_att': [],
            }
        
        # 1. 获取原始嵌入
        input_data = batch['X']  # (B, T, N, 3)
        _, time_series_emb, tid_emb, diw_emb, spatio_emb = self._get_embeddings(batch)
        # time_series_emb: (B, T, N, D_time_series)
        # tid_emb: (B, T, N, D_tid) 或 None
        # diw_emb: (B, T, N, D_diw) 或 None
        # spatio_emb: (N, D_spatio) 用于主模型的残差连接
        # adj_mx_emb: (N, D_adj_mx) 用于 ProtoModule 的空间码本

        # 2. 保存时间信息（用于可视化对齐）- 传递给 ProtoModule
        if collect_cache:
            # 提取 time_of_day 索引
            # input_data: (B, T, N, 3) 其中第1通道是 time_of_day，第2通道是 dow
            tod_data = input_data[..., 1]  # (B, T, N)
            
            # 获取最后一个时间步的索引作为 key
            tod_last = tod_data[:, -1, 0]  # (B,) - 每个样本最后一个时间步对应的 tod 值 (0~1)
            # 转换为索引 (0 ~ time_of_day_size-1)
            tod_last_idx = (tod_last * self.time_of_day_size).long().clamp(0, self.time_of_day_size - 1)
            
            # 初始化累积器
            if not hasattr(self, '_proto_vis_accumulator'):
                self._proto_vis_accumulator = {
                    'tod_last': [],
                    'temporal_att': [],
                }
            
            self._proto_vis_accumulator['tod_last'].extend(tod_last_idx.cpu().numpy().tolist())

        # 3. 应用原型模块（使用主模型已学习的时间嵌入）
        if self.use_proto and self.proto_module is not None:
            proto_enhanced, proto_info = self.proto_module.apply(
                time_series_emb=time_series_emb,
                adj_mx_emb=self.adj_mx_emb,
                tid_emb=tid_emb,  # 直接传递主模型的时间嵌入
                diw_emb=diw_emb,  # 直接传递主模型的星期嵌入
                collect_cache=collect_cache,
            )
            # 缓存 proto_enhanced（首样本）用于 visualization
            if collect_cache and proto_enhanced is not None:
                self._last_proto_enhanced_first_batch = [proto_enhanced[0, 0, :, :].detach().cpu()]

            # 累加 temporal attention 到 PSTID 的 accumulator（与 tod_last 配对）
            if collect_cache and proto_info is not None:
                temporal_att = proto_info.get('attention', {}).get('temporal', None)
                assert temporal_att is not None, \
                    "PSTID.forward (collect_cache): proto_info has no 'attention.temporal'"
                # temporal_att shape: (B, T, N, n_temporal)，对 T 和 N 求均值 → (B, n_temporal)
                att_per_sample = temporal_att.mean(dim=[1, 2]).detach().cpu().numpy()
                self._proto_vis_accumulator['temporal_att'].append(att_per_sample)
        else:
            proto_enhanced = None
        
        # 3. 拼接所有嵌入和原型增强结果
        # 扩展 spatio_emb 到 (B, T, N, D_spatio)
        B, T, N, _ = time_series_emb.shape
        spatio_emb_expanded = spatio_emb.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)
        
        # 拼接: time_series_emb + tid_emb + diw_emb + spatio_emb_expanded
        emb_list = [time_series_emb]
        if tid_emb is not None:
            emb_list.append(tid_emb)
        if diw_emb is not None:
            emb_list.append(diw_emb)
        emb_list.append(spatio_emb_expanded)
        
        hidden = torch.cat(emb_list, dim=-1)  # (B, T, N, hidden_dim)

        if proto_enhanced is not None:
            hidden = hidden + proto_enhanced
        
        # 4. 转换为 (B, hidden_dim, N, T) 格式
        hidden = hidden.permute(0, 3, 2, 1)  # (B, hidden_dim, N, T)
        
        # 5. 通过 MLP Layers
        hidden = self.encoder(hidden)
        
        # 6. 通过 Regression Layer
        prediction = self.regression_layer(hidden)[..., 0:1]
        
        return prediction

    def compute_laplacian_loss(self, adj_mx: torch.Tensor = None, weight: float = 0.01) -> torch.Tensor:
        """拉普拉斯正则化损失：相邻节点的 adj_mx_emb 应该相似
        
        L = I - D^{-1/2} A D^{-1/2}（对称归一化拉普拉斯）
        loss = tr(emb^T L emb) / tr(emb^T emb)
        
        Args:
            adj_mx: 邻接矩阵 (N, N)，如果为 None 则使用缓存的 self._cached_adj_mx
            weight: 正则化权重
        
        Returns:
            laplacian_loss: 标量损失
        """
        # 获取 adj_mx
        if adj_mx is None:
            adj_mx = getattr(self, '_cached_adj_mx', None)
        if adj_mx is None:
            return torch.tensor(0.0, device=self.adj_mx_emb.device)
        
        adj = adj_mx.to(self.adj_mx_emb.device)
        emb = self.adj_mx_emb  # (N, D)
        
        # 计算度矩阵
        d = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (N, 1)
        D_inv_sqrt = torch.pow(d, -0.5)  # (N, 1)
        
        # 对称归一化拉普拉斯: L = I - D^{-1/2} A D^{-1/2}
        L = torch.eye(adj.shape[0], device=adj.device) - D_inv_sqrt * adj * D_inv_sqrt.t()
        
        # 损失: tr(emb^T L emb) / tr(emb^T emb)
        loss = torch.trace(emb.T @ L @ emb) / (emb.pow(2).sum() + 1e-8)
        
        return weight * loss

    def get_proto_usage_summary(self, decimals: int = 2) -> str:
        """生成原型使用情况摘要

        与 PSTID.get_proto_usage_summary 保持一致
        """
        if not self.use_proto or self.proto_module is None:
            return ''

        usages = getattr(self.proto_module, '_last_proto_usage', None)
        if usages is None or len(usages) == 0:
            return '⚠️ _last_proto_usage is None or empty'

        u = usages[0]
        if u is None:
            return '⚠️ usages[0] is None'

        parts = []

        if self.use_spatio:
            if 'spatial' in u and u['spatial']:
                spatial_vals = u['spatial']
                n = len(spatial_vals)
                # 统计每个原型的"理论"期望分配节点数
                total_nodes = 137  # 假设
                expected_per_proto = total_nodes / n
                parts.append(f"sp(n={n},exp={expected_per_proto:.1f})[{', '.join(f'{v:.2f}' for v in spatial_vals)}]")
            else:
                parts.append(f"sp=[]")

        if self.use_temporal:
            if 'temporal' in u and u['temporal']:
                temporal_vals = u['temporal']
                n = len(temporal_vals)
                parts.append(f"tp[{', '.join(f'{v:.2f}' for v in temporal_vals)}]")
            else:
                parts.append(f"tp=[]")

        return ' '.join(parts)

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
            use_spatio=_get_arg(args, 'use_spatio', True),
            use_temporal=_get_arg(args, 'use_temporal', True),
            proto_uniformity_weight=_get_arg(args, 'proto_uniformity_weight', 0.0),
            # 新增参数
            proto_emb_dim=_get_arg(args, 'proto_emb_dim', 64),
            # adj_mx 衍生的可学习节点嵌入维度
            adj_mx_emb_dim=_get_arg(args, 'adj_mx_emb_dim', 64),
            # 传入 adj_mx 用于谱嵌入初始化
            adj_mx=adj_mx,
        )

    def cache_adj_mx(self, adj_mx: torch.Tensor):
        """缓存 adj_mx 用于拉普拉斯正则化（不需要梯度）"""
        self._cached_adj_mx = adj_mx.detach()
