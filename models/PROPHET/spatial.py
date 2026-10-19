"""PROPHET spatial components."""

import torch
from torch import nn
import torch.nn.functional as F


class SpatialCodebook(nn.Module):
    """空间码本：编码静态/半静态节点身份

    职责：
    - 接收主特征 (B,T,N,D) 和 adj_mx_emb（可学习节点嵌入）
    - 自行完成节点均值计算、维度对齐、投影、码本注意力融合
    - 返回原型增强特征和用于残差的 Query 投影

    分配策略：
    - 使用 softmax 软分配，根据相似度加权所有原型
    """

    def __init__(self, num_protos: int, proto_emb_dim: int = None,
                 adj_mx_emb_dim: int = None, spatial_emb_dim: int = None,
                 time_series_dim: int = None,
                 temperature: float = 2.0, use_his: bool = True):
        super().__init__()
        self.num_protos = num_protos
        self.proto_emb_dim = proto_emb_dim
        self.temperature = temperature
        self.use_his = use_his

        # 原型参数
        self.prototypes = nn.Parameter(torch.randn(num_protos, self.proto_emb_dim))
        nn.init.orthogonal_(self.prototypes)

        # 统一投影层：拼接 (空间嵌入 + time_series_mean) → proto_emb_dim
        # use_adj=True 时用 adj_mx_emb_dim，use_adj=False 时用 spatial_emb_dim
        codebook_emb_dim = adj_mx_emb_dim if adj_mx_emb_dim is not None else spatial_emb_dim
        combined_dim = codebook_emb_dim + (time_series_dim if use_his else 0)
        self.proj = nn.Linear(combined_dim, proto_emb_dim)

    def _compute_attention(self, proj_feat: torch.Tensor) -> tuple:
        """计算原型注意力（proj_feat 形状: (N, D)）

        Args:
            proj_feat: 投影特征 (N, proto_emb_dim)

        Returns:
            attention: (N, num_protos) 软分配权重
            proto_enhanced: (N, proto_emb_dim) 原型增强特征
        """
        # 点积相似度：各自 L2 归一化后做内积，即余弦相似度
        proj_norm = F.normalize(proj_feat, p=2, dim=-1)   # (N, D)
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)  # (num_protos, D)
        sim = proj_norm @ proto_norm.t()  # (N, num_protos)

        # 软分配（温度越大分布越均匀）
        soft_weights = F.softmax(sim / self.temperature, dim=-1)  # (N, num_protos)

        # 软加权求和
        proto_enhanced = soft_weights @ self.prototypes  # (N, D)

        return soft_weights, proto_enhanced

    def forward(self, time_series_emb: torch.Tensor, adj_mx_emb: torch.Tensor = None,
                node_emb: torch.Tensor = None) -> dict:
        """完整前向传播：投影 + 码本注意力

        Args:
            time_series_emb: 时间序列嵌入 (B, T, N, D_time_series)，use_his=False 时仅取形状
            adj_mx_emb: adj_mx 衍生的可学习节点嵌入 (N, D_adj_mx)
            node_emb: 节点可学习嵌入 (N, spatial_emb_dim)，当 use_adj=False 时使用

        Returns:
            dict with keys:
                proto_enhanced: (B, T, N, proto_emb_dim) 原型增强特征
                proj_feat: (B, T, N, proto_emb_dim) 投影特征
        """
        B, T, N, _ = time_series_emb.shape

        # ===== 静态 Query：所有样本共享 =====
        # 空间嵌入优先使用 adj_mx_emb（包含图结构信息），否则回退到 node_emb
        spatial_emb = adj_mx_emb if adj_mx_emb is not None else node_emb
        x_combined = spatial_emb
        if self.use_his:
            x_node_mean = time_series_emb.mean(dim=1).mean(dim=0)  # (N, D_time_series)
            x_combined = torch.cat([spatial_emb, x_node_mean], dim=-1)
        proj_feat = self.proj(x_combined)  # (N, proto_emb_dim)

        _, proto_enhanced = self._compute_attention(proj_feat)
        # proto_enhanced: (N, proto_emb_dim)

        # reshape 回 (B, T, N, dim)
        proto_enhanced = proto_enhanced.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)
        proj_feat = proj_feat.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)

        return {
            'proto_enhanced': proto_enhanced,
            'proj_feat': proj_feat,
        }
