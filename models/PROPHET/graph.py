"""PROPHET graph components."""

import torch
import numpy as np


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
