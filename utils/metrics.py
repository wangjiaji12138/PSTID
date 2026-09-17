"""
评估指标模块

提供 PyTorch 和 NumPy 两套评估接口:
- masked_* 系列: PyTorch 张量计算 (训练时使用)
- *_np 系列: NumPy 数组计算 (评估时使用)
"""

import numpy as np
import torch


def masked_mae(preds: torch.Tensor,
               labels: torch.Tensor,
               null_val: float = np.nan,
               mask: torch.Tensor = None) -> torch.Tensor:
    """Masked MAE: 忽略 null_val 的预测"""
    if mask is None:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels > null_val + 0.1)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse(preds: torch.Tensor,
               labels: torch.Tensor,
               null_val: float = np.nan,
               mask: torch.Tensor = None) -> torch.Tensor:
    """Masked MSE: 忽略 null_val 的预测"""
    if mask is None:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels > null_val + 0.1)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds: torch.Tensor,
                labels: torch.Tensor,
                null_val: float = np.nan,
                mask: torch.Tensor = None) -> torch.Tensor:
    """Masked RMSE: 忽略 null_val 的预测"""
    return torch.sqrt(masked_mse(preds, labels, null_val, mask))


def masked_mape(preds: torch.Tensor,
                labels: torch.Tensor,
                null_val: float = np.nan,
                mask: torch.Tensor = None,
                threshold: float = 0.5) -> torch.Tensor:
    """Masked MAPE: 只选择 labels >= threshold 的样本计算"""
    if mask is None:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels > null_val + 0.1)
        mask = mask & (labels >= threshold)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) * 100


def masked_r2(preds: torch.Tensor,
             labels: torch.Tensor,
             null_val: float = np.nan,
             mask: torch.Tensor = None) -> torch.Tensor:
    """Masked R² (Coefficient of Determination): 衡量模型解释方差的比例
    
    R² = 1 - SS_res / SS_tot
    其中:
    - SS_res = Σ(y_true - y_pred)² (残差平方和)
    - SS_tot = Σ(y_true - y_mean)² (总平方和)
    
    范围通常为 (-∞, 1]，1表示完美预测，0表示预测等于均值
    """
    if mask is None:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels > null_val + 0.1)
    
    mask = mask.float()
    
    # 计算残差平方和
    residuals = (labels - preds) * mask
    ss_res = torch.sum(residuals ** 2)
    
    # 计算总平方和 (相对于均值的方差)
    labels_masked = labels * mask
    mean_label = torch.sum(labels_masked) / torch.sum(mask)
    ss_tot = torch.sum(((labels - mean_label) * mask) ** 2)
    
    # 避免除零
    if ss_tot < 1e-10:
        return torch.tensor(0.0, device=preds.device)
    
    r2 = 1 - ss_res / ss_tot
    return r2


def masked_smape(preds: torch.Tensor,
               labels: torch.Tensor,
               null_val: float = np.nan,
               mask: torch.Tensor = None,
               threshold: float = 0.1) -> torch.Tensor:
    """Masked SMAPE (Symmetric Mean Absolute Percentage Error): 对称MAPE
    """
    if mask is None:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels > null_val + 0.1)
        # SMAPE对接近零的值也是稳定的，不需要额外阈值过滤
        # threshold < 0 时不会过滤任何数据
        if threshold > 0:
            mask = mask & (labels >= threshold)

    mask = mask.float()

    abs_diff = torch.abs(labels - preds) * mask
    abs_sum = (torch.abs(labels) + torch.abs(preds)) * mask

    # 避免除零: 当abs_sum为0时（y_true和y_pred都为0），该项贡献为0
    eps = 1e-6
    ratio = abs_diff / (abs_sum / 2 + eps)

    # 只有当abs_sum > eps时才计入
    valid_mask = abs_sum > eps
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device)

    smape = torch.mean(ratio[valid_mask]) * 100
    return smape


def masked_mae_np(y_pred: np.ndarray,
                  y_true: np.ndarray,
                  null_val: float = np.nan) -> float:
    """NumPy 版本 Masked MAE"""
    if np.isnan(null_val):
        mask = ~np.isnan(y_true)
    else:
        mask = y_true > null_val + 0.1

    masked_pred = np.where(mask, y_pred, 0)
    masked_true = np.where(mask, y_true, 0)

    return np.mean(np.abs(masked_pred - masked_true))


def masked_mse_np(y_pred: np.ndarray,
                  y_true: np.ndarray,
                  null_val: float = np.nan) -> float:
    """NumPy 版本 Masked MSE"""
    if np.isnan(null_val):
        mask = ~np.isnan(y_true)
    else:
        mask = y_true > null_val + 0.1

    masked_pred = np.where(mask, y_pred, 0)
    masked_true = np.where(mask, y_true, 0)

    return np.mean((masked_pred - masked_true) ** 2)


def masked_rmse_np(y_pred: np.ndarray,
                   y_true: np.ndarray,
                   null_val: float = np.nan) -> float:
    """NumPy 版本 Masked RMSE"""
    return np.sqrt(masked_mse_np(y_pred, y_true, null_val))


def masked_mape_np(y_pred: np.ndarray,
                   y_true: np.ndarray,
                   threshold: float = 0.5) -> float:
    """NumPy 版本 Masked MAPE (过滤掉接近0的值)"""
    mask = y_true > threshold
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs(y_pred[mask] - y_true[mask]) / y_true[mask]) * 100


def compute_all_metrics_torch(pred: torch.Tensor,
                              real: torch.Tensor,
                              null_value: float = np.nan) -> tuple:
    """综合评估指标 (PyTorch版本)

    Returns:
        tuple: (mae, rmse, mape, r2, smape)
    """
    mae = masked_mae(pred, real, null_value).item()
    rmse = masked_rmse(pred, real, null_value).item()
    mape = masked_mape(pred, real, null_value).item()
    r2 = masked_r2(pred, real, null_value).item()
    smape = masked_smape(pred, real, null_value).item()
    return mae, rmse, mape, r2, smape


def masked_r2_np(y_pred: np.ndarray,
                 y_true: np.ndarray,
                 null_val: float = np.nan) -> float:
    """NumPy 版本 Masked R²"""
    if np.isnan(null_val):
        mask = ~np.isnan(y_true)
    else:
        mask = y_true > null_val + 0.1

    y_true_masked = np.where(mask, y_true, 0)
    y_pred_masked = np.where(mask, y_pred, 0)

    residuals = y_true_masked - y_pred_masked
    ss_res = np.sum(residuals ** 2)

    mean_label = np.sum(y_true_masked) / np.sum(mask)
    ss_tot = np.sum(((y_true_masked - mean_label) ** 2) * mask)

    if ss_tot < 1e-10:
        return 0.0

    r2 = 1 - ss_res / ss_tot
    return r2


def masked_smape_np(y_pred: np.ndarray,
                   y_true: np.ndarray,
                   threshold: float = 0.1) -> float:
    """NumPy 版本 Masked SMAPE (Symmetric MAPE)
    
    当 threshold < 0 时不过滤任何数据，保留SMAPE的对称稳定性优势。
    """
    if threshold > 0:
        mask = y_true > threshold
        if mask.sum() == 0:
            return 0.0
    else:
        mask = np.ones_like(y_true, dtype=bool)

    abs_diff = np.abs(y_true - y_pred)
    abs_sum = np.abs(y_true) + np.abs(y_pred)

    ratio = abs_diff / (abs_sum / 2)
    smape = 2 * np.mean(ratio[mask]) * 100
    return smape


def compute_all_metrics(pred: np.ndarray,
                        real: np.ndarray,
                        null_value: float = np.nan) -> dict:
    """综合评估指标 (NumPy版本)

    Returns:
        dict: 包含 MAE, RMSE, MAPE, R2, SMAPE 指标
    """
    pred = pred.flatten()
    real = real.flatten()
    return {
        'MAE': masked_mae_np(pred, real, null_value),
        'RMSE': masked_rmse_np(pred, real, null_value),
        'MAPE': masked_mape_np(pred, real),
        'R2': masked_r2_np(pred, real, null_value),
        'SMAPE': masked_smape_np(pred, real)
    }


mae_torch = masked_mae
"""MAE 损失函数 (PyTorch版本), 用于训练时的 criterion"""
