"""Shared value-channel normalization; time features remain unchanged."""

import numpy as np
import torch


def compute_scaler_stats(dataset, input_dim=3, output_dim=1):
    """Compute mean and std for data normalization.

    Args:
        dataset: Dataset dict with train/val/test splits
        input_dim: Number of input channels
        output_dim: Number of output channels
    """
    # Use first data channel for scaler (exclude time features)
    train_X = dataset['train']['X'][..., :1]
    train_y = dataset['train']['y']
    all_values = np.concatenate([train_X.flatten(), train_y.flatten()])

    mean = float(np.mean(all_values))
    std = float(np.std(all_values))

    return mean, std


def normalize_input(X, scaler, input_dim=3):
    """Normalize input for multi-channel data.

    Args:
        X: Input tensor (B, T, N, input_dim) - first (input_dim-2) channels are data,
           last 2 channels are time features (hour, dow)
        scaler: StandardScaler for data channels
        input_dim: Number of input channels

    Returns:
        Normalized tensor with data channels normalized and time features as-is
    """
    if input_dim != 3 or X.shape[-1] != 3:
        raise ValueError("Expected three channels: value, time_of_day, day_of_week")
    time_feats = X[..., -2:]  # Last 2 channels: hour, dow

    if isinstance(X, torch.Tensor):
        mean = torch.as_tensor(scaler.mean, device=X.device, dtype=X.dtype)
        std = torch.as_tensor(scaler.std, device=X.device, dtype=X.dtype)
        assert std > 1e-6, "Standard deviation is too small"
        data_norm = (X[..., :1] - mean) / std

        return torch.cat([data_norm, time_feats], dim=-1)

    if np.any(np.asarray(scaler.std) <= 1e-6):
        raise ValueError("Standard deviation is too small")
    data_norm = (X[..., :1] - scaler.mean) / scaler.std
    return np.concatenate([data_norm, time_feats], axis=-1)
