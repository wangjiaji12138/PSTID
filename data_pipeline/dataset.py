"""
数据加载工具
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


def load_dataset(data_dir: str, dataset_name: str,
                 train_ratio: float = 0.6, val_ratio: float = 0.2,
                 seed: int = 42):
    """
    加载数据集

    Args:
        data_dir: 数据目录
        dataset_name: 数据集名称 (如 "cq")
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子 (用于打乱数据)

    Returns:
        dict: 包含 train/val/test npz 数据和 adj 矩阵
    """
    data_path = os.path.join(data_dir, dataset_name)

    dataset = {}

    # 加载 all.npz
    npz_file = os.path.join(data_path, 'all.npz')
    if os.path.exists(npz_file):
        all_data = np.load(npz_file)
    else:
        raise FileNotFoundError(f"Data file not found: {npz_file}")

    # 划分数据集
    num_samples = len(all_data['X'])
    indices = np.arange(num_samples)

    # 设置随机种子并打乱
    rng = np.random.RandomState(seed)
    # rng.shuffle(indices)

    num_train = round(num_samples * train_ratio)
    num_val = round(num_samples * val_ratio)

    train_idx = indices[:num_train]
    val_idx = indices[num_train:num_train + num_val]
    test_idx = indices[num_train + num_val:]

    # 构建分割后的数据字典
    dataset['train'] = {
        'X': all_data['X'][train_idx],
        'y': all_data['y'][train_idx]
    }
    dataset['val'] = {
        'X': all_data['X'][val_idx],
        'y': all_data['y'][val_idx]
    }
    dataset['test'] = {
        'X': all_data['X'][test_idx],
        'y': all_data['y'][test_idx]
    }

    # 加载邻接矩阵
    adj_file = os.path.join(data_path, 'adj.pkl')
    if os.path.exists(adj_file):
        with open(adj_file, 'rb') as f:
            dataset['adj_mx'] = pickle.load(f)
    else:
        raise FileNotFoundError(f"Adjacency file not found: {adj_file}")

    return dataset


class PSTIDDataset(Dataset):
    """时空图预测数据集

    设计原则:
    - Dataset返回原始值(X和y都不做归一化)
    - 归一化在训练循环中显式进行
    - 训练/验证/测试使用完全一致的流程
    """

    def __init__(self, data_dict, split='train'):
        """
        Args:
            data_dict: load_dataset 返回的字典
            split: 'train', 'val', 'test'
        """
        self.X = data_dict[split]['X']  # [B, T, N, 3] 原始值
        self.y = data_dict[split]['y']  # [B, T, N, 1] 原始值

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 返回原始值，不做任何归一化
        return {
            'X': torch.from_numpy(self.X[idx]).float(),  # [T, N, 3] 原始
            'y': torch.from_numpy(self.y[idx]).float(),  # [T, N, 1] 原始
        }


class StandardScaler:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)
        return self

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """逆标准化: y_unnorm = y_pred * σ + μ"""
        if isinstance(data, torch.Tensor):
            if not hasattr(self, '_mean_tensor') or self._mean_tensor.device != data.device:
                if isinstance(self.mean, torch.Tensor):
                    self._mean_tensor = self.mean.to(device=data.device, dtype=data.dtype)
                    self._std_tensor = self.std.to(device=data.device, dtype=data.dtype)
                else:
                    self._mean_tensor = torch.as_tensor(self.mean, device=data.device, dtype=data.dtype)
                    self._std_tensor = torch.as_tensor(self.std, device=data.device, dtype=data.dtype)
            return data * self._std_tensor + self._mean_tensor
        else:
            return data * self.std + self.mean
