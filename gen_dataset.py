"""
数据处理脚本
生成标准深度学习格式的数据集

输出结构:
    data/
    └── processed/
        └── dataset_name/
            ├── train.npz      # 训练集
            ├── val.npz        # 验证集
            ├── test.npz       # 测试集
            ├── adj.pkl        # 邻接矩阵
            └── metadata.json  # 元数据
"""

import pandas as pd
import numpy as np
import argparse
import os
import json
import pickle
from geopy.distance import geodesic


class DataProcessor:
    """数据处理器：生成训练所需的标准格式数据"""

    def __init__(self,
                 city: str = "sh",
                 grid_size: float = 0.05,
                 time_col: str = "pickup_time",
                 time_size: str = "30min",
                 day_start_hour: int = 8,
                 day_end_hour: int = 20,
                 sigma_scale: float = 2.0,
                 train_ratio: float = 0.6,
                 val_ratio: float = 0.2,
                 input_window: int = 24,
                 output_window: int = 24):
        """
        Args:
            city: 城市名称
            grid_size: 网格大小（经纬度）
            time_col: 时间列名
            time_size: 时间粒度
            day_start_hour: 每天开始小时
            day_end_hour: 每天结束小时
            sigma_scale: 高斯核sigma缩放因子
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            input_window: 输入序列长度
            output_window: 预测序列长度
        """
        self.city = city
        self.grid_size = grid_size
        self.time_col = time_col
        self.time_size = time_size
        self.day_start_hour = day_start_hour
        self.day_end_hour = day_end_hour
        self.sigma_scale = sigma_scale
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.input_window = input_window
        self.output_window = output_window

        # 路径设置
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(script_dir, "data")
        self.data_raw = os.path.join(self.data_dir, "raw")
        self.raw_data_path = os.path.join(self.data_raw, f"pickup_{city.lower()}.csv")

        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data not found: {self.raw_data_path}")

        self.raw_data = pd.read_csv(self.raw_data_path)
        print(f"Data loading completed, {len(self.raw_data)} records")

        self.processed_data = None
        self.grid_idx = None
        self.gridSplitter = GridSplitter(city, grid_size, self.raw_data)
        self.gridSplitter.find_border()
        self.gridSplitter.create_grid_idx()
        self.gridSplitter.find_grid_id()

    def process(self):
        """执行完整的数据处理流程"""
        self._preprocess_data()
        self._build_time_series()
        self._generate_samples()
        self._save_dataset()

    def _preprocess_data(self):
        """数据预处理：时间转换、网格划分"""
        self.processed_data = self.raw_data.copy()

        def add_year(time_str):
            if pd.isna(time_str):
                return time_str
            time_str = str(time_str).strip()
            if len(time_str.split()[0].split('-')) == 2:
                month_day = time_str.split()[0]
                time_part = time_str.split()[1] if len(time_str.split()) > 1 else "00:00:00"
                return f"2022-{month_day} {time_part}"
            return time_str

        self.processed_data['datetime'] = pd.to_datetime(
            self.processed_data[self.time_col].apply(add_year),
            errors='coerce'
        )
        self.processed_data = self.processed_data.dropna(subset=['datetime'])

        self.processed_data['grid_id'] = self.gridSplitter.get_mapped_grid_ids(
            self.processed_data['lng'].values,
            self.processed_data['lat'].values
        )

        self.processed_data = self.processed_data.set_index('datetime')
        self.processed_data = self.processed_data.between_time(
            f'{self.day_start_hour:02d}:00:00',
            f'{self.day_end_hour:02d}:59:59'
        )

    def _build_time_series(self):
        """构建完整时空矩阵"""
        groups = self.processed_data.groupby(pd.Grouper(freq=self.time_size, label='left', closed='left'))

        min_time = self.processed_data.index.min()
        max_time = self.processed_data.index.max()
        full_time_range = pd.date_range(
            start=min_time.floor('D') + pd.Timedelta(hours=self.day_start_hour),
            end=max_time.floor('D') + pd.Timedelta(hours=self.day_end_hour, minutes=59, seconds=59),
            freq=self.time_size
        )

        from datetime import time as dt_time
        start_time = dt_time(hour=self.day_start_hour)
        end_time = dt_time(hour=self.day_end_hour, minute=59, second=59)
        valid_mask = [(ts.time() >= start_time and ts.time() <= end_time) for ts in full_time_range]
        self.full_time_range = full_time_range[valid_mask]
        time_to_idx = {ts: i for i, ts in enumerate(self.full_time_range)}

        valid_grids = self.gridSplitter.valid_grids
        self.N = len(valid_grids)

        self.dataset = np.zeros(shape=(len(self.full_time_range), self.N, 1))

        for ts, group in groups:
            if ts in time_to_idx:
                idx = time_to_idx[ts]
                for i, grid in enumerate(valid_grids):
                    self.dataset[idx, i, 0] = len(group[group.grid_id == grid[0]])

        print(f'Time stamps: {len(self.full_time_range)}, Nodes: {self.N}')

    def _generate_samples(self):
        """生成训练样本，包含完整的特征矩阵"""
        x_offsets = np.sort(np.concatenate((np.arange(-(self.input_window - 1), 1),)))
        y_offsets = np.sort(np.arange(1, self.output_window + 1))

        x_samples = []
        y_samples = []
        time_indices = []

        min_t = abs(min(x_offsets))
        max_t = len(self.full_time_range) - max(y_offsets)

        for t in range(min_t, max_t):
            x_ = self.dataset[t + x_offsets]
            y_ = self.dataset[t + y_offsets]
            x_samples.append(x_)
            y_samples.append(y_)
            time_indices.append(t)

        self.num_samples = len(x_samples)
        self.x_all = np.stack(x_samples, axis=0)
        self.y_all = np.stack(y_samples, axis=0)

        print(f"Total samples: {self.num_samples}")

    def _build_feature_matrix(self, time_indices: np.ndarray) -> np.ndarray:
        """
        构建特征矩阵 X

        返回 shape: (num_samples, input_window, N, 3)
        - X[..., 0]: 数值特征 (订单数)
        - X[..., 1]: hour of day (归一化到 [0, 1])
        - X[..., 2]: day of week (归一化到 [0, 1])
        """
        num_samples = len(time_indices)
        X = np.zeros((num_samples, self.input_window, self.N, 3), dtype=np.float32)

        for i, t_idx in enumerate(time_indices):
            for j in range(self.input_window):
                ts = self.full_time_range[t_idx + j]

                # 数值特征 (原始订单数)
                X[i, j, :, 0] = self.x_all[i, j, :, 0]

                # Hour of day (归一化)
                hour = ts.hour + ts.minute / 60.0
                X[i, j, :, 1] = hour / 24.0

                # Day of week (归一化)
                dow = ts.dayofweek
                X[i, j, :, 2] = dow / 7.0

        return X

    def _save_dataset(self):
        """保存完整数据集到文件"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        dataset_name = self.city.lower()
        output_dir = os.path.join(data_dir, "processed", dataset_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 构建完整特征矩阵
        all_indices = np.arange(self.num_samples)
        X = self._build_feature_matrix(all_indices)

        # 保存所有数据到一个 npz 文件
        npz_path = os.path.join(output_dir, 'all.npz')
        np.savez_compressed(
            npz_path,
            X=X,
            y=self.y_all
        )
        print(f"Saved all.npz: X shape {X.shape}, y shape {self.y_all.shape}")

        # 保存邻接矩阵
        adj_matrix = self.gridSplitter.get_adjacency_matrix()
        adj_path = os.path.join(output_dir, 'adj.pkl')
        with open(adj_path, 'wb') as f:
            pickle.dump(adj_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved adj.pkl: shape {adj_matrix.shape}")

        # 保存元数据
        metadata = {
            'city': self.city,
            'grid_size': self.grid_size,
            'time_size': self.time_size,
            'num_nodes': self.N,
            'input_window': self.input_window,
            'output_window': self.output_window,
            'total_samples': self.num_samples,
            'num_timestamps': len(self.full_time_range),
            'time_range': {
                'start': str(self.full_time_range[0]),
                'end': str(self.full_time_range[-1])
            }
        }
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata.json")

        print(f"\nDataset saved to: {output_dir}")


class GridSplitter:
    """网格分割器：管理空间网格和邻接矩阵"""

    def __init__(self, city: str, grid_size: float, raw_data: pd.DataFrame):
        self.city = city
        self.grid_size = grid_size
        self.raw_data = raw_data
        self.border = None
        self.grids = []
        self.valid_grids = []
        self._id_mapping = {}
        self._inverse_mapping = {}

    def find_border(self):
        lng_min = float(self.raw_data["lng"].min())
        lng_max = float(self.raw_data["lng"].max())
        lat_min = float(self.raw_data["lat"].min())
        lat_max = float(self.raw_data["lat"].max())

        self.lng_min = float(int(lng_min / self.grid_size)) * self.grid_size
        self.lng_max = (float(int(lng_max / self.grid_size)) + 1) * self.grid_size
        self.lat_min = float(int(lat_min / self.grid_size)) * self.grid_size
        self.lat_max = (float(int(lat_max / self.grid_size)) + 1) * self.grid_size

    def create_grid_idx(self):
        idx = 0
        n_lng = int((self.lng_max - self.lng_min) / self.grid_size) + 1
        n_lat = int((self.lat_max - self.lat_min) / self.grid_size) + 1

        for i in range(n_lng):
            for j in range(n_lat):
                lng_min_val = self.lng_min + i * self.grid_size
                lng_max_val = lng_min_val + self.grid_size
                lat_min_val = self.lat_min + j * self.grid_size
                lat_max_val = lat_min_val + self.grid_size
                self.grids.append((idx, lng_min_val, lng_max_val, lat_min_val, lat_max_val))
                idx += 1

    def find_grid_id(self):
        n_lng = int((self.lng_max - self.lng_min) / self.grid_size) + 1
        n_lat = int((self.lat_max - self.lat_min) / self.grid_size) + 1

        lng_arr = self.raw_data['lng'].values
        lat_arr = self.raw_data['lat'].values
        lng_idx = ((lng_arr - self.lng_min) / self.grid_size).astype(int)
        lat_idx = ((lat_arr - self.lat_min) / self.grid_size).astype(int)
        original_grid_ids = (lng_idx * n_lat + lat_idx).tolist()

        unique_original_ids = sorted(set(original_grid_ids))

        self._id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_original_ids)}
        self._inverse_mapping = {new_id: old_id for new_id, old_id in enumerate(unique_original_ids)}

        for new_id, old_id in enumerate(unique_original_ids):
            grid = self.grids[old_id]
            self.valid_grids.append((new_id, grid[1], grid[2], grid[3], grid[4]))

        print(f"Grid mapping: {len(unique_original_ids)} unique grids -> IDs 0-{len(unique_original_ids)-1}")

    def get_mapped_grid_ids(self, lng_arr, lat_arr):
        n_lng = int((self.lng_max - self.lng_min) / self.grid_size) + 1
        n_lat = int((self.lat_max - self.lat_min) / self.grid_size) + 1

        lng_idx = ((lng_arr - self.lng_min) / self.grid_size).astype(int)
        lat_idx = ((lat_arr - self.lat_min) / self.grid_size).astype(int)
        original_ids = lng_idx * n_lat + lat_idx

        mapped_ids = np.array([self._id_mapping.get(id, -1) for id in original_ids])
        return mapped_ids

    def get_adjacency_matrix(self, sigma_scale: float = 2.0) -> np.ndarray:
        """
        计算基于距离的邻接矩阵

        使用 log-距离高斯核: w_ij = exp(-(log(1+d_ij)/sigma_log)^2)

        Returns:
            np.ndarray: (N, N) 邻接矩阵
        """
        N = len(self.valid_grids)
        centers = []

        for grid in self.valid_grids:
            lng_center = (grid[1] + grid[2]) / 2
            lat_center = (grid[3] + grid[4]) / 2
            centers.append((lat_center, lng_center))

        dist_mat = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if i != j:
                    dist_mat[i, j] = geodesic(centers[i], centers[j]).kilometers

        log_d = np.log1p(dist_mat)
        finite_mask = np.isfinite(dist_mat)
        sigma_log = sigma_scale * np.std(log_d[finite_mask])

        weights = np.zeros((N, N), dtype=np.float32)
        weights[finite_mask] = np.exp(-np.square(log_d[finite_mask] / sigma_log))

        return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processor')
    parser.add_argument('--city', type=str, help='City name (e.g., sh, cq, jl)')
    parser.add_argument('--all', action='store_true', help='Process all cities')
    parser.add_argument('--grid_size', type=float, default=0.05)
    parser.add_argument('--time_size', type=str, default="30min")
    parser.add_argument('--time_col', type=str, default='pickup_time')
    parser.add_argument('--day_start_hour', type=int, default=8)
    parser.add_argument('--day_end_hour', type=int, default=20)
    parser.add_argument('--sigma_scale', type=float, default=2.0)
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--input_window', type=int, default=24, help='输入序列长度')
    parser.add_argument('--output_window', type=int, default=24, help='预测序列长度')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(script_dir, "data", "raw")

    if args.all:
        cities = []
        for fn in os.listdir(raw_dir):
            if fn.startswith("pickup_") and fn.endswith(".csv"):
                city = fn[len("pickup_"):-len(".csv")]
                if city:
                    cities.append(city)
        cities = sorted(set(cities))
    elif args.city:
        cities = [args.city]
    else:
        raise ValueError("Please specify --city or use --all")

    for city in cities:
        print(f"\n{'='*60}\nProcessing: {city}\n{'='*60}")
        try:
            processor = DataProcessor(
                city=city,
                grid_size=args.grid_size,
                time_col=args.time_col,
                time_size=args.time_size,
                day_start_hour=args.day_start_hour,
                day_end_hour=args.day_end_hour,
                sigma_scale=args.sigma_scale,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                input_window=args.input_window,
                output_window=args.output_window,
            )
            processor.process()
            print(f"Done: {city}")
        except Exception as e:
            import traceback
            print(f"Failed: {city}\n{traceback.format_exc()}")
