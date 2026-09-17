"""
数据集处理脚本 - 支持多个数据集
将原始NPZ格式数据转换为PSTID训练所需的格式

用法:
    python prepare_dataset.py NYC_TAXI
    python prepare_dataset.py CHI_TAXI
    python prepare_dataset.py --all

支持的数据集:
    - NYC_TAXI: 纽约出租车数据 (30分钟间隔)
    - CHI_TAXI: 芝加哥出租车数据 (30分钟间隔)

输入:
    data/raw/<dataset_name>/
        ├── <dataset_name>.npz      # 原始数据 (N, T, C) - 节点数, 时间步, 通道
        └── <dataset_name>_rn_adj.npy # 邻接矩阵 (N, N)

输出:
    data/processed/<dataset_name_lower>/
        ├── all.npz           # 全部数据 (X, y)
        └── adj.pkl           # 邻接矩阵

时间特征 (time_of_day 连续值):
    X: (num_samples, input_window, N, 3) - 数据 + time_of_day + dow
    time_of_day: (time_index % intervals_per_day) / intervals_per_day  # 0~1 连续值
    dow: (time_index // intervals_per_day) % 7 / 7.0  # 0~1 归一化
"""

import os
import argparse
import pickle
import numpy as np

# 数据集配置
DATASET_CONFIG = {
    'NYC_TAXI': {
        'time_interval': 1800,  # 30分钟
        'default_start': '20160101',
        'default_end': '20170630',
        'raw_path': 'data/raw/NYC_TAXI',
    },
    'CHI_TAXI': {
        'time_interval': 1800,  # 30分钟
        'default_start': None,  # 全部数据
        'default_end': None,
        'raw_path': 'data/raw/CHI_TAXI',
    },
}


def parse_date(date_str):
    """解析日期字符串 YYYYMMDD"""
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    return year, month, day


def load_raw_data(raw_dir: str, dataset_name: str):
    """加载原始数据"""
    npz_path = os.path.join(raw_dir, f"{dataset_name}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"数据文件不存在: {npz_path}")

    data_dict = np.load(npz_path)
    raw_data = data_dict['data']
    data = raw_data.astype(np.float32)

    print(f"加载数据: {data.shape}")
    print(f"  - 节点数 (N): {data.shape[0]}")
    print(f"  - 时间步 (T): {data.shape[1]}")
    print(f"  - 通道数 (C): {data.shape[2]}")

    # CHI_TAXI 只取第一维特征（上车）
    if dataset_name == 'CHI_TAXI':
        data = data[:, :, 0:1]
        print(f"  - 取第一维特征后: {data.shape}")

    adj_path = os.path.join(raw_dir, f"{dataset_name}_rn_adj.npy")
    if not os.path.exists(adj_path):
        raise FileNotFoundError(f"邻接矩阵不存在: {adj_path}")

    adj = np.load(adj_path)
    print(f"加载邻接矩阵: {adj.shape}")

    return data, adj.astype(np.float32)


def filter_by_date_range(data: np.ndarray,
                         start_date: str = None,
                         end_date: str = None,
                         time_interval: int = 1800):
    """根据日期范围过滤数据"""
    if start_date is None and end_date is None:
        print("未指定日期范围，使用全部数据")
        return data, 0

    from datetime import date

    intervals_per_day = 24 * 60 * 60 // time_interval

    start_idx = 0
    end_idx = data.shape[1]

    if start_date is not None:
        start_year, start_month, start_day = parse_date(start_date)
        start_date_obj = date(start_year, start_month, start_day)
        ref_date = date(2016, 1, 1)
        days_from_start = (start_date_obj - ref_date).days
        start_idx = max(0, days_from_start * intervals_per_day)
        print(f"起始日期: {start_date} (索引: {start_idx})")

    if end_date is not None:
        end_year, end_month, end_day = parse_date(end_date)
        end_date_obj = date(end_year, end_month, end_day)
        ref_date = date(2016, 1, 1)
        days_from_start = (end_date_obj - ref_date).days + 1
        end_idx = min(data.shape[1], days_from_start * intervals_per_day)
        print(f"结束日期: {end_date} (索引: {end_idx})")

    filtered_data = data[:, start_idx:end_idx, :]
    print(f"过滤后数据: {filtered_data.shape}")

    return filtered_data, start_idx


def create_sliding_windows(data: np.ndarray,
                          input_window: int = 24,
                          output_window: int = 24,
                          stride: int = 1,
                          time_interval: int = 1800):
    """
    创建滑动窗口样本

    时间特征:
        X: (num_samples, input_window, N, 3) - 数值 + time_of_day + dow
        y: (num_samples, output_window, N, 1)
    """
    N, T, C = data.shape
    total_window = input_window + output_window

    values = data[:, :, 0:1] if data.shape[-1] > 1 else data

    intervals_per_hour = 3600 // time_interval
    intervals_per_day = intervals_per_hour * 24

    time_indices = np.arange(T)

    # Time of day (连续值，0~1)
    time_of_day = ((time_indices % intervals_per_day) / intervals_per_day)[..., None, None]

    # Day of week (归一化)
    dow = (time_indices // intervals_per_day) % 7
    dow_feat = (dow / 7.0)[..., None, None]

    time_of_day_feat = np.broadcast_to(time_of_day, (T, N, 1)).copy()
    dow_feat = np.broadcast_to(dow_feat, (T, N, 1)).copy()

    time_features = np.concatenate([time_of_day_feat, dow_feat], axis=-1)

    x_samples = []
    y_samples = []

    for t in range(0, T - total_window + 1, stride):
        x_w = values[:, t:t + input_window]
        y_w = values[:, t + input_window:t + total_window]

        x_w = x_w.transpose(1, 0, 2)
        y_w = y_w.transpose(1, 0, 2)

        x_samples.append(x_w)
        y_samples.append(y_w)

    X = np.stack(x_samples, axis=0)
    y = np.stack(y_samples, axis=0)

    # 为每个样本生成完整的 input_window 时间特征
    start_indices = np.arange(0, len(x_samples) * stride, stride)  # 每个样本的起始时间步
    X_time_list = []
    for start_idx in start_indices:
        # 该样本的时间特征: (input_window, N, 2)
        sample_time = time_features[start_idx:start_idx + input_window][None, :, :, :]
        X_time_list.append(sample_time)

    X_time = np.concatenate(X_time_list, axis=0)

    X = np.concatenate([X, X_time], axis=-1)

    print(f"滑动窗口样本:")
    print(f"  - X shape: {X.shape}  (数值 + time_of_day + dow)")
    print(f"  - y shape: {y.shape}")

    return X, y


def save_all_dataset(X, y, output_dir: str):
    """保存全部数据到npz文件"""
    os.makedirs(output_dir, exist_ok=True)

    npz_path = os.path.join(output_dir, "all.npz")
    np.savez_compressed(npz_path, X=X, y=y)
    print(f"保存 all.npz: X {X.shape}, y {y.shape}")


def save_adjacency(adj: np.ndarray, output_dir: str):
    """保存邻接矩阵为pkl文件"""
    adj_path = os.path.join(output_dir, "adj.pkl")
    with open(adj_path, 'wb') as f:
        pickle.dump(adj.astype(np.float32), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"保存 adj.pkl: {adj.shape}")


def process_dataset(dataset_name: str, args):
    """处理单个数据集"""
    if dataset_name not in DATASET_CONFIG:
        print(f"错误: 不支持的数据集 '{dataset_name}'")
        print(f"支持的 datasets: {list(DATASET_CONFIG.keys())}")
        return False

    config = DATASET_CONFIG[dataset_name]
    time_interval = args.time_interval or config['time_interval']

    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(script_dir, config['raw_path'])
    output_dir = os.path.join(script_dir, "data", "processed", dataset_name.lower())

    print("=" * 60)
    print(f"处理数据集: {dataset_name}")
    print(f"日期范围: {args.start_date or config['default_start'] or '全部'} - {args.end_date or config['default_end'] or '全部'}")
    print(f"输入窗口: {args.input_window}, 输出窗口: {args.output_window}")
    print(f"滑动步长: {args.stride}")
    print(f"时间间隔: {time_interval} 秒 ({time_interval // 60} 分钟)")
    print("=" * 60)

    print("\n[1/4] 加载原始数据...")
    data, adj = load_raw_data(raw_dir, dataset_name)

    print("\n[2/4] 按日期范围过滤...")
    data, start_idx = filter_by_date_range(
        data,
        start_date=args.start_date or config['default_start'],
        end_date=args.end_date or config['default_end'],
        time_interval=time_interval
    )

    print("\n[3/4] 创建滑动窗口样本...")
    X, y = create_sliding_windows(
        data,
        input_window=args.input_window,
        output_window=args.output_window,
        stride=args.stride,
        time_interval=time_interval
    )

    print("\n[4/4] 保存数据...")
    save_all_dataset(X, y, output_dir)
    save_adjacency(adj, output_dir)

    print("\n" + "=" * 60)
    print(f"处理完成! 数据保存在: {output_dir}")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description='数据集处理')
    parser.add_argument('dataset_name', nargs='?', type=str, default=None,
                       help=f'数据集名称 ({", ".join(DATASET_CONFIG.keys())}) 或 --all')
    parser.add_argument('--all', action='store_true', help='处理所有数据集')
    parser.add_argument('--start_date', type=str, default=None, help='起始日期 YYYYMMDD')
    parser.add_argument('--end_date', type=str, default=None, help='结束日期 YYYYMMDD')
    parser.add_argument('--input_window', type=int, default=24, help='输入序列长度 (默认: 24)')
    parser.add_argument('--output_window', type=int, default=24, help='预测序列长度 (默认: 24)')
    parser.add_argument('--stride', type=int, default=1, help='滑动窗口步长 (默认: 1)')
    parser.add_argument('--time_interval', type=int, default=None, help='时间间隔秒数')

    args = parser.parse_args()

    # 确定要处理的数据集列表
    if args.all:
        datasets = list(DATASET_CONFIG.keys())
    elif args.dataset_name:
        datasets = [args.dataset_name]
    else:
        print("错误: 请指定数据集名称或使用 --all")
        print(f"用法: python prepare_dataset.py {','.join(DATASET_CONFIG.keys())}")
        print(f"      python prepare_dataset.py --all")
        return

    print(f"\n将处理以下数据集: {datasets}\n")

    success_count = 0
    for ds in datasets:
        if process_dataset(ds, args):
            success_count += 1
        print()

    print(f"\n完成! 成功处理 {success_count}/{len(datasets)} 个数据集。")


if __name__ == "__main__":
    main()
