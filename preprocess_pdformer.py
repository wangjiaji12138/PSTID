#!/usr/bin/env python3
"""
PDFormer Data Preprocessing Script

This script processes raw traffic data for PDFormer, generating:
1. DTW (Dynamic Time Warping) matrix for semantic attention
2. Pattern keys for spatial attention
3. Laplacian positional encoding
4. Geographic and semantic masks

Output is saved to data/processed/pdformer/{dataset_name}/
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import eigsh
from fastdtw import fastdtw
from tslearn.clustering import KShape
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compute_laplacian_pe(adj_mx, lape_dim=8, max_freq=10):
    """
    Compute Laplacian Positional Encoding from adjacency matrix.
    
    Args:
        adj_mx: Adjacency matrix (N x N)
        lape_dim: Dimension of positional encoding
        max_freq: Maximum frequency for positional encoding
    
    Returns:
        lap_pe: Laplacian positional encoding (N, lape_dim)
    """
    n = adj_mx.shape[0]
    
    # Normalize adjacency matrix
    d = np.sum(adj_mx, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    
    # Normalized Laplacian: I - D^{-1/2} A D^{-1/2}
    laplacian = np.eye(n) - d_mat_inv_sqrt @ adj_mx @ d_mat_inv_sqrt
    
    # Compute eigenvalues and eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    except np.linalg.LinAlgError:
        # Fallback: use identity for very small matrices
        return np.eye(n, lape_dim)
    
    # Sort by eigenvalues (ascending)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top k eigenvectors (excluding the first constant one)
    k = min(lape_dim, n - 1)
    eigenvectors = eigenvectors[:, 1:k+1]
    eigenvalues = eigenvalues[1:k+1]

    # Avoid division by zero
    eigenvalues = np.maximum(eigenvalues, 0)
    eigenvalues = np.minimum(eigenvalues, max_freq)

    # Compute positional encoding: [sin(freq * v), cos(freq * v), ...]
    # Note: k * 2 columns are produced (sin + cos per eigenvector). Truncate to lape_dim.
    lap_pe = np.zeros((n, k * 2))
    for i, ev in enumerate(eigenvalues):
        freq = np.log((ev + 1) / (max_freq + 1))
        lap_pe[:, 2*i] = np.sin(freq * eigenvectors[:, i])
        lap_pe[:, 2*i+1] = np.cos(freq * eigenvectors[:, i])

    # Ensure final shape is (n, lape_dim)
    if lap_pe.shape[1] > lape_dim:
        lap_pe = lap_pe[:, :lape_dim]
    elif lap_pe.shape[1] < lape_dim:
        lap_pe = np.pad(lap_pe, ((0, 0), (0, lape_dim - lap_pe.shape[1])))

    return lap_pe


def compute_dtw_matrix(X_train, num_nodes, radius=6):
    """
    Compute DTW distance matrix for semantic attention.
    
    Args:
        X_train: Training data (S, T, N, C) - S samples, T time steps, N nodes
        num_nodes: Number of nodes
        radius: DTW radius for speed optimization
    
    Returns:
        dtw_distance: (N, N) DTW distance matrix
    """
    print("Computing DTW matrix...")
    
    # X_train shape: (S, T, N, C) where T is the number of time steps per sample
    S, T, N, C = X_train.shape
    
    # For DTW, we need to compare time series between nodes
    # We'll use the first feature (traffic flow) for comparison
    # Treat each sample as a sequence of T time steps
    
    dtw_distance = np.zeros((num_nodes, num_nodes))
    
    # Extract traffic data: (S*T, N) - flatten samples and time steps
    traffic_data = X_train[:, :, :, 0].reshape(-1, num_nodes)  # (S*T, N)
    
    # Compute daily averages for efficiency
    # If we have T time steps per sample, we can aggregate them
    # We'll compute DTW on node-wise time series
    
    # Get time series for each node: (N, S*T)
    node_time_series = traffic_data.T  # (N, S*T)
    
    # For efficiency, we'll sample every few time steps
    sample_step = max(1, len(node_time_series[0]) // (24 * 14))  # Use ~14 days of data
    sampled_series = node_time_series[:, ::sample_step]
    
    print(f"  DTW input shape: {sampled_series.shape}")
    
    for i in tqdm(range(num_nodes)):
        for j in range(i, num_nodes):
            # Compute DTW distance between node i and node j time series
            dist, _ = fastdtw(
                sampled_series[i].reshape(-1, 1),
                sampled_series[j].reshape(-1, 1),
                radius=radius
            )
            dtw_distance[i, j] = dist
            dtw_distance[j, i] = dist
    
    return dtw_distance


def compute_pattern_keys(x_train, s_attn_size=3, n_cluster=16, cluster_max_iter=5, cand_key_days=14):
    """
    Compute pattern keys for spatial attention using KShape clustering.
    
    Args:
        x_train: Training data (S, T, N, C)
        s_attn_size: Spatial attention window size
        n_cluster: Number of clusters
        cluster_max_iter: Maximum iterations for clustering
        cand_key_days: Number of days to use for candidate keys
    
    Returns:
        pattern_keys: (n_cluster, s_attn_size, output_dim) where output_dim=1
    """
    print("Computing pattern keys using KShape clustering...")

    # Extract spatial patterns: (S, s_attn_size, N, 1) -> (N*s, s_attn_size)
    # We use the first cand_key_days * 4 intervals (assuming 15-min intervals)
    cand_key_samples = cand_key_days * 24 * 4
    x_cand = x_train[:cand_key_samples]

    # Extract patterns: for each sample, take first s_attn_size time steps
    # Shape: (S, s_attn_size, N, 1) -> (S*N, s_attn_size)
    n_samples = x_cand.shape[0]
    num_nodes = x_cand.shape[2]

    # Get patterns for all nodes and samples
    patterns = x_cand[:, :s_attn_size, :, 0]  # (S, s_attn_size, N)
    patterns = patterns.transpose(0, 2, 1).reshape(-1, s_attn_size)  # (S*N, s_attn_size)

    # Remove invalid patterns (all zeros or with NaN)
    valid_mask = ~np.any(np.isnan(patterns), axis=1) & ~np.all(np.abs(patterns) < 1e-6, axis=1)
    patterns = patterns[valid_mask]

    if len(patterns) < n_cluster:
        print(f"Warning: Only {len(patterns)} valid patterns, duplicating to reach {n_cluster}")
        while len(patterns) < n_cluster:
            patterns = np.concatenate([patterns, patterns[:min(n_cluster - len(patterns), len(patterns))]], axis=0)

    # KShape clustering
    try:
        km = KShape(n_clusters=n_cluster, max_iter=cluster_max_iter, random_state=42)
        km.fit(patterns[:n_cluster * 10])  # Use subset for efficiency
        pattern_keys = km.cluster_centers_  # (n_cluster, s_attn_size)
    except Exception as e:
        print(f"KShape failed: {e}, using random initialization")
        # Fallback: random initialization
        pattern_keys = patterns[:n_cluster]

    # Reshape to (n_cluster, s_attn_size, output_dim=1) to match model expectation
    # model.forward indexes pattern_keys[..., i] for i in range(output_dim)
    pattern_keys = np.asarray(pattern_keys).reshape(n_cluster, s_attn_size, 1)

    return pattern_keys


def compute_spatial_masks(adj_mx, sh_mx, dtw_matrix, far_mask_delta=5, dtw_delta=5, num_nodes=None):
    """
    Compute geographic and semantic masks.
    
    Args:
        adj_mx: Adjacency matrix (distance-based)
        sh_mx: Spatial mask matrix (hop-based)
        dtw_matrix: DTW distance matrix
        far_mask_delta: Threshold for geographic mask
        dtw_delta: Number of top similar nodes for semantic mask
        num_nodes: Number of nodes
    
    Returns:
        geo_mask: Boolean mask for geographic attention
        sem_mask: Boolean mask for semantic attention
        sd_mx: Shortest distance matrix
    """
    if num_nodes is None:
        num_nodes = adj_mx.shape[0]
    
    # Hop-based spatial mask
    sh_mx = sh_mx.T.copy()
    geo_mask = np.zeros((num_nodes, num_nodes))
    geo_mask[sh_mx >= far_mask_delta] = 1
    geo_mask = geo_mask.astype(bool)
    
    # Semantic mask based on DTW
    sem_mask = np.ones((num_nodes, num_nodes))
    dtw_sorted_idx = np.argsort(dtw_matrix, axis=1)
    for i in range(num_nodes):
        # Keep only top dtw_delta most similar nodes (smallest DTW distance)
        sem_mask[i, dtw_sorted_idx[i, :dtw_delta]] = 0
    sem_mask = sem_mask.astype(bool)
    
    # Compute shortest distance matrix (if needed)
    sd_mx = adj_mx.copy()
    sd_mx[sd_mx == 0] = np.inf
    np.fill_diagonal(sd_mx, 0)
    
    # Floyd-Warshall for all-pairs shortest path
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if sd_mx[i, j] > sd_mx[i, k] + sd_mx[k, j]:
                    sd_mx[i, j] = sd_mx[i, k] + sd_mx[k, j]
    
    return geo_mask, sem_mask, sd_mx


def preprocess_dataset(dataset_name, data_dir='data/processed', output_dir='data/processed/pdformer',
                       s_attn_size=3, n_cluster=16, cluster_max_iter=5, cand_key_days=14,
                       far_mask_delta=5, dtw_delta=5, lape_dim=8):
    """
    Preprocess a single dataset for PDFormer.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'cq', 'hz')
        data_dir: Input data directory
        output_dir: Output directory
        s_attn_size: Spatial attention window size
        n_cluster: Number of clusters for pattern keys
        cluster_max_iter: Maximum iterations for clustering
        cand_key_days: Days for candidate keys
        far_mask_delta: Geographic mask threshold
        dtw_delta: Semantic mask threshold
        lape_dim: Laplacian PE dimension
    """
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Paths
    input_path = Path(data_dir) / dataset_name
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {input_path}...")
    
    # Load all.npz
    npz_file = input_path / 'all.npz'
    if not npz_file.exists():
        raise FileNotFoundError(f"Data file not found: {npz_file}")
    
    all_data = np.load(npz_file)
    X = all_data['X']  # (S, T, N, 3)
    y = all_data['y']  # (S, T, N, 1)
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Load adjacency matrix
    adj_file = input_path / 'adj.pkl'
    if not adj_file.exists():
        raise FileNotFoundError(f"Adjacency file not found: {adj_file}")
    
    with open(adj_file, 'rb') as f:
        adj_mx = pickle.load(f)
    
    print(f"Adjacency matrix shape: {adj_mx.shape}")
    
    # Get dataset properties
    num_nodes = X.shape[2]
    time_intervals = 1800  # 30 min intervals
    points_per_hour = 3600 // time_intervals
    points_per_day = 24 * points_per_hour
    
    # =========================================================================
    # 1. Compute Laplacian Positional Encoding
    # =========================================================================
    print("\n1. Computing Laplacian Positional Encoding...")
    lap_pe = compute_laplacian_pe(adj_mx, lape_dim=lape_dim)
    print(f"   Laplacian PE shape: {lap_pe.shape}")
    
    # =========================================================================
    # 2. Compute DTW matrix
    # =========================================================================
    print("\n2. Computing DTW matrix...")
    dtw_cache_file = output_path / 'dtw_matrix.npy'
    if dtw_cache_file.exists():
        print("   Loading cached DTW matrix...")
        dtw_matrix = np.load(dtw_cache_file)
    else:
        train_size = int(len(X) * 0.7)
        X_train = X[:train_size]
        dtw_matrix = compute_dtw_matrix(X_train, num_nodes, radius=6)
        np.save(dtw_cache_file, dtw_matrix)
        print(f"   DTW matrix saved to {dtw_cache_file}")
    
    print(f"   DTW matrix shape: {dtw_matrix.shape}")
    
    # =========================================================================
    # 3. Compute pattern keys
    # =========================================================================
    print("\n3. Computing pattern keys...")
    pattern_keys_cache_file = output_path / 'pattern_keys.npy'
    if pattern_keys_cache_file.exists():
        print("   Loading cached pattern keys...")
        pattern_keys = np.load(pattern_keys_cache_file)
    else:
        train_size = int(len(X) * 0.7)
        X_train = X[:train_size]
        pattern_keys = compute_pattern_keys(
            X_train, s_attn_size=s_attn_size, n_cluster=n_cluster,
            cluster_max_iter=cluster_max_iter, cand_key_days=cand_key_days
        )
        np.save(pattern_keys_cache_file, pattern_keys)
        print(f"   Pattern keys saved to {pattern_keys_cache_file}")
    
    print(f"   Pattern keys shape: {pattern_keys.shape}")
    
    # =========================================================================
    # 4. Compute spatial masks
    # =========================================================================
    print("\n4. Computing spatial masks...")
    
    # Hop-based adjacency (1 for neighbors, 511 for non-neighbors)
    sh_mx = adj_mx.copy()
    sh_mx[sh_mx > 0] = 1
    sh_mx[sh_mx == 0] = 511
    np.fill_diagonal(sh_mx, 0)
    
    # Floyd-Warshall to compute all-pairs shortest path
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                sh_mx[i, j] = min(sh_mx[i, j], sh_mx[i, k] + sh_mx[k, j], 511)
    
    geo_mask, sem_mask, sd_mx = compute_spatial_masks(
        adj_mx, sh_mx, dtw_matrix,
        far_mask_delta=far_mask_delta, dtw_delta=dtw_delta, num_nodes=num_nodes
    )
    
    print(f"   Geo mask: {geo_mask.sum()} / {num_nodes*num_nodes} connections masked")
    print(f"   Sem mask: {sem_mask.sum()} / {num_nodes*num_nodes} connections masked")
    
    # =========================================================================
    # 5. Save all processed data
    # =========================================================================
    print("\n5. Saving processed data...")
    
    # Save Laplacian PE
    np.save(output_path / 'laplacian_pe.npy', lap_pe)
    
    # Save masks
    np.save(output_path / 'geo_mask.npy', geo_mask)
    np.save(output_path / 'sem_mask.npy', sem_mask)
    np.save(output_path / 'sd_mx.npy', sd_mx)
    
    # Save metadata
    metadata = {
        'dataset': dataset_name,
        'num_nodes': num_nodes,
        'input_dim': X.shape[-1],
        'output_dim': y.shape[-1],
        's_attn_size': s_attn_size,
        'n_cluster': n_cluster,
        'cluster_max_iter': cluster_max_iter,
        'cand_key_days': cand_key_days,
        'far_mask_delta': far_mask_delta,
        'dtw_delta': dtw_delta,
        'lape_dim': lape_dim,
        'points_per_hour': points_per_hour,
        'points_per_day': points_per_day,
        'adj_shape': list(adj_mx.shape),
        'dtw_matrix_shape': list(dtw_matrix.shape),
        'pattern_keys_shape': list(pattern_keys.shape),
        'lap_pe_shape': list(lap_pe.shape),
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Copy original data files
    import shutil
    shutil.copy2(input_path / 'all.npz', output_path / 'all.npz')
    shutil.copy2(input_path / 'adj.pkl', output_path / 'adj.pkl')
    
    print(f"\nDataset '{dataset_name}' processed successfully!")
    print(f"Output directory: {output_path}")
    print(f"Files created:")
    for f in output_path.iterdir():
        print(f"   - {f.name}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='PDFormer Data Preprocessing')
    
    parser.add_argument('--dataset', type=str, default='cq',
                        help='Dataset name (cq, hz, sh, jl, yt, chi_taxi, nyc_taxi)')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Input data directory')
    parser.add_argument('--output_dir', type=str, default='data/processed/pdformer',
                        help='Output data directory')
    parser.add_argument('--s_attn_size', type=int, default=3,
                        help='Spatial attention window size')
    parser.add_argument('--n_cluster', type=int, default=16,
                        help='Number of clusters for pattern keys')
    parser.add_argument('--cluster_max_iter', type=int, default=5,
                        help='Maximum iterations for clustering')
    parser.add_argument('--cand_key_days', type=int, default=7,
                        help='Days for candidate keys')
    parser.add_argument('--far_mask_delta', type=int, default=5,
                        help='Geographic mask threshold (hop distance)')
    parser.add_argument('--dtw_delta', type=int, default=5,
                        help='Semantic mask threshold (top-k similar nodes)')
    parser.add_argument('--lape_dim', type=int, default=8,
                        help='Laplacian PE dimension')
    parser.add_argument('--all', action='store_true',
                        help='Process all datasets')
    
    args = parser.parse_args()
    
    if args.all:
        datasets = ['cq', 'hz', 'sh', 'jl', 'yt', 'chi_taxi', 'nyc_taxi']
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        try:
            preprocess_dataset(
                dataset_name=dataset,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                s_attn_size=args.s_attn_size,
                n_cluster=args.n_cluster,
                cluster_max_iter=args.cluster_max_iter,
                cand_key_days=args.cand_key_days,
                far_mask_delta=args.far_mask_delta,
                dtw_delta=args.dtw_delta,
                lape_dim=args.lape_dim
            )
        except FileNotFoundError as e:
            print(f"Error processing {dataset}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("All datasets processed!")
    print("="*60)


if __name__ == '__main__':
    main()
