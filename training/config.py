"""training.config: extracted from the original training entry point."""

import argparse
from models.registry import MODEL_LIST
from utils.paths import DATA_ROOT, RESULTS_ROOT


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(description='PROPHET Training')

    # ============================================================
    # 1) Basic
    # ============================================================
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--data', type=str.lower, default='cq',
                        help='数据集名称，如 cq、hz、sh、jl、yt、chi_taxi、nyc_taxi')
    parser.add_argument('--data_dir', type=str, default=str(DATA_ROOT / 'processed'))
    parser.add_argument('--model', type=str.upper, choices=MODEL_LIST, default='PROPHET')

    # ============================================================
    # 2) Training
    # ============================================================
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--early_stop_patience', type=int, default=15)
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[20, 40, 50, 60, 70])
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--clip_grad', type=float, default=5.0)

    # ============================================================
    # 3) Device
    # ============================================================
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu', type=int, default=2)

    # ============================================================
    # 4) Model architecture
    # ============================================================
    parser.add_argument('--input_window', type=int, default=24)
    parser.add_argument('--output_window', type=int, default=24)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--input_embedding_dim', type=int, default=32)
    parser.add_argument('--node_emb_dim', type=int, default=16)
    parser.add_argument('--proto_emb_dim', type=int, default=64)
    parser.add_argument('--adj_mx_emb_dim', type=int, default=16)
    parser.add_argument('--use_adj', type=int, default=1)
    parser.add_argument('--tid', type=int, default=16)
    parser.add_argument('--diw', type=int, default=16)
    parser.add_argument('--time_intervals', type=int, default=1800)
    parser.add_argument('--tcn_kernel_sizes', type=int, nargs='+', default=[3])
    parser.add_argument('--dropout', type=float, default=0.1)

    # ============================================================
    # 5) Prototype codebooks
    # ============================================================
    # 5a) Codebook sizes
    parser.add_argument('--num_spatial_prototypes', type=int, default=16)
    parser.add_argument('--num_temporal_prototypes', type=int, default=8)

    # 5b) Prototype temperature
    parser.add_argument('--proto_temperature', type=float, default=0.5)

    # ============================================================
    # 6) Ablation switches (0=off, 1=on)
    # ============================================================
    parser.add_argument('--use_proto', type=int, default=1)
    parser.add_argument('--use_spatio', type=int, default=1)
    parser.add_argument('--use_temporal', type=int, default=1)

    parser.add_argument('--results_dir', type=str, default=str(RESULTS_ROOT), help='实验结果根目录')
    return parser
