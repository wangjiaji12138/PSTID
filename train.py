import os
import sys
import gc
import json
import time
import random
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 配置中文字体支持（解决"测试集"等中文显示为方块的问题）
import matplotlib

matplotlib.use('Agg')  # 非交互式后端，避免阻塞（供 viz 模块使用）
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'DejaVu Sans', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示

from model_builder import build_model
from utils.data_utils import load_dataset, PSTIDDataset, StandardScaler
from utils.metrics import compute_all_metrics_torch, mae_torch
from utils.log_record import create_logger

# 可视化（独立模块，可单独运行 viz.py 来加载 checkpoint 可视化）
import viz as _viz


MODEL_LIST = ["HA", "GRU", "STGCN", "AGCRN", "GWNET", "MEGACRN", "STNORM",
               "MTGNN", "STAEFORMER", "STDN", "STID", "PSTID", "STSSL", "STSSDL", "PDFORMER"]

def set_seed(seed: int = 42, deterministic: bool = True):
    """Set random seed for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        
        
def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(description='PSTID Training')

    # ============================================================
    # 1) Basic
    # ============================================================
    parser.add_argument('--seed', type=int, default=88)
    parser.add_argument('--data', type=str, default='cq')
    parser.add_argument('--data_dir', type=str, default='data/processed/')
    parser.add_argument('--pdformer_data_dir', type=str, default='data/processed/pdformer',
                        help='PDFormer 预处理数据目录（来自 preprocess_pdformer.py 输出）')
    parser.add_argument('--model', type=str, default='PSTID')

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
    parser.add_argument('--gpu', type=int, default=0)

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

    # 5c) 原型均匀性损失
    parser.add_argument('--proto_uniformity_weight', type=float, default=0.0)

    # 5d) 原型 dropout
    parser.add_argument('--spatial_idx_dropout', type=float, default=0.1)

    # ============================================================
    # 6) Ablation switches (0=off, 1=on)
    # ============================================================
    parser.add_argument('--use_proto', type=int, default=1)
    parser.add_argument('--use_spatio', type=int, default=1)
    parser.add_argument('--use_temporal', type=int, default=1)
    
    return parser


def build_dataloaders(dataset, batch_size, seed, num_workers=0):
    """Build train/val/test dataloaders."""
    train_dataset = PSTIDDataset(dataset, 'train')
    val_dataset = PSTIDDataset(dataset, 'val')
    test_dataset = PSTIDDataset(dataset, 'test')

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    common_config = {
        'pin_memory': True,
        'num_workers': num_workers,
        'worker_init_fn': seed_worker,
        'generator': g,
    }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, **common_config)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,**common_config)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, **common_config)

    return train_loader, val_loader, test_loader, len(train_dataset), len(val_dataset), len(test_dataset)


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
    time_feats = X[..., -2:]  # Last 2 channels: hour, dow

    if isinstance(X, torch.Tensor):
        mean = torch.as_tensor(scaler.mean, device=X.device, dtype=X.dtype)
        std = torch.as_tensor(scaler.std, device=X.device, dtype=X.dtype)
        assert std > 1e-6, "Standard deviation is too small"
        data_norm = (X[..., :1] - mean) / std
    
    return torch.cat([data_norm, time_feats], dim=-1)


def train_epoch(model, dataloader, optimizer, criterion, device, scaler, 
                clip_grad=None, model_name='PSTID', input_dim=3, scaler_amp=None, epoch=0):
    """Train for one epoch with multi-channel support and optional mixed precision training."""
    model.train()

    epoch_loss = 0.0
    epoch_pred_loss = 0.0
    epoch_stssl_temporal_loss = 0.0
    epoch_stssl_spatial_loss = 0.0
    epoch_uniformity_loss = 0.0
    n_batches = 0

    use_stssl = model_name in ['STSSL', 'STSSDL']

    for batch in dataloader:
        X = batch['X'].to(device)
        y_raw = batch['y'].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            if use_stssl:
                # STSSL模型使用自己的calculate_loss方法
                loss, sep_losses = model.calculate_loss({'X': X, 'y': y_raw})
                epoch_pred_loss += sep_losses.get('pred', 0.0)
                epoch_stssl_temporal_loss += sep_losses.get('temporal', 0.0)
                epoch_stssl_spatial_loss += sep_losses.get('spatial', 0.0)
            else:
                # 标准模型使用标准流程
                X_norm = normalize_input(X, scaler, input_dim)
                output = model({'X': X_norm})  # (B, T, N, C_out)
                
                y_pred_norm = output.mean(dim=-1, keepdim=True)  # (B, T, N, 1)
                y_pred_raw = scaler.inverse_transform(y_pred_norm) # (B, T, N, D_out)
                
                y_target_raw = y_raw[..., 0:1] # (B, T, N, 1)

                loss = criterion(y_pred_raw, y_target_raw)
                epoch_pred_loss += loss.detach().item()

        # 混合精度训练的 backward
        if scaler_amp is not None:
            scaler_amp.scale(loss).backward()
            if clip_grad:
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        epoch_loss += loss.detach().item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
    avg_pred = epoch_pred_loss / n_batches if n_batches > 0 else 0.0
    avg_stssl_temporal = epoch_stssl_temporal_loss / n_batches if n_batches > 0 and use_stssl else 0.0
    avg_stssl_spatial = epoch_stssl_spatial_loss / n_batches if n_batches > 0 and use_stssl else 0.0
    avg_uniformity = epoch_uniformity_loss / n_batches if n_batches > 0 else 0.0

    return {'loss': avg_loss, 'pred_loss': avg_pred,
            'stssl_temporal_loss': avg_stssl_temporal, 'stssl_spatial_loss': avg_stssl_spatial,
            'uniformity_loss': avg_uniformity}


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, scaler, 
             input_dim=3, output_dim=1, model_name='PSTID'):
    """Evaluate model on validation/test set.

    For multi-channel output, predictions are averaged across channels.

    Returns:
        dict: 包含 loss, MAE, RMSE, MAPE, R2, SMAPE 指标
    """
    model.eval()
    all_preds_raw, all_targets_raw = [], []
    
    use_stssl = model_name in ['STSSL', 'STSSDL']

    for batch in dataloader:
        X = batch['X'].to(device)
        y_raw = batch['y'].to(device)  # (B, T, N, 1)

        if use_stssl:
            # STSSL: 使用predict方法获取预测
            output = model.predict({'X': X})  # (B, T, N, output_dim)
            y_pred_raw = output[..., 0:output_dim]  # (B, T, N, output_dim)
        else:
            X_norm = normalize_input(X, scaler, input_dim)
            output = model({'X': X_norm})  # (B, T, N, output_dim)

            y_pred_norm = output[..., 0:output_dim]  # (B, T, N, output_dim)
            y_pred_raw = scaler.inverse_transform(y_pred_norm) # (B, T, N, output_dim)

        all_preds_raw.append(y_pred_raw)
        all_targets_raw.append(y_raw)

    all_preds = torch.cat(all_preds_raw, dim=0)
    all_targets = torch.cat(all_targets_raw, dim=0)

    loss = criterion(all_preds, all_targets).item()

    mae, rmse, mape, r2, smape = compute_all_metrics_torch(all_preds, all_targets, null_value=-1.0)

    return {'loss': loss, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2, 'SMAPE': smape}


def save_checkpoint(model, optimizer, scheduler, val_metrics, args, scaler, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': args.current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_metrics': val_metrics,
        'config': vars(args),
        'scaler_mean': scaler.mean,
        'scaler_std': scaler.std,
    }, path)


def save_results(results_dir, training_history, test_metrics, args):
    """Save training history, test results, and config."""
    history_path = Path(results_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    test_clean = {k: v.item() if hasattr(v, 'item') else v for k, v in test_metrics.items()}
    with open(Path(results_dir) / 'test_results.json', 'w') as f:
        json.dump(test_clean, f, indent=2)

    with open(Path(results_dir) / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)


def train_single_model(args, model, train_loader, val_loader, test_loader,
                       optimizer, scheduler, criterion, device, scaler, logger,
                       model_name, num_nodes):
    """Train a single model with early stopping and optional mixed precision training."""
    best_val_mae = float('inf')
    patience_counter = 0
    training_history = []
    best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')

    # 判断是否为 HA 模型
    is_ha = model_name.upper() == 'HA'
    
    # 创建混合精度训练的 GradScaler
    scaler_amp = torch.amp.GradScaler('cuda', enabled=not is_ha)

    # HA 模型使用过去历史平均，没有可训练参数
    # 直接评估一次，不需要多轮训练
    if is_ha:
        logger.info("HA model: using historical average (last 144 steps) for prediction")
        logger.info("Evaluating on val/test set...")
        # 评估验证集
        val_metrics = evaluate(model, val_loader, criterion, device, scaler,
                               input_dim=args.input_dim, output_dim=args.output_dim,
                               model_name=model_name)
        logger.info(f"Val MAE: {val_metrics['MAE']:.4f} | Val RMSE: {val_metrics['RMSE']:.4f} | Val MAPE: {val_metrics['MAPE']:.2f}%")
        training_history = [{'epoch': 1, 'train_loss': 0.0, 'val_loss': val_metrics['loss'], 'val_mae': val_metrics['MAE']}]
        # 评估测试集
        test_metrics = evaluate(model, test_loader, criterion, device, scaler,
                                input_dim=args.input_dim, output_dim=args.output_dim,
                                model_name=model_name)
        logger.info(f"Test MAE: {test_metrics['MAE']:.4f} | Test RMSE: {test_metrics['RMSE']:.4f} | Test MAPE: {test_metrics['MAPE']:.2f}%")
        return best_model_path, training_history, test_metrics

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, args.clip_grad, model_name,
            input_dim=args.input_dim, scaler_amp=scaler_amp,
            epoch=epoch,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, scaler,
                              input_dim=args.input_dim, output_dim=args.output_dim,
                              model_name=model_name)

        args.current_epoch = epoch
        current_lr = optimizer.param_groups[0]['lr'] if optimizer else 0
        epoch_time = time.time() - epoch_start

        logger.info("-" * 60)
        
        # 构建日志信息
        if model_name in ['STSSL', 'STSSDL']:
            # SSL模型日志
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.0e} | Model: {model_name.upper()}")
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Pred Loss: {train_metrics.get('pred_loss', 0.0):.4f} | "
                f"Temporal: {train_metrics.get('stssl_temporal_loss', 0.0):.4f} | "
                f"Spatial: {train_metrics.get('stssl_spatial_loss', 0.0):.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val MAE: {val_metrics['MAE']:.4f}"
            )
        elif model_name.upper() in ['PSTID', 'STID', 'STDN']:
            # 主模型日志
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.0e} | Model: {model_name.upper()}")
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"UnifLoss: {train_metrics.get('uniformity_loss', 0.0):.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val MAE: {val_metrics['MAE']:.4f}"
            )
        else:
            # 其他模型日志 (GRU, STGCN, AGCRN, etc.)
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.0e} | Model: {model_name.upper()}")
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val MAE: {val_metrics['MAE']:.4f}"
            )

        if model_name.upper() == 'PSTID' and args.use_proto:
            proto_summary = model.get_proto_usage_summary()
            if proto_summary:
                logger.info(f"Proto Info: {proto_summary}")

        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_mae': val_metrics['MAE'],
            'lr': current_lr,
            'epoch_time': epoch_time
        })

        if val_metrics['MAE'] < best_val_mae:
            best_val_mae = val_metrics['MAE']
            patience_counter = 0
            # HA 模型不需要保存 checkpoint
            if model_name.upper() != 'HA':
                save_checkpoint(model, optimizer, scheduler, val_metrics, args, scaler, best_model_path)
            logger.info(f"✅ Best model (Val MAE: {best_val_mae:.4f})")
        else:
            patience_counter += 1

        # HA 模型跳过 scheduler
        if scheduler is not None:
            scheduler.step()

        # HA 模型没有参数，只评估一次后退出
        if is_ha:
            logger.info(f"HA model: no training needed, completing after 1 epoch(s)")
            break

        if patience_counter >= args.early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience_counter} epochs)")
            break

        if device.type == 'cuda' and epoch % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Test evaluation
    logger.info("-" * 50)
    logger.info(f"Model: {model_name.upper()} - Test evaluation...")
    
    # HA 模型跳过 checkpoint 加载
    if model_name.upper() != 'HA':
        checkpoint = torch.load(best_model_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    test_metrics = evaluate(model, test_loader, criterion, device, scaler,
                            input_dim=args.input_dim, output_dim=args.output_dim, model_name=model_name)

    logger.info(f"Test Results ({model_name.upper()}):")
    logger.info(f"  Loss:  {test_metrics['loss']:.4f}")
    logger.info(f"  MAE:   {test_metrics['MAE']:.4f}")
    logger.info(f"  RMSE:  {test_metrics['RMSE']:.4f}")
    logger.info(f"  MAPE:  {test_metrics['MAPE']:.4f}%")
    logger.info(f"  R2:    {test_metrics['R2']:.4f}")
    logger.info(f"  SMAPE:  {test_metrics['SMAPE']:.4f}%")

    return best_model_path, training_history, test_metrics


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Convert int flags to bool
    args.use_proto = bool(args.use_proto)
    args.use_spatio = bool(args.use_spatio)
    args.use_temporal = bool(args.use_temporal)

    # Validate model (case-sensitive)
    if args.model.upper() not in MODEL_LIST:
        raise ValueError(f"Unknown model '{args.model}'. Available: {MODEL_LIST}")

    set_seed(args.seed)

    # Setup directories
    script_dir = Path(__file__).parent.resolve()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results_dir = script_dir / 'results' / args.data / f"{args.model}_{timestamp}"

    model_name = args.model

    log_dir = results_dir / 'log'
    checkpoint_dir = results_dir / 'checkpoint'

    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    args.results_dir = str(results_dir)
    args.log_dir = str(log_dir)
    args.checkpoint_dir = str(checkpoint_dir)

    # Setup logger for visualization functions that use logging.getLogger('train')
    train_logger = logging.getLogger('train')
    train_logger.setLevel(logging.INFO)
    # Only add handler if not already configured (avoid duplicate handlers)
    if not train_logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        train_logger.addHandler(console_handler)

    logger = create_logger(str(log_dir), 'TRAIN')
    logger.info(f"Results: {results_dir}")
    logger.info(f"Model: {model_name}")
    logger.log_model_config(args)

    train_start = datetime.now()
    logger.info(f"Training started at: {train_start.strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Device: {device}")

    # Check if data exists (support both split .npz files and unified all.npz)
    import os
    data_path = os.path.join(args.data_dir, args.data)
    # Try unified all.npz first, then split files
    all_file = os.path.join(data_path, 'all.npz')
    train_file = os.path.join(data_path, 'train.npz')
    if not os.path.exists(all_file) and not os.path.exists(train_file):
        logger.error(f"Data not found: {data_path}")
        logger.error(f"Expected 'all.npz' or 'train.npz', 'val.npz', 'test.npz' files")
        return None, None, None
    logger.info(f"Data path: {data_path}")
    # Load data
    logger.info(f"Loading data: {args.data}")
    dataset = load_dataset(args.data_dir, args.data, seed=args.seed)

    train_loader, val_loader, test_loader, n_train, n_val, n_test = build_dataloaders(
        dataset, args.batch_size, args.seed
    )
    logger.info(f"Dataset: train={n_train}, val={n_val}, test={n_test}")

    # Get model info
    num_nodes = dataset['train']['X'].shape[2]
    adj_mx = torch.from_numpy(dataset['adj_mx']).float().to(device)

    # ========== Load PDFormer preprocessed data if needed ==========
    pdformer_data = None
    if model_name.upper() == 'PDFORMER':
        logger.info(f"Loading PDFormer preprocessed data from {args.pdformer_data_dir}/{args.data}/")
        pdformer_data_path = os.path.join(args.pdformer_data_dir, args.data)
        if not os.path.exists(pdformer_data_path):
            logger.error(f"PDFormer data not found at {pdformer_data_path}")
            logger.error("Please run preprocessing first: python preprocess_pdformer.py --dataset " + args.data)
            return None, None, None
        
        pdformer_data = {}
        try:
            # Load Laplacian PE
            lap_mx_path = os.path.join(pdformer_data_path, 'laplacian_pe.npy')
            if os.path.exists(lap_mx_path):
                lap_mx = np.load(lap_mx_path)
                # Defensive: ensure shape is (N, lape_dim)
                lape_dim = getattr(args, 'lape_dim', 8)
                if lap_mx.ndim != 2:
                    raise ValueError(f"Unexpected lap_mx ndim={lap_mx.ndim}, shape={lap_mx.shape}")
                if lap_mx.shape[1] != lape_dim:
                    if lap_mx.shape[1] > lape_dim:
                        lap_mx = lap_mx[:, :lape_dim]
                    else:
                        lap_mx = np.pad(lap_mx, ((0, 0), (0, lape_dim - lap_mx.shape[1])))
                pdformer_data['lap_mx'] = lap_mx
                logger.info(f"  - lap_mx: {pdformer_data['lap_mx'].shape}")
            
            # Load pattern keys
            pattern_keys_path = os.path.join(pdformer_data_path, 'pattern_keys.npy')
            if os.path.exists(pattern_keys_path):
                pattern_keys = np.load(pattern_keys_path)
                # Normalize pattern_keys to shape (n_cluster, s_attn_size, output_dim)
                # Older preprocessing may produce (n_cluster, s_attn_size) or (n_cluster, s_attn_size, 1, 1)
                # The model indexes pattern_keys[..., i] for i in range(output_dim)
                s_attn_size = getattr(args, 's_attn_size', 3)
                n_cluster = pattern_keys.shape[0]
                if pattern_keys.ndim == 2:
                    # (n_cluster, s_attn_size) -> (n_cluster, s_attn_size, 1)
                    pattern_keys = pattern_keys.reshape(n_cluster, s_attn_size, 1)
                elif pattern_keys.ndim == 4:
                    # (n_cluster, s_attn_size, 1, 1) -> (n_cluster, s_attn_size, 1)
                    pattern_keys = pattern_keys.reshape(n_cluster, s_attn_size, 1)
                elif pattern_keys.ndim != 3:
                    raise ValueError(
                        f"Unexpected pattern_keys ndim={pattern_keys.ndim}, shape={pattern_keys.shape}"
                    )
                pdformer_data['pattern_keys'] = pattern_keys
                logger.info(f"  - pattern_keys: {pdformer_data['pattern_keys'].shape}")
            
            # Load geographic mask
            geo_mask_path = os.path.join(pdformer_data_path, 'geo_mask.npy')
            if os.path.exists(geo_mask_path):
                pdformer_data['geo_mask'] = np.load(geo_mask_path)
                logger.info(f"  - geo_mask: {pdformer_data['geo_mask'].shape}")
            
            # Load semantic mask
            sem_mask_path = os.path.join(pdformer_data_path, 'sem_mask.npy')
            if os.path.exists(sem_mask_path):
                pdformer_data['sem_mask'] = np.load(sem_mask_path)
                logger.info(f"  - sem_mask: {pdformer_data['sem_mask'].shape}")
            
            # Load metadata
            metadata_path = os.path.join(pdformer_data_path, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"  - metadata: {metadata}")
            
            # Pass to args for model building
            args.pdformer_data = pdformer_data
            logger.info("PDFormer data loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading PDFormer data: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    model_name = args.model

    model = build_model(model_name, args, num_nodes, adj_mx, device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: total={total_params:,}, trainable={trainable_params:,}")
    logger.log_model_architecture(model)

    # Build scaler
    mean, std = compute_scaler_stats(dataset)
    scaler = StandardScaler(mean=mean, std=std)
    logger.info(f"Scaler: mean={mean:.6f}, std={std:.6f}")

    # Optimizer & scheduler (HA 模型跳过优化器)
    if model_name.upper() == 'HA':
        # HA 是非参数模型，直接使用验证/测试评估
        logger.info("HA model: skipping optimizer, using direct evaluation")
        # 创建一个虚拟的 optimizer 用于 train_single_model 接口兼容
        optimizer = None
        scheduler = None
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)
        # 确保 checkpoint 目录存在
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Training
    best_model_path, training_history, test_metrics = train_single_model(
        args, model, train_loader, val_loader, test_loader,
        optimizer, scheduler, mae_torch, device, scaler, logger,
        model_name, num_nodes
    )

    # Save results
    save_results(results_dir, training_history, test_metrics, args)

    # Generate visualizations (delegated to viz module)
    # 训练完成后的所有可视化都委托给 viz.py：
    #   - 也可以独立运行 `python viz.py --checkpoint <path>` 来重新可视化
    _viz.run_all_visualizations(model, test_loader, scaler, device, args, results_dir)

    train_end = datetime.now()
    duration = train_end - train_start
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info(f"\nTraining completed. Results saved to: {results_dir}")
    logger.info(f"Finished at: {train_end.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total time: {hours}h {minutes}m {seconds}s")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    main()
