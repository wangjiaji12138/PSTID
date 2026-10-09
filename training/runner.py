"""Assemble and execute one forecasting experiment."""
import os
import gc
from datetime import datetime
from pathlib import Path
import torch
from models.registry import MODEL_LIST, build_model
from data_pipeline.dataset import load_dataset, StandardScaler
from data_pipeline.loaders import build_dataloaders
from data_pipeline.normalization import compute_scaler_stats
from utils.metrics import mae_torch
from utils.log_record import create_logger
from .config import create_parser
from .reproducibility import set_seed
from .engine import train_single_model
from .checkpoint import save_results


def main():
    parser = create_parser()
    args = parser.parse_args()

    args.use_adj = bool(args.use_adj)
    # Convert int flags to bool
    args.use_proto = bool(args.use_proto)
    args.use_spatio = bool(args.use_spatio)
    args.use_temporal = bool(args.use_temporal)

    # Validate model against the shared registry
    if args.model.upper() not in MODEL_LIST:
        raise ValueError(f"Unknown model '{args.model}'. Available: {MODEL_LIST}")

    set_seed(args.seed)

    # Setup directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results_dir = Path(args.results_dir) / args.data / f"{args.model}_{timestamp}"

    model_name = args.model

    log_dir = results_dir / 'log'
    checkpoint_dir = results_dir / 'checkpoint'

    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    args.results_dir = str(results_dir)
    args.log_dir = str(log_dir)
    args.checkpoint_dir = str(checkpoint_dir)

    logger = create_logger(str(log_dir), 'TRAIN')
    logger.info(f"Results: {results_dir}")
    logger.info(f"Model: {model_name}")
    logger.log_model_config(args)

    train_start = datetime.now()
    logger.info(f"Training started at: {train_start.strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Device: {device}")

    # load_dataset 需要预处理生成的 all.npz 和 adj.pkl。
    data_path = os.path.join(args.data_dir, args.data)
    all_file = os.path.join(data_path, 'all.npz')
    adj_file = os.path.join(data_path, 'adj.pkl')
    missing_files = [path for path in (all_file, adj_file) if not os.path.isfile(path)]
    if missing_files:
        logger.error(f"Missing dataset files: {', '.join(missing_files)}")
        if args.data in ('nyc_taxi', 'chi_taxi', 'nyc_bike'):
            logger.error(f"请先完成预处理: python prepare_dataset.py {args.data.upper()}")
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
        model_name, num_nodes, adj_mx=adj_mx,
    )

    # Save results
    save_results(results_dir, training_history, test_metrics, args)

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
