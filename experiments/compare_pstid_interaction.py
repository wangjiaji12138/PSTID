"""Matched-seed PSTID interaction ablation, using validation data only.

python -m experiments.compare_pstid_interaction --gpu 1 --epochs 20
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from data_pipeline.dataset import load_dataset, StandardScaler
from data_pipeline.loaders import build_dataloaders
from data_pipeline.normalization import compute_scaler_stats
from models.PSTID.model import PSTID
from training.config import create_parser
from training.engine import train_epoch, evaluate
from training.reproducibility import set_seed
from utils.metrics import mae_torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mechanism', choices=['logits', 'change'], default='logits')
    parser.add_argument('--proto_change_context', choices=['raw', 'encoded'], default='encoded')
    parser.add_argument('--data', default='cq')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=88)
    parser.add_argument('--output', default='results/pstid_interaction_comparison.json')
    options = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(f'cuda:{options.gpu}')
    torch.cuda.set_device(device)
    args = create_parser().parse_args(['--model', 'PSTID', '--data', options.data,
                                      '--seed', str(options.seed)])
    args.proto_change_context = options.proto_change_context
    dataset = load_dataset(args.data_dir, args.data, seed=args.seed)
    args.pstid_time_slots = np.unique(np.rint(dataset['train']['X'][:, :, 0, 1]
                                            * (86400 // args.time_intervals)).astype(int)).tolist()
    scaler = StandardScaler(*compute_scaler_stats(dataset))
    adj = torch.as_tensor(dataset['adj_mx'], dtype=torch.float32, device=device)
    report = {'dataset': args.data, 'seed': args.seed, 'epochs': options.epochs,
              'mechanism': options.mechanism, 'config': vars(args).copy(), 'runs': {}}
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for enabled in [False, True]:
        set_seed(args.seed)
        args.use_time_interaction = enabled and options.mechanism == 'logits'
        args.use_proto_change = enabled and options.mechanism == 'change'
        model = PSTID.from_args(args, adj.shape[0], adj, device).to(device)
        train_loader, val_loader, _, *_ = build_dataloaders(dataset, args.batch_size, args.seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_milestones, gamma=args.lr_gamma)
        amp = torch.amp.GradScaler('cuda')
        history = []
        for epoch in range(options.epochs):
            start = time.perf_counter()
            train = train_epoch(model, train_loader, optimizer, mae_torch, device, scaler,
                                clip_grad=args.clip_grad, model_name='PSTID', scaler_amp=amp, epoch=epoch)
            val = evaluate(model, val_loader, mae_torch, device, scaler, model_name='PSTID')
            scheduler.step()
            entry = {'epoch': epoch + 1, 'train_loss': float(train['loss']),
                     'val_mae': float(val['MAE']), 'seconds': time.perf_counter() - start}
            history.append(entry)
            print(json.dumps({'interaction': enabled, **entry}), flush=True)
        report['runs'][str(enabled)] = history
        output.write_text(json.dumps(report, indent=2))
        del model, optimizer, scheduler
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
