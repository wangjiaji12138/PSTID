"""training.checkpoint: extracted from the original training entry point."""

import json
from pathlib import Path
import torch


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
