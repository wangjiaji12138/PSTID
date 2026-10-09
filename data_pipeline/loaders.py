"""data_pipeline.loaders: extracted from the original training entry point."""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import PSTIDDataset


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
