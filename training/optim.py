"""Optimizer/scheduler selection from resolved model registry settings."""
import torch


def build_optimizer(model, args):
    kwargs = dict(lr=args.learning_rate, weight_decay=args.weight_decay,
                  eps=getattr(args, 'optimizer_eps', 1e-8))
    name = getattr(args, 'optimizer', 'adam')
    if name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **kwargs)
    elif name == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(), alpha=getattr(args, 'rmsprop_alpha', .9), **kwargs)
    else:
        raise ValueError(f'Unsupported optimizer: {name}')
    schedule = getattr(args, 'lr_scheduler', 'multistep')
    if schedule == 'none':
        # Keep a serializable scheduler for existing checkpoint code.
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[])
    elif schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)
    elif schedule == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    else:
        raise ValueError(f'Unsupported lr_scheduler: {schedule}')
    return optimizer, scheduler
