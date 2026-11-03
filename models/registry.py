"""Model factories backed by independently sourced upstream configuration."""

import torch

from types import SimpleNamespace
from copy import deepcopy

from .presets import MODEL_CONFIG_REGISTRY, get_model_config, model_defaults


# Fixed baseline settings. Only PSTID and PROPHET accept hyperparameter overrides.
# Upstream evidence remains in official_configs.json; runtime baseline settings
# are deliberately frozen here and do not follow train.py or JSON edits.
BASELINE_CONFIGS = {'AGCRN': {'model': {'input_embedding_dim': 64, 'num_layers': 2, 'embed_dim': 10, 'cheb_k': 2},
           'training': {'learning_rate': 0.003,
                        'weight_decay': 0,
                        'early_stop_patience': 15,
                        'clip_grad': 0,
                        'lr_milestones': [],
                        'lr_gamma': 0.1,
                        'optimizer': 'adam',
                        'optimizer_eps': 1e-08,
                        'lr_scheduler': 'none',
                        'train_loss': 'mae'}},
 'GWNET': {'model': {'input_embedding_dim': 32,
                     'num_layers': 2,
                     'blocks': 4,
                     'dropout': 0.3,
                     'gcn_bool': True,
                     'addaptadj': True,
                     'randomadj': True,
                     'aptonly': False,
                     'adjtype': 'doubletransition',
                     'kernel_size': 2},
           'training': {'learning_rate': 0.001,
                        'weight_decay': 0.0001,
                        'early_stop_patience': 0,
                        'clip_grad': 5,
                        'lr_milestones': [],
                        'lr_gamma': 0.1,
                        'optimizer': 'adam',
                        'optimizer_eps': 1e-08,
                        'lr_scheduler': 'none',
                        'train_loss': 'mae'}},
 'MTGNN': {'model': {'conv_channels': 32,
                     'residual_channels': 32,
                     'skip_channels': 64,
                     'end_channels': 128,
                     'gcn_depth': 2,
                     'dropout': 0.3,
                     'subgraph_size': 20,
                     'node_dim': 40,
                     'num_layers': 3,
                     'propalpha': 0.05,
                     'tanhalpha': 3,
                     'dilation_exponential': 1},
           'training': {'learning_rate': 0.001,
                        'weight_decay': 0.0001,
                        'early_stop_patience': 0,
                        'clip_grad': 5,
                        'lr_milestones': [],
                        'lr_gamma': 0.1,
                        'optimizer': 'adam',
                        'optimizer_eps': 1e-08,
                        'lr_scheduler': 'none',
                        'train_loss': 'mae'}},
 'STAEFORMER': {'model': {'input_embedding_dim': 24,
                          'tod_embedding_dim': 24,
                          'dow_embedding_dim': 24,
                          'spatial_embedding_dim': 0,
                          'adaptive_embedding_dim': 80,
                          'feed_forward_dim': 256,
                          'num_heads': 4,
                          'num_layers': 3,
                          'dropout': 0.1,
                          'use_mixed_proj': True},
                'training': {'learning_rate': 0.001,
                             'weight_decay': 0.0005,
                             'early_stop_patience': 20,
                             'clip_grad': 0,
                             'lr_milestones': [15, 30, 50],
                             'lr_gamma': 0.1,
                             'optimizer': 'adam',
                             'optimizer_eps': 1e-08,
                             'lr_scheduler': 'multistep',
                             'train_loss': 'mae'}},
 'HIMNET': {'model': {'input_embedding_dim': 64,
                      'num_layers': 1,
                      'cheb_k': 2,
                      'tod_embedding_dim': 12,
                      'dow_embedding_dim': 4,
                      'node_embedding_dim': 16,
                      'st_embedding_dim': 16,
                      'tf_decay_steps': 4000,
                      'use_teacher_forcing': True},
            'training': {'learning_rate': 0.001,
                         'weight_decay': 0.0001,
                         'early_stop_patience': 20,
                         'clip_grad': 5,
                         'lr_milestones': [30, 50],
                         'lr_gamma': 0.1,
                         'optimizer': 'adam',
                         'optimizer_eps': 0.001,
                         'lr_scheduler': 'multistep',
                         'train_loss': 'mae'}},
 'MEGACRN': {'model': {'input_embedding_dim': 64,
                       'num_layers': 1,
                       'cheb_k': 3,
                       'mem_num': 20,
                       'mem_dim': 64,
                       'cl_decay_steps': 2000,
                       'use_curriculum_learning': True,
                       'ycov_dim': 2,
                       'lamb': 0.01,
                       'lamb1': 0.01},
             'training': {'learning_rate': 0.01,
                          'weight_decay': 0,
                          'early_stop_patience': 20,
                          'clip_grad': 5,
                          'lr_milestones': [50, 100],
                          'lr_gamma': 0.1,
                          'optimizer': 'adam',
                          'optimizer_eps': 0.001,
                          'lr_scheduler': 'multistep',
                          'train_loss': 'mae'}},
 'STDN': {'model': {'input_embedding_dim': 128,
                    'K': 16,
                    'd': 8,
                    'L': 2,
                    'order': 3,
                    'bn_decay': 0.1},
          'training': {'learning_rate': 0.001,
                       'weight_decay': 0,
                       'early_stop_patience': 10,
                       'clip_grad': 0,
                       'lr_milestones': [],
                       'lr_gamma': 0.9,
                       'optimizer': 'adam',
                       'optimizer_eps': 1e-08,
                       'lr_scheduler': 'step',
                       'lr_step_size': 10,
                       'train_loss': 'mae'}},
 'DCRNN': {'model': {'input_embedding_dim': 64,
                     'num_rnn_layers': 2,
                     'rnn_units': 64,
                     'max_diffusion_step': 2,
                     'filter_type': 'dual_random_walk',
                     'use_curriculum_learning': True,
                     'cl_decay_steps': 2000},
           'training': {'learning_rate': 0.01,
                        'weight_decay': 0,
                        'early_stop_patience': 50,
                        'clip_grad': 5,
                        'lr_milestones': [20, 30, 40, 50],
                        'lr_gamma': 0.1,
                        'optimizer': 'adam',
                        'optimizer_eps': 0.001,
                        'lr_scheduler': 'multistep',
                        'min_learning_rate': 2e-06,
                        'train_loss': 'mae'}},
 'STSSL': {'model': {'input_embedding_dim': 64,
                     'num_layers': 2,
                     'dropout': 0.1,
                     'nmb_prototype': 4,
                     'spatial_temp': 0.5,
                     'augmentation_percent': 0.1},
           'training': {'learning_rate': 0.001,
                        'weight_decay': 0,
                        'early_stop_patience': 15,
                        'clip_grad': 5,
                        'lr_milestones': [],
                        'lr_gamma': 0.1,
                        'optimizer': 'adam',
                        'optimizer_eps': 1e-08,
                        'lr_scheduler': 'none',
                        'train_loss': 'mae'}},
 'STSSDL': {'model': {'input_embedding_dim': 128,
                      'rnn_units': 128,
                      'rnn_layers': 1,
                      'cheb_k': 3,
                      'prototype_num': 20,
                      'prototype_dim': 64,
                      'tod_embed_dim': 10,
                      'node_embed_dim': 20,
                      'adaptive_embed_dim': 48,
                      'lambda_contrastive': 0.1,
                      'lambda_deviation': 1.0,
                      'cl_decay_steps': 2000,
                      'use_curriculum_learning': False},
            'training': {'learning_rate': 0.01,
                         'weight_decay': 0,
                         'early_stop_patience': 30,
                         'clip_grad': 5,
                         'lr_milestones': [50, 100],
                         'lr_gamma': 0.1,
                         'optimizer': 'adam',
                         'optimizer_eps': 0.001,
                         'lr_scheduler': 'multistep',
                         'train_loss': 'mae'}},
 'STGCN': {'model': {'input_embedding_dim': 64, 'num_layers': 2, 'dropout': 0},
           'training': {'learning_rate': 0.001,
                        'weight_decay': 0,
                        'early_stop_patience': 0,
                        'clip_grad': 0,
                        'lr_milestones': [],
                        'lr_gamma': 0.7,
                        'optimizer': 'rmsprop',
                        'optimizer_eps': 1e-10,
                        'lr_scheduler': 'step',
                        'lr_step_size': 5,
                        'rmsprop_alpha': 0.9,
                        'train_loss': 'mse'}},
 'STNORM': {'model': {'input_embedding_dim': 16,
                      'blocks': 1,
                      'num_layers': 4,
                      'kernel_size': 2,
                      'snorm_bool': True,
                      'tnorm_bool': True,
                      'dropout': 0},
            'training': {'learning_rate': 0.0001,
                         'weight_decay': 0,
                         'early_stop_patience': 50,
                         'clip_grad': 10,
                         'lr_milestones': [],
                         'lr_gamma': 0.1,
                         'optimizer': 'adam',
                         'optimizer_eps': 1e-08,
                         'lr_scheduler': 'none',
                         'train_loss': 'mse'}},
 'STID': {'model': {'input_embedding_dim': 32,
                    'num_layers': 3,
                    'spatial_emb_dim': 32,
                    'temp_dim_tid': 32,
                    'temp_dim_diw': 32,
                    'if_spatial': True,
                    'if_time_in_day': True,
                    'if_day_in_week': True},
          'training': {'learning_rate': 0.002,
                       'weight_decay': 0.0001,
                       'early_stop_patience': 0,
                       'clip_grad': 5,
                       'lr_milestones': [1, 50, 80],
                       'lr_gamma': 0.5,
                       'optimizer': 'adam',
                       'optimizer_eps': 1e-08,
                       'lr_scheduler': 'multistep',
                       'train_loss': 'mae'}},
 'HA': {'model': {'output_dim': 1},
        'training': {'learning_rate': 0.001,
                     'weight_decay': 1e-05,
                     'early_stop_patience': 15,
                     'clip_grad': 5,
                     'lr_milestones': [20, 40, 50, 60, 70],
                     'lr_gamma': 0.1,
                     'optimizer': 'adam',
                     'optimizer_eps': 1e-08,
                     'lr_scheduler': 'multistep',
                     'train_loss': 'mae'}},
 'GRU': {'model': {'input_embedding_dim': 64,
                   'num_layers': 3,
                   'dropout': 0.1,
                   'input_dim': 3,
                   'output_dim': 1},
         'training': {'learning_rate': 0.001,
                      'weight_decay': 1e-05,
                      'early_stop_patience': 15,
                      'clip_grad': 5,
                      'lr_milestones': [20, 40, 50, 60, 70],
                      'lr_gamma': 0.1,
                      'optimizer': 'adam',
                      'optimizer_eps': 1e-08,
                      'lr_scheduler': 'multistep',
                      'train_loss': 'mae'}}}


def resolve_model_args(name, args):
    name = name.upper()
    supplied = dict(args) if isinstance(args, dict) else vars(args).copy()
    # Normalize legacy spellings once; downstream code uses only max_epoch.
    for alias in ('epochs', 'max_epochs', 'epoch'):
        if alias in supplied:
            maximum = supplied.pop(alias)
            if 'max_epoch' in supplied and supplied['max_epoch'] != maximum:
                raise ValueError('epoch aliases must agree with max_epoch')
            supplied['max_epoch'] = maximum
    if name in BASELINE_CONFIGS:
        config = deepcopy(BASELINE_CONFIGS[name])
        values = {**supplied, **config['model'], **config['training']}
        # Shared engine feature layout and LR floor are also fixed for baselines.
        values.update(input_dim=3, output_dim=1)
        values['min_learning_rate'] = config['training'].get('min_learning_rate', 0.0)
        interval = values.get('time_intervals', 1800)
        values['steps_per_day'] = 86400 // interval
        values['time_slice_size'] = interval // 60
    else:
        defaults = model_defaults(name)
        defaults.pop('epochs', None)
        defaults.pop('batch_size', None)
        values = {**defaults, **supplied}
        interval = values.get('time_intervals', 1800)
        values.setdefault('steps_per_day', 86400 // interval)
        values.setdefault('time_slice_size', interval // 60)
    values.setdefault('max_epoch', 100)
    values.setdefault('batch_size', 64)
    return SimpleNamespace(**values)


def _get_arg(args, name, default):
    """Get argument with fallback to default."""
    return getattr(args, name, default)


def _adj_to_numpy(adj_mx):
    """Convert adjacency matrix to numpy array."""
    if isinstance(adj_mx, torch.Tensor):
        return adj_mx.cpu().numpy()
    return adj_mx


def _build_stgcn(args, num_nodes, adj_mx, device):
    """Build STGCN model."""
    from models.STGCN.STCGN import STGCN
    adj_mx_np = _adj_to_numpy(adj_mx)
    return STGCN(
        num_nodes=num_nodes,
        model_dim=64,
        output_dim=1,
        in_window=_get_arg(args, 'input_window', 24),
        out_window=_get_arg(args, 'output_window', 24),
        num_layers=2,
        dropout=0,
        adj_mx=adj_mx_np,
        device=device
    ).to(device)


def _build_agcrn(args, num_nodes, adj_mx, device):
    """Build AGCRN model."""
    from models.AGCRN.AGCRN import AGCRN
    return AGCRN(
        num_nodes=num_nodes,
        feature_dim=1,
        hidden_dim=64,
        output_dim=1,
        in_window=_get_arg(args, 'input_window', 24),
        out_window=_get_arg(args, 'output_window', 24),
        embed_dim=10,
        num_layers=2,
        cheb_k=2,
        device=device
    ).to(device)


def _build_gwnet(args, num_nodes, adj_mx, device):
    """Build GWNET model."""
    from models.GWNET.GWNET import GWNET
    adj_mx_np = _adj_to_numpy(adj_mx)
    return GWNET(
        num_nodes=num_nodes,
        feature_dim=3,
        output_dim=1,
        in_window=_get_arg(args, 'input_window', 24),
        out_window=_get_arg(args, 'output_window', 24),
        dropout=0.3,
        blocks=4,
        layers=2,
        gcn_bool=True,
        addaptadj=True,
        adjtype='doubletransition',
        randomadj=True,
        aptonly=False,
        kernel_size=2,
        nhid=32,
        adj_mx=adj_mx_np,
        device=device
    ).to(device)


def _build_megacrn(args, num_nodes, adj_mx, device):
    """Build MegaCRN model."""
    from models.MegaCRN.MegaCRN import MegaCRN
    return MegaCRN(
        num_nodes=num_nodes,
        rnn_units=64,
        output_dim=1,
        in_window=_get_arg(args, 'input_window', 24),
        out_window=_get_arg(args, 'output_window', 24),
        num_layers=1,
        cheb_k=3,
        mem_num=20,
        mem_dim=64,
        cl_decay_steps=2000,
        use_curriculum_learning=True,
        ycov_dim=2,
        lamb=0.01,
        lamb1=0.01,
        device=device
    ).to(device)


def _build_stnorm(args, num_nodes, adj_mx, device):
    """Build STNorm model."""
    from models.STNorm.STNorm import STNorm
    return STNorm(
        num_nodes=num_nodes,
        feature_dim=3,
        output_dim=1,
        in_window=_get_arg(args, 'input_window', 24),
        out_window=_get_arg(args, 'output_window', 24),
        blocks=1,
        layers=4,
        kernel_size=2,
        channels=16,
        snorm_bool=True,
        tnorm_bool=True,
        dropout=0,
        device=device
    ).to(device)


def _build_mtgnn(args, num_nodes, adj_mx, device):
    """Build MTGNN model."""
    from models.MTGNN.MTGNN import MTGNN
    adj_mx_np = _adj_to_numpy(adj_mx)
    return MTGNN(
        num_nodes=num_nodes,
        feature_dim=3,
        output_dim=1,
        in_window=_get_arg(args, 'input_window', 24),
        out_window=_get_arg(args, 'output_window', 24),
        conv_channels=32,
        residual_channels=32,
        skip_channels=64,
        end_channels=128,
        gcn_depth=2,
        dropout=0.3,
        subgraph_size=20,
        node_dim=40,
        layers=3,
        propalpha=0.05,
        tanhalpha=3,
        dilation_exponential=1,
        adj_mx=adj_mx_np,
        device=device
    ).to(device)


def _build_staeformer(args, num_nodes, adj_mx, device):
    """Build STAEformer model."""
    from models.STAEformer.STAEformer import STAEformer
    return STAEformer(
        num_nodes=num_nodes,
        in_steps=_get_arg(args, 'input_window', 24),
        out_steps=_get_arg(args, 'output_window', 24),
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
        steps_per_day=_get_arg(args, 'steps_per_day', 48),
        device=device
    ).to(device)


def _build_stdn(args, num_nodes, adj_mx, device):
    """Build STDN model."""
    from models.STDN.STDN import STDN
    adj_mx_np = _adj_to_numpy(adj_mx)
    return STDN(
        num_nodes=num_nodes,
        model_dim=128,
        input_dim=1,
        output_dim=1,
        input_window=_get_arg(args, 'input_window', 24),
        output_window=_get_arg(args, 'output_window', 24),
        K=16,
        d=8,
        L=2,
        order=3,
        bn_decay=0.1,
        time_slice_size=_get_arg(args, 'time_slice_size', 60),
        adj_mx=adj_mx_np,
        device=device
    ).to(device)


def _build_stid(args, num_nodes, adj_mx, device):
    """Build STID model."""
    from models.STID.STID import STID
    return STID(
        num_nodes=num_nodes,
        input_window=_get_arg(args, 'input_window', 24),
        output_window=_get_arg(args, 'output_window', 24),
        feature_dim=3,
        output_dim=1,
        time_intervals=_get_arg(args, 'time_intervals', 1800),
        num_block=3,
        time_series_emb_dim=32,
        spatial_emb_dim=32,
        temp_dim_tid=32,
        temp_dim_diw=32,
        if_spatial=True,
        if_time_in_day=True,
        if_day_in_week=True,
        device=device
    ).to(device)


def _build_prophet(args, num_nodes, adj_mx, device):
    """PROPHET owns its argument mapping; keep the factory as a thin adapter."""
    from .PROPHET.model import PROPHET
    return PROPHET.from_args(args, num_nodes, adj_mx, device).to(device)


def _build_pstid(args, num_nodes, adj_mx, device):
    """Build the independent PSTID development copy of PROPHET."""
    from .PSTID.model import PSTID
    return PSTID.from_args(args, num_nodes, adj_mx, device).to(device)


def _build_dcrnn(args, num_nodes, adj_mx, device):
    """Build DCRNN model."""
    from models.DCRNN.DCRNN import DCRNN
    adj_mx_np = _adj_to_numpy(adj_mx)
    return DCRNN(
        num_nodes=num_nodes,
        input_embedding_dim=64,
        output_dim=1,
        input_window=_get_arg(args, 'input_window', 12),
        output_window=_get_arg(args, 'output_window', 12),
        num_rnn_layers=2,
        rnn_units=64,
        max_diffusion_step=2,
        filter_type='dual_random_walk',
        cl_decay_steps=2000,
        use_curriculum_learning=True,
        adj_mx=adj_mx_np,
        device=device
    ).to(device)


def _build_stssl(args, num_nodes, adj_mx, device):
    """Build STSSL model."""
    from models.STSSL.STSSL import STSSL
    adj_mx_np = _adj_to_numpy(adj_mx)
    return STSSL(
        num_nodes=num_nodes,
        model_dim=64,
        output_dim=1,
        in_window=_get_arg(args, 'input_window', 24),
        out_window=_get_arg(args, 'output_window', 24),
        num_layers=2,
        dropout=0.1,
        Kt=3, Ks=3,
        nmb_prototype=4,
        spatial_temp=0.5,
        augmentation_percent=0.1,
        adj_mx=adj_mx_np,
        device=device,
        time_intervals=_get_arg(args, 'time_intervals', 1800)
    ).to(device)


def _build_stssdl(args, num_nodes, adj_mx, device):
    """Build STSSDL model."""
    from models.STSSDL.STSSDL import STSSDL
    adj_mx_np = _adj_to_numpy(adj_mx)
    return STSSDL(
        num_nodes=num_nodes,
        model_dim=128,
        output_dim=1,
        in_window=_get_arg(args, 'input_window', 24),
        out_window=_get_arg(args, 'output_window', 24),
        rnn_units=128,
        rnn_layers=1,
        cheb_k=3,
        prototype_num=20,
        prototype_dim=64,
        tod_embed_dim=10,
        node_embed_dim=20,
        adaptive_embed_dim=48,
        lambda_contrastive=0.1,
        lambda_deviation=1.0,
        cl_decay_steps=2000,
        use_curriculum_learning=False,
        adj_mx=adj_mx_np,
        device=device,
        time_intervals=_get_arg(args, 'time_intervals', 1800)
    ).to(device)


def _build_himnet(args, num_nodes, adj_mx, device):
    """Build HimNet model."""
    from models.HimNet.HimNet import HimNet
    adj_mx_np = _adj_to_numpy(adj_mx)
    return HimNet(
        num_nodes=num_nodes,
        input_dim=3,
        output_dim=1,
        out_steps=_get_arg(args, 'output_window', 24),
        in_window=_get_arg(args, 'input_window', 24),
        hidden_dim=64,
        num_layers=1,
        cheb_k=2,
        ycov_dim=2,
        tod_embedding_dim=12,
        dow_embedding_dim=4,
        node_embedding_dim=16,
        st_embedding_dim=16,
        tf_decay_steps=4000,
        use_teacher_forcing=True,
        adj_mx=adj_mx_np,
        device=device,
        time_intervals=_get_arg(args, 'time_intervals', 1800),
        time_slots=_get_arg(args, 'himnet_time_slots', None),
    ).to(device)


def _build_ha(args, num_nodes, adj_mx, device):
    """Build HA (Historical Average) baseline model."""
    from models.HA.HA import HA
    model = HA(
        num_nodes=num_nodes,
        input_window=_get_arg(args, 'input_window', 24),
        output_window=_get_arg(args, 'output_window', 24),
        output_dim=1,
    )
    return model.to(device)


def _build_gru(args, num_nodes, adj_mx, device):
    """Build GRU baseline model."""
    from models.GRU.GRU import GRU
    return GRU(
        num_nodes=num_nodes,
        input_window=_get_arg(args, 'input_window', 24),
        output_window=_get_arg(args, 'output_window', 24),
        input_dim=3,
        output_dim=1,
        input_embedding_dim=64,
        num_layers=3,
        dropout=0.1,
    ).to(device)


MODEL_REGISTRY = {
    "HA": _build_ha,
    "GRU": _build_gru,
    "STGCN": _build_stgcn,
    "AGCRN": _build_agcrn,
    "GWNET": _build_gwnet,
    "MEGACRN": _build_megacrn,
    "STNORM": _build_stnorm,
    "MTGNN": _build_mtgnn,
    "STAEFORMER": _build_staeformer,
    "STDN": _build_stdn,
    "STID": _build_stid,
    "PROPHET": _build_prophet,
    "PSTID": _build_pstid,
    "DCRNN": _build_dcrnn,
    "STSSL": _build_stssl,
    "STSSDL": _build_stssdl,
    "HIMNET": _build_himnet,
}
MODEL_LIST = list(MODEL_REGISTRY)


def build_model(model_name, args, num_nodes, adj_mx, device):
    """Build a model by case-insensitive name using the registered adapter."""
    name = model_name.upper()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {MODEL_LIST}")
    args = resolve_model_args(name, args)
    return MODEL_REGISTRY[name](args, num_nodes, adj_mx, device)
