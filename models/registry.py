"""Model registry with lazy imports and unchanged constructor adapters."""

import torch


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
        model_dim=_get_arg(args, 'input_embedding_dim', 64),
        output_dim=1,
        in_window=_get_arg(args, 'input_window', 24),
        out_window=_get_arg(args, 'output_window', 24),
        num_layers=_get_arg(args, 'num_layers', 2),
        dropout=_get_arg(args, 'dropout', 0.1),
        adj_mx=adj_mx_np,
        device=device
    ).to(device)


def _build_agcrn(args, num_nodes, adj_mx, device):
    """Build AGCRN model."""
    from models.AGCRN.AGCRN import AGCRN
    return AGCRN(
        num_nodes=num_nodes,
        feature_dim=1,
        hidden_dim=_get_arg(args, 'input_embedding_dim', 64),
        output_dim=1,
        in_window=_get_arg(args, 'input_window', 24),
        out_window=_get_arg(args, 'output_window', 24),
        embed_dim=_get_arg(args, 'embed_dim', 10),
        num_layers=_get_arg(args, 'num_layers', 2),
        cheb_k=_get_arg(args, 'cheb_k', 2),
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
        dropout=_get_arg(args, 'dropout', 0.3),
        blocks=_get_arg(args, 'blocks', 4),
        layers=_get_arg(args, 'num_layers', 3),
        gcn_bool=_get_arg(args, 'gcn_bool', True),
        addaptadj=_get_arg(args, 'addaptadj', True),
        adjtype=_get_arg(args, 'adjtype', 'doubletransition'),
        randomadj=_get_arg(args, 'randomadj', True),
        aptonly=_get_arg(args, 'aptonly', True),
        kernel_size=_get_arg(args, 'kernel_size', 2),
        nhid=_get_arg(args, 'input_embedding_dim', 64),
        adj_mx=adj_mx_np,
        device=device
    ).to(device)


def _build_megacrn(args, num_nodes, adj_mx, device):
    """Build MegaCRN model."""
    from models.MegaCRN.MegaCRN import MegaCRN
    return MegaCRN(
        num_nodes=num_nodes,
        rnn_units=_get_arg(args, 'input_embedding_dim', 64),
        output_dim=1,
        in_window=_get_arg(args, 'input_window', 24),
        out_window=_get_arg(args, 'output_window', 24),
        num_layers=_get_arg(args, 'num_layers', 3),
        cheb_k=_get_arg(args, 'cheb_k', 2),
        mem_num=_get_arg(args, 'mem_num', 20),
        mem_dim=_get_arg(args, 'mem_dim', 64),
        cl_decay_steps=_get_arg(args, 'cl_decay_steps', 200),
        use_curriculum_learning=_get_arg(args, 'use_curriculum_learning', False),
        ycov_dim=_get_arg(args, 'ycov_dim', 2),
        lamb=_get_arg(args, 'lamb', 0.01),
        lamb1=_get_arg(args, 'lamb1', 0.01),
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
        blocks=_get_arg(args, 'blocks', 1),
        layers=_get_arg(args, 'num_layers', 3),
        kernel_size=_get_arg(args, 'kernel_size', 2),
        channels=_get_arg(args, 'input_embedding_dim', 64),
        snorm_bool=_get_arg(args, 'snorm_bool', True),
        tnorm_bool=_get_arg(args, 'tnorm_bool', True),
        dropout=_get_arg(args, 'dropout', 0.1),
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
        conv_channels=_get_arg(args, 'input_embedding_dim', 64),
        residual_channels=_get_arg(args, 'input_embedding_dim', 64),
        skip_channels=_get_arg(args, 'input_embedding_dim', 64),
        end_channels=_get_arg(args, 'input_embedding_dim', 64),
        gcn_depth=_get_arg(args, 'gcn_depth', 2),
        dropout=_get_arg(args, 'dropout', 0.1),
        subgraph_size=_get_arg(args, 'subgraph_size', 20),
        node_dim=_get_arg(args, 'node_dim', 40),
        layers=_get_arg(args, 'num_layers', 3),
        propalpha=_get_arg(args, 'propalpha', 0.05),
        tanhalpha=_get_arg(args, 'tanhalpha', 3),
        dilation_exponential=_get_arg(args, 'dilation_exponential', 1),
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
        input_embedding_dim=_get_arg(args, 'input_embedding_dim', 64),
        tod_embedding_dim=_get_arg(args, 'tod_embedding_dim', 16),
        dow_embedding_dim=_get_arg(args, 'dow_embedding_dim', 16),
        spatial_embedding_dim=_get_arg(args, 'spatial_embedding_dim', 0),
        adaptive_embedding_dim=_get_arg(args, 'adaptive_embedding_dim', 64),
        feed_forward_dim=_get_arg(args, 'feed_forward_dim', 256),
        num_heads=_get_arg(args, 'num_heads', 4),
        num_layers=_get_arg(args, 'num_layers', 3),
        dropout=_get_arg(args, 'dropout', 0.1),
        use_mixed_proj=_get_arg(args, 'use_mixed_proj', True),
        steps_per_day=_get_arg(args, 'steps_per_day', 48),
        device=device
    ).to(device)


def _build_stdn(args, num_nodes, adj_mx, device):
    """Build STDN model."""
    from models.STDN.STDN import STDN
    adj_mx_np = _adj_to_numpy(adj_mx)
    return STDN(
        num_nodes=num_nodes,
        model_dim=_get_arg(args, 'input_embedding_dim', 64),
        input_dim=1,
        output_dim=1,
        input_window=_get_arg(args, 'input_window', 24),
        output_window=_get_arg(args, 'output_window', 24),
        K=_get_arg(args, 'K', 8),
        d=_get_arg(args, 'd', 16),
        L=_get_arg(args, 'L', 2),
        order=_get_arg(args, 'order', 2),
        bn_decay=_get_arg(args, 'bn_decay', 0.1),
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
        num_block=_get_arg(args, 'num_layers', 3),
        time_series_emb_dim=_get_arg(args, 'input_embedding_dim', 32),
        spatial_emb_dim=_get_arg(args, 'spatial_emb_dim', 16),
        temp_dim_tid=_get_arg(args, 'temp_dim_tid', 16),
        temp_dim_diw=_get_arg(args, 'temp_dim_diw', 16),
        if_spatial=_get_arg(args, 'if_spatial', True),
        if_time_in_day=_get_arg(args, 'if_time_in_day', True),
        if_day_in_week=_get_arg(args, 'if_day_in_week', True),
        device=device
    ).to(device)


def _build_pstid(args, num_nodes, adj_mx, device):
    """PSTID owns its argument mapping; keep the factory as a thin adapter."""
    from .PSTID.model import PSTID
    return PSTID.from_args(args, num_nodes, adj_mx, device).to(device)


def _build_dcrnn(args, num_nodes, adj_mx, device):
    """Build DCRNN model."""
    from models.DCRNN.DCRNN import DCRNN
    adj_mx_np = _adj_to_numpy(adj_mx)
    return DCRNN(
        num_nodes=num_nodes,
        input_embedding_dim=_get_arg(args, 'input_embedding_dim', 64),
        output_dim=1,
        input_window=_get_arg(args, 'input_window', 12),
        output_window=_get_arg(args, 'output_window', 12),
        num_rnn_layers=_get_arg(args, 'num_rnn_layers', 2),
        rnn_units=_get_arg(args, 'rnn_units', 64),
        max_diffusion_step=_get_arg(args, 'max_diffusion_step', 2),
        filter_type=_get_arg(args, 'filter_type', 'laplacian'),
        use_curriculum_learning=_get_arg(args, 'use_curriculum_learning', False),
        adj_mx=adj_mx_np,
        device=device
    ).to(device)


def _build_stssl(args, num_nodes, adj_mx, device):
    """Build STSSL model."""
    from models.STSSL.STSSL import STSSL
    adj_mx_np = _adj_to_numpy(adj_mx)
    return STSSL(
        num_nodes=num_nodes,
        model_dim=_get_arg(args, 'input_embedding_dim', 32),
        output_dim=1,
        in_window=_get_arg(args, 'input_window', 24),
        out_window=_get_arg(args, 'output_window', 24),
        num_layers=_get_arg(args, 'num_layers', 3),
        dropout=_get_arg(args, 'dropout', 0.1),
        Kt=3, Ks=3,
        nmb_prototype=_get_arg(args, 'nmb_prototype', 32),
        spatial_temp=_get_arg(args, 'spatial_temp', 0.5),
        augmentation_percent=_get_arg(args, 'augmentation_percent', 0.1),
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
        model_dim=_get_arg(args, 'input_embedding_dim', 32),
        output_dim=1,
        in_window=_get_arg(args, 'input_window', 24),
        out_window=_get_arg(args, 'output_window', 24),
        rnn_units=_get_arg(args, 'rnn_units', 64),
        rnn_layers=_get_arg(args, 'rnn_layers', 1),
        cheb_k=_get_arg(args, 'cheb_k', 2),
        prototype_num=_get_arg(args, 'prototype_num', 16),
        prototype_dim=_get_arg(args, 'prototype_dim', 32),
        tod_embed_dim=_get_arg(args, 'tod_embed_dim', 16),
        node_embed_dim=_get_arg(args, 'node_embed_dim', 32),
        adaptive_embed_dim=_get_arg(args, 'adaptive_embed_dim', 32),
        lambda_contrastive=_get_arg(args, 'lambda_contrastive', 1e-2),
        lambda_deviation=_get_arg(args, 'lambda_deviation', 1e-2),
        cl_decay_steps=_get_arg(args, 'cl_decay_steps', 200),
        use_curriculum_learning=_get_arg(args, 'use_curriculum_learning', True),
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
        hidden_dim=_get_arg(args, 'input_embedding_dim', 16),
        num_layers=_get_arg(args, 'num_layers', 2),
        cheb_k=_get_arg(args, 'cheb_k', 2),
        ycov_dim=2,
        tod_embedding_dim=_get_arg(args, 'tod_embedding_dim', 8),
        dow_embedding_dim=_get_arg(args, 'dow_embedding_dim', 8),
        node_embedding_dim=_get_arg(args, 'node_embedding_dim', 16),
        st_embedding_dim=_get_arg(args, 'st_embedding_dim', 16),
        tf_decay_steps=_get_arg(args, 'tf_decay_steps', 4000),
        use_teacher_forcing=_get_arg(args, 'use_teacher_forcing', True),
        adj_mx=adj_mx_np,
        device=device,
    ).to(device)


def _build_ha(args, num_nodes, adj_mx, device):
    """Build HA (Historical Average) baseline model."""
    from models.HA.HA import HA
    model = HA(
        num_nodes=num_nodes,
        input_window=_get_arg(args, 'input_window', 24),
        output_window=_get_arg(args, 'output_window', 24),
        output_dim=_get_arg(args, 'output_dim', 1),
    )
    return model.to(device)


def _build_gru(args, num_nodes, adj_mx, device):
    """Build GRU baseline model."""
    from models.GRU.GRU import GRU
    return GRU(
        num_nodes=num_nodes,
        input_window=_get_arg(args, 'input_window', 24),
        output_window=_get_arg(args, 'output_window', 24),
        input_dim=_get_arg(args, 'input_dim', 3),
        output_dim=_get_arg(args, 'output_dim', 1),
        input_embedding_dim=_get_arg(args, 'input_embedding_dim', 64),
        num_layers=_get_arg(args, 'num_layers', 3),
        dropout=_get_arg(args, 'dropout', 0.1),
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
    return MODEL_REGISTRY[name](args, num_nodes, adj_mx, device)
