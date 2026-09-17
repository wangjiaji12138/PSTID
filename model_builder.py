"""
Model Builder: Creates models based on model name with proper parameter mapping.
"""

import torch
import numpy as np

from models.STGCN.STCGN import STGCN
from models.AGCRN.AGCRN import AGCRN
from models.GWNET.GWNET import GWNET
from models.MegaCRN.MegaCRN import MegaCRN
from models.STNorm.STNorm import STNorm
from models.MTGNN.MTGNN import MTGNN
from models.STAEformer.STAEformer import STAEformer
from models.STDN.STDN import STDN
from models.STID.STID import STID
from models.PSTID.PSTID import PSTID
from models.DCRNN.DCRNN import DCRNN
from models.STSSL.STSSL import STSSL
from models.STSSDL.STSSDL import STSSDL
from models.PDFormer.PDFormer import PDFormer
from models.HA.HA import HA
from models.GRU.GRU import GRU


MODEL_LIST = ["HA", "GRU", "STGCN", "AGCRN", "GWNET", "MEGACRN", "STNORM",
              "MTGNN", "STAEFORMER", "STDN", "STID", "PSTID", "DCRNN", "STSSL", "STSSDL",
              "PDFORMER"]


def build_model(model_name: str, args, num_nodes: int, adj_mx: torch.Tensor, device: torch.device):
    """Build model by name with parameters from args.

    Args:
        model_name: Name of the model (case-insensitive)
        args: Argument namespace from train.py
        num_nodes: Number of nodes in the graph
        adj_mx: Adjacency matrix (can be numpy array or tensor)
        device: torch device

    Returns:
        Initialized model
    """
    model_name = model_name.upper()

    if model_name == "STGCN":
        return _build_stgcn(args, num_nodes, adj_mx, device)
    elif model_name == "AGCRN":
        return _build_agcrn(args, num_nodes, adj_mx, device)
    elif model_name == "GWNET":
        return _build_gwnet(args, num_nodes, adj_mx, device)
    elif model_name == "MEGACRN":
        return _build_megacrn(args, num_nodes, adj_mx, device)
    elif model_name == "STNORM":
        return _build_stnorm(args, num_nodes, adj_mx, device)
    elif model_name == "MTGNN":
        return _build_mtgnn(args, num_nodes, adj_mx, device)
    elif model_name == "STAEFORMER":
        return _build_staeformer(args, num_nodes, adj_mx, device)
    elif model_name == "STDN":
        return _build_stdn(args, num_nodes, adj_mx, device)
    elif model_name == "STID":
        return _build_stid(args, num_nodes, adj_mx, device)
    elif model_name == "PSTID":
        return _build_pstid(args, num_nodes, adj_mx, device)
    elif model_name == "DCRNN":
        return _build_dcrnn(args, num_nodes, adj_mx, device)
    elif model_name == "STSSL":
        return _build_stssl(args, num_nodes, adj_mx, device)
    elif model_name == "STSSDL":
        return _build_stssdl(args, num_nodes, adj_mx, device)
    elif model_name == "PDFORMER":
        return _build_pdformer(args, num_nodes, adj_mx, device, pdformer_data=args.pdformer_data)
    elif model_name == "HA":
        return _build_ha(args, num_nodes, adj_mx, device)
    elif model_name == "GRU":
        return _build_gru(args, num_nodes, adj_mx, device)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {MODEL_LIST}")


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
    return AGCRN(
        num_nodes=num_nodes,
        feature_dim=3,
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
    """Build PSTID model: STID with Prototype Module.
    
    PSTID 在 STID 的基础上，在输入嵌入和 MLP 之间加原型模块，
    通过原型模块将节点嵌入归类成空间码本信息，将时间嵌入归类成时间码本信息，
    从而减少噪声并提升模型的可解释性。
    """
    return PSTID(
        num_nodes=num_nodes,
        input_window=_get_arg(args, 'input_window', 24),
        output_window=_get_arg(args, 'output_window', 24),
        feature_dim=3,
        output_dim=1,
        time_intervals=_get_arg(args, 'time_intervals', 1800),
        num_block=_get_arg(args, 'num_layers', 3),
        time_series_emb_dim=_get_arg(args, 'input_embedding_dim', 32),
        spatial_emb_dim=_get_arg(args, 'node_emb_dim', 16),
        temp_dim_tid=_get_arg(args, 'tid', 16),
        temp_dim_diw=_get_arg(args, 'diw', 16),
        if_spatial=_get_arg(args, 'if_spatial', True),
        if_time_in_day=_get_arg(args, 'if_time_in_day', True),
        if_day_in_week=_get_arg(args, 'if_day_in_week', True),
        device=device,
        # ProtoModule 参数
        num_spatial_prototypes=_get_arg(args, 'num_spatial_prototypes', 16),
        num_temporal_prototypes=_get_arg(args, 'num_temporal_prototypes', 8),
        proto_temperature=_get_arg(args, 'proto_temperature', 0.5),
        use_proto=_get_arg(args, 'use_proto', True),
        use_spatio=_get_arg(args, 'use_spatio', True),
        use_temporal=_get_arg(args, 'use_temporal', True),
        proto_uniformity_weight=_get_arg(args, 'proto_uniformity_weight', 0.0),
        proto_emb_dim=_get_arg(args, 'proto_emb_dim', 64),
    ).to(device)


def _build_dcrnn(args, num_nodes, adj_mx, device):
    """Build DCRNN model."""
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


def _build_pdformer(args, num_nodes, adj_mx, device, pdformer_data=None):
    """Build PDFormer model.
    
    Args:
        args: Argument namespace from train.py
        num_nodes: Number of nodes in the graph
        adj_mx: Adjacency matrix
        device: torch device
        pdformer_data: dict with preprocessed PDFormer data:
            - lap_mx: Laplacian PE matrix (N, lape_dim)
            - pattern_keys: Pattern centroids (n_cluster, s_attn_size, 1)
            - geo_mask: Geographic mask (N, N)
            - sem_mask: Semantic mask (N, N)
    
    Returns:
        PDFormer model
    """
    adj_mx_np = _adj_to_numpy(adj_mx)
    
    # Extract PDFormer-specific data
    lap_mx = pdformer_data.get('lap_mx') if pdformer_data else None
    pattern_keys = pdformer_data.get('pattern_keys') if pdformer_data else None
    geo_mask = pdformer_data.get('geo_mask') if pdformer_data else None
    sem_mask = pdformer_data.get('sem_mask') if pdformer_data else None
    
    return PDFormer(
        num_nodes=num_nodes,
        # In our pipeline, the last 2 input channels are time-in-day and day-of-week
        # So the data feature dimension is input_dim - 2
        feature_dim=max(1, _get_arg(args, 'input_dim', 3) - 2),
        output_dim=_get_arg(args, 'output_dim', 1),
        in_window=_get_arg(args, 'input_window', 24),
        out_window=_get_arg(args, 'output_window', 24),
        embed_dim=_get_arg(args, 'embed_dim', 64),
        lape_dim=_get_arg(args, 'lape_dim', 8),
        s_attn_size=_get_arg(args, 's_attn_size', 3),
        t_attn_size=_get_arg(args, 't_attn_size', 1),
        geo_num_heads=_get_arg(args, 'geo_num_heads', 4),
        sem_num_heads=_get_arg(args, 'sem_num_heads', 2),
        t_num_heads=_get_arg(args, 't_num_heads', 2),
        enc_depth=_get_arg(args, 'enc_depth', 6),
        mlp_ratio=_get_arg(args, 'mlp_ratio', 4),
        qkv_bias=_get_arg(args, 'qkv_bias', True),
        drop=_get_arg(args, 'dropout', 0.0),
        attn_drop=_get_arg(args, 'attn_drop', 0.0),
        drop_path=_get_arg(args, 'drop_path', 0.3),
        adj_mx=adj_mx_np,
        lap_mx=lap_mx,
        pattern_keys=pattern_keys,
        geo_mask=geo_mask,
        sem_mask=sem_mask,
        device=device,
    ).to(device)


def _build_ha(args, num_nodes, adj_mx, device):
    """Build HA (Historical Average) baseline model."""
    model = HA(
        num_nodes=num_nodes,
        input_window=_get_arg(args, 'input_window', 24),
        output_window=_get_arg(args, 'output_window', 24),
        output_dim=_get_arg(args, 'output_dim', 1),
    )
    return model.to(device)


def _build_gru(args, num_nodes, adj_mx, device):
    """Build GRU baseline model."""
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
