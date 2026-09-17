"""
PDFormer: Spatial-Temporal Transformer for Traffic Prediction

Reference: https://github.com/AsiKur/PDFormer
Paper: PDFormer: Propagation Delay-aware Dynamic Long-range Transformer for Traffic Forecasting

This is an adapted version for the unified training framework.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def drop_path(x, drop_prob=0., training=False):
    """Drop path (stochastic depth) operation."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class TokenEmbedding(nn.Module):
    """Token embedding with linear projection and optional normalization."""
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()


class LaplacianPE(nn.Module):
    """Laplacian Positional Encoding from graph structure."""
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc


class DataEmbedding(nn.Module):
    """Complete data embedding module for PDFormer."""
    def __init__(
        self, feature_dim, embed_dim, lape_dim, adj_mx=None, drop=0.,
        add_time_in_day=True, add_day_in_week=True, device=torch.device('cpu'),
    ):
        super().__init__()
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.device = device
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        
        # Value embedding
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)
        
        # Position encoding
        self.position_encoding = PositionalEncoding(embed_dim)
        
        # Time-of-day embedding
        if self.add_time_in_day:
            self.minute_size = 48  # 30-min intervals per day
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)
        
        # Day-of-week embedding
        if self.add_day_in_week:
            weekday_size = 7
            self.weekday_embedding = nn.Embedding(weekday_size, embed_dim)
        
        # Spatial embedding (Laplacian PE)
        self.spatial_embedding = LaplacianPE(lape_dim, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, lap_mx=None):
        """
        Args:
            x: (B, T, N, feature_dim) - feature_dim includes value + time features
            lap_mx: (N, lape_dim) Laplacian PE matrix
        Returns:
            embedded: (B, T, N, embed_dim)
        """
        origin_x = x
        # Embed values (first feature_dim channels)
        x = self.value_embedding(origin_x[..., :self.feature_dim])
        # Add positional encoding
        x = x + self.position_encoding(x)
        
        # Add time-of-day encoding
        if self.add_time_in_day:
            # x[..., self.feature_dim] is normalized time in day (0-1)
            time_in_day = (origin_x[..., self.feature_dim] * self.minute_size).round().long()
            time_in_day = torch.clamp(time_in_day, 0, self.minute_size - 1)
            x = x + self.daytime_embedding(time_in_day)
        
        # Add day-of-week encoding
        if self.add_day_in_week:
            # x[..., self.feature_dim + 1] is day of week (0-6)
            day_of_week = origin_x[..., self.feature_dim + 1].long()
            day_of_week = torch.clamp(day_of_week, 0, 6)
            x = x + self.weekday_embedding(day_of_week)
        
        # Add Laplacian PE from graph structure
        if lap_mx is not None:
            x = x + self.spatial_embedding(lap_mx)
        
        x = self.dropout(x)
        return x


class DropPath(nn.Module):
    """Drop path (stochastic depth) module."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Chomp2d(nn.Module):
    """Chomp along time dimension."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :x.shape[2] - self.chomp_size, :].contiguous()


class STSelfAttention(nn.Module):
    """
    Spatial-Temporal Self-Attention with three parallel attention branches:
    1. Geographic attention (based on graph structure)
    2. Semantic attention (based on DTW similarity)
    3. Temporal attention (along time dimension)
    """
    def __init__(
        self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2,
        qkv_bias=False, attn_drop=0., proj_drop=0., device=torch.device('cpu'), output_dim=1,
    ):
        super().__init__()
        assert dim % (geo_num_heads + sem_num_heads + t_num_heads) == 0
        self.geo_num_heads = geo_num_heads
        self.sem_num_heads = sem_num_heads
        self.t_num_heads = t_num_heads
        self.head_dim = dim // (geo_num_heads + sem_num_heads + t_num_heads)
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.s_attn_size = s_attn_size
        self.t_attn_size = t_attn_size
        self.geo_ratio = geo_num_heads / (geo_num_heads + sem_num_heads + t_num_heads)
        self.sem_ratio = sem_num_heads / (geo_num_heads + sem_num_heads + t_num_heads)
        self.t_ratio = 1 - self.geo_ratio - self.sem_ratio
        self.output_dim = output_dim

        # Pattern attention projections (for geographic attention enhancement)
        self.pattern_q_linears = nn.ModuleList([
            nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)
        ])
        self.pattern_k_linears = nn.ModuleList([
            nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)
        ])
        self.pattern_v_linears = nn.ModuleList([
            nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)
        ])

        # Geographic attention conv projections
        self.geo_q_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_k_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_v_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_attn_drop = nn.Dropout(attn_drop)

        # Semantic attention conv projections
        self.sem_q_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_k_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_v_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_attn_drop = nn.Dropout(attn_drop)

        # Temporal attention conv projections
        self.t_q_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_patterns, pattern_keys, geo_mask=None, sem_mask=None):
        """
        Args:
            x: (B, T, N, D) input tensor
            x_patterns: (B, T, N, s_attn_size, output_dim) spatial patterns
            pattern_keys: (n_cluster, s_attn_size, output_dim) pattern centroids
            geo_mask: (N, N) boolean mask for geographic attention
            sem_mask: (N, N) boolean mask for semantic attention
        Returns:
            output: (B, T, N, D) transformed tensor
        """
        B, T, N, D = x.shape
        
        # ===== Temporal Attention =====
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, int(D * self.t_ratio)).transpose(1, 2)

        # ===== Geographic Attention =====
        geo_q = self.geo_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_k = self.geo_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # Enhance geographic key with pattern attention
        for i in range(self.output_dim):
            pattern_q = self.pattern_q_linears[i](x_patterns[..., i])
            pattern_k = self.pattern_k_linears[i](pattern_keys[..., i])
            pattern_v = self.pattern_v_linears[i](pattern_keys[..., i])
            pattern_attn = (pattern_q @ pattern_k.transpose(-2, -1)) * self.scale
            pattern_attn = pattern_attn.softmax(dim=-1)
            geo_k += pattern_attn @ pattern_v
        geo_v = self.geo_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_q = geo_q.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_k = geo_k.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_v = geo_v.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale
        # Apply geographic mask (mask out far nodes)
        if geo_mask is not None:
            # geo_attn: (B, T, H, N, N); geo_mask: (N, N) - broadcast over B, T, H
            geo_attn.masked_fill_(geo_mask.to(x.device), float('-inf'))
        geo_attn = geo_attn.softmax(dim=-1)
        geo_attn = self.geo_attn_drop(geo_attn)
        geo_x = (geo_attn @ geo_v).transpose(2, 3).reshape(B, T, N, int(D * self.geo_ratio))

        # ===== Semantic Attention =====
        sem_q = self.sem_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_k = self.sem_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_v = self.sem_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_q = sem_q.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_k = sem_k.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_v = sem_v.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_attn = (sem_q @ sem_k.transpose(-2, -1)) * self.scale
        # Apply semantic mask (mask out dissimilar nodes based on DTW)
        if sem_mask is not None:
            # sem_attn: (B, T, H, N, N); sem_mask: (N, N) - broadcast over B, T, H
            sem_attn.masked_fill_(sem_mask.to(x.device), float('-inf'))
        sem_attn = sem_attn.softmax(dim=-1)
        sem_attn = self.sem_attn_drop(sem_attn)
        sem_x = (sem_attn @ sem_v).transpose(2, 3).reshape(B, T, N, int(D * self.sem_ratio))

        # ===== Combine all attention outputs =====
        x = self.proj(torch.cat([t_x, geo_x, sem_x], dim=-1))
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP with GELU activation."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TemporalSelfAttention(nn.Module):
    """Temporal self-attention layer (simplified for single temporal branch)."""
    def __init__(
        self, dim, dim_out, t_attn_size, t_num_heads=6, qkv_bias=False,
        attn_drop=0., proj_drop=0., device=torch.device('cpu'),
    ):
        super().__init__()
        assert dim % t_num_heads == 0
        self.t_num_heads = t_num_heads
        self.head_dim = dim // t_num_heads
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.t_attn_size = t_attn_size

        self.t_q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, N, D = x.shape
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, D).transpose(1, 2)
        x = self.proj(t_x)
        x = self.proj_drop(x)
        return x


class STEncoderBlock(nn.Module):
    """Spatial-Temporal Encoder Block with ST Self-Attention and MLP."""
    def __init__(
        self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2,
        mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
        act_layer=nn.GELU, norm_layer=nn.LayerNorm, device=torch.device('cpu'),
        type_ln="pre", output_dim=1,
    ):
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = norm_layer(dim)
        self.st_attn = STSelfAttention(
            dim, s_attn_size, t_attn_size, geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads,
            t_num_heads=t_num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            device=device, output_dim=output_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_patterns, pattern_keys, geo_mask=None, sem_mask=None):
        if self.type_ln == 'pre':
            x = x + self.drop_path(self.st_attn(self.norm1(x), x_patterns, pattern_keys, geo_mask, sem_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            x = self.norm1(x + self.drop_path(self.st_attn(x, x_patterns, pattern_keys, geo_mask, sem_mask)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


class PDFormer(nn.Module):
    """
    PDFormer: Propagation Delay-aware Dynamic Long-range Transformer
    
    Key components:
    - DataEmbedding: positional, temporal, and spatial embeddings
    - STEncoderBlocks: stacked spatial-temporal transformer blocks
    - Pattern-based attention enhancement
    - Skip connections with projection
    
    Args:
        num_nodes: Number of nodes in the graph
        feature_dim: Input feature dimension (default 3 for 3D input)
        output_dim: Output dimension (default 1)
        in_window: Input sequence length
        out_window: Output sequence length
        embed_dim: Embedding dimension
        lape_dim: Laplacian PE dimension
        s_attn_size: Spatial attention window size
        t_attn_size: Temporal attention window size
        geo_num_heads: Number of geographic attention heads
        sem_num_heads: Number of semantic attention heads
        t_num_heads: Number of temporal attention heads
        enc_depth: Number of encoder blocks
        mlp_ratio: MLP hidden dim ratio
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Drop path rate
        adj_mx: Adjacency matrix (for preprocessing)
        lap_mx: Laplacian PE matrix (precomputed)
        pattern_keys: Pattern centroids (precomputed)
        geo_mask: Geographic mask (precomputed)
        sem_mask: Semantic mask (precomputed)
        device: Device
    """
    
    def __init__(
        self, num_nodes, feature_dim=3, output_dim=1, in_window=24, out_window=24,
        embed_dim=64, lape_dim=8, s_attn_size=3, t_attn_size=1,
        geo_num_heads=4, sem_num_heads=2, t_num_heads=2, enc_depth=6,
        mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.3,
        adj_mx=None, lap_mx=None, pattern_keys=None, geo_mask=None, sem_mask=None,
        device=torch.device('cpu'),
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.in_window = in_window
        self.out_window = out_window
        self.embed_dim = embed_dim
        self.device = device
        self.s_attn_size = s_attn_size
        
        # Store preprocessed data as buffers
        if lap_mx is not None:
            self.register_buffer('lap_mx', torch.from_numpy(lap_mx).float())
        else:
            self.lap_mx = None
            
        if pattern_keys is not None:
            self.register_buffer('pattern_keys', torch.from_numpy(pattern_keys).float())
        else:
            self.pattern_keys = None
            
        if geo_mask is not None:
            self.register_buffer('geo_mask', torch.from_numpy(geo_mask).bool())
        else:
            self.geo_mask = None
            
        if sem_mask is not None:
            self.register_buffer('sem_mask', torch.from_numpy(sem_mask).bool())
        else:
            self.sem_mask = None
        
        # Pattern embeddings for spatial attention
        self.pattern_embeddings = nn.ModuleList([
            TokenEmbedding(s_attn_size, embed_dim) for _ in range(output_dim)
        ])
        
        # Data embedding
        self.enc_embed_layer = DataEmbedding(
            feature_dim, embed_dim, lape_dim, adj_mx, drop=drop,
            add_time_in_day=True, add_day_in_week=True, device=device,
        )
        
        # Encoder blocks
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]
        self.encoder_blocks = nn.ModuleList([
            STEncoderBlock(
                dim=embed_dim, s_attn_size=s_attn_size, t_attn_size=t_attn_size,
                geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads, t_num_heads=t_num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=enc_dpr[i], act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), device=device, type_ln="post",
                output_dim=output_dim,
            ) for i in range(enc_depth)
        ])
        
        # Skip connections
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(in_channels=embed_dim, out_channels=256, kernel_size=1)
            for _ in range(enc_depth)
        ])
        
        # Output projection
        self.end_conv1 = nn.Conv2d(
            in_channels=in_window, out_channels=out_window, kernel_size=1, bias=True,
        )
        self.end_conv2 = nn.Conv2d(
            in_channels=256, out_channels=output_dim, kernel_size=1, bias=True,
        )

    def forward(self, batch):
        """
        Forward pass for PDFormer.
        
        Args:
            batch: dict with keys:
                - 'X': (B, T, N, feature_dim) input tensor with 3D features
                - optionally: 'lap_mx', 'pattern_keys', 'geo_mask', 'sem_mask'
        
        Returns:
            output: (B, T_out, N, output_dim) prediction
        """
        x = batch['X']  # (B, T, N, feature_dim)
        T = x.shape[1]
        
        # Get masks (from buffer or batch)
        geo_mask = batch.get('geo_mask', self.geo_mask)
        sem_mask = batch.get('sem_mask', self.sem_mask)
        lap_mx = batch.get('lap_mx', self.lap_mx)
        pattern_keys = batch.get('pattern_keys', self.pattern_keys)
        
        # Create spatial patterns for each time step
        x_pattern_list = []
        for i in range(self.s_attn_size):
            # Extract spatial patterns from input
            x_pattern = F.pad(
                x[:, :T + i + 1 - self.s_attn_size, :, :self.output_dim],
                (0, 0, 0, 0, self.s_attn_size - 1 - i, 0),
                "constant", 0,
            ).unsqueeze(-2)
            x_pattern_list.append(x_pattern)
        x_patterns = torch.cat(x_pattern_list, dim=-2)  # (B, T, N, s_attn_size, output_dim)
        
        # Process patterns through embedding
        x_pattern_processed_list = []
        pattern_key_processed_list = []
        for i in range(self.output_dim):
            x_pattern_processed_list.append(
                self.pattern_embeddings[i](x_patterns[..., i]).unsqueeze(-1)
            )
            pattern_key_processed_list.append(
                self.pattern_embeddings[i](pattern_keys[..., i]).unsqueeze(-1)
            )
        x_patterns = torch.cat(x_pattern_processed_list, dim=-1)  # (B, T, N, embed_dim, output_dim)
        pattern_keys_processed = torch.cat(pattern_key_processed_list, dim=-1)  # (n_cluster, embed_dim, output_dim)
        
        # Data embedding
        enc = self.enc_embed_layer(x, lap_mx)  # (B, T, N, embed_dim)
        
        # Encoder blocks with skip connections
        skip = 0
        for i, encoder_block in enumerate(self.encoder_blocks):
            enc = encoder_block(
                enc, x_patterns, pattern_keys_processed,
                geo_mask=geo_mask, sem_mask=sem_mask
            )
            skip = skip + self.skip_convs[i](enc.permute(0, 3, 2, 1))
        
        # Output projection
        skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))
        
        output = skip.permute(0, 3, 2, 1)  # (B, T_out, N, output_dim)
        
        return output
    
    def get_loss_func(self, set_loss='masked_mae'):
        """Return loss function for training."""
        if set_loss.lower() == 'mae':
            return self._masked_mae_loss
        elif set_loss.lower() == 'mse':
            return self._masked_mse_loss
        elif set_loss.lower() == 'huber':
            return partial(self._huber_loss, delta=1.0)
        else:
            return self._masked_mae_loss
    
    @staticmethod
    def _masked_mae_loss(preds, labels, null_val=0.):
        """Masked MAE loss."""
        mask = (labels.abs() > null_val).float()
        mask = mask / mask.mean()
        loss = (preds - labels).abs() * mask
        return loss.mean()
    
    @staticmethod
    def _masked_mse_loss(preds, labels, null_val=0.):
        """Masked MSE loss."""
        mask = (labels.abs() > null_val).float()
        mask = mask / mask.mean()
        loss = ((preds - labels) ** 2) * mask
        return loss.mean()
    
    @staticmethod
    def _huber_loss(preds, labels, delta=1.0, null_val=0.):
        """Huber loss."""
        mask = (labels.abs() > null_val).float()
        mask = mask / mask.mean()
        diff = (preds - labels).abs()
        loss = torch.where(diff < delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta)) * mask
        return loss.mean()
