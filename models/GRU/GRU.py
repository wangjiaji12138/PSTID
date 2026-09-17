"""
GRU: 门控循环单元基线模型

使用标准的 GRU 进行时序预测，作为深度学习基线。
GRU 比 LSTM 参数量更少，训练速度更快。
"""

import torch
import torch.nn as nn
from models.base import BaseModel


class GRU(BaseModel):
    """GRU 时序预测模型
    
    Encoder-Decoder 结构：GRU 编码输入序列，线性层解码输出序列。
    """
    
    model_name = "gru"
    
    def __init__(self, num_nodes: int, input_window: int = 24, output_window: int = 24,
                 input_dim: int = 3, output_dim: int = 1,
                 input_embedding_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_window = input_window
        self.output_window = output_window
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.hidden_dim = input_embedding_dim * 2
        self.num_layers = num_layers
        
        # 输入投影：将 input_dim -> input_embedding_dim
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        
        # Encoder GRU
        self.encoder = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Decoder GRU (输出 input_embedding_dim)
        self.decoder = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # 输出层：input_embedding_dim -> output_dim
        self.output_proj = nn.Linear(self.hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def from_args(args, num_nodes, adj_mx, device):
        """从命令行参数创建 GRU 模型实例"""
        return GRU(
            num_nodes=num_nodes,
            input_window=getattr(args, 'input_window', 24),
            output_window=getattr(args, 'output_window', 24),
            input_dim=getattr(args, 'input_dim', 3),
            output_dim=getattr(args, 'output_dim', 1),
            input_embedding_dim=getattr(args, 'input_embedding_dim', 64),
            num_layers=getattr(args, 'num_layers', 2),
            dropout=getattr(args, 'dropout', 0.1),
        ).to(device)

    def forward(self, batch):
        """前向传播 - Encoder-Decoder 架构
        
        Args:
            batch: dict, 包含 'X' (B, T_in, N, C_in)
        
        Returns:
            predictions: (B, T_out, N, C_out)
        """
        X = batch['X']  # (B, T_in, N, C_in)
        B, T_in, N, C_in = X.shape
        
        # 重塑: (B, T_in, N, C_in) -> (B*T, T_in, C_in)
        X = X.view(B * N, T_in, C_in)
        
        # 输入投影
        X = self.input_proj(X)  # (B*T, T_in, self.hidden_dim)
        X = self.dropout(X)
        
        # Encoder: 编码输入序列
        _, hidden = self.encoder(X)  # hidden: (num_layers, B*T, self.hidden_dim)
        
        # Decoder: 解码输出序列
        # 初始化 decoder 输入为零
        decoder_input = torch.zeros(B * N, 1, self.hidden_dim, device=X.device)
        decoder_outputs = []
        
        for _ in range(self.output_window):
            # 一步解码
            output, hidden = self.decoder(decoder_input, hidden)
            # 预测
            pred = self.output_proj(output)  # (B*T, 1, output_dim)
            decoder_outputs.append(pred)
            # 下一个时间步的输入
            decoder_input = output
        
        # 合并所有时间步: (B*T, T_out, output_dim)
        predictions = torch.cat(decoder_outputs, dim=1)
        
        # 重塑回: (B*T, T_out, output_dim) -> (B, T_out, N, output_dim)
        predictions = predictions.view(B, N, self.output_window, self.output_dim)
        predictions = predictions.permute(0, 2, 1, 3)  # (B, T_out, N, output_dim)
        
        return predictions

    def predict(self, batch):
        """预测接口（同 forward）"""
        return self.forward(batch)
