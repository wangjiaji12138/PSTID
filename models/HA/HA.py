"""
HA: 历史平均基线模型

使用过去 144 个时间步的平均值作为预测结果。
"""

import torch
import torch.nn as nn
from models.base import BaseModel


class HA(BaseModel):
    """历史平均法 (Historical Average)

    使用输入 X 的最后 144 个时间步的平均值作为预测值。
    """
    
    model_name = "ha"

    def __init__(self, num_nodes: int, input_window: int = 24, output_window: int = 24,
                 output_dim: int = 1, lookback_window: int = 144, **kwargs):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_window = input_window
        self.output_window = output_window
        self.output_dim = output_dim
        self.lookback_window = lookback_window  # 使用的历史时间步数

    @staticmethod
    def from_args(args, num_nodes, adj_mx, device):
        """从命令行参数创建 HA 模型实例"""
        return HA(
            num_nodes=num_nodes,
            input_window=getattr(args, 'input_window', 24),
            output_window=getattr(args, 'output_window', 24),
            output_dim=getattr(args, 'output_dim', 1),
            lookback_window=getattr(args, 'lookback_window', 144),
        ).to(device)

    def forward(self, batch):
        """前向传播：使用过去 144 个时间步的平均值作为预测
        
        Args:
            batch: dict, 包含 'X' 和 'y'
        
        Returns:
            predictions: (B, T_out, N, C_out) 使用历史平均
        """
        X = batch['X']  # (B, T_in, N, C_in)
        B, T_in, N, C_in = X.shape
        
        # 取最后 lookback_window 个时间步
        lookback = min(self.lookback_window, T_in)
        historical = X[:, -lookback:, :, 0:self.output_dim]  # (B, lookback, N, C_out)
        
        # 对时间维度求平均
        predictions = historical.mean(dim=1, keepdim=True)  # (B, 1, N, C_out)
        
        # 扩展到 output_window
        predictions = predictions.expand(-1, self.output_window, -1, -1)  # (B, T_out, N, C_out)
        
        return predictions

    def predict(self, batch):
        """预测接口（同 forward）"""
        return self.forward(batch)
