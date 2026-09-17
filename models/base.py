"""
Base Model for PSTID Project
"""

from abc import ABC, abstractmethod
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """抽象模型基类，所有模型必须继承此类并实现 from_args 方法"""

    @staticmethod
    @abstractmethod
    def from_args(args, num_nodes, adj_mx, device):
        """从命令行参数创建模型实例"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回模型名称"""
        pass
