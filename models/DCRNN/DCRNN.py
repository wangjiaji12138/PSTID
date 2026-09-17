"""
DCRNN: Diffused Convolutional Recurrent Neural Network
基于图扩散的卷积循环神经网络，用于时空交通预测。
参考: https://arxiv.org/abs/1707.01926
"""

from __future__ import annotations

import scipy.sparse as sp
from scipy.sparse import linalg
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from models.base import BaseModel


def calculate_normalized_laplacian(adj):
    """计算归一化拉普拉斯矩阵 L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2"""
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=None, undirected=True):
    """计算缩放的拉普拉斯矩阵"""
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        # 计算最大的特征值
        try:
            lambda_max, _ = linalg.eigsh(lap.asfptype(), 1, which='LM')
            lambda_max = lambda_max[0]
            # 确保lambda_max不是0或负数
            if lambda_max <= 0:
                lambda_max = 2.0
        except Exception:
            lambda_max = 2.0
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32)


def count_parameters(model):
    """计算模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GCONV(nn.Module):
    """图卷积层，使用Chebyshev多项式近似"""
    
    def __init__(self, num_nodes: int, max_diffusion_step: int, supports: list, device: torch.device,
                 input_dim: int, hid_dim: int, output_dim: int, bias_start: float = 0.0):
        super().__init__()
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._supports = supports
        self._device = device
        self._num_matrices = len(self._supports) * self._max_diffusion_step + 1
        self._output_dim = output_dim
        input_size = input_dim + hid_dim
        shape = (input_size * self._num_matrices, self._output_dim)
        self.weight = nn.Parameter(torch.empty(*shape, device=self._device))
        self.biases = nn.Parameter(torch.empty(self._output_dim, device=self._device))
        nn.init.xavier_normal_(self.weight)
        nn.init.constant_(self.biases, bias_start)

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, inputs: Tensor, state: Tensor) -> Tensor:
        """前向传播"""
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        x = torch.reshape(x, shape=[self._num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * self._num_matrices])
        x = torch.matmul(x, self.weight)
        x += self.biases
        return torch.reshape(x, [batch_size, self._num_nodes * self._output_dim])


class FC(nn.Module):
    """全连接层"""
    
    def __init__(self, num_nodes: int, device: torch.device, input_dim: int, hid_dim: int,
                 output_dim: int, bias_start: float = 0.0):
        super().__init__()
        self._num_nodes = num_nodes
        self._device = device
        self._output_dim = output_dim
        input_size = input_dim + hid_dim
        shape = (input_size, self._output_dim)
        self.weight = nn.Parameter(torch.empty(*shape, device=self._device))
        self.biases = nn.Parameter(torch.empty(self._output_dim, device=self._device))
        nn.init.xavier_normal_(self.weight)
        nn.init.constant_(self.biases, bias_start)

    def forward(self, inputs: Tensor, state: Tensor) -> Tensor:
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        value = torch.sigmoid(torch.matmul(inputs_and_state, self.weight))
        value += self.biases
        return torch.reshape(value, [batch_size, self._num_nodes * self._output_dim])


class DCGRUCell(nn.Module):
    """扩散卷积GRU单元"""
    
    def __init__(self, input_dim: int, num_units: int, adj_mx: np.ndarray,
                 max_diffusion_step: int, num_nodes: int, device: torch.device,
                 filter_type: str = "laplacian", use_gc_for_ru: bool = True):
        super().__init__()
        self._activation = torch.tanh if filter_type == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._device = device
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru

        supports = []
        if filter_type == "laplacian":
            supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None))
        else:
            supports.append(calculate_scaled_laplacian(adj_mx))
        
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support, self._device))

        if self._use_gc_for_ru:
            self._fn = GCONV(self._num_nodes, self._max_diffusion_step, self._supports, self._device,
                             input_dim=input_dim, hid_dim=self._num_units, output_dim=2*self._num_units, bias_start=1.0)
        else:
            self._fn = FC(self._num_nodes, self._device, input_dim=input_dim,
                          hid_dim=self._num_units, output_dim=2*self._num_units, bias_start=1.0)
        self._gconv = GCONV(self._num_nodes, self._max_diffusion_step, self._supports, self._device,
                            input_dim=input_dim, hid_dim=self._num_units, output_dim=self._num_units, bias_start=0.0)

    @staticmethod
    def _build_sparse_matrix(lap, device):
        lap = lap.tocoo()
        indices = np.column_stack((lap.row, lap.col))
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        lap = torch.sparse_coo_tensor(indices.T, lap.data, lap.shape, device=device)
        return lap

    def forward(self, inputs: Tensor, hx: Tensor) -> Tensor:
        """GRU前向传播"""
        output_size = 2 * self._num_units
        value = torch.sigmoid(self._fn(inputs, hx))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))

        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs, r * hx)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state


class Seq2SeqAttrs:
    """Seq2Seq模型属性"""
    
    def __init__(self, config, adj_mx):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(config.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(config.get('cl_decay_steps', 1000))
        self.filter_type = config.get('filter_type', 'laplacian')
        self.num_nodes = int(config.get('num_nodes', 1))
        self.num_rnn_layers = int(config.get('num_rnn_layers', 2))
        self.rnn_units = int(config.get('rnn_units', 64))
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.input_dim = config.get('feature_dim', 1)
        self.device = config.get('device', torch.device('cpu'))


class EncoderModel(nn.Module, Seq2SeqAttrs):
    """编码器模型"""
    
    def __init__(self, config, adj_mx):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, adj_mx)
        self.dcgru_layers = nn.ModuleList()
        self.dcgru_layers.append(DCGRUCell(self.input_dim, self.rnn_units, adj_mx, self.max_diffusion_step,
                                           self.num_nodes, self.device, filter_type=self.filter_type))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(DCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type))

    def forward(self, inputs: Tensor, hidden_state: Optional[Tensor] = None) -> tuple:
        """编码器前向传播"""
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size), device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        return output, torch.stack(hidden_states)


class DecoderModel(nn.Module, Seq2SeqAttrs):
    """解码器模型"""
    
    def __init__(self, config, adj_mx):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, adj_mx)
        self.output_dim = config.get('output_dim', 1)
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList()
        self.dcgru_layers.append(DCGRUCell(self.output_dim, self.rnn_units, adj_mx, self.max_diffusion_step,
                                           self.num_nodes, self.device, filter_type=self.filter_type))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(DCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type))

    def forward(self, inputs: Tensor, hidden_state: Optional[Tensor] = None) -> tuple:
        """解码器前向传播"""
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)
        return output, torch.stack(hidden_states)


class DCRNN(BaseModel):
    """扩散卷积循环神经网络
    
    使用扩散卷积GRU进行时空交通预测。
    数据格式: batch['X']: (B, T, N, F), batch['y']: (B, T, N, 1)
    """
    
    model_name = "dcrnn"

    def __init__(self, num_nodes: int, input_embedding_dim: int = 64, output_dim: int = 1,
                 input_window: int = 12, output_window: int = 12,
                 num_rnn_layers: int = 2, rnn_units: int = 64,
                 max_diffusion_step: int = 2, filter_type: str = "laplacian",
                 use_curriculum_learning: bool = False,
                 adj_mx: np.ndarray = None, device: torch.device = None):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.input_window = input_window
        self.output_window = output_window
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self.max_diffusion_step = max_diffusion_step
        self.filter_type = filter_type
        self.use_curriculum_learning = use_curriculum_learning
        self.cl_decay_steps = 1000
        self.device = device or torch.device('cpu')
        
        self.adj_mx = adj_mx if adj_mx is not None else np.eye(self.num_nodes)
        
        # 配置字典用于Seq2SeqAttrs
        config = {
            'max_diffusion_step': max_diffusion_step,
            'cl_decay_steps': self.cl_decay_steps,
            'filter_type': filter_type,
            'num_nodes': num_nodes,
            'num_rnn_layers': num_rnn_layers,
            'rnn_units': rnn_units,
            'feature_dim': 1,  # 只使用时序值
            'device': self.device,
            'output_dim': output_dim,
        }
        
        # 输入嵌入层
        self.input_proj = nn.Linear(1, rnn_units)  # 只使用时序特征
        
        # 编码器和解码器
        self.encoder_model = EncoderModel(config, self.adj_mx)
        self.decoder_model = DecoderModel(config, self.adj_mx)
        
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        pass

    def _compute_sampling_threshold(self, batches_seen: int) -> float:
        """计算课程学习采样阈值"""
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs: Tensor) -> Tensor:
        """编码器前向传播"""
        encoder_hidden_state = None
        for t in range(self.input_window):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)
        return encoder_hidden_state

    def decoder(self, encoder_hidden_state: Tensor, labels: Optional[Tensor] = None,
                batches_seen: Optional[int] = None) -> Tensor:
        """解码器前向传播"""
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.output_dim), device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []
        for t in range(self.output_window):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning and batches_seen is not None:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, batch: dict, batches_seen: Optional[int] = None) -> Tensor:
        """前向传播
        
        Args:
            batch: 包含 'X' 和 'y' 的字典
            batches_seen: 训练中已见过的batch数量（用于课程学习）
        Returns:
            (batch_size, output_window, num_nodes, output_dim)
        """
        inputs = batch['X']
        labels = batch.get('y')
        
        # 提取时序特征 (只使用第一个通道)
        time_series = inputs[..., :1]  # (B, T, N, 1)
        
        batch_size, _, num_nodes, input_dim = time_series.shape
        time_series = time_series.permute(1, 0, 2, 3)  # (T, B, N, 1)
        time_series = time_series.view(self.input_window, batch_size, num_nodes * input_dim).to(self.device)
        
        if labels is not None:
            labels = labels.permute(1, 0, 2, 3)
            labels = labels[..., :self.output_dim].contiguous().view(
                self.output_window, batch_size, num_nodes * self.output_dim).to(self.device)

        encoder_hidden_state = self.encoder(time_series)
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        
        outputs = outputs.view(self.output_window, batch_size, num_nodes, self.output_dim)
        return outputs.permute(1, 0, 2, 3)

    def predict(self, batch: dict) -> Tensor:
        """预测接口"""
        return self.forward(batch)

    @staticmethod
    def from_args(args, num_nodes, adj_mx, device):
        """从命令行参数创建模型实例"""
        return DCRNN(
            num_nodes=num_nodes,
            input_embedding_dim=args.input_embedding_dim,
            output_dim=1,
            input_window=args.input_window,
            output_window=args.output_window,
            num_rnn_layers=args.num_rnn_layers,
            rnn_units=args.rnn_units,
            max_diffusion_step=args.max_diffusion_step,
            filter_type=getattr(args, 'filter_type', 'laplacian'),
            use_curriculum_learning=getattr(args, 'use_curriculum_learning', False),
            adj_mx=adj_mx,
            device=device
        ).to(device)
