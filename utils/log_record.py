"""
统一日志记录器
支持：模型架构、参数配置、训练损失和评估指标等
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import torch



class Logger:
    """统一日志记录器"""

    def __init__(self, log_dir: str, name: str = 'PROPHET'):
        self.log_dir = log_dir
        self.name = name
        self.logger = self._setup_logger()
        self.metrics_history = []

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, 'training.log')

        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        logger.handlers = []

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 阻止日志向上传播到 root logger，避免重复输出
        logger.propagate = False

        return logger

    def info(self, msg: str):
        """普通信息"""
        self.logger.info(msg)

    def error(self, msg: str):
        """错误信息"""
        self.logger.error(msg)

    def warning(self, msg: str):
        """警告信息"""
        self.logger.warning(msg)

    def section(self, title: str):
        """分隔标题"""
        separator = "=" * 60
        self.info(separator)
        self.info(f"  {title}")
        self.info(separator)

    def log_model_config(self, args):
        """记录详细的模型配置信息（与 DSTGN 格式一致）"""
        self.section("Model Configuration")

        config_str = f"""City: {getattr(args, 'data', 'N/A')}
Number of nodes: {getattr(args, 'num_nodes', 'N/A')}
Input dimension: {getattr(args, 'input_dim', 'N/A')}
Input length: {getattr(args, 'input_window', 'N/A')}
Output length: {getattr(args, 'output_window', 'N/A')}
Model dimension: {getattr(args, 'input_embedding_dim', 'N/A')}
Num layers: {getattr(args, 'num_layers', 'N/A')}
Dropout: {getattr(args, 'dropout', 'N/A')}
Learning rate: {getattr(args, 'learning_rate', 'N/A')}
Epochs: {getattr(args, 'epochs', 'N/A')}
Batch size: {getattr(args, 'batch_size', 'N/A')}
Device: {getattr(args, 'device', 'N/A')}
Seed: {getattr(args, 'seed', 'N/A')}
--------------------------------------------------------------------------------
Proto: {getattr(args, 'use_proto', 'N/A')} (spatio={getattr(args, 'use_spatio', 'N/A')}, temporal={getattr(args, 'use_temporal', 'N/A')}, his={getattr(args, 'use_his', 'N/A')})
Num spatial protos: {getattr(args, 'num_spatial_prototypes', 'N/A')}
Num temporal protos: {getattr(args, 'num_temporal_prototypes', 'N/A')}
Proto temperature: {getattr(args, 'proto_temperature', 'N/A')}
Proto transition weight: {getattr(args, 'proto_transition_weight', 'N/A')}"""
        self.info(config_str)

    def log_model_architecture(self, model):
        """记录模型架构详细信息"""
        self.section("Model Architecture")

        # --- 1. 基础信息 ---
        model_name = model.__class__.__name__
        num_nodes = getattr(model, 'num_nodes', 'N/A')
        model_dim = getattr(model, 'model_dim', 'N/A')
        num_layers = getattr(model, 'num_layers', 'N/A')

        self.info(f"  Model Class: {model_name}")
        self.info(f"  Graph Nodes: {num_nodes}")
        self.info(f"  Hidden Dim:  {model_dim}")
        self.info(f"  Num Layers:  {num_layers}")

        # --- 2. 总参数量统计 ---
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.info(f"  Total Params:     {total_params:,}")
        self.info(f"  Trainable Params:{trainable_params:,}")

        # --- 3. 完整模型结构 + 逐参数明细（与 DSTGN 一致）---
        self.info(str(model))
        self.info("")

    def log_epoch(self, epoch: int, total_epochs: int,
                  train_loss: float,
                  val_metrics: Dict[str, float],
                  model_losses: Optional[Dict[str, Any]] = None):
        """记录每个epoch的训练结果"""
        msg_parts = [f"Epoch {epoch}/{total_epochs}"]

        msg_parts.append(f"Train Loss: {train_loss:.4f}")
        msg_parts.append(f"Val MAE: {val_metrics.get('MAE', 0):.4f}")
        msg_parts.append(f"Val RMSE: {val_metrics.get('RMSE', 0):.4f}")
        msg_parts.append(f"Val MAPE: {val_metrics.get('MAPE', 0):.2f}%")

        if model_losses:
            if 'pred' in model_losses:
                msg_parts.append(f"| Pred Loss: {model_losses['pred']:.4f}")
            if 'contrastive' in model_losses:
                msg_parts.append(f"| Contra Loss: {model_losses['contrastive']:.4f}")

        self.info(" | ".join(msg_parts))


    def log_test_results(self, test_metrics: Dict[str, float]):
        """记录测试结果"""
        self.section("Test Results")
        self.info(f"  {'Metric':<10} {'Value':<15}")
        self.info(f"  {'-'*25}")
        self.info(f"  {'MAE':<10} {test_metrics.get('MAE', 0):<15.4f}")
        self.info(f"  {'RMSE':<10} {test_metrics.get('RMSE', 0):<15.4f}")
        self.info(f"  {'MAPE':<10} {test_metrics.get('MAPE', 0):<15.2f}%")

    def save_history(self, history: list):
        """保存训练历史"""
        history_path = os.path.join(self.log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        self.metrics_history = history

    def save_test_results(self, test_metrics: Dict[str, float]):
        """保存测试结果"""
        test_path = os.path.join(self.log_dir, 'test_results.json')
        with open(test_path, 'w') as f:
            json.dump({
                'MAE': float(test_metrics.get('MAE', 0)),
                'RMSE': float(test_metrics.get('RMSE', 0)),
                'MAPE': float(test_metrics.get('MAPE', 0))
            }, f, indent=2)


def create_logger(log_dir: str, name: str = 'PROPHET') -> Logger:
    """创建日志记录器"""
    return Logger(log_dir, name)
