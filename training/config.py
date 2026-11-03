"""training.config: extracted from the original training entry point."""

import argparse
from models.registry import MODEL_LIST, BASELINE_CONFIGS, resolve_model_args
from models.presets import MODEL_CONFIG_REGISTRY, model_defaults, get_model_config
from utils.paths import DATA_ROOT, RESULTS_ROOT


class ModelArgumentParser(argparse.ArgumentParser):
    """Parse runtime options; freeze baselines and allow research-model tuning."""

    def parse_known_args(self, args=None, namespace=None):
        selector = argparse.ArgumentParser(add_help=False)
        selector.add_argument('--model', type=str.upper, default='PROPHET')
        selected, _ = selector.parse_known_args(args)
        if selected.model not in MODEL_CONFIG_REGISTRY:
            self.error(f'unknown model: {selected.model}')
        if selected.model in BASELINE_CONFIGS:
            fixed = BASELINE_CONFIGS[selected.model]
            defaults = {**fixed['model'], **fixed['training']}
        else:
            defaults = model_defaults(selected.model)
        # Shared experiment settings must not inherit a model's upstream values.
        defaults.pop('epochs', None)
        defaults.pop('batch_size', None)
        # Restore defaults after each call so a parser can be reused safely.
        previous = {action.dest: action.default for action in self._actions}
        previous_defaults = self._defaults.copy()
        try:
            self.set_defaults(**defaults)
            result, unknown = super().parse_known_args(args, namespace)
        finally:
            for action in self._actions:
                action.default = previous[action.dest]
            self._defaults = previous_defaults
        if result.time_intervals <= 0 or 86400 % result.time_intervals:
            self.error('--time_intervals must be a positive divisor of 86400')
        result = argparse.Namespace(**vars(resolve_model_args(result.model, result)))
        result.hyperparameter_policy = ('fixed_baseline' if result.model in BASELINE_CONFIGS
                                        else 'tunable')
        config = get_model_config(result.model)
        result.registry_provenance = {
            key: config[key] for key in ('status', 'repository', 'commit', 'profile', 'sources', 'notes')
        }
        return result, unknown


def create_parser():
    """Create argument parser."""
    parser = ModelArgumentParser(description='Model-specific training (official registry defaults)')

    # ============================================================
    # 1) Basic
    # ============================================================
    parser.add_argument('--seed', type=int, default=88)
    parser.add_argument('--data', type=str.lower, default='cq',
                        help='数据集名称，如 cq、hz、sh、jl、yt、chi_taxi、nyc_taxi')
    parser.add_argument('--data_dir', type=str, default=str(DATA_ROOT / 'processed'))
    parser.add_argument('--model', type=str.upper, choices=MODEL_LIST, default='PROPHET')

    # ============================================================
    # 2) Training
    # ============================================================
    parser.add_argument('--max_epoch', '--epochs', '--max_epochs', '--epoch',
                        dest='max_epoch', type=int, default=100,
                        help='统一训练轮数上限，内部字段为 max_epoch')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--early_stop_patience', type=int, default=15)
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[20, 40, 50, 60, 70])
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--clip_grad', type=float, default=5.0)

    # ============================================================
    # 3) Device
    # ============================================================
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu', type=int, default=2)

    # ============================================================
    # 4) Model architecture
    # ============================================================
    parser.add_argument('--input_window', type=int, default=24)
    parser.add_argument('--output_window', type=int, default=24)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--input_embedding_dim', type=int, default=32)
    parser.add_argument('--node_emb_dim', type=int, default=16)
    parser.add_argument('--proto_emb_dim', type=int, default=64)
    parser.add_argument('--adj_mx_emb_dim', type=int, default=16)
    parser.add_argument('--use_adj', type=int, default=1)
    parser.add_argument('--tid', type=int, default=16)
    parser.add_argument('--diw', type=int, default=16)
    parser.add_argument('--time_intervals', type=int, default=1800)
    parser.add_argument('--tcn_kernel_sizes', type=int, nargs='+', default=[3])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_teacher_forcing', type=int, choices=[0, 1], default=0,
                        help='HimNet: enable scheduled sampling during training')
    parser.add_argument('--tf_decay_steps', type=int, default=2,
                        help='HimNet: scheduled sampling decay in training batches')

    # ============================================================
    # 5) Prototype codebooks
    # ============================================================
    # 5a) Codebook sizes
    parser.add_argument('--num_spatial_prototypes', type=int, default=16)
    parser.add_argument('--num_temporal_prototypes', type=int, default=8)

    # 5b) Prototype temperature
    parser.add_argument('--proto_temperature', type=float, default=0.5)

    # ============================================================
    # 6) Ablation switches (0=off, 1=on)
    # ============================================================
    parser.add_argument('--use_proto', type=int, choices=[0, 1], default=1,
                        help='原型模块开关（0=embeddings 直接输入 MLP，1=启用原型模块）')
    # 区分直接输入 MLP 与旧版线性替换等 use_proto=0 实验。
    parser.set_defaults(proto_ablation='direct_embeddings_mlp')
    parser.add_argument('--use_spatio', type=int, default=1)
    parser.add_argument('--use_temporal', type=int, default=1)
    parser.add_argument('--use_time_interaction', type=int, choices=[0, 1], default=1,
                        help='PROPHET/PSTID: condition future prototypes on current calendar and prototype assignment')
    parser.add_argument('--use_proto_change', type=int, choices=[0, 1], default=1,
                        help='PROPHET/PSTID: state-conditioned future-minus-current prototype readout')
    parser.add_argument('--proto_change_context', choices=['raw', 'encoded'], default='encoded')
    parser.add_argument('--use_his', type=int, choices=[0, 1], default=1,
                        help='原型分支是否使用时间序列嵌入（0=仅身份嵌入，1=包含历史序列）')

    parser.add_argument('--results_dir', type=str, default=str(RESULTS_ROOT), help='实验结果根目录')
    # Expose model-only fields without duplicating their defaults in train.
    known = {action.dest for action in parser._actions}
    for config in MODEL_CONFIG_REGISTRY.values():
        for key, value in {**config['model'], **config['training']}.items():
            if key in known or key in ('epochs', 'max_epochs', 'epoch'):
                continue
            kwargs = {'default': argparse.SUPPRESS}
            if isinstance(value, bool):
                kwargs.update(type=int, choices=[0, 1])
            elif isinstance(value, list):
                kwargs.update(type=int, nargs='*')
            else:
                kwargs['type'] = type(value)
            choices = {'optimizer': ['adam', 'rmsprop'],
                       'lr_scheduler': ['none', 'multistep', 'step'],
                       'train_loss': ['mae', 'mse']}
            if key in choices:
                kwargs['choices'] = choices[key]
            parser.add_argument('--' + key, **kwargs)
            known.add(key)
    return parser
