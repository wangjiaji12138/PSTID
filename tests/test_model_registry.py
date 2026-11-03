"""Regression coverage for upstream defaults and explicit override precedence."""
from types import SimpleNamespace

import pytest
import torch

from models.presets import get_model_config, resolve_model_args
from models.registry import MODEL_LIST, MODEL_CONFIG_REGISTRY, BASELINE_CONFIGS, build_model
from training.config import create_parser
from training.optim import build_optimizer


def test_catalog_covers_factories_with_pinned_sources():
    assert set(MODEL_LIST) == set(MODEL_CONFIG_REGISTRY)
    for name in MODEL_LIST:
        config = get_model_config(name)
        if config['status'] == 'official_adapted':
            assert len(config['commit']) == 40
            assert config['sources'] and config['official_config']
            assert all('/blob/' + config['commit'] + '/' in url for url in config['sources'])


def test_parser_baselines_ignore_hyperparameter_overrides():
    parser = create_parser()
    agcrn = parser.parse_args(['--model', 'agcrn'])
    assert (agcrn.learning_rate, agcrn.num_layers, agcrn.input_embedding_dim) == (.003, 2, 64)
    explicit = parser.parse_args(['--model', 'AGCRN', '--learning_rate', '.001',
                                 '--num_layers', '3', '--input_embedding_dim', '32'])
    assert (explicit.learning_rate, explicit.num_layers, explicit.input_embedding_dim) == (.003, 2, 64)
    stae = parser.parse_args(['--model=STAEformer'])
    assert (stae.input_embedding_dim, stae.adaptive_embedding_dim, stae.batch_size) == (24, 80, 64)
    assert parser.parse_args([]).learning_rate == .001
    assert parser.parse_args(['--model', 'AGCRN']).learning_rate == .003


def test_direct_builder_and_calendar_resolution_do_not_mutate_input():
    original = SimpleNamespace(time_intervals=3600)
    resolved = resolve_model_args('STAEFORMER', original)
    assert resolved.input_embedding_dim == 24
    assert resolved.steps_per_day == 24
    assert vars(original) == {'time_intervals': 3600}
    config = get_model_config('STAEFORMER')
    config['model']['input_embedding_dim'] = 999
    assert get_model_config('STAEFORMER')['model']['input_embedding_dim'] == 24
    # Baseline settings also override direct dictionary arguments.
    args = resolve_model_args('STID', {'input_embedding_dim': 16})
    assert args.input_embedding_dim == 32


@pytest.mark.parametrize('name,lr_after_ten', [('AGCRN', .003), ('STDN', .0009),
                                            ('STGCN', .00049)])
def test_training_schedule_is_model_specific(name, lr_after_ten):
    args = create_parser().parse_args(['--model', name])
    model = torch.nn.Linear(1, 1)
    optimizer, scheduler = build_optimizer(model, args)
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(lr_after_ten)
    if name == 'STGCN':
        assert isinstance(optimizer, torch.optim.RMSprop)


@pytest.mark.parametrize('name', MODEL_LIST)
def test_every_preset_constructs(name):
    args = create_parser().parse_args(['--model', name])
    model = build_model(name, args, 32, torch.eye(32), torch.device('cpu'))
    assert isinstance(model, torch.nn.Module)
    if name == 'HIMNET':
        assert args.use_teacher_forcing and args.tf_decay_steps == 4000
    elif name == 'MTGNN':
        assert (args.conv_channels, args.residual_channels, args.skip_channels,
                args.end_channels) == (32, 32, 64, 128)
    elif name == 'DCRNN':
        assert model.cl_decay_steps == 2000
        assert model.use_curriculum_learning


@pytest.mark.parametrize('window', [12, 24])
def test_gwnet_keeps_official_layers_with_local_windows(window):
    model = build_model('GWNET', {'input_window': window, 'output_window': 6},
                        4, torch.eye(4), torch.device('cpu')).eval()
    assert model.layers == 2
    assert len(model.filter_convs) == 8
    with torch.no_grad():
        result = model({'X': torch.rand(2, window, 4, 3)})
    assert result.shape == (2, 6, 4, 1)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize('name', ['PSTID', 'PROPHET'])
def test_research_models_remain_tunable(name):
    args = create_parser().parse_args(['--model', name, '--learning_rate', '.02',
                                      '--num_layers', '5', '--input_embedding_dim', '48'])
    assert (args.learning_rate, args.num_layers, args.input_embedding_dim) == (.02, 5, 48)
    assert args.hyperparameter_policy == 'tunable'


@pytest.mark.parametrize('name', sorted(BASELINE_CONFIGS))
def test_all_baseline_settings_are_fixed_and_runtime_options_survive(name):
    config = BASELINE_CONFIGS[name]
    expected = {**config['model'], **config['training']}
    supplied = {key: 'must not survive' for key in expected}
    supplied.update(input_window=12, output_window=6, seed=42, gpu=4, data='hz',
                    time_intervals=3600, steps_per_day=999, min_learning_rate=99)
    resolved = resolve_model_args(name, supplied)
    for key, value in expected.items():
        assert getattr(resolved, key) == value
    assert (resolved.input_window, resolved.output_window, resolved.seed,
            resolved.gpu, resolved.data) == (12, 6, 42, 4, 'hz')
    assert resolved.steps_per_day == 24
    assert resolved.min_learning_rate == config['training'].get('min_learning_rate', 0)


def test_factory_ignores_baseline_args_even_when_called_directly():
    model = build_model('AGCRN', {'input_embedding_dim': 8, 'num_layers': 9},
                        4, None, torch.device('cpu'))
    reference = build_model('AGCRN', {}, 4, None, torch.device('cpu'))
    assert {k: v.shape for k, v in model.state_dict().items()} == {
        k: v.shape for k, v in reference.state_dict().items()}


@pytest.mark.parametrize('name', MODEL_LIST)
@pytest.mark.parametrize('epoch_flag', ['--max_epoch', '--epochs', '--max_epochs', '--epoch'])
def test_shared_training_controls_override_every_model(name, epoch_flag):
    args = create_parser().parse_args(['--model', name, epoch_flag, '7',
                                      '--batch_size', '11'])
    assert args.max_epoch == 7
    assert args.batch_size == 11
    assert not hasattr(args, 'max_epochs')
    assert not hasattr(args, 'epochs')
    resolved = resolve_model_args(name, args)
    assert (resolved.max_epoch, resolved.batch_size) == (7, 11)
    if name in BASELINE_CONFIGS:
        assert 'epochs' not in BASELINE_CONFIGS[name]['training']
        assert 'batch_size' not in BASELINE_CONFIGS[name]['training']


@pytest.mark.parametrize('name', MODEL_LIST)
def test_shared_training_defaults_and_direct_alias(name):
    args = create_parser().parse_args(['--model', name])
    assert (args.max_epoch, args.batch_size) == (100, 64)
    direct = resolve_model_args(name, {'max_epochs': 3, 'batch_size': 9})
    assert (direct.max_epoch, direct.batch_size) == (3, 9)
    assert not hasattr(direct, 'max_epochs')
    assert not hasattr(direct, 'epochs')


def test_conflicting_epoch_names_are_rejected_for_direct_calls():
    with pytest.raises(ValueError, match='must agree'):
        resolve_model_args('AGCRN', {'epochs': 3, 'max_epochs': 5})


def test_canonical_max_epoch_survives_resolution():
    args = resolve_model_args('PSTID', {'max_epoch': 12})
    assert args.max_epoch == 12
    assert not hasattr(args, 'epochs')
