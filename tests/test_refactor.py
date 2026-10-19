"""Regression coverage for model, data, CLI and checkpoint compatibility.

Run with: python -m unittest discover -s tests -v
"""
import importlib
import itertools
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from data_pipeline.dataset import StandardScaler, load_dataset
from data_pipeline.normalization import normalize_input
from data_pipeline.npz import create_sliding_windows
from models.registry import build_model
from models.PROPHET.model import PROPHET
from training.config import create_parser

ROOT = Path(__file__).resolve().parents[1]


def model_kwargs():
    return dict(num_nodes=6, input_window=4, output_window=2, num_block=1,
                time_series_emb_dim=8, spatial_emb_dim=4, temp_dim_tid=4,
                temp_dim_diw=4, proto_emb_dim=8, adj_mx_emb_dim=4,
                num_spatial_prototypes=4, num_temporal_prototypes=4,
                adj_mx=torch.eye(6))


def example_input():
    x = torch.arange(2 * 4 * 6 * 3, dtype=torch.float32).reshape(2, 4, 6, 3)
    x[..., 0] = x[..., 0] / 100
    x[..., 1] = (x[..., 1] % 48) / 48
    x[..., 2] = (x[..., 2] % 7) / 7
    return x


class RefactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pre_refactor_forecasts_and_gradients(self):
        fixture = json.loads((ROOT / 'tests/fixtures/prophet_reference.json').read_text())
        for case in fixture['cases']:
            # 旧版 use_proto=False 删除整个分支；新语义由 Linear 对照测试覆盖。
            if not case['flags'].get('use_proto', True):
                continue
            with self.subTest(flags=case['flags']):
                torch.manual_seed(42)
                model = PROPHET(**model_kwargs(), **case['flags']).eval()
                self.assertEqual(list(model.state_dict()), case['state_keys'])
                output = model({'X': example_input()})
                torch.testing.assert_close(output, torch.tensor(case['prediction']), atol=1e-6, rtol=1e-5)
                output.square().mean().backward()
                gradient = model.time_series_emb_layer.weight.grad
                torch.testing.assert_close(gradient.flatten(), torch.tensor(case['input_gradient']), atol=1e-6, rtol=1e-5)

    def test_linear_ablation_trains_both_branches_and_roundtrips(self):
        for use_adj, use_his in itertools.product((False, True), repeat=2):
            with self.subTest(use_adj=use_adj, use_his=use_his):
                torch.manual_seed(42)
                kwargs = dict(model_kwargs(), num_spatial_prototypes=3, num_temporal_prototypes=5)
                model = PROPHET(**kwargs, use_proto=False,
                                use_adj=use_adj, use_his=use_his).eval()
                spatial = model.proto_module.spatial_codebook
                temporal = model.proto_module.temporal_basis
                for branch, count in ((spatial, 3), (temporal, 5)):
                    self.assertIsInstance(branch.linear[0], torch.nn.Linear)
                    self.assertIsInstance(branch.linear[1], torch.nn.Softmax)
                    self.assertIsInstance(branch.linear[2], torch.nn.Linear)
                    self.assertEqual(branch.linear[0].out_features, count)
                    self.assertEqual(branch.linear[2].in_features, count)
                    self.assertEqual(branch.linear[2].out_features, model.proto_emb_dim)
                self.assertFalse(hasattr(spatial, 'prototypes'))
                for name in ('prototypes', 'allocator', 'temp'):
                    self.assertFalse(hasattr(temporal, name))
                self.assertEqual(spatial.linear[0].in_features, model.proto_emb_dim)
                self.assertEqual(temporal.linear[0].in_features, 8 + (8 if use_his else 0))
                assignments = []
                handles = [branch.linear[1].register_forward_hook(
                    lambda module, inputs, output: assignments.append(output.detach())
                ) for branch in (spatial, temporal)]
                output = model({'X': example_input()})
                for handle in handles:
                    handle.remove()
                self.assertEqual(len(assignments), 2)
                for weights, count in zip(assignments, (3, 5)):
                    self.assertEqual(weights.shape[-1], count)
                    self.assertTrue((weights >= 0).all())
                    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]))
                self.assertEqual(output.shape, (2, 2, 6, 1))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                for layer in (spatial.linear[0], spatial.linear[2],
                              temporal.linear[0], temporal.linear[2], spatial.proj,
                              model.proto_module.proto_out_proj):
                    self.assertTrue(torch.isfinite(layer.weight.grad).all())
                    self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
                for parameter in temporal.proj.parameters():
                    self.assertIsNone(parameter.grad)
                restored = PROPHET(**kwargs, use_proto=False,
                                   use_adj=use_adj, use_his=use_his).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                torch.testing.assert_close(restored({'X': example_input()}), output)

    def test_query_residuals_only_affect_prototype_mode(self):
        def corrupt_spatial_query(module, inputs, output):
            return dict(output, proj_feat=torch.full_like(output['proj_feat'], float('nan')))

        def corrupt_temporal_query(module, inputs, output):
            return output[0], torch.full_like(output[1], float('nan'))

        for use_proto, use_spatio, use_temporal in itertools.product((False, True), repeat=3):
            if not (use_spatio or use_temporal):
                continue
            with self.subTest(use_proto=use_proto, use_spatio=use_spatio, use_temporal=use_temporal):
                model = PROPHET(**model_kwargs(), use_proto=use_proto,
                                use_spatio=use_spatio, use_temporal=use_temporal).eval()
                expected = model({'X': example_input()})
                handles = []
                if use_spatio:
                    handles.append(model.proto_module.spatial_codebook.register_forward_hook(corrupt_spatial_query))
                if use_temporal:
                    handles.append(model.proto_module.temporal_basis.register_forward_hook(corrupt_temporal_query))
                try:
                    actual = model({'X': example_input()})
                finally:
                    for handle in handles:
                        handle.remove()
                if use_proto:
                    self.assertFalse(torch.isfinite(actual).all())
                else:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_linear_ablation_preserves_history_exclusion(self):
        for use_adj in (False, True):
            model = PROPHET(**model_kwargs(), use_proto=False,
                            use_adj=use_adj, use_his=False).eval()
            _, history, tid, diw, node, adj = model._get_embeddings({'X': example_input()})
            clean = model.proto_module(history, adj, node, tid, diw)
            invalid = model.proto_module(torch.full_like(history, float('nan')), adj, node, tid, diw)
            torch.testing.assert_close(clean, invalid, rtol=0, atol=0)

    def test_linear_ablation_config_and_legacy_results(self):
        from unittest.mock import patch
        from experiments import ablation
        from training.checkpoint import save_results

        args = create_parser().parse_args(['--use_proto', '0'])
        self.assertEqual(args.proto_ablation, 'linear_softmax_linear_no_query_residual')
        self.assertEqual(ablation.ABL_LABELS[ablation.identify_abl_type(vars(args))], 'Linear-Softmax-Linear')
        model = PROPHET.from_args(args, 6, torch.eye(6), torch.device('cpu'))
        self.assertFalse(model.proto_module.spatial_codebook.use_proto)
        self.assertFalse(model.proto_module.temporal_basis.use_proto)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'sample'
            base.mkdir()
            new_dir = base / 'PROPHET_20261019_120000'
            new_dir.mkdir()
            save_results(new_dir, {}, {'MAE': 1.0}, args)
            saved = json.loads((new_dir / 'config.json').read_text())
            self.assertEqual(saved['proto_ablation'], 'linear_softmax_linear_no_query_residual')
            # 即使旧实验的时间戳更晚，也不能覆盖 Linear 对照。
            for name, config in (
                ('PSTID_20261020_120000', {'use_proto': False}),
                ('PROPHET_20261021_120000', {'use_proto': False, 'proto_ablation': 'linear'}),
                ('PROPHET_20261022_120000', {'use_proto': False, 'proto_ablation': 'linear_softmax_linear'}),
            ):
                old_dir = base / name
                old_dir.mkdir()
                (old_dir / 'config.json').write_text(json.dumps(config))
                (old_dir / 'test_results.json').write_text(json.dumps({'MAE': 2.0}))
            with patch.object(ablation, 'RESULTS_ROOT', Path(tmp)):
                result = ablation.load_results('sample')
            self.assertEqual(result['minus_proto']['exp_dir'], new_dir.name)

    def test_forward_has_no_diagnostic_hooks_or_caches(self):
        model = PROPHET(**model_kwargs())
        output = model({'X': example_input()})
        output.square().mean().backward()
        for name in ('collect_visualization_data', 'get_proto_usage_summary',
                     '_proto_vis_accumulator', '_last_proto_enhanced_first_batch',
                     '_last_batch_size', '_cached_adj_mx'):
            self.assertFalse(hasattr(model, name), name)
        for name in ('_cache_proto_vis_data', '_compute_proto_usage', '_last_pa',
                     '_last_proto_assigns', '_last_proto_usage'):
            self.assertFalse(hasattr(model.proto_module, name), name)
        # Prototype enhancement no longer shadows recursive PyTorch initialization.
        visited = []
        model.apply(lambda module: visited.append(module))
        self.assertIn(model.proto_module, visited)

    def test_prototype_history_ablation_is_independent_of_history(self):
        for use_adj in (True, False):
            with self.subTest(use_adj=use_adj):
                model = PROPHET(**model_kwargs(), use_his=False, use_adj=use_adj).eval()
                _, history, tid, diw, node, adj = model._get_embeddings({'X': example_input()})
                history = history.detach().requires_grad_()
                spatial = model.proto_module.spatial_codebook
                temporal = model.proto_module.temporal_basis
                self.assertEqual(spatial.proj.in_features, 4)
                self.assertEqual(temporal.allocator[0].in_features, 8)
                self.assertEqual(temporal.proj[0].in_features, 8)

                def features(hist):
                    return (*spatial(hist, adj, node).values(), *temporal(hist, tid, diw))

                original = features(history)
                # Even invalid history values must not enter either prototype path.
                altered = features(torch.full_like(history, float('nan')))
                for before, after in zip(original, altered):
                    torch.testing.assert_close(before, after, rtol=0, atol=0)
                sum(feature.square().mean() for feature in original).backward()
                self.assertIsNone(history.grad)
                self.assertGreater(model.time_in_day_emb.grad.abs().sum().item(), 0)
                self.assertGreater(model.day_in_week_emb.grad.abs().sum().item(), 0)
                spatial_embedding = model.adj_mx_emb if use_adj else model.node_emb
                self.assertGreater(spatial_embedding.grad.abs().sum().item(), 0)

    def test_history_ablation_with_other_switches_preserves_backbone_gradients(self):
        for use_adj, use_proto, use_spatio, use_temporal in itertools.product((False, True), repeat=4):
            flags = dict(use_adj=use_adj, use_proto=use_proto,
                         use_spatio=use_spatio, use_temporal=use_temporal)
            with self.subTest(**flags):
                model = PROPHET(**model_kwargs(), use_his=False, **flags).eval()
                output = model({'X': example_input()})
                self.assertEqual(output.shape, (2, 2, 6, 1))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                gradient = model.time_series_emb_layer.weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0)

    def test_history_ablation_cli_factory_and_result_label(self):
        from experiments.ablation import identify_abl_type

        parser = create_parser()
        self.assertTrue(parser.parse_args([]).use_his)
        args = parser.parse_args(['--use_his', '0'])
        for config in (args, vars(args)):
            model = PROPHET.from_args(config, 6, torch.eye(6), torch.device('cpu'))
            self.assertFalse(model.use_his)
            self.assertFalse(model.proto_module.spatial_codebook.use_his)
            self.assertFalse(model.proto_module.temporal_basis.use_his)
        self.assertEqual(identify_abl_type(vars(args)), 'minus_his')
        self.assertEqual(identify_abl_type({}), 'full')
        self.assertEqual(identify_abl_type({'use_his': True}), 'full')
        self.assertTrue(PROPHET.from_args({}, 6, None, torch.device('cpu')).use_his)

    def test_legacy_import_order_and_factory_mapping(self):
        importlib.import_module('models.PROPHET.PROPHET')
        from models.PROPHET import PROPHET as package_class
        from models.PROPHET.PROPHET import PROPHET as legacy_class
        self.assertIs(package_class, PROPHET)
        self.assertIs(legacy_class, PROPHET)
        args = create_parser().parse_args(['--model', 'prophet', '--use_adj', '0'])
        torch.manual_seed(17)
        factory = build_model(args.model, args, 6, torch.eye(6), torch.device('cpu'))
        torch.manual_seed(17)
        direct = PROPHET.from_args(args, 6, torch.eye(6), torch.device('cpu'))
        self.assertFalse(factory.use_adj)
        for name, parameter in factory.state_dict().items():
            torch.testing.assert_close(parameter, direct.state_dict()[name], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            build_model('missing-model', args, 6, None, torch.device('cpu'))

    def test_training_import_has_no_file_side_effects(self):
        script = '''
import sys
import train
from models import build_model
assert 'matplotlib' not in sys.modules
assert 'sklearn.manifold' not in sys.modules
assert 'models.STSSL.STSSL' not in sys.modules
'''
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
            subprocess.run([sys.executable, '-c', script], cwd=tmp, env=env, check=True, capture_output=True)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_windows_split_and_shared_normalization(self):
        raw = np.arange(2 * 20, dtype=np.float32).reshape(2, 20, 1)
        x, y = create_sliding_windows(raw, 4, 2)
        self.assertEqual(x.shape, (15, 4, 2, 3))
        np.testing.assert_array_equal(x[0, :, :, 0], raw[:, :4, 0].T)
        np.testing.assert_array_equal(y[0, :, :, 0], raw[:, 4:6, 0].T)
        scaler = StandardScaler(mean=10., std=2.)
        normalized = normalize_input(torch.from_numpy(x), scaler)
        torch.testing.assert_close(normalized[..., 1:], torch.from_numpy(x[..., 1:]), rtol=0, atol=0)
        torch.testing.assert_close(scaler.inverse_transform(normalized[..., :1]), torch.from_numpy(x[..., :1]))
        np.testing.assert_allclose(normalize_input(x, scaler), normalized.numpy())
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / 'sample'
            ds.mkdir()
            np.savez(ds / 'all.npz', X=x, y=y)
            with (ds / 'adj.pkl').open('wb') as stream:
                pickle.dump(np.eye(2, dtype=np.float32), stream)
            splits = load_dataset(tmp, 'sample')
            np.testing.assert_array_equal(splits['train']['X'], x[:9])
            np.testing.assert_array_equal(splits['val']['X'], x[9:12])
            np.testing.assert_array_equal(splits['test']['X'], x[12:])

    def test_training_cli_checkpoint_reload_and_metrics(self):
        self._check_training_cli_checkpoint_reload_and_metrics()

    def test_training_without_history_cli_checkpoint_reload_and_metrics(self):
        self._check_training_cli_checkpoint_reload_and_metrics(use_his=False)

    def test_training_linear_cli_checkpoint_reload_and_metrics(self):
        self._check_training_cli_checkpoint_reload_and_metrics(use_proto=False)

    def _check_training_cli_checkpoint_reload_and_metrics(self, use_his=True, use_proto=True):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ds = tmp / 'data' / 'sample'
            ds.mkdir(parents=True)
            rng = np.random.default_rng(9)
            x, y = create_sliding_windows(rng.uniform(1, 20, (6, 40, 1)).astype('float32'), 4, 2)
            np.savez(ds / 'all.npz', X=x, y=y)
            with (ds / 'adj.pkl').open('wb') as stream:
                pickle.dump(np.eye(6, dtype=np.float32), stream)
            command = [sys.executable, str(ROOT / 'train.py'), '--model', 'prophet',
                       '--data', 'sample', '--data_dir', str(tmp / 'data'),
                       '--results_dir', str(tmp / 'results'), '--device', 'cpu',
                       '--epochs', '1', '--batch_size', '8', '--input_window', '4',
                       '--output_window', '2', '--input_embedding_dim', '8',
                       '--node_emb_dim', '4', '--proto_emb_dim', '8', '--adj_mx_emb_dim', '4',
                       '--tid', '4', '--diw', '4', '--num_layers', '1',
                       '--num_spatial_prototypes', '4', '--num_temporal_prototypes', '4']
            if not use_his:
                command.extend(['--use_his', '0'])
            if not use_proto:
                command.extend(['--use_proto', '0'])
            guard = """
import importlib.abc
import runpy
import sys
class BlockPlottingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'matplotlib', 'seaborn', 'plotly', 'visualization'}:
            raise AssertionError(f'Training attempted to import {fullname}')
sys.meta_path.insert(0, BlockPlottingImports())
sys.argv = sys.argv[1:]
sys.path.insert(0, str(__import__('pathlib').Path(sys.argv[0]).parent))
runpy.run_path(sys.argv[0], run_name='__main__')
"""
            command = [command[0], '-c', guard, *command[1:]]
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            result = subprocess.run(command, cwd=tmp, env=env, text=True, capture_output=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            checkpoints = list((tmp / 'results').rglob('best_model.pth'))
            self.assertEqual(len(checkpoints), 1)
            checkpoint = torch.load(checkpoints[0], map_location='cpu', weights_only=False)
            self.assertIs(checkpoint['config']['use_his'], use_his)
            self.assertIs(checkpoint['config']['use_proto'], use_proto)
            if not use_proto:
                self.assertEqual(checkpoint['config']['proto_ablation'], 'linear_softmax_linear_no_query_residual')
            args = create_parser().parse_args([])
            vars(args).update(checkpoint['config'])
            model = build_model('PROPHET', args, 6, torch.eye(6), torch.device('cpu')).eval()
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            from data_pipeline.loaders import build_dataloaders
            from training.engine import evaluate
            from utils.metrics import mae_torch
            dataset = load_dataset(str(tmp / 'data'), 'sample', seed=args.seed)
            _, _, loader, *_ = build_dataloaders(dataset, args.batch_size, args.seed)
            scaler = StandardScaler(checkpoint['scaler_mean'], checkpoint['scaler_std'])
            batch = next(iter(loader))
            with torch.no_grad():
                predictions = model({'X': normalize_input(batch['X'], scaler)})
            self.assertEqual(predictions.shape, (7, 2, 6, 1))
            restored_metrics = evaluate(model, loader, mae_torch, torch.device('cpu'), scaler)
            self.assertFalse(any((tmp / 'results').rglob('*.png')))
            self.assertFalse(any((tmp / 'results').rglob('*.npy')))
            metrics = json.loads((checkpoints[0].parent.parent / 'test_results.json').read_text())
            for metric in ('MAE', 'RMSE', 'MAPE', 'R2', 'SMAPE'):
                self.assertTrue(np.isfinite(metrics[metric]), metric)
                self.assertAlmostEqual(metrics[metric], restored_metrics[metric], places=5)


if __name__ == '__main__':
    unittest.main()
