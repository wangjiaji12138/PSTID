"""Regression coverage for model, data, CLI and checkpoint compatibility.

Run with: python -m unittest discover -s tests -v
"""
import importlib
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
            with self.subTest(flags=case['flags']):
                torch.manual_seed(42)
                model = PROPHET(**model_kwargs(), **case['flags']).eval()
                self.assertEqual(list(model.state_dict()), case['state_keys'])
                output = model({'X': example_input()})
                torch.testing.assert_close(output, torch.tensor(case['prediction']), atol=1e-6, rtol=1e-5)
                output.square().mean().backward()
                gradient = model.time_series_emb_layer.weight.grad
                torch.testing.assert_close(gradient.flatten(), torch.tensor(case['input_gradient']), atol=1e-6, rtol=1e-5)

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
