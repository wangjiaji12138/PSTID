"""Inspect trained prototype routing without changing or retraining the model.

Run: python -m experiments.diagnose_prototypes EXPERIMENT_DIR --output report.json
Affine surrogates and constant replacements are fitted on training batches only.
All reported errors and frozen-weight interventions use the complete test split.
"""
import argparse
import json
import math
import types
from pathlib import Path

import torch
import torch.nn.functional as F

from data_pipeline.dataset import StandardScaler, load_dataset
from data_pipeline.normalization import normalize_input
from models.PROPHET.model import PROPHET
from utils.metrics import compute_all_metrics_torch


def historical_model(config, nodes):
    """Reconstruct the archived residual architecture for old checkpoint analysis only."""
    if config.get('representation_path') == 'encoded_only_v1':
        raise ValueError('This historical diagnostic only supports checkpoints with the old residual paths.')
    model = PROPHET.from_args(config, nodes, None, torch.device('cpu')).eval()
    del model._encoded_only_version  # Historical checkpoints predate the topology marker.
    if model.use_proto:
        del model.proto_module._hard_assignment_version
    temporal = model.proto_module.temporal_basis
    spatial = model.proto_module.spatial_codebook
    td = (model.time_series_emb_dim if model.use_his else 0) + model.temp_dim_tid + model.temp_dim_diw
    temporal.proj = torch.nn.Sequential(torch.nn.Linear(td, td * 2), torch.nn.GELU(),
                                        torch.nn.Linear(td * 2, model.proto_emb_dim))
    if not model.use_proto:
        sd = (model.adj_mx_emb_dim if model.use_adj else model.spatial_emb_dim)
        sd += model.time_series_emb_dim if model.use_his else 0
        spatial.proj = torch.nn.Linear(sd, model.proto_emb_dim)
    model.forward = types.MethodType(historical_forward, model)
    return model.eval()


def historical_forward(model, batch):
    hidden, s, t = components(model, batch['X'])
    return forecast(model, hidden, s['z'], t['z'], s['q'], t['q'])


def rms(x):
    return x.square().mean().sqrt().item()


def with_bias(x):
    return torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)


def affine_fit(x, y):
    # Standardization and SVD handle redundant temporal/identity coordinates.
    mean = x.double().mean(0)
    scale = x.double().std(0).clamp_min(1e-8)
    a = with_bias((x.double() - mean) / scale)
    weight = torch.linalg.lstsq(a, y.double(), driver='gelsd').solution
    return mean, scale, weight


def affine_predict(fit, x):
    mean, scale, weight = fit
    return (with_bias((x.double() - mean) / scale) @ weight).float()


def r2(target, prediction):
    denominator = (target - target.mean(0)).square().sum()
    return (1 - (target - prediction).square().sum() / denominator.clamp_min(1e-20)).item()


def components(model, x):
    hist, tid, diw, node, adj = model._get_embeddings({'X': x})
    b, steps = hist.shape[:2]
    parts = [hist, tid, diw, node[None, None].expand(b, steps, -1, -1)]
    if model.use_adj:
        parts.append(adj[None, None].expand(b, steps, -1, -1))
    hidden = torch.cat(parts, -1)
    module = model.proto_module
    spatial, temporal = module.spatial_codebook, module.temporal_basis
    xs = adj if model.use_adj else node
    if model.use_his:
        xs = torch.cat([xs, hist.mean(1).mean(0)], dim=-1)
    xt = torch.cat(([hist[:, -1]] if model.use_his else []) + [tid[:, -1], diw[:, -1]], -1)
    qs, qt = spatial.proj(xs), temporal.proj(xt)
    ls = lt = None
    if model.use_proto:
        ls = F.normalize(qs, dim=-1) @ F.normalize(spatial.prototypes, dim=-1).T / spatial.temperature
        lt = temporal.allocator(xt) / temporal.temp
        # Preserve the soft routing of the archived checkpoints, independently
        # of the current production encoder's hard assignment implementation.
        zs = ls.softmax(-1) @ spatial.prototypes
        zt = lt.softmax(-1) @ temporal.prototypes
    else:
        zs, zt = spatial.linear_encoder(xs), temporal.linear_encoder(xt)
    # Current model broadcasts every embedding and branch output across T.
    return hidden[:, :1], {'x': xs, 'q': qs, 'z': zs, 'logits': ls}, {
        'x': xt, 'q': qt, 'z': zt, 'logits': lt}


def forecast(model, hidden, zs, zt, qs, qt):
    enhanced = model.proto_module.proto_out_proj(zs + zt + qs + qt).unsqueeze(1)
    return model.regression_layer(model.encoder((hidden + enhanced).permute(0, 3, 2, 1)))


@torch.no_grad()
def diagnose(experiment, train_batches=16, linear_experiment=None):
    checkpoint = torch.load(experiment / 'checkpoint/best_model.pth', map_location='cpu', weights_only=False)
    config = checkpoint['config']
    if not all(config.get(k, True) for k in ['use_proto', 'use_spatio', 'use_temporal']):
        raise ValueError('Diagnosis requires both prototype branches enabled.')
    dataset = load_dataset(config['data_dir'], config['data'], seed=config['seed'])
    nodes = dataset['test']['X'].shape[2]
    model = historical_model(config, nodes)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    scaler = StandardScaler(checkpoint['scaler_mean'], checkpoint['scaler_std'])
    batch_size = config['batch_size']
    count = math.ceil(len(dataset['train']['X']) / batch_size)
    selected = torch.linspace(0, count - 1, min(train_batches, count)).round().long().tolist()
    train = {b: {k: [] for k in ['x', 'z']} for b in ['spatial', 'temporal']}
    static_spatial = []
    for index in selected:
        x = torch.from_numpy(dataset['train']['X'][index * batch_size:(index + 1) * batch_size]).float()
        _, s, t = components(model, normalize_input(x, scaler))
        static_spatial.append(s['z'])
        for name, values in [('spatial', s), ('temporal', t)]:
            for key in train[name]:
                train[name][key].append(values[key].reshape(-1, values[key].shape[-1]))
    train = {b: {k: torch.cat(v) for k, v in d.items()} for b, d in train.items()}
    fits = {b: affine_fit(v['x'], v['z']) for b, v in train.items()}
    constants = {'spatial': torch.stack(static_spatial).mean(0),
                 'temporal': train['temporal']['z'].mean(0)}
    modes = ['original', 'zero_spatial_prototype', 'zero_temporal_prototype',
             'zero_both_prototypes', 'zero_both_queries', 'mean_prototypes',
             'affine_spatial', 'affine_temporal', 'affine_both', 'uniform_prototypes']
    predictions = {name: [] for name in modes}
    stats = {b: {k: [] for k in ['x', 'q', 'z', 'logits']} for b in train}
    max_reconstruction_error = 0.0
    for start in range(0, len(dataset['test']['X']), batch_size):
        x = torch.from_numpy(dataset['test']['X'][start:start + batch_size]).float()
        x = normalize_input(x, scaler)
        hidden, s, t = components(model, x)
        for name, values in [('spatial', s), ('temporal', t)]:
            for key in stats[name]:
                stats[name][key].append(values[key].reshape(-1, values[key].shape[-1]))
        sa, ta = affine_predict(fits['spatial'], s['x']), affine_predict(fits['temporal'], t['x'])
        zs, zt, qs, qt = s['z'], t['z'], s['q'], t['q']
        variants = {
            'original': (zs, zt, qs, qt),
            'zero_spatial_prototype': (torch.zeros_like(zs), zt, qs, qt),
            'zero_temporal_prototype': (zs, torch.zeros_like(zt), qs, qt),
            'zero_both_prototypes': (torch.zeros_like(zs), torch.zeros_like(zt), qs, qt),
            'zero_both_queries': (zs, zt, torch.zeros_like(qs), torch.zeros_like(qt)),
            'mean_prototypes': (constants['spatial'], constants['temporal'], qs, qt),
            'affine_spatial': (sa, zt, qs, qt),
            'affine_temporal': (zs, ta, qs, qt),
            'affine_both': (sa, ta, qs, qt),
            'uniform_prototypes': (model.proto_module.spatial_codebook.prototypes.mean(0),
                                   model.proto_module.temporal_basis.prototypes.mean(0), qs, qt),
        }
        for name, values in variants.items():
            pred = forecast(model, hidden, *values)
            if name == 'original' and start == 0:
                original = model({'X': x})
                torch.testing.assert_close(pred, original, rtol=1e-5, atol=1e-6)
                max_reconstruction_error = (pred - original).abs().max().item()
            predictions[name].append(scaler.inverse_transform(pred))
    target = torch.from_numpy(dataset['test']['y']).float()
    predictions = {k: torch.cat(v) for k, v in predictions.items()}
    intervention = {}
    for name, pred in predictions.items():
        metrics = compute_all_metrics_torch(pred, target, null_value=-1.0)
        intervention[name] = dict(zip(['MAE', 'RMSE', 'MAPE', 'R2', 'SMAPE'], metrics))
        intervention[name]['mean_abs_prediction_change'] = (pred - predictions['original']).abs().mean().item()
    original_metrics = json.loads((experiment / 'test_results.json').read_text())
    for metric in ['MAE', 'RMSE']:
        # Saved runs used CUDA; allow small CPU/CUDA numerical differences.
        if abs(intervention['original'][metric] - original_metrics[metric]) > 1e-3 * max(1., abs(original_metrics[metric])):
            raise AssertionError(f'Original {metric}: CPU={intervention["original"][metric]}, saved={original_metrics[metric]}')
    branches = {}
    out_weight = model.proto_module.proto_out_proj.weight
    for name, values in stats.items():
        d = {k: torch.cat(v) for k, v in values.items()}
        branch = model.proto_module.spatial_codebook if name == 'spatial' else model.proto_module.temporal_basis
        p = branch.prototypes
        alpha = d['logits'].softmax(-1)
        entropy = -(alpha * alpha.clamp_min(1e-20).log()).sum(-1)
        k = p.shape[0]
        centered_logits = d['logits'] - d['logits'].mean(-1, keepdim=True)
        first_order = (1 / k + centered_logits / k) @ p
        affine = affine_predict(fits[name], d['x'])
        zc = d['z'] - d['z'].mean(0)
        qc = d['q'] - d['q'].mean(0)
        cos = F.normalize(p, dim=-1) @ F.normalize(p, dim=-1).T
        branches[name] = {
            'num_prototypes': k,
            'temperature': float(branch.temperature if name == 'spatial' else branch.temp),
            'normalized_entropy_mean': (entropy.mean() / math.log(k)).item(),
            'top1_probability_mean': alpha.max(-1).values.mean().item(),
            'mean_assignment': alpha.mean(0).tolist(),
            'assignment_std_mean': alpha.std(0).mean().item(),
            'centered_logits_rms': rms(centered_logits),
            'prototype_pair_cosine_mean': cos[~torch.eye(k, dtype=torch.bool)].mean().item(),
            'prototype_rms': rms(d['z']), 'query_rms': rms(d['q']),
            'centered_prototype_rms': rms(zc), 'centered_query_rms': rms(qc),
            'projected_centered_prototype_rms': rms(zc @ out_weight.T),
            'projected_centered_query_rms': rms(qc @ out_weight.T),
            'affine_surrogate_test_r2': r2(d['z'], affine),
            'affine_surrogate_relative_rmse': rms(d['z'] - affine) / max(rms(zc), 1e-20),
            'uniform_softmax_taylor_r2': r2(d['z'], first_order),
        }
    comparison = None
    if linear_experiment is not None:
        other = torch.load(linear_experiment / 'checkpoint/best_model.pth', map_location='cpu', weights_only=False)
        if other['config'].get('proto_encoder_type') != 'linear_mlp':
            raise ValueError('Comparison checkpoint must use linear_mlp.')
        allowed = {'use_proto', 'proto_encoder_type', 'results_dir', 'log_dir', 'checkpoint_dir', 'current_epoch'}
        differences = {k: [config.get(k), other['config'].get(k)]
                       for k in config.keys() | other['config'].keys()
                       if config.get(k) != other['config'].get(k)}
        if set(differences) - allowed:
            raise ValueError(f'Unexpected comparison configuration differences: {differences}')
        linear = historical_model(other['config'], nodes)
        linear.load_state_dict(other['model_state_dict'], strict=True)
        other_scaler = StandardScaler(other['scaler_mean'], other['scaler_std'])
        outputs = []
        for start in range(0, len(target), batch_size):
            x = torch.from_numpy(dataset['test']['X'][start:start + batch_size]).float()
            outputs.append(other_scaler.inverse_transform(linear({'X': normalize_input(x, other_scaler)})))
        pred = torch.cat(outputs)
        metrics = compute_all_metrics_torch(pred, target, null_value=-1.0)
        comparison = {
            'experiment': str(linear_experiment), 'configuration_differences': differences,
            'test_metrics': dict(zip(['MAE', 'RMSE', 'MAPE', 'R2', 'SMAPE'], metrics)),
            'mean_abs_prediction_difference': (pred - predictions['original']).abs().mean().item(),
            'prediction_correlation': torch.corrcoef(torch.stack([pred.flatten(), predictions['original'].flatten()]))[0, 1].item(),
            'prototype_model_parameters': sum(p.numel() for p in model.parameters()),
            'linear_model_parameters': sum(p.numel() for p in linear.parameters()),
        }
    return {'experiment': str(experiment), 'seed': config['seed'],
            'saved_test_metrics': original_metrics,
            'train_batches_for_surrogates': len(selected), 'test_samples': len(target),
            'batch_size': batch_size, 'reconstruction_max_error': max_reconstruction_error,
            'notes': ['CPU float32 inference; no checkpoint updates or retraining.',
                      'Affine fits use only evenly spaced training batches, float64 SVD least squares.',
                      'Mean replacement is per-node for spatial, global for temporal.',
                      'Spatial routing statistics count node-batch pairs; temporal counts sample-node pairs.',
                      'Frozen-weight interventions measure reliance, not retrained ablation performance.'],
            'branches': branches, 'frozen_weight_interventions': intervention,
            'trained_linear_comparison': comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-batches', type=int, default=16)
    parser.add_argument('--linear-experiment', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = diagnose(args.experiment, args.train_batches, args.linear_experiment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
