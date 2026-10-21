"""Read-only probes for the current residual, soft-prototype PROPHET.

python -m experiments.audit_prophet_validation --output docs/prophet_optimization/validation_audit.json
Uses eight evenly spaced validation batches per dataset, seed 42 checkpoints.
Frozen interventions diagnose reliance; they are not retrained model comparisons.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data_pipeline.dataset import StandardScaler
from data_pipeline.normalization import normalize_input
from models.PROPHET.model import PROPHET
from utils.metrics import compute_all_metrics_torch


RUNS = [
    'results_42/cq/PROPHET_20261010_045203',
    'results_42/hz/PROPHET_20261010_083451',
    'results_42/nyc_taxi/PROPHET_20261011_200251',
    'results_42/chi_taxi/PROPHET_20261012_063407',
    'results/nyc_bike/PROPHET_20261021_132813',
]


def rms(x):
    return x.square().mean().sqrt().item()


def metrics(pred, target):
    return dict(zip(['MAE', 'RMSE', 'MAPE', 'R2', 'SMAPE'],
                    compute_all_metrics_torch(pred, target, null_value=-1.0)))


@torch.no_grad()
def audit(run, batches):
    checkpoint_path = Path(run) / 'checkpoint/best_model.pth'
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg, state = ck['config'], ck['model_state_dict']
    nodes = state['node_emb'].shape[0]
    model = PROPHET.from_args(cfg, nodes, None, torch.device('cpu')).eval()
    model.load_state_dict(state, strict=True)
    scaler = StandardScaler(ck['scaler_mean'], ck['scaler_std'])
    with np.load(Path(cfg['data_dir']) / cfg['data'] / 'all.npz') as data:
        full_x, full_y = data['X'], data['y']
    n = len(full_x)
    start, count = round(n * .6), round(n * .2)
    # Own compact arrays so that training and test arrays can be released.
    xv = full_x[start:start + count].copy()
    yv = full_y[start:start + count].copy()
    del full_x, full_y
    bs = cfg['batch_size']
    selected = np.unique(np.linspace(0, math.ceil(count / bs) - 1,
                                     min(batches, math.ceil(count / bs))).round().astype(int))
    module = model.proto_module
    spatial, temporal = module.spatial_codebook, module.temporal_basis
    modes = ['original', 'uniform_spatial', 'uniform_temporal',
             'zero_spatial_retrieval', 'zero_temporal_retrieval',
             'zero_both_retrievals', 'zero_both_queries']
    preds, targets = {k: [] for k in modes}, []
    stats = {b: {k: [] for k in ['a', 'q', 'r']} for b in ['spatial', 'temporal']}
    sensitivity = []
    reconstruction_error = 0.
    for batch_idx in selected:
        sl = slice(batch_idx * bs, min((batch_idx + 1) * bs, count))
        x = normalize_input(torch.from_numpy(xv[sl]).float(), scaler)
        target = torch.from_numpy(yv[sl]).float()
        hidden, hist, tid, diw, node, adj = model._get_embeddings({'X': x})
        xs = adj if model.use_adj else node
        if model.use_his:
            xs = torch.cat([xs, hist.mean(1).mean(0)], -1)
        xt = torch.cat(([hist[:, -1]] if model.use_his else []) + [tid[:, -1], diw[:, -1]], -1)
        qs, qt = spatial.proj(xs), temporal.proj(xt)
        a_s = (F.normalize(qs, dim=-1) @ F.normalize(spatial.prototypes, dim=-1).T
               / spatial.temperature).softmax(-1)
        a_t = (temporal.allocator(xt) / temporal.temp).softmax(-1)
        rs, rt = a_s @ spatial.prototypes, a_t @ temporal.prototypes
        for branch, a, q, r in [('spatial', a_s, qs, rs), ('temporal', a_t, qt, rt)]:
            for key, value in [('a', a), ('q', q), ('r', r)]:
                stats[branch][key].append(value.reshape(-1, value.shape[-1]))
        variants = {
            'original': (rs, rt, qs, qt),
            'uniform_spatial': (spatial.prototypes.mean(0), rt, qs, qt),
            'uniform_temporal': (rs, temporal.prototypes.mean(0), qs, qt),
            'zero_spatial_retrieval': (torch.zeros_like(rs), rt, qs, qt),
            'zero_temporal_retrieval': (rs, torch.zeros_like(rt), qs, qt),
            'zero_both_retrievals': (torch.zeros_like(rs), torch.zeros_like(rt), qs, qt),
            'zero_both_queries': (rs, rt, torch.zeros_like(qs), torch.zeros_like(qt)),
        }
        for name, (s, t, sq, tq) in variants.items():
            # All replicated time slices are identical in eval; 1x1 layers do not mix them.
            h = hidden[:, :1] + module.proto_out_proj(s + t + sq + tq).unsqueeze(1)
            pred = model.regression_layer(model.encoder(h.permute(0, 3, 2, 1)))
            if name == 'original':
                original = model({'X': x})
                torch.testing.assert_close(pred, original, atol=1e-5, rtol=1e-5)
                reconstruction_error = max(reconstruction_error, (pred - original).abs().max().item())
                single = model({'X': x[:1]})
                delta = scaler.inverse_transform(single) - scaler.inverse_transform(original[:1])
                sensitivity.append({'batch_index': int(batch_idx), 'mean_abs_change': delta.abs().mean().item(),
                                    'max_abs_change': delta.abs().max().item()})
            preds[name].append(scaler.inverse_transform(pred))
        targets.append(target)
    target = torch.cat(targets)
    predictions = {k: torch.cat(v) for k, v in preds.items()}
    interventions = {k: metrics(v, target) for k, v in predictions.items()}
    baseline = interventions['original']['MAE']
    for name, m in interventions.items():
        m['MAE_change_percent'] = (m['MAE'] / baseline - 1) * 100
        m['mean_abs_prediction_change'] = (predictions[name] - predictions['original']).abs().mean().item()
    branches = {}
    for name, raw in stats.items():
        a, q, r = (torch.cat(raw[k]) for k in ['a', 'q', 'r'])
        w = module.proto_out_proj.weight
        qc, rc = q - q.mean(0), r - r.mean(0)
        rq, rr = rms(qc @ w.T), rms(rc @ w.T)
        branch = spatial if name == 'spatial' else temporal
        pn = F.normalize(branch.prototypes, dim=-1)
        cosine = pn @ pn.T
        branches[name] = {
            'temperature': float(branch.temperature if name == 'spatial' else branch.temp),
            'normalized_entropy': float(-(a * a.clamp_min(1e-20).log()).sum(-1).mean() / math.log(a.shape[-1])),
            'mean_max_probability': a.max(-1).values.mean().item(),
            'mean_assignment': a.mean(0).tolist(),
            'projected_centered_retrieval_rms': rr,
            'projected_centered_query_rms': rq,
            'retrieval_query_rms_ratio': rr / max(rq, 1e-20),
            'mean_prototype_pair_cosine': cosine[~torch.eye(len(pn), dtype=torch.bool)].mean().item(),
        }
    per_horizon = [metrics(predictions['original'][:, h], target[:, h]) for h in range(target.shape[1])]
    return {'run': run, 'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            'seed': cfg['seed'], 'best_epoch_zero_based': ck['epoch'], 'saved_full_validation_metrics': ck['val_metrics'],
            'validation_samples': count, 'sampled_validation_samples': len(target),
            'validation_batch_indices': selected.tolist(), 'batch_size': bs,
            'forward_reconstruction_max_abs_error': reconstruction_error,
            'branches': branches, 'frozen_weight_interventions': interventions,
            'batch_sensitivity_first_sample': sensitivity, 'per_horizon_metrics': per_horizon}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batches', type=int, default=8)
    args = parser.parse_args()
    if args.batches < 1:
        parser.error('--batches must be positive')
    torch.set_num_threads(2)
    report = {'scope': 'validation subset only; no training or test metrics computed',
              'limitations': ['One seed per dataset.', 'Interventions change the distribution seen by frozen weights.',
                              'RMS ratio is not a contribution percentage.', 'Subset metrics do not replace full validation metrics.'],
              'runs': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        result = audit(run, args.batches)
        report['runs'].append(result)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'run': run, 'branches': result['branches'],
                          'interventions': {k: v['MAE_change_percent'] for k, v in result['frozen_weight_interventions'].items()}},
                         ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
