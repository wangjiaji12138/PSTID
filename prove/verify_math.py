"""Numerical cross-checks of the analytical proofs, using production modules.

Run from any directory: python /path/to/PSTID/prove/verify_math.py
Prints JSON only; does not train models or modify checkpoints or datasets.
Finite numerical checks are not substitutes for the proofs in the document.
"""

import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.PROPHET.spatial import SpatialCodebook
from models.PROPHET.temporal import TemporalBasis


def max_error(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    return (actual - expected).abs().max().item()


def spatial_checks():
    d, k = 5, 3
    proto = SpatialCodebook(k, proto_emb_dim=d, spatial_emb_dim=d,
                            time_series_dim=2, use_his=False, temperature=0.7)
    linear = SpatialCodebook(k, proto_emb_dim=d, spatial_emb_dim=d,
                             time_series_dim=2, use_his=False, use_proto=False)
    q = torch.randn(7, d)
    _, z = proto._compute_attention(q)
    _, scaled_z = proto._compute_attention(q * torch.linspace(0.2, 3.0, 7)[:, None])
    scale_error = max_error(scaled_z, z)

    # Proposition 6 requires normalized Query as the comparator input.
    with torch.no_grad():
        linear.linear[0].weight.copy_(F.normalize(proto.prototypes, dim=-1) / proto.temperature)
        linear.linear[0].bias.zero_()
        linear.linear[2].weight.copy_(proto.prototypes.T)
        linear.linear[2].bias.zero_()
    tied_error = max_error(linear.linear(F.normalize(q, dim=-1)), z)

    query = q[0].detach().requires_grad_()
    jac = torch.autograd.functional.jacobian(lambda v: proto._compute_attention(v)[1], query)
    radial_error = max_error(jac @ query, torch.zeros(d))
    norm_p = torch.linalg.matrix_norm(proto.prototypes, ord=2)
    norm_keys = torch.linalg.matrix_norm(F.normalize(proto.prototypes, dim=-1), ord=2)
    bound = (norm_p * norm_keys / (2 * proto.temperature * query.norm())).item()
    actual_norm = torch.linalg.matrix_norm(jac, ord=2).item()
    assert actual_norm <= bound + 1e-10

    # Theorem 5: two directions, two positive scales per direction.
    two = SpatialCodebook(2, proto_emb_dim=2, spatial_emb_dim=2,
                          time_series_dim=1, use_his=False, temperature=1.0)
    with torch.no_grad():
        two.prototypes.copy_(torch.eye(2))
    points = torch.tensor([[1., 0.], [2., 0.], [0., 1.], [0., 2.]])
    target = torch.sigmoid(torch.tensor([1., 1., -1., -1.]))
    _, spatial_output = two._compute_attention(points)
    construction_error = max_error(spatial_output[:, 0], target)
    a = torch.sigmoid(torch.tensor(1.0))
    saturation_risks = {}
    for magnitude in (1.0, 4.0, 10.0):
        prediction = 1 - a + (2 * a - 1) * torch.sigmoid(points @ torch.tensor([magnitude, -magnitude]))
        saturation_risks[str(magnitude)] = ((prediction - target) ** 2).mean().item()

    return {
        'query_scale_invariance_max_error': scale_error,
        'normalized_query_tied_LSL_max_error': tied_error,
        'radial_jacobian_max_error': radial_error,
        'jacobian_spectral_norm': actual_norm,
        'jacobian_proven_upper_bound': bound,
        'four_point_prototype_max_error': construction_error,
        'unbounded_LSL_saturation_MSE': saturation_risks,
    }


def temporal_containment_check():
    history_dim, tid_dim, diw_dim, d, k = 3, 2, 1, 5, 3
    m = history_dim + tid_dim + diw_dim
    common = dict(num_protos=k, proto_emb_dim=d, time_series_dim=history_dim,
                  temp_dim_tid=tid_dim, temp_dim_diw=diw_dim)
    linear = TemporalBasis(**common, use_proto=False)
    proto = TemporalBasis(**common, temp=0.7)
    with torch.no_grad():
        a, b = linear.linear[0], linear.linear[2]
        proto.allocator[0].weight.copy_(torch.cat([torch.eye(m), -torch.eye(m)]))
        proto.allocator[0].bias.zero_()
        proto.allocator[2].weight.copy_(proto.temp * torch.cat([a.weight, -a.weight], dim=1))
        proto.allocator[2].bias.copy_(proto.temp * a.bias)
        proto.prototypes.copy_(b.weight.T + b.bias)
        proto.proj.load_state_dict(linear.proj.state_dict())
    history = torch.randn(4, 3, 2, history_dim)
    tid = torch.randn(4, 3, 2, tid_dim)
    diw = torch.randn(4, 3, 2, diw_dim)
    zp, qp = proto(history, tid, diw)
    zl, ql = linear(history, tid, diw)
    values = torch.linspace(-10, 10, 101)
    return {
        'GELU_identity_max_error': max_error(F.gelu(values) - F.gelu(-values), values),
        'production_module_output_max_error': max_error(zp, zl),
        'unchanged_query_max_error': max_error(qp, ql),
    }


def temporal_interaction_check():
    # Theorem 9, with input carried by two temporal identity coordinates.
    proto = TemporalBasis(2, proto_emb_dim=1, time_series_dim=1,
                          temp_dim_tid=1, temp_dim_diw=1, use_his=False, temp=1.0)
    x = torch.tensor([[1., 1.], [-1., -1.], [1., -1.], [-1., 1.]])
    target = torch.tensor([1., 1., 0., 0.])
    c = F.gelu(torch.tensor(2.)) + F.gelu(torch.tensor(-2.))
    with torch.no_grad():
        proto.allocator[0].weight.copy_(torch.tensor([[1., 1.], [-1., -1.], [1., -1.], [-1., 1.]]))
        proto.allocator[0].bias.zero_()
        proto.allocator[2].weight.zero_()
        proto.allocator[2].weight[0].copy_(math.log(3) / c * torch.tensor([1., 1., -1., -1.]))
        proto.allocator[2].bias.zero_()
        proto.prototypes.copy_(torch.tensor([[1.5], [-0.5]]))
        for parameter in proto.proj.parameters():
            parameter.zero_()
    history = torch.zeros(4, 1, 1, 1)
    tid, diw = x[:, 0].reshape(4, 1, 1, 1), x[:, 1].reshape(4, 1, 1, 1)
    prediction, _ = proto(history, tid, diw)
    prediction = prediction.flatten()
    error = max_error(prediction, target)

    # Sanity check over finite random comparators, not a proof of the lower bound.
    count = 4096
    w = 3 * torch.randn(count, 2)
    offset = 3 * torch.randn(count, 1)
    amplitude = 3 * torch.randn(count, 1)
    bias = torch.randn(count, 1)
    sampled = bias + amplitude * torch.sigmoid(w @ x.T + offset)
    mse = ((sampled - target) ** 2).mean(dim=1)
    mae = (sampled - target).abs().mean(dim=1)
    assert mse.min().item() >= 1 / 16 - 1e-12
    assert mae.min().item() >= 1 / 8 - 1e-12
    return {
        'prototype_predictions': prediction.tolist(),
        'prototype_max_error': error,
        'prototype_MSE': ((prediction - target) ** 2).mean().item(),
        'prototype_MAE': (prediction - target).abs().mean().item(),
        'analytically_proven_LSL_MSE_lower_bound': 1 / 16,
        'analytically_proven_LSL_MAE_lower_bound': 1 / 8,
        'sampled_LSL_count_not_a_proof': count,
        'sampled_LSL_min_MSE': mse.min().item(),
        'sampled_LSL_min_MAE': mae.min().item(),
    }


def parameter_count_check():
    d, m, ks, kt = 64, 64, 16, 8
    temporal_kwargs = dict(num_protos=kt, proto_emb_dim=d, time_series_dim=32,
                           temp_dim_tid=16, temp_dim_diw=16)
    tp = TemporalBasis(**temporal_kwargs)
    tl = TemporalBasis(**temporal_kwargs, use_proto=False)
    spatial_kwargs = dict(num_protos=ks, proto_emb_dim=d, spatial_emb_dim=16, time_series_dim=32)
    sp = SpatialCodebook(**spatial_kwargs)
    sl = SpatialCodebook(**spatial_kwargs, use_proto=False)

    def excluded_query_count(module):
        return sum(p.numel() for name, p in module.named_parameters() if not name.startswith('proj.'))

    observed = {
        'spatial_prototype': excluded_query_count(sp),
        'spatial_LSL': excluded_query_count(sl),
        'temporal_prototype': excluded_query_count(tp),
        'temporal_LSL': excluded_query_count(tl),
    }
    expected = {
        'spatial_prototype': ks * d,
        'spatial_LSL': 2 * ks * d + ks + d,
        'temporal_prototype': 2 * m * m + 2 * m + 2 * m * kt + kt + kt * d + 1,
        'temporal_LSL': m * kt + kt + kt * d + d,
    }
    assert observed == expected
    return observed


def affine_risk_check():
    # Enumerate a finite isotropic noise distribution exactly, not Monte Carlo.
    d, r, n, variance = 5, 2, 20, 0.3
    covariance_scale = variance / n
    noise = math.sqrt(d * covariance_scale) * torch.cat([torch.eye(d), -torch.eye(d)])
    anchor = torch.tensor([0.2, -0.1, 0.3, 0.4, -0.2])
    projector = torch.diag(torch.tensor([1., 1., 0., 0., 0.]))
    cases = {}
    for label, shift in (
        ('well_specified', torch.tensor([0.7, -0.3, 0., 0., 0.])),
        ('misspecified', torch.tensor([0.7, -0.3, 0.5, 0., 0.])),
    ):
        theta = anchor + shift
        estimate = theta + noise
        constrained = anchor + (estimate - anchor) @ projector
        risk_free = (estimate - theta).square().sum(-1).mean()
        risk_constrained = (constrained - theta).square().sum(-1).mean()
        bias_squared = ((torch.eye(d) - projector) @ shift).square().sum()
        max_error(risk_free, torch.tensor(covariance_scale * d))
        max_error(risk_constrained, bias_squared + covariance_scale * r)
        cases[label] = {
            'unconstrained_risk': risk_free.item(),
            'constrained_risk': risk_constrained.item(),
            'structural_bias_squared': bias_squared.item(),
            'predicted_variance_saving': covariance_scale * (d - r),
        }
    return cases


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(20261019)
    source_paths = [
        'models/PROPHET/spatial.py', 'models/PROPHET/temporal.py',
        'models/PROPHET/prototype.py', 'models/PROPHET/model.py',
        'training/config.py', 'prove/verify_math.py',
    ]
    result = {
        'status': 'passed',
        'scope': 'Numerical cross-checks only; analytical proofs are in prototype_advantage.md.',
        'torch_version': torch.__version__,
        'dtype': str(torch.get_default_dtype()),
        'seed': 20261019,
        'source_sha256': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths},
        'spatial': spatial_checks(),
        'temporal_containment': temporal_containment_check(),
        'temporal_interaction': temporal_interaction_check(),
        'replacement_parameter_counts': parameter_count_check(),
        'known_affine_subspace_risk': affine_risk_check(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
