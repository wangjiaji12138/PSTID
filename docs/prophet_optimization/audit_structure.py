"""Reproduce structural properties of the current PROPHET on CPU.

This audit uses a tiny, randomly initialized model. It does not load data,
update parameters through training, or estimate trained-model metric changes.
Run from the repository root:
    python -m docs.prophet_optimization.audit_structure
"""

import argparse
import copy
import json
from pathlib import Path

import torch

from models.PROPHET.model import PROPHET


def max_difference(left, right):
    return float((left - right).abs().max())


def temporal_weights(model, embeddings):
    _, history, tid, diw, _, _ = embeddings
    branch = model.proto_module.temporal_basis
    inputs = torch.cat([history[:, -1], tid[:, -1], diw[:, -1]], dim=-1)
    return (branch.allocator(inputs) / branch.temp).softmax(dim=-1)


def spatial_weights(model, embeddings):
    _, history, _, _, node, adj = embeddings
    branch = model.proto_module.spatial_codebook
    identity = adj if adj is not None else node
    inputs = torch.cat([identity, history.mean(1).mean(0)], dim=-1)
    return branch._compute_attention(branch.proj(inputs))[0]


@torch.no_grad()
def run_audit():
    torch.manual_seed(20261021)
    torch.set_num_threads(1)
    model = PROPHET(
        num_nodes=4,
        input_window=6,
        output_window=4,
        num_block=1,
        time_series_emb_dim=8,
        spatial_emb_dim=4,
        temp_dim_tid=4,
        temp_dim_diw=4,
        adj_mx_emb_dim=4,
        proto_emb_dim=8,
        num_spatial_prototypes=3,
        num_temporal_prototypes=4,
        device=torch.device("cpu"),
    ).eval()
    sample = torch.zeros(1, 6, 4, 3)
    sample[..., 0] = torch.randn(1, 6, 4)
    sample[..., 1] = (torch.arange(6) + 10).view(1, 6, 1) / 48
    sample[..., 2] = 2 / 7
    original = model({"X": sample})

    companion = sample.clone()
    companion[..., 0] += 5
    mixed = torch.cat([sample, companion], dim=0)
    mixed_prediction = model({"X": mixed})
    duplicate_prediction = model({"X": torch.cat([sample, sample], dim=0)})
    permuted_prediction = model({"X": mixed.flip(0)}).flip(0)
    batch_change = max_difference(original[0], mixed_prediction[0])
    duplicate_change = max_difference(original[0], duplicate_prediction[0])
    permutation_change = max_difference(mixed_prediction, permuted_prediction)
    assert batch_change > 1e-5, "Expected the spatial history batch mean to affect predictions."
    torch.testing.assert_close(original[0], duplicate_prediction[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(mixed_prediction, permuted_prediction, atol=1e-6, rtol=1e-5)

    changed_node = sample.clone()
    changed_node[:, :, 1, 0] += 100
    changed_prediction = model({"X": changed_node})
    other_node_change = max_difference(original[:, :, 0], changed_prediction[:, :, 0])
    perturbed_node_change = max_difference(original[:, :, 1], changed_prediction[:, :, 1])
    torch.testing.assert_close(original[:, :, 0], changed_prediction[:, :, 0], atol=1e-6, rtol=1e-5)
    assert perturbed_node_change > 1e-5

    embeddings = model._get_embeddings({"X": sample})
    hidden, history, tid, diw, node, adj = embeddings
    enhancement = model.proto_module(history, adj, node, tid, diw)
    repeated = {name: max_difference(value, value[:, :1].expand_as(value))
                for name, value in {"hidden": hidden, "history": history,
                                    "tid": tid, "diw": diw,
                                    "prototype_enhancement": enhancement}.items()}
    assert all(value == 0 for value in repeated.values())
    one_slice = (hidden[:, :1] + enhancement[:, :1]).permute(0, 3, 2, 1)
    one_slice_prediction = model.regression_layer(model.encoder(one_slice))
    torch.testing.assert_close(original, one_slice_prediction, atol=1e-6, rtol=1e-5)

    changed_identity_model = copy.deepcopy(model)
    changed_identity_model.adj_mx_emb[0] += torch.tensor([5.0, -3.0, 2.0, -4.0])
    changed_identity_model.node_emb[0] += torch.tensor([-2.0, 6.0, -1.0, 3.0])
    changed_embeddings = changed_identity_model._get_embeddings({"X": sample})
    temporal_before = temporal_weights(model, embeddings)
    temporal_after = temporal_weights(changed_identity_model, changed_embeddings)
    spatial_before = spatial_weights(model, embeddings)
    spatial_after = spatial_weights(changed_identity_model, changed_embeddings)
    time_identity_change = max_difference(temporal_before, temporal_after)
    space_identity_change = max_difference(spatial_before, spatial_after)
    torch.testing.assert_close(temporal_before, temporal_after, atol=0, rtol=0)
    assert space_identity_change > 1e-5

    # Algebraic control for a proposed conditional-logit term; no model change.
    assignments = spatial_before
    zero_relation = torch.zeros(3, 4)
    temporal = model.proto_module.temporal_basis
    inputs = torch.cat([history[:, -1], tid[:, -1], diw[:, -1]], dim=-1)
    logits = temporal.allocator(inputs)
    zero_relation_prediction = ((logits + assignments @ zero_relation) / temporal.temp).softmax(-1)
    torch.testing.assert_close(temporal_before, zero_relation_prediction, atol=0, rtol=0)
    relation = torch.randn(3, 4)
    centered = relation - relation.mean(1, keepdim=True) - relation.mean(0, keepdim=True) + relation.mean()
    uniform_effect = torch.full_like(assignments, 1 / 3) @ centered

    return {
        "scope": "Random initialization, CPU eval(), seed=20261021; structural checks only, not accuracy evidence.",
        "torch_version": torch.__version__,
        "batch_dependence": {
            "changed_companion_target_max_abs_difference": batch_change,
            "duplicate_companion_target_max_abs_difference": duplicate_change,
            "permuting_same_batch_max_abs_difference": permutation_change,
            "interpretation": "Batch composition matters; permutation of the same batch preserves predictions.",
        },
        "cross_node_observation_path": {
            "perturbation": "Add 100 only to historical flow at node 1; keep calendars and target node 0 unchanged.",
            "target_node_0_max_abs_difference": other_node_change,
            "perturbed_node_1_max_abs_difference": perturbed_node_change,
            "interpretation": "Current operations transmit no other-node current observations to a target node. Static learned graph identity remains present.",
        },
        "broadcast_time_axis": {
            "max_abs_differences_from_first_slice": repeated,
            "one_slice_eval_prediction_max_abs_difference": max_difference(original, one_slice_prediction),
            "interpretation": "Internal time slices are copies. Forecast horizons still use distinct regression output channels. Training dropout has independent masks; this check establishes eval equivalence only.",
        },
        "temporal_router_spatial_identity": {
            "explicit_spatial_identity_change_temporal_weights_max_abs_difference": time_identity_change,
            "explicit_spatial_identity_change_spatial_weights_max_abs_difference": space_identity_change,
            "explicit_spatial_identity_change_prediction_max_abs_difference": max_difference(original, changed_identity_model({"X": sample})),
            "interpretation": "The temporal router lacks explicit node/graph/spatial-prototype inputs. Own-node history can still carry spatial information, and the downstream MLP can learn joint effects.",
        },
        "conditional_relation_algebra": {
            "zero_initialized_relation_router_max_abs_difference": max_difference(temporal_before, zero_relation_prediction),
            "uniform_spatial_assignment_double_centered_relation_max_abs_effect": float(uniform_effect.abs().max()),
            "interpretation": "A zero C preserves the current router. Identical C rows are an existing temporal-logit bias; rowwise constants cancel in softmax. Double centering removes both redundancies.",
            "fair_controls": "Keep backbone and both query projections identical for Q-only, Q+S, Q+T, Q+S+T, and Q+S+T+C. Also compare concat or a matched-budget identity-conditioned allocator: improvements from extra parameters or identity input alone do not establish a unique benefit of prototype relations.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("structure_audit.json"))
    args = parser.parse_args()
    report = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
