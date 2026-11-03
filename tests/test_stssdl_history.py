import numpy as np
import pytest
import torch

from data_pipeline.history import fit_weekly_flow_anchor
from models.STSSDL.STSSDL import STSSDL
from models.registry import build_model
from training.config import create_parser


def windows(values, slots=None, days=None, width=3):
    values = np.asarray(values, dtype=np.float32)
    frames = np.zeros((len(values), 2, 3), dtype=np.float32)
    frames[..., 0] = values[:, None] * np.array([1, 10])
    frames[..., 1] = np.asarray(slots if slots is not None else np.arange(len(values)) % 2)[:, None] / 2
    frames[..., 2] = np.asarray(days if days is not None else np.zeros(len(values)))[:, None] / 7
    return np.stack([frames[i:i + width] for i in range(len(values) - width + 1)])


def make_model():
    return STSSDL(num_nodes=2, model_dim=3, in_window=3, out_window=2,
                  rnn_units=4, prototype_num=3, prototype_dim=3,
                  tod_embed_dim=2, node_embed_dim=2, adaptive_embed_dim=2,
                  cheb_k=2, time_intervals=43200)


def test_anchor_deduplicates_overlap_and_preserves_zeros_and_nodes():
    x = windows([0, 2, 10, 6, 20])
    original = x.copy()
    anchor, counts = fit_weekly_flow_anchor(x, 2)
    np.testing.assert_allclose(anchor[0, 0], [10, 100])
    np.testing.assert_allclose(anchor[0, 1], [4, 40])
    assert counts[0, 0, 0] == 3
    assert counts.sum() == 5 * 2
    np.testing.assert_allclose(anchor[6, 1], [7.6, 76])
    np.testing.assert_array_equal(x, original)


def test_lookup_uses_fitted_flow_not_weekday_or_validation_values():
    model = make_model()
    model.fit_history(windows([0, 2, 10, 6, 20]))
    query = torch.from_numpy(windows([999, 999, 999]))
    actual = model._history_input(query)
    torch.testing.assert_close(actual[0, :, 0, 0], torch.tensor([10., 4., 10.]))
    query[..., 0] = -999
    torch.testing.assert_close(model._history_input(query), actual)
    query[..., 2] = 6 / 7
    torch.testing.assert_close(model._history_input(query)[0, :, 0, 0], torch.full((3,), 7.6))


def test_calendar_alignment_separates_weekdays():
    x = windows([1, 2, 11, 12], slots=[0, 1, 0, 1], days=[0, 0, 1, 1])
    anchor, _ = fit_weekly_flow_anchor(x, 2)
    np.testing.assert_allclose(anchor[:2, :, 0], [[1, 2], [11, 12]])


def test_unfitted_model_fails_and_checkpoint_restores_history():
    x = torch.from_numpy(windows([1, 2, 3]))
    model = make_model().eval()
    with pytest.raises(RuntimeError, match='history is not fitted'):
        model({'X': x})
    model.fit_history(x.numpy())
    restored = make_model().eval()
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(model({'X': x}), restored({'X': x}))
    assert restored.history_fitted.item()
    assert not restored.use_curriculum_learning


def test_training_and_inference_share_history_and_receive_gradients():
    x = torch.from_numpy(windows([0, 2, 10, 6, 20]))
    model = make_model()
    model.fit_history(x.numpy())
    history_inputs = []
    hook = model.encoder.register_forward_pre_hook(lambda module, args: history_inputs.append(args[0].detach()))
    output = model({'X': x})
    loss, parts = model.calculate_loss({'X': x, 'y': torch.ones_like(output)})
    hook.remove()
    torch.testing.assert_close(history_inputs[1], history_inputs[3])
    loss.backward()
    assert model.input_proj.weight.grad is not None
    assert torch.isfinite(model.input_proj.weight.grad).all()
    assert parts['pred'] == pytest.approx(torch.nn.functional.l1_loss(output, torch.ones_like(output)).item())


def test_registry_disables_curriculum_learning():
    args = create_parser().parse_args(['--model', 'STSSDL'])
    model = build_model('STSSDL', args, 2, np.eye(2, dtype=np.float32), torch.device('cpu'))
    assert not model.use_curriculum_learning


def test_history_with_adaptive_embedding_disabled():
    model = STSSDL(num_nodes=2, model_dim=3, in_window=3, out_window=2,
                   rnn_units=4, prototype_num=3, prototype_dim=3,
                   tod_embed_dim=2, node_embed_dim=2, adaptive_embed_dim=0,
                   cheb_k=2, time_intervals=43200)
    x = torch.from_numpy(windows([1, 2, 3]))
    model.fit_history(x.numpy())
    assert model({'X': x}).shape == (1, 2, 2, 1)
