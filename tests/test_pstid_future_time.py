import pytest
import torch

from models.PSTID.model import PSTID


def make_model(**kwargs):
    kwargs.setdefault("use_time_interaction", True)
    return PSTID(num_nodes=2, input_window=3, output_window=5,
                 time_series_emb_dim=4, spatial_emb_dim=3,
                 temp_dim_tid=3, temp_dim_diw=2, proto_emb_dim=4,
                 use_adj=False, **kwargs)


def history(slot=47, day=6):
    x = torch.zeros(2, 3, 2, 3)
    x[..., 1] = slot / 48
    x[..., 2] = day / 7
    return x


def test_calendar_rollover_and_shared_encoding():
    model = make_model()
    future = model._future_calendar(history())
    torch.testing.assert_close(future[0, :, 0, 0] * 48, torch.arange(5).float())
    assert torch.count_nonzero(future[..., 1]) == 0
    tid, diw = model._encode_time(future)
    torch.testing.assert_close(tid[:, 0], model.time_in_day_emb[0].expand(2, 2, -1))
    torch.testing.assert_close(diw[:, 0], model.day_in_week_emb[0].expand(2, 2, -1))


def test_partial_daily_schedule():
    model = make_model(time_slots=[12, 13, 14])
    future = model._future_calendar(history(14, 6))
    torch.testing.assert_close(future[0, :, 0, 0] * 48, torch.tensor([12., 13., 14., 12., 13.]))
    torch.testing.assert_close(future[0, :, 0, 1] * 7, torch.tensor([0., 0., 0., 1., 1.]))


@pytest.mark.parametrize('use_his', [True, False])
def test_future_conditions_temporal_prototypes_and_backprop(use_his):
    torch.manual_seed(0)
    model = make_model(use_his=use_his).eval()
    with torch.no_grad():
        model.future_readout.weight.fill_(0.1)
    x = history()
    hidden, series, tid, diw, _, _ = model._get_embeddings({'X': x})
    future_tid, future_diw = model._encode_time(model._future_calendar(x))
    series = torch.cat([series, series[:, :1].expand(-1, 5, -1, -1)], dim=1)
    basis = model.proto_module.temporal_basis
    a, _ = basis(series, tid, diw, future_tid, future_diw)
    b, _ = basis(series, tid, diw, future_tid + 1, future_diw + 1)
    assert not torch.allclose(a, b)
    output = model({'X': x})
    assert output.shape == (2, 5, 2, 1)
    output.square().mean().backward()
    assert model.time_in_day_emb.grad[4].abs().sum() > 0
    assert model.day_in_week_emb.grad[0].abs().sum() > 0


@pytest.mark.parametrize('options', [{'use_proto': False}, {'use_temporal': False}])
def test_ablation_without_temporal_branch(options):
    assert make_model(**options)({'X': history()}).shape == (2, 5, 2, 1)


def test_history_last_and_all_future_steps_use_own_calendar_and_prototype():
    torch.manual_seed(7)
    model = make_model().eval()
    with torch.no_grad():
        model.future_readout.weight.fill_(0.1)
    x = history()
    x[:, :, :, 1] = torch.tensor([45., 46., 47.])[None, :, None] / 48
    _, series, tid, diw, _, _ = model._get_embeddings({'X': x})
    torch.testing.assert_close(tid[0, :, 0], model.time_in_day_emb[47:48])
    future_tid, future_diw = model._encode_time(model._future_calendar(x))
    torch.testing.assert_close(future_tid[0, :, 0], model.time_in_day_emb[:5])
    series = torch.cat([series, series[:, :1].expand(-1, 5, -1, -1)], dim=1)
    basis = model.proto_module.temporal_basis
    original, _ = basis(series, tid, diw, future_tid, future_diw)
    changed = future_tid.clone()
    changed[:, 1] += 2
    modified, _ = basis(series, tid, diw, changed, future_diw)
    assert original.shape == (2, 6, 2, 4)
    assert not torch.allclose(original[:, 2], modified[:, 2])
    torch.testing.assert_close(original[:, [0, 1, 3, 4, 5]],
                               modified[:, [0, 1, 3, 4, 5]])
    output = model({'X': x})
    output.square().mean().backward()
    for slot in [47, 0, 1, 2, 3, 4]:
        assert model.time_in_day_emb.grad[slot].abs().sum() > 0


def test_future_position_reaches_corresponding_prediction():
    torch.manual_seed(8)
    model = make_model().eval()
    with torch.no_grad():
        model.future_readout.weight.fill_(0.1)
    x = history()
    original = model({'X': x})
    # Slot 1 appears only at future step 1, and never in this history.
    with torch.no_grad():
        model.time_in_day_emb[1].add_(2)
    changed = model({'X': x})
    assert not torch.allclose(original[:, 1], changed[:, 1])
    torch.testing.assert_close(original[:, [0, 2, 3, 4]], changed[:, [0, 2, 3, 4]])


def test_zero_initialized_correction_and_single_position_encoder():
    model = make_model().eval()
    assert torch.count_nonzero(model.future_readout.weight) == 0
    assert model.regression_layer.out_channels == model.output_window
    shapes = []
    handle = model.encoder.register_forward_pre_hook(lambda module, args: shapes.append(args[0].shape))
    output = model({'X': history()})
    handle.remove()
    assert shapes == [torch.Size([2, model.hidden_dim, 2, 1])]
    output.square().mean().backward()
    assert model.future_readout.weight.grad.abs().sum() > 0


def test_earlier_calendar_ignored_but_entire_history_flow_used():
    torch.manual_seed(17)
    model = make_model().eval()
    x = history()
    original = model({'X': x})
    changed = x.clone()
    changed[:, :-1, :, 1:3] = 0
    torch.testing.assert_close(model({'X': changed}), original)
    changed[:, 0, :, 0] = 10
    assert not torch.allclose(model({'X': changed}), original)


def test_embedding_lookup_matches_indexing_outputs_and_gradients():
    import copy
    import types

    torch.manual_seed(31)
    optimized = make_model().double().eval()
    with torch.no_grad():
        optimized.future_readout.weight.normal_()
    reference = copy.deepcopy(optimized)

    def indexed_time(self, calendar):
        tid = torch.round(calendar[..., 0] * self.time_of_day_size).long().clamp(0, self.time_of_day_size - 1)
        diw = torch.round(calendar[..., 1] * 7).long().clamp(0, 6)
        return self.time_in_day_emb[tid], self.day_in_week_emb[diw]

    reference._encode_time = types.MethodType(indexed_time, reference)
    x = history().double().requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    actual = optimized({'X': x})
    expected = reference({'X': ref_x})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=1e-8, atol=1e-9)
    for (name, param), (ref_name, ref_param) in zip(optimized.named_parameters(), reference.named_parameters()):
        assert name == ref_name
        if param.grad is not None:
            torch.testing.assert_close(param.grad, ref_param.grad, rtol=1e-8, atol=1e-9, msg=name)
        else:
            assert ref_param.grad is None


def test_interaction_zero_init_preserves_previous_model_and_learns():
    torch.manual_seed(52)
    baseline = make_model(use_time_interaction=False).eval()
    with torch.no_grad():
        baseline.future_readout.weight.fill_(0.1)
    interaction = make_model(use_time_interaction=True).eval()
    loaded = interaction.load_state_dict(baseline.state_dict(), strict=False)
    assert set(loaded.missing_keys) == {
        'proto_module.temporal_basis.calendar_interaction.weight',
        'proto_module.temporal_basis.prototype_transition',
    }
    assert not loaded.unexpected_keys
    expected = baseline({'X': history()})
    actual = interaction({'X': history()})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    basis = interaction.proto_module.temporal_basis
    assert basis.calendar_interaction.weight.grad.abs().sum() > 0
    assert basis.prototype_transition.grad.abs().sum() > 0


@pytest.mark.parametrize('use_his', [True, False])
def test_history_calendar_conditions_future_prototypes(use_his):
    torch.manual_seed(53)
    model = make_model(use_his=use_his).eval()
    _, series, tid, diw, _, _ = model._get_embeddings({'X': history()})
    future_tid, future_diw = model._encode_time(model._future_calendar(history()))
    series = series.expand(-1, 6, -1, -1)
    basis = model.proto_module.temporal_basis
    with torch.no_grad():
        basis.calendar_interaction.weight.normal_(std=0.2)
        basis.prototype_transition.normal_(std=0.2)
    original, _ = basis(series, tid, diw, future_tid, future_diw)
    changed, _ = basis(series, tid + 1, diw, future_tid, future_diw)
    assert not torch.allclose(original[:, 1:], changed[:, 1:])
    with_interaction_history = original[:, :1].detach().clone()
    # Turning off interaction restores independent future allocations.
    basis.use_time_interaction = False
    original, _ = basis(series, tid, diw, future_tid, future_diw)
    changed, _ = basis(series, tid + 1, diw, future_tid, future_diw)
    torch.testing.assert_close(original[:, 1:], changed[:, 1:])
    torch.testing.assert_close(original[:, :1], with_interaction_history)


def test_interaction_switch_preserves_shared_initialization():
    torch.manual_seed(88)
    baseline = make_model(use_time_interaction=False)
    torch.manual_seed(88)
    interaction = make_model(use_time_interaction=True)
    state = interaction.state_dict()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("context", ["raw", "encoded"])
def test_change_readout_zero_init_preserves_baseline_and_learns(context):
    torch.manual_seed(88)
    baseline = make_model(use_time_interaction=False, use_proto_change=False).eval()
    torch.manual_seed(88)
    changed = make_model(use_time_interaction=False, use_proto_change=True,
                         proto_change_context=context).eval()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(changed.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(changed({'X': history()}), baseline({'X': history()}), rtol=0, atol=0)
    optimizer = torch.optim.Adam(changed.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        changed({'X': history()}).square().mean().backward()
        assert changed.proto_change_readout.weight.grad.abs().sum() > 0
        optimizer.step()
    assert changed.proto_change_query.weight.grad.abs().sum() > 0
    assert changed.proto_change_delta.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('options', [{'use_proto': False}, {'use_temporal': False}])
def test_change_readout_disabled_without_temporal_prototypes(options):
    model = make_model(use_proto_change=True, **options)
    assert model.proto_change_readout is None
    assert model({'X': history()}).shape == (2, 5, 2, 1)
