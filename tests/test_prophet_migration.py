"""Check the promoted model against PSTID, including trained residual branches."""

import pytest
import torch

from models.PSTID import PSTID
from models.PROPHET import PROPHET
from models.registry import build_model


@pytest.mark.parametrize('options', [
    {}, {'use_adj': False}, {'use_proto': False}, {'use_temporal': False},
    {'use_spatio': False}, {'use_his': False},
    {'use_time_interaction': False}, {'use_proto_change': False},
    {'proto_change_context': 'raw'},
])
def test_pstid_weights_predictions_and_gradients_match(options):
    config = dict(num_nodes=2, input_window=3, output_window=5,
                  time_series_emb_dim=4, spatial_emb_dim=3,
                  temp_dim_tid=3, temp_dim_diw=2, proto_emb_dim=4,
                  adj_mx_emb_dim=2, time_slots=[12, 13, 14])
    config.update(options)
    torch.manual_seed(88)
    source = PSTID(**config).eval()
    torch.manual_seed(88)
    target = PROPHET(**config).eval()
    assert source.state_dict().keys() == target.state_dict().keys()
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)

    x = torch.randn(2, 3, 2, 3)
    x[..., 1] = 14 / 48
    x[..., 2] = 6 / 7
    # Learn nonzero corrections so equality also covers the newly added branches.
    optimizer = torch.optim.Adam(source.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        source({'X': x}).square().mean().backward()
        optimizer.step()
    target.load_state_dict(source.state_dict(), strict=True)
    source.zero_grad(set_to_none=True)
    expected = source({'X': x})
    actual = target({'X': x})
    assert actual.shape == (2, 5, 2, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for key, parameter in source.named_parameters():
        target_grad = target.get_parameter(key).grad
        if parameter.grad is None:
            assert target_grad is None
        else:
            torch.testing.assert_close(target_grad, parameter.grad, rtol=0, atol=0)


@pytest.mark.parametrize('schedule', [
    {'prophet_time_slots': [12, 13, 14]},
    {'pstid_time_slots': [12, 13, 14]},
    {'prophet_time_slots': [12, 13, 14], 'pstid_time_slots': [0, 1]},
])
def test_registered_prophet_restores_schedule_and_switches(schedule):
    config = dict(schedule, use_adj=False, use_time_interaction=0,
                  use_proto_change=0, proto_change_context='raw')
    model = build_model('PROPHET', config, 2, None, torch.device('cpu'))
    assert isinstance(model, PROPHET)
    assert model.model_name == 'prophet'
    assert model.time_slots.tolist() == [12, 13, 14]
    assert not model.proto_module.temporal_basis.use_time_interaction
    assert model.proto_change_readout is None
    assert model.proto_change_context == 'raw'
