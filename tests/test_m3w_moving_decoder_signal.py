import numpy as np
import pytest
import torch

from src.evaluation.m3w_moving_decoder_signal import probe, signal
from src.world_model import m3w_shared_motion_cost as api


def test_probe_matches_original_forward_and_rejects_label_fields():
    rng = np.random.default_rng(17)
    c = dict(x=rng.normal(size=(4, 7, 428)).astype(np.float32),
        envelope=np.ones((4, 7)), scales=np.ones(4), allowed=np.ones((4, 7), bool),
        groups=[np.arange(3), np.array([3])], source_groups=[np.arange(2)],
        sites=np.array(['source']*4), ids=np.arange(4))
    pr = dict(mean=np.zeros(428), std=np.ones(428), clip=8., bias=np.zeros(5, np.float32))
    model = api.base.base.CostHead(428, 8, 17, pr['bias'], 'context')
    with torch.no_grad():
        model.action.weight.normal_(0, .1)
        model.reference.weight.normal_(0, .1)
    state = dict(identity=dict(feature_arm='local_history'), preprocess=pr,
        settings=dict(width=8), seed=17, model=model.state_dict())
    result = probe(state, c)
    np.testing.assert_array_equal(result['float32'], api.predict(state, c))
    assert result['raw_action6'].shape == (4, 3)
    assert np.isfinite(result['same_float32_inputs_float64']).all()
    with pytest.raises(ValueError, match='causal input'):
        probe(state, dict(c, future_endpoint=np.ones((4, 2))))


def test_true_cost_mass_not_row_count_weights_link_diagnosis():
    raw = np.array([[0., -30., 0.], [0., 0., -30.]])
    p = np.ones((2, 7, 5))
    y = np.zeros_like(p)
    y[:, 6, 4] = [9, 1]
    result = signal(raw, p, p, y, np.ones(2, bool), np.array([6, 6]))
    at = result['pools']['independent_selected']['attenuation'][-1]
    assert at['known_EH_mass_fraction_small_h_link'] == .9
    assert at['known_EH_mass_fraction_small_EH_gate_link'] == .1
    assert at['known_EH_mass_fraction_either_small_link'] == 1
    empty = signal(raw, p, p, y, np.ones(2, bool), np.array([-1, -1]))
    assert empty['pools']['independent_selected']['attenuation'][0]['known_EH_mass_fraction_either_small_link'] is None


def test_reported_link_factors_match_direct_autograd():
    raw = torch.tensor([.2, -.7, -3.], dtype=torch.float64, requires_grad=True)
    fraction = torch.softmax(torch.cat((raw[:2], torch.zeros(1, dtype=raw.dtype))), 0)
    h = 5*fraction[1]
    derivative = torch.autograd.grad(h, raw, retain_graph=True)[0][1]/5
    torch.testing.assert_close(derivative, fraction[1]*(1-fraction[1]))
    gate = torch.sigmoid(raw[2])
    eh = h.detach()*gate
    derivative = torch.autograd.grad(eh, raw)[0][2]/h.detach()
    torch.testing.assert_close(derivative, gate*(1-gate))
