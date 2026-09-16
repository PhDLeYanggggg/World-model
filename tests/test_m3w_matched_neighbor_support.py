from copy import deepcopy

import pytest
import torch

from test_m3w_eqmotion_adapter import batch
from src.world_model.m3w_supervised_intervention import PastContextForecaster


def model():
    return PastContextForecaster(width=8, heads=2, layers=1,
                                 neighbor_policy='complete_aligned_history').eval()


def test_partial_and_misaligned_neighbors_have_no_effect():
    torch.set_num_threads(2)
    predictor, inputs = model(), batch()
    inputs['neighbors'][1, 1, 0, 2] -= .01
    changed = deepcopy(inputs)
    changed['neighbors'][0, 1, :, :2] = 999
    changed['neighbors'][1, 1, :, :2] = -999
    changed['neighbors'][~changed['neighbor_mask']] = float('nan')
    with torch.no_grad():
        torch.testing.assert_close(predictor(inputs), predictor(changed), atol=0, rtol=0)


def test_support_rule_equals_explicit_past_masking():
    inputs, matched = batch(), model()
    inputs['neighbors'][1, 1, 0, 2] -= .01
    old = PastContextForecaster(width=8, heads=2, layers=1).eval()
    old.load_state_dict(matched.state_dict())
    masked = deepcopy(inputs)
    aligned = torch.isclose(inputs['neighbors'][..., 2], inputs['history'][:, None, :, 2],
                            atol=1e-6, rtol=1e-6).all(-1)
    eligible = inputs['neighbor_mask'].all(-1) & aligned
    masked['neighbor_mask'] &= eligible[..., None]
    with torch.no_grad():
        torch.testing.assert_close(matched(inputs), old(masked), atol=0, rtol=0)


def test_future_observed_token_is_not_silently_filtered():
    inputs = batch()
    inputs['neighbors'][0, 1, -1, 2] = .1
    with pytest.raises(ValueError, match='post-current'):
        model()(inputs)


def test_unknown_support_is_refused():
    with pytest.raises(ValueError, match='support policy'):
        PastContextForecaster(width=8, heads=2, layers=1, neighbor_policy='future_available')
