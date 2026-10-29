import numpy as np
import pytest
from src.evaluation.m3w_support_action_diagnostic import real_intervention_mask, causal_bank_envelope


def test_analytic_floor_identity_is_not_a_fake_intervention():
    floor = np.zeros((3, 12, 2))
    candidate = floor.copy(); candidate[1, 0, 0] = 1.
    np.testing.assert_array_equal(real_intervention_mask(candidate, floor), [False, True, False])
    with pytest.raises(ValueError):
        real_intervention_mask(candidate, floor[:1])
    candidate[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        real_intervention_mask(candidate, floor)


def test_partial_label_envelope_cannot_use_a_mean_over_requested_steps():
    baseline = np.zeros((1, 12, 2)); candidate = baseline.copy()
    candidate[0, :, 0] = np.arange(1, 13)
    envelope = causal_bank_envelope(baseline, baseline, candidate)
    assert envelope[0] == 12.
    assert envelope[0] > np.linalg.norm(candidate, axis=-1).mean()
