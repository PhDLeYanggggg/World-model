import numpy as np
import pytest

from scripts.diagnose_m3w_source_cost_deferral import huber_location, summarize


def test_huber_location_is_not_necessarily_mean_gain():
    values = np.array([-.1, -.1, -.1, 10.])
    root = huber_location(values)
    assert root == pytest.approx(1/3-.1)
    assert root < values.mean()
    assert np.clip(root-values, -1, 1).mean() == pytest.approx(0., abs=1e-12)


def test_bounded_symmetric_target_recovers_mean_and_constant_case():
    assert huber_location(np.array([-.2, 0., .2])) == pytest.approx(0., abs=1e-12)
    assert huber_location(np.ones(4)*.4) == pytest.approx(.4)


def test_score_summary_distinguishes_hard_and_soft_action():
    result = summarize(np.array([.5, -.1]), np.array([0., 0.]),
                       np.array([.5, .5]), np.array([False, False]))
    assert result['hard_gain'] == 0
    assert result['soft_expected_gain'] == pytest.approx(.1)
    assert result['expected_risk_score_gradient'] == pytest.approx(-.05)
    assert result['requested_rate'] == 0


@pytest.mark.parametrize('values', [[], [[1]], [float('nan')]])
def test_invalid_diagnostic_targets_rejected(values):
    with pytest.raises(ValueError):
        huber_location(values)
