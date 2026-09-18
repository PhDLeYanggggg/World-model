import numpy as np
import pytest

from src.evaluation.m3w_deferral_transfer import outputs, errors, seed_mean, grouped_summary
from src.evaluation.m3w_recording_diagnostic import recording_resamples


def test_action_threshold_is_fixed_and_label_free():
    proposal = np.ones((3, 12, 2))
    emitted = outputs(proposal, np.array([-.1, 0., .1]))
    assert not emitted['hard_action'][:2].any()
    np.testing.assert_array_equal(emitted['hard_action'][2], proposal[2])
    assert set(outputs(proposal)) == {'proposal'}


def test_seed_errors_are_not_prediction_ensemble_errors():
    target = np.zeros((2, 12, 2))
    cells = [errors(np.ones_like(target)*sign, target, np.ones(2, bool)) for sign in (-1, 0, 1)]
    mean = seed_mean(cells)
    assert mean['ade'][0] == pytest.approx(2*np.sqrt(2)/3)
    assert mean['changed'][0] == pytest.approx(2/3)


def test_group_readout_preserves_zero_error_ambiguity_and_dependence():
    baseline = np.array([0., 1., 3., 4.])
    values = dict(ade=baseline.copy(), fde=baseline.copy(), changed=np.zeros(4), requested=np.zeros(4))
    rec = np.array(['a', 'a', 'b', 'b'])
    _, inverse, counts = recording_resamples(rec, 2000, 17)
    result, rows = grouped_summary(values, baseline, np.ones(4), baseline > 2,
        {'recording':rec, 'scoped_agent':np.array(['a1','a1','b1','b2'])}, inverse, counts)
    assert result['gain_percent'] == result['equal_recording_gain_percent'] == 0
    assert result['gain_interval']['conditional_recording_ci95'] == [0., 0.]
    assert result['easy_percentage_degradation'] is None
    assert len(rows) == 2


@pytest.mark.parametrize('score', [np.array([np.nan]), np.ones(2)])
def test_invalid_score_rejected(score):
    with pytest.raises(ValueError):
        outputs(np.ones((1,12,2)), score)


def test_missing_seed_is_rejected():
    with pytest.raises(ValueError):
        seed_mean([{'ade':np.zeros(2)}])
