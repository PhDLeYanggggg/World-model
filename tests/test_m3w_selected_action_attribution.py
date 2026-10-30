import copy
import numpy as np
import pytest

from src.evaluation import m3w_selected_action_attribution as api


def fixture():
    harm = np.array([.05, .05, .2, .05, 0., np.nan])
    predicted_harm = np.array([.005, .05, .05, .018, .001, .001])
    predicted_reference = np.array([1., 10., 3., 3., 1., 1.])
    y = np.zeros((6, 7, 5))
    y[..., 1] = harm[:, None]
    y[..., 2:4] = 1.
    y[..., 4] = harm[:, None]
    y[-1] = np.nan
    p = np.zeros_like(y)
    p[..., 0] = .1
    p[..., 1] = predicted_harm[:, None]
    p[..., 2:4] = predicted_reference[:, None, None]
    p[..., 4] = predicted_harm[:, None]
    packet = dict(ids=np.arange(6), y=y, cv_ade=np.array([.5] * 5 + [np.nan]),
        recordings=np.array(['r0'] * 6), envelope=np.ones((6, 7)), scales=np.ones(6),
        floor_ade=np.array([1.] * 5 + [np.nan]))
    choices = np.array([6, 6, 6, 6, -1, 6])
    return packet, p, choices


def test_four_false_admission_witnesses_are_disjoint_exhaustive_and_not_policies():
    packet, p, c = fixture()
    result = api.analyze(packet, p, c, easy_cut=1.)
    for channel in ('all', 'easy'):
        value = result['channels'][channel]
        assert value['false_admitted_known_rows'] == 4
        assert value['witness_counts'] == dict(numerator_only=1, denominator_only=1,
            both_single_replacements_fail=1, only_joint_replacement_fails=1)
        assert value['decomposition_residual'] == pytest.approx(0, abs=1e-10)
    assert result['selected_unknown_rows'] == 1
    assert result['rows'] == 6 and result['selected_known_rows'] == 4
    assert result['future_informed_diagnostic_only']
    assert not result['new_policy_constructed']
    assert result['selected_unknown_envelope_mass'] == 1.
    assert np.isnan(packet['y'][-1]).all()


def test_future_easy_event_cannot_be_inferred_from_positive_reference_alone():
    packet, p, c = fixture()
    packet['floor_ade'][0] = 0
    packet['y'][0, :, 2:4] = 0
    result = api.analyze(packet, p, c, easy_cut=1.)
    support = result['easy_support']
    assert support['selected_known_easy_rows'] == 4
    assert support['selected_known_easy_zero_reference_rows'] == 1
    assert support['selected_known_noneasy_rows'] == 0
    changed = copy.deepcopy(packet)
    changed['cv_ade'][:4] = 2.
    changed['y'][:4, :, 3:5] = 0
    result = api.analyze(changed, p, c, easy_cut=1.)
    assert result['easy_support']['selected_known_easy_rows'] == 0
    assert result['easy_support']['predicted_easy_reference_on_known_noneasy'] > 0
    assert result['records']['r0']['easy_support']['selected_easy_reference_mass'] == 0


def test_unknown_rows_never_receive_imputed_supervised_costs_or_arithmetic_attribution():
    packet, p, c = fixture()
    first = api.analyze(packet, p, c, easy_cut=1.)
    p[-1] *= 100
    second = api.analyze(packet, p, c, easy_cut=1.)
    assert first['channels'] == second['channels']
    assert first['selected_unknown_rows'] == second['selected_unknown_rows'] == 1
    bad = copy.deepcopy(packet)
    bad['y'][-1, :, 1] = 0
    with pytest.raises(ValueError):
        api.analyze(bad, p, c, easy_cut=1.)


def test_hash_equivalent_selected_choices_and_future_labels_do_not_mutate_inputs():
    packet, p, c = fixture()
    before_p, before_c = p.copy(), c.copy()
    before_y = packet['y'].copy()
    api.analyze(packet, p, c, easy_cut=1.)
    np.testing.assert_array_equal(p, before_p)
    np.testing.assert_array_equal(c, before_c)
    np.testing.assert_array_equal(packet['y'], before_y)


def test_predicted_risk_violation_is_not_silently_analyzed_as_causally_admitted():
    packet, p, c = fixture()
    p[0, 6, 1] = 1.
    with pytest.raises(ValueError, match='causal decoder'):
        api.analyze(packet, p, c, easy_cut=1.)
