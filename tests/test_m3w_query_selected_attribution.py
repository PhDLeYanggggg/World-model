import numpy as np
import pytest
from src.evaluation import m3w_query_selected_attribution as api
from tests.test_m3w_active_bank_readout import fixture
from tests.test_m3w_frame_action_allocation import problem


def data(missing=False):
    packet = fixture()[0]['new_neural']; p, ok = problem()
    y = np.zeros((2, 7, 5)); y[..., 2:4] = 1.
    y[0, 1, 1] = y[0, 1, 4] = .1; y[1, 0, 0] = .1
    if missing:
        y[0] = np.nan
        packet['cv_ade'][0] = packet['floor_ade'][0] = packet['floor_fde'][0] = np.nan
        packet['ade'][0] = packet['fde'][0] = np.nan
    packet.update(y=y, allowed=ok, envelope=np.full((2, 7), .5), frames=np.zeros(2, int))
    return packet, p, np.array([1, 0])


def test_pooled_diagnostic_does_not_mislabel_intended_row_risk_as_false_query():
    packet, p, choice = data()
    with pytest.raises(ValueError, match='unchanged causal decoder'):
        api.old.analyze(packet, p, choice, easy_cut=1.5)
    result = api.analyze(packet, p, choice, easy_cut=1.5)
    assert result['row_excess_not_necessarily_a_query_violation']
    assert result['fully_known_selected_queries'] == 1
    assert result['query_channels']['easy']['fully_known_selected_queries_with_observed_excess'] == 1
    assert result['query_channels']['easy']['witness_counts']['numerator_only'] == 1
    assert result['full_policy_unknowns_retained']
    assert not result['optimized_physical_only_comparator']
    assert not result['deployment_policy_constructed']


def test_selected_unknown_cannot_be_zeroed_into_fully_known_query_evidence():
    packet, p, choice = data(missing=True)
    result = api.analyze(packet, p, choice, easy_cut=1.5)
    assert result['selected_unknown_rows'] == 1 and result['selected_unknown_queries'] == 1
    assert result['fully_known_selected_queries'] == 0
    assert result['query_channels']['easy']['fully_known_selected_queries_with_observed_excess'] == 0
    assert result['channels']['easy']['observed_positive_excess_rows'] == 0
    assert result['future_informed_diagnostic_only'] and not result['causal_identification']


def test_incremental_neural_use_has_own_bound_and_cancels_shared_unknown_actions():
    packet = fixture(missing=True)[0]['new_neural']
    shared = np.array([6, 6]); empty = np.array([-1, -1])
    result = api.incremental_neural_use(packet, shared, shared)
    assert result['known_pp'] == 0 and result['lower_pp'] == 0 and result['unknown_disagreement_mass'] == 0
    result = api.incremental_neural_use(packet, empty, shared)
    assert result['known_pp'] == pytest.approx(60.)
    assert result['lower_pp'] == pytest.approx(0.)
    assert result['upper_pp'] == pytest.approx(120.)
    assert result['not_difference_of_independent_lower_bounds']
    assert not result['causal_identification']
