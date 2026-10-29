import numpy as np
import pytest
from src.evaluation.m3w_selected_cost_diagnosis import subset_summary, audit_record
from src.world_model.m3w_multiaction_cost import select_actions


def fixture():
    p = np.tile([1., .001, 2., 1., .001], (4, 7, 1))
    y = np.tile([.5, .3, 1., .5, .2], (4, 7, 1)); y[-1] = np.nan
    return p, y, np.ones((4, 7), bool)


def test_selected_risk_decomposition_retains_unknowns_and_excludes_inactive_bank():
    p, y, allowed = fixture(); choice, _ = select_actions(p, allowed, arm='physical_only')
    out = audit_record(p, y, allowed, choice, np.ones(4), tuple(range(6)))
    assert out['all_legal']['action_views'] == 24
    assert out['selected']['unknown_action_views'] == 1
    assert out['selected']['observed_all_risk_on_known'] == pytest.approx(.3)
    assert out['selected']['predicted_all_risk_on_known'] == pytest.approx(.0005)
    decomposition = out['selected']['all_excess_decomposition']
    assert decomposition['harm_underprediction'] > 0 and decomposition['reference_inflation'] > 0
    assert decomposition['identity_error'] == pytest.approx(0)
    assert out['choice_counts']['6'] == 0


def test_diagnostic_labels_cannot_change_the_frozen_choice():
    p, y, allowed = fixture(); choice, _ = select_actions(p, allowed, arm='physical_neural')
    first = audit_record(p, y, allowed, choice, np.ones(4), tuple(range(7)))
    y[:3, :, 1] *= 3
    other = audit_record(p, y, allowed, choice, np.ones(4), tuple(range(7)))
    assert first['choice_counts'] == other['choice_counts']
    assert first['selected']['observed_all_risk_on_known'] != other['selected']['observed_all_risk_on_known']


def test_empty_diagnosis_is_undefined_and_partial_unknown_labels_are_rejected():
    p, y, allowed = fixture()
    out = subset_summary(p, y, np.zeros((4, 7), bool), np.ones(4))
    assert out['observed_all_risk_on_known'] is None and out['normalized_MSE'] is None
    y[-1, 0, 0] = 0
    with pytest.raises(ValueError):
        subset_summary(p, y, allowed, np.ones(4))
