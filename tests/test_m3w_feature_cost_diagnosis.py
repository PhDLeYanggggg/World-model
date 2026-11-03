from copy import deepcopy

import numpy as np
import pytest

from src.evaluation.m3w_cost_mass_decomposition import cost_masses, risks
from src.evaluation.m3w_feature_cost_diagnosis import describe_records, packet_for_role, summarize


def test_fitting_packet_is_explicit_without_learning_or_mutation():
    job = dict(critic_sites=['G'], evaluation_sites=['H'], fit_scope={'rows': 3},
               evaluation_scope={'rows': 4})
    old = deepcopy(job)
    packet = packet_for_role(job, 'critic_fit')
    assert packet['evaluation_sites'] == ['G']
    assert packet['evaluation_scope'] == {'rows': 3}
    assert job == old
    assert packet_for_role(job, 'double_excluded') is job
    with pytest.raises(ValueError):
        packet_for_role(job, 'confirmation')


def test_false_safe_harm_and_reference_are_measured_separately_on_identical_set():
    target = np.array([[0., 1., 10., 4., 1.]])
    pred = np.array([[0., .01, 10., 2., .01]])
    m = cost_masses(pred, target, np.array([True]))
    row = dict(selected=1, unknown_selected=0, selected_actual_risks=risks(m),
        full_legal_neural=m, selected_full_proposal=m, selected_scaled_proposal=m,
        selected_actual_interpolation=m)
    summary = describe_records([row])
    for key in ('all_risk', 'easy_risk'):
        assert summary[key]['predicted_safe_actual_violations'] == 1
        assert summary[key]['false_safe_numerator_underpredicted'] == 1
        assert summary[key]['false_safe_denominator_overpredicted'] == 0
    assert summary['posthoc_no_unknown_selected']['median_EH_prediction_target_ratio'] == .01


def test_zero_support_is_kept_undefined_and_incomplete_rosters_refused():
    m = cost_masses(np.zeros((1, 5)), np.zeros((1, 5)), np.array([False]))
    row = dict(selected=0, unknown_selected=0, selected_actual_risks=risks(m),
        full_legal_neural=m, selected_full_proposal=m, selected_scaled_proposal=m,
        selected_actual_interpolation=m)
    summary = describe_records([row])
    assert summary['easy_risk']['defined_views'] == 0
    assert summary['easy_risk']['actual_violations'] == 0
    assert summary['selected_actual_interpolation']['EH']['undefined_record_views'] == 1
    assert summary['posthoc_no_unknown_selected']['record_views'] == 0
    with pytest.raises(ValueError):
        summarize([], [])
