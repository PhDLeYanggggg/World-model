import copy

import numpy as np
import pytest

from scripts.audit_m3w_temporal_intervention import decomposition, selected_costs, verify_gates


def test_crossover_decomposition_preserves_both_paths():
    r = decomposition([2., 4.], [3., 5.], [2.5, 4.2], [2.8, 4.8])
    assert r['total']['equal_scene_pp'] == -1
    for action, choices in [('action_at_ramp_choices', 'choices_on_uniform'),
                            ('action_at_uniform_choices', 'choices_on_ramp')]:
        np.testing.assert_allclose(np.array(r[action]['by_scene_pp'])+r[choices]['by_scene_pp'], [-1, -1])


@pytest.mark.parametrize('bad', [[np.nan, 1], [1], []])
def test_decomposition_rejects_bad_support(bad):
    with pytest.raises(ValueError):
        decomposition([1, 2], [1, 2], bad, [1, 2])


def test_costs_separate_unknown_and_incomplete_not_safe():
    r = selected_costs([2, 3, np.nan, 0], [1, 5, np.nan, 4],
                       np.array([True, False, False, True]), np.ones(4, bool), np.ones((4, 2)))
    assert r['complete']['rows'] == 2 and r['complete']['realized_harm_mean'] == 2
    assert r['complete']['harm_underpredicted']
    assert r['incomplete_observed']['realized_harm_mean'] == 2
    assert r['unknown']['rows'] == 1 and 'realized_harm_mean' not in r['unknown']


def test_costs_reject_support_mismatch():
    with pytest.raises(ValueError):
        selected_costs([np.nan], [1], np.array([False]), np.array([True]), np.ones((1, 2)))


def fixture_report():
    checks = dict(positive_primary_ci=True, each_seed_positive_cv=True, aggregate_easy=True,
                  each_scene_seed_easy=True, exact_zero=True, positive_old_scalar_ci=True)
    return dict(summaries={'ramp_strict': {'seeds': {'17': dict(ADE={'equal_scene_gain_percent': 1.},
        subsets={'positive_easy': {'equal_scene_gain_percent': -2., 'by_scene': {'a': {'gain_percent': -2.}}}},
        zero_CV_harmed=0)}}}, contrasts={'uniform_strict': {'ci95_pp': [.1, 1]},
        'scalar_log_strict': {'ci95_pp': [.1, 1]}}, primary_gates=checks, primary_joint_empirical_pass=True)


def test_fixed_gate_boundary_and_reject_modified_zero_harm():
    r = fixture_report()
    assert all(verify_gates(r).values())
    r['summaries']['ramp_strict']['seeds']['17']['zero_CV_harmed'] = 1
    with pytest.raises(ValueError):
        verify_gates(r)


def test_fixed_gate_reject_modified_result_flag():
    r = copy.deepcopy(fixture_report()); r['primary_joint_empirical_pass'] = False
    with pytest.raises(ValueError):
        verify_gates(r)
