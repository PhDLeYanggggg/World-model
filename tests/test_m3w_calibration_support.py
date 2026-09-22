from copy import deepcopy

import numpy as np
import pytest

from src.evaluation.m3w_calibration_support import (
    audit_view, best_case_cluster_support, require_calibration_exclusion, required_producers,
)


ROSTER = ['a', 'b', 'c', 'd']


def producer(excluded, seed=17):
    fit = sorted(set(ROSTER)-set(excluded))
    return dict(id='_'.join(excluded)+str(seed), training_sites=fit,
        preprocessing_fit_sites=fit, excluded_sites=excluded, seed=seed,
        parents=[], initialization='random_seed', checkpoint_selection_sites=[],
        calibration_sites=[], research_design_exposed_sites=ROSTER, objective='native_coordinate')


def view():
    return dict(outer_site='a', seed=17,
        outer_producer=dict(producer=producer(['a'])),
        groups=[dict(inner_site=s, producer=producer(['a', s]), rows=10) for s in ROSTER[1:]])


def test_detects_all_three_shortcuts_and_retains_valid_outer_exclusion():
    rows = audit_view(view(), ROSTER[1:], ROSTER)
    assert len(rows) == 3
    assert all(r['current_outer_fitting_exclusion'] for r in rows)
    assert all(not a['accepted'] for r in rows for a in r['attempts'].values())
    assert all(len(r['exposed_cost_target_producers']) == 2 for r in rows)
    assert all('cost target' in r['attempts']['drop_rows_and_replace_scoring_predictor']['reason'] for r in rows)
    assert all(not r['current_outer_is_independent_confirmation'] for r in rows)


def test_complete_prospective_exclusion_is_not_independence_certificate():
    result = require_calibration_exclusion(calibration_sites=['b'], head_fit_sites=['c', 'd'],
        preprocessing_fit_sites=['c', 'd'], scoring_producer=producer(['a', 'b']),
        target_producers=[producer(['a', 'b', 'c']), producer(['a', 'b', 'd'])])
    assert result['fitting_exclusion_pass']
    assert not result['independence_verified'] and not result['data_roles_assigned']


@pytest.mark.parametrize('field', ['training_sites', 'preprocessing_fit_sites',
                                 'checkpoint_selection_sites', 'calibration_sites'])
def test_hidden_producer_exposure_is_rejected(field):
    target = producer(['a', 'b', 'c'])
    target[field] = ['b']
    with pytest.raises(ValueError, match='exposes'):
        require_calibration_exclusion(calibration_sites=['b'], head_fit_sites=['c', 'd'],
            preprocessing_fit_sites=['c', 'd'], scoring_producer=producer(['a', 'b']),
            target_producers=[target])


def test_preprocessing_and_undeclared_ancestry_fail_closed():
    args = dict(calibration_sites=['b'], head_fit_sites=['c', 'd'],
        preprocessing_fit_sites=['b', 'c', 'd'], scoring_producer=producer(['a', 'b']),
        target_producers=[producer(['a', 'b', 'c'])])
    with pytest.raises(ValueError, match='preprocessing'):
        require_calibration_exclusion(**args)
    args['preprocessing_fit_sites'] = ['c', 'd']
    args['target_producers'][0]['parents'] = ['unknown_teacher']
    with pytest.raises(ValueError, match='without parents'):
        require_calibration_exclusion(**args)


def test_missing_or_duplicate_group_rejected():
    v = view()
    v['groups'][1] = deepcopy(v['groups'][0])
    with pytest.raises(ValueError, match='Complete fixed'):
        audit_view(v, ROSTER[1:], ROSTER)


def test_required_producers_are_not_pair_exclusions():
    available = [producer(['a', 'b'])]
    requirements = required_producers(ROSTER, [17, 29, 43], available)
    assert len(requirements) == 12 and not any(r['available'] for r in requirements)
    assert all(len(r['training_sites']) == 1 for r in requirements)
    assert sum(r['available'] for r in required_producers(ROSTER, [17], [producer(['a', 'b', 'c'])])) == 1


def test_cluster_best_case_is_synthetic_and_not_easy_relative_risk():
    rows = best_case_cluster_support()
    expected = [min(1., np.sqrt(np.log(20)/(2*n))) for n in [1, 2, 4]]
    np.testing.assert_allclose([r['best_case_unit_range_upper_bound'] for r in rows], expected)
    assert all(not r['real_calibration'] and not r['same_as_relative_easy_degradation'] for r in rows)
    with pytest.raises(ValueError):
        best_case_cluster_support([0])
