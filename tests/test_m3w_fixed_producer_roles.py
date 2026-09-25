import numpy as np
import pytest
from src.world_model.m3w_fixed_producer_roles import role_rosters, role_indices, bounded_ridge_scores, label_free_replay, forecaster_baseline
from src.world_model.m3w_producer_conditioned import safe_choice


def fixture():
    halves = {str(i): [[f's{4*i}', f's{4*i+1}'], [f's{4*i+2}', f's{4*i+3}']] for i in range(3)}
    return halves, np.repeat([f's{i}' for i in range(12)], 3)


def test_all_six_ordered_rotations_exclude_both_training_roles():
    halves, sites = fixture(); rosters = role_rosters(halves); seen = []
    for a in range(3):
        for b in range(3):
            if a == b: continue
            r = role_indices(sites, rosters, a, b); seen.append((a, b, r['readout_fold']))
            assert set(sites[r['producer']]).isdisjoint(sites[r['controller']])
            assert set(sites[r['readout']]).isdisjoint(sites[r['producer']])
            assert set(sites[r['readout']]).isdisjoint(sites[r['controller']])
            np.testing.assert_array_equal(np.sort(np.concatenate([r[k] for k in ('producer', 'controller', 'readout')])), np.arange(len(sites)))
    assert len(set(seen)) == 6


def test_row_order_not_future_labels_determines_roles():
    halves, sites = fixture(); roster = role_rosters(halves)
    order = np.random.default_rng(5).permutation(len(sites))
    r = role_indices(sites[order], roster, 0, 1)
    assert set(sites[order[r['readout']]]) == set(roster[2])


def test_reject_overlapping_producers():
    halves, _ = fixture(); halves['1'][0][0] = 's0'
    with pytest.raises(ValueError): role_rosters(halves)


def test_reject_undeclared_source_and_same_role():
    halves, sites = fixture(); roster = role_rosters(halves)
    with pytest.raises(ValueError): role_indices(np.append(sites, 'reserved'), roster, 0, 1)
    with pytest.raises(ValueError): role_indices(sites, roster, 0, 0)


def test_ridge_projection_preserves_reference_moment_not_utility_cap():
    raw = np.array([[8., -1.], [-2., 9.]])
    np.testing.assert_array_equal(bounded_ridge_scores(raw, [3., 2.], 'utility'), [[3., 0.], [0., 2.]])
    np.testing.assert_array_equal(bounded_ridge_scores(raw, [3., 2.], 'risk'), [[8., 0.], [0., 2.]])


def test_invalid_ridge_input_not_silently_safe():
    with pytest.raises(ValueError): bounded_ridge_scores([[np.nan, 0.]], [1.], 'risk')
    with pytest.raises(ValueError): bounded_ridge_scores([[1., 0.]], [-1.], 'utility')


def test_scalar_replay_of_stopping_and_risk_budget():
    u = np.array([[1., 0.], [1., 0.], [1., 2.], [1., 0.]])
    r = np.array([[1., .01], [1., .03], [1., 0.], [1., 0.]])
    moving = np.array([True, True, True, False])
    np.testing.assert_array_equal(label_free_replay((u, r), moving), [True, False, False, False])
    np.testing.assert_array_equal(label_free_replay((u, r), moving), safe_choice(u, r, moving))


def test_checkpoint_selected_baseline_is_not_evaluation_cv_index():
    state = dict(step=4000, identity=dict(kind='single', fold=0, seed=17, fit_sites=['a', 'b'], baseline_index=2))
    assert forecaster_baseline(state, ['b', 'a'], 0, 17) == 2
    with pytest.raises(ValueError): forecaster_baseline(state, ['a', 'c'], 0, 17)
    with pytest.raises(ValueError): forecaster_baseline(state, ['a', 'b'], 1, 17)
