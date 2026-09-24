import inspect
import json

import numpy as np
import pytest

from src.world_model.m3w_european_conditional_risk import (
    event_labels, fitting_support, nonnegative_moments, pointwise_rule, query_controls, decisions,
)


def test_event_targets_preserve_unknowns_and_exclude_zero_from_positive_easy():
    cv = np.array([0., .5, 1., 2., np.nan])
    harm = np.array([.2, .1, .4, .8, np.nan])
    easy = event_labels(cv, harm, easy_cut=1., event='easy')
    full = event_labels(cv, harm, easy_cut=1., event='all')
    np.testing.assert_equal(easy, [[0., 0.], [.5, .1], [1., .4], [0., 0.], [np.nan, np.nan]])
    np.testing.assert_equal(full[:, 0], cv)
    np.testing.assert_equal(full[:, 1], harm)
    with pytest.raises(ValueError):
        event_labels(cv, np.zeros(5), easy_cut=1., event='easy')


def test_zero_support_absence_is_not_a_small_positive_probability():
    report = fitting_support(np.array([.1, 2., np.nan]), np.array(['a', 'b', 'c']))
    assert not report['gate_available'] and report['zero_rows'] == 0
    assert report['unknown_rows'] == 1
    report = fitting_support(np.array([0., 2.]), np.array(['a', 'b']))
    assert report['zero_rows'] == 1 and report['zero_localities'] == ['a']
    assert report['interpretation'] == 'presence_is_not_a_safety_certificate'


def test_equal_rollout_zeroes_only_harm_not_reference_mass():
    p = np.array([[2., .1], [-1., -1.], [1., .2]])
    value = nonnegative_moments(p, np.array([True, False, False]))
    np.testing.assert_equal(value, [[2., 0.], [0., 0.], [1., .2]])
    assert p[0, 1] == .1


def test_easy_ratio_is_not_an_unconditional_risk_budget():
    gain = np.ones(3)
    moments = np.array([[1., .015], [.1, .015], [0., 0.]])
    bits = pointwise_rule(gain, moments, np.ones(3, bool), budget=.02, support_available=True)
    np.testing.assert_equal(bits, [True, False, False])
    assert not pointwise_rule(gain, moments, np.ones(3, bool), budget=.02, support_available=False).any()


def test_query_moment_budget_and_exact_counts_use_no_future_api():
    assert not {'target', 'valid', 'future', 'future_endpoint'} & set(inspect.signature(decisions).parameters)
    xy = np.array([[0., 0.], [20., 0.], [40., 0.]])
    baseline = np.tile(xy[:, None], (1, 12, 1))
    result = query_controls(np.array([3., 2., 1.]), np.tile([1., .015], (3, 1)),
        np.ones(3, bool), xy, np.ones(3), baseline, baseline+1,
        support_available=True, budget=.02, pair_weight=.1, radius_widths=3., threshold_widths=.5, seconds=2.)
    assert result['matched'] and result['reference_count'] == 3
    for arm in ('independent', 'unary_exact', 'joint_exact'):
        assert result[arm]['switch'].sum() == 3
        assert result[arm]['predicted_event_ratio'] <= .02
    closed = query_controls(np.array([3., 2., 1.]), np.tile([1., .015], (3, 1)),
        np.ones(3, bool), xy, np.ones(3), baseline, baseline+1,
        support_available=False, budget=.02, pair_weight=.1, radius_widths=3., threshold_widths=.5, seconds=2.)
    assert all(not closed[a]['switch'].any() for a in ('independent', 'joint', 'unary_exact', 'joint_exact', 'scene_uniform'))


def test_held_future_mutation_cannot_change_event_fit_or_support(tmp_path, monkeypatch):
    from test_m3w_european_cv_reference import nested_fixture
    from scripts import run_m3w_european_conditional_risk as runner
    old, data, identity, design, ti, _ = nested_fixture(tmp_path, monkeypatch)
    a = old.assemble(data, identity, 'complement0_seed17', design, ti)
    y, pr, lineage = runner.prepare_event(a, data, design, 'easy')
    changed = dict(data, target_eval=data['target_eval'].copy(), baseline_ade=data['baseline_ade'].copy())
    changed['target_eval'][:2] += 1000
    changed['baseline_ade'][:2] = 0
    b = old.assemble(changed, identity, 'complement0_seed17', design, ti)
    yy, ppr, ll = runner.prepare_event(b, changed, design, 'easy')
    np.testing.assert_equal(a['x'], b['x'])
    np.testing.assert_equal(y, yy)
    for k in ('mean', 'std', 'weights', 'known', 'constant'):
        np.testing.assert_equal(pr[k], ppr[k])
    assert ll == lineage and not lineage['source_support']['gate_available']
    assert np.isnan(y).all(1).sum() == 1


def test_reference_mass_head_keeps_equal_forecast_training_rows(tmp_path):
    import torch
    from src.world_model.m3w_native_gain_harm import preprocess, fit_neural, predict_neural
    torch.set_num_threads(1)
    x = np.arange(24, dtype=np.float32).reshape(6, 4)/24
    cv = np.ones(6); sites = np.array(['a']*3+['b']*3)
    y = event_labels(cv, np.zeros(6), easy_cut=1., event='easy')
    pr = preprocess(x, y, cv, sites, 'held')
    settings = dict(width=4, steps=4, batch_size=4, learning_rate=.001,
                    gradient_clip=5., checkpoint_every=2, heartbeat_every=2)
    model, report = fit_neural(x, y, sites, np.zeros(6, bool), pr, seed=17,
        arm='underharm4', settings=settings, identity={'fixture': True}, directory=tmp_path,
        heartbeat=lambda **kw: None)
    assert report['complete'] and report['step'] == 4
    out = nonnegative_moments(predict_neural(model, x, np.zeros(6, bool), pr), np.ones(6, bool))
    assert (out[:, 0] > 0).all() and not out[:, 1].any()


def test_missing_controls_and_incomplete_matrix_fail_closed(tmp_path, monkeypatch):
    from scripts import run_m3w_european_conditional_risk as runner
    monkeypatch.setattr(runner, 'PRIVATE', tmp_path)
    with pytest.raises(ValueError, match='36'):
        runner.checked_head('missing', {})
    with pytest.raises(ValueError, match='Missing decisions'):
        runner.get_decisions({}, {}, {}, np.array([0]), np.ones(1, bool), np.ones((1, 2)),
            np.ones(1), 'missing', {'identity': {}}, 'no_guard', True)


def test_decisions_keep_past_population_and_guard_does_not_select_a_subset():
    h = np.zeros((3, 8, 2)); h[:, :, 0] = np.arange(-7, 1)
    baseline = np.ones((3, 12, 2))
    reg = dict(predicted_risk_budget=.02, pair_weight=.1, edge_radius_bbox_widths=3.,
                proximity_threshold_bbox_widths=.5, solver_seconds=2.)
    kw = dict(history=h, origin=np.array([[0., 0.], [20., 0.], [40., 0.]]), widths=np.ones(3),
        recordings=np.array([0, 0, 1]), frames=np.array([10, 10, 20]), sites=np.array(['a']*3),
        baseline=baseline, candidate=baseline+.1, utility=np.ones(3), moments=np.tile([1., .001], (3, 1)),
        held_ids=np.arange(3), query_mask=np.ones(3, bool), support_available=False)
    choice, reports = decisions(reg, **kw)
    assert choice['pointwise_ids'].tolist() == [0, 1, 2] and choice['ids'].tolist() == [0, 1, 2]
    assert not choice['pointwise'].any()
    assert len(reports) == 2
    json.dumps(reports, allow_nan=False)
