import numpy as np
import pytest
import torch
from src.world_model.m3w_native_protected_risk import risk_targets, risk_features, build_head, fit_head, predict_head
from src.world_model.m3w_native_gain_harm import preprocess
from src.evaluation.m3w_native_protected_risk_eval import risk_decisions, event_quality


def test_targets_keep_partial_outcomes_unknown():
    cv, harm = np.array([0., 1., 0., np.nan]), np.array([2., 3., 0., np.nan])
    complete = np.array([True, True, False, False])
    np.testing.assert_equal(risk_targets(cv, harm, complete, 'all_harm'), [[1, 2], [1, 3], [np.nan, np.nan], [np.nan, np.nan]])
    np.testing.assert_equal(risk_targets(cv, harm, complete, 'zero_reference_harm'), [[1, 2], [0, 0], [np.nan, np.nan], [np.nan, np.nan]])
    with pytest.raises(ValueError):
        risk_targets(cv, harm, np.ones(4, bool), 'zero_reference_harm')


def test_zero_target_uses_exact_reference_not_epsilon():
    target = risk_targets(np.array([0., 1e-12]), np.ones(2), np.ones(2, bool), 'zero_reference_harm')
    np.testing.assert_equal(target, [[1, 1], [0, 0]])


def test_input_features_have_only_causal_motion_flags():
    g = np.zeros((2, 476), np.float32)
    g[:, 16:24] = np.arange(-7, 1)/12
    g[1, :16] = np.column_stack((np.arange(8), np.zeros(8))).ravel()
    pred = np.ones((2, 12, 2), np.float32)
    x, same = risk_features(g, pred, np.ones(2))
    assert x.shape == (2, 357)
    np.testing.assert_equal(x[:, -2:], [[1, 1], [0, 0]])
    assert not same.any()
    g[:, 356:] = 987
    np.testing.assert_array_equal(risk_features(g, pred, np.ones(2))[0], x)


def test_shared_initialization_and_exact_resume(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(8)
    x = rng.normal(size=(40, 5)).astype(np.float32)
    sites = np.repeat(['a', 'b'], 20)
    same = np.zeros(40, bool); same[0] = True
    cv = np.linspace(0, 5, 40); h = rng.uniform(0, 2, 40); h[0] = 0
    complete = np.ones(40, bool); complete[4] = False
    y = risk_targets(cv, h, complete, 'all_harm')
    pr = preprocess(x, y, np.where(complete, cv, np.nan), sites, 'held')
    settings = dict(width=8, steps=6, batch_size=8, learning_rate=.001, gradient_clip=5., checkpoint_every=2, heartbeat_every=2)
    a = build_head(5, 8, 17); b = build_head(5, 8, 17)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    args = dict(seed=17, arm='all_harm', settings=settings, identity={'version':1}, heartbeat=lambda **_:None)
    full, record = fit_head(x, y, sites, same, pr, directory=tmp_path/'full', **args)
    fit_head(x, y, sites, same, pr, directory=tmp_path/'resumed', stop_at=2, **args)
    resumed, rc = fit_head(x, y, sites, same, pr, directory=tmp_path/'resumed', resume=True, **args)
    np.testing.assert_array_equal(predict_head(full, x, same, pr), predict_head(resumed, x, same, pr))
    out = predict_head(full, x, same, pr)
    assert (out[:, 0] >= 0).all() and (out[:, 0] <= 1).all() and (out[:, 1] >= 0).all()
    np.testing.assert_array_equal(out[0], [0, 0])
    assert record['unknown_rows_sampled'] == 0 and rc['total_draws'] == 48
    with pytest.raises(ValueError):
        fit_head(x, y, sites, same, pr, directory=tmp_path/'resumed', **args)


def test_common_budget_is_causal_and_never_forces_a_rejected_risk():
    gain = np.array([[5., 0], [4, 0], [3, 0], [2, 0]])
    risk = dict(all_harm=np.array([[.02, 1], [0, 0], [.001, 0], [0, 0]]),
        zero_reference_harm=np.zeros((4, 2)))
    policies, meta = risk_decisions(gain, risk, np.zeros(4, bool), np.array([False, True, False, False]),
        np.arange(4), np.ones(4, bool))
    assert meta['requested_count'] == 4 and meta['common_count'] == 3
    assert all(v.sum() == 3 for k,v in policies.items() if k.endswith('_matched'))
    assert not policies['all_harm_guard_matched'][0]
    assert not policies['stop_veto_matched'][1]


def test_event_quality_zero_and_one_are_valid_not_silent_nan():
    q = event_quality(np.array([0., 1., .99]), np.array([0., 1., np.nan]), np.array([True, True, False]))
    assert q['brier'] == 0 and q['ece'] == 0 and q['auroc'] == 1
    assert q['rows'] == 2 and q['positive_rows'] == 1
