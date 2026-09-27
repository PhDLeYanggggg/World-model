import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.world_model.m3w_dimensionless_intervention import inference_features, allowed, joint_controls
from src.world_model.m3w_floor_relative import relative_targets
from src.world_model.m3w_fixed_producer_roles import role_indices
from src.world_model.m3w_geometric_cost_head import fit, predict
from src.world_model.m3w_native_gain_harm import preprocess
from test_m3w_agent_track_context import geometry


def test_point_guard_and_zero_reference_labels():
    utility = np.array([[2., 1.], [1., 2.], [2., 1.], [2., 1.]])
    a = np.array([[100., 1.], [100., 1.], [0., 1.], [100., 1.]])
    e = np.array([[20., .1], [20., .1], [0., 0.], [1., 1.]])
    assert allowed(utility, a, e, np.ones(4, bool), .02).tolist() == [True, False, False, False]
    with pytest.raises(ValueError): allowed(utility, a, e, np.ones(4), .02)
    cv = np.array([0., 1., np.nan])
    u, risk = relative_targets(cv, cv, np.array([2., .5, np.nan]), reference='cv', event='all', easy_cut=1.)
    assert risk[0].tolist() == [0., 2.] and u[0].tolist() == [0., 2.]
    assert np.isnan(risk[2]).all()


def test_features_have_no_target_argument_and_poisoning_invariance():
    g = geometry()
    b = g[:, 332:356].reshape(-1, 12, 2).copy()
    p = b + .25
    data = dict(geometry=g, target=np.ones_like(p), valid=np.ones(p.shape[:2], bool))
    a = inference_features(data['geometry'], b, p)
    data['target'][:] = np.nan
    data['valid'][:] = False
    z = inference_features(data['geometry'], b, p)
    assert a[0].shape == (len(g), 355)
    for left, right in zip(a, z): np.testing.assert_array_equal(left, right)


def test_rosters_and_nonzero_matched_count():
    rosters = [[f'{i}{j}' for j in range(4)] for i in range(3)]
    sites = np.array(sum(rosters, []))
    r = role_indices(sites, rosters, 0, 1)
    assert not set(r['producer']) & set(r['controller']) and r['readout_fold'] == 2
    cfg = json.loads(Path('configs/m3w_european_dimensionless_intervention_v1.json').read_text())
    current = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
    b = np.repeat(current[:, None], 12, axis=1)
    n = b.copy(); n[0, :, 0] += .9; n[1, :, 0] += 1.
    u = np.array([[3., .01], [2., .01], [1., .01], [.5, .01]])
    risk = np.tile([10., .01], (4, 1))
    out, report = joint_controls(u, risk, risk, np.ones(4, bool), current, np.ones(4), b, n, 1., cfg)
    assert report['matched_nonzero']
    assert all(out[k].sum() == 2 for k in ('half_independent', 'half_unary', 'half_joint'))
    assert not report['realized_risk_certified']


def test_cost_head_resume_exact(tmp_path):
    torch.set_num_threads(2)
    x = np.random.default_rng(17).normal(size=(12, 5)).astype(np.float32)
    sites = np.repeat(['a', 'b', 'c'], 4)
    cv = np.linspace(.5, 1., 12)
    y = np.column_stack((cv, cv*.03)); env = np.full(12, 2.)
    pr = preprocess(x, y, cv, sites, '__outer__')
    settings = dict(width=8, steps=4, batch_size=6, learning_rate=.001, gradient_clip=5,
                    checkpoint_every=2, heartbeat_every=1)
    def run(path, resume=False, stop=None):
        return fit(x, y, sites, env, pr, seed=17, task='all', settings=settings,
                   identity={'test': True}, directory=path, heartbeat=lambda **kw: None, resume=resume, stop_at=stop)
    m, _ = run(tmp_path/'full')
    run(tmp_path/'resume', stop=2)
    resumed, report = run(tmp_path/'resume', resume=True)
    assert report['new_updates'] == 2 and report['unknown_rows_sampled'] == 0
    np.testing.assert_array_equal(predict(m, x, env, pr), predict(resumed, x, env, pr))
