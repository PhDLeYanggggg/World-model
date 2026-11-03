import numpy as np
import pytest

from scripts import run_m3w_moving_shared_motion_cost as runner


def test_policy_paths_only_use_saved_causal_rollouts():
    paths = np.arange(3*7*12*2, dtype=float).reshape(3, 7, 12, 2)
    floor = paths[:, 1]
    chosen = dict(independent=np.array([-1, 6, -1]))
    p = runner.policy_paths(paths, floor, chosen)
    np.testing.assert_array_equal(p['independent'][[0, 2]], floor[[0, 2]])
    np.testing.assert_array_equal(p['independent'][1], paths[1, 6])
    np.testing.assert_array_equal(p['uncontrolled_neural'], paths[:, 6])


def test_joint_receipt_replay_checks_fresh_cost_binding_and_original_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'PRIVATE', tmp_path)
    p = np.zeros((3, 7, 5))
    p[:, :, 2] = 10
    p[:, :, 3] = 5
    p[:, 6, 0] = 2
    p[:, 6, 1] = .01
    p[:, 6, 4] = .001
    allowed = np.zeros((3, 7), bool)
    allowed[:, 6] = True
    ids = np.arange(3)
    held = dict(recordings=np.zeros(3, int), frames=np.full(3, 7), ids=ids)
    causal = dict(allowed=allowed, groups=[ids])
    job = dict(key='synthetic')
    cfg = dict(joint_solver_seconds_per_query=.25)
    choices, report = runner.allocated(p, causal, held, job, 'local_history', cfg, lambda **_:None)
    assert report['labels_used'] is False
    assert all(np.array_equal(c, [6, 6, 6]) for c in choices.values())
    replay, checks = runner.allocated(p, causal, held, job, 'local_history', cfg, lambda **_:None, verify=True)
    for key in choices:
        np.testing.assert_array_equal(replay[key], choices[key])
    assert checks['cached_feasibility_verified']
    changed = p.copy()
    changed[0, 6, 0] += .01
    with pytest.raises(ValueError, match='different freshly replayed'):
        runner.allocated(changed, causal, held, job, 'local_history', cfg, lambda **_:None, verify=True)


def test_group_intervals_average_seed_producers_and_keep_missing_undefined():
    groups = [dict(site=site, value=float(index)) for site in ('g0', 'g1') for index in range(6)]
    cfg = dict(bootstrap_resamples=100, bootstrap_seed=17)
    value = runner.group_interval(groups, lambda r:r['value'], ['g0', 'g1'], cfg)
    assert value['point'] == 2.5
    assert value['CI95'] == [2.5, 2.5]
    assert value['by_source_group'] == dict(g0=2.5, g1=2.5)
    assert 'not_verified_IID' in value['bootstrap_unit']
    missing = runner.group_interval(groups, lambda _:None, ['g0', 'g1'], cfg)
    assert missing['point'] is None
    assert missing['CI95'] is None
    assert missing['status'] == 'required_view_not_estimable'
