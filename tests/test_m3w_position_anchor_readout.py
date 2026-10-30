import copy
import numpy as np
import pytest
from src.evaluation import m3w_position_anchor as api
from src.world_model.m3w_european_source_forecast import baseline_numpy
from src.evaluation.m3w_native_metrics import native_errors


def fixture():
    h = np.broadcast_to(np.arange(8)[None, :, None]*np.array([.5, -.2]), (2, 8, 2)).copy()
    h[:, -1] += [.3, .1]; origin = h[:, -1].copy()
    y = np.broadcast_to(np.arange(8, 20)[None, :, None]*np.array([.5, -.2]), (2, 12, 2)).copy()
    valid = np.ones((2, 12), bool); valid[1] = False; valid[0, -1] = False
    physical = np.stack([baseline_numpy(h, k)-origin[:, None] for k in range(6)], 1)
    paths = np.concatenate((physical, physical[:, 5:6]), 1)
    errors = [native_errors(paths[:, i]+origin[:, None], y, valid, np.ones(2)) for i in range(7)]
    p = dict(rollouts=paths, floor_rollout=paths[:, 1], floor_ade=errors[1][0], floor_fde=errors[1][1],
        cv_ade=errors[1][0], ade=np.stack([e[0] for e in errors], 1), fde=np.stack([e[1] for e in errors], 1),
        ids=np.arange(2), recordings=np.array(['r0', 'r1']), frames=np.array([7, 7]),
        scales=np.ones(2), hard=np.zeros(2, bool))
    return h, origin, y, valid, p, np.ones(2, bool), 100.


def test_unknown_rows_and_endpoint_absence_are_not_safe_zero_or_removed():
    r = api.evaluate(*fixture())
    assert r['rows'] == 2 and r['known_rows'] == r['unknown_rows'] == 1
    assert r['neutral_raw_paths_ADE_FDE_exact'] and not r['inference_labels_used']
    assert all(v['known_FDE_gain_floor_percent'] is None for v in r['policies'].values())
    assert not r['same_intervention_count_claim'] and r['new_training_updates'] == 0
    assert all(not r['recordings']['r1'][a]['budget_status']['supported'] for a in api.NAMES)


def test_changed_original_control_refuses():
    args = list(fixture()); args[4]['ade'][0, 4] += .1
    with pytest.raises(AssertionError): api.evaluate(*args)


def test_full_roster_and_deployment_are_separate():
    contexts = {f'c{i}': [f's{4*(i%3)+j}' for j in range(4)] for i in range(6)}
    result = api.evaluate(*fixture())
    rows = [dict(context=c, source=s, seed=43, result=copy.deepcopy(result), identity_control_exact=True)
        for c, ss in contexts.items() for s in ss]
    cfg = dict(bootstrap_seed=3, bootstrap_resamples=32)
    s = api.summarize(rows, contexts, cfg)
    assert s['groups'] == 24 and s['new_training_updates'] == 0
    assert not s['fitting_advancement_gate'] and not s['fitting_gates']['complete_original_recording_risk']
    assert s['FDE']['ols8_fitted']['point'] is None and not s['deployment_changed']
    assert not s['independent_confirmation'] and not s['submission_ready'] and not s['smc_enabled'] and not s['stage5c_executed']
    with pytest.raises(ValueError): api.summarize(rows[:-1], contexts, cfg)
