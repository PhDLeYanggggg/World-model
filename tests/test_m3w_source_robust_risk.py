import copy
import numpy as np
import pytest
import torch
from src.world_model import m3w_source_robust_risk as api
from tests.test_m3w_kinematic_risk import fixture as raw_fixture
from src.world_model import m3w_admission_priority as setup


def fixture(tmp_path):
    torch.set_num_threads(1)
    data = raw_fixture()
    data['causal']['sites'] = np.array(['a', 'a', 'b', 'b'])
    data['causal']['source_groups'] = [np.array([0]), np.array([1])]
    data['preprocess']['source_names'] = ['a', 'b']
    data['truth_by_arm']['bound_aware'][2:, :, 1] = 2.
    data['truth_by_arm']['bound_aware'][2:, :, 4] = 1.2
    settings = dict(width=4, learning_rate=.0001, query_batch_size=2,
        minimum_updates=4, largest_source_cycles=1, gradient_clip=5., heartbeat_every=1, checkpoint_every=1)
    first = setup.parent.parent.fit(data, arm='mean', seed=17, settings=settings, identity={}, path=tmp_path/'first.pt.gz')
    data = setup.parent.prepare(data, first)
    origin = setup.parent.fit(data, first, arm='kinematic', settings=settings, identity={}, path=tmp_path/'warm.pt.gz')
    data = setup.prepare(data, origin)
    origin = dict(origin, observed_mask_hash=api.parent.parent.kin.digest(data['observed_future']))
    data = api.parent.prepare(data, origin)
    warm = api.parent.fit(data, origin, arm='relational', settings=settings, identity={}, path=tmp_path/'rel.pt.gz')
    return data, warm, settings


def test_exact_control_and_source_objective_decomposition(tmp_path):
    data, warm, settings = fixture(tmp_path); z, sites = api.validate(data, warm)
    ix = np.arange(4); seg = torch.tensor([0, 0, 1, 1]); model = api.model_from(warm)
    pred = model(z, seg, 2)
    original = api.parent.previous.previous.objective(pred, data, ix, seg, 2, 'joint')
    q = api.query_terms(pred, data, ix, seg, 2)
    for key, val in q.items():
        torch.testing.assert_close(val.mean(), original[key])
    a = api.objective(pred, data, ix, seg, 2, 'mean_continue', sites)
    b = api.objective(pred, data, ix, seg, 2, 'source_robust', sites)
    assert torch.equal(a['total'], original['total'])
    torch.testing.assert_close(b['total'], .75*a['total']+.25*q['total'].max())
    assert b['total'] >= a['total']
    ga = torch.autograd.grad(a['total'], pred, retain_graph=True)
    gb = torch.autograd.grad(original['total'], pred, retain_graph=True)
    api.base.exact(ga, gb)


def test_unknown_bounds_retained_and_no_query_split(tmp_path):
    data, warm, _ = fixture(tmp_path); changed = copy.deepcopy(data)
    data['observed_future'][:] = False
    pred = (torch.zeros((4, 7, 2)), torch.ones((4, 7, 2)))
    q = api.query_terms(pred, data, np.arange(4), torch.tensor([0, 0, 1, 1]), 2)
    assert q['composed_observed_cost'].sum() == 0
    assert q['risk_MSE'].sum() > 0 and q['decision_objective'].sum() > 0
    changed['causal']['sites'][1] = 'different'
    with pytest.raises(ValueError, match='Homogeneous'):
        api.validate(changed, warm)


def test_matched_training_resume_exact_and_causal_inference(tmp_path):
    data, warm, settings = fixture(tmp_path); original = copy.deepcopy(warm)
    states = []
    for arm in api.ARMS:
        path = tmp_path/(arm+'.pt.gz'); kw = dict(settings=settings, identity={}, arm=arm)
        api.fit(data, warm, **kw, path=path, stop_at=2)
        a = api.fit(data, warm, **kw, path=path, resume=True)
        b = api.fit(data, warm, **kw, path=tmp_path/(arm+'_direct.pt.gz'))
        api.parent.parent.exact_state(a, b); states.append(a)
        before = api.predict(a, data['causal'])
        changed = copy.deepcopy(data); changed['truth_by_arm']['bound_aware'][:] = 999
        changed['observed_future'][:] = False
        api.base.exact(before, api.predict(a, changed['causal']))
        with pytest.raises(ValueError, match='causal'):
            api.predict(a, dict(data['causal'], future_endpoint=np.zeros((4, 2))))
    api.assert_matched(states); api.base.exact(warm, original)
    assert not torch.equal(states[0]['model']['risk.encoder.weight'], states[1]['model']['risk.encoder.weight'])
    assert api.source_metrics(states[0], data)['worst_source_objective'] >= 0


def test_partial_optimizer_failure_keeps_last_completed_checkpoint(tmp_path, monkeypatch):
    data, warm, settings = fixture(tmp_path); path = tmp_path/'risk.pt.gz'
    kw = dict(settings=settings, identity={}, arm='source_robust', path=path)
    api.fit(data, warm, **kw, stop_at=2); before = path.read_bytes()
    def fail(opt, *a, **kw):
        with torch.no_grad():
            opt.param_groups[0]['params'][0].add_(1)
        raise RuntimeError('partial update')
    monkeypatch.setattr(torch.optim.AdamW, 'step', fail)
    with pytest.raises(RuntimeError):
        api.fit(data, warm, **kw, resume=True)
    assert path.read_bytes() == before


def test_followup_screen_cannot_ignore_mean_or_harm_worsening():
    jobs = [dict(key=str(i)) for i in range(3)]; rows = []
    for j in jobs:
        for arm in api.ARMS:
            rows.append(dict(job=j['key'], arm=arm, final_sources=dict(worst_source_objective=2 if arm == api.ARMS[0] else 1),
                joint_attribution=dict(total_joint_cost=1, false_safe_cost=1)))
    assert api.screen(rows, jobs)['followup_readout_allowed']
    for row in rows:
        if row['arm'] == 'source_robust':
            row['joint_attribution']['false_safe_cost'] = 1.01
    assert not api.screen(rows, jobs)['followup_readout_allowed']


def test_full_source_decomposition_matches_original_fitting_mse(tmp_path):
    data, warm, settings = fixture(tmp_path)
    state = api.fit(data, warm, arm='mean_continue', settings=settings, identity={}, path=tmp_path/'mean.pt.gz')
    by_source = api.source_metrics(state, data)['by_source']
    c = data['causal']; counts = state['sampler']['counts']; group_sites = [c['sites'][g[0]] for g in c['groups']]
    expected = sum(v['risk_MSE']*counts[np.asarray(group_sites) == s].sum()/counts.sum() for s,v in by_source.items())
    assert expected == pytest.approx(api.full(state, data)['mean_MSE'], rel=1e-6)


def test_failed_screen_does_not_open_new_readout(tmp_path, monkeypatch):
    import json
    from scripts import run_m3w_moving_source_robust as run
    monkeypatch.setattr(run, 'PUBLIC', tmp_path)
    (tmp_path/'training_freeze.json').write_text(json.dumps(dict(complete_heads=6, all_before_validation=True)))
    (tmp_path/'fitting_screen.json').write_text(json.dumps(dict(followup_readout_allowed=False)))
    def forbidden(*a, **kw):
        raise AssertionError('New development readout forbidden after failed fitting screen')
    monkeypatch.setattr(run.readout, 'evaluate', forbidden)
    run.evaluate({}, None, lambda **kw: None, lambda: None)
    value = json.loads((tmp_path/'analysis.json').read_text())
    assert value['status'] == 'fitting_complete_held_readout_not_run'
    assert value['original_rules_unchanged'] and not value['identity_certification']
