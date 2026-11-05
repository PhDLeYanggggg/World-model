import copy
import json
import numpy as np
import pytest
import torch
from src.world_model import m3w_peer_participation_risk as api
from tests.test_m3w_proposal_context_risk import fixture


def test_factorial_contexts_include_floor_and_exclude_ego():
    h = torch.arange(6*7*2).float().reshape(6, 7, 2)
    seg = torch.tensor([0, 0, 0, 1, 1, 2]); mask = torch.zeros(6, 7)
    mask[0, 2] = mask[1, 6] = mask[3, 1] = 1
    active, complete = api.added_contexts(h, mask, seg, 3)
    torch.testing.assert_close(active[0], h[1])
    torch.testing.assert_close(active[1], h[0])
    torch.testing.assert_close(active[2], (h[0]+h[1])/2)
    assert not active[3].any() and not active[5].any()
    torch.testing.assert_close(complete[0], ((h[1, 6]+h[2, 0])/2).expand(7, 2))
    torch.testing.assert_close(complete[3], h[4, 0].expand(7, 2))
    assert not complete[5].any()
    empty = api.added_contexts(h, torch.zeros_like(mask), seg, 3)
    assert not empty[0].any()
    torch.testing.assert_close(empty[1][0], ((h[1, 0]+h[2, 0])/2).expand(7, 2))
    changed = h.clone(); changed[0] += 1000; changed[3:] += 2000
    torch.testing.assert_close(api.added_contexts(changed, mask, seg, 3)[1][0], complete[0])
    order = torch.tensor([2, 0, 1, 4, 3, 5])
    a, b = api.added_contexts(h[order], mask[order], seg[order], 3)
    torch.testing.assert_close(a, active[order]); torch.testing.assert_close(b, complete[order])


def test_floor_context_gets_gradient_not_inactive_proposal():
    h = torch.randn(3, 7, 4, requires_grad=True); mask = torch.zeros(3, 7); mask[0, 2] = 1
    api.added_contexts(h, mask, torch.zeros(3, dtype=torch.long), 1)[1][0].sum().backward()
    assert not h.grad[0].any()
    assert h.grad[1:, 0].abs().sum() > 0 and not h.grad[1:, 1:].any()


def test_four_arms_initial_function_and_legacy_exact(tmp_path):
    d, warm, _ = fixture(tmp_path); z = api.validate(d, warm); seg = torch.tensor([0, 0, 1, 1])
    original = api.parent.model_from(warm)(z[..., :524], seg, 2)
    initial = None
    for arm in api.ARMS:
        model = api.initialize(warm, arm)
        api.base.exact(model(z, seg, 2), original)
        if initial is not None:
            api.base.exact(model.state_dict(), initial)
        initial = copy.deepcopy(model.state_dict())
        assert sum(p.numel() for p in model.parameters()) == warm['parameters']+32
        if arm in api.previous.ARMS:
            api.base.exact(model(z, seg, 2), api.previous.initialize(warm, arm)(z, seg, 2))
    illegal = z.clone(); illegal[0, 0, -1] = 1
    with pytest.raises(ValueError, match='proposal'):
        api.initialize(warm, 'proposed_all')(illegal, seg, 2)


def test_matched_engine_resume_and_no_future_input(tmp_path):
    d, warm, settings = fixture(tmp_path); before = copy.deepcopy(warm); states = []
    proposal = api.proposal_from_mask(d['causal']['allowed'])
    for arm in api.ARMS:
        path = tmp_path/(arm+'.pt.gz'); kw = dict(arm=arm, settings=settings, identity=dict(job='same', registration_sha256=arm))
        api.fit(d, warm, **kw, path=path, stop_at=2)
        a = api.fit(d, warm, **kw, path=path, resume=True)
        b = api.fit(d, warm, **kw, path=tmp_path/(arm+'_direct.pt.gz'))
        api.parent.parent.exact_state(a, b); states.append(a)
        if arm in api.previous.ARMS:
            old = api.previous.fit(d, warm, **kw, path=tmp_path/(arm+'_old.pt.gz'))
            api.parent.parent.exact_state(a, old)
            api.base.exact(api.predict(a, d['causal'], proposal), api.previous.predict(old, d['causal'], proposal))
        changed = copy.deepcopy(d); changed['truth_by_arm']['bound_aware'][:] = 999
        changed['observed_future'][:] = False
        api.base.exact(api.predict(a, d['causal'], proposal), api.predict(a, changed['causal'], proposal))
        with pytest.raises(ValueError, match='causal'):
            api.predict(a, dict(d['causal'], future_endpoint=np.zeros((4, 2))), proposal)
        for head in ('risk', 'decision'):
            assert a['model'][head+'.bridge.weight'].abs().sum() > 0
            assert not torch.equal(a['model'][head+'.encoder.weight'], a['initial_model'][head+'.encoder.weight'])
        assert api.full(a, d)['mean_MSE'] >= 0
        assert api.attribution(a, d)['total_joint_cost'] >= 0
    api.assert_matched(states); api.base.exact(warm, before)
    bad = copy.deepcopy(states); bad[-1]['identity']['job'] = 'different'
    with pytest.raises(AssertionError):
        api.assert_matched(bad)


def test_partial_optimizer_failure_preserves_checkpoint(tmp_path, monkeypatch):
    d, warm, settings = fixture(tmp_path); path = tmp_path/'test.pt.gz'
    kw = dict(arm='proposed_all', settings=settings, identity={}, path=path)
    api.fit(d, warm, **kw, stop_at=2); before = path.read_bytes()
    def fail(opt, *a, **kw):
        with torch.no_grad():
            opt.param_groups[0]['params'][0].add_(1)
        raise RuntimeError('partial update')
    monkeypatch.setattr(torch.optim.AdamW, 'step', fail)
    with pytest.raises(RuntimeError):
        api.fit(d, warm, **kw, resume=True)
    assert path.read_bytes() == before


def test_primary_screen_cannot_choose_favorable_secondary():
    jobs = [dict(key=str(i)) for i in range(3)]; rows = []
    for j in jobs:
        for arm in api.ARMS:
            rows.append(dict(job=j['key'], arm=arm, final_fitting=dict(mean_MSE=1),
                joint_attribution=dict(total_joint_cost=.5 if arm in ('same_active', 'proposed_action') else 1., false_safe_cost=1)))
    assert not api.screen(rows, jobs)['followup_readout_allowed']
    for r in rows:
        if r['arm'] == 'proposed_all':
            r['joint_attribution']['total_joint_cost'] = .95
    assert api.screen(rows, jobs)['followup_readout_allowed']
    for r in rows:
        if r['arm'] == 'proposed_all':
            r['joint_attribution']['false_safe_cost'] = 1.01
    assert not api.screen(rows, jobs)['followup_readout_allowed']


def test_failed_screen_never_accesses_held_readout(tmp_path, monkeypatch):
    from scripts import run_m3w_moving_peer_participation as run
    monkeypatch.setattr(run, 'PUBLIC', tmp_path)
    (tmp_path/'training_freeze.json').write_text(json.dumps(dict(fresh_heads=6, cached_verified_heads=6, all_before_validation=True)))
    (tmp_path/'fitting_screen.json').write_text(json.dumps(dict(followup_readout_allowed=False)))
    def forbidden(*a, **kw):
        raise AssertionError('No new held readout')
    monkeypatch.setattr(run.readout, 'evaluate', forbidden)
    run.evaluate({}, None, lambda **kw:None, lambda:None)
    r = json.loads((tmp_path/'analysis.json').read_text())
    assert r['status'] == 'factorial_fitting_complete_held_readout_not_run'
    assert not r['identity_certification'] and r['original_rules_unchanged']
