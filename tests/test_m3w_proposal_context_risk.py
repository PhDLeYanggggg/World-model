import copy
import numpy as np
import pytest
import torch
from src.world_model import m3w_proposal_context_risk as api
from src.world_model import m3w_admission_priority as setup
from tests.test_m3w_kinematic_risk import fixture as raw_fixture


def fixture(tmp_path):
    torch.set_num_threads(1); data = raw_fixture()
    data['causal']['allowed'][:] = False
    data['causal']['allowed'][[0, 1, 2], [1, 3, 6]] = True
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


def test_explicit_proposal_contract_not_legal_mask():
    mask = np.zeros((4, 7), bool); mask[[0, 1, 2], [1, 3, 6]] = True
    np.testing.assert_array_equal(api.proposal_from_mask(mask), [1, 3, 6, -1])
    c = dict(x=np.zeros((4, 7, 428)), allowed=mask)
    np.testing.assert_array_equal(api.proposal_mask(c, np.array([1, 3, 6, -1])), mask)
    mask[0, 2] = True
    with pytest.raises(ValueError, match='legal-action'):
        api.proposal_from_mask(mask)
    for bad in (np.array([0, 3, 6, -1]), np.array([7, 3, 6, -1]), np.array([1., 3, 6, -1]), np.array([3, 3, 6, -1])):
        with pytest.raises(ValueError):
            api.proposal_mask(c, bad)


def test_peer_pool_excludes_ego_inactive_peers_and_other_queries():
    h = torch.arange(5*7*2).float().reshape(5, 7, 2)
    seg = torch.tensor([0, 0, 0, 1, 1]); mask = torch.zeros(5, 7)
    mask[0, 2] = mask[1, 6] = mask[3, 1] = 1
    same, proposed = api.peer_context(h, mask, seg, 2)
    torch.testing.assert_close(proposed[0], h[1, 6].expand(7, 2))
    torch.testing.assert_close(proposed[1], h[0, 2].expand(7, 2))
    torch.testing.assert_close(proposed[2], ((h[0, 2]+h[1, 6])/2).expand(7, 2))
    assert not proposed[3].any()
    torch.testing.assert_close(proposed[4], h[3, 1].expand(7, 2))
    torch.testing.assert_close(same[0], (h[1]+h[2])/2)
    changed = h.clone(); changed[2] += 1000; changed[3:] += 2000
    torch.testing.assert_close(api.peer_context(changed, mask, seg, 2)[1][0], proposed[0])
    assert not api.peer_context(h, torch.zeros_like(mask), seg, 2)[1].any()
    order = torch.tensor([2, 0, 1, 4, 3])
    a, b = api.peer_context(h[order], mask[order], seg[order], 2)
    torch.testing.assert_close(a, same[order]); torch.testing.assert_close(b, proposed[order])


def test_matched_capacity_initial_parent_identity_and_proposal_effect(tmp_path):
    d, warm, _ = fixture(tmp_path); z = api.validate(d, warm); seg = torch.tensor([0, 0, 1, 1])
    original = api.parent.model_from(warm)(z[..., :524], seg, 2)
    a, b = (api.initialize(warm, arm) for arm in api.ARMS)
    api.base.exact(a.state_dict(), b.state_dict())
    assert sum(p.numel() for p in a.parameters()) == warm['parameters']+2*4*4
    for m in (a, b):
        api.base.exact(m(z, seg, 2), original)
        torch.nn.init.eye_(m.risk.bridge.weight)
    changed = z.clone(); changed[1, :, -1] = 0; changed[1, 1, -1] = 1
    api.base.exact(a(z, seg, 2), a(changed, seg, 2))
    assert not torch.equal(b(z, seg, 2)[0][0], b(changed, seg, 2)[0][0])


def test_exact_resume_matched_updates_and_no_future_inference(tmp_path):
    d, warm, settings = fixture(tmp_path); original = copy.deepcopy(warm); states = []
    proposal = api.proposal_from_mask(d['causal']['allowed'])
    for arm in api.ARMS:
        path = tmp_path/(arm+'.pt.gz'); kw = dict(arm=arm, settings=settings, identity={})
        api.fit(d, warm, **kw, path=path, stop_at=2)
        a = api.fit(d, warm, **kw, path=path, resume=True)
        b = api.fit(d, warm, **kw, path=tmp_path/(arm+'_direct.pt.gz'))
        api.parent.parent.exact_state(a, b); states.append(a)
        pred = api.predict(a, d['causal'], proposal)
        changed = copy.deepcopy(d); changed['truth_by_arm']['bound_aware'][:] = 999
        changed['observed_future'][:] = False
        api.base.exact(pred, api.predict(a, changed['causal'], proposal))
        with pytest.raises(ValueError, match='causal'):
            api.predict(a, dict(d['causal'], future_endpoint=np.zeros((4, 2))), proposal)
        for head in ('risk', 'decision'):
            assert a['model'][head+'.bridge.weight'].abs().sum() > 0
            assert not torch.equal(a['model'][head+'.encoder.weight'], a['initial_model'][head+'.encoder.weight'])
        assert api.full(a, d)['mean_MSE'] >= 0
        assert api.attribution(a, d)['total_joint_cost'] >= 0
    api.assert_matched(states); api.base.exact(warm, original)


def test_partial_update_failure_does_not_corrupt_checkpoint(tmp_path, monkeypatch):
    data, warm, settings = fixture(tmp_path); path = tmp_path/'test.pt.gz'
    kw = dict(arm='proposed_action', settings=settings, identity={}, path=path)
    api.fit(data, warm, **kw, stop_at=2); before = path.read_bytes()
    def fail(opt, *a, **kw):
        with torch.no_grad():
            opt.param_groups[0]['params'][0].add_(1)
        raise RuntimeError('partial update')
    monkeypatch.setattr(torch.optim.AdamW, 'step', fail)
    with pytest.raises(RuntimeError):
        api.fit(data, warm, **kw, resume=True)
    assert path.read_bytes() == before


def test_screen_retains_mse_and_false_safe_requirements():
    jobs = [dict(key=str(i)) for i in range(3)]; rows = []
    for j in jobs:
        for arm in api.ARMS:
            rows.append(dict(job=j['key'], arm=arm, final_fitting=dict(mean_MSE=1),
                joint_attribution=dict(total_joint_cost=1 if arm == api.ARMS[0] else .9, false_safe_cost=1)))
    assert api.screen(rows, jobs)['followup_readout_allowed']
    for r in rows:
        if r['arm'] == 'proposed_action':
            r['joint_attribution']['false_safe_cost'] = 1.01
    assert not api.screen(rows, jobs)['followup_readout_allowed']


def test_failed_screen_does_not_access_readout(tmp_path, monkeypatch):
    import json
    from scripts import run_m3w_moving_proposal_context as run
    monkeypatch.setattr(run, 'PUBLIC', tmp_path)
    (tmp_path/'training_freeze.json').write_text(json.dumps(dict(complete_heads=6, all_before_validation=True)))
    (tmp_path/'fitting_screen.json').write_text(json.dumps(dict(followup_readout_allowed=False)))
    def forbidden(*a, **kw):
        raise AssertionError('No extra readout permitted')
    monkeypatch.setattr(run.readout, 'evaluate', forbidden)
    run.evaluate({}, None, lambda **kw:None, lambda:None)
    result = json.loads((tmp_path/'analysis.json').read_text())
    assert result['status'] == 'fitting_complete_held_readout_not_run'
    assert not result['identity_certification'] and result['original_rules_unchanged']
