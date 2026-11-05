import numpy as np
import pytest
import torch

from src.world_model import m3w_proposal_conditioned_risk as api
from src.world_model import m3w_bound_aware_cost as cost
from tests.test_m3w_moving_deferral import packet


def learning():
    d = packet()
    d['defined_by_arm'] = dict(bound_aware=np.ones(4, bool))
    d['preprocess'].update(rms=np.ones(5, np.float32), bias=np.ones(5, np.float32), source_names=['s'])
    return d


def test_mask_is_only_causal_and_retains_query_roster_and_unknown_bounds():
    d = learning(); d['observed_future'][0] = False
    logits = np.zeros((4, 7)); logits[0, 6] = 1; logits[2, 3] = 1
    a, choice = api.condition(d, logits)
    assert choice.tolist() == [6, -1, 3, -1]
    assert a['causal']['allowed'].sum() == 2
    assert a['causal']['groups'] is d['causal']['groups']
    assert a['z'] is d['z'] and a['preprocess'] is d['preprocess']
    pred = torch.ones(4, 7, 5, requires_grad=True)
    terms = cost.objective(pred, a, np.arange(4), torch.tensor([0, 0, 1, 1]), 2, 'bound_aware')
    assert terms['unknown_bound_supervised_rows'] == 1
    assert terms['observed_supervised_rows'] == 1
    terms['total'].backward()
    assert not pred.grad[1].any() and not pred.grad[3].any()
    assert not pred.grad[0, :6].any() and pred.grad[0, 6].any()
    assert d['causal']['allowed'].sum() == 24
    d['truth_by_arm']['bound_aware'][:] = 1000
    b, second = api.condition(d, logits)
    np.testing.assert_array_equal(choice, second)
    np.testing.assert_array_equal(a['causal']['allowed'], b['causal']['allowed'])
    with pytest.raises(ValueError, match='causal schema'):
        api.condition(dict(d, causal=dict(d['causal'], future_endpoint=np.ones((4, 2)))), logits)


def test_source_isolation_and_original_veto_matching():
    d = learning(); c = d['causal']
    state = dict(fit_ids=np.array([9]), preprocess=dict(source_names=['other']), source_sites=['other'])
    api.assert_excluded(state, c, ['s'])
    with pytest.raises(ValueError, match='disjoint'):
        api.assert_excluded(dict(state, fit_ids=np.array([0])), c, ['s'])
    logits = np.tile(np.arange(7), (4, 1)).astype(float)
    p = np.zeros((4, 7, 5)); p[..., 2:4] = 1
    q = p.copy(); q[0, :, 1] = .02001; q[2, :, 4] = .02; q[3, :, 3] = 0
    choices = api.causal_choices(logits, p, q, c)
    assert choices['conditioned_veto'].tolist() == [-1, 6, 6, -1]
    for group in c['groups']:
        np.testing.assert_array_equal(np.sort(choices['conditioned_veto'][group]),
            np.sort(choices['conditioned_matched'][group]))


def test_resume_exact_and_no_query_sampling_change(tmp_path):
    torch.set_num_threads(1)
    d = learning(); logits = np.tile(np.arange(7), (4, 1)).astype(float)
    selected, _ = api.condition(d, logits)
    settings = dict(width=4, learning_rate=.001, query_batch_size=2, minimum_updates=3,
        largest_source_cycles=1, gradient_clip=5., heartbeat_every=1, checkpoint_every=1)
    path = tmp_path/'resume.pt.gz'
    args = dict(seed=17, settings=settings, identity={'proposal_hash': 'fixed'})
    api.fit(selected, **args, path=path, stop_at=1)
    a = api.fit(selected, **args, path=path, resume=True)
    b = api.fit(selected, **args, path=tmp_path/'direct.pt.gz')
    api.router.exact_state(a, b)
    all_action = api.fit(d, **args, path=tmp_path/'control.pt.gz')
    for key in ('initial_model', 'preprocess', 'sampler', 'draw_hash', 'fit_ids'):
        api.base.exact(b[key], all_action[key])
    np.testing.assert_array_equal(api.predict(a, d['causal']), api.predict(b, d['causal']))
