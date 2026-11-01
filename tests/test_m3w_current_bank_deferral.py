import copy
import numpy as np
import pytest

from src.world_model import m3w_current_bank_deferral as api
from tests.test_m3w_own_decision_cost import data as base_data, warm, original_settings


def sample():
    d = base_data(); y = np.ones((8, 8))
    y[:, 1:] += np.arange(7) / 10
    y[:, 0] = 1.; y[~d['known']] = np.nan
    y[:, 1:] = np.where(d['causal']['allowed'], y[:, 1:], y[:, :1])
    return dict(d, costs=y, input_hashes=dict(d['input_hashes'], canonical_ADE_costs=api.base.digest(y)))


def test_canonical_costs_trace_ids_unknown_and_unavailable_floor_aliases():
    d = base_data(); c = d['causal']; floor = np.ones(8); floor[~d['known']] = np.nan
    packet = dict(ids=c['ids'], allowed=c['allowed'], floor_ade=floor, ade=np.tile(floor[:, None], (1, 7)) + .1)
    v = api.training_data(d, {'s': packet}, {'s': slice(0, 8)})
    assert np.isnan(v['costs'][~d['known']]).all()
    np.testing.assert_array_equal(v['costs'][:, 1:][~c['allowed']], np.broadcast_to(floor[:, None], (8, 7))[~c['allowed']])
    packet['ids'] = packet['ids'][::-1]
    with pytest.raises(AssertionError):
        api.training_data(d, {'s': packet}, {'s': slice(0, 8)})


@pytest.mark.parametrize('arm', api.ARMS)
def test_unknown_output_supervised_gradient_zero_and_complete_query_reduction(arm):
    d = sample(); scores = api.torch.zeros((8, 8), requires_grad=True, dtype=api.torch.float64)
    segments = api.torch.repeat_interleave(api.torch.arange(4), 2)
    y = api.torch.from_numpy(d['costs']); known = api.torch.from_numpy(d['known'])
    value = api.loss(scores, y, known, segments, 4, arm)
    manual = sum(api.loss(scores[segments == q], y[segments == q], known[segments == q],
        api.torch.zeros(2, dtype=api.torch.int64), 1, arm) for q in range(4)) / 4
    api.torch.testing.assert_close(value, manual, rtol=0, atol=1e-15)
    value.backward()
    assert api.torch.isfinite(scores.grad).all() and (scores.grad[~known] == 0).all()


def test_real_published_surrogate_reused_not_best_class_cross_entropy():
    s = api.torch.tensor([[.2, -.1, .3, 0., .2, -.2, .1, .4]], requires_grad=True)
    y = api.torch.tensor([[1., 2., 1.5, .5, 2., 3., 4., 1.]])
    a = api.loss(s, y, api.torch.ones(1, dtype=api.torch.bool), api.torch.zeros(1, dtype=api.torch.int64), 1, 'tdef')
    b = api.published.deferral_loss(s, y)
    api.torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not api.torch.isclose(a, api.torch.nn.functional.cross_entropy(s, y.argmin(1)))


@pytest.mark.parametrize('arm', api.ARMS)
def test_real_training_and_exact_resume_recovery(tmp_path, arm):
    d = sample(); w = warm(base_data()); kw = dict(data=d, warm=w, arm=arm, settings=original_settings(), identity={})
    direct = api.fit(path=tmp_path / 'direct.pt.gz', **kw)
    api.fit(path=tmp_path / 'resume.pt.gz', stop_at=1, **kw)
    recovered = api.fit(path=tmp_path / 'resume.pt.gz', resume=True, **kw)
    api.assert_replay(direct, recovered)
    assert direct['unknown_rows_drawn'] > 0 and direct['unknown_rows_with_supervised_labels'] == 0
    assert any(not api.torch.equal(v, direct['initial_model'][k]) for k, v in direct['model'].items())
    scores = api.predict(direct, d['causal']); assert scores.shape == (8, 8)
    np.testing.assert_array_equal(scores[:, 1:][~d['causal']['allowed']], np.broadcast_to(scores[:, :1], (8, 7))[~d['causal']['allowed']])
    with pytest.raises(ValueError):
        api.predict(direct, dict(d['causal'], future_endpoint=np.zeros((8, 2))))


def test_matched_capacity_draws_and_causal_guard(tmp_path):
    d = sample(); states = [api.fit(d, warm=warm(base_data()), arm=a, settings=original_settings(), identity={},
        path=tmp_path / (a + '.pt.gz')) for a in api.ARMS]
    api.assert_matched(*states)
    assert sum(p.numel() for p in api.QueryScores(462, 64, 17).parameters()) == 30726
    risk = np.broadcast_to([.1, .001, 1., .5, .0001], (8, 7, 5)).copy()
    risk[:, 6, 1] = 10.
    scores = np.zeros((8, 8)); scores[:, 7] = 100.; scores[:, 2] = 1.
    choices = api.choices(scores, np.ones((8, 7), bool), risk)
    assert (choices['raw'][0] == 6).all() and (choices['guarded'][0] == 1).all()
    assert api.parent.allocation.base.feasible(risk, np.ones((8, 7), bool), choices['guarded'][0])
    with pytest.raises(ValueError):
        api.fit(d, warm=warm(base_data()), arm='invalid', settings=original_settings(), identity={}, path=tmp_path/'bad')
