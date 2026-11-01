import copy
import numpy as np
import pytest
import torch
from src.world_model import m3w_decision_cost_critic as api
from tests.test_m3w_coupled_benefit_risk import data
from tests.test_m3w_selected_query_critic import settings


DECISION = dict(cost_weight=1., dual_every=2, dual_learning_rate=1., dual_maximum=100.)


def test_probabilities_have_fallback_and_respect_only_causal_availability():
    p = torch.zeros((3, 7, 5), dtype=torch.float64, requires_grad=True)
    allowed = torch.ones((3, 7), dtype=torch.bool); allowed[1] = False
    q = api.probabilities(p, allowed, 1.)
    torch.testing.assert_close(q[0], torch.ones(8, dtype=torch.float64)/8)
    assert q[1, 0] == 1 and q[1, 1:].sum() == 0
    q[:, 1].sum().backward(); assert torch.isfinite(p.grad).all()


def test_unknowns_receive_bounds_not_zero_imputed_outcomes():
    d = data(); lay = api.training_layout(d); unknown = ~d['known']; c = d['causal']
    assert unknown.any()
    np.testing.assert_array_equal(lay['gain'][unknown], -c['envelope'][unknown])
    for k in range(3): np.testing.assert_array_equal(lay['risk'][unknown, :, k], c['envelope'][unknown])
    assert np.isfinite(lay['gain']).all() and np.isfinite(lay['risk']).all()


def test_query_sampling_weights_equal_exact_source_and_record_objectives():
    d = data(); c = d['causal']; lay = api.training_layout(d)
    rng = np.random.default_rng(13); prob = rng.random((len(c['ids']), 7)); prob /= 2*prob.sum(1)[:, None]
    gain = (prob*lay['gain']).sum(1); risk = (prob[:, :, None]*lay['risk']).sum(1)
    sampled_gain = 0.; sampled_risk = np.zeros(3)
    for groups in c['source_groups']:
        for qid in groups:
            ix = c['groups'][qid]; weight = 1/(len(c['source_groups'])*len(groups))
            sampled_gain += weight*(gain[ix]*lay['objective_weight'][ix]).sum()
            sampled_risk += weight*(risk[ix]*lay['risk_weight'][ix, None]).sum(0)
    exact_gain = np.mean([gain[c['sites'] == s].sum()/np.where(d['known'], d['targets'][:, 0, 2], 0.)[c['sites'] == s].sum()
        for s in sorted(set(c['sites']))])
    record_risk = np.zeros((len(lay['record_keys']), 3)); np.add.at(record_risk, lay['record'], risk)
    exact_risk = (record_risk/lay['record_scales'][:, None]*lay['record_weights'][:, None]).sum(0)
    np.testing.assert_allclose(sampled_gain, exact_gain, rtol=1e-14)
    np.testing.assert_allclose(sampled_risk, exact_risk, rtol=1e-14)


def test_full_risk_matches_independent_exact_record_sums_including_easy_constant():
    d = data(); c = d['causal']; lay = api.training_layout(d)
    model = api.base.CoupledCritic(4, 8, 17, 'released_benefit')
    allp = np.zeros((len(c['ids']), 7))
    for g in c['groups']:
        args = api.parent.batch(c, g, np.zeros(len(g), int), 1)
        with torch.no_grad(): allp[g] = api.probabilities(model(*args), torch.from_numpy(c['allowed'][g]), lay['temperature'])[:, 1:].numpy()
    expected = np.zeros((len(lay['record_keys']), 3))
    np.add.at(expected, lay['record'], (allp[:, :, None]*lay['risk']).sum(1)); expected[:, 2] -= lay['easy_constant']
    risk, mass = api.full_risk(model, d, lay, chunk_rows=2)
    np.testing.assert_allclose(risk, expected/lay['record_scales'][:, None], rtol=1e-13)
    assert mass == pytest.approx(allp.sum()/len(allp))


def test_cost_term_has_finite_nonzero_gradient_and_unknowns_never_supervise():
    d = data(); c = d['causal']; rows = np.arange(8); segments = torch.from_numpy(np.repeat(np.arange(4), 2))
    args = api.parent.batch(c, rows, segments, 4); model = api.base.CoupledCritic(4, 8, 17, 'released_benefit')
    t = api.losses(model(*args), d, rows, segments, 4, api.base.units.query_normalizer(d), api.training_layout(d), torch.ones((4, 3)), decision_weight=1.)
    t['decision_cost'].backward()
    assert t['unknown_supervised_rows'] == 0 and t['unknown_bound_rows'] > 0
    assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None) > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_zero_decision_weight_reproduces_original_regression_exactly(tmp_path):
    d = data(); old = api.base.fit(d, mode='released_benefit', seed=17, settings=settings(), identity={}, path=tmp_path/'old.pt.gz')
    new = api.fit(d, seed=17, settings=settings(), decision=DECISION, identity={}, path=tmp_path/'new.pt.gz', regression_control=True)
    api.assert_matched(new, old); api.parent.exact(old['model'], new['model'])


def test_resume_reproduces_every_scientific_field_and_dual_exactly(tmp_path):
    d = data(); kw = dict(seed=17, settings=settings(), decision=DECISION, identity={})
    direct = api.fit(d, path=tmp_path/'direct.pt.gz', **kw)
    api.fit(d, path=tmp_path/'resumed.pt.gz', stop_at=2, **kw)
    resumed = api.fit(d, path=tmp_path/'resumed.pt.gz', resume=True, **kw)
    for k in direct:
        if k != 'seconds': api.parent.exact(direct[k], resumed[k])
    assert float(direct['dual'].max()) > 0
    prediction = api.predict(direct, d['causal'])
    np.testing.assert_array_equal(prediction[:, :, [2, 3]], d['causal']['prediction'][:, :, [2, 3]])
    assert direct['unknown_rows_drawn'] > 0 and direct['unknown_rows_with_loss'] == 0
    with pytest.raises(ValueError): api.predict(direct, dict(d['causal'], future_endpoint=np.zeros((8, 2))))
    with pytest.raises(ValueError): api.predict(direct, dict(d['causal'], outcome_mask=d['known']))


def test_changed_resume_definition_rejected(tmp_path):
    d = data(); path = tmp_path/'head.pt.gz'; kw = dict(seed=17, settings=settings(), identity={})
    api.fit(d, path=path, decision=DECISION, stop_at=2, **kw)
    with pytest.raises(AssertionError): api.fit(d, path=path, decision=dict(DECISION, cost_weight=2.), resume=True, **kw)
