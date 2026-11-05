import copy
import numpy as np
import pytest
import torch
from tests.test_m3w_harm_floor_risk import prepared
from src.evaluation import m3w_paired_gradient as diagnostic

api = diagnostic.api


def test_weighted_terms_reproduce_original_equal_query_objective():
    risk = torch.randn(6, 7, 2, requires_grad=True)
    decision = torch.randn(6, 7, 2, requires_grad=True)
    y = torch.zeros(6, 7, 2); y[:, 0, 0] = torch.tensor([1., -2., 1., -3., .5, -.4])
    allowed = torch.zeros(6, 7, dtype=torch.bool); allowed[:, 0] = True
    observed = torch.tensor([True, True, True, False, True, True])
    segments = torch.tensor([0, 0, 1, 1, 2, 2])
    terms = diagnostic.weighted_terms((risk, decision), y, allowed, observed, segments, torch.ones(3).double()/3)
    reduce = lambda e: api.parent.aggregate(e, allowed, segments, 3)
    torch.testing.assert_close(terms['risk'].float(), .5*reduce((risk-y).square()))
    torch.testing.assert_close(terms['decision'].float(), .5*reduce(torch.where(observed[:, None, None],
        api.parent.signed_cost(decision, y), (decision-y).square())))
    torch.testing.assert_close(terms['joint'].float(), .5*reduce(torch.where(observed[:, None, None],
        api.previous.joint_cost(risk, decision, y), 0.)))
    for step in (0, 1999):
        score = torch.maximum(risk, decision).amax(-1)
        for mode in ('point', 'rank'):
            expected = score.sum()*0
            for _, p, n, w in api.pairs(y, allowed, observed, segments, 3, step):
                expected += .5*api.contrast(score[p[:, 0], p[:, 1]], score[n[:, 0], n[:, 1]],
                    w, 'paired_'+mode).mean()/3
            torch.testing.assert_close(terms[mode+str(step)].float(), expected)


def test_projection_protects_only_risk_auxiliary_and_zero_norm_is_defined():
    names = ['risk.weight', 'decision.weight']
    g = {key: [torch.tensor([1., 0.]).double(), torch.zeros(2).double()]
        for key in diagnostic.TERMS}
    g['rank1999'] = [torch.tensor([-2., 3.]).double(), torch.tensor([4., 5.]).double()]
    full, _ = diagnostic.combine(g, names, 'rank')
    projected, relation = diagnostic.combine(g, names, 'risk_protected_rank')
    assert relation['first_order_conflict']
    torch.testing.assert_close(projected[0], torch.tensor([3., 3.]).double())
    torch.testing.assert_close(projected[1], full[1])
    detached, _ = diagnostic.combine(g, names, 'risk_detached_rank')
    torch.testing.assert_close(detached[0], torch.tensor([3., 0.]).double())
    assert diagnostic.geometry([torch.zeros(2)], [torch.ones(2)])['cosine'] is None


def test_unknown_rows_do_not_enter_pairs_and_virtual_step_does_not_mutate_saved_model(tmp_path):
    torch.set_num_threads(1); data, origin, settings = prepared(tmp_path)
    state = api.fit(data, origin, arm='paired_rank', settings=settings, identity={}, path=tmp_path/'fit.pt.gz')
    saved = copy.deepcopy(state)
    values, gradients, names = diagnostic.full(state, data)
    assert all(np.isfinite(list(values.values())))
    for variant in diagnostic.VARIANTS:
        changed, _ = diagnostic.virtual(state, gradients, names, variant)
        assert any(not torch.equal(changed['model'][k], state['model'][k]) for k in state['model'])
        api.parent.exact_state(state, saved)
    data['observed_future'][:] = False
    with pytest.raises(AssertionError):
        diagnostic.full(state, data)
    z = api.parent.kin.batch_features(data, 'kinematic')
    for ix, seg in api.parent.batches(data['causal']):
        count = int(seg.max())+1
        y = api.parent.kin.parent.signed(data['truth_by_arm']['bound_aware'][ix]) / torch.from_numpy(
            api.parent.kin.parent.normalizer(data['preprocess']))
        values = diagnostic.weighted_terms(api.model_from(state)(z[ix], torch.from_numpy(seg), count),
            y, torch.from_numpy(data['causal']['allowed'][ix]),
            torch.from_numpy(data['observed_future'][ix]), torch.from_numpy(seg), torch.ones(count).double()/count)
        assert all(float(values[k].detach()) == 0 for k in ('point0', 'rank0', 'point1999', 'rank1999', 'joint'))
