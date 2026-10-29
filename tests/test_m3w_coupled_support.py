import copy
import numpy as np
import pytest

from src.evaluation import m3w_coupled_support as api
from tests.test_m3w_local_cost_analogue import inputs


def test_joint_optimum_matches_dense_triangle_and_respects_cap():
    means = np.array([[.2, .8], [-2, 3], [.9, -.1], [.1, .2]])
    caps = np.array([1., .3, .7, 0.])
    answer = api.joint(means, caps)
    for m, cap, pred in zip(means, caps, answer):
        g = np.linspace(0, cap, 501)
        h, e = np.meshgrid(g, g)
        loss = (h-m[0])**2+(e-m[1])**2
        assert ((pred-m)**2).sum() <= loss[e <= h].min()+1e-12
        assert 0 <= pred[1] <= pred[0] <= cap
    np.testing.assert_allclose(answer[0], [.5, .5])


def test_joint_uses_unclipped_average():
    np.testing.assert_allclose(api.joint(np.array([[-2., 3.]]), np.ones(1)), [[.5, .5]])
    with pytest.raises(ValueError): api.joint(np.array([[np.nan, 1]]), np.ones(1))


def test_alpha_is_quadratic_minimum_with_equal_recording_mass():
    error = np.array([[[1., 0.], [1., 0.], [-1., 0.]]])
    delta = np.array([[[-2., 0.], [-2., 0.], [1., 0.]]])
    recs = np.array([[0, 0, 1]])
    a, before, after = api.blend_coefficient(error, delta, recs, np.array([True]))
    assert a[0] == pytest.approx(.6)
    grid = np.linspace(0, 1, 10001)
    losses = .5*(1-2*grid)**2+.5*(-1+grid)**2
    assert after[0] <= losses.min()+1e-12
    assert after[0] < before[0]
    a, _, _ = api.blend_coefficient(error, delta*0, recs, np.array([True]))
    assert a[0] == 0


def test_nested_excludes_whole_record_and_adjusts_source_counts():
    recs = np.array([[0, 0, 1, 2]])
    sources = np.array([[0, 0, 0, 1]])
    counts = np.array([[10., 10., 10., 7.]])
    record_counts = np.array([[4., 4., 6., 7.]])
    target = np.array([[[.1, .1], [.2, .1], [.7, .3], [.9, .4]]])
    got, supported, weights = api.nested_fit(target, np.ones((1,4)), counts,
        record_counts, recs, sources, np.ones((1,4)))
    assert supported.tolist() == [True]
    assert not weights[0,0,:2].any() and not weights[0,1,:2].any()
    np.testing.assert_allclose(weights[0,0], [0,0,1/6,1/7])
    expected = api.joint(np.average(target[0,2:],axis=0,weights=[1/6,1/7])[None],np.ones(1))
    np.testing.assert_allclose(got[0,0],expected[0])
    changed = target.copy(); changed[0,:2] = 999
    new, _, _ = api.nested_fit(changed,np.ones((1,4)),counts,record_counts,recs,sources,np.ones((1,4)))
    np.testing.assert_array_equal(new[0,:2],got[0,:2])


def test_one_reference_record_cannot_learn_blend():
    target = np.ones((1,4,2)); recs = np.zeros((1,4),int)
    got, support, weights = api.nested_fit(target,np.ones((1,4)),np.full((1,4),4.),
        np.full((1,4),4.),recs,recs,np.ones((1,4)))
    assert not support.any() and not weights.any() and not got.any()


def test_query_future_and_knownness_do_not_enter_any_new_arm():
    b,q,p = inputs(); out = api.retrieve(b,q,p,k=4)
    changed = copy.deepcopy(b); changed['labels'][:20] = np.nan
    new = api.retrieve(changed,q,p,k=4)
    for arm in api.MODES:
        np.testing.assert_array_equal(out['predictions'][arm][:10],new['predictions'][arm][:10])
    np.testing.assert_array_equal(out['diagnostics']['alpha'][:10],new['diagnostics']['alpha'][:10])


def test_controls_exact_and_joint_arms_coherent():
    b,q,p = inputs(); out = api.retrieve(b,q,p,k=4)
    old = api.previous.retrieve(b,q,p,k=4)
    np.testing.assert_array_equal(out['predictions']['independent_cost'],old['predictions']['local_signed32'])
    assert out['references'] == old['references']
    for arm in api.JOINT_ARMS:
        v = out['predictions'][arm]
        assert np.all(v[:,1] <= v[:,0]+1e-12)
        assert np.all(v[:,0]+p[q,0] <= b['envelope'][q]+1e-12)
    assert np.all(out['diagnostics']['nested_loss_after'] <= out['diagnostics']['nested_loss_before']+1e-12)


def test_mean_and_shuffle_preserve_record_alpha_mass():
    b,q,p = inputs(); out = api.retrieve(b,q,p,k=4)
    d = out['diagnostics']
    for rec in np.unique(b['recordings'][q]):
        take = b['recordings'][q] == rec
        assert d['mean_alpha'][take].sum() == pytest.approx(d['alpha'][take].sum())
        np.testing.assert_array_equal(np.sort(d['alpha'][take]),np.sort(d['shuffled_alpha'][take]))


def test_unselected_targets_cannot_affect_fits():
    b,q,p = inputs(); out = api.retrieve(b,q,p,k=4)
    b['labels'][~q] = np.nan; new = api.retrieve(b,q,p,k=4)
    for arm in api.MODES: np.testing.assert_array_equal(out['predictions'][arm],new['predictions'][arm])


def test_units_rescale_and_zero_reference_support_returns_original_not_safe_zero():
    b,q,p = inputs(); out = api.retrieve(b,q,p,k=4)
    b['envelope'] *= 7; b['labels'] *= 7; b['scale'] *= 7; p *= 7
    new = api.retrieve(b,q,p,k=4)
    for arm in api.MODES:
        np.testing.assert_allclose(out['predictions'][arm]*7,new['predictions'][arm],atol=1e-12)
    b['labels'][20:] = np.nan; new = api.retrieve(b,q,p,k=32)
    assert not new['supported'].any()
    np.testing.assert_array_equal(new['predictions']['nested_support_blend'],p[q][:,[1,4]])
    assert np.isnan(new['predictions']['independent_cost']).all()


def test_incoherent_original_cannot_be_silently_repaired():
    b,q,p = inputs(); p[0,4] = p[0,1]+1
    with pytest.raises(ValueError,match='Frozen'): api.retrieve(b,q,p,k=4)
