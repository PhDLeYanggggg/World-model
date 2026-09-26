import copy
import numpy as np
import pytest
from src.world_model import m3w_nested_residual as m
from src.world_model import m3w_context_residual as residual
from src.world_model import m3w_membership_auxiliary as neural


def fixture():
    rng = np.random.default_rng(123); n = 120
    sites = np.repeat(['a', 'b', 'c'], 40)
    x = rng.normal(size=(n, 7)); cv = np.tile(np.linspace(.1, 4, 40), 3)
    raw = np.column_stack((cv, np.full(n, .3)))
    return x, raw, cv, sites


def bank(sites):
    return {s:dict(prediction=np.tile([3, 2, 1, .1*(i+1)], (len(sites), 1)),
                   training_sites=[v for v in 'abc' if v != s], cut=.5+i*.2)
            for i,s in enumerate('abc')}


def test_inner_labels_stats_ignore_inner_held_values():
    x, raw, cv, sites = fixture(); a = m.inner_inputs(x, raw, cv, sites, 'd', 'a')
    x[sites == 'a'] += 1e6; raw[sites == 'a'] += 9e6; cv[sites == 'a'] += 8e6
    b = m.inner_inputs(x, raw, cv, sites, 'd', 'a')
    for k in a[1]: np.testing.assert_array_equal(a[1][k], b[1][k])
    for i in (0,2,3): np.testing.assert_array_equal(a[i], b[i])
    assert a[1]['training_sites'] == ['b','c']


@pytest.mark.parametrize('variant,offset', [('oof',0),('in_sample_next',1),('in_sample_prev',-1)])
def test_producer_routing_and_row_permutation(variant, offset):
    x, raw, cv, sites = fixture(); bb = bank(sites)
    p,y,cuts,producers = m.assemble(bb, raw, cv, sites, 'd', variant)
    for i,s in enumerate('abc'):
        h = 'abc'[(i+offset)%3]; use = sites == s
        assert (producers[use] == h).all()
        np.testing.assert_array_equal(p[use], bb[h]['prediction'][use])
        assert ((s in bb[h]['training_sites']) == (variant != 'oof'))
    order = np.random.default_rng(6).permutation(len(sites))
    cc = copy.deepcopy(bb)
    for v in cc.values(): v['prediction'] = v['prediction'][order]
    z = m.assemble(cc, raw[order], cv[order], sites[order], 'd', variant)
    for expected,actual in zip((p,y,cuts,producers), z): np.testing.assert_array_equal(expected[order], actual)


def test_lineage_guards_and_unknown_labels():
    x, raw, cv, sites = fixture(); raw[0] = np.nan; cv[0] = np.nan
    take,pr,y,e = m.inner_inputs(x, raw, cv, sites, 'd', 'b')
    assert pr['weights'][0] == 0 and np.isnan(y[0]).all() and np.isnan(e[0])
    with pytest.raises(ValueError): m.inner_inputs(x, raw, cv, sites, 'a', 'b')
    bb = bank(sites); bb['a']['training_sites'] = ['a','b']
    with pytest.raises(ValueError): m.assemble(bb, raw, cv, sites, 'd', 'oof')


def test_cut_transport_visible_and_probe_keeps_outer_costs():
    x, raw, cv, sites = fixture(); bb = bank(sites)
    p,y,cuts,_ = m.assemble(bb, raw, cv, sites, 'd', 'oof')
    drift = m.cut_drift(cv, cuts, .2)
    assert drift['easy_label_disagreement_fraction'] > 0
    models = residual.fit(x, p, y, np.ones(len(x))/len(x), sites, 'd')
    outer = np.tile([7., 5., 2., .8], (len(x), 1))
    for model in models.values():
        out = residual.predict(model, x, outer)
        np.testing.assert_array_equal(out[:,:3], outer[:,:3])
        assert (out[:,3] >= 0).all() and (out[:,3] <= outer[:,1]).all()


def test_two_locality_neural_resume_and_unknown(tmp_path):
    x,raw,cv,sites = fixture(); raw[45] = np.nan; cv[45] = np.nan
    take,pr,y,e = m.inner_inputs(x,raw,cv,sites,'d','a')
    settings = dict(width=8,steps=6,batch_size=16,learning_rate=.0003,
                    gradient_clip=5.,checkpoint_every=2,heartbeat_every=2)
    kw = dict(arm='cost_only',seed=17,settings=settings,identity={'outer':'d','inner':'a'},heartbeat=lambda **_:None)
    args = (x[take],y,e,sites[take],np.ones(take.sum()),pr)
    a,fit = neural.fit(*args,directory=tmp_path/'a',**kw)
    neural.fit(*args,directory=tmp_path/'b',stop_at=2,**kw)
    b,_ = neural.fit(*args,directory=tmp_path/'b',resume=True,**kw)
    np.testing.assert_array_equal(neural.predict(a,x,np.ones(len(x)),pr)[0],neural.predict(b,x,np.ones(len(x)),pr)[0])
    assert fit['unknown_rows_sampled'] == 0
