import numpy as np
import pytest
from src.evaluation import m3w_forecast_tail_probe as m


def test_ties_never_use_row_order_or_outcomes():
    w=np.array([1.,2.,4.]); cost=np.array([9.,1.,3.]); score=np.ones(3)
    assert m.capture(score,cost,w,.1) == pytest.approx(.1)
    for order in ([2,0,1],[1,2,0]):
        assert m.capture(score[order],cost[order],w[order],.1) == pytest.approx(.1)


def test_fraction_is_weight_mass_not_row_count():
    assert m.capture([2,1],[4,2],[1,9],.1) == pytest.approx(4/22)
    assert m.capture([2,1],[4,2],[1,9],1) == pytest.approx(1)


def test_empty_cost_is_not_a_pass_or_zero():
    assert m.capture([1,2],[0,0],[1,1],.1) is None
    r=m.score_table(np.zeros((2,2)),np.ones(2),
        {k:np.zeros((2,2)) for k in ('signed_risk','positive_tail')},np.ones(2))
    assert not m.source_summary([dict(ranking=r)])['passes_information_screen']


@pytest.mark.parametrize('cost,weights,fraction', [([-1,0],[1,1],.1),([1,2],[0,0],.1),
    ([1,2],[1,-1],.1),([1,2],[1,1],0),([np.nan,0],[1,1],.1)])
def test_invalid_ranking_inputs_rejected(cost,weights,fraction):
    with pytest.raises(ValueError):
        m.capture([1,2],cost,weights,fraction)


def test_held_labels_features_and_weights_do_not_fit_parameters():
    rng=np.random.default_rng(17); x=rng.normal(size=(240,4)); x[:,0]=np.sign(x[:,0])
    y=np.column_stack((np.where(x[:,0]>0,2.,-.2),np.where(x[:,0]>0,.5,-.01)))
    train=np.arange(240)<180; w=np.ones(240)
    pred,a=m.fit(x,y,train,w)
    xx,yy,ww=x.copy(),y.copy(),w.copy(); xx[~train]+=100; yy[~train]-=500; ww[~train]*=100
    _,b=m.fit(xx,yy,train,ww)
    for k,v in m.parent.fingerprint(a).items():
        np.testing.assert_array_equal(v,m.parent.fingerprint(b)[k])
    assert (pred>=0).all()
    assert pred[x[~train,0]>0,1].mean()>pred[x[~train,0]<0,1].mean()


def test_mass_attribution_preserves_total_and_normalization():
    rows=m.group_masses(np.array([[1,-2],[3,4],[-1,0]]),np.array([1.,2.,3.]),np.array(['a','b','a']))
    np.testing.assert_allclose(np.sum([r['positive_mass'] for r in rows],0),[7/6,8/6])
    np.testing.assert_allclose(np.sum([r['positive_mass_share'] for r in rows],0),[1,1])


def test_oracle_only_diagnostic_and_screen_cannot_claim_safety():
    y=np.array([[1.,.5],[0.,0.],[-1.,0.]])
    r=m.score_table(y,np.ones(3),{'signed_risk':y,'positive_tail':y},np.ones(3))
    s=m.source_summary([dict(ranking=r)])
    assert not s['safety_certificate'] and s['no_policy_selected']
    assert not s['passes_information_screen']
