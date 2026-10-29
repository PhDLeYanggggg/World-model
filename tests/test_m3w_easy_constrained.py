import copy
import numpy as np
import pytest
from src.evaluation import m3w_easy_constrained as api
from tests.test_m3w_local_cost_analogue import inputs


def test_conflict_and_constrained_quadratic_optimum():
    error=np.array([[[10.,-1.],[10.,-1.]]]);delta=np.array([[[-10.,3.],[-10.,3.]]])
    recs=np.array([[0,1]]);out=api.coefficients(error,delta,recs,np.ones(1,bool))
    assert out['parent_alpha'][0] > 2/3
    assert out['easy_cap'][0] == pytest.approx(2/3)
    assert out['alpha'][0] == pytest.approx(2/3)
    assert out['conflict'][0]
    assert out['easy_only_alpha'][0] == pytest.approx(1/3)
    assert out['after'][0,1] <= out['before'][0,1]+1e-12
    grid=np.linspace(0,2/3,10001)
    loss=(10-10*grid)**2+(-1+3*grid)**2
    assert out['after'][0].sum()<=loss.min()+1e-12


def test_positive_easy_direction_forces_fallback_not_safe_zero():
    e=np.array([[[10.,1.],[10.,1.]]]);d=np.array([[[-10.,1.],[-10.,1.]]])
    out=api.coefficients(e,d,np.array([[0,1]]),np.ones(1,bool))
    assert out['alpha'][0]==0 and out['easy_cap'][0]==0
    np.testing.assert_array_equal(out['after'],out['before'])


def test_zero_easy_direction_does_not_block_total_fit():
    e=np.array([[[2.,1.],[2.,1.]]]);d=np.array([[[-2.,0.],[-2.,0.]]])
    out=api.coefficients(e,d,np.array([[0,1]]),np.ones(1,bool))
    assert out['alpha'][0]==1 and out['easy_cap'][0]==1 and out['easy_only_alpha'][0]==0
    out=api.coefficients(e,d,np.array([[0,1]]),np.zeros(1,bool))
    assert out['alpha'][0]==out['easy_only_alpha'][0]==0


def test_many_fits_minimize_total_with_easy_constraint():
    rng=np.random.default_rng(903);e=rng.normal(size=(60,9,2));d=rng.normal(size=e.shape)
    recs=np.tile(np.array([0,0,0,1,1,2,3,4,4]),(60,1))
    out=api.coefficients(e,d,recs,np.ones(60,bool))
    assert np.all(out['after'][:,1]<=out['before'][:,1]+1e-12)
    w=1/(recs[:,:,None]==recs[:,None,:]).sum(-1);w/=w.sum(-1,keepdims=True)
    for i in range(60):
        grid=np.linspace(0,out['easy_cap'][i],501)
        loss=(w[i][None,:,None]*(e[i][None]+grid[:,None,None]*d[i][None])**2).sum((1,2))
        assert out['after'][i].sum()<=loss.min()+1e-12


def test_query_future_knownness_does_not_change_own_fit():
    b,q,p=inputs();old=api.retrieve(b,q,p,k=4)
    b['labels'][:20]=np.nan;new=api.retrieve(b,q,p,k=4)
    for a in api.MODES:
        np.testing.assert_array_equal(old['predictions'][a][:10],new['predictions'][a][:10])
    for name in old['channel']:
        np.testing.assert_array_equal(old['channel'][name][:10],new['channel'][name][:10])


def test_parent_predictions_and_receipts_exact():
    b,q,p=inputs();out=api.retrieve(b,q,p,k=4);old=api.parent.retrieve(b,q,p,k=4)
    assert out['references']==old['references']
    np.testing.assert_array_equal(out['predictions']['parent_blend'],old['predictions'][api.parent.PRIMARY])
    np.testing.assert_array_equal(out['channel']['parent_alpha'],old['diagnostics']['alpha'])
    assert np.all(out['channel']['after'][:,1]<=out['channel']['before'][:,1]+1e-12)
    for a in api.MODES:
        pred=out['predictions'][a]
        assert np.all(pred[:,1]<=pred[:,0]+1e-12)
        assert np.all(p[q,0]+pred[:,0]<=b['envelope'][q]+1e-12)


def test_unselected_labels_and_unit_scale():
    b,q,p=inputs();old=api.retrieve(b,q,p,k=4)
    b['labels'][~q]=np.nan;new=api.retrieve(b,q,p,k=4)
    for a in api.MODES:np.testing.assert_array_equal(old['predictions'][a],new['predictions'][a])
    b['scale']*=7;b['labels']*=7;b['envelope']*=7;p*=7
    new=api.retrieve(b,q,p,k=4)
    for a in api.MODES:np.testing.assert_allclose(old['predictions'][a]*7,new['predictions'][a],atol=1e-12)


def test_no_reference_retains_original_and_marks_support_missing():
    b,q,p=inputs();b['labels'][20:]=np.nan;out=api.retrieve(b,q,p,k=32)
    assert not out['supported'].any()
    for a in api.MODES:np.testing.assert_array_equal(out['predictions'][a],p[q][:,[1,4]])


def test_matched_alpha_controls_and_determinism():
    b,q,p=inputs();a=api.retrieve(b,q,p,k=4);c=api.retrieve(b,q,p,k=4)
    for arm in api.MODES:np.testing.assert_array_equal(a['predictions'][arm],c['predictions'][arm])
    for rec in np.unique(b['recordings'][q]):
        take=b['recordings'][q]==rec;d=a['channel']
        assert d['mean_alpha'][take].sum()==pytest.approx(d['alpha'][take].sum())
        np.testing.assert_array_equal(np.sort(d['alpha'][take]),np.sort(d['shuffled_alpha'][take]))
