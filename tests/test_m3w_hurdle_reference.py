import numpy as np
import pytest
from src.world_model import m3w_hurdle_reference as api
from tests.test_m3w_harm_refinement import fixture


def data():
    d,_=fixture();c=d['causal'];n=len(c['ids'])
    c=dict(c,sites=np.array(['a'if i%2 else'b'for i in range(n)]))
    # Source/query membership must remain homogeneous in this synthetic fixture.
    c['groups']=[np.array([i])for i in range(n)]
    target=np.array([float(i%3)for i in range(n)])
    return c,target,np.ones(n,bool)


def test_features_are_causal_and_inner_sources_are_metadata_only():
    c,y,known=data();x=api.inputs(c);assert x.shape==(len(y),382)
    with pytest.raises(ValueError):api.inputs(dict(c,future_endpoint=np.zeros((len(y),2))))
    assert api.inner_folds(['a','b','c','d'])==api.inner_folds(['d','c','b','a'])
    use=known.copy();use[0]=False;w=api.weights(c,use)
    assert w[0]==0 and w.sum()==pytest.approx(1.)


@pytest.mark.parametrize('task',api.TASKS)
def test_reference_forests_resume_exactly_and_keep_support_masks_out_of_input(tmp_path,task):
    c,y,k=data();settings=dict(trees=dict(direct=8,event=8,positive=8),max_depth=3,min_samples_leaf=1,max_features=8)
    kw=dict(task=task,settings=settings,seed=17,identity={},checkpoint_every=2)
    a=api.fit(c,y,k,['a','b'],path=tmp_path/'full.pt.gz',**kw)
    api.fit(c,y,k,['a','b'],path=tmp_path/'resume.pt.gz',stop_at=4,**kw)
    b=api.fit(c,y,k,['a','b'],path=tmp_path/'resume.pt.gz',resume=True,**kw)
    api.exact_refit(a,b);np.testing.assert_array_equal(api.predict_task(a,api.inputs(c)),api.predict_task(b,api.inputs(c)))
    assert a['dimension']==382 and all(t.n_features_in_==382 for t in a['model'].estimators_)


def test_unknown_targets_do_not_enter_reference_fitting(tmp_path):
    c,y,k=data();k[0]=False
    settings=dict(trees=dict(direct=4,event=4,positive=4),max_depth=3,min_samples_leaf=1,max_features=8)
    kw=dict(task='direct',settings=settings,seed=17,identity={},checkpoint_every=2)
    a=api.fit(c,y,k,['a','b'],path=tmp_path/'a.pt.gz',**kw)
    yy=y.copy();yy[0]=1e9
    b=api.fit(c,yy,k,['a','b'],path=tmp_path/'b.pt.gz',**kw)
    api.exact_refit(a,b)


def test_conditional_positive_fit_preserves_original_sampling_measure(tmp_path):
    c,y,k=data();y[:]=0;y[:3]=[1,2,3]
    common=api.weights(c,k&c['allowed'][:,6]);use=k&(y>0)&c['allowed'][:,6]
    common[~use]=0;common/=common.sum()
    settings=dict(trees=dict(direct=2,event=2,positive=2),max_depth=3,min_samples_leaf=1,max_features=8)
    state=api.fit(c,y,k,['a','b'],task='positive',settings=settings,seed=17,identity={},
        path=tmp_path/'positive.pt.gz',checkpoint_every=2)
    assert state['weight_sha256']==api.old.digest(common[use])
