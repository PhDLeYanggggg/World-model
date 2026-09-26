import numpy as np
import pytest
from src.evaluation.m3w_harm_tail_diagnostics import event_targets,weights_for_sites,quantiles,top_mass_share,summarize


def test_target_definition_preserves_unknown_and_nested_events():
    y=np.array([[1,2,0,0],[3,4,0,0],[np.nan]*4]); cv=np.array([1.,3.,np.nan])
    out=event_targets(y,cv,2.)
    np.testing.assert_array_equal(out[:2],[[1,2,1,2],[3,4,0,0]])
    assert np.isnan(out[2]).all()


def test_equal_locality_weights_and_train_bins():
    w=weights_for_sites([True,True,True,False],['a','a','b','b'])
    np.testing.assert_array_equal(w,[.25,.25,.5,0])
    assert quantiles(np.array([0.,1.,2.,999.]),w,[.5,.9])==[1.,2.]
    with pytest.raises(ValueError): weights_for_sites([True,False],['a','b'])


def test_tied_scores_cannot_cherry_pick_high_harm_rows():
    s=np.ones(4); h=np.array([0.,0.,0.,8.]); w=np.full(4,.25)
    assert top_mass_share(s,h,w,.1)==pytest.approx(.1)
    assert top_mass_share(h,h,w,.1)==pytest.approx(.4)
    assert top_mass_share(s,np.zeros(4),w,.1) is None


def test_flat_ranking_and_unknown_rows():
    y=np.array([[1.,2.,1.,2.],[1.,0.,0.,0.],[np.nan]*4])
    pred=np.ones((3,4)); env=np.ones(3)*2; edges={k:[1.] for k in ('moment','fraction','envelope')}
    r=summarize(pred,y,env,np.array(['a']*3),edges)
    assert r['rows']==2 and r['positive']==1
    assert r['scores']['moment']['AUROC']==.5
    assert r['scores']['moment']['AUPRC']==.5
    assert r['scores']['moment']['top10_harm_mass_share']==pytest.approx(.1)
    assert sum(b['rows'] for b in r['scores']['moment']['bins'])==2
    assert not r['stable_event_support']


def test_no_event_is_unidentifiable_not_perfect_prediction():
    r=summarize(np.ones((3,4)),np.zeros((3,4)),np.ones(3),np.array(['a']*3),
        {k:[1.] for k in ('moment','fraction','envelope')})
    assert r['harm_coverage'] is None and r['scores']['moment']['AUROC'] is None
