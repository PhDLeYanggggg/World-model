import numpy as np

from scripts.run_m3w_opendd_support_exchange import observed_numerators
from src.evaluation.m3w_support_exchange import labels, pick, rollout


def test_exchange_is_same_support_and_budget_and_matches_transferred_mass():
    cv = np.zeros((5,12,2))
    inc = cv+np.array([.1,.2,.3,.4,0.])[:,None,None]
    scores = np.zeros((5,2,4))
    scores[0,1] = [2.,-.2,-.1,0.]
    scores[1,0] = [2.,-.1,0.,0.]
    h = np.array([0.,1.,2.,1.,5.])
    paths,stats = rollout(cv,inc,np.array([0,3,5]),scores,h)
    assert stats['active_queries']['guarded']==stats['active_queries']['matched_hindcast']==1
    assert stats['transferred_radius']['guarded']==stats['transferred_radius']['matched_hindcast']
    for p in paths.values():
        r = np.linalg.norm(p-cv,axis=-1).max(1)
        np.testing.assert_array_equal(r>0.,[True,True,True,True,False])
        np.testing.assert_allclose(np.add.reduceat(r,[0,3]),np.sqrt(2)*np.array([.3,.2]))


def test_guard_rejects_positive_predicted_numerator_change():
    m = np.zeros((2,2,4))
    m[0,1] = [2.,.01,0.,0.]
    m[1,0] = [1.,0.,0.,0.]
    p,_ = pick(np.ones(2),m,np.zeros(2))
    assert p['utility'] is not None and p['guarded'] is None


def test_training_action_deltas_match_the_original_harm_numerators():
    cv = np.zeros((3,12,2))
    inc = cv+.4
    y = cv.copy()
    y[0] = .05
    y[1] = 1.
    y[2] = np.nan
    valid = np.isfinite(y).all(2)
    d = dict(cv=cv,y=y,valid=valid)
    v = labels(cv,y,valid,inc,np.ones(3),.2)
    ref = observed_numerators(d,cv+.5*(inc-cv),.2)
    for i,a in enumerate((.25,.75)):
        actual = observed_numerators(d,cv+a*(inc-cv),.2)-ref
        np.testing.assert_allclose(v[:,i,1:].sum(0),actual,atol=1e-13)
    assert v[2,0,0]<0 and v[2,1,0]<0
