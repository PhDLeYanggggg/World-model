import numpy as np

from scripts.run_m3w_opendd_direct_exchange import actions, apply


def test_inference_action_features_need_no_targets_and_match_training_order():
    cv = np.zeros((3,12,2))
    inc = cv+np.array([.1,.2,.3])[:,None,None]
    r = np.linalg.norm(inc-cv,axis=-1).max(1)
    d = dict(cv=cv,scale=np.ones(3),offsets=[0,3],origin=np.zeros((3,2)))
    c = dict(inc=inc,r=r)
    x = np.arange(15.).reshape(3,5)
    X,q,info,_ = actions(d,c,x)
    X_train,Y,w = actions(dict(**d,y=inc,valid=np.ones((3,12),bool)),c,x,1.)
    np.testing.assert_array_equal(X,X_train)
    assert X.shape[0]==Y.shape[0]==6
    np.testing.assert_allclose(w.sum(),1.)
    est = np.zeros((len(q),5))
    est[0] = [1.,0.,0.,0.,.5]
    p,n = apply(cv,inc,d['offsets'],q,info,est)
    assert n=={'direct_utility':1,'direct_guard':1,'event_guard':0}
    np.testing.assert_array_equal(p['event_guard'],.5*inc)
