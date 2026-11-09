import numpy as np
import pytest

from scripts.audit_m3w_cv_budget_menu_factorial import check, result
from scripts.verify_m3w_cv_budget_menu_inclusion import propagate
from src.evaluation.m3w_cv_budget_menu_factorial import prefix_orders
from src.world_model.m3w_cv_slate_supervision import menu


def paths(x):
    out = np.zeros((len(x),12,2))
    out[:,:,0] = np.asarray(x)[:,None]
    return out


def test_count_is_a_real_restriction():
    cv,ols,y = paths([1,1,1,1]),paths([.8,.8,.8,.8]),paths([0,0,0,0])
    bank = np.stack((paths([.5]*4),paths([1.5]*4)),axis=1)
    args = cv,ols,bank,y,np.ones((4,12),bool),np.repeat([0,1],2),2.,5.
    anchor = np.full(4,-1)
    for width in ('narrow','full'):
        assert result(args,anchor,width+'_exact')['status']=='undefined_support'
        assert result(args,anchor,width+'_relaxed')['status']=='verified'
    with pytest.raises(ValueError,match='query count'):
        check([[0,0]],args,anchor,'full_exact')


def test_full_menu_can_pick_action_outside_every_narrow_prefix():
    cv,ols,y = paths([1,1]),paths([1.1,1.1]),paths([0,0])
    bank = np.stack((paths([1.1,1.1]),paths([.3,.3])),axis=1)
    args = cv,ols,bank,y,np.ones((2,12),bool),np.zeros(2,int),2.,5.
    anchor = np.array([0,-1])
    assert result(args,anchor,'narrow_exact')['status']=='risk_infeasible'
    assert result(args,anchor,'full_exact')['status']=='verified'
    with pytest.raises(ValueError,match='out-of-menu'):
        check([[0,1]],args,anchor,'narrow_exact')


def test_prefix_actions_equal_the_registered_narrow_menu():
    rng = np.random.default_rng(28)
    cv,ols = rng.normal(size=(2,5,12,2))
    bank = rng.normal(size=(5,6,12,2))
    for count in (1,3,5):
        for (r,arm),old in zip(prefix_orders(cv,ols,bank),menu(cv,ols,bank,count),strict=True):
            action = np.full(len(cv),-1)
            action[r[:count]] = arm[:count]
            np.testing.assert_array_equal(action,old)


def test_nested_witnesses_do_not_relabel_infeasible_certificates():
    original = {k:dict(status='unresolved',upper_pp=2.) for k in
                ('narrow_exact','narrow_relaxed','full_exact','full_relaxed')}
    original['narrow_exact'] = dict(status='verified',actions=[[0,0]],upper_pp=1.)
    newer = propagate(original,lambda a,k:dict(verified=a==[[0,0]]))
    assert all(z['status']=='verified' for z in newer.values())
    assert original['full_relaxed']['status']=='unresolved'
    original['full_exact']['status']='risk_infeasible'
    with pytest.raises(ValueError,match='contradicts'):
        propagate(original,lambda a,k:dict(verified=True))
