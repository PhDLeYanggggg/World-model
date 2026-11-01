import numpy as np
import pytest
from src.evaluation import m3w_pair_distance_diagnostic as api


def fixture():
    floor=np.zeros((2,12,2)); floor[1,:,0]=2.
    rollout=np.repeat(floor[:,None],7,axis=1); rollout[1,0,:,0]=1.; rollout[1,1,:,0]=2.5
    packet=dict(ids=np.arange(2),scales=np.ones(2),rollouts=rollout,floor_rollout=floor,
        recordings=np.array(['a','a']),frames=np.array([0,0]))
    target=floor.copy(); target[1,:,0]=3.
    choices=dict(unary_joint=np.zeros(2,np.int64),pair_joint=np.ones(2,np.int64))
    return packet,np.zeros((2,2)),choices,target,np.ones((2,12),bool)


def test_known_pair_error_and_full_completion_identity():
    p,o,c,y,m=fixture(); result=api.evaluate(p,o,c,target_labels=y,valid_labels=m)['total']
    assert result['requested_pair_steps']==12 and result['known_pair_steps']==12
    assert result['known_only_mean_error']==dict(floor=1.,unary_joint=2.,pair_joint=.5)
    assert result['full_requested_mean_gain']==dict(known_contribution=1.5,completion_lower=1.5,completion_upper=1.5)


@pytest.mark.parametrize('seed',range(12))
def test_missing_pair_bounds_cover_arbitrary_completion(seed):
    p,o,c,y,m=fixture(); rng=np.random.default_rng(seed); y=rng.normal(size=y.shape)*10
    m[:,6:]=False; result=api.evaluate(p,o,c,target_labels=y,valid_labels=m)['total']
    actual=api.evaluate(p,o,c,target_labels=y,valid_labels=np.ones_like(m))['total']['full_requested_mean_gain']['completion_lower']
    assert result['full_requested_mean_gain']['completion_lower']<=actual+1e-12
    assert result['full_requested_mean_gain']['completion_upper']>=actual-1e-12
    y[:,6:]=np.nan; missing=api.evaluate(p,o,c,target_labels=y,valid_labels=m)['total']
    assert missing==result


def test_zero_pair_is_missing_not_safe_zero():
    p,o,c,y,m=fixture(); p['frames']=np.arange(2)
    result=api.evaluate(p,o,c,target_labels=y,valid_labels=m)['total']
    assert result['requested_pair_steps']==0 and result['zero_pair_status']=='not_estimable'
    assert result['full_requested_mean_gain']['completion_lower'] is None


def test_labels_do_not_modify_frozen_actions_and_invalid_label_rejected():
    p,o,c,y,m=fixture(); saved={a:v.copy() for a,v in c.items()}
    api.evaluate(p,o,c,target_labels=y,valid_labels=m)
    for a in c: np.testing.assert_array_equal(c[a],saved[a])
    y[0,0]=np.nan
    with pytest.raises(ValueError): api.evaluate(p,o,c,target_labels=y,valid_labels=m)
