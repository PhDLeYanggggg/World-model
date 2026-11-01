import numpy as np
import pytest
from src.evaluation import m3w_pair_distance_diagnostic as api


def fixture():
    rng=np.random.default_rng(43); n=3
    packet=dict(ids=np.arange(n),scales=np.array([.1,3.,100.],np.float32),
        floor_rollout=(rng.normal(size=(n,12,2))*300).astype(np.float32),
        rollouts=(rng.normal(size=(n,7,12,2))*300).astype(np.float32),
        recordings=np.array(['a']*n),frames=np.zeros(n,dtype=int))
    origin=(rng.normal(size=(n,2))*1000).astype(np.float32)
    target=(rng.normal(size=(n,12,2))*1000).astype(np.float32)
    mask=rng.uniform(size=(n,12))>.2
    choices=dict(unary_joint=np.array([0,-1,2]),pair_joint=np.array([1,3,-1]))
    return packet,origin,choices,target,mask


def test_float32_cache_agrees_exactly_with_explicit_float64_evaluation():
    p,o,c,y,m=fixture(); actual=api.evaluate(p,o,c,target_labels=y,valid_labels=m)
    cast={k:v.astype(float) if k in ('scales','rollouts','floor_rollout') else v for k,v in p.items()}
    expected=api.evaluate(cast,o.astype(float),c,target_labels=y.astype(float),valid_labels=m)
    assert actual==expected


def test_common_translation_does_not_create_a_geometry_gain():
    p,o,c,y,m=fixture(); original=api.evaluate(p,o,c,target_labels=y,valid_labels=m)['total']
    shift=np.array([12000.,-45000.]); moved=api.evaluate(p,o.astype(float)+shift,c,
        target_labels=y.astype(float)+shift,valid_labels=m)['total']
    for k,v in original['known_error_mass'].items(): assert moved['known_error_mass'][k]==pytest.approx(v,abs=1e-9)
    assert moved['unknown_disagreement_mass']==pytest.approx(original['unknown_disagreement_mass'],abs=1e-9)


def test_all_missing_labels_keep_unknown_bound_not_zero_error():
    p,o,c,y,m=fixture(); m[:]=False; y[:]=np.nan
    result=api.evaluate(p,o,c,target_labels=y,valid_labels=m)['total']
    assert result['known_pair_steps']==0 and result['unknown_pair_steps']==36
    assert all(v is None for v in result['known_only_mean_error'].values())
    gain=result['full_requested_mean_gain']
    assert gain['completion_lower']<0 and gain['completion_upper']>0
    assert gain['completion_lower']==-gain['completion_upper']
