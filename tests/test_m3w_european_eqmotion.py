from copy import deepcopy
import numpy as np
import pytest
from src.evaluation.m3w_european_eqmotion import paired_group,interval
from scripts.run_m3w_european_eqmotion import reference_matching,scientific_state


def test_unknown_rows_use_direct_disagreement_and_endpoint_mask():
    p=np.ones((3,12,2));q=np.zeros_like(p);y=np.zeros_like(p)
    mask=np.ones((3,12),bool);mask[1,6:]=False;mask[2]=False
    cv=np.array([2.,2.,np.nan])
    r=paired_group(p,q,y,mask,cv)
    assert (r['known_rows'],r['partial_rows'],r['unknown_rows'],r['endpoint_rows'])==(2,1,1,1)
    assert r['known_pp']<0 and r['completion_lower_pp']<r['known_pp']<r['completion_upper_pp']
    y[~mask]=1e20
    assert paired_group(p,q,y,mask,cv)==r


def test_full_roster_and_actual_seed_fold_support_required():
    groups=[dict(site=s,trial=f'{f}_{seed}',seed=seed,scores={'known_pp':1.})
        for s in ('a','b') for f in (0,1) for seed in (17,29,43)]
    assert interval(groups,'known_pp',['a','b'],100,17)['CI95']==[1.,1.]
    with pytest.raises(ValueError):interval(groups[:-1],'known_pp',['a','b'],100,17)
    with pytest.raises(ValueError):interval(groups,'known_pp',['a','b','c'],100,17)


def test_matching_rejects_changed_draws_and_incomplete_budget():
    d=dict(train_ids=np.array([0,1]),held_ids=np.array([2]),factors=np.array([1.,1.,0.]))
    a=dict(train_ids=d['train_ids'],factors=d['factors'],draws=np.array([4,4,0]),
        sampler_rng=np.array([1]),step=4)
    reference_matching(a,deepcopy(a),d,4,2)
    b=deepcopy(a);b['draws'][0]-=1;b['draws'][2]+=1
    with pytest.raises(AssertionError):reference_matching(b,a,d,4,2)
    with pytest.raises(ValueError):reference_matching(a,a,d,5,2)
    scientific_state(a,deepcopy(a))
