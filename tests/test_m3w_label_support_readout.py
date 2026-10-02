import numpy as np
from src.world_model import m3w_label_support_diagnostic as api
from scripts.verify_m3w_label_support import interval, check
from scripts.replay_m3w_label_support import verify_receipt
import pytest


def test_random_temporal_signs_match_scalar_exclusion():
    rng=np.random.default_rng(56)
    r,c,t=[rng.normal(size=(60,12,2)) for _ in range(3)]
    m=rng.random((60,12)) > .35; m[0]=False; m[1]=False; m[1,0]=True
    out=api.temporal_errors(r,c,t,m)
    for i in range(len(r)):
        d=[float(np.linalg.norm(c[i,j]-t[i,j])-np.linalg.norm(r[i,j]-t[i,j])) for j in range(12) if m[i,j]]
        assert out['valid_steps'][i] == len(d)
        if d: np.testing.assert_allclose(out['signed_error'][i],sum(d)/len(d),atol=1e-14)
        else: assert np.isnan(out['signed_error'][i])
        if len(d)>1:
            mean=sum(d)/len(d)
            removed=[sum(d[:j]+d[j+1:])/(len(d)-1) for j in range(len(d))]
            assert bool(out['leave_one_out_sign_flip'][i]) == any(mean*x <= 0 and mean != 0 for x in removed)


def test_scalar_bootstrap_and_vector_bootstrap_agree_with_missing_sites():
    rows=[dict(source=str(i//3),value=None if i<3 else .1*i) for i in range(18)]
    check(api.locality_interval(rows,lambda r:r['value']),interval(rows,lambda r:r['value']))


def test_cohort_strata_partition_occurrences_without_summing_ratios():
    rng=np.random.default_rng(10); n=80
    count=rng.integers(0,13,n); known=count>0
    d=rng.normal(size=n); ref=rng.uniform(1,5,n)
    harm=np.maximum(d,0); benefit=np.maximum(-d,0); easy=rng.random(n)<.5
    y=np.stack([benefit,harm,ref,ref*easy,harm*easy],1); y[~known]=np.nan
    take=rng.random(n)<.8; env=np.abs(d)+1
    temporal={k:np.zeros(n,bool) for k in ('leave_one_out_defined','leave_one_out_sign_flip','early_late_defined','early_late_opposite_sign')}
    out=api.cohort(y,take,env,np.arange(n),np.zeros(n),np.arange(n),count,temporal)['strata']
    assert sum(v['rows'] for v in out.values()) == n
    assert sum(v['selected'] for v in out.values()) == int(take.sum())
    np.testing.assert_allclose(sum(v['harm'] for v in out.values()),sum(harm[i] for i in range(n) if take[i] and known[i]))


def test_repeat_receipt_may_change_runtime_not_science():
    a=dict(pid=1,seconds=2.,peak_RSS_bytes=10,available_disk_bytes=20,summary_sha256='a',groups=[1])
    verify_receipt(a,dict(a,pid=5,seconds=4.,peak_RSS_bytes=12,available_disk_bytes=19))
    with pytest.raises(AssertionError): verify_receipt(a,dict(a,summary_sha256='b'))


def test_repeat_receipt_schema_change_fails():
    with pytest.raises(AssertionError): verify_receipt(dict(groups=[1]),dict(groups=[1],ignored=False))
