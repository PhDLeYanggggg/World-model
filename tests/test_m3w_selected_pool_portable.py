import numpy as np
from scripts import run_m3w_selected_pool_accounting as local
from scripts import run_m3w_selected_pool_portable as portable


def test_portable_analysis_matches_registered_local_implementation():
    rng=np.random.default_rng(24); n=41
    p=rng.random((n,5)); q=p*.8; y=rng.random((n,5)); y[::9]=np.nan
    env=np.ones(n)*10; raw=rng.random(n)>.3; kept=raw&(rng.random(n)>.5)
    rec=np.array(['record'+str(i//8) for i in range(n)]); supported=np.ones(n,bool)
    meta=dict(role='transfer',mode='joint',head_seed=17,site='a')
    assert local.analyze(y,p,q,env,raw,kept,rec,meta,supported)==portable.analyze(y,p,q,env,raw,kept,rec,meta,supported)
    assert portable.array_hash(y)==local.base.inter.array_hash(y)


def test_portable_keeps_unsupported_pool_and_unknowns():
    p=np.ones((3,5)); q=p.copy(); y=p.copy(); y[1]=np.nan
    env=np.ones(3); raw=np.ones(3,bool); supported=np.array([True,True,False]); kept=supported.copy()
    rec=np.array(['a','a','b']); meta=dict(role='transfer')
    assert local.analyze(y,p,q,env,raw,kept,rec,meta,supported)==portable.analyze(y,p,q,env,raw,kept,rec,meta,supported)
