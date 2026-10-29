import copy
import numpy as np
import pytest

from src.world_model import m3w_shared_train_risk as api
from tests.test_m3w_easy_harm_deviance_training import settings
from tests.test_m3w_task_risk_training import CONFIG


def packet(name='a'):
    n=32; rng=np.random.default_rng(91)
    x=rng.normal(size=(n,380)); env=np.full(n,2.)
    y=np.array([[.3,0.,1.,1.,0.] if i%2 else [0.,.2,1.,1.,.2] for i in range(n)])
    series=np.stack([np.tile([r[1]-r[0],r[2]],(12,1)) for r in y])
    y[-1]=np.nan;series[-1]=np.nan
    sites=np.repeat(name,n);recs=np.repeat(np.arange(4).astype(str),8);frames=np.tile(np.arange(8),4)
    pr=api.old.core.preprocess(x,env,y,sites,recs,frames,training_site=name)
    return x,env,y,series,sites,recs,frames,pr


def test_partition_excludes_held_statistics_and_targets():
    args=packet(); part=api.recording_partition('a',args[5])
    train,held=api.prepare(args,part)
    assert not set(train[5]) & set(held[5])
    changed=list(copy.deepcopy(args));mask=np.isin(args[5],part['held_recordings'])
    changed[0][mask] += 1e5; changed[2][mask]=1e6
    again,_=api.prepare(changed,part)
    api.old.core.exact(train[-1],again[-1])


def fit(path,mode,**kw):
    packets={name:api.prepare(packet(name),api.recording_partition(name,packet(name)[5]))[0]
             for name in 'abcd'}
    return api.fit_shared(packets,mode=mode,settings=settings(),objective_config=CONFIG,
        seed=17,identity={'test':True},experiment_sha256='test',path=path,heartbeat=lambda **_:None,**kw)


@pytest.mark.parametrize('mode',['shared_global','shared_source'])
def test_exact_resume_and_causal_inference(tmp_path,mode):
    direct=fit(tmp_path/'direct.pt.gz',mode)
    fit(tmp_path/'resume.pt.gz',mode,stop_at=3)
    resumed=fit(tmp_path/'resume.pt.gz',mode,resume=True)
    for k in direct:
        if k!='seconds': api.old.core.exact(direct[k],resumed[k])
    args=packet();p=api.predict_shared(direct,'a',args[0],args[1])
    assert p.shape==(32,5) and np.isfinite(p).all() and (p>=0).all()
    assert (p[:,:2].sum(1)<=args[1]+1e-6).all()
    assert (p[:,3]<=p[:,2]+1e-8).all() and (p[:,4]<=p[:,1]+1e-8).all()


def test_shared_arms_match_initialization_and_source_draws(tmp_path):
    a,b=[fit(tmp_path/(m+'.pt.gz'),m) for m in ('shared_global','shared_source')]
    for k in ('preprocess','source_preprocess','initial_model','samplers','input_hashes'):
        api.old.core.exact(a[k],b[k])
    assert a['dual'].shape==(2,) and b['dual'].shape==(4,2)


def test_single_recording_is_a_real_support_blocker():
    with pytest.raises(ValueError): api.recording_partition('a',['1']*10)
