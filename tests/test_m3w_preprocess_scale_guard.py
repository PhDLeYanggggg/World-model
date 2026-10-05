import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.world_model import m3w_temporal_auxiliary as api
from src.world_model.m3w_preprocess_portability import frozen_preprocess, validate_scale_only
from test_m3w_temporal_auxiliary import data, settings


def shifted(x, count):
    for _ in range(count): x=math.nextafter(x, math.inf)
    return x


def test_only_four_ulp_scalar_difference_is_permitted():
    a=dict(scale=1.1, mean=np.array([1.,2.]), training_site='TRAIN')
    validate_scale_only(a,dict(a,scale=shifted(a['scale'],4)),api.core.exact)
    with pytest.raises(ValueError):validate_scale_only(a,dict(a,scale=shifted(a['scale'],5)),api.core.exact)
    with pytest.raises(AssertionError):validate_scale_only(a,dict(a,mean=np.array([math.nextafter(1.,2.),2.])),api.core.exact)
    with pytest.raises(AssertionError):validate_scale_only(a,dict(a,training_site='OTHER'),api.core.exact)
    with pytest.raises(ValueError):validate_scale_only(a,dict(a,mean=np.array([1.,2.],dtype=np.float32)),api.core.exact)


@pytest.mark.parametrize('scale',[0.,-1.,math.nan,math.inf,1,np.float32(1.1)])
def test_invalid_scale_never_admitted(scale):
    with pytest.raises(ValueError):validate_scale_only(dict(scale=1.1),dict(scale=scale),api.core.exact)


def test_exact_comparator_and_original_function_restored_after_exception():
    pr=dict(scale=1.1);fn=lambda:dict(scale=shifted(1.1,2))
    core=SimpleNamespace(preprocess=fn,exact=api.core.exact)
    with pytest.raises(RuntimeError):
        with frozen_preprocess(core,pr):
            assert core.preprocess() is pr
            assert core.exact is api.core.exact
            raise RuntimeError('interrupted')
    assert core.preprocess is fn


@pytest.mark.parametrize('arm',api.ARMS)
def test_optimizer_and_resume_unchanged_with_frozen_scale(tmp_path, monkeypatch, arm):
    args=data();expected=copy.deepcopy(args[-1]);original=api.core.preprocess
    kwargs=dict(arm=arm,settings=settings(),seed=17,identity={'test':'scale_guard'},heartbeat=lambda **kw:None)
    baseline=api.fit(*args,**kwargs,path=tmp_path/'original.pt.gz')
    def alternate_reduction(*a,**kw):
        pr=original(*a,**kw);pr['scale']=shifted(pr['scale'],2);return pr
    monkeypatch.setattr(api.core,'preprocess',alternate_reduction)
    path=tmp_path/'guarded.pt.gz'
    with frozen_preprocess(api.core,args[-1]):
        api.fit(*args,**kwargs,path=path,stop_at=3)
        actual=api.fit(*args,**kwargs,path=path,resume=True)
    assert api.core.preprocess is alternate_reduction
    api.core.exact(args[-1],expected)
    for key in baseline:
        if key!='seconds':api.core.exact(baseline[key],actual[key])
