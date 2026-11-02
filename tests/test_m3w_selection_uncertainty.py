import numpy as np
import pytest
import torch
from src.world_model import m3w_selection_uncertainty as api
from tests.test_m3w_selection_controller import data


def fixture():
    d=data();u=np.random.default_rng(3).uniform(size=(8,16)).astype(np.float32)
    return api.augment_data(d,u),u


def test_spread_is_order_rotation_and_scale_invariant():
    rng=np.random.default_rng(3);p=rng.normal(size=(5,3,12,2));s=rng.uniform(.1,2,5)
    a=api.spread_features(p,s)
    np.testing.assert_allclose(a,api.spread_features(p[:,[2,0,1]],s),rtol=1e-6,atol=1e-6)
    angle=.37;r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    np.testing.assert_allclose(a,api.spread_features(p@r*11,s*11),rtol=1e-6,atol=1e-6)
    zero=api.spread_features(np.repeat(p[:,:1],3,1),s)
    np.testing.assert_allclose(zero,0,atol=1e-12)
    with pytest.raises(TypeError):api.spread_features(p,s,future_endpoint=np.zeros((5,2)))


def test_noise_catalog_is_fitting_only_and_row_order_stable():
    d,u=fixture();pr=d["uncertainty_preprocess"];c=d["causal"]
    assert set(pr["catalog_source_ids"])<=set(c["ids"])
    a=api.noise_rows(c["ids"],pr,"selection")
    np.testing.assert_array_equal(a[::-1],api.noise_rows(c["ids"][::-1],pr,"selection"))
    assert not np.array_equal(a,api.standardize(u,pr))


def test_zero_initialized_adapter_retains_original_model_and_rng():
    d,u=fixture();bias=d["preprocess"]["bias"]
    a=api.base.CostHead(444,8,17,bias,"context");rng=torch.get_rng_state()
    b=api.SpreadCostHead(444,8,17,bias,"real")
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k])
    z=d["z"];env=torch.ones(8,7)*3;seg=torch.arange(8);uu=torch.from_numpy(u)
    torch.testing.assert_close(a(z,env,seg,8),b(z,env,seg,8,uu),rtol=0,atol=0)


@pytest.mark.parametrize("arm",api.ARMS)
def test_resume_and_input_separation(tmp_path,arm):
    d,u=fixture();settings=dict(width=8,learning_rate=.001,query_batch_size=2,minimum_updates=8,
        largest_source_cycles=2,gradient_clip=5.,heartbeat_every=2,checkpoint_every=2)
    kw=dict(arm=arm,seed=17,settings=settings,identity={"fixture":True})
    api.fit(d,path=tmp_path/"resume.gz",stop_at=2,**kw)
    a=api.fit(d,path=tmp_path/"resume.gz",resume=True,**kw);b=api.fit(d,path=tmp_path/"direct.gz",**kw)
    for key in a:
        if key!="seconds":api.base.exact(a[key],b[key])
    assert api.predict(a,d["causal"],u,domain="selection").shape==(8,7,5)
    with pytest.raises(ValueError):api.predict(a,dict(d["causal"],future=np.zeros((8,12,2))),u,domain="selection")
    if arm=="disabled":
        old=api.cardinal.fit(d,arm="mean_absolute",seed=17,settings=settings,identity={"fixture":True},path=tmp_path/"original.gz")
        api.compare_disabled(a,old)
