import copy
import numpy as np
import pytest
import torch

from src.world_model import m3w_selection_controller as api


def causal():
    rng = np.random.default_rng(7)
    n = 8
    groups = [np.arange(i, i+2) for i in range(0, n, 2)]
    return dict(x=rng.normal(size=(n, 7, 444)).astype(np.float32), envelope=np.ones((n, 7))*3,
        allowed=np.ones((n, 7), bool), scales=np.ones(n, np.float32), groups=groups,
        source_groups=[np.array([0, 1]), np.array([2, 3])], sites=np.repeat(["a", "b"], 4), ids=np.arange(n))


def data():
    c = causal()
    y = np.ones((8, 7, 5), np.float64)
    y[..., :2] = .1; y[..., 3:] = .2; y[-1] = np.nan
    return api.training_data(c, y)


def test_same_weights_context_changes_only_when_other_agents_present():
    a = api.CostHead(444, 8, 17, np.zeros(5, np.float32), "row")
    b = api.CostHead(444, 8, 17, np.zeros(5, np.float32), "context")
    with torch.no_grad(): a.action.weight.fill_(.1); a.reference.weight.fill_(.1)
    b.load_state_dict(a.state_dict()); c = causal()
    z = torch.from_numpy(c["x"][:2]); e = torch.ones(2, 7)
    torch.testing.assert_close(a(z, e, torch.arange(2), 2), b(z, e, torch.arange(2), 2), rtol=0, atol=0)
    assert not torch.equal(a(z, e, torch.zeros(2, dtype=torch.long), 1), b(z, e, torch.zeros(2, dtype=torch.long), 1))
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_unknown_rows_do_not_become_zero_targets_and_references_are_shared():
    d = data(); model = api.CostHead(444, 8, 17, d["preprocess"]["bias"], "context")
    p = model(d["z"], torch.ones(8, 7)*3, torch.arange(8), 8)
    assert torch.equal(p[:, :, 2], p[:, :1, 2].expand(-1, 7))
    assert torch.equal(p[:, :, 3], p[:, :1, 3].expand(-1, 7))
    assert (p[..., 0] <= p[..., 2]).all() and (p[..., 4] <= p[..., 1]).all()
    terms = api.loss(p, d, np.arange(8), torch.arange(8), 8)
    altered = copy.copy(d); altered["truth"] = d["truth"].clone(); altered["truth"][-1] = 100
    assert torch.equal(terms["total"], api.loss(p, altered, np.arange(8), torch.arange(8), 8)["total"])
    assert terms["unknown_supervised_rows"] == 0


def test_resume_scientific_state_exact_and_labels_refused_at_inference(tmp_path):
    d = data(); settings = dict(width=8, learning_rate=.001, query_batch_size=2,
        minimum_updates=4, largest_source_cycles=2, gradient_clip=5., heartbeat_every=2, checkpoint_every=2)
    kw = dict(arm="context", seed=17, settings=settings, identity={"fixture": True})
    api.fit(d, path=tmp_path/"resumed.gz", stop_at=2, **kw)
    resumed = api.fit(d, path=tmp_path/"resumed.gz", resume=True, **kw)
    direct = api.fit(d, path=tmp_path/"direct.gz", **kw)
    for key in direct:
        if key != "seconds": api.exact(resumed[key], direct[key])
    a = api.predict(direct, d["causal"])
    assert a.shape == (8, 7, 5) and np.isfinite(a).all()
    with pytest.raises(ValueError): api.predict(direct, dict(d["causal"], future_endpoint=np.zeros((8, 2))))


def test_feature_constructor_preserves_partial_context_and_keeps_labels_separate():
    n = 3; g = np.zeros((n,476),np.float32)
    g[:,:16] = np.tile(np.column_stack((np.arange(-7,1),np.zeros(8))).ravel(),(n,1))
    g[:,16:24] = np.arange(-7,1)/12
    g[:,230:238] = 1; g[:,166:174] = np.arange(-7,1)/12
    history = np.stack([np.column_stack((np.arange(-7,1),np.zeros(8))) for _ in range(n)])
    from src.world_model.m3w_european_source_forecast import baseline_numpy
    physical = np.stack([baseline_numpy(history,k) for k in range(6)],1)
    p = np.concatenate((physical,physical[:,:1]),1)
    args = dict(geometry=g,detector_history=np.ones((n,64),np.float32),rollouts=p,
        floor_index=2,recordings=np.array([0,0,1]),frames=np.array([100,100,100]),
        ids=np.arange(n),sites=np.array(["a","a","b"]))
    c = api.features(**args)
    assert c["x"].shape == (n,7,444) and len(c["groups"])==2
    assert not c["allowed"][:,2].any()
    with pytest.raises(TypeError): api.features(**args,future_endpoint=np.zeros((n,2)))
    target=p[:,1].copy();valid=np.ones((n,12),bool);valid[-1]=False
    y,known=api.labels(p,target,valid,2,10.)
    assert known.tolist()==[True,True,False] and np.isnan(y[-1]).all()
