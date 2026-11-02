import copy
import numpy as np
import pytest
import torch

from src.world_model import m3w_selection_cardinal_cost as api
from tests.test_m3w_selection_controller import data


def test_original_objective_is_bitwise_identical():
    d = data(); model = api.base.CostHead(444, 8, 17, d["preprocess"]["bias"], "context")
    p = model(d["z"], torch.ones(8, 7)*3, torch.arange(8), 8)
    a = api.base.loss(p, d, np.arange(8), torch.arange(8), 8)
    b = api.objective(p, d, np.arange(8), torch.arange(8), 8, "weighted_centered")
    for key in a:
        if isinstance(a[key], torch.Tensor): assert torch.equal(a[key], b[key])
        else: assert a[key] == b[key]


def test_absolute_gain_penalizes_common_shift_centered_loss_ignores():
    d = data(); truth = d["truth"].clone(); p = truth.clone(); p[..., 0] += .5
    args = (p, d, np.arange(8), torch.arange(8), 8)
    centered = api.objective(*args, "mean_centered")
    absolute = api.objective(*args, "mean_absolute")
    assert float(centered["ranking"]) < 1e-12 and float(absolute["ranking"]) > .01
    assert torch.equal(centered["regression"], absolute["regression"])


def test_unknown_labels_cannot_change_any_arm():
    d = data(); altered = copy.copy(d); altered["truth"] = d["truth"].clone(); altered["truth"][-1] = 1000
    p = torch.ones(8, 7, 5)*.1
    for arm in api.ARMS:
        a = api.objective(p, d, np.arange(8), torch.arange(8), 8, arm)
        b = api.objective(p, altered, np.arange(8), torch.arange(8), 8, arm)
        assert torch.equal(a["total"], b["total"]) and a["unknown_supervised_rows"] == 0


@pytest.mark.parametrize("arm", api.NEW_ARMS)
def test_recovery_and_causal_input_rejection(tmp_path, arm):
    d = data(); settings = dict(width=8, learning_rate=.001, query_batch_size=2,
        minimum_updates=8, largest_source_cycles=2, gradient_clip=5., heartbeat_every=2, checkpoint_every=2)
    kw = dict(arm=arm, seed=17, settings=settings, identity={"fixture":True})
    api.fit(d, path=tmp_path/"resume.gz", stop_at=2, **kw)
    a = api.fit(d, path=tmp_path/"resume.gz", resume=True, **kw)
    b = api.fit(d, path=tmp_path/"direct.gz", **kw)
    for key in a:
        if key != "seconds": api.base.exact(a[key], b[key])
    assert api.base.predict(a, d["causal"]).shape == (8,7,5)
    with pytest.raises(ValueError): api.base.predict(a, dict(d["causal"], future_endpoint=np.ones((8,2))))


def test_asymmetric_score_not_same_mean_estimand():
    # A two-point target makes the conditional-mean/expectile distinction exact.
    target = torch.tensor([0.]*9+[1.])
    p = torch.tensor(.1, requires_grad=True)
    squared = (p-target).square().mean()
    asymmetric = ((p-target).square()*torch.where(p < target,4.,1.)).mean()
    assert abs(float(torch.autograd.grad(squared,p,retain_graph=True)[0])) < 1e-7
    assert float(torch.autograd.grad(asymmetric,p)[0]) < -.5
