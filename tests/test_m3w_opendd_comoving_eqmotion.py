import numpy as np
import torch

from src.world_model import m3w_opendd_comoving_eqmotion as model


def test_comoving_inputs_are_invariant_to_common_velocity_and_translation():
    rng = np.random.default_rng(51)
    h = rng.normal(size=(7, 8, 2)).cumsum(1)
    offsets = np.array([0, 4, 7])
    v = np.array([2., -1.])
    q = h + np.arange(-7, 1)[None, :, None]*v + [31., 15.]
    a, b = (model.context(x, offsets)[:3] for x in (h, q))
    left, s, cv = model.frame(*a, "comoving_peers")
    right, t, dv = model.frame(*b, "comoving_peers")
    for k in left:
        torch.testing.assert_close(left[k], right[k], atol=3e-6, rtol=1e-5)
    torch.testing.assert_close(s, t)
    torch.testing.assert_close(dv-cv, torch.tensor(v, dtype=cv.dtype)[None, None]
                               *torch.arange(1, 13)[None, :, None].expand_as(cv))


def test_context_is_query_local_and_masks_empty_slots():
    h = np.zeros((5, 8, 2))
    h[:, :, 0] = np.array([0, 1, 2, 9, 10])[:, None]
    a, p, mask, _ = model.context(h, np.array([0, 3, 5]))
    assert mask.sum(1).tolist() == [2, 2, 2, 1, 1]
    assert torch.max(p.abs()) == 2.
    x, scale, _ = model.frame(a, p, mask, "comoving_ego")
    assert not x["neighbor_mask"].any()
    _, scale0, _ = model.frame(a, p, mask, "centered_peers")
    torch.testing.assert_close(scale, scale0)


def test_pinned_fixed_head_runs_all_frames_without_future_inputs():
    torch.set_num_threads(2)
    h = np.random.default_rng(62).normal(size=(4, 8, 2)).cumsum(1)
    batch = model.context(h, np.array([0, 4]))
    for k in model.ARMS:
        net = model.Forecast(k)
        pred = net(*batch[:3])
        assert pred.shape == (4, 12, 2)
        assert torch.isfinite(pred).all()
        y, valid = torch.ones_like(pred), torch.ones(pred.shape[:2], dtype=torch.bool)
        y[0, 3:] = torch.nan
        valid[0, 3:] = False
        loss = model.loss(net, batch, y, valid)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


@torch.inference_mode()
def test_comoving_model_restores_rotation_and_common_velocity():
    torch.set_num_threads(2)
    torch.manual_seed(29)
    h = np.random.default_rng(80).normal(size=(4, 8, 2)).cumsum(1)
    r = np.array([[0., -1.], [1., 0.]])
    v = np.array([-1., 2.])
    boosted = h @ r + np.arange(-7, 1)[None, :, None]*v + [40., 52.]
    a, b = (model.context(z, np.array([0, 4]))[:3] for z in (h, boosted))
    net = model.Forecast("comoving_peers").eval()
    p, q = net(*a), net(*b)
    t = torch.arange(1, 13, dtype=p.dtype)[None, :, None]
    torch.testing.assert_close(q, p @ torch.tensor(r, dtype=p.dtype) + t*torch.tensor(v, dtype=p.dtype),
                               atol=3e-4, rtol=3e-4)
