import torch

from src.world_model.m3w_gated_motion_residual import GatedMotionResidual
from src.world_model.m3w_neighbor_motion_adapter import ARMS, NeighborMotionAdapter


def fixture():
    t = torch.arange(8.)
    h = torch.stack((t, .05*t*t), 1)[None]
    return dict(history=h, neighbors=h[:, None]+torch.tensor([[[[3., 1.]]]]),
                neighbor_mask=torch.ones(1, 1, 8, dtype=torch.bool))


def test_context_arms_agree_with_no_valid_neighbors_and_freeze_backbone():
    x = fixture()
    x['neighbor_mask'][:] = False
    first = NeighborMotionAdapter(GatedMotionResidual(False), ARMS[0])
    expected = first(x)
    for arm in ARMS[1:]:
        candidate = NeighborMotionAdapter(GatedMotionResidual(False), arm)
        candidate.load_state_dict(first.state_dict())
        torch.testing.assert_close(candidate(x), expected)
        assert all(not p.requires_grad for p in candidate.backbone.parameters())


def test_local_model_rotation_equivariance():
    x = fixture()
    r = torch.tensor([[0., 1.], [-1., 0.]])
    model = NeighborMotionAdapter(GatedMotionResidual(False), ARMS[-1])
    rotated = dict(x, history=x['history'] @ r+10., neighbors=x['neighbors'] @ r+10.)
    torch.testing.assert_close(model(rotated), model(x) @ r, atol=1e-5, rtol=1e-5)
