import numpy as np
import torch

from scripts.audit_m3w_opendd_exchange_context_matched import matched_choice
from scripts.run_m3w_opendd_exchange_context import design
from src.world_model.m3w_exchange_context import MAP, PEER, WIDTH, aligned_and_rotated, descriptors


def test_context_reads_only_observed_and_predicted_geometry():
    class Raster:
        def sample(self, xy):
            return np.where(xy[..., 0] < 20., xy[..., 0], np.nan)

    h = np.zeros((2, 8, 2))
    h[0, :, 0] = np.arange(8)
    h[1, :, 0] = np.arange(8)+3
    cv = np.zeros((2, 12, 2))
    cv[..., 0] = np.arange(1, 13)
    edited = cv.copy()
    edited[0, :, 0] *= .5
    map_, peer = descriptors(h, cv, edited, np.array([0, 0]), np.array([[0, -1]]), Raster())
    assert map_.shape == (1, 2, MAP) and peer.shape == (1, 2, PEER)
    assert map_[0, 0, 0] == peer[0, 0, 0] == 1.
    assert map_[0, 0, 3] == 0. and map_[0, 0, 4] > 0.
    assert peer[0, 0, -2] > 0.
    assert not map_[0, 1].any() and not peer[0, 1].any()


def test_rotation_preserves_support_and_removes_singletons():
    road = np.zeros((4, 2, MAP))
    peer = np.zeros((4, 2, PEER))
    road[:, 0, 0] = peer[:, 0, 0] = 1.
    road[:, 0, 1] = np.arange(1, 5)
    road[2, 1, 0] = 1.
    real, wrong, active = aligned_and_rotated(road, peer, np.array(['a', 'a', 'a', 'b']))
    np.testing.assert_array_equal(active, [True, True, False, False])
    np.testing.assert_array_equal(real[:2], wrong[[1, 0]])
    assert not real[2:].any() and not wrong[2:].any()


def test_misaligned_control_uses_the_identical_active_weight_columns():
    width, w = 4, 4+2*WIDTH
    x = torch.arange(4*w+9, dtype=torch.float32)[None, :]
    aligned, wrong, motion = (design(x, width, k) for k in ('aligned', 'misaligned', 'motion'))
    for start in (0, w, 2*w+6, 3*w+6):
        i = start+width
        torch.testing.assert_close(wrong[:, i:i+WIDTH], x[:, i+WIDTH:i+2*WIDTH])
        torch.testing.assert_close(aligned[:, i:i+WIDTH], x[:, i:i+WIDTH])
        assert not motion[:, i:i+2*WIDTH].any()
        assert not aligned[:, i+WIDTH:i+2*WIDTH].any() and not wrong[:, i+WIDTH:i+2*WIDTH].any()


def test_matched_choice_commits_exactly_one_exchange_if_possible():
    scores = torch.tensor([[0., 0., 3., -.1], [0., 0., 2., .1]])
    assert matched_choice(scores) == 1
    assert matched_choice(scores[:1]) == 0
    assert matched_choice(scores[:0]) is None
