import numpy as np

from scripts.audit_m3w_opendd_map_support import strata
from src.world_model.m3w_opendd_map_direction import RoadDistance, choose, world_to_pixel


def test_rotated_world_file_uses_pixel_centers_in_correct_axis_order():
    world = [0., 2., 3., 0., 10., 20.]
    np.testing.assert_allclose(world_to_pixel(np.array([[10., 20.], [16., 22.]]), world), [[0., 0.], [1., 2.]])
    mask = np.zeros((9, 9), dtype=bool)
    mask[:, 4:] = True
    raster = RoadDistance(mask, world)
    d = raster.sample(np.array([[16., 22.], [16., 32.], [1000., 20.]]))
    assert d[0] > 0 > d[1] and np.isnan(d[2])


def test_candidate_without_map_support_falls_back_and_keeps_bank_unchanged():
    world = [1., 0., 0., 1., 0., 0.]
    mask = np.zeros((12, 12), dtype=bool)
    mask[:, 6:] = True
    h = np.full((2, 8, 2), 5.)
    bank = np.zeros((2, 2, 12, 2))
    bank[:, 0, :, 0] = 2.
    bank[1, 1, :, 0] = 10.
    out, supported, _ = choose(h, bank, np.ones(2, dtype=bool), RoadDistance(mask, world), 0)
    assert supported.tolist() == [True, False]
    assert out["map_persistence"].tolist() == [1, 0]
    assert out["map_continuation"].tolist() == [1, 0]


def test_shuffle_strata_keep_unsupported_and_recent_crossings_separate():
    distances = np.array([[np.nan, 1], [1, 2], [-2, -1], [-1, 1], [1, -1]])
    observed = np.array([False, True, True, True, True])
    assert strata(distances, observed).tolist() == [0, 1, 2, 3, 4]
