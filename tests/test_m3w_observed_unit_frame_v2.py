import numpy as np

from src.world_model.m3w_observed_unit_frame import observed_unit_frame as v1
from src.world_model.m3w_observed_unit_frame_v2 import observed_unit_frame as v2
from src.world_model.m3w_sdd_step_adapter import SDDStepAdapter


def turning(factor):
    rows = []
    for f in range(20):
        xy = factor*1e-7*np.array([np.cos(.13*f), np.sin(.13*f)])
        half = np.array([.2, .4])*factor
        rows.append([0, *(xy-half), *(xy+half), f, 0, 0, 0])
    a = SDDStepAdapter(np.array(rows), np.full(20, 'Pedestrian'), 'synthetic/video0', 1)
    return a.get_geometry(0)[None]


def test_legacy_turn_threshold_counterexample_and_repaired_internal_rollouts():
    normal, small = turning(1), turning(1e-4)
    # An inherited native-unit speed cutoff changes the turn diagnostic branch.
    assert np.max(np.abs(v1(normal)[0]-v1(small)[0])) > .01
    np.testing.assert_allclose(v2(normal)[0], v2(small)[0], atol=5e-5, rtol=5e-5)
    np.testing.assert_allclose(v2(normal)[0], v2(turning(1e4))[0], atol=5e-5, rtol=5e-5)


def test_cached_rollout_changes_cannot_override_past_recomputation():
    x = turning(1)
    changed = x.copy(); changed[:, 308:] += 800
    for a, b in zip(v2(x), v2(changed)):
        np.testing.assert_array_equal(a, b)
