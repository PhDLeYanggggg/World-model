import numpy as np

from scripts import run_m3w_pdestre_camera as camera
from scripts.audit_m3w_pdestre_box_order import swap_size


def test_box_parser_honors_xy_height_width_and_ignores_unknown_id():
    rows = camera.labels(b"3,1,10,20,7,4,ignored\n3,-1,irrelevant\n4,1,11,20,7,4,unused\n", 3)
    np.testing.assert_array_equal(rows[0][1], [10, 20, 14, 27])
    np.testing.assert_array_equal(swap_size(rows)[0][1], [10, 20, 17, 24])
    assert set(rows[0]) == {1}
    assert len(rows) == 20


def test_camera_readout_keeps_unmatched_online_paths_in_the_denominator():
    a, b = np.array([0, 0, 4, 4]), np.array([20, 0, 24, 4])
    truth = [{1: a, 2: b} for _ in range(20)]
    path_a, path_b = [[2, 2]]*12, [[22, 2]]*12
    frames = [dict(boxes={"rapid": [[0, *a, .9, 1], [1, *b, .9, 1],
                                     [2, 40, 0, 44, 4, .9, 1]]}) for _ in range(8)]
    z = dict(frames=frames, paths={"rapid": [[0, path_a, path_a], [1, path_b, path_b], [2, path_b, path_b]]})
    m = camera.metrics(z, truth, "rapid")
    assert (m["online"], m["stable"], m["complete"], m["absent"]) == (3, 2, 2, 1)
    assert not m["full_roster"]
    assert m["ade_px"] == [0., 0.]
    z["paths"]["rapid"].pop()
    assert camera.metrics(z, truth, "rapid")["full_roster"]


def test_framewise_matches_do_not_imply_stable_track():
    a = np.array([0, 0, 4, 4])
    truth = [{1: a} for _ in range(20)]
    frames = [dict(boxes={"rapid": [[t, *a, .9, 1]]}) for t in range(8)]
    m = camera.metrics(dict(frames=frames, paths={"rapid": []}), truth, "rapid")
    assert m["complete_match_ignoring_id"] == 1 and m["complete"] == 0
