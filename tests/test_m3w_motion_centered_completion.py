import numpy as np

from src.evaluation.m3w_motion_centered_completion import completion_ledger, recenter


def test_only_terminal_center_changes_and_radius_stays_fixed():
    agent = np.array([19, 19]); frame = np.array([7, 8])
    h = np.zeros((2, 8, 2)); h[:,:,0] = frame[:,None]+np.arange(-7, 1)
    target = np.full((2, 5, 2), np.nan)
    valid = np.array([[True, False, False, True, True], [False, False, True, True, False]])
    target[..., 0][valid] = (frame[:,None]+np.arange(1,6))[valid]
    target[..., 1][valid] = 0.
    ids, seen, original, radius, gaps = completion_ledger(agent, frame, h, target, valid)
    x, counts = recenter(seen, original, radius, gaps, 1)
    np.testing.assert_array_equal(x, original)  # only two consecutive terminal anchors
    assert counts == dict(terminal_steps=1, shifted_eligible_steps=0, changed_coordinates=0)

    track = np.array([19]); frame = np.array([7]); h = h[:1]
    target = np.full((1, 4, 2), np.nan); valid = np.zeros((1,4), bool)
    ids, seen, original, radius, gaps = completion_ledger(track, frame, h, target, valid)
    x, counts = recenter(seen, original, radius, gaps, 1)
    np.testing.assert_allclose(x[ids[0], 0], [8., 9., 10., 11.])
    np.testing.assert_array_equal(x[seen], original[seen])
    np.testing.assert_array_equal(radius[ids[0]], [1., 2., 3., 4.])
    assert counts['changed_coordinates'] == 4
