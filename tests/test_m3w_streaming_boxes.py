import numpy as np
import pytest

from src.world_model.m3w_streaming_boxes import StreamingTracks


def moving(i):
    return np.array([[i, 1, i+8, 9, .9, 3.], [40-i, 2, 48-i, 10, .8, 1.]])


def test_stream_is_prefix_invariant_and_forecasts_constant_motion():
    model = StreamingTracks()
    prior = []
    for i in range(10):
        rows, paths = model.step(i, moving(i))
        prior.append(rows)
        if i == 7:
            np.testing.assert_allclose(paths[0][1], paths[0][2])
            np.testing.assert_allclose(paths[0][1][-1], [23, 5])
    replay = StreamingTracks()
    for i in range(8):
        rows, _ = replay.step(i, moving(i))
        np.testing.assert_array_equal(rows, prior[i])
    mutated = moving(8) + np.array([100, 100, 100, 100, 0, 0])
    replay.step(8, mutated)
    np.testing.assert_array_equal(prior[0][:, 0], [0, 1])


def test_no_interpolation_and_no_class_switch():
    track = StreamingTracks()
    for i in range(9):
        rows, forecasts = track.step(i, moving(i) if i != 3 else np.empty((0, 6)))
        assert not forecasts
    bad = moving(9)
    bad[0, 5] = 6
    assert track.step(9, bad)[0][0, 0] == 2
    with pytest.raises(ValueError):
        track.step(11, moving(11))
