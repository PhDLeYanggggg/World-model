import numpy as np
import pandas as pd

from src.data.tgsim_loader import dense_frame_ids, fill_acceleration_components


def test_dense_frame_id_is_index_not_dt():
    times = pd.Series([10.0, 10.1, 10.2])
    frames = dense_frame_ids(times)
    np.testing.assert_array_equal(frames, np.asarray([0, 1, 2]))


def test_acceleration_uses_time_when_available():
    table = pd.DataFrame(
        {
            "agent_id": ["a", "a"],
            "frame_id": [0, 1],
            "time": [0.0, 0.1],
            "vx": [0.0, 1.0],
            "vy": [0.0, 0.0],
            "ax": [np.nan, np.nan],
            "ay": [np.nan, np.nan],
        }
    )
    out = fill_acceleration_components(table)
    np.testing.assert_allclose(out.iloc[1]["ax"], 10.0, atol=1e-6)
