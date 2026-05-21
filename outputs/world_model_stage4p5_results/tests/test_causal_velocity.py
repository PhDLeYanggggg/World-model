import numpy as np
import pandas as pd

from src.data.tgsim_loader import compute_velocity_variants


def test_causal_velocity_uses_only_previous_frame():
    raw = pd.DataFrame(
        {
            "agent_id": ["a", "a", "a"],
            "time": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 100.0],
            "y_m": [0.0, 0.0, 0.0],
            "vx_mps": [0.0, 0.0, 0.0],
            "vy_mps": [0.0, 0.0, 0.0],
            "ax_mps2": [0.0, 0.0, 0.0],
            "ay_mps2": [0.0, 0.0, 0.0],
        }
    )
    variants = compute_velocity_variants(raw)
    assert variants.loc[1, "causal_vx"] == 1.0
    assert variants.loc[1, "central_vx"] == 50.0
    assert variants.loc[1, "causal_vx"] != variants.loc[1, "central_vx"]


def test_dt_from_time_not_dense_frame_assumption():
    raw = pd.DataFrame(
        {
            "agent_id": ["a", "a"],
            "time": [10.0, 10.1],
            "x_m": [0.0, 0.2],
            "y_m": [0.0, 0.0],
            "vx_mps": [0.0, 0.0],
            "vy_mps": [0.0, 0.0],
            "ax_mps2": [0.0, 0.0],
            "ay_mps2": [0.0, 0.0],
        }
    )
    variants = compute_velocity_variants(raw)
    np.testing.assert_allclose(variants.loc[1, "dt"], 0.1, atol=1e-6)
    np.testing.assert_allclose(variants.loc[1, "causal_vx"], 2.0, atol=1e-6)
