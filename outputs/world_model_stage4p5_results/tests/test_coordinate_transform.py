import numpy as np
import pandas as pd

from src.data.build_real_episodes import make_episode


def test_absolute_to_local_preserves_relative_displacement():
    rows = []
    for frame in range(6):
        for agent in ["a", "b"]:
            base = 100.0 if agent == "a" else 103.0
            rows.append(
                {
                    "scene_id": "s",
                    "frame_id": frame,
                    "agent_id": agent,
                    "x": base + frame,
                    "y": 50.0,
                    "vx": 1.0,
                    "vy": 0.0,
                    "ax": 0.0,
                    "ay": 0.0,
                    "causal_vx": 1.0,
                    "causal_vy": 0.0,
                    "causal_ax": 0.0,
                    "causal_ay": 0.0,
                    "heading": 0.0,
                    "agent_type": "unknown",
                    "time": frame * 0.1,
                }
            )
    df = pd.DataFrame(rows)
    ep = make_episode("s", df, list(range(6)), ["a", "b"], 1, 0, {"whether_scene_geometry_available": False}, "causal_fd")
    original_delta = df[(df.frame_id == 5) & (df.agent_id == "b")]["x"].iloc[0] - df[(df.frame_id == 5) & (df.agent_id == "a")]["x"].iloc[0]
    local_delta = ep["states"][5, 1, 0] - ep["states"][5, 0, 0]
    np.testing.assert_allclose(local_delta, original_delta, atol=1e-6)
