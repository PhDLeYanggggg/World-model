from __future__ import annotations

from src.m3w.token_schema import TOKEN_NAMES, build_token_schema
from src.m3w.numpy_backend import _token_pool_features


def test_m3w_token_schema_contains_required_tokens():
    features = [
        "speed_now",
        "speed_mean_past",
        "scene_image_width",
        "goal_count",
        "density_r50",
        "cv_rollout_displacement",
        "horizon_norm",
        "start_frame_norm",
    ]
    schema = build_token_schema(features)
    assert set(TOKEN_NAMES) == set(schema.token_names)
    assert "future_endpoint" not in " ".join(features)
    assert "central_velocity" not in " ".join(features)


def test_m3w_numpy_token_pool_shapes():
    import numpy as np

    features = [f"f{i}" for i in range(12)]
    pooled = _token_pool_features(np.ones((4, 12), dtype=np.float32), features)
    assert pooled.shape == (4, len(TOKEN_NAMES) * 2)
