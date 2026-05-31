from __future__ import annotations

import numpy as np

from src import stage43_scene_raster_proxy_tokens as aa


def _payload() -> dict:
    split = {
        "rows": 10,
        "cache_exists": True,
        "feature_dim": len(aa.FEATURE_NAMES),
        "row_hash": "row",
        "feature_hash": "feature",
        "source_proxy_coverage": 0.5,
        "domain_or_source_coverage": 1.0,
    }
    return {
        "source": aa.SOURCE,
        "stage43_z_precondition": {"verdict": "stage43_z_latent_token_schema_coverage_pass"},
        "proxy_build": {
            "build_split": "stage43_train_only",
            "source_proxy_count": 2,
            "domain_proxy_count": 1,
        },
        "split_features": {split_name: dict(split) for split_name in aa.SPLITS},
        "feature_names": list(aa.FEATURE_NAMES),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "test_endpoint_goal_construction": False,
            "scene_proxy_built_from_stage43_train_only": True,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "true_scene_image_token_claim": False,
            "true_sdf_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "deployment_contract": {
            "standalone_world_model_deployable": False,
            "requires_stage37_stage42_floor": True,
        },
    }


def test_gate_accepts_train_only_scene_proxy_tokens() -> None:
    gate = aa._gate(_payload())
    assert gate["passed"] == gate["total"]
    assert gate["scene_raster_proxy_token_ready"] is True
    assert gate["standalone_world_model_deployable"] is False


def test_gate_rejects_test_endpoint_goal_usage() -> None:
    payload = _payload()
    payload["no_leakage"]["test_endpoint_goal_construction"] = True
    gate = aa._gate(payload)
    assert gate["passed"] < gate["total"]
    assert gate["scene_raster_proxy_token_ready"] is False


def test_boundary_sdf_is_positive_inside_and_negative_outside() -> None:
    proxy = aa.SceneProxy(
        level="source",
        key="toy",
        rows=4,
        centroid=np.asarray([0.5, 0.5], dtype=np.float32),
        bounds_min=np.asarray([0.0, 0.0], dtype=np.float32),
        bounds_max=np.asarray([1.0, 1.0], dtype=np.float32),
        scale=1.0,
        route_grid=np.ones((aa.GRID_SIZE, aa.GRID_SIZE), dtype=np.float32),
        density_mean=1.0,
        goal_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        entropy_mean=0.1,
        ambiguity_mean=0.2,
    )
    assert aa._boundary_sdf(proxy, np.asarray([0.5, 0.5], dtype=np.float32)) > 0
    assert aa._boundary_sdf(proxy, np.asarray([1.5, 0.5], dtype=np.float32)) < 0


def test_alignment_handles_zero_velocity_safely() -> None:
    assert aa._alignment(np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)) == 0.0
