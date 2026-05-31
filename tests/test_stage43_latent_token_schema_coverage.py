from __future__ import annotations

from src import stage43_latent_token_schema_coverage as z


FEATURE_NAMES = [
    "current_x_over_scale",
    "current_y_over_scale",
    "horizon_norm",
    "domain_UCY",
    "horizon_50",
    "history_dx_tail0",
    "history_dy_tail0",
    "history_speed_tail0",
    "history_accel_tail0",
    "history_heading_tail0",
    "history_valid_mask_tail0",
    "history_curvature",
    "history_turn_angle",
    "history_stop_go",
    "history_dwell",
    "history_path_length",
    "history_velocity_decay",
    "history_neighbor_count",
    "history_min_neighbor_dist",
    "history_density",
    "history_TTC",
    "history_closing_speed",
    "prototype_likelihood_0",
    "prototype_distance_0",
    "prototype_angle_0",
    "prototype_entropy",
    "goal_ambiguity",
    "baseline_endpoint_rel_0",
    "floor_endpoint_rel_x",
    "floor_endpoint_rel_y",
]


CACHE_KEYS = {
    "current_xy",
    "agent_id",
    "frame_id",
    "horizon",
    "source_file",
    "scene_id",
    "dataset",
    "dt_frame_step",
    "future_xy",
    "waypoint_xy",
    "waypoint_valid",
    "failure",
    "easy",
    "hard",
    "valid_mask",
}


def _payload() -> dict:
    coverage = z._coverage(
        FEATURE_NAMES,
        {
            "train": {"keys": list(CACHE_KEYS)},
            "val": {"keys": list(CACHE_KEYS)},
            "test": {"keys": list(CACHE_KEYS)},
        },
    )
    return {
        "source": z.SOURCE,
        "stage43_y_precondition": {"verdict": "stage43_y_protected_multimodal_latent_head_suite_candidate"},
        "split_schemas": {
            split: {"exists": True, "rows": 10, "row_hash": f"{split}-hash"}
            for split in z.SPLITS
        },
        "feature_schema": {"feature_dim": len(FEATURE_NAMES), "feature_schema_hash": "feature-hash"},
        "token_coverage": coverage,
        "explicit_gaps": {
            "explicit_scene_image_raster_token": "missing",
            "explicit_scene_sdf_token": "missing",
            "future_occupancy_true_label": "missing_proxy_only",
            "true_physical_validity_label": "missing_proxy_only",
            "human_interaction_annotation": "missing_proxy_only",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "deployment_contract": {
            "safety_floor_required": True,
            "standalone_ungated_policy": False,
        },
    }


def test_label_only_tokens_are_separated_from_features() -> None:
    payload = _payload()
    assert payload["token_coverage"]["future_endpoint_label"]["status"] == "label_only_separated"
    assert payload["token_coverage"]["future_full_waypoint_label"]["status"] == "label_only_separated"


def test_gate_accepts_complete_schema_with_recorded_scene_gaps() -> None:
    gate = z._gate(_payload())
    assert gate["passed"] == gate["total"]
    assert gate["protected_multimodal_latent_state_schema_supported"] is True
    assert gate["standalone_world_model_deployable"] is False


def test_gate_rejects_future_waypoint_input_leakage() -> None:
    payload = _payload()
    payload["no_leakage"]["future_waypoint_input"] = True
    gate = z._gate(payload)
    assert gate["passed"] < gate["total"]
    assert gate["protected_multimodal_latent_state_schema_supported"] is False


def test_coverage_records_missing_scene_tokens_as_gap_not_success() -> None:
    payload = _payload()
    scene_patch = payload["token_coverage"]["scene_patch"]
    assert scene_patch["status"] == "recorded_gap"
    assert scene_patch["present"] is False
