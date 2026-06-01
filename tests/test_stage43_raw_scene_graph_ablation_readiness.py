from pathlib import Path

from src.stage43_raw_scene_graph_ablation_readiness import (
    REPORT_JSON,
    REPORT_MD,
    GAP_MD,
    GATE_MD,
    WORLD_GATE_JSON,
    build_raw_scene_graph_ablation_readiness,
    run_raw_scene_graph_ablation_readiness,
)


def test_stage43_raw_scene_graph_ablation_readiness_gate_passes() -> None:
    payload = build_raw_scene_graph_ablation_readiness()
    gate = payload["stage43_bl_gate"]
    assert gate["verdict"] == "stage43_bl_raw_scene_graph_ablation_readiness_pass_blocker_documented"
    assert gate["passed"] == gate["total"]
    assert gate["raw_scene_retrained_ablation_ready_now"] is False
    assert gate["graph_rich_retrained_ablation_ready_now"] is False
    assert gate["raw_scene_or_graph_rich_main_claim_allowed"] is False


def test_stage43_raw_scene_graph_ablation_readiness_records_proxy_boundary() -> None:
    payload = build_raw_scene_graph_ablation_readiness()
    proxy = payload["current_proxy_evidence"]
    claim = payload["claim_boundary"]
    blockers = {row["blocker"]: row for row in payload["blocker_matrix"]}
    assert proxy["scene_proxy_only"] is True
    assert proxy["interaction_proxy_only"] is True
    assert proxy["scene_proxy_delta"]["full_scene_minus_no_scene_t50"] > 0.0
    assert proxy["feature_family_delta"]["full_minus_no_neighbor_interaction_t50"] > 0.0
    assert claim["raw_scene_main_claim"] is False
    assert claim["graph_rich_interaction_main_claim"] is False
    assert claim["proxy_scene_goal_interaction_evidence_only"] is True
    assert "raw_scene_or_verified_sdf_tensor_missing" in blockers
    assert "graph_rich_all_agent_tensor_missing" in blockers


def test_stage43_raw_scene_graph_ablation_readiness_cache_schema_is_not_raw_scene_or_graph_rich() -> None:
    payload = build_raw_scene_graph_ablation_readiness()
    cache = payload["cache_schema"]
    assert cache["raw_scene_tensor_ready"] is False
    assert cache["graph_rich_tensor_ready"] is False
    assert all(row["has_row_geometry"] for row in cache["full_waypoint_cache"].values())
    assert all(row["has_future_labels"] for row in cache["full_waypoint_cache"].values())
    assert not any(row["has_raw_scene_keys"] for row in cache["full_waypoint_cache"].values())
    assert not any(row["has_graph_rich_keys"] for row in cache["full_waypoint_cache"].values())


def test_stage43_raw_scene_graph_ablation_readiness_no_leakage_and_no_stage5c() -> None:
    payload = build_raw_scene_graph_ablation_readiness()
    leak = payload["no_leakage_and_execution"]
    claim = payload["claim_boundary"]
    assert leak["future_endpoint_input"] is False
    assert leak["future_waypoint_input"] is False
    assert leak["future_labels_eval_or_loss_only"] is True
    assert leak["central_velocity_input"] is False
    assert leak["test_endpoint_goal_construction"] is False
    assert leak["test_statistics_normalization"] is False
    assert leak["new_training_executed"] is False
    assert leak["new_conversion_executed"] is False
    assert claim["metric_or_seconds_claim"] is False
    assert claim["true_3d_world_model"] is False
    assert claim["foundation_world_model"] is False
    assert claim["stage5c_executed"] is False
    assert claim["smc_enabled"] is False


def test_stage43_raw_scene_graph_ablation_readiness_writes_reports() -> None:
    payload = run_raw_scene_graph_ablation_readiness()
    assert payload["stage43_bl_gate"]["passed"] == payload["stage43_bl_gate"]["total"]
    assert Path(REPORT_JSON).exists()
    assert Path(REPORT_MD).exists()
    assert Path(GAP_MD).exists()
    assert Path(GATE_MD).exists()
    assert Path(WORLD_GATE_JSON).exists()
