from pathlib import Path

import numpy as np

from src.stage43_all_agent_history_graph_cache import (
    CACHE_DIR,
    REPORT_JSON,
    REPORT_MD,
    GATE_MD,
    WORLD_GATE_JSON,
    build_all_agent_history_graph_cache,
    run_all_agent_history_graph_cache,
)


def test_stage43_all_agent_history_graph_cache_gate_passes() -> None:
    payload = build_all_agent_history_graph_cache(history_k=16)
    gate = payload["stage43_bn_gate"]
    assert gate["verdict"] == "stage43_bn_all_agent_history_graph_cache_pass_raw_scene_blocker"
    assert gate["passed"] == gate["total"]
    assert gate["all_agent_history_graph_cache_ready"] is True
    assert gate["raw_scene_or_sdf_cache_ready"] is False
    assert gate["retrained_graph_ablation_executed"] is False


def test_stage43_all_agent_history_graph_cache_shapes_and_validation() -> None:
    payload = build_all_agent_history_graph_cache(history_k=16)
    assert payload["readiness_decision"]["all_agent_history_graph_ready"] is True
    assert payload["readiness_decision"]["raw_scene_or_sdf_ready"] is False
    assert all(row["row_alignment_preserved"] for row in payload["validation"].values())
    assert all(row["valid_shapes"] for row in payload["validation"].values())
    assert all(row["finite_history_values"] for row in payload["validation"].values())
    assert all(row["future_label_keys_absent_from_history_graph_cache"] for row in payload["validation"].values())
    assert all(row["rows_with_full_target_history"] > 0 for row in payload["validation"].values())
    assert all(row["rows_with_any_neighbor_history"] > 0 for row in payload["validation"].values())


def test_stage43_all_agent_history_graph_cache_no_leakage_and_no_overclaim() -> None:
    payload = build_all_agent_history_graph_cache(history_k=16)
    leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    assert leak["future_endpoint_input"] is False
    assert leak["future_waypoint_input"] is False
    assert leak["future_labels_cached_as_input"] is False
    assert leak["central_velocity_input"] is False
    assert leak["test_endpoint_goal_construction"] is False
    assert leak["test_statistics_normalization"] is False
    assert claim["graph_rich_history_cache_claim_allowed"] is True
    assert claim["retrained_graph_ablation_executed"] is False
    assert claim["raw_scene_or_sdf_main_claim_allowed"] is False
    assert claim["metric_or_seconds_claim"] is False
    assert claim["stage5c_executed"] is False
    assert claim["smc_enabled"] is False


def test_stage43_all_agent_history_graph_cache_writes_npz_and_reports() -> None:
    payload = run_all_agent_history_graph_cache(history_k=16)
    assert payload["stage43_bn_gate"]["passed"] == payload["stage43_bn_gate"]["total"]
    for split in ["train", "val", "test"]:
        path = CACHE_DIR / f"stage43_all_agent_history_graph_{split}.npz"
        assert path.exists()
        with np.load(path, allow_pickle=False) as data:
            assert "all_agent_history_xy" in data.files
            assert "all_agent_history_dxdy" in data.files
            assert "all_agent_history_valid_mask" in data.files
            assert "edge_history_attr" in data.files
            assert "future_xy" not in data.files
            assert "waypoint_xy" not in data.files
            assert data["all_agent_history_xy"].shape[-2:] == (16, 2)
            assert data["all_agent_history_xy"].shape[1] == data["neighbor_index"].shape[1] + 1
    assert Path(REPORT_JSON).exists()
    assert Path(REPORT_MD).exists()
    assert Path(GATE_MD).exists()
    assert Path(WORLD_GATE_JSON).exists()
