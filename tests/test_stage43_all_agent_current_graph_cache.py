from pathlib import Path

import numpy as np

from src.stage43_all_agent_current_graph_cache import (
    CACHE_DIR,
    REPORT_JSON,
    REPORT_MD,
    GATE_MD,
    WORLD_GATE_JSON,
    build_all_agent_current_graph_cache,
    run_all_agent_current_graph_cache,
)


def test_stage43_all_agent_current_graph_cache_gate_passes() -> None:
    payload = build_all_agent_current_graph_cache(top_k=8)
    gate = payload["stage43_bm_gate"]
    assert gate["verdict"] == "stage43_bm_all_agent_current_graph_cache_pass_partial_history_blocker"
    assert gate["passed"] == gate["total"]
    assert gate["all_agent_current_graph_cache_ready"] is True
    assert gate["all_agent_history_graph_cache_ready"] is False
    assert gate["raw_scene_or_sdf_cache_ready"] is False


def test_stage43_all_agent_current_graph_cache_validation_and_boundary() -> None:
    payload = build_all_agent_current_graph_cache(top_k=8)
    assert payload["readiness_decision"]["all_agent_current_graph_ready"] is True
    assert payload["readiness_decision"]["all_agent_history_graph_ready"] is False
    assert payload["claim_boundary"]["current_frame_graph_cache_claim_allowed"] is True
    assert payload["claim_boundary"]["graph_rich_history_main_claim_allowed"] is False
    assert payload["claim_boundary"]["raw_scene_or_sdf_main_claim_allowed"] is False
    assert all(row["row_alignment_preserved"] for row in payload["validation"].values())
    assert all(row["edge_index_in_range"] for row in payload["validation"].values())
    assert all(row["no_self_edges"] for row in payload["validation"].values())
    assert all(row["future_label_keys_absent_from_graph_cache"] for row in payload["validation"].values())
    assert all(row["multi_agent_rows"] > 0 for row in payload["validation"].values())


def test_stage43_all_agent_current_graph_cache_no_leakage() -> None:
    payload = build_all_agent_current_graph_cache(top_k=8)
    leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    assert leak["future_endpoint_input"] is False
    assert leak["future_waypoint_input"] is False
    assert leak["future_labels_cached_as_input"] is False
    assert leak["central_velocity_input"] is False
    assert leak["test_endpoint_goal_construction"] is False
    assert leak["test_statistics_normalization"] is False
    assert claim["metric_or_seconds_claim"] is False
    assert claim["true_3d_world_model"] is False
    assert claim["foundation_world_model"] is False
    assert claim["stage5c_executed"] is False
    assert claim["smc_enabled"] is False


def test_stage43_all_agent_current_graph_cache_writes_npz_and_reports() -> None:
    payload = run_all_agent_current_graph_cache(top_k=8)
    assert payload["stage43_bm_gate"]["passed"] == payload["stage43_bm_gate"]["total"]
    for split in ["train", "val", "test"]:
        path = CACHE_DIR / f"stage43_all_agent_current_graph_{split}.npz"
        assert path.exists()
        with np.load(path, allow_pickle=False) as data:
            assert "edge_index" in data.files
            assert "edge_attr" in data.files
            assert "all_agent_current_xy" in data.files
            assert "future_xy" not in data.files
            assert "waypoint_xy" not in data.files
            assert data["edge_index"].shape[0] == 2
            assert data["edge_attr"].shape[1] == 6
    assert Path(REPORT_JSON).exists()
    assert Path(REPORT_MD).exists()
    assert Path(GATE_MD).exists()
    assert Path(WORLD_GATE_JSON).exists()
