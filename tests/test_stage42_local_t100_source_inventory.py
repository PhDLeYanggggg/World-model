from pathlib import Path

from src import stage42_local_t100_source_inventory as s42bd


def test_candidate_group_mapping_for_trajnet_and_ucy() -> None:
    trajnet = Path("external_data/OpenTraj/datasets/TrajNet/Train/crowds/students003.txt")
    ucy = Path("external_data/OpenTraj/datasets/UCY/zara03/crowds_zara03.txt")
    assert "TrajNet::TrajNet/Train/crowds/students003.txt" in s42bd._candidate_groups(trajnet)
    groups = s42bd._candidate_groups(ucy)
    assert "UCY::UCY/zara03/crowds_zara03.txt" in groups
    assert "ETH_UCY::UCY/zara03/crowds_zara03.txt" in groups


def test_summarize_rows_marks_t100_capable() -> None:
    rows = [(i, "a", float(i), 0.0) for i in range(101)]
    out = s42bd._summarize_rows(Path("external_data/OpenTraj/datasets/UCY/foo.txt"), rows, skipped=0)
    assert out["t100_capable"] is True
    assert out["estimated_t100_windows"] == 1
    assert out["max_track_points"] == 101


def test_annotate_inventory_identifies_novel_candidate() -> None:
    row = {
        "path": "external_data/OpenTraj/datasets/TrajNet/Train/crowds/new_source.txt",
        "relative_path": "TrajNet/Train/crowds/new_source.txt",
        "t100_capable": True,
        "synthetic_or_diagnostic": False,
        "estimated_t100_windows": 7,
    }
    annotated = s42bd._annotate_inventory([row], known_groups={})
    assert annotated[0]["suggested_domain"] == "TrajNet"
    assert annotated[0]["already_in_current_source_groups"] is False
    assert annotated[0]["recommended_next_step"] == "candidate_for_stage42_be_conversion_and_train_only_source_cv"


def test_gate_passes_with_inventory_summary() -> None:
    payload = {
        "source": "fresh_local_path_inventory",
        "known_current_group_count": 2,
        "summary": {
            "files_scanned": 10,
            "parseable_files": 8,
            "t100_capable_files": 3,
            "novel_t100_candidate_files": 1,
            "stage42_be_conversion_recommended": True,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_metrics_for_threshold": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42bd._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_bd_local_t100_source_inventory_pass"
