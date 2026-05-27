from src import stage42_floor_relaxation_source_robustness as gv


def test_source_name_shortens_parent_and_file():
    assert gv._source_name("/tmp/source/example.txt") == "source/example.txt"
    assert gv._source_name("example.txt") == "example.txt"


def test_gate_blocks_broad_source_overclaim():
    payload = {
        "summary": {
            "gt_verdict": "stage42_gt_floor_relaxation_safety_stress_pass",
            "gu_verdict": "stage42_gu_floor_relaxation_paper_refresh_pass",
            "source_safety_positive_slices": ["TrajNet|50", "UCY|50"],
            "source_concentration_limited_slices": ["UCY|50"],
            "broad_source_generalization_claim_allowed": True,
            "training_executed": False,
            "download_executed": False,
            "conversion_executed": False,
            "threshold_tuned_on_test": False,
        },
        "slice_audits": {
            "TrajNet|50": {"total_rows": 10, "source_count": 2},
            "UCY|50": {"total_rows": 10, "source_count": 1},
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "broad_source_generalization_claim_allowed": True,
            "global_floor_removal_allowed": False,
            "floor_free_neural_deployable": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = gv._gate(payload)
    assert gate["verdict"] == "stage42_gv_floor_relaxation_source_robustness_partial"
    assert not gate["gates"]["broad_source_generalization_not_overclaimed"]


def test_gate_passes_with_source_concentration_caveat():
    payload = {
        "summary": {
            "gt_verdict": "stage42_gt_floor_relaxation_safety_stress_pass",
            "gu_verdict": "stage42_gu_floor_relaxation_paper_refresh_pass",
            "source_safety_positive_slices": ["TrajNet|50", "UCY|50"],
            "source_concentration_limited_slices": ["UCY|50"],
            "broad_source_generalization_claim_allowed": False,
            "training_executed": False,
            "download_executed": False,
            "conversion_executed": False,
            "threshold_tuned_on_test": False,
        },
        "slice_audits": {
            "TrajNet|50": {"total_rows": 10, "source_count": 2},
            "UCY|50": {"total_rows": 10, "source_count": 1},
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "broad_source_generalization_claim_allowed": False,
            "global_floor_removal_allowed": False,
            "floor_free_neural_deployable": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = gv._gate(payload)
    assert gate["passed"] == gate["total"]
