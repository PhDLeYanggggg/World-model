from __future__ import annotations

from src import stage42_t50_source_robustness_audit as cb


def test_source_name_shortens_parent_and_file() -> None:
    assert cb._source_name("/tmp/foo/bar.txt") == "foo/bar.txt"
    assert cb._source_name("bar.txt") == "bar.txt"


def test_ci_positive_handles_easy_boundary() -> None:
    assert cb._ci_positive({"low": 0.1, "high": 0.2, "bootstrap_n": cb.BOOTSTRAP_N})
    assert not cb._ci_positive({"low": -0.1, "high": 0.2, "bootstrap_n": cb.BOOTSTRAP_N})
    assert cb._ci_positive({"low": -0.2, "high": 0.01, "bootstrap_n": cb.BOOTSTRAP_N}, easy=True)
    assert not cb._ci_positive({"low": -0.2, "high": 0.03, "bootstrap_n": cb.BOOTSTRAP_N}, easy=True)


def test_gate_passes_with_source_concentration_limit() -> None:
    payload = {
        "source": "unit",
        "input_reports": {
            "stage42_bz_verdict": "stage42_bz_t50_repair_statistical_evidence_pass",
            "stage42_ca_verdict": "stage42_ca_post_bz_paper_package_refresh_pass",
        },
        "slice_audits": {"TrajNet|50": {}, "UCY|50": {}},
        "summary": {
            "robust_major_source_slices": ["TrajNet|50", "UCY|50"],
            "concentration_limited_slices": ["TrajNet|50", "UCY|50"],
            "broad_source_generalization_claim_allowed": False,
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
            "floor_free_neural_deployable": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = cb._gate(payload)
    assert gate["verdict"] == "stage42_cb_t50_source_robustness_pass_with_source_diversity_limit"
    assert gate["passed"] == gate["total"]


def test_gate_blocks_broad_source_overclaim() -> None:
    payload = {
        "source": "unit",
        "input_reports": {
            "stage42_bz_verdict": "stage42_bz_t50_repair_statistical_evidence_pass",
            "stage42_ca_verdict": "stage42_ca_post_bz_paper_package_refresh_pass",
        },
        "slice_audits": {"TrajNet|50": {}, "UCY|50": {}},
        "summary": {
            "robust_major_source_slices": ["TrajNet|50", "UCY|50"],
            "concentration_limited_slices": ["TrajNet|50"],
            "broad_source_generalization_claim_allowed": True,
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
            "floor_free_neural_deployable": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = cb._gate(payload)
    assert gate["verdict"] == "stage42_cb_t50_source_robustness_partial"
    assert not gate["gates"]["broad_source_generalization_not_overclaimed"]
