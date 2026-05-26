from src import stage42_post_repair_claim_refresh as s42ah


def test_stage42ah_status_classifies_positive_floor_and_safety_limited() -> None:
    positive = {
        "rows": 5,
        "ade_all": {"mean": 0.1, "ci_low": 0.01},
        "ade_hard_failure": {"mean": 0.1, "ci_low": 0.01},
        "ade_easy_degradation": {"mean": 0.0, "ci_high": 0.01},
        "fde_t50": {"mean": 0.2, "ci_low": 0.02},
    }
    floor = {
        "rows": 5,
        "ade_all": {"mean": 0.0, "ci_low": 0.0},
        "ade_hard_failure": {"mean": 0.0, "ci_low": 0.0},
        "ade_easy_degradation": {"mean": 0.0, "ci_high": 0.0},
        "switch_rate": {"mean": 0.0},
    }
    safety = {
        "rows": 5,
        "ade_all": {"mean": 0.1, "ci_low": 0.01},
        "ade_hard_failure": {"mean": 0.1, "ci_low": 0.01},
        "ade_easy_degradation": {"mean": 0.05, "ci_high": 0.06},
    }
    assert s42ah._status(positive, require_fde=True) == "positive_supported"
    assert s42ah._status(floor) == "floor_non_harm"
    assert s42ah._status(safety) == "safety_limited"


def test_stage42ah_gate_requires_metric_rejection_and_eth_repair() -> None:
    payload = {
        "stage42ag_gate": {"verdict": "stage42_ag_eth_t50_fde_source_repair_pass"},
        "summary": {
            "ade_all": {"ci_low": 0.01},
            "ade_t50": {"ci_low": 0.01},
            "ade_hard_failure": {"ci_low": 0.01},
            "ade_easy_degradation": {"ci_high": 0.01},
        },
        "slice_table": [
            {"slice": "ETH_UCY|25", "horizon": 25, "status": "floor_non_harm"},
            {"slice": "TrajNet|25", "horizon": 25, "status": "floor_non_harm"},
            {"slice": "UCY|25", "horizon": 25, "status": "floor_non_harm"},
            {"slice": "ETH_UCY|50", "horizon": 50, "status": "positive_supported"},
        ],
        "claim_matrix": [
            {"claim": "t100 can be written as a uniformly deployable long-horizon result.", "status": "rejected"},
            {"claim": "Metric or seconds-level pedestrian claims are allowed.", "status": "rejected"},
        ],
        "claim_boundary": {"stage5c_executed": False, "smc_enabled": False},
    }
    gate = s42ah._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ah_post_repair_claim_refresh_pass"
