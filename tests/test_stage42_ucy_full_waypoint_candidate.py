import numpy as np

from src import stage42_ucy_full_waypoint_candidate as v


def test_full_label_alignment_matches_requested_ids():
    ds = v._dataset("val")
    assert ds["waypoint_xy"].shape[0] == ds["ids"].shape[0]
    assert ds["waypoint_delta"].shape[1:] == (4, 2)
    assert ds["waypoint_valid"].shape[1] == 4
    assert np.any(ds["waypoint_valid"])


def test_gate_rejects_negative_t50_candidate():
    result = {
        "data": {"train": {"rows": 10}, "val": {"rows": 10}, "test": {"rows": 10}},
        "rows": [{} for _ in range(len(v.TRIALS) * len(v.SEEDS))],
        "best_summary": {
            "ade_all": {"mean": 0.01},
            "ade_t50": {"mean": -0.1},
            "ade_hard_failure": {"mean": 0.02},
            "ade_easy_degradation": {"mean": 0.0},
        },
        "deployment_decision": "keep_stage42r_s_policy_ucy_blocked",
        "no_leakage": {"future_waypoint_input": False},
        "claim_boundary": {"stage5c_executed": False, "smc_enabled": False},
    }
    gate = v._gate(result)
    assert gate["passed"] == 10
    assert gate["total"] == 11
    assert gate["verdict"] == "stage42_v_ucy_full_waypoint_candidate_not_deployable"
