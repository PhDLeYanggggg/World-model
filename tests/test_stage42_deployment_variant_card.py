from __future__ import annotations

from src import stage42_deployment_variant_card as dn


def _payload() -> dict:
    cr = {
        "stage42_cr_gate": {"passed": 19, "total": 19},
        "ablation_rows": {
            "endpoint_linear_reference": {"all_improvement": 0.0},
            "no_proximity_guard": {
                "all_improvement": 0.03,
                "t50_improvement": 0.01,
                "t100_raw_frame_diagnostic_improvement": 0.06,
                "hard_failure_improvement": 0.03,
                "easy_degradation": 0.002,
                "switch_rate": 0.2,
                "near_collision_005_delta_vs_endpoint": 0.003,
            },
            "proximity_guard": {
                "all_improvement": 0.02,
                "t50_improvement": 0.01,
                "t100_raw_frame_diagnostic_improvement": 0.03,
                "hard_failure_improvement": 0.02,
                "easy_degradation": 0.002,
                "switch_rate": 0.16,
                "near_collision_005_delta_vs_endpoint": -0.001,
            },
        },
    }
    cq = {
        "stage42_cq_gate": {"passed": 19, "total": 19},
        "bootstrap_vs_endpoint_ade": {"all": {"low": 0.01}, "t50": {"low": 0.005}},
    }
    di = {"stage42_di_gate": {"passed": 17, "total": 17}}
    dl = {
        "stage42_dl_gate": {"passed": 30, "total": 30},
        "real_batch_replay": {
            "selected_xy_max_abs_diff": 0.0,
            "switch_exact_match": True,
            "metric": {
                "all_improvement": 0.24,
                "t50_improvement": 0.22,
                "t100_raw_frame_diagnostic_improvement": 0.14,
                "hard_failure_improvement": 0.23,
                "easy_degradation": -0.25,
                "switch_rate": 0.58,
            },
            "diagnostics": {"base_near_005": 0.02, "final_near_005": 0.01, "floor_near_005": 0.02},
        },
    }
    dm = {"stage42_dm_gate": {"passed": 21, "total": 21}}
    return {
        "inputs": {
            "proximity_guard_ablation": cr,
            "proximity_aware_guard": cq,
            "group_consistency_repair": di,
            "group_consistency_runtime": dl,
            "reviewer_replay_package": dm,
        },
        "deployment_variants": dn._build_variants(cr, cq, dl),
        "recommended_policy": {"safety_sensitive_default": "proximity_guard"},
        "baseline_mixing_caveat": True,
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def test_gate_passes_for_clean_variant_card() -> None:
    gate = dn._gate(_payload())
    assert gate["verdict"] == "stage42_dn_deployment_variant_card_pass"
    assert gate["passed"] == gate["total"]


def test_no_guard_is_not_marked_safety_deployable() -> None:
    variants = {row["variant"]: row for row in _payload()["deployment_variants"]}
    assert variants["no_proximity_guard"]["deployment_status"] == "diagnostic_not_safety_sensitive"
    assert variants["no_proximity_guard"]["safety"]["near_collision_005_delta_vs_endpoint"] > 0


def test_gate_rejects_missing_baseline_mixing_caveat() -> None:
    payload = _payload()
    payload["baseline_mixing_caveat"] = False
    gate = dn._gate(payload)
    assert gate["gates"]["baseline_mixing_caveat_present"] is False
    assert gate["passed"] == gate["total"] - 1
