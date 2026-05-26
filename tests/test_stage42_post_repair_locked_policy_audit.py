from src import stage42_post_repair_locked_policy_audit as s42ak


def test_stage42ak_policy_rules_are_validation_only() -> None:
    af = {"guard_rule": {"threshold": 0.02}, "repair_effect": {}}
    ag = {"source_repair_rule": {"uses_test_metrics_for_threshold": False}, "repair_effect": {}}
    ai = {"source_repair_rule": {"uses_test_metrics_for_threshold": False}, "repair_effect": {}}
    x = {"stage": "Stage42-X", "cache_hash": "abc", "input_hash": "def", "rows": {"test": 10}}
    split = {"proposed_split_no_source_overlap": True, "no_leakage": {"frozen_eval_uses_old_train_rows": False}}
    policy = s42ak._build_policy(af, ag, ai, x, split)
    assert [rule["rule_id"] for rule in policy["ordered_policy_rules"]] == [
        "base_stage42x_row_level_full_waypoint_policy",
        "stage42af_validation_margin_guard",
        "stage42ag_eth_ucy_t50_fde_source_repair",
        "stage42ai_trajnet_t100_easy_safety_repair",
    ]
    assert all(rule["uses_test_metrics_for_threshold"] is False for rule in policy["ordered_policy_rules"])
    assert policy["claim_boundary"]["stage5c_executed"] is False
    assert policy["claim_boundary"]["smc_enabled"] is False


def test_stage42ak_no_leakage_combines_split_and_sources() -> None:
    policy = {
        "ordered_policy_rules": [
            {"rule_id": "stage42af_validation_margin_guard", "uses_test_metrics_for_threshold": False},
            {"rule_id": "stage42ag_eth_ucy_t50_fde_source_repair", "uses_test_metrics_for_threshold": False},
            {"rule_id": "stage42ai_trajnet_t100_easy_safety_repair", "uses_test_metrics_for_threshold": False},
        ]
    }
    split = {
        "proposed_split_no_source_overlap": True,
        "no_leakage": {"frozen_eval_uses_old_train_rows": False},
    }
    no_leakage = s42ak._combined_no_leakage(policy, {"no_leakage": {}}, {"no_leakage": {}}, {"no_leakage": {}}, split)
    assert no_leakage["passed"] is True
    assert no_leakage["checks"]["source_overlap_pass"] is True
    assert no_leakage["checks"]["frozen_eval_uses_old_train_rows"] is False


def test_stage42ak_gate_requires_locked_policy_and_claim_boundary(tmp_path, monkeypatch) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(s42ak, "POLICY_JSON", policy_path)
    payload = {
        "inputs": {
            "stage42af": {"stage42_af_gate": {"passed": 13, "total": 13, "verdict": "stage42_af_weak_slice_guard_repair_pass_with_eth_t50_limitation"}},
            "stage42ag": {"stage42_ag_gate": {"passed": 13, "total": 13, "verdict": "stage42_ag_eth_t50_fde_source_repair_pass"}},
            "stage42ai": {"stage42_ai_gate": {"passed": 13, "total": 13, "verdict": "stage42_ai_trajnet_t100_safety_repair_pass"}},
            "stage42aj": {"stage42_aj_gate": {"passed": 10, "total": 10, "verdict": "stage42_aj_post_repair_paper_package_refresh_pass"}},
        },
        "external_source_split": {"source_overlap_pass": True},
        "no_leakage": {
            "passed": True,
            "checks": {"frozen_eval_uses_old_train_rows": False},
        },
        "post_repair_summary": {
            "ade_all_ci_low": 0.01,
            "ade_t50_ci_low": 0.01,
            "ade_hard_failure_ci_low": 0.01,
            "easy_degradation_ci_high": 0.01,
        },
        "policy": {
            "ordered_policy_rules": [{"uses_test_metrics_for_threshold": False}],
            "claim_boundary": {
                "t100_seconds_claim": False,
                "metric_or_seconds_claim": False,
                "stage5c_executed": False,
                "smc_enabled": False,
            },
        },
        "policy_hash": "policyhash",
        "input_hash": "inputhash",
    }
    gate = s42ak._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_ak_post_repair_locked_policy_audit_pass"
