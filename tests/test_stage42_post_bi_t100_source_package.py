from src import stage42_post_bi_t100_source_package as s42bj


def test_domain_support_keeps_ucy_repair_and_blocks_global_domains() -> None:
    bi = {
        "summary": {"ucy_t100_source_cv_supported": True},
        "support_matrix": {
            "ETH_UCY": {"independent_sources": 1},
            "UCY": {"independent_sources": 4},
            "TrajNet": {"independent_sources": 0},
        },
        "domain_summary": {
            "UCY": {
                "by_horizon": {
                    "100": {
                        "mean_holdout_improvement_vs_fallback": 0.4,
                        "minimum_holdout_improvement_vs_fallback": 0.3,
                        "maximum_easy_degradation": 0.01,
                    }
                }
            }
        },
    }
    support = s42bj._domain_support_from_bi(bi)
    assert support["UCY"]["t100_source_cv_supported"] is True
    assert support["UCY"]["additional_independent_sources_needed"] == 0
    assert support["ETH_UCY"]["additional_independent_sources_needed"] == 2
    assert support["TrajNet"]["additional_independent_sources_needed"] == 3


def test_inventory_exhaustion_marks_domains_without_novel_candidates() -> None:
    bd = {
        "inventory": [
            {"relative_path": "ETH/seq_eth/obsmat.txt", "t100_capable": True, "already_used_by_stage42": True},
            {"relative_path": "ETH/seq_eth/biwi_eth_10fps.txt", "t100_capable": True},
            {"relative_path": "UCY/zara01/obsmat.txt", "t100_capable": True, "novel_candidate": True},
        ]
    }
    support = {
        "ETH_UCY": {"independent_sources": 1, "additional_independent_sources_needed": 2},
        "UCY": {"independent_sources": 1, "additional_independent_sources_needed": 0},
        "TrajNet": {"independent_sources": 0, "additional_independent_sources_needed": 3},
    }
    audit = s42bj._local_inventory_exhaustion(bd, support)
    assert audit["independent_t100_group_count_by_domain"]["ETH_UCY"] == 1
    assert "ETH_UCY" in audit["local_inventory_exhausted_for_domains"]
    assert "TrajNet" in audit["local_inventory_exhausted_for_domains"]
    assert "UCY" not in audit["local_inventory_exhausted_for_domains"]


def test_inventory_exhaustion_does_not_mark_new_independent_group_exhausted() -> None:
    bd = {
        "inventory": [
            {"relative_path": "ETH/seq_eth/obsmat.txt", "t100_capable": True},
            {"relative_path": "ETH/seq_hotel/obsmat.txt", "t100_capable": True},
        ]
    }
    support = {
        "ETH_UCY": {"independent_sources": 1, "additional_independent_sources_needed": 2},
        "UCY": {"additional_independent_sources_needed": 0},
        "TrajNet": {"independent_sources": 0, "additional_independent_sources_needed": 3},
    }
    audit = s42bj._local_inventory_exhaustion(bd, support)
    assert audit["independent_t100_group_count_by_domain"]["ETH_UCY"] == 2
    assert "ETH_UCY" not in audit["local_inventory_exhausted_for_domains"]
    assert "TrajNet" in audit["local_inventory_exhausted_for_domains"]


def test_action_queue_uses_bc_candidates_for_blocked_domains() -> None:
    support = {
        "ETH_UCY": {"additional_independent_sources_needed": 2},
        "UCY": {"additional_independent_sources_needed": 0},
        "TrajNet": {"additional_independent_sources_needed": 3},
    }
    bc = {
        "candidates": [
            {
                "id": "trajnetpp",
                "dataset_name": "TrajNet++",
                "target_domains": ["TrajNet"],
                "priority_score": 90,
                "official_url": "official",
                "local_status": {"local_path_found": False, "found_paths": []},
                "download_policy": {"auto_download_allowed": False, "blocked_reasons": ["terms"]},
                "expected_t100_role": "repair TrajNet",
            },
            {
                "id": "eth_ucy",
                "dataset_name": "ETH/UCY",
                "target_domains": ["ETH_UCY"],
                "priority_score": 85,
                "official_url": "official2",
                "local_status": {"local_path_found": True, "found_paths": ["path"]},
                "download_policy": {"auto_download_allowed": False, "blocked_reasons": ["manual"]},
                "expected_t100_role": "repair ETH_UCY",
            },
        ]
    }
    actions = s42bj._rank_actions(support, bc)
    assert {action["domain"] for action in actions} == {"ETH_UCY", "TrajNet"}
    assert any(action["candidate_source_ids"] == ["trajnetpp"] for action in actions)


def test_gate_requires_no_overclaim_and_explicit_blockers() -> None:
    payload = {
        "source": "fresh_post_bi_t100_source_package",
        "bi_verdict": "stage42_bi_ucy_t100_easy_guard_repair_pass_with_global_blocker",
        "bc_verdict": "stage42_bc_t100_source_acquisition_plan_pass",
        "bd_verdict": "stage42_bd_local_t100_source_inventory_pass",
        "domain_support": {
            "ETH_UCY": {"additional_independent_sources_needed": 2},
            "UCY": {"additional_independent_sources_needed": 0},
            "TrajNet": {"additional_independent_sources_needed": 3},
        },
        "local_inventory_exhaustion": {"local_inventory_exhausted_for_domains": ["ETH_UCY", "TrajNet"]},
        "action_queue": [{"domain": "ETH_UCY"}, {"domain": "TrajNet"}],
        "strict_protocol": {
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_metrics_for_threshold": False,
        },
        "summary": {
            "ucy_t100_repaired": True,
            "auto_download_executed": False,
            "global_t100_positive_claim_allowed": False,
        },
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "t100_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    gate = s42bj._gate(payload)
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_bj_post_bi_t100_source_package_pass"
