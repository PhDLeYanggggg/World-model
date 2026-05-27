from __future__ import annotations

from src.stage42_post_ea_paper_refresh import _gate, _refresh_lines, _summary


def _fake_dy() -> dict:
    return {
        "stage42_dy_gate": {"passed": 16, "total": 16},
        "summary": {
            "loss_family_any_promotable_over_stage42_am": False,
            "best_loss_family_candidate": "proximity_occupancy_loss",
            "best_loss_family_all": 0.255061,
            "best_loss_family_t50": 0.221366,
            "best_loss_family_hard": 0.237393,
            "group_consistency_promotable_over_stage42_am": True,
            "group_consistency_all": 0.247157,
            "group_consistency_t50": 0.223630,
            "group_consistency_t100_raw_frame_diagnostic": 0.143461,
            "group_consistency_hard": 0.238874,
            "group_consistency_easy": -0.256309,
            "group_consistency_delta_all_vs_stage42_am": 0.001368,
            "group_consistency_delta_hard_vs_stage42_am": 0.001380,
            "group_consistency_near005_base": 0.019364,
            "group_consistency_near005_final": 0.013823,
        },
    }


def _fake_dz() -> dict:
    return {
        "stage42_dz_gate": {"passed": 15, "total": 15},
        "summary": {
            "positive_safe_domains": 2,
            "ucy_all": 0.355808,
            "ucy_t50": 0.227206,
            "ucy_hard": 0.337848,
            "trajnet_all": 0.320715,
            "trajnet_t50": 0.281804,
            "trajnet_hard": 0.312868,
        },
    }


def _ci(low: float, high: float) -> dict:
    return {"low": low, "high": high}


def _fake_ea() -> dict:
    return {
        "stage42_ea_gate": {"passed": 12, "total": 12},
        "summary": {"bootstrap_n": 2000, "ci_positive_safe_domains": 2},
        "bootstrap_ci": {
            "global": {
                "all": _ci(0.325616, 0.332),
                "t50": _ci(0.265328, 0.274),
                "hard_failure": _ci(0.315115, 0.323),
                "easy_degradation": _ci(-0.329, -0.312813),
            },
            "by_domain": {
                "UCY": {
                    "all": _ci(0.346983, 0.365),
                    "t50": _ci(0.213784, 0.241),
                    "hard_failure": _ci(0.328373, 0.347),
                },
                "TrajNet": {
                    "all": _ci(0.317175, 0.324),
                    "t50": _ci(0.277244, 0.286),
                    "hard_failure": _ci(0.308982, 0.317),
                },
            },
        },
        "near_collision_ci": {
            "global": {"delta_final_minus_base": _ci(-0.008, -0.006722)}
        },
    }


def test_stage42_eb_summary_blocks_overclaims() -> None:
    summary = _summary(_fake_dy(), _fake_dz(), _fake_ea())

    assert summary["loss_family_boundary"]["any_loss_family_promotable"] is False
    assert summary["explicit_group_consistency"]["source_level_promoted"] is True
    assert summary["dual_domain_support"]["positive_safe_domains"] == 2
    assert summary["paper_verdict"]["loss_family_primary_claim_allowed"] is False
    assert summary["paper_verdict"]["stage5c_execution_allowed"] is False
    assert summary["paper_verdict"]["smc_allowed"] is False


def test_stage42_eb_refresh_lines_record_dual_domain_and_boundaries() -> None:
    lines = "\n".join(_refresh_lines(_summary(_fake_dy(), _fake_dz(), _fake_ea())))

    assert "UCY+TrajNet bootstrap-backed raw-frame evidence" in lines
    assert "scalar loss-family promotion remains blocked" in lines
    assert "explicit physical/group-consistency as a source-level" in lines
    assert "Still forbidden: true 3D" in lines
    assert "Stage5C execution" in lines
    assert "SMC readiness" in lines


def test_stage42_eb_gate_passes_with_refreshed_paper_files() -> None:
    summary = _summary(_fake_dy(), _fake_dz(), _fake_ea())
    payload = {
        "source": "unit",
        "inputs": {
            "stage42_dy": _fake_dy(),
            "stage42_dz": _fake_dz(),
            "stage42_ea": _fake_ea(),
        },
        "paper_refresh_summary": summary,
        "paper_file_status": [
            {
                "contains_stage42_eb": True,
                "contains_dual_domain_bootstrap": True,
                "contains_loss_family_blocker": True,
                "contains_group_consistency_claim": True,
                "contains_non_claims": True,
            }
        ],
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }

    gate = _gate(payload)

    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage42_eb_post_ea_paper_refresh_pass"
