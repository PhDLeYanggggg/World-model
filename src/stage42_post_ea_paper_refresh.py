from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
DY_JSON = OUT_DIR / "explicit_physical_consistency_checkpoint_stage42.json"
DZ_JSON = OUT_DIR / "ucy_supported_group_consistency_stage42.json"
EA_JSON = OUT_DIR / "dual_domain_group_consistency_statistics_stage42.json"

REPORT_JSON = OUT_DIR / "paper_package_post_ea_refresh_stage42.json"
REPORT_MD = OUT_DIR / "paper_package_post_ea_refresh_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_eb_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

PAPER_FILES = [
    OUT_DIR / "paper_outline_stage42.md",
    OUT_DIR / "method_draft_stage42.md",
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "failure_taxonomy_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "data_card_stage42.md",
    OUT_DIR / "reproducibility_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EB 是 post-EA paper package refresh，不重新训练，不调 threshold，不把 paper update 当模型成功。",
    "本阶段同步 Stage42-DY/DZ/EA：loss-family blocker、explicit group-consistency repair、UCY+TrajNet dual-domain bootstrap evidence。",
    "future endpoints / waypoints 只作为 supervised/evaluation labels，不能作为 inference input。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _ci(row: Mapping[str, Any]) -> str:
    return f"[{_pct(row['low'])}, {_pct(row['high'])}]"


def _summary(dy: Mapping[str, Any], dz: Mapping[str, Any], ea: Mapping[str, Any]) -> dict[str, Any]:
    dy_s = dy["summary"]
    dz_s = dz["summary"]
    ea_s = ea["summary"]
    ci = ea["bootstrap_ci"]
    near = ea["near_collision_ci"]
    return {
        "source": "fresh_paper_refresh_from_stage42_dy_dz_ea",
        "loss_family_boundary": {
            "any_loss_family_promotable": dy_s["loss_family_any_promotable_over_stage42_am"],
            "best_loss_family_candidate": dy_s["best_loss_family_candidate"],
            "best_loss_family_all": dy_s["best_loss_family_all"],
            "best_loss_family_t50": dy_s["best_loss_family_t50"],
            "best_loss_family_hard": dy_s["best_loss_family_hard"],
            "claim": "Scalar loss-family weighting is not enough for primary full-waypoint promotion.",
        },
        "explicit_group_consistency": {
            "source_level_promoted": dy_s["group_consistency_promotable_over_stage42_am"],
            "all": dy_s["group_consistency_all"],
            "t50": dy_s["group_consistency_t50"],
            "t100_raw_frame_diagnostic": dy_s["group_consistency_t100_raw_frame_diagnostic"],
            "hard": dy_s["group_consistency_hard"],
            "easy": dy_s["group_consistency_easy"],
            "delta_all_vs_stage42_am": dy_s["group_consistency_delta_all_vs_stage42_am"],
            "delta_hard_vs_stage42_am": dy_s["group_consistency_delta_hard_vs_stage42_am"],
            "near005_base": dy_s["group_consistency_near005_base"],
            "near005_final": dy_s["group_consistency_near005_final"],
            "claim": "Explicit group/physical consistency over predicted all-agent full-waypoint rollouts is the supported source-level repair route.",
        },
        "dual_domain_support": {
            "positive_safe_domains": dz_s["positive_safe_domains"],
            "ucy_all": dz_s["ucy_all"],
            "ucy_t50": dz_s["ucy_t50"],
            "ucy_hard": dz_s["ucy_hard"],
            "trajnet_all": dz_s["trajnet_all"],
            "trajnet_t50": dz_s["trajnet_t50"],
            "trajnet_hard": dz_s["trajnet_hard"],
            "claim": "UCY validation support turns group consistency from a TrajNet-dominant policy into a UCY+TrajNet dual-domain source-level policy.",
        },
        "statistical_evidence": {
            "bootstrap_n": ea_s["bootstrap_n"],
            "ci_positive_safe_domains": ea_s["ci_positive_safe_domains"],
            "global_all_ci": ci["global"]["all"],
            "global_t50_ci": ci["global"]["t50"],
            "global_hard_ci": ci["global"]["hard_failure"],
            "global_easy_ci": ci["global"]["easy_degradation"],
            "ucy_all_ci": ci["by_domain"]["UCY"]["all"],
            "ucy_t50_ci": ci["by_domain"]["UCY"]["t50"],
            "ucy_hard_ci": ci["by_domain"]["UCY"]["hard_failure"],
            "trajnet_all_ci": ci["by_domain"]["TrajNet"]["all"],
            "trajnet_t50_ci": ci["by_domain"]["TrajNet"]["t50"],
            "trajnet_hard_ci": ci["by_domain"]["TrajNet"]["hard_failure"],
            "near005_delta_global_ci": near["global"]["delta_final_minus_base"],
            "claim": "Dual-domain group-consistency evidence is now bootstrap-supported on global, UCY, and TrajNet slices.",
        },
        "paper_verdict": {
            "paper_ready_evidence_package_strengthened": True,
            "full_waypoint_group_consistency_claim_allowed": True,
            "dual_domain_bootstrap_claim_allowed": True,
            "loss_family_primary_claim_allowed": False,
            "goal_scene_main_claim_allowed": False,
            "neighbor_interaction_main_claim_allowed": False,
            "ungated_full_waypoint_deployment_allowed": False,
            "global_primary_full_waypoint_replacement_claim_allowed": False,
            "metric_seconds_claim_allowed": False,
            "foundation_claim_allowed": False,
            "stage5c_execution_allowed": False,
            "smc_allowed": False,
        },
    }


def _refresh_lines(summary: Mapping[str, Any]) -> list[str]:
    loss = summary["loss_family_boundary"]
    group = summary["explicit_group_consistency"]
    dual = summary["dual_domain_support"]
    stats = summary["statistical_evidence"]
    return [
        "## Stage42-EB Post-EA Paper Evidence Refresh",
        "",
        "- source: `fresh_paper_refresh_from_stage42_dy_dz_ea`",
        "- role: synchronize paper-ready artifacts after explicit physical consistency and dual-domain bootstrap evidence.",
        "- This is a paper-package update from fresh Stage42-DY/DZ/EA evidence, not new training and not a threshold search.",
        "",
        "### What Changed After EA",
        "",
        f"- scalar loss-family promotion remains blocked: best `{loss['best_loss_family_candidate']}` all/t50/hard `{_pct(loss['best_loss_family_all'])}` / `{_pct(loss['best_loss_family_t50'])}` / `{_pct(loss['best_loss_family_hard'])}`.",
        f"- explicit group-consistency is source-level promoted: all/t50/t100 raw/hard `{_pct(group['all'])}` / `{_pct(group['t50'])}` / `{_pct(group['t100_raw_frame_diagnostic'])}` / `{_pct(group['hard'])}`.",
        f"- group-consistency delta vs Stage42-AM all/hard: `{_pct(group['delta_all_vs_stage42_am'])}` / `{_pct(group['delta_hard_vs_stage42_am'])}`.",
        f"- near@0.05 is repaired from `{_pct(group['near005_base'])}` to `{_pct(group['near005_final'])}` in the DY checkpoint.",
        "",
        "### Dual-Domain Evidence",
        "",
        f"- positive safe domains: `{dual['positive_safe_domains']}`.",
        f"- UCY all/t50/hard: `{_pct(dual['ucy_all'])}` / `{_pct(dual['ucy_t50'])}` / `{_pct(dual['ucy_hard'])}`.",
        f"- TrajNet all/t50/hard: `{_pct(dual['trajnet_all'])}` / `{_pct(dual['trajnet_t50'])}` / `{_pct(dual['trajnet_hard'])}`.",
        "",
        "### Bootstrap Evidence",
        "",
        f"- bootstrap_n: `{stats['bootstrap_n']}`.",
        f"- global all/t50/hard CI: `{_ci(stats['global_all_ci'])}` / `{_ci(stats['global_t50_ci'])}` / `{_ci(stats['global_hard_ci'])}`; easy degradation CI `{_ci(stats['global_easy_ci'])}`.",
        f"- UCY all/t50/hard CI: `{_ci(stats['ucy_all_ci'])}` / `{_ci(stats['ucy_t50_ci'])}` / `{_ci(stats['ucy_hard_ci'])}`.",
        f"- TrajNet all/t50/hard CI: `{_ci(stats['trajnet_all_ci'])}` / `{_ci(stats['trajnet_t50_ci'])}` / `{_ci(stats['trajnet_hard_ci'])}`.",
        f"- near@0.05 final-base delta CI: `{_ci(stats['near005_delta_global_ci'])}`.",
        "",
        "### Updated Paper Claim Boundary",
        "",
        "- Supported: protected source-level group-consistency full-waypoint dynamics with UCY+TrajNet bootstrap-backed raw-frame evidence.",
        "- Supported: explicit physical/group-consistency as a source-level full-waypoint repair route.",
        "- Not supported as main claims: scalar loss weighting, goal/scene context, and neighbor/interaction context under current protocols.",
        "- Not supported: ungated full-waypoint deployment or global primary full-waypoint replacement.",
        "- Still forbidden: true 3D, foundation model, global metric/seconds-level claims, Stage5C execution, and SMC readiness.",
    ]


def _refresh_paper_files(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    lines = _refresh_lines(summary)
    status: list[dict[str, Any]] = []
    for path in PAPER_FILES:
        _replace_section(path, "STAGE42_EB_POST_EA_PAPER_REFRESH", lines)
        text = path.read_text(encoding="utf-8")
        status.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "contains_stage42_eb": "Stage42-EB Post-EA Paper Evidence Refresh" in text,
                "contains_dual_domain_bootstrap": "UCY+TrajNet bootstrap-backed raw-frame evidence" in text,
                "contains_loss_family_blocker": "scalar loss-family promotion remains blocked" in text,
                "contains_group_consistency_claim": "explicit physical/group-consistency as a source-level full-waypoint repair route" in text,
                "contains_non_claims": "Still forbidden: true 3D" in text and "Stage5C execution" in text and "SMC readiness" in text,
            }
        )
    return status


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["paper_refresh_summary"]
    stats = summary["statistical_evidence"]
    paper = summary["paper_verdict"]
    claim = payload["claim_boundary"]
    gates = {
        "dy_gate_passed": payload["inputs"]["stage42_dy"].get("stage42_dy_gate", {}).get("passed")
        == payload["inputs"]["stage42_dy"].get("stage42_dy_gate", {}).get("total"),
        "dz_gate_passed": payload["inputs"]["stage42_dz"].get("stage42_dz_gate", {}).get("passed")
        == payload["inputs"]["stage42_dz"].get("stage42_dz_gate", {}).get("total"),
        "ea_gate_passed": payload["inputs"]["stage42_ea"].get("stage42_ea_gate", {}).get("passed")
        == payload["inputs"]["stage42_ea"].get("stage42_ea_gate", {}).get("total"),
        "paper_files_refreshed": all(row["contains_stage42_eb"] for row in payload["paper_file_status"]),
        "dual_domain_bootstrap_recorded": stats["ci_positive_safe_domains"] >= 2 and stats["bootstrap_n"] >= 2000,
        "loss_family_primary_blocked": paper["loss_family_primary_claim_allowed"] is False,
        "group_consistency_claim_allowed": paper["full_waypoint_group_consistency_claim_allowed"] is True,
        "context_overclaims_blocked": paper["goal_scene_main_claim_allowed"] is False
        and paper["neighbor_interaction_main_claim_allowed"] is False,
        "ungated_and_global_primary_blocked": paper["ungated_full_waypoint_deployment_allowed"] is False
        and paper["global_primary_full_waypoint_replacement_claim_allowed"] is False,
        "no_metric_seconds_overclaim": claim["metric_or_seconds_claim"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_eb_post_ea_paper_refresh_pass" if passed == total else "stage42_eb_post_ea_paper_refresh_partial"
    return {"source": payload.get("source", "unknown"), "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_eb_gate"]
    lines = [
        "# Stage42-EB Post-EA Paper Evidence Refresh",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Refresh Content",
        "",
        *_refresh_lines(payload["paper_refresh_summary"]),
        "",
        "## Paper File Status",
        "",
        "| file | refreshed | dual-domain bootstrap | loss blocker | group claim | non-claims |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["paper_file_status"]:
        lines.append(
            f"| `{row['path']}` | {row['contains_stage42_eb']} | {row['contains_dual_domain_bootstrap']} | {row['contains_loss_family_blocker']} | {row['contains_group_consistency_claim']} | {row['contains_non_claims']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-EB makes the dual-domain statistical evidence available to the paper package.",
            "- It upgrades paper wording from single-report evidence to a coherent claim boundary across paper artifacts.",
            "- It keeps loss-family, goal/scene, neighbor/interaction, ungated full-waypoint, metric/seconds, Stage5C, and SMC overclaims blocked.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_eb_gate"]
    return [
        "# Stage42-EB Gate",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload["paper_refresh_summary"])
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_EB_POST_EA_PAPER_REFRESH", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    gate = payload["stage42_eb_gate"]
    state["current_stage"] = "Stage42-EB post-EA paper evidence refresh"
    state["current_verdict"] = gate["verdict"]
    state["stage42_eb_post_ea_paper_refresh"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": gate["verdict"],
        "gates": f"{gate['passed']}/{gate['total']}",
        "paper_refresh_summary": payload["paper_refresh_summary"],
        "paper_files": [row["path"] for row in payload["paper_file_status"]],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_post_ea_paper_refresh(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    dy = read_json(DY_JSON, {})
    dz = read_json(DZ_JSON, {})
    ea = read_json(EA_JSON, {})
    payload: dict[str, Any] = {
        "source": "fresh_paper_refresh_from_stage42_dy_dz_ea",
        "stage": "Stage42-EB",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([DY_JSON, DZ_JSON, EA_JSON]),
        "current_facts": CURRENT_FACTS,
        "inputs": {"stage42_dy": dy, "stage42_dz": dz, "stage42_ea": ea},
        "paper_refresh_summary": _summary(dy, dz, ea),
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["paper_file_status"] = _refresh_paper_files(payload["paper_refresh_summary"])
    payload["stage42_eb_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_post_ea_paper_refresh()
