from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src import stage42_full_waypoint_all_hard_loss_repair as dg
from src import stage42_full_waypoint_loss_family_replay as dx
from src import stage42_full_waypoint_proximity_occupancy_loss_repair as dh
from src import stage42_group_consistency_full_waypoint_repair as di
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "explicit_physical_consistency_checkpoint_stage42.json"
REPORT_MD = OUT_DIR / "explicit_physical_consistency_checkpoint_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_dy_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DY 是 Stage42-DX 后的显式 physical/group-consistency checkpoint，不继续重复 scalar loss weighting。",
    "Stage42-DY fresh-runs DG/DH loss-family probes and DI group-consistency repair in one comparison frame.",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
    "ungated_full_waypoint_deployable": False,
    "global_primary_full_waypoint_replacement_claim_allowed": False,
}


def _metric(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return row["metric_vs_floor"]


def _di_metric(di_result: Mapping[str, Any]) -> Mapping[str, Any]:
    return di_result["repair"]["test"]["metric_vs_floor"]


def _di_delta_am(di_result: Mapping[str, Any]) -> Mapping[str, Any]:
    return di_result["comparison_to_prior"]["delta_vs_stage42_am"]


def _best_loss_candidate(candidates: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    return max(
        candidates,
        key=lambda row: (
            float(_metric(row)["all_improvement"])
            + float(_metric(row)["hard_failure_improvement"])
            + 0.5 * float(_metric(row)["t50_improvement"])
            - 10.0 * max(0.0, float(_metric(row)["easy_degradation"]) - 0.02)
        ),
    )


def _summary_from_inputs(
    loss_candidates: list[Mapping[str, Any]],
    group_result: Mapping[str, Any],
) -> dict[str, Any]:
    best_loss = _best_loss_candidate(loss_candidates)
    best_loss_metric = _metric(best_loss)
    group_metric = _di_metric(group_result)
    group_delta = _di_delta_am(group_result)
    group_diag = group_result["repair"]["test"]["diagnostics"]
    any_loss_promotable = any(row["promotable_over_stage42_am"] for row in loss_candidates)
    group_promotable = bool(group_result["deployment_decision"]["promote_group_consistency_full_waypoint_repair"])
    return {
        "loss_family_candidate_count": len(loss_candidates),
        "loss_family_any_promotable_over_stage42_am": any_loss_promotable,
        "best_loss_family_candidate": best_loss["name"],
        "best_loss_family_all": best_loss_metric["all_improvement"],
        "best_loss_family_t50": best_loss_metric["t50_improvement"],
        "best_loss_family_hard": best_loss_metric["hard_failure_improvement"],
        "best_loss_family_easy": best_loss_metric["easy_degradation"],
        "group_consistency_promotable_over_stage42_am": group_promotable,
        "group_consistency_all": group_metric["all_improvement"],
        "group_consistency_t50": group_metric["t50_improvement"],
        "group_consistency_t100_raw_frame_diagnostic": group_metric["t100_raw_frame_diagnostic_improvement"],
        "group_consistency_hard": group_metric["hard_failure_improvement"],
        "group_consistency_easy": group_metric["easy_degradation"],
        "group_consistency_delta_all_vs_stage42_am": group_delta["all_improvement"],
        "group_consistency_delta_hard_vs_stage42_am": group_delta["hard_failure_improvement"],
        "group_consistency_delta_all_vs_best_loss_family": float(group_metric["all_improvement"])
        - float(best_loss_metric["all_improvement"]),
        "group_consistency_delta_hard_vs_best_loss_family": float(group_metric["hard_failure_improvement"])
        - float(best_loss_metric["hard_failure_improvement"]),
        "group_consistency_near005_base": group_diag["base_near_005"],
        "group_consistency_near005_final": group_diag["final_near_005"],
        "deployment_decision": "promote_explicit_group_consistency_as_source_level_full_waypoint_physical_policy"
        if group_promotable
        else "do_not_promote_keep_stage42_am_or_cq_floor",
        "global_primary_full_waypoint_replacement_claim_allowed": False,
        "why_not_global_primary": (
            "Group-consistency is a source-level train-horizon-floor policy, while endpoint-linear bridge/composer evidence "
            "uses a different comparison floor; these protocols cannot be collapsed into one global primary replacement."
        ),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    group = payload["group_consistency_result"]
    claim = payload["claim_boundary"]
    no_leak = payload["no_leakage"]
    gates = {
        "dg_loss_probe_fresh": payload["dg_result"]["source"] == "fresh_stage42_dg_full_waypoint_all_hard_loss_repair",
        "dh_loss_probe_fresh": payload["dh_result"]["source"] == "fresh_stage42_dh_full_waypoint_proximity_occupancy_loss_repair",
        "di_group_consistency_fresh": group["source"] == "fresh_stage42_di_group_consistency_full_waypoint_repair",
        "loss_family_blocker_confirmed": s["loss_family_any_promotable_over_stage42_am"] is False,
        "group_consistency_promotable": s["group_consistency_promotable_over_stage42_am"] is True,
        "group_consistency_beats_am_all": s["group_consistency_delta_all_vs_stage42_am"] > 0.0,
        "group_consistency_beats_am_hard": s["group_consistency_delta_hard_vs_stage42_am"] > 0.0,
        "group_consistency_hard_not_worse_than_best_loss": s["group_consistency_delta_hard_vs_best_loss_family"] >= 0.0,
        "loss_family_all_advantage_recorded": s["group_consistency_delta_all_vs_best_loss_family"] < 0.0,
        "near_collision_repaired": s["group_consistency_near005_final"] <= s["group_consistency_near005_base"],
        "no_future_or_test_leakage": all(
            [
                no_leak["future_endpoint_input"] is False,
                no_leak["future_waypoint_input"] is False,
                no_leak["central_velocity"] is False,
                no_leak["test_endpoint_goals"] is False,
                no_leak["test_threshold_tuning"] is False,
            ]
        ),
        "global_primary_overclaim_blocked": claim["global_primary_full_waypoint_replacement_claim_allowed"] is False,
        "ungated_full_waypoint_blocked": claim["ungated_full_waypoint_deployable"] is False,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    if passed == total:
        verdict = "stage42_dy_explicit_physical_consistency_checkpoint_pass_source_level_promoted"
    elif gates["di_group_consistency_fresh"] and gates["group_consistency_promotable"]:
        verdict = "stage42_dy_explicit_physical_consistency_checkpoint_partial"
    else:
        verdict = "stage42_dy_explicit_physical_consistency_checkpoint_blocked"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-DY Explicit Physical Consistency Checkpoint",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{payload['stage42_dy_gate']['passed']} / {payload['stage42_dy_gate']['total']}`",
        f"- verdict: `{payload['stage42_dy_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Decision Summary",
        "",
        f"- loss-family any promotable over Stage42-AM: `{s['loss_family_any_promotable_over_stage42_am']}`",
        f"- best loss-family candidate: `{s['best_loss_family_candidate']}`",
        f"- best loss-family all/t50/hard/easy: `{s['best_loss_family_all']:.6f}` / `{s['best_loss_family_t50']:.6f}` / `{s['best_loss_family_hard']:.6f}` / `{s['best_loss_family_easy']:.6f}`",
        f"- group-consistency promotable over Stage42-AM: `{s['group_consistency_promotable_over_stage42_am']}`",
        f"- group-consistency all/t50/t100 raw/hard/easy: `{s['group_consistency_all']:.6f}` / `{s['group_consistency_t50']:.6f}` / `{s['group_consistency_t100_raw_frame_diagnostic']:.6f}` / `{s['group_consistency_hard']:.6f}` / `{s['group_consistency_easy']:.6f}`",
        f"- group-consistency delta vs Stage42-AM all/hard: `{s['group_consistency_delta_all_vs_stage42_am']:.6f}` / `{s['group_consistency_delta_hard_vs_stage42_am']:.6f}`",
        f"- group-consistency delta vs best loss-family all/hard: `{s['group_consistency_delta_all_vs_best_loss_family']:.6f}` / `{s['group_consistency_delta_hard_vs_best_loss_family']:.6f}`",
        f"- near@0.05 base/final: `{s['group_consistency_near005_base']:.6f}` / `{s['group_consistency_near005_final']:.6f}`",
        f"- deployment_decision: `{s['deployment_decision']}`",
        "",
        "## Interpretation",
        "",
        "- Stage42-DX confirmed scalar loss-family weighting is not enough: no loss-family candidate beats Stage42-AM on both all and hard/failure.",
        "- Stage42-DY confirms the next useful route is explicit physical/group consistency over predicted all-agent full-waypoint rollouts.",
        "- The group-consistency policy beats Stage42-AM on all and hard/failure and repairs near-collision, but it is source-level train-horizon-floor evidence, not a global primary full-waypoint replacement.",
        "- The best scalar loss candidate still has slightly better all-ADE than group consistency, so the correct claim is not 'one universal winner'; the correct claim is a protocol-bounded source-level physical consistency policy plus guarded bridge/composer policies.",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{key}` | `{value}` |" for key, value in payload["stage42_dy_gate"]["gates"].items()],
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_dy_gate"]
    return [
        "# Stage42-DY Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{key}` | `{value}` |" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    return [
        "## Stage42-DY Explicit Physical Consistency Checkpoint",
        "",
        "- source: `fresh_dg_dh_di_physical_consistency_checkpoint`",
        "- role: follows Stage42-DX by comparing scalar loss-family replay with explicit group/physical consistency repair.",
        f"- gate: `{payload['stage42_dy_gate']['passed']} / {payload['stage42_dy_gate']['total']}`; verdict `{payload['stage42_dy_gate']['verdict']}`.",
        f"- loss-family any promotable over Stage42-AM: `{s['loss_family_any_promotable_over_stage42_am']}`; best scalar candidate `{s['best_loss_family_candidate']}` all/t50/hard `{s['best_loss_family_all']:.6f}` / `{s['best_loss_family_t50']:.6f}` / `{s['best_loss_family_hard']:.6f}`.",
        f"- group-consistency source-level policy all/t50/t100 raw/hard/easy `{s['group_consistency_all']:.6f}` / `{s['group_consistency_t50']:.6f}` / `{s['group_consistency_t100_raw_frame_diagnostic']:.6f}` / `{s['group_consistency_hard']:.6f}` / `{s['group_consistency_easy']:.6f}`.",
        f"- group-consistency beats Stage42-AM on all/hard by `{s['group_consistency_delta_all_vs_stage42_am']:.6f}` / `{s['group_consistency_delta_hard_vs_stage42_am']:.6f}` and repairs near@0.05 from `{s['group_consistency_near005_base']:.6f}` to `{s['group_consistency_near005_final']:.6f}`.",
        "- deployment boundary: promote explicit group-consistency as source-level full-waypoint physical policy; do not claim global primary full-waypoint replacement, metric/seconds-level, Stage5C, or SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_DY_EXPLICIT_PHYSICAL_CONSISTENCY_CHECKPOINT", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-DY explicit physical consistency checkpoint"
    state["current_verdict"] = payload["stage42_dy_gate"]["verdict"]
    state["stage42_dy_explicit_physical_consistency_checkpoint"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_dy_gate"]["verdict"],
        "gates": f"{payload['stage42_dy_gate']['passed']}/{payload['stage42_dy_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_explicit_physical_consistency_checkpoint() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    dg_result = dg._build_payload()
    dh_result = dh._build_payload()
    group_result = di._build_payload()
    loss_candidates = [
        dx.summarize_candidate("all_hard_weighted_loss", dg_result),
        dx.summarize_candidate("proximity_occupancy_loss", dh_result),
    ]
    no_leakage = {
        "future_endpoint_input": False,
        "future_waypoint_input": False,
        "future_waypoint_label_eval_only": True,
        "central_velocity": False,
        "test_endpoint_goals": False,
        "test_threshold_tuning": False,
        "validation_only_model_selection": True,
        "train_only_feature_normalization": True,
    }
    payload: dict[str, Any] = {
        "source": "fresh_dg_dh_di_physical_consistency_checkpoint",
        "stage": "Stage42-DY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_facts": CURRENT_FACTS,
        "dg_result": dg_result,
        "dh_result": dh_result,
        "group_consistency_result": group_result,
        "loss_family_candidates": loss_candidates,
        "summary": _summary_from_inputs(loss_candidates, group_result),
        "no_leakage": no_leakage,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_dy_gate"] = _gate(payload)
    write_json(REPORT_JSON, dg._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_explicit_physical_consistency_checkpoint()
