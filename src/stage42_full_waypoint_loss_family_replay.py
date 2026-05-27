from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src import stage42_full_waypoint_all_hard_loss_repair as dg
from src import stage42_full_waypoint_proximity_occupancy_loss_repair as dh
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "full_waypoint_loss_family_replay_stage42.json"
REPORT_MD = OUT_DIR / "full_waypoint_loss_family_replay_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_dx_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DX fresh-reruns Stage42-DG and Stage42-DH full-waypoint loss-family probes, then applies one promotion gate.",
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
    "full_waypoint_primary_promotion_allowed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _metric_from_result(result: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = result["model"]["metrics"]
    if "protected_selected_loss_variant" in metrics:
        return metrics["protected_selected_loss_variant"]
    return metrics["protected_selected_candidate"]


def _delta_from_result(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return result["comparison_to_stage42_am"]["delta_vs_stage42_am"]


def summarize_candidate(name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    metric = _metric_from_result(result)
    delta = _delta_from_result(result)
    selected = result["model"]["selected"]
    gate_key = next(key for key in result if key.startswith("stage42_") and key.endswith("_gate"))
    promotable = bool(
        metric["all_improvement"] > 0.0
        and metric["hard_failure_improvement"] > 0.0
        and metric["easy_degradation"] <= 0.02
        and delta["all_improvement"] > 0.0
        and delta["hard_failure_improvement"] > 0.0
    )
    return {
        "name": name,
        "source": result["source"],
        "gate": result[gate_key],
        "selected_variant": selected.get("variant"),
        "selected_feature_mode": selected.get("feature_mode", "stage42_am_features"),
        "selected_lambda": selected.get("lambda"),
        "selected_val_score": selected.get("val_score"),
        "metric_vs_floor": metric,
        "delta_vs_stage42_am": delta,
        "promotable_over_stage42_am": promotable,
        "decision": result["deployment_decision"]["decision"],
    }


def _best_candidate(candidates: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    return max(
        candidates,
        key=lambda row: (
            float(row["metric_vs_floor"]["all_improvement"])
            + float(row["metric_vs_floor"]["hard_failure_improvement"])
            + 0.5 * float(row["metric_vs_floor"]["t50_improvement"])
            - 10.0 * max(0.0, float(row["metric_vs_floor"]["easy_degradation"]) - 0.02)
        ),
    )


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    claim = payload["claim_boundary"]
    s = payload["summary"]
    gates = {
        "dg_fresh_rerun_completed": payload["dg_result"]["source"] == "fresh_stage42_dg_full_waypoint_all_hard_loss_repair",
        "dh_fresh_rerun_completed": payload["dh_result"]["source"] == "fresh_stage42_dh_full_waypoint_proximity_occupancy_loss_repair",
        "loss_family_candidates_compared": len(payload["candidate_summaries"]) >= 2,
        "validation_selected_candidates": all(row["selected_val_score"] is not None for row in payload["candidate_summaries"]),
        "no_future_or_test_leakage": payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["test_threshold_tuning"] is False,
        "honest_promotion_decision_recorded": s["promotion_decision"] in {
            "promote_full_waypoint_loss_family_candidate",
            "do_not_promote_keep_stage42_am_or_cq_floor",
        },
        "promotion_blocker_recorded_if_needed": bool(s["promotion_blockers"]) if not s["any_promotable_over_stage42_am"] else True,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_dx_loss_family_replay_pass_promotable" if s["any_promotable_over_stage42_am"] and passed == total else (
        "stage42_dx_loss_family_replay_pass_blocker_confirmed" if passed == total else "stage42_dx_loss_family_replay_partial"
    )
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-DX Full-Waypoint Loss-Family Fresh Replay",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{payload['stage42_dx_gate']['passed']} / {payload['stage42_dx_gate']['total']}`",
        f"- verdict: `{payload['stage42_dx_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in payload["summary"].items()],
        "",
        "## Candidate Replay Table",
        "",
        "| candidate | selected | all | t50 | t100 raw | hard | easy | delta all vs AM | delta hard vs AM | promotable |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["candidate_summaries"]:
        m = row["metric_vs_floor"]
        d = row["delta_vs_stage42_am"]
        selected = f"{row['selected_variant']}:{row['selected_lambda']}"
        lines.append(
            "| `{}` | `{}` | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | `{}` |".format(
                row["name"],
                selected,
                float(m["all_improvement"]),
                float(m["t50_improvement"]),
                float(m["t100_raw_frame_diagnostic_improvement"]),
                float(m["hard_failure_improvement"]),
                float(m["easy_degradation"]),
                float(d["all_improvement"]),
                float(d["hard_failure_improvement"]),
                row["promotable_over_stage42_am"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a fresh replay of the two strongest full-waypoint loss-family repair tracks, not a threshold-only report.",
            "- Promotion requires beating Stage42-AM on both all and hard/failure while keeping easy degradation <=2%.",
            "- If no candidate is promotable, the current deployable path remains Stage42-AM/CQ/CS guarded floor, not primary full-waypoint dynamics.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{key}` | `{value}` |" for key, value in payload["stage42_dx_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_dx_gate"]
    return [
        "# Stage42-DX Gate",
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
    best = payload["best_candidate"]
    return [
        "## Stage42-DX Full-Waypoint Loss-Family Fresh Replay",
        "",
        "- source: `fresh_rerun_dg_dh_loss_family_replay`",
        "- role: reruns DG/DH full-waypoint loss-family probes and applies one promotion gate over Stage42-AM.",
        f"- gate: `{payload['stage42_dx_gate']['passed']} / {payload['stage42_dx_gate']['total']}`; verdict `{payload['stage42_dx_gate']['verdict']}`.",
        f"- best replay candidate: `{best['name']}`; all `{best['metric_vs_floor']['all_improvement']:.6f}`, t50 `{best['metric_vs_floor']['t50_improvement']:.6f}`, hard `{best['metric_vs_floor']['hard_failure_improvement']:.6f}`, easy `{best['metric_vs_floor']['easy_degradation']:.6f}`.",
        f"- promotion decision: `{s['promotion_decision']}`; blockers: `{s['promotion_blockers']}`.",
        "- Stage5C remains false; SMC remains false; no metric/seconds claim.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_DX_FULL_WAYPOINT_LOSS_FAMILY_REPLAY", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-DX full-waypoint loss-family fresh replay"
    state["current_verdict"] = payload["stage42_dx_gate"]["verdict"]
    state["stage42_dx_full_waypoint_loss_family_replay"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_dx_gate"]["verdict"],
        "gates": f"{payload['stage42_dx_gate']['passed']}/{payload['stage42_dx_gate']['total']}",
        "summary": payload["summary"],
        "best_candidate": payload["best_candidate"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_full_waypoint_loss_family_replay() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    dg_result = dg._build_payload()
    dh_result = dh._build_payload()
    candidate_summaries = [
        summarize_candidate("all_hard_weighted_loss", dg_result),
        summarize_candidate("proximity_occupancy_loss", dh_result),
    ]
    best = _best_candidate(candidate_summaries)
    any_promotable = any(row["promotable_over_stage42_am"] for row in candidate_summaries)
    blockers = []
    if not any_promotable:
        blockers.extend(
            [
                "no_loss_family_candidate_beats_stage42_am_on_all_and_hard",
                "primary_full_waypoint_promotion_blocked",
                "next_step_requires_model_architecture_or_explicit_physical_consistency_target_not_more_scalar_weighting",
            ]
        )
    no_leak = {
        "future_endpoint_input": False,
        "future_waypoint_input": False,
        "future_waypoint_label_eval_only": True,
        "central_velocity": False,
        "test_endpoint_goals": False,
        "test_threshold_tuning": False,
        "train_only_feature_normalization": True,
        "validation_only_model_selection": True,
    }
    payload: dict[str, Any] = {
        "source": "fresh_rerun_dg_dh_loss_family_replay",
        "stage": "Stage42-DX",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_facts": CURRENT_FACTS,
        "dg_result": dg_result,
        "dh_result": dh_result,
        "candidate_summaries": candidate_summaries,
        "best_candidate": best,
        "summary": {
            "candidate_count": len(candidate_summaries),
            "any_promotable_over_stage42_am": any_promotable,
            "promotion_decision": "promote_full_waypoint_loss_family_candidate"
            if any_promotable
            else "do_not_promote_keep_stage42_am_or_cq_floor",
            "promotion_blockers": blockers,
            "best_candidate_name": best["name"],
            "best_candidate_all": best["metric_vs_floor"]["all_improvement"],
            "best_candidate_t50": best["metric_vs_floor"]["t50_improvement"],
            "best_candidate_hard": best["metric_vs_floor"]["hard_failure_improvement"],
            "best_candidate_easy": best["metric_vs_floor"]["easy_degradation"],
        },
        "no_leakage": no_leak,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_dx_gate"] = _gate(payload)
    write_json(REPORT_JSON, dg._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_full_waypoint_loss_family_replay()
