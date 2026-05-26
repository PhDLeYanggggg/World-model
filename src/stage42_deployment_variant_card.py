from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "deployment_variant_card_stage42.json"
REPORT_MD = OUT_DIR / "deployment_variant_card_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_dn_gate.md"

CR_JSON = OUT_DIR / "proximity_guard_ablation_stage42.json"
CQ_JSON = OUT_DIR / "proximity_aware_composer_guard_stage42.json"
DI_JSON = OUT_DIR / "group_consistency_full_waypoint_repair_stage42.json"
DL_JSON = OUT_DIR / "group_consistency_runtime_policy_stage42.json"
DM_JSON = OUT_DIR / "reviewer_replay_package_stage42.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
RETRO_README = Path("README_M3W_GOAL_RETROSPECTIVE_CURRENT_ZH.md")
GOAL_SUMMARY_README = Path("README_M3W_CURRENT_GOAL_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DN 是 deployment variant card，不重新训练，不调 threshold。",
    "DN 区分 safety-sensitive deployable、accuracy-priority diagnostic、source-level full-waypoint runtime policy，避免混用 claim。",
    "future waypoints / endpoints 只作为 supervised/evaluation labels，不能作为 inference input。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "metric_or_seconds_claim": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _passed(payload: Mapping[str, Any], gate_name: str) -> bool:
    gate = payload.get(gate_name, {})
    return bool(gate) and gate.get("passed") == gate.get("total")


def _metric_summary(metric: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "all_improvement": metric.get("all_improvement"),
        "t50_improvement": metric.get("t50_improvement"),
        "t100_raw_frame_diagnostic_improvement": metric.get("t100_raw_frame_diagnostic_improvement"),
        "hard_failure_improvement": metric.get("hard_failure_improvement"),
        "easy_degradation": metric.get("easy_degradation"),
        "switch_rate": metric.get("switch_rate"),
    }


def _build_variants(cr: Mapping[str, Any], cq: Mapping[str, Any], dl: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = cr.get("ablation_rows", {})
    no_guard = rows.get("no_proximity_guard", {})
    proximity_guard = rows.get("proximity_guard", {})
    dl_replay = dl.get("real_batch_replay", {})
    dl_metric = dl_replay.get("metric", {})
    dl_diag = dl_replay.get("diagnostics", {})
    cq_bootstrap = cq.get("bootstrap_vs_endpoint_ade", {})
    return [
        {
            "variant": "endpoint_linear_reference",
            "role": "reference_floor",
            "deployment_status": "reference_only",
            "comparison_baseline": "endpoint_linear_bridge",
            "metrics": _metric_summary(rows.get("endpoint_linear_reference", {})),
            "safety": {"near_collision_005_delta_vs_endpoint": 0.0},
            "claim_use": "reference floor for bridge/shape composer; not a new M3W deployment claim",
        },
        {
            "variant": "no_proximity_guard",
            "role": "accuracy_priority_diagnostic",
            "deployment_status": "diagnostic_not_safety_sensitive",
            "comparison_baseline": "endpoint_linear_bridge",
            "metrics": _metric_summary(no_guard),
            "safety": {
                "near_collision_005_delta_vs_endpoint": no_guard.get("near_collision_005_delta_vs_endpoint"),
                "safety_caveat": "worsens near-collision@0.05 versus endpoint-linear",
            },
            "claim_use": "may be reported as an accuracy-priority diagnostic Pareto point, not as safety-sensitive deployment",
        },
        {
            "variant": "proximity_guard",
            "role": "safety_sensitive_deployable_bridge_shape_policy",
            "deployment_status": "deployable_when_joint_proximity_safety_is_required",
            "comparison_baseline": "endpoint_linear_bridge",
            "metrics": _metric_summary(proximity_guard),
            "safety": {
                "near_collision_005_delta_vs_endpoint": proximity_guard.get("near_collision_005_delta_vs_endpoint"),
                "all_ci_low": cq_bootstrap.get("all", {}).get("low"),
                "t50_ci_low": cq_bootstrap.get("t50", {}).get("low"),
            },
            "claim_use": "safety-sensitive protected bridge/shape composer variant",
        },
        {
            "variant": "group_consistency_full_waypoint_runtime",
            "role": "source_level_full_waypoint_group_consistency_runtime_policy",
            "deployment_status": "runtime_ready_for_its_source_level_protocol",
            "comparison_baseline": "train_horizon_causal_floor_not_endpoint_linear_bridge",
            "metrics": _metric_summary(dl_metric),
            "safety": {
                "base_near_005": dl_diag.get("base_near_005"),
                "final_near_005": dl_diag.get("final_near_005"),
                "floor_near_005": dl_diag.get("floor_near_005"),
                "selected_xy_max_abs_diff": dl_replay.get("selected_xy_max_abs_diff"),
                "switch_exact_match": dl_replay.get("switch_exact_match"),
            },
            "claim_use": "stronger full-waypoint runtime evidence, but its baseline/protocol differs from endpoint-linear composer; do not rank-mix without stating baseline",
        },
    ]


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _input_summary(name: str, path: Path, payload: Mapping[str, Any], gate_name: str) -> dict[str, Any]:
    gate = payload.get(gate_name, {})
    return {
        "name": name,
        "path": str(path),
        "sha256": _sha256(path),
        gate_name: {
            "passed": gate.get("passed"),
            "total": gate.get("total"),
            "verdict": gate.get("verdict"),
        },
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    variants = {row["variant"]: row for row in payload["deployment_variants"]}
    no_guard = variants["no_proximity_guard"]
    prox = variants["proximity_guard"]
    group = variants["group_consistency_full_waypoint_runtime"]
    claim = payload["claim_boundary"]
    gates = {
        "cr_ablation_passed": _passed(payload["inputs"]["proximity_guard_ablation"], "stage42_cr_gate"),
        "cq_guard_passed": _passed(payload["inputs"]["proximity_aware_guard"], "stage42_cq_gate"),
        "di_group_repair_passed": _passed(payload["inputs"]["group_consistency_repair"], "stage42_di_gate"),
        "dl_runtime_passed": _passed(payload["inputs"]["group_consistency_runtime"], "stage42_dl_gate"),
        "dm_reviewer_replay_passed": _passed(payload["inputs"]["reviewer_replay_package"], "stage42_dm_gate"),
        "no_guard_marked_diagnostic": no_guard["deployment_status"] == "diagnostic_not_safety_sensitive",
        "no_guard_proximity_caveat_visible": no_guard["safety"].get("near_collision_005_delta_vs_endpoint", 0.0) > 0.0,
        "proximity_guard_marked_safety_deployable": prox["deployment_status"]
        == "deployable_when_joint_proximity_safety_is_required",
        "proximity_guard_near_collision_not_worse": prox["safety"].get("near_collision_005_delta_vs_endpoint", 1.0) <= 0.0,
        "proximity_guard_ci_positive": prox["safety"].get("all_ci_low", 0.0) > 0.0
        and prox["safety"].get("t50_ci_low", 0.0) > 0.0,
        "group_runtime_marked_protocol_specific": "train_horizon_causal_floor" in group["comparison_baseline"],
        "group_runtime_exact_replay_visible": group["safety"].get("selected_xy_max_abs_diff") == 0.0
        and group["safety"].get("switch_exact_match") is True,
        "group_runtime_near_collision_reduced": group["safety"].get("final_near_005", 1.0)
        <= group["safety"].get("base_near_005", 0.0),
        "baseline_mixing_caveat_present": payload["baseline_mixing_caveat"] is True,
        "recommended_policy_declared": bool(payload["recommended_policy"]["safety_sensitive_default"]),
        "no_metric_seconds_overclaim": claim["metric_or_seconds_claim"] is False,
        "true_3d_overclaim_blocked": claim["true_3d"] is False,
        "foundation_overclaim_blocked": claim["foundation_world_model"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_dn_deployment_variant_card_pass" if passed == total else "stage42_dn_deployment_variant_card_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-DN Deployment Variant Card",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{payload['stage42_dn_gate']['passed']} / {payload['stage42_dn_gate']['total']}`",
        f"- verdict: `{payload['stage42_dn_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Deployment Variants",
        "",
        "| variant | role | status | comparison baseline | all | t50 | t100 raw | hard | easy | safety note |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["deployment_variants"]:
        metrics = row["metrics"]
        safety_note = row["safety"].get("safety_caveat") or (
            f"near@0.05 {row['safety'].get('near_collision_005_delta_vs_endpoint')}"
            if row["comparison_baseline"] == "endpoint_linear_bridge"
            else f"base near {row['safety'].get('base_near_005')} -> final {row['safety'].get('final_near_005')}"
        )
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | {} |".format(
                row["variant"],
                row["role"],
                row["deployment_status"],
                row["comparison_baseline"],
                metrics.get("all_improvement"),
                metrics.get("t50_improvement"),
                metrics.get("t100_raw_frame_diagnostic_improvement"),
                metrics.get("hard_failure_improvement"),
                metrics.get("easy_degradation"),
                safety_note,
            )
        )
    lines.extend(
        [
            "",
            "## Recommended Use",
            "",
            *[f"- {key}: `{value}`" for key, value in payload["recommended_policy"].items()],
            "",
            "## Claim Boundary",
            "",
            "- Do not present `no_proximity_guard` as safety-sensitive deployment.",
            "- Do not rank-mix `group_consistency_full_waypoint_runtime` directly against endpoint-linear composer variants without stating its different train-horizon causal-floor comparison baseline.",
            "- t+100 remains raw-frame diagnostic; dataset-local coordinates remain non-metric.",
            "- Stage5C remains unexecuted and SMC remains disabled.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{key}` | `{value}` |" for key, value in payload["stage42_dn_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_dn_gate"]
    return [
        "# Stage42-DN Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{key}` | `{value}` |" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_dn_gate"]
    return [
        "## Stage42-DN Deployment Variant Card",
        "",
        "- source: `fresh_deployment_variant_card_from_stage42_cr_cq_di_dl_dm`",
        "- role: separates safety-sensitive deployment, accuracy-priority diagnostics, and protocol-specific group-consistency runtime policy.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        "- safety-sensitive default: `proximity_guard` for endpoint-linear bridge/shape deployment with joint-proximity safety.",
        "- strongest full-waypoint runtime evidence: `group_consistency_full_waypoint_runtime`, but it uses train-horizon causal-floor comparison and must not be rank-mixed with endpoint-linear composer variants without that caveat.",
        "- accuracy-priority diagnostic: `no_proximity_guard`; it has higher ADE gains but worsens near-collision@0.05 and is not the safety-sensitive deployment claim.",
        "- No Stage5C, no SMC, no metric/seconds/true-3D/foundation claim.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, RETRO_README, GOAL_SUMMARY_README]:
        _replace_section(path, "STAGE42_DN_DEPLOYMENT_VARIANT_CARD", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-DN deployment variant card"
    state["current_verdict"] = payload["stage42_dn_gate"]["verdict"]
    state["stage42_dn_deployment_variant_card"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_dn_gate"]["verdict"],
        "gates": f"{payload['stage42_dn_gate']['passed']}/{payload['stage42_dn_gate']['total']}",
        "recommended_policy": payload["recommended_policy"],
        "baseline_mixing_caveat": payload["baseline_mixing_caveat"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_deployment_variant_card() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cr = read_json(CR_JSON, {})
    cq = read_json(CQ_JSON, {})
    di = read_json(DI_JSON, {})
    dl = read_json(DL_JSON, {})
    dm = read_json(DM_JSON, {})
    variants = _build_variants(cr, cq, dl)
    payload: dict[str, Any] = {
        "source": "fresh_deployment_variant_card_from_stage42_cr_cq_di_dl_dm",
        "stage": "Stage42-DN",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "proximity_guard_ablation": cr,
            "proximity_aware_guard": cq,
            "group_consistency_repair": di,
            "group_consistency_runtime": dl,
            "reviewer_replay_package": dm,
        },
        "deployment_variants": variants,
        "input_summaries": [
            _input_summary("proximity_guard_ablation", CR_JSON, cr, "stage42_cr_gate"),
            _input_summary("proximity_aware_guard", CQ_JSON, cq, "stage42_cq_gate"),
            _input_summary("group_consistency_repair", DI_JSON, di, "stage42_di_gate"),
            _input_summary("group_consistency_runtime", DL_JSON, dl, "stage42_dl_gate"),
            _input_summary("reviewer_replay_package", DM_JSON, dm, "stage42_dm_gate"),
        ],
        "recommended_policy": {
            "safety_sensitive_default": "proximity_guard",
            "accuracy_priority_diagnostic": "no_proximity_guard",
            "source_level_full_waypoint_runtime_candidate": "group_consistency_full_waypoint_runtime",
            "deployment_floor": "Stage37 / teacher floor remains required",
        },
        "baseline_mixing_caveat": True,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["inputs"] = {
        "proximity_guard_ablation": payload["input_summaries"][0],
        "proximity_aware_guard": payload["input_summaries"][1],
        "group_consistency_repair": payload["input_summaries"][2],
        "group_consistency_runtime": payload["input_summaries"][3],
        "reviewer_replay_package": payload["input_summaries"][4],
    }
    payload["stage42_dn_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_deployment_variant_card()
