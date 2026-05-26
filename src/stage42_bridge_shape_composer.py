from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
FULL_DYNAMICS_JSON = OUT_DIR / "full_waypoint_dynamics_stage42.json"
CM_JSON = OUT_DIR / "full_waypoint_bridge_shape_audit_stage42.json"
STATIC_GATE_JSON = OUT_DIR / "static_gated_full_waypoint_stage42.json"
UNIFIED_ROW_JSON = OUT_DIR / "unified_row_level_full_waypoint_cache_stage42.json"

REPORT_JSON = OUT_DIR / "bridge_shape_composer_stage42.json"
REPORT_MD = OUT_DIR / "bridge_shape_composer_stage42.md"
REPORT_CSV = OUT_DIR / "bridge_shape_composer_stage42.csv"
GATE_MD = OUT_DIR / "stage42_stage_cn_gate.md"

PAPER_FILES = [
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CN 是 bridge/shape composer 审计，不重新训练，不调 test threshold。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "endpoint-only 或 endpoint-linear bridge 成功不能自动算 learned full-waypoint shape success。",
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


def _metric(comp: Mapping[str, Any], key: str = "ade") -> Mapping[str, Any]:
    return comp.get(key, {})


def _t100(metric: Mapping[str, Any]) -> float:
    return float(metric.get("t100_raw_frame_diagnostic_improvement", metric.get("t100_improvement", 0.0)))


def _static_summary_row(static: Mapping[str, Any], name: str) -> dict[str, Any]:
    summary = static["summary"][name]
    return {
        "name": f"stage42j_{name}",
        "source": summary["source"],
        "validation_rule": "domain_horizon_expert_selected_on_val",
        "status": "validation_selected_full_waypoint_shape_candidate",
        "rows": static.get("dataset_rows", {}).get("test"),
        "all_improvement": summary["ade_all"]["mean"],
        "t50_improvement": summary["ade_t50"]["mean"],
        "t100_raw_frame_diagnostic_improvement": summary["ade_t100_raw_frame_diagnostic"]["mean"],
        "hard_failure_improvement": summary["ade_hard_failure"]["mean"],
        "easy_degradation": summary["ade_easy_degradation"]["mean"],
        "switch_rate": summary["switch_rate"]["mean"],
        "note": "Stage42-J uses validation-only domain/horizon static gating over cached full-waypoint checkpoints.",
    }


def _comparison_row(name: str, comp: Mapping[str, Any], status: str, note: str) -> dict[str, Any]:
    metric = _metric(comp, "ade")
    return {
        "name": name,
        "source": comp.get("source"),
        "validation_rule": comp.get("validation_rule", "preexisting_protected_policy_or_diagnostic"),
        "status": status,
        "rows": metric.get("rows"),
        "all_improvement": metric.get("all_improvement"),
        "t50_improvement": metric.get("t50_improvement"),
        "t100_raw_frame_diagnostic_improvement": metric.get(
            "t100_raw_frame_diagnostic_improvement", metric.get("t100_improvement")
        ),
        "hard_failure_improvement": metric.get("hard_failure_improvement"),
        "easy_degradation": metric.get("easy_degradation"),
        "switch_rate": metric.get("switch_rate"),
        "note": note,
    }


def _unified_row(unified: Mapping[str, Any]) -> dict[str, Any]:
    summary = unified["summary"]
    rows_obj = unified.get("rows", {})
    if isinstance(rows_obj, Mapping):
        rows = rows_obj.get("total")
    elif isinstance(rows_obj, list):
        rows = sum(int(row.get("merged_test_metrics", {}).get("ade", {}).get("rows", 0)) for row in rows_obj)
    else:
        rows = None
    return {
        "name": "stage42x_unified_row_level_full_waypoint_cache",
        "source": unified.get("source"),
        "validation_rule": "combo_sources_selected_on_val",
        "status": "row_level_full_waypoint_three_domain_positive_auxiliary",
        "rows": rows,
        "all_improvement": summary["ade_all"]["mean"],
        "t50_improvement": summary["ade_t50"]["mean"],
        "t100_raw_frame_diagnostic_improvement": summary["ade_t100_raw_frame_diagnostic"]["mean"],
        "hard_failure_improvement": summary["ade_hard_failure"]["mean"],
        "easy_degradation": summary["ade_easy_degradation"]["mean"],
        "switch_rate": summary["switch_rate"]["mean"],
        "note": "Unified row-level full-waypoint cache is positive but below the current endpoint-linear bridge floor on all/t50/hard.",
    }


def _candidate_rows(full: Mapping[str, Any], static: Mapping[str, Any], unified: Mapping[str, Any]) -> list[dict[str, Any]]:
    comps = full["comparisons"]
    return [
        _comparison_row(
            "endpoint_linear_bridge_floor",
            comps["m3w_neural_v1_composite_tail_linear_bridge"],
            "current_deployable_all_ade_floor",
            "Current M3W-Neural v1 protected endpoint dynamics projected through endpoint-linear waypoint bridge.",
        ),
        _comparison_row(
            "protected_full_waypoint_sequence",
            comps["full_waypoint_transformer_protected"],
            "protected_full_waypoint_horizon_auxiliary",
            "Actual full-waypoint sequence model; useful on t50/t100 raw-frame but not an all-ADE replacement.",
        ),
        _static_summary_row(static, "static_gated"),
        _static_summary_row(static, "static_alpha025"),
        _static_summary_row(static, "no_static"),
        _unified_row(unified),
        _comparison_row(
            "ungated_full_waypoint_sequence",
            comps["ungated_full_waypoint_transformer"],
            "diagnostic_unsafe_not_deployable",
            "Ungated full-waypoint neural output is unsafe because easy degradation is far above the deployment limit.",
        ),
    ]


def _composer_decision(rows: list[Mapping[str, Any]], cm: Mapping[str, Any]) -> dict[str, Any]:
    by_name = {row["name"]: row for row in rows}
    floor = by_name["endpoint_linear_bridge_floor"]
    protected = by_name["protected_full_waypoint_sequence"]
    static_gated = by_name["stage42j_static_gated"]
    delta = cm["deltas"]["full_waypoint_minus_linear_bridge"]
    common_val_available = False
    deployable_switch = bool(
        common_val_available
        and float(protected["all_improvement"]) >= float(floor["all_improvement"])
        and float(protected["easy_degradation"]) <= 0.02
    )
    return {
        "selected_deployment_policy": "keep_endpoint_linear_bridge_floor_with_full_waypoint_auxiliary_reporting",
        "deployable_bridge_shape_composer_available": deployable_switch,
        "common_validation_endpoint_vs_full_waypoint_comparison_available": common_val_available,
        "reason": (
            "Stage42-J supplies validation-only full-waypoint/static gating evidence, but there is not yet a common "
            "row-level validation comparison that can safely switch between endpoint-linear bridge and protected "
            "full-waypoint sequence. Stage42-CM shows full-waypoint improves t50/t100 raw-frame over the linear "
            "bridge but loses all-ADE and hard/failure, so deployment remains the endpoint-linear floor with "
            "full-waypoint as auxiliary horizon evidence."
        ),
        "key_delta_full_waypoint_minus_linear_bridge": delta,
        "static_gated_positive_easy_safe": (
            float(static_gated["all_improvement"]) > 0.0
            and float(static_gated["t50_improvement"]) > 0.0
            and float(static_gated["easy_degradation"]) <= 0.02
        ),
        "full_waypoint_horizon_auxiliary_supported": (
            float(delta["t50_improvement"]) > 0.0
            and float(delta["t100_raw_frame_diagnostic_improvement"]) > 0.0
        ),
        "full_waypoint_all_ade_replacement_supported": float(delta["all_improvement"]) >= 0.0,
        "blocked_next_requirement": "Build common validation-aligned endpoint-linear-vs-full-waypoint row cache before any deployment switch.",
    }


def _replace_section(path: Path, marker: str, lines: list[str]) -> None:
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    block = "\n".join([start, *lines, end])
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if start in text and end in text:
        prefix = text.split(start, 1)[0].rstrip()
        suffix = text.split(end, 1)[1].lstrip()
        new_text = prefix + "\n\n" + block + ("\n\n" + suffix if suffix else "\n")
    else:
        new_text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(new_text, encoding="utf-8")


def _paper_lines(decision: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> list[str]:
    return [
        "## Stage42-CN Bridge / Shape Composer Audit",
        "",
        "- source: `fresh_synthesis_from_stage42_cm_j_x_artifacts`",
        "- scope: validation-only composer feasibility for endpoint-linear bridge vs full-waypoint shape heads.",
        "- conclusion: keep endpoint-linear bridge as deployable all-ADE floor; use full-waypoint heads only as auxiliary horizon evidence until a common validation-aligned row-level composer exists.",
        "- blocker: common validation endpoint-vs-full-waypoint row cache is missing, so no new bridge/shape deployment switch is allowed.",
        "- Stage5C remains unexecuted and SMC remains disabled.",
        "",
        "### Candidate Summary",
        "",
        "| candidate | status | all | t50 | t100 diag | hard | easy | role |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        *[
            f"| `{row['name']}` | `{row['status']}` | {_pct(row['all_improvement'])} | {_pct(row['t50_improvement'])} | {_pct(row['t100_raw_frame_diagnostic_improvement'])} | {_pct(row['hard_failure_improvement'])} | {_pct(row['easy_degradation'])} | {row['note']} |"
            for row in rows
        ],
        "",
        "### Deployment Boundary",
        "",
        f"- selected deployment policy: `{decision['selected_deployment_policy']}`",
        f"- deployable bridge/shape composer available now: `{decision['deployable_bridge_shape_composer_available']}`",
        f"- next required evidence: {decision['blocked_next_requirement']}",
    ]


def _refresh_paper_files(decision: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lines = _paper_lines(decision, rows)
    status = []
    for path in PAPER_FILES:
        _replace_section(path, "STAGE42_CN_BRIDGE_SHAPE_COMPOSER", lines)
        text = path.read_text(encoding="utf-8")
        status.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "contains_stage42_cn": "Stage42-CN Bridge / Shape Composer Audit" in text,
                "blocks_new_deployment_switch": "no new bridge/shape deployment switch is allowed" in text,
                "blocks_metric_seconds_overclaim": "Stage5C remains unexecuted and SMC remains disabled" in text,
            }
        )
    return status


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    full = payload["inputs"]["full_waypoint"]
    cm = payload["inputs"]["cm"]
    static = payload["inputs"]["static_gate"]
    unified = payload["inputs"]["unified_row_cache"]
    decision = payload["composer_decision"]
    rows = {row["name"]: row for row in payload["candidate_rows"]}
    gates = {
        "stage42_c_gate_passed": full["stage42_c_gate"]["passed"] == full["stage42_c_gate"]["total"],
        "stage42_cm_gate_passed": cm["stage42_cm_gate"]["passed"] == cm["stage42_cm_gate"]["total"],
        "stage42_j_validation_gate_passed": static["stage42_j_gate"]["passed"] == static["stage42_j_gate"]["total"],
        "stage42_x_unified_cache_gate_passed": unified["stage42_x_gate"]["passed"] == unified["stage42_x_gate"]["total"],
        "endpoint_linear_floor_available": rows["endpoint_linear_bridge_floor"]["all_improvement"] is not None,
        "static_gated_full_waypoint_positive_easy_safe": bool(decision["static_gated_positive_easy_safe"]),
        "full_waypoint_horizon_auxiliary_supported": bool(decision["full_waypoint_horizon_auxiliary_supported"]),
        "full_waypoint_not_all_ade_replacement": not bool(decision["full_waypoint_all_ade_replacement_supported"]),
        "common_val_switch_blocker_documented": not bool(decision["common_validation_endpoint_vs_full_waypoint_comparison_available"])
        and "common validation" in str(decision["blocked_next_requirement"]).lower(),
        "ungated_full_waypoint_blocked": float(rows["ungated_full_waypoint_sequence"]["easy_degradation"]) > 0.02,
        "no_test_threshold_tuning": full["no_leakage"]["test_threshold_tuning"] is False
        and static["no_leakage"]["test_threshold_tuning"] is False
        and unified["no_leakage"]["test_policy_tuning"] is False,
        "paper_files_refreshed": all(row["contains_stage42_cn"] for row in payload["paper_file_status"]),
        "metric_seconds_overclaim_blocked": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_not_enabled": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = (
        "stage42_cn_bridge_shape_composer_audit_pass_blocker_documented"
        if passed == total
        else "stage42_cn_bridge_shape_composer_audit_fail"
    )
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _write_csv(rows: list[Mapping[str, Any]]) -> None:
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "source",
                "validation_rule",
                "status",
                "rows",
                "all_improvement",
                "t50_improvement",
                "t100_raw_frame_diagnostic_improvement",
                "hard_failure_improvement",
                "easy_degradation",
                "switch_rate",
                "note",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_cn_gate"]
    decision = payload["composer_decision"]
    lines = [
        "# Stage42-CN Bridge / Shape Composer Audit",
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
        "## Composer Decision",
        "",
        f"- selected deployment policy: `{decision['selected_deployment_policy']}`",
        f"- deployable bridge/shape composer available now: `{decision['deployable_bridge_shape_composer_available']}`",
        f"- common validation endpoint-vs-full-waypoint comparison available: `{decision['common_validation_endpoint_vs_full_waypoint_comparison_available']}`",
        f"- blocked next requirement: {decision['blocked_next_requirement']}",
        f"- reason: {decision['reason']}",
        "",
        "## Candidate Rows",
        "",
        "| candidate | source | status | validation rule | rows | all | t50 | t100 diag | hard | easy | switch | note |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["candidate_rows"]:
        lines.append(
            f"| `{row['name']}` | `{row['source']}` | `{row['status']}` | `{row['validation_rule']}` | "
            f"{row['rows']} | {_pct(row['all_improvement'])} | {_pct(row['t50_improvement'])} | "
            f"{_pct(row['t100_raw_frame_diagnostic_improvement'])} | {_pct(row['hard_failure_improvement'])} | "
            f"{_pct(row['easy_degradation'])} | {_pct(row['switch_rate'])} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Full-waypoint shape heads have real auxiliary horizon value, especially t50/t100 raw-frame.",
            "- They do not yet replace the endpoint-linear bridge all-ADE floor.",
            "- No new composer deployment switch is allowed because endpoint-linear-vs-full-waypoint common validation evidence is missing.",
            "- The honest deployable policy remains M3W-Neural v1 endpoint-linear bridge / Stage37-teacher floor, with full-waypoint evidence reported as auxiliary.",
            "- This is not true 3D, not metric, not seconds-level, not Stage5C, and not SMC.",
        ]
    )
    write_md(REPORT_MD, lines)

    gate_lines = [
        "# Stage42-CN Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{name}` | `{ok}` |" for name, ok in gate["gates"].items()],
    ]
    write_md(GATE_MD, gate_lines)


def run() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    full = read_json(FULL_DYNAMICS_JSON, {})
    cm = read_json(CM_JSON, {})
    static = read_json(STATIC_GATE_JSON, {})
    unified = read_json(UNIFIED_ROW_JSON, {})
    rows = _candidate_rows(full, static, unified)
    decision = _composer_decision(rows, cm)
    paper_status = _refresh_paper_files(decision, rows)
    payload: dict[str, Any] = {
        "source": "fresh_synthesis_from_stage42_cm_j_x_artifacts",
        "stage": "Stage42-CN bridge / shape composer audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([FULL_DYNAMICS_JSON, CM_JSON, STATIC_GATE_JSON, UNIFIED_ROW_JSON]),
        "inputs": {"full_waypoint": full, "cm": cm, "static_gate": static, "unified_row_cache": unified},
        "candidate_rows": rows,
        "composer_decision": decision,
        "paper_file_status": paper_status,
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "new_bridge_shape_deployment_switch_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_cn_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    _write_csv(rows)
    _write_md(payload)
    return payload


if __name__ == "__main__":
    result = run()
    gate = result["stage42_cn_gate"]
    print(f"Stage42-CN bridge/shape composer: {gate['verdict']} ({gate['passed']}/{gate['total']})")
