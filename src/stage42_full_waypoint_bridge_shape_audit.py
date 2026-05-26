from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
FULL_WAYPOINT_JSON = OUT_DIR / "full_waypoint_dynamics_stage42.json"
UNIFIED_ROW_JSON = OUT_DIR / "unified_row_level_full_waypoint_cache_stage42.json"
UCY_BRIDGE_JSON = OUT_DIR / "ucy_candidate_bridge_stage42.json"
GRAPH_CLAIM_JSON = OUT_DIR / "neighbor_interaction_gated_expert_stage42.json"

REPORT_JSON = OUT_DIR / "full_waypoint_bridge_shape_audit_stage42.json"
REPORT_MD = OUT_DIR / "full_waypoint_bridge_shape_audit_stage42.md"
REPORT_CSV = OUT_DIR / "full_waypoint_bridge_shape_audit_stage42.csv"
GATE_MD = OUT_DIR / "stage42_stage_cm_gate.md"

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
    "Stage42-CM 是 endpoint/bridge/full-waypoint shape audit，不重新训练，不调 threshold，不执行 Stage5C，不启用 SMC。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "endpoint-only 成功不能自动算 full-waypoint world-state dynamics 成功。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _pct(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


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


def _metric(comp: Mapping[str, Any], metric: str = "ade") -> Mapping[str, Any]:
    return comp.get(metric, {})


def _row(name: str, source: str, status: str, metric: Mapping[str, Any], note: str) -> dict[str, Any]:
    return {
        "name": name,
        "source": source,
        "status": status,
        "rows": metric.get("rows"),
        "all_improvement": metric.get("all_improvement"),
        "t50_improvement": metric.get("t50_improvement"),
        "t100_raw_frame_diagnostic_improvement": metric.get("t100_raw_frame_diagnostic_improvement", metric.get("t100_improvement")),
        "hard_failure_improvement": metric.get("hard_failure_improvement"),
        "easy_degradation": metric.get("easy_degradation"),
        "switch_rate": metric.get("switch_rate"),
        "note": note,
    }


def _comparison_rows(
    full: Mapping[str, Any],
    unified: Mapping[str, Any],
    ucy_bridge: Mapping[str, Any],
) -> list[dict[str, Any]]:
    comps = full["comparisons"]
    rows = [
        _row(
            "endpoint_only_final_fde",
            comps["endpoint_only_final_fde"]["source"],
            "diagnostic_only_not_full_waypoint",
            _metric(comps["endpoint_only_final_fde"], "fde"),
            "Endpoint-only final FDE is not a full-waypoint world-state model.",
        ),
        _row(
            "m3w_neural_v1_composite_tail_linear_bridge",
            comps["m3w_neural_v1_composite_tail_linear_bridge"]["source"],
            "deployable_endpoint_linear_bridge_floor",
            _metric(comps["m3w_neural_v1_composite_tail_linear_bridge"], "ade"),
            "Current protected endpoint dynamics projected through a linear bridge.",
        ),
        _row(
            "full_waypoint_transformer_protected",
            comps["full_waypoint_transformer_protected"]["source"],
            "protected_full_waypoint_positive_two_domains",
            _metric(comps["full_waypoint_transformer_protected"], "ade"),
            "Actual full-waypoint sequence model under protected switch policy.",
        ),
        _row(
            "ungated_full_waypoint_transformer",
            comps["ungated_full_waypoint_transformer"]["source"],
            "diagnostic_unsafe_not_deployable",
            _metric(comps["ungated_full_waypoint_transformer"], "ade"),
            "Ungated neural/full-waypoint output has unsafe easy degradation.",
        ),
    ]
    graph_metric = comps["graph_interaction_group_consistency"]["report"]["test_metrics"]
    rows.append(
        _row(
            "graph_interaction_group_consistency",
            comps["graph_interaction_group_consistency"]["source"],
            "protected_positive_with_proximity_caveat",
            graph_metric,
            f"Protected graph/group policy positive, but collision_delta_vs_floor_005={graph_metric.get('collision_delta_vs_floor_005')}; CK blocks graph as independent main claim.",
        )
    )
    unified_summary = unified["summary"]
    unified_rows_obj = unified.get("rows", {})
    if isinstance(unified_rows_obj, Mapping):
        unified_row_count = unified_rows_obj.get("total")
    elif isinstance(unified_rows_obj, list):
        unified_row_count = sum(
            int(row.get("merged_test_metrics", {}).get("ade", {}).get("rows", 0))
            for row in unified_rows_obj
        )
    else:
        unified_row_count = None
    rows.append(
        {
            "name": "unified_row_level_full_waypoint_cache",
            "source": unified["source"],
            "status": "row_level_full_waypoint_three_domain_positive",
            "rows": unified_row_count,
            "all_improvement": unified_summary["ade_all"]["mean"],
            "t50_improvement": unified_summary["ade_t50"]["mean"],
            "t100_raw_frame_diagnostic_improvement": unified_summary["ade_t100_raw_frame_diagnostic"]["mean"],
            "hard_failure_improvement": unified_summary["ade_hard_failure"]["mean"],
            "easy_degradation": unified_summary["ade_easy_degradation"]["mean"],
            "switch_rate": unified_summary["switch_rate"]["mean"],
            "note": "Unified row-level cache merges verified external full-waypoint policy sources across ETH_UCY, TrajNet, and UCY.",
        }
    )
    rows.append(
        {
            "name": "ucy_endpoint_to_full_linear_bridge",
            "source": ucy_bridge["source"],
            "status": "failed_blocker",
            "rows": None,
            "all_improvement": None,
            "t50_improvement": None,
            "t100_raw_frame_diagnostic_improvement": None,
            "hard_failure_improvement": None,
            "easy_degradation": None,
            "switch_rate": None,
            "note": ucy_bridge["interpretation"]["root_cause"],
        }
    )
    return rows


def _deltas(full: Mapping[str, Any]) -> dict[str, Any]:
    comps = full["comparisons"]
    linear = _metric(comps["m3w_neural_v1_composite_tail_linear_bridge"], "ade")
    waypoint = _metric(comps["full_waypoint_transformer_protected"], "ade")
    graph = comps["graph_interaction_group_consistency"]["report"]["test_metrics"]
    linear_t100 = linear.get("t100_raw_frame_diagnostic_improvement", linear.get("t100_improvement", 0.0))
    waypoint_t100 = waypoint.get("t100_raw_frame_diagnostic_improvement", waypoint.get("t100_improvement", 0.0))
    graph_t100 = graph.get("t100_raw_frame_diagnostic_improvement", graph.get("t100_improvement", 0.0))
    return {
        "full_waypoint_minus_linear_bridge": {
            "all_improvement": waypoint["all_improvement"] - linear["all_improvement"],
            "t50_improvement": waypoint["t50_improvement"] - linear["t50_improvement"],
            "t100_raw_frame_diagnostic_improvement": waypoint_t100 - linear_t100,
            "hard_failure_improvement": waypoint["hard_failure_improvement"] - linear["hard_failure_improvement"],
        },
        "graph_group_minus_full_waypoint": {
            "all_improvement": graph["all_improvement"] - waypoint["all_improvement"],
            "t50_improvement": graph["t50_improvement"] - waypoint["t50_improvement"],
            "t100_raw_frame_diagnostic_improvement": graph_t100 - waypoint_t100,
            "hard_failure_improvement": graph["hard_failure_improvement"] - waypoint["hard_failure_improvement"],
            "collision_delta_vs_floor_005": graph.get("collision_delta_vs_floor_005"),
        },
    }


def _paper_lines(rows: list[dict[str, Any]], deltas: Mapping[str, Any]) -> list[str]:
    fw_delta = deltas["full_waypoint_minus_linear_bridge"]
    graph_delta = deltas["graph_group_minus_full_waypoint"]
    return [
        "## Stage42-CM Endpoint Bridge / Full-Waypoint Shape Audit",
        "",
        "- source: `fresh_synthesis_from_stage42_full_waypoint_artifacts`",
        "- scope: protected dataset-local raw-frame 2.5D full-waypoint evidence boundary.",
        "- Endpoint-only FDE is diagnostic; endpoint success cannot be counted as full-waypoint world-state success.",
        "- Protected full-waypoint sequence dynamics has horizon/full-waypoint evidence, but the endpoint linear bridge remains stronger on all-ADE.",
        "- Graph/group consistency has positive protected metrics but still carries a proximity caveat and is not an independent neighbor/interaction main claim after Stage42-CK.",
        "- Stage5C remains unexecuted and SMC remains disabled.",
        "",
        "### Key Deltas",
        "",
        f"- protected full-waypoint minus composite-tail linear bridge: all `{_pct(fw_delta['all_improvement'])}`, t50 `{_pct(fw_delta['t50_improvement'])}`, t100 raw diagnostic `{_pct(fw_delta['t100_raw_frame_diagnostic_improvement'])}`, hard `{_pct(fw_delta['hard_failure_improvement'])}`.",
        f"- graph/group minus protected full-waypoint: all `{_pct(graph_delta['all_improvement'])}`, t50 `{_pct(graph_delta['t50_improvement'])}`, hard `{_pct(graph_delta['hard_failure_improvement'])}`, collision_delta_005 `{graph_delta['collision_delta_vs_floor_005']}`.",
        "",
        "### Evidence Rows",
        "",
        "| variant | status | all | t50 | t100 diag | hard | easy | note |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        *[
            f"| `{row['name']}` | `{row['status']}` | {_fmt_metric(row['all_improvement'])} | {_fmt_metric(row['t50_improvement'])} | {_fmt_metric(row['t100_raw_frame_diagnostic_improvement'])} | {_fmt_metric(row['hard_failure_improvement'])} | {_fmt_metric(row['easy_degradation'])} | {row['note']} |"
            for row in rows
        ],
        "",
        "### Claim Boundary",
        "",
        "- Supported: protected full-waypoint raw-frame evidence exists, especially for horizon/full-waypoint slices and unified row-level three-domain package.",
        "- Supported with caveat: graph/group consistency can be useful inside protected policies, but current source-level kNN graph expert did not become a main contribution.",
        "- Rejected: endpoint-only success as full-waypoint success; ungated full-waypoint neural deployment; true 3D; foundation; metric/seconds-level; Stage5C; SMC.",
    ]


def _fmt_metric(value: Any) -> str:
    return "n/a" if value is None else _pct(value)


def _refresh_paper_files(rows: list[dict[str, Any]], deltas: Mapping[str, Any]) -> list[dict[str, Any]]:
    status = []
    lines = _paper_lines(rows, deltas)
    for path in PAPER_FILES:
        _replace_section(path, "STAGE42_CM_FULL_WAYPOINT_BRIDGE_SHAPE_AUDIT", lines)
        text = path.read_text(encoding="utf-8")
        status.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "contains_stage42_cm": "Stage42-CM Endpoint Bridge / Full-Waypoint Shape Audit" in text,
                "blocks_endpoint_as_full_waypoint": "Endpoint-only FDE is diagnostic" in text,
                "blocks_ungated_full_waypoint": "ungated full-waypoint neural deployment" in text,
                "contains_no_metric_boundary": "metric/seconds-level" in text,
            }
        )
    return status


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    full = payload["inputs"]["full_waypoint"]
    unified = payload["inputs"]["unified_row_cache"]
    ucy = payload["inputs"]["ucy_bridge"]
    comps = full["comparisons"]
    full_metric = comps["full_waypoint_transformer_protected"]["ade"]
    linear_metric = comps["m3w_neural_v1_composite_tail_linear_bridge"]["ade"]
    ungated_metric = comps["ungated_full_waypoint_transformer"]["ade"]
    graph_metric = comps["graph_interaction_group_consistency"]["report"]["test_metrics"]
    full_t100 = full_metric.get("t100_raw_frame_diagnostic_improvement", full_metric.get("t100_improvement", 0.0))
    linear_t100 = linear_metric.get(
        "t100_raw_frame_diagnostic_improvement", linear_metric.get("t100_improvement", 0.0)
    )
    gates = {
        "stage42_c_gate_passed": full["stage42_c_gate"]["passed"] == full["stage42_c_gate"]["total"],
        "stage42_x_gate_passed": unified["stage42_x_gate"]["passed"] == unified["stage42_x_gate"]["total"],
        "endpoint_only_marked_diagnostic": comps["endpoint_only_final_fde"]["claim_boundary"]["full_waypoint_model"] is False,
        "protected_full_waypoint_two_domains_positive": bool(full["stage42_c_gate"]["gates"]["two_external_domains_positive"]),
        "full_waypoint_has_horizon_lift_over_linear_bridge": full_metric["t50_improvement"] > linear_metric["t50_improvement"]
        and full_t100 > linear_t100,
        "linear_bridge_all_advantage_recorded": linear_metric["all_improvement"] > full_metric["all_improvement"],
        "ucy_endpoint_bridge_blocker_recorded": ucy["verdict"] == "stage42_u_ucy_endpoint_to_full_bridge_failed_blocker",
        "graph_positive_with_proximity_caveat_recorded": graph_metric["all_improvement"] > 0.0
        and graph_metric.get("collision_delta_vs_floor_005", 0.0) > 0.0,
        "ungated_full_waypoint_unsafe_blocked": ungated_metric["easy_degradation"] > 0.02,
        "paper_files_refreshed": all(row["contains_stage42_cm"] for row in payload["paper_file_status"]),
        "no_leakage_pass": full["no_leakage"]["future_endpoint_input"] is False
        and full["no_leakage"]["future_waypoints_input"] is False
        and full["no_leakage"]["test_threshold_tuning"] is False,
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_cm_full_waypoint_bridge_shape_audit_pass" if passed == total else "stage42_cm_full_waypoint_bridge_shape_audit_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-CM Full-Waypoint Bridge / Shape Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_cm_gate']['passed']} / {payload['stage42_cm_gate']['total']}`",
        f"- verdict: `{payload['stage42_cm_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Key Deltas",
        "",
    ]
    for group, vals in payload["deltas"].items():
        lines.append(f"### {group}")
        for key, val in vals.items():
            lines.append(f"- {key}: `{_fmt_metric(val) if isinstance(val, (int, float)) else val}`")
        lines.append("")
    lines += [
        "## Comparison Rows",
        "",
        "| variant | source | status | rows | all | t50 | t100 diag | hard | easy | switch | note |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["comparison_rows"]:
        lines.append(
            f"| `{row['name']}` | `{row['source']}` | `{row['status']}` | {row['rows'] if row['rows'] is not None else 'n/a'} | {_fmt_metric(row['all_improvement'])} | {_fmt_metric(row['t50_improvement'])} | {_fmt_metric(row['t100_raw_frame_diagnostic_improvement'])} | {_fmt_metric(row['hard_failure_improvement'])} | {_fmt_metric(row['easy_degradation'])} | {_fmt_metric(row['switch_rate'])} | {row['note']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Full-waypoint evidence exists and is strongest on horizon/full-waypoint slices, but endpoint-linear bridge remains stronger on all-ADE.",
        "- Endpoint-only or endpoint-to-linear evidence must not be counted as learned full-waypoint shape by itself.",
        "- Ungated full-waypoint neural remains unsafe; deployment still requires protected switch/floor.",
        "- Graph/group interaction can help protected policy metrics, but CK blocks it as a standalone source-level graph main claim.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_cm_gate"]
    lines = [
        "# Stage42-CM Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | {bool(value)} |")
    return lines


def run_stage42_full_waypoint_bridge_shape_audit() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    full = read_json(FULL_WAYPOINT_JSON, {})
    unified = read_json(UNIFIED_ROW_JSON, {})
    ucy_bridge = read_json(UCY_BRIDGE_JSON, {})
    graph_claim = read_json(GRAPH_CLAIM_JSON, {})
    rows = _comparison_rows(full, unified, ucy_bridge)
    deltas = _deltas(full)
    paper_file_status = _refresh_paper_files(rows, deltas)
    payload: dict[str, Any] = {
        "source": "fresh_synthesis_from_stage42_full_waypoint_artifacts",
        "stage": "Stage42-CM endpoint bridge / full-waypoint shape audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([FULL_WAYPOINT_JSON, UNIFIED_ROW_JSON, UCY_BRIDGE_JSON, GRAPH_CLAIM_JSON]),
        "inputs": {
            "full_waypoint": full,
            "unified_row_cache": unified,
            "ucy_bridge": ucy_bridge,
            "graph_claim_boundary": {
                "verdict": graph_claim.get("stage42_ck_gate", {}).get("verdict"),
                "neighbor_interaction_rescue_success": graph_claim.get("neighbor_interaction_rescue_success"),
            },
        },
        "comparison_rows": rows,
        "deltas": deltas,
        "paper_file_status": paper_file_status,
        "claim_boundary": {
            "endpoint_only_as_full_waypoint_claim_allowed": False,
            "ungated_full_waypoint_deployable": False,
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_cm_gate"] = _gate(payload)

    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "name",
                "source",
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
        writer.writerows(rows)
    return payload


if __name__ == "__main__":
    run_stage42_full_waypoint_bridge_shape_audit()
