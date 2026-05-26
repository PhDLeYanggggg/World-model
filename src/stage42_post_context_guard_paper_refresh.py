from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
CJ_JSON = OUT_DIR / "goal_scene_gated_expert_stage42.json"
CK_JSON = OUT_DIR / "neighbor_interaction_gated_expert_stage42.json"

REPORT_JSON = OUT_DIR / "paper_package_post_context_guard_refresh_stage42.json"
REPORT_MD = OUT_DIR / "paper_package_post_context_guard_refresh_stage42.md"
REPORT_CSV = OUT_DIR / "paper_package_post_context_guard_refresh_stage42.csv"
GATE_MD = OUT_DIR / "stage42_stage_cl_gate.md"

PAPER_FILES = [
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "failure_taxonomy_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CL 是 paper package refresh，不重新训练，不调 threshold，不执行 Stage5C，不启用 SMC。",
    "Stage42-CJ/CK 是 validation-only gated expert audits；test 只最终评估，不用于选择。",
    "goal/scene 与 neighbor/interaction 仍是 diagnostic / auxiliary evidence，不是独立主贡献。",
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


def _metric(payload: Mapping[str, Any], variant: str, key: str) -> float:
    return float(payload["variants"][variant]["protected"][key])


def _variant_summary(payload: Mapping[str, Any], variant: str) -> dict[str, Any]:
    protected = payload["variants"][variant]["protected"]
    return {
        "variant": variant,
        "all_improvement": protected["all_improvement"],
        "t50_improvement": protected["t50_improvement"],
        "t100_raw_frame_diagnostic_improvement": protected["t100_raw_frame_diagnostic_improvement"],
        "hard_failure_improvement": protected["hard_failure_improvement"],
        "easy_degradation": protected["easy_degradation"],
        "switch_rate": protected["switch_rate"],
    }


def _evidence_rows(cj: Mapping[str, Any], ck: Mapping[str, Any]) -> list[dict[str, str]]:
    cj_base = _variant_summary(cj, "baseline_family_control")
    cj_goal = _variant_summary(cj, "baseline_plus_goal_scene")
    cj_motion_goal = _variant_summary(cj, "baseline_plus_motion_goal_context")
    ck_base = _variant_summary(ck, "baseline_family_control")
    ck_scalar = _variant_summary(ck, "baseline_plus_scalar_neighbor")
    ck_graph = _variant_summary(ck, "baseline_plus_knn_graph")
    ck_graph_goal = _variant_summary(ck, "baseline_plus_graph_goal")
    graph_stats = ck["graph_info"]["graph_stats"]

    return [
        {
            "item": "Stage42-CJ goal/scene gated expert",
            "status": "diagnostic_negative",
            "paper_use": "claim boundary / limitation",
            "evidence": (
                f"gate={cj['stage42_cj_gate']['passed']}/{cj['stage42_cj_gate']['total']}; "
                f"selected={cj['validation_only_selection']['selected_variant']}; "
                f"goal_scene_rescue_success={cj['goal_scene_rescue_success']}; "
                f"control all/t50/hard={_pct(cj_base['all_improvement'])}/{_pct(cj_base['t50_improvement'])}/{_pct(cj_base['hard_failure_improvement'])}; "
                f"goal all/t50/hard={_pct(cj_goal['all_improvement'])}/{_pct(cj_goal['t50_improvement'])}/{_pct(cj_goal['hard_failure_improvement'])}"
            ),
        },
        {
            "item": "Stage42-CJ motion+goal context",
            "status": "diagnostic_negative",
            "paper_use": "ablation boundary",
            "evidence": (
                f"motion_goal all/t50/hard={_pct(cj_motion_goal['all_improvement'])}/{_pct(cj_motion_goal['t50_improvement'])}/{_pct(cj_motion_goal['hard_failure_improvement'])}; "
                f"delta_t50_vs_control={_pct(cj_motion_goal['t50_improvement'] - cj_base['t50_improvement'])}"
            ),
        },
        {
            "item": "Stage42-CK neighbor/interaction gated expert",
            "status": "diagnostic_negative",
            "paper_use": "claim boundary / limitation",
            "evidence": (
                f"gate={ck['stage42_ck_gate']['passed']}/{ck['stage42_ck_gate']['total']}; "
                f"selected={ck['validation_only_selection']['selected_variant']}; "
                f"neighbor_interaction_rescue_success={ck['neighbor_interaction_rescue_success']}; "
                f"graph_rows={graph_stats['rows']}; rows_with_neighbors={graph_stats['rows_with_neighbors']}; "
                f"control all/t50/hard={_pct(ck_base['all_improvement'])}/{_pct(ck_base['t50_improvement'])}/{_pct(ck_base['hard_failure_improvement'])}"
            ),
        },
        {
            "item": "Stage42-CK graph/scalar candidates",
            "status": "diagnostic_negative",
            "paper_use": "ablation boundary",
            "evidence": (
                f"scalar all/t50/hard={_pct(ck_scalar['all_improvement'])}/{_pct(ck_scalar['t50_improvement'])}/{_pct(ck_scalar['hard_failure_improvement'])}; "
                f"knn_graph all/t50/hard={_pct(ck_graph['all_improvement'])}/{_pct(ck_graph['t50_improvement'])}/{_pct(ck_graph['hard_failure_improvement'])}; "
                f"graph_goal all/t50/hard={_pct(ck_graph_goal['all_improvement'])}/{_pct(ck_graph_goal['t50_improvement'])}/{_pct(ck_graph_goal['hard_failure_improvement'])}"
            ),
        },
    ]


def _refresh_lines(rows: list[dict[str, str]]) -> list[str]:
    return [
        "## Stage42-CL Post-CJ/CK Context Guard Refresh",
        "",
        "- source: `fresh_synthesis_from_stage42_cj_ck_artifacts`",
        "- scope: protected dataset-local raw-frame 2.5D paper package only.",
        "- Stage42-CJ tested whether goal/scene context can become a validation-only gated expert over baseline-family rollout context.",
        "- Stage42-CK tested whether scalar neighbor or kNN interaction graph context can become a validation-only gated expert.",
        "- Both gates selected `baseline_family_control`, so goal/scene and neighbor/interaction remain diagnostic/auxiliary rather than main claims.",
        "- This refresh updates paper-package language to prevent context overclaims.",
        "- Stage5C remains unexecuted and SMC remains disabled.",
        "",
        "### Evidence Rows",
        "",
        "| item | status | paper use | evidence |",
        "| --- | --- | --- | --- |",
        *[f"| {row['item']} | `{row['status']}` | {row['paper_use']} | {row['evidence']} |" for row in rows],
        "",
        "### Claim Boundary",
        "",
        "- Supported main mechanism remains baseline-family rollout context plus causal history under a conservative safety floor.",
        "- Goal/scene context is not a standalone main contribution under the current source-level ridge/full-waypoint protocol.",
        "- Neighbor/interaction context is not a standalone main contribution under the current source-level ridge/full-waypoint protocol.",
        "- Rejected: true 3D, foundation model, global metric prediction, seconds-level horizon, Stage5C execution, SMC readiness, and ungated/floor-free neural deployment.",
    ]


def _refresh_paper_files(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    status = []
    lines = _refresh_lines(rows)
    for path in PAPER_FILES:
        _replace_section(path, "STAGE42_CL_CONTEXT_GUARD_REFRESH", lines)
        text = path.read_text(encoding="utf-8")
        status.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "contains_stage42_cl": "Stage42-CL Post-CJ/CK Context Guard Refresh" in text,
                "blocks_goal_scene_main_claim": "Goal/scene context is not a standalone main contribution" in text,
                "blocks_neighbor_interaction_main_claim": "Neighbor/interaction context is not a standalone main contribution" in text,
                "contains_no_metric_boundary": "global metric prediction" in text,
            }
        )
    return status


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    cj = payload["inputs_loaded"]["cj"]
    ck = payload["inputs_loaded"]["ck"]
    gates = {
        "cj_gate_passed": cj["stage42_cj_gate"]["passed"] == cj["stage42_cj_gate"]["total"],
        "ck_gate_passed": ck["stage42_ck_gate"]["passed"] == ck["stage42_ck_gate"]["total"],
        "goal_scene_rescue_not_overclaimed": cj["goal_scene_rescue_success"] is False
        and payload["claim_boundary"]["goal_scene_main_claim_allowed"] is False,
        "neighbor_interaction_rescue_not_overclaimed": ck["neighbor_interaction_rescue_success"] is False
        and payload["claim_boundary"]["neighbor_interaction_main_claim_allowed"] is False,
        "baseline_family_control_selected_cj": cj["validation_only_selection"]["selected_variant"] == "baseline_family_control",
        "baseline_family_control_selected_ck": ck["validation_only_selection"]["selected_variant"] == "baseline_family_control",
        "paper_files_refreshed": all(row["contains_stage42_cl"] for row in payload["paper_file_status"]),
        "no_future_or_test_leakage": bool(cj["no_leakage"]["test_threshold_tuning"] is False)
        and bool(ck["no_leakage"]["test_threshold_tuning"] is False)
        and bool(cj["no_leakage"]["future_endpoint_input"] is False)
        and bool(ck["no_leakage"]["future_endpoint_input"] is False),
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_cl_context_guard_paper_refresh_pass" if passed == total else "stage42_cl_context_guard_paper_refresh_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-CL Post-CJ/CK Context Guard Paper Refresh",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_cl_gate']['passed']} / {payload['stage42_cl_gate']['total']}`",
        f"- verdict: `{payload['stage42_cl_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Evidence Rows",
        "",
        "| item | status | paper use | evidence |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['item']} | `{row['status']}` | {row['paper_use']} | {row['evidence']} |"
        for row in payload["evidence_rows"]
    )
    lines += [
        "",
        "## Paper File Status",
        "",
        "| file | refreshed | blocks goal/scene | blocks neighbor/interaction | metric boundary |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["paper_file_status"]:
        lines.append(
            f"| `{row['path']}` | {row['contains_stage42_cl']} | {row['blocks_goal_scene_main_claim']} | {row['blocks_neighbor_interaction_main_claim']} | {row['contains_no_metric_boundary']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- CJ/CK are fresh diagnostic negative evidence, not failures hidden under a broader success claim.",
        "- The paper package can claim baseline-family rollout context, causal history, guarded domain expert, and protected safe-switch evidence.",
        "- The paper package must not claim goal/scene or neighbor/interaction as independent uniformly positive main contributions under the current protocol.",
        "- Stage5C and SMC remain disabled.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_cl_gate"]
    lines = [
        "# Stage42-CL Gate",
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


def run_stage42_post_context_guard_paper_refresh() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cj = read_json(CJ_JSON, {})
    ck = read_json(CK_JSON, {})
    rows = _evidence_rows(cj, ck)
    paper_file_status = _refresh_paper_files(rows)
    payload: dict[str, Any] = {
        "source": "fresh_synthesis_from_stage42_cj_ck_artifacts",
        "stage": "Stage42-CL post-CJ/CK context guard paper refresh",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([CJ_JSON, CK_JSON]),
        "inputs_loaded": {
            "cj": cj,
            "ck": ck,
        },
        "evidence_rows": rows,
        "paper_file_status": paper_file_status,
        "claim_boundary": {
            "goal_scene_main_claim_allowed": False,
            "neighbor_interaction_main_claim_allowed": False,
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_cl_gate"] = _gate(payload)

    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["item", "status", "paper_use", "evidence"])
        writer.writeheader()
        writer.writerows(rows)
    return payload


if __name__ == "__main__":
    run_stage42_post_context_guard_paper_refresh()
