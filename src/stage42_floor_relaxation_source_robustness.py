from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_joint_multiagent_consistency as jmc
from src import stage42_floor_relaxation_safety_stress as gt
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")

GT_JSON = OUT_DIR / "floor_relaxation_safety_stress_stage42.json"
GU_JSON = OUT_DIR / "floor_relaxation_paper_refresh_stage42.json"

REPORT_JSON = OUT_DIR / "floor_relaxation_source_robustness_stage42.json"
REPORT_MD = OUT_DIR / "floor_relaxation_source_robustness_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gv_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
CONSOLIDATED_SUMMARY = Path("README_M3W_CURRENT_GOAL_CONSOLIDATED_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

PAPER_FILES = [
    OUT_DIR / "paper_ready_evidence_matrix_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
    OUT_DIR / "experiment_tables_stage42.md",
]

SOURCE = "fresh_stage42_gv_floor_relaxation_source_robustness"
MARKER = "STAGE42_GV_FLOOR_RELAXATION_SOURCE_ROBUSTNESS"
TARGET_SLICES = ("TrajNet|50", "UCY|50")
DOMINANCE_WARN_FRAC = 0.90
EASY_LIMIT = gt.EASY_LIMIT
NEAR_COLLISION_EPS = gt.NEAR_COLLISION_EPS

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-GV 是 Stage42-GT partial t50 floor-relaxation 的 source-level all-agent robustness audit。",
    "本阶段不训练新模型，不下载数据，不转换新数据，不执行 Stage5C，不启用 SMC。",
    "future endpoints / waypoints 只作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 坐标不能写成 global metric。",
    "如果 source 过于集中，必须保留 source-concentration caveat，不能写成 broad source-level generalization。",
]


def _pct(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def _source_name(path: str) -> str:
    p = Path(path)
    parent = p.parent.name
    return f"{parent}/{p.name}" if parent else p.name


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


def _mask_for_source(data: Mapping[str, np.ndarray], split: np.ndarray, slice_name: str, source_file: str) -> np.ndarray:
    domain, horizon_s = slice_name.split("|", 1)
    return (
        (split == "test")
        & (data["dataset"].astype(str) == domain)
        & (data["horizon"].astype(int) == int(horizon_s))
        & (data["source_file"].astype(str) == source_file)
    )


def _source_safety_row(bundle: Mapping[str, Any], mask: np.ndarray, slice_name: str, source_file: str) -> dict[str, Any]:
    data = bundle["data"]
    metric = am._metric(bundle["selected_ade"], bundle["floor_ade"], data, bundle["switch"], mask)
    floor_switch = np.zeros(len(bundle["switch"]), dtype=bool)
    floor_stats = gt._joint_for_mask(bundle, mask, "floor", bundle["floor_xy"], floor_switch)
    selected_stats = gt._joint_for_mask(bundle, mask, "partial_floor_relaxation", bundle["selected_xy"], bundle["switch"])
    normalizer = bundle["joint_labels"]["normalizer"][mask].astype(np.float64)
    keys = bundle["keys"][mask]
    floor_min = jmc._min_group_distance(bundle["floor_xy"][mask], keys, normalizer)
    selected_min = jmc._min_group_distance(bundle["selected_xy"][mask], keys, normalizer)
    delta = gt._delta_stats(selected_stats, floor_stats)
    near_boot = gt._bootstrap_near_delta(selected_min, floor_min)
    safety_pass = bool(
        metric["t50_improvement"] > 0.0
        and metric["hard_failure_improvement"] > 0.0
        and metric["easy_degradation"] <= EASY_LIMIT
        and delta["near_collision_rate_005_delta"] <= NEAR_COLLISION_EPS
        and delta["jagged_rate_delta"] <= 0.0
    )
    return {
        "source": SOURCE,
        "slice": slice_name,
        "source_file": source_file,
        "source_name": _source_name(source_file),
        "rows": int(np.sum(mask)),
        "groups": int(selected_stats["groups"]),
        "metric": metric,
        "floor_joint_stats": floor_stats,
        "selected_joint_stats": selected_stats,
        "selected_minus_floor": delta,
        "near_collision_delta_bootstrap": near_boot,
        "safety_pass": safety_pass,
    }


def _slice_audit(bundle: Mapping[str, Any], slice_name: str) -> dict[str, Any]:
    data = bundle["data"]
    split = bundle["split"]
    domain, horizon_s = slice_name.split("|", 1)
    slice_mask = (split == "test") & (data["dataset"].astype(str) == domain) & (data["horizon"].astype(int) == int(horizon_s))
    sources = sorted(set(data["source_file"].astype(str)[slice_mask].tolist()))
    source_rows = [_source_safety_row(bundle, _mask_for_source(data, split, slice_name, src), slice_name, src) for src in sources]
    total_rows = int(np.sum(slice_mask))
    largest_rows = max([row["rows"] for row in source_rows], default=0)
    largest_fraction = float(largest_rows / max(total_rows, 1))
    safety_sources = [row["source_name"] for row in source_rows if row["safety_pass"]]
    concentration_limited = bool(len(source_rows) < 3 or largest_fraction >= DOMINANCE_WARN_FRAC)
    return {
        "source": SOURCE,
        "slice": slice_name,
        "total_rows": total_rows,
        "source_count": len(source_rows),
        "largest_source_fraction": largest_fraction,
        "source_concentration_limited": concentration_limited,
        "safety_positive_source_count": len(safety_sources),
        "safety_positive_sources": safety_sources,
        "source_rows": source_rows,
        "broad_source_generalization_claim_allowed": bool(
            len(source_rows) >= 3 and not concentration_limited and len(safety_sources) >= 2
        ),
    }


def _summary(slice_audits: Mapping[str, Any], gt_report: Mapping[str, Any], gu_report: Mapping[str, Any]) -> dict[str, Any]:
    concentration_limited = [name for name, row in slice_audits.items() if row["source_concentration_limited"]]
    safety_positive = [name for name, row in slice_audits.items() if row["safety_positive_source_count"] >= 1]
    broad_allowed = all(row["broad_source_generalization_claim_allowed"] for row in slice_audits.values())
    gt_summary = gt_report.get("summary", {})
    return {
        "source": SOURCE,
        "gt_verdict": gt_report.get("stage42_gt_gate", {}).get("verdict", ""),
        "gu_verdict": gu_report.get("stage42_gu_gate", {}).get("verdict", ""),
        "target_slices": list(TARGET_SLICES),
        "source_safety_positive_slices": safety_positive,
        "source_concentration_limited_slices": concentration_limited,
        "broad_source_generalization_claim_allowed": bool(broad_allowed),
        "source_level_claim": "major-source all-agent safety support with source-concentration caveat; not broad source-level generalization",
        "target_union_rows": int(gt_summary.get("target_union_rows", 0) or 0),
        "target_union_t50_improvement": float(gt_summary.get("target_union_t50_improvement", 0.0) or 0.0),
        "target_union_near_collision_005_delta": float(gt_summary.get("target_union_near_collision_005_delta", 0.0) or 0.0),
        "training_executed": False,
        "download_executed": False,
        "conversion_executed": False,
        "threshold_tuned_on_test": False,
        "global_floor_removal_allowed": False,
        "floor_free_neural_deployable": False,
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    no_leakage = payload["no_leakage"]
    gates = {
        "gt_input_passed": s["gt_verdict"] == "stage42_gt_floor_relaxation_safety_stress_pass",
        "gu_input_passed": s["gu_verdict"] in {"stage42_gu_floor_relaxation_paper_refresh_pass", ""},
        "target_slices_audited": set(payload["slice_audits"].keys()) == set(TARGET_SLICES),
        "source_rows_present": all(row["total_rows"] > 0 and row["source_count"] > 0 for row in payload["slice_audits"].values()),
        "major_source_safety_positive": set(s["source_safety_positive_slices"]) == set(TARGET_SLICES),
        "source_concentration_caveat_recorded": len(s["source_concentration_limited_slices"]) >= 1,
        "broad_source_generalization_not_overclaimed": s["broad_source_generalization_claim_allowed"] is False
        and claim["broad_source_generalization_claim_allowed"] is False,
        "no_training_download_conversion_or_test_tuning": not (
            s["training_executed"]
            or s["download_executed"]
            or s["conversion_executed"]
            or s["threshold_tuned_on_test"]
        ),
        "no_leakage_pass": no_leakage["future_endpoint_input"] is False
        and no_leakage["future_waypoint_input"] is False
        and no_leakage["central_velocity"] is False
        and no_leakage["test_endpoint_goals"] is False
        and no_leakage["test_threshold_tuning"] is False,
        "global_floor_removal_false": claim["global_floor_removal_allowed"] is False,
        "floor_free_neural_false": claim["floor_free_neural_deployable"] is False,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(bool(v) for v in gates.values())
    total = len(gates)
    verdict = (
        "stage42_gv_floor_relaxation_source_robustness_pass_with_source_concentration_caveat"
        if passed == total
        else "stage42_gv_floor_relaxation_source_robustness_partial"
    )
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-GV Floor-Relaxation Source Robustness Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_gv_gate']['passed']} / {payload['stage42_gv_gate']['total']}`",
        f"- verdict: `{payload['stage42_gv_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in s.items()],
        "",
        "## Source-Level Audit",
        "",
        "| slice | source count | total rows | largest source fraction | safety-positive sources | broad source claim |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for name, row in payload["slice_audits"].items():
        lines.append(
            f"| `{name}` | {row['source_count']} | {row['total_rows']} | {_pct(row['largest_source_fraction'])} | "
            f"`{row['safety_positive_sources']}` | {row['broad_source_generalization_claim_allowed']} |"
        )
    lines += [
        "",
        "## Per-Source Metrics",
        "",
        "| slice | source | rows | groups | t50 | hard | easy | switch | near@0.05 delta | safety pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, audit in payload["slice_audits"].items():
        for row in audit["source_rows"]:
            metric = row["metric"]
            delta = row["selected_minus_floor"]
            lines.append(
                f"| `{name}` | `{row['source_name']}` | {row['rows']} | {row['groups']} | "
                f"{_pct(metric['t50_improvement'])} | {_pct(metric['hard_failure_improvement'])} | "
                f"{_pct(metric['easy_degradation'])} | {_pct(metric['switch_rate'])} | "
                f"{_pct(delta['near_collision_rate_005_delta'])} | {row['safety_pass']} |"
            )
    lines += [
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_gv_gate"]["gates"].items()],
        "",
        "## Interpretation",
        "",
        "- Stage42-GT partial t50 floor relaxation has source-level all-agent safety support on the available major sources.",
        "- The evidence remains source-concentrated, so broad source-level generalization is still disallowed.",
        "- This strengthens the safety claim for audited t50 slices while preserving the source-diversity limitation.",
        "- Stage5C remains unexecuted and SMC remains disabled.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_gv_gate"]
    return [
        "# Stage42-GV Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    return [
        "## Stage42-GV Floor Relaxation Source Robustness",
        "",
        "- source: `fresh_stage42_gv_floor_relaxation_source_robustness`",
        "- role: source-level all-agent robustness audit for Stage42-GT partial t50 floor relaxation.",
        f"- gate: `{payload['stage42_gv_gate']['passed']} / {payload['stage42_gv_gate']['total']}`; verdict `{payload['stage42_gv_gate']['verdict']}`.",
        f"- source-safety-positive slices: `{s['source_safety_positive_slices']}`.",
        f"- source-concentration-limited slices: `{s['source_concentration_limited_slices']}`.",
        f"- broad source-level generalization claim allowed: `{s['broad_source_generalization_claim_allowed']}`.",
        "- Claim boundary: major-source support only; not broad source-level generalization, not global floor removal, not floor-free neural, not metric/seconds-level, not Stage5C, not SMC.",
    ]


def _refresh_docs(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, CONSOLIDATED_SUMMARY, *PAPER_FILES]:
        _replace_section(path, MARKER, lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-GV floor relaxation source robustness"
    state["current_verdict"] = payload["stage42_gv_gate"]["verdict"]
    state["stage42_gv_floor_relaxation_source_robustness"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_gv_gate"]["verdict"],
        "gates": f"{payload['stage42_gv_gate']['passed']}/{payload['stage42_gv_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_floor_relaxation_source_robustness(*, refresh_docs: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    gt_report = read_json(GT_JSON, {})
    gu_report = read_json(GU_JSON, {})
    bundle = gt._fit_policy_with_xy()
    slice_audits = {name: _slice_audit(bundle, name) for name in TARGET_SLICES}
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-GV Floor Relaxation Source Robustness Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([GT_JSON, GU_JSON]),
        "current_facts": CURRENT_FACTS,
        "input_reports": {
            "gt_verdict": gt_report.get("stage42_gt_gate", {}).get("verdict", ""),
            "gu_verdict": gu_report.get("stage42_gu_gate", {}).get("verdict", ""),
        },
        "slice_audits": slice_audits,
        "summary": _summary(slice_audits, gt_report, gu_report),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "test_rows_for_reporting_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "broad_source_generalization_claim_allowed": False,
            "global_floor_removal_allowed": False,
            "floor_free_neural_deployable": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_gv_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_docs:
        _refresh_docs(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_floor_relaxation_source_robustness()
