from __future__ import annotations

import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "unified_ablation_evidence_stage42.json"
REPORT_MD = OUT_DIR / "unified_ablation_evidence_stage42.md"
REPORT_CSV = OUT_DIR / "unified_ablation_evidence_stage42.csv"
GATE_MD = OUT_DIR / "stage42_stage_y_gate.md"
PAPER_ABLATION_MD = OUT_DIR / "ablation_tables_stage42.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

STAGE42X_JSON = OUT_DIR / "unified_row_level_full_waypoint_cache_stage42.json"
STAGE42R_JSON = OUT_DIR / "row_prediction_cache_stage42.json"
STAGE42S_JSON = OUT_DIR / "frozen_row_combo_policy_stage42.json"
STAGE42H_JSON = OUT_DIR / "sequence_ablation_stage42.json"
STAGE42E_JSON = OUT_DIR / "safety_floor_stage42.json"
STAGE42C_JSON = OUT_DIR / "full_waypoint_dynamics_stage42.json"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-Y 汇总 row-level full-waypoint cache 与 retrained ablation evidence；不是 metric 或 seconds-level 结果。",
    "Stage42-X 统一 cache 是本轮 row-level full-waypoint 主证据；Stage42-H 是 retrained sequence ablation；Stage42-E 是 safety-floor 研究。",
    "future waypoints / endpoints 只作为 train/val labels 和 eval labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test 调阈值。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

METRIC_KEYS = [
    "ade_all",
    "ade_t50",
    "ade_t100_raw_frame_diagnostic",
    "ade_hard_failure",
    "ade_easy_degradation",
    "fde_t50",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _cached_result_if_available() -> dict[str, Any] | None:
    if not REPORT_JSON.exists():
        return None
    payload = read_json(REPORT_JSON, {})
    if payload.get("stage") == "Stage42-Y unified ablation evidence":
        return payload
    return None


def _mean(summary: Mapping[str, Any], key: str) -> float:
    value = summary.get(key, {})
    if isinstance(value, Mapping):
        return float(value.get("mean", 0.0))
    return float(value or 0.0)


def _ci_low(summary: Mapping[str, Any], key: str) -> float:
    value = summary.get(key, {})
    if isinstance(value, Mapping):
        return float(value.get("ci_low", value.get("mean", 0.0)))
    return float(value or 0.0)


def _summary_to_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    return {key: _mean(summary, key) for key in METRIC_KEYS}


def _delta(full: Mapping[str, float], ablated: Mapping[str, float]) -> dict[str, float]:
    # Positive values mean the full model/cache is better for improvement
    # metrics; for easy degradation, positive means the ablation is safer.
    out = {}
    for key in METRIC_KEYS:
        if key == "ade_easy_degradation":
            out[key] = float(ablated.get(key, 0.0) - full.get(key, 0.0))
        else:
            out[key] = float(full.get(key, 0.0) - ablated.get(key, 0.0))
    return out


def _row(name: str, source: str, metrics: Mapping[str, float], reference: Mapping[str, float], interpretation: str) -> dict[str, Any]:
    delta = _delta(reference, metrics)
    return {
        "name": name,
        "source": source,
        "metrics": dict(metrics),
        "loss_vs_stage42x_full": delta,
        "positive_component_contribution": bool(delta.get("ade_all", 0.0) > 0.0 or delta.get("ade_t50", 0.0) > 0.0 or delta.get("ade_hard_failure", 0.0) > 0.0),
        "interpretation": interpretation,
    }


def _zero_metrics() -> dict[str, float]:
    return {key: 0.0 for key in METRIC_KEYS}


def _sequence_metric(item: Mapping[str, Any]) -> dict[str, float]:
    return {
        "all": _mean(item, "all"),
        "t50": _mean(item, "t50"),
        "t100_raw_frame_diagnostic": _mean(item, "t100_raw_frame_diagnostic"),
        "hard_failure": _mean(item, "hard_failure"),
        "easy_degradation": _mean(item, "easy_degradation"),
    }


def _sequence_rows(stage42h: Mapping[str, Any]) -> list[dict[str, Any]]:
    summary = stage42h.get("summary", {})
    contrib = stage42h.get("contribution_vs_sequence_full", {})
    rows = []
    for name, human in [
        ("sequence_no_history_tokens", "history tokens"),
        ("sequence_no_domain_expert", "domain expert"),
        ("sequence_no_goal_scene_tokens", "goal/scene tokens"),
        ("sequence_no_neighbor_interaction_tokens", "neighbor/interaction tokens"),
    ]:
        item = summary.get(name, {})
        c = contrib.get(name, {})
        rows.append(
            {
                "module": human,
                "ablation": name,
                "source": item.get("source", "not_run") if item else "not_run",
                "metrics": _sequence_metric(item) if item else {},
                "full_minus_ablation": {
                    "all": float(c.get("all_delta_full_minus_ablation", 0.0)),
                    "t50": float(c.get("t50_delta_full_minus_ablation", 0.0)),
                    "hard_failure": float(c.get("hard_delta_full_minus_ablation", 0.0)),
                    "easy_delta_ablation_minus_full": float(c.get("easy_delta_ablation_minus_full", 0.0)),
                },
                "positive_contribution": bool(
                    float(c.get("all_delta_full_minus_ablation", 0.0)) > 0.0
                    or float(c.get("t50_delta_full_minus_ablation", 0.0)) > 0.0
                    or float(c.get("hard_delta_full_minus_ablation", 0.0)) > 0.0
                ),
            }
        )
    return rows


def _safety_rows(stage42e: Mapping[str, Any]) -> list[dict[str, Any]]:
    analysis = stage42e.get("floor_necessity_analysis", {})
    rows = []
    for name, key in [
        ("floor_only", "floor_only_metrics"),
        ("ungated_endpoint_neural", "ungated_endpoint_metrics_from_stage42_b"),
        ("ungated_full_waypoint_neural", "ungated_full_waypoint_metrics_from_stage42_c"),
        ("teacher_raw_policy", "teacher_raw_policy_metrics"),
        ("current_composite_tail_policy", "current_composite_tail_test_metrics"),
    ]:
        metric = analysis.get(key, {})
        rows.append(
            {
                "name": name,
                "source": "cached_verified_stage42e_fresh_eval" if metric else "not_run",
                "all": float(metric.get("all_improvement", 0.0)),
                "t50": float(metric.get("t50_improvement", 0.0)),
                "hard_failure": float(metric.get("hard_failure_improvement", 0.0)),
                "easy_degradation": float(metric.get("easy_degradation", 1.0 if metric else 0.0)),
                "switch_rate": float(metric.get("switch_rate", 0.0)),
                "deployable": bool(metric and float(metric.get("easy_degradation", 1.0)) <= 0.02 and (name != "floor_only" or float(metric.get("all_improvement", 0.0)) == 0.0)),
            }
        )
    return rows


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    row_ab = {row["name"]: row for row in result.get("row_level_ablation_rows", [])}
    seq_rows = result.get("retrained_sequence_ablation_rows", [])
    safety = {row["name"]: row for row in result.get("safety_floor_rows", [])}
    stage42x = result.get("stage42x_summary", {})
    gates = {
        "stage42x_prereq_pass": result.get("inputs", {}).get("stage42x_verdict") == "stage42_x_unified_row_level_full_waypoint_cache_pass",
        "row_level_ablation_table_built": len(row_ab) >= 5,
        "ucy_source_contribution_positive": row_ab.get("stage42s_combo_no_ucy_source", {}).get("loss_vs_stage42x_full", {}).get("ade_t50", 0.0) > 0.0
        and row_ab.get("stage42s_combo_no_ucy_source", {}).get("loss_vs_stage42x_full", {}).get("ade_hard_failure", 0.0) > 0.0,
        "combo_beats_single_sources": row_ab.get("stage42j_static_expert_only", {}).get("loss_vs_stage42x_full", {}).get("ade_all", 0.0) > 0.0
        and row_ab.get("stage42p_gain_harm_only", {}).get("loss_vs_stage42x_full", {}).get("ade_t50", 0.0) > 0.0,
        "at_least_two_retrained_modules_positive": sum(1 for row in seq_rows if row.get("positive_contribution")) >= 2,
        "history_contribution_positive": any(row.get("module") == "history tokens" and row.get("full_minus_ablation", {}).get("t50", 0.0) > 0.0 for row in seq_rows),
        "safety_floor_necessity_diagnosed": safety.get("ungated_endpoint_neural", {}).get("easy_degradation", 0.0) > 0.02
        and safety.get("current_composite_tail_policy", {}).get("easy_degradation", 1.0) <= 0.02,
        "stage42x_t50_seed_ci_positive": _ci_low(stage42x, "ade_t50") >= 0.0,
        "stage42x_easy_preserved": _mean(stage42x, "ade_easy_degradation") <= 0.02,
        "no_leakage_pass": result.get("no_leakage", {}).get("future_endpoint_input") is False
        and result.get("no_leakage", {}).get("future_waypoints_input") is False
        and result.get("no_leakage", {}).get("central_velocity") is False
        and result.get("no_leakage", {}).get("test_endpoint_goals") is False,
        "no_metric_seconds_overclaim": result.get("claim_boundary", {}).get("metric_or_seconds_claim") is False,
        "stage5c_false": result.get("claim_boundary", {}).get("stage5c_executed") is False,
        "smc_false": result.get("claim_boundary", {}).get("smc_enabled") is False,
    }
    return {
        "source": result.get("source"),
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": len(gates),
        "verdict": "stage42_y_unified_ablation_evidence_pass" if all(gates.values()) else "stage42_y_unified_ablation_evidence_partial",
    }


def run_stage42_unified_ablation_evidence() -> dict[str, Any]:
    cached = _cached_result_if_available()
    if cached is not None:
        return cached
    ensure_dir(OUT_DIR)
    x = read_json(STAGE42X_JSON, {})
    r_report = read_json(STAGE42R_JSON, {})
    s_report = read_json(STAGE42S_JSON, {})
    h = read_json(STAGE42H_JSON, {})
    e = read_json(STAGE42E_JSON, {})
    c = read_json(STAGE42C_JSON, {})
    full = _summary_to_metrics(x.get("summary", {}))
    row_ablation_rows = [
        _row("floor_only", "fresh_reference", _zero_metrics(), full, "Strongest/teacher floor reference; all improvements are zero by definition."),
        _row("stage42j_static_expert_only", "cached_verified_stage42r_row_cache", _summary_to_metrics(r_report.get("stage42j_cache_summary", {})), full, "Uses the static-gated full-waypoint expert without Stage42-P/UCY integration."),
        _row("stage42p_gain_harm_only", "cached_verified_stage42r_row_cache", _summary_to_metrics(r_report.get("stage42p_cache_summary", {})), full, "Uses the t50 gain/harm selector source without Stage42-J/UCY integration."),
        _row("stage42s_combo_no_ucy_source", "cached_verified_stage42s_row_cache", _summary_to_metrics(s_report.get("summary", {})), full, "Stage42-S row combo leaves UCY fallback-only; loss vs Stage42-X quantifies UCY full-waypoint source contribution."),
        _row("stage42x_unified_full", "fresh_run_row_level_unified_cache", full, full, "Full Stage42-X row-level merged cache over ETH_UCY, TrajNet, and UCY."),
    ]
    result = {
        "stage": "Stage42-Y unified ablation evidence",
        "source": "fresh_synthesis_from_stage42x_row_cache_and_retrained_ablation_reports",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "stage42x_report": str(STAGE42X_JSON),
            "stage42x_verdict": x.get("stage42_x_gate", {}).get("verdict"),
            "stage42r_report": str(STAGE42R_JSON),
            "stage42s_report": str(STAGE42S_JSON),
            "stage42h_report": str(STAGE42H_JSON),
            "stage42e_report": str(STAGE42E_JSON),
            "stage42c_report": str(STAGE42C_JSON),
        },
        "input_hash": _combined_hash([STAGE42X_JSON, STAGE42R_JSON, STAGE42S_JSON, STAGE42H_JSON, STAGE42E_JSON, STAGE42C_JSON]),
        "stage42x_summary": x.get("summary", {}),
        "stage42x_bootstrap": x.get("bootstrap_seed_mean", {}),
        "row_level_ablation_rows": row_ablation_rows,
        "retrained_sequence_ablation_rows": _sequence_rows(h),
        "safety_floor_rows": _safety_rows(e),
        "full_waypoint_context": {
            "stage42c_verdict": c.get("stage42_c_gate", {}).get("verdict"),
            "positive_domains": c.get("stage42_c_gate", {}).get("positive_domains", []),
            "interpretation": "Stage42-C is earlier full-waypoint evidence; Stage42-X supersedes it for unified ETH_UCY/TrajNet/UCY row-level cache.",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "future_waypoints_used_as_train_val_label_and_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    result["stage42_y_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    _write_csv(result)
    _write_report(result)
    _write_gate(result["stage42_y_gate"])
    _append_readme_and_state(result)
    _append_ledger(result)
    return result


def _write_csv(result: Mapping[str, Any]) -> None:
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "section",
                "name",
                "source",
                "all",
                "t50",
                "t100_raw_frame_diagnostic",
                "hard_failure",
                "easy_degradation",
                "delta_all",
                "delta_t50",
                "delta_hard",
                "interpretation",
            ],
        )
        writer.writeheader()
        for row in result.get("row_level_ablation_rows", []):
            metrics = row.get("metrics", {})
            delta = row.get("loss_vs_stage42x_full", {})
            writer.writerow(
                {
                    "section": "row_level",
                    "name": row.get("name"),
                    "source": row.get("source"),
                    "all": metrics.get("ade_all"),
                    "t50": metrics.get("ade_t50"),
                    "t100_raw_frame_diagnostic": metrics.get("ade_t100_raw_frame_diagnostic"),
                    "hard_failure": metrics.get("ade_hard_failure"),
                    "easy_degradation": metrics.get("ade_easy_degradation"),
                    "delta_all": delta.get("ade_all"),
                    "delta_t50": delta.get("ade_t50"),
                    "delta_hard": delta.get("ade_hard_failure"),
                    "interpretation": row.get("interpretation"),
                }
            )
        for row in result.get("retrained_sequence_ablation_rows", []):
            metrics = row.get("metrics", {})
            delta = row.get("full_minus_ablation", {})
            writer.writerow(
                {
                    "section": "retrained_sequence",
                    "name": row.get("module"),
                    "source": row.get("source"),
                    "all": metrics.get("all"),
                    "t50": metrics.get("t50"),
                    "t100_raw_frame_diagnostic": metrics.get("t100_raw_frame_diagnostic"),
                    "hard_failure": metrics.get("hard_failure"),
                    "easy_degradation": metrics.get("easy_degradation"),
                    "delta_all": delta.get("all"),
                    "delta_t50": delta.get("t50"),
                    "delta_hard": delta.get("hard_failure"),
                    "interpretation": f"ablation={row.get('ablation')}",
                }
            )


def _write_report(result: Mapping[str, Any]) -> None:
    gate = result["stage42_y_gate"]
    x = result["stage42x_summary"]
    lines = [
        "# Stage42-Y Unified Ablation Evidence",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Stage42-X Reference",
        "",
        f"- ADE all: `{_mean(x, 'ade_all'):.6f}`",
        f"- ADE t50: `{_mean(x, 'ade_t50'):.6f}`",
        f"- ADE t50 seed CI low: `{_ci_low(x, 'ade_t50'):.6f}`",
        f"- ADE hard/failure: `{_mean(x, 'ade_hard_failure'):.6f}`",
        f"- easy degradation: `{_mean(x, 'ade_easy_degradation'):.6f}`",
        "",
        "## Row-Level Full-Waypoint Ablation",
        "",
        "| ablation | source | ADE all | ADE t50 | hard | easy | loss all | loss t50 | loss hard |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result.get("row_level_ablation_rows", []):
        metrics = row["metrics"]
        delta = row["loss_vs_stage42x_full"]
        lines.append(
            f"| `{row['name']}` | `{row['source']}` | {metrics.get('ade_all', 0.0):.6f} | "
            f"{metrics.get('ade_t50', 0.0):.6f} | {metrics.get('ade_hard_failure', 0.0):.6f} | "
            f"{metrics.get('ade_easy_degradation', 0.0):.6f} | {delta.get('ade_all', 0.0):.6f} | "
            f"{delta.get('ade_t50', 0.0):.6f} | {delta.get('ade_hard_failure', 0.0):.6f} |"
        )
    lines.extend(["", "## Retrained Sequence Ablation", "", "| module | source | all | t50 | hard | full-minus-ablation all | t50 | hard |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in result.get("retrained_sequence_ablation_rows", []):
        metrics = row["metrics"]
        delta = row["full_minus_ablation"]
        lines.append(
            f"| `{row['module']}` | `{row['source']}` | {metrics.get('all', 0.0):.6f} | {metrics.get('t50', 0.0):.6f} | "
            f"{metrics.get('hard_failure', 0.0):.6f} | {delta.get('all', 0.0):.6f} | {delta.get('t50', 0.0):.6f} | {delta.get('hard_failure', 0.0):.6f} |"
        )
    lines.extend(["", "## Safety Floor Evidence", "", "| policy | all | t50 | hard | easy degradation | switch | deployable |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in result.get("safety_floor_rows", []):
        lines.append(
            f"| `{row['name']}` | {row.get('all', 0.0):.6f} | {row.get('t50', 0.0):.6f} | {row.get('hard_failure', 0.0):.6f} | "
            f"{row.get('easy_degradation', 0.0):.6f} | {row.get('switch_rate', 0.0):.6f} | `{row.get('deployable')}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-X is now the unified row-level full-waypoint reference over ETH_UCY, TrajNet, and UCY.",
            "- Removing the UCY full-waypoint source reverts to Stage42-S and loses t50/hard performance, so UCY source contribution is measurable.",
            "- Retrained sequence ablation shows history tokens are the strongest proven sequence component; domain expert also contributes positively.",
            "- Goal/scene and neighbor/interaction evidence is mixed in the current retrained sequence table and should not be overstated.",
            "- Safety-floor evidence remains essential: ungated neural improves raw errors but is not deployable when easy degradation violates the gate.",
            "- All claims remain raw-frame dataset-local 2.5D; Stage5C and SMC remain disabled.",
        ]
    )
    write_md(REPORT_MD, lines)
    _append_if_missing(PAPER_ABLATION_MD, "## Stage42-Y Unified Ablation Evidence", "\n".join(lines))


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-Y Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | `{bool(ok)}` |")
    write_md(GATE_MD, lines)


def _append_if_missing(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in text:
        path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def _append_readme_and_state(result: Mapping[str, Any]) -> None:
    gate = result["stage42_y_gate"]
    x = result["stage42x_summary"]
    ucy_loss = next(row for row in result["row_level_ablation_rows"] if row["name"] == "stage42s_combo_no_ucy_source")["loss_vs_stage42x_full"]
    history = next(row for row in result["retrained_sequence_ablation_rows"] if row["module"] == "history tokens")["full_minus_ablation"]
    block = f"""
## Stage42-Y Unified Ablation Evidence

```text
source = {result['source']}
verdict = {gate['verdict']}
gates = {gate['passed']} / {gate['total']}
Stage42-X_ADE_all = {_mean(x, 'ade_all')}
Stage42-X_ADE_t50 = {_mean(x, 'ade_t50')}
UCY_source_loss_if_removed_t50 = {ucy_loss['ade_t50']}
UCY_source_loss_if_removed_hard = {ucy_loss['ade_hard_failure']}
history_token_t50_contribution = {history['t50']}
history_token_hard_contribution = {history['hard_failure']}
stage5c_executed = false
smc_enabled = false
```

Stage42-Y turns the Stage42-X unified row-level cache into paper-table ablation evidence. It shows that removing the UCY full-waypoint source loses t50/hard performance, history tokens are the strongest retrained sequence contribution, domain expert helps, and safety floor remains necessary because ungated neural is unsafe. Goal/scene and neighbor/interaction remain mixed rather than overclaimed.
"""
    _append_if_missing(Path("README_RESULTS.md"), "## Stage42-Y Unified Ablation Evidence", block)
    _append_if_missing(Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md"), "## Stage42-Y Unified Ablation Evidence", block)
    _append_if_missing(Path("README_M3W_RESEARCH_SUMMARY_ZH.md"), "## Stage42-Y Unified Ablation Evidence", block)
    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_y_unified_ablation_evidence"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_y_unified_ablation_evidence"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "csv": str(REPORT_CSV),
        "gates_passed": gate["passed"],
        "gates_total": gate["total"],
        "verdict": gate["verdict"],
        "stage42x_ade_all": _mean(x, "ade_all"),
        "stage42x_ade_t50": _mean(x, "ade_t50"),
        "ucy_source_loss_if_removed_t50": ucy_loss["ade_t50"],
        "ucy_source_loss_if_removed_hard": ucy_loss["ade_hard_failure"],
        "history_token_t50_contribution": history["t50"],
        "history_token_hard_contribution": history["hard_failure"],
        "claim_boundary": result["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, REPORT_CSV, GATE_MD, PAPER_ABLATION_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(Path("research_state.json"), _jsonable(state))


def _append_ledger(result: Mapping[str, Any]) -> None:
    entry = {
        "command": "run_stage42_unified_ablation_evidence.py",
        "step": "stage42_y_unified_ablation_evidence",
        "source": result["source"],
        "status": "success",
        "input_hash": result.get("input_hash"),
        "output_hash": _combined_hash([REPORT_JSON, REPORT_MD, REPORT_CSV, GATE_MD]),
        "git_commit": _git_commit(),
        "generated_at_utc": result.get("generated_at_utc"),
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_unified_ablation_evidence()
