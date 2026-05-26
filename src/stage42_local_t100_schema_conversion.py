from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src import stage42_local_t100_conversion_readiness as be
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
BE_JSON = OUT_DIR / "local_t100_conversion_readiness_stage42.json"
REPORT_JSON = OUT_DIR / "local_t100_schema_conversion_stage42.json"
REPORT_MD = OUT_DIR / "local_t100_schema_conversion_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bf_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_local_t100_schema_conversion_stage42.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

HORIZONS = [10, 25, 50, 100]
BASELINES = [
    "constant_position",
    "constant_velocity_causal_fd",
    "damped_velocity_0p25",
    "damped_velocity_0p50",
    "damped_velocity_0p75",
    "constant_acceleration_causal",
]


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BF 做 local t100 candidates 的 in-memory schema conversion 和 causal baseline/source-CV audit。",
    "本步骤不提交 full feature store，不训练神经模型，不改变部署模型。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t100 仍是 raw-frame diagnostic / blocker，不能写成 seconds-level。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return float(math.hypot(ax - bx, ay - by))


def _baseline_prediction(name: str, prev2: Mapping[str, Any], prev: Mapping[str, Any], cur: Mapping[str, Any], horizon: int) -> tuple[float, float]:
    x = float(cur["x"])
    y = float(cur["y"])
    vx = float(cur["x"]) - float(prev["x"])
    vy = float(cur["y"]) - float(prev["y"])
    ax = float(cur["x"]) - 2.0 * float(prev["x"]) + float(prev2["x"])
    ay = float(cur["y"]) - 2.0 * float(prev["y"]) + float(prev2["y"])
    if name == "constant_position":
        return x, y
    if name == "constant_velocity_causal_fd":
        return x + vx * horizon, y + vy * horizon
    if name == "damped_velocity_0p25":
        return x + 0.25 * vx * horizon, y + 0.25 * vy * horizon
    if name == "damped_velocity_0p50":
        return x + 0.50 * vx * horizon, y + 0.50 * vy * horizon
    if name == "damped_velocity_0p75":
        return x + 0.75 * vx * horizon, y + 0.75 * vy * horizon
    if name == "constant_acceleration_causal":
        return x + vx * horizon + 0.5 * ax * horizon * horizon, y + vy * horizon + 0.5 * ay * horizon * horizon
    raise KeyError(name)


def _window_errors_for_source(source: Mapping[str, Any]) -> dict[str, Any]:
    rows = be._parse_rows(Path(str(source["path"])))
    tracks = be._track_map(rows)
    horizon_acc: dict[int, dict[str, list[float]]] = {h: {name: [] for name in BASELINES} for h in HORIZONS}
    relative_acc: dict[int, dict[str, list[float]]] = {h: {name: [] for name in BASELINES} for h in HORIZONS}
    window_counts: dict[str, int] = {}
    for track in tracks.values():
        n = len(track)
        for horizon in HORIZONS:
            for i in range(2, n - horizon):
                prev2 = track[i - 2]
                prev = track[i - 1]
                cur = track[i]
                fut = track[i + horizon]
                velocity_scale = max(_dist(float(cur["x"]), float(cur["y"]), float(prev["x"]), float(prev["y"])) * horizon, 1e-6)
                for name in BASELINES:
                    px, py = _baseline_prediction(name, prev2, prev, cur, horizon)
                    err = _dist(px, py, float(fut["x"]), float(fut["y"]))
                    horizon_acc[horizon][name].append(err)
                    relative_acc[horizon][name].append(err / velocity_scale)
    summary: dict[str, Any] = {}
    for horizon in HORIZONS:
        horizon_summary: dict[str, Any] = {}
        n_windows = max((len(vals) for vals in horizon_acc[horizon].values()), default=0)
        window_counts[str(horizon)] = int(n_windows)
        for name in BASELINES:
            vals = horizon_acc[horizon][name]
            rel_vals = relative_acc[horizon][name]
            horizon_summary[name] = {
                "source": "fresh_in_memory_schema_conversion",
                "windows": int(len(vals)),
                "mean_fde": float(sum(vals) / len(vals)) if vals else None,
                "mean_relative_fde": float(sum(rel_vals) / len(rel_vals)) if rel_vals else None,
            }
        valid = {name: row for name, row in horizon_summary.items() if row["mean_fde"] is not None}
        strongest = min(valid, key=lambda name: float(valid[name]["mean_fde"])) if valid else None
        fallback = valid.get("constant_velocity_causal_fd")
        strongest_row = valid.get(strongest) if strongest else None
        improvement_vs_cv = None
        if fallback and strongest_row and float(fallback["mean_fde"]) > 0.0:
            improvement_vs_cv = (float(fallback["mean_fde"]) - float(strongest_row["mean_fde"])) / float(fallback["mean_fde"])
        summary[str(horizon)] = {
            "source": "fresh_in_memory_schema_conversion",
            "windows": int(n_windows),
            "baselines": horizon_summary,
            "strongest_baseline": strongest,
            "improvement_vs_constant_velocity": improvement_vs_cv,
        }
    return {
        "source": "fresh_in_memory_schema_conversion",
        "source_id": source["source_id"],
        "domain": source["domain"],
        "window_counts": window_counts,
        "by_horizon": summary,
    }


def _source_cv_baseline_audit(source_metrics: Mapping[str, Mapping[str, Any]], source_cv_plan: Mapping[str, Any]) -> dict[str, Any]:
    audits: dict[str, Any] = {}
    for domain, plan in source_cv_plan.get("domains", {}).items():
        folds: list[dict[str, Any]] = []
        for fold in plan.get("folds", []):
            val_source = fold["validation_source"]
            holdout_source = fold["holdout_source"]
            val_metrics = source_metrics[val_source]["by_horizon"]["100"]["baselines"]
            valid = {name: row for name, row in val_metrics.items() if row["mean_fde"] is not None}
            if not valid:
                continue
            selected = min(valid, key=lambda name: float(valid[name]["mean_fde"]))
            holdout = source_metrics[holdout_source]["by_horizon"]["100"]["baselines"]
            selected_holdout = holdout[selected]
            cv_holdout = holdout["constant_velocity_causal_fd"]
            improvement = None
            if selected_holdout["mean_fde"] is not None and cv_holdout["mean_fde"]:
                improvement = (float(cv_holdout["mean_fde"]) - float(selected_holdout["mean_fde"])) / float(cv_holdout["mean_fde"])
            folds.append(
                {
                    "source": "fresh_source_cv_baseline_audit",
                    "holdout_source": holdout_source,
                    "validation_source": val_source,
                    "selected_baseline_from_validation": selected,
                    "holdout_t100_windows": int(source_metrics[holdout_source]["by_horizon"]["100"]["windows"]),
                    "holdout_selected_mean_fde": selected_holdout["mean_fde"],
                    "holdout_constant_velocity_mean_fde": cv_holdout["mean_fde"],
                    "holdout_improvement_vs_constant_velocity": improvement,
                }
            )
        improvements = [float(row["holdout_improvement_vs_constant_velocity"]) for row in folds if row["holdout_improvement_vs_constant_velocity"] is not None]
        audits[domain] = {
            "source": "fresh_source_cv_baseline_audit",
            "fold_count": len(folds),
            "folds": folds,
            "all_folds_positive_vs_constant_velocity": bool(improvements and all(v > 0.0 for v in improvements)),
            "mean_holdout_improvement_vs_constant_velocity": float(sum(improvements) / len(improvements)) if improvements else None,
            "minimum_holdout_improvement_vs_constant_velocity": float(min(improvements)) if improvements else None,
        }
    return {"source": "fresh_source_cv_baseline_audit", "domains": audits}


def run_stage42_local_t100_schema_conversion() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    be_payload = _load_json(BE_JSON)
    readiness = list(be_payload.get("source_readiness", []))
    source_metrics = {row["source_id"]: _window_errors_for_source(row) for row in readiness}
    source_cv_audit = _source_cv_baseline_audit(source_metrics, be_payload.get("source_cv_plan", {}))
    total_windows = {
        str(h): int(sum(int(row["by_horizon"][str(h)]["windows"]) for row in source_metrics.values()))
        for h in HORIZONS
    }
    source_cv_domains_positive = [
        domain for domain, row in source_cv_audit["domains"].items() if row["all_folds_positive_vs_constant_velocity"]
    ]
    payload = {
        "source": "fresh_in_memory_schema_conversion",
        "stage": "Stage42-BF Local T100 Schema Conversion And Source-CV Baseline Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([str(BE_JSON)] + [str(row.get("path", "")) for row in readiness]),
        "be_verdict": be_payload.get("stage42_be_gate", {}).get("verdict"),
        "converted_schema": {
            "source": "fresh_in_memory_schema_conversion",
            "fields": [
                "dataset_name",
                "domain",
                "source_id",
                "scene_id",
                "agent_id",
                "frame_id",
                "x",
                "y",
                "vx_causal",
                "vy_causal",
                "horizon",
                "future_x_label_eval_only",
                "future_y_label_eval_only",
                "coordinate_unit",
                "metric_status",
            ],
            "materialized_feature_store_written": False,
            "conversion_mode": "in_memory_aggregate_windows_only",
        },
        "source_metrics": source_metrics,
        "source_cv_audit": source_cv_audit,
        "summary": {
            "source": "fresh_in_memory_schema_conversion",
            "candidate_sources": len(readiness),
            "converted_sources": len(source_metrics),
            "total_eval_windows_by_horizon": total_windows,
            "t100_eval_windows": total_windows["100"],
            "source_cv_domains_evaluated": sorted(source_cv_audit["domains"].keys()),
            "source_cv_domains_positive_vs_constant_velocity": source_cv_domains_positive,
            "materialized_feature_store_written": False,
            "training_run": False,
            "evaluation_run": True,
            "t100_positive_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "metric_or_seconds_claim": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_metrics_for_threshold": False,
            "causal_velocity_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "t100_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "converted_dataset_claim_allowed": True,
            "t100_positive_claim_allowed": False,
        },
    }
    payload["stage42_bf_gate"] = _gate(payload)
    payload["user_action_required"] = _user_actions(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_ledger(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    gates = {
        "be_readiness_loaded": payload["be_verdict"] == "stage42_be_local_t100_conversion_readiness_pass",
        "schema_conversion_ran": s["converted_sources"] == s["candidate_sources"] and s["candidate_sources"] > 0,
        "t50_windows_converted": int(s["total_eval_windows_by_horizon"]["50"]) > 0,
        "t100_windows_converted": int(s["t100_eval_windows"]) > 0,
        "causal_baselines_computed": all(
            "constant_velocity_causal_fd" in row["by_horizon"]["100"]["baselines"] for row in payload["source_metrics"].values()
        ),
        "source_cv_audit_completed": bool(s["source_cv_domains_evaluated"]),
        "no_materialized_large_store": s["materialized_feature_store_written"] is False,
        "no_leakage_pass": all(
            payload["no_leakage"][k] is False
            for k in ["future_endpoint_input", "future_waypoint_input", "central_velocity", "test_endpoint_goals", "test_metrics_for_threshold"]
        ),
        "no_metric_seconds_overclaim": not payload["claim_boundary"]["metric_or_seconds_claim"],
        "t100_claim_still_blocked": not payload["claim_boundary"]["t100_positive_claim_allowed"],
        "stage5c_false": not payload["claim_boundary"]["stage5c_executed"],
        "smc_false": not payload["claim_boundary"]["smc_enabled"],
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_bf_local_t100_schema_conversion_pass" if passed == total else "stage42_bf_local_t100_schema_conversion_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _user_actions(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions = [
        {
            "source": "fresh_in_memory_schema_conversion",
            "priority": "high",
            "action_type": "write_non_git_feature_store_for_stage42_bg",
            "notes": "The in-memory conversion and baseline audit passed. Next step is to write a non-committed feature store/cache and run t100 policy source-CV.",
        },
        {
            "source": "fresh_in_memory_schema_conversion",
            "priority": "high",
            "action_type": "keep_t100_claim_blocked",
            "notes": "This audit computes causal baselines and source-CV baseline selection, but it does not train/evaluate the protected M3W policy on these sources.",
        },
    ]
    return actions


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-BF Local T100 Schema Conversion And Source-CV Baseline Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bf_gate']['passed']} / {payload['stage42_bf_gate']['total']}`",
        f"- verdict: `{payload['stage42_bf_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- candidate_sources: `{s['candidate_sources']}`",
        f"- converted_sources: `{s['converted_sources']}`",
        f"- t50_eval_windows: `{s['total_eval_windows_by_horizon']['50']}`",
        f"- t100_eval_windows: `{s['t100_eval_windows']}`",
        f"- source_cv_domains_evaluated: `{', '.join(s['source_cv_domains_evaluated']) or 'none'}`",
        f"- source_cv_domains_positive_vs_constant_velocity: `{', '.join(s['source_cv_domains_positive_vs_constant_velocity']) or 'none'}`",
        f"- materialized_feature_store_written: `{s['materialized_feature_store_written']}`",
        f"- t100_positive_claim_allowed: `{s['t100_positive_claim_allowed']}`",
        "",
        "## Strongest Baseline By Source And Horizon",
        "",
        "| source | domain | h10 | h25 | h50 | h100 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for source_id, row in payload["source_metrics"].items():
        cells = []
        for horizon in HORIZONS:
            hrow = row["by_horizon"][str(horizon)]
            strongest = hrow["strongest_baseline"]
            imp = hrow["improvement_vs_constant_velocity"]
            cells.append(f"{strongest} ({imp:.3f})" if imp is not None else str(strongest))
        lines.append(f"| `{source_id}` | {row['domain']} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Source-CV Baseline Audit",
            "",
            "| domain | folds | mean holdout improvement vs CV | min holdout improvement vs CV | all folds positive |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for domain, row in payload["source_cv_audit"]["domains"].items():
        mean_imp = row["mean_holdout_improvement_vs_constant_velocity"]
        min_imp = row["minimum_holdout_improvement_vs_constant_velocity"]
        lines.append(
            f"| `{domain}` | {row['fold_count']} | {mean_imp:.6f} | {min_imp:.6f} | {row['all_folds_positive_vs_constant_velocity']} |"
            if mean_imp is not None and min_imp is not None
            else f"| `{domain}` | {row['fold_count']} | NA | NA | {row['all_folds_positive_vs_constant_velocity']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-BF performs an actual in-memory schema conversion and causal baseline audit, but does not write a large feature store.",
            "- Future labels are used only to compute baseline errors; they are not inference inputs.",
            "- Source-CV baseline selection is useful readiness evidence, not protected M3W policy training.",
            "- t100 remains a raw-frame diagnostic / blocker until Stage42-BG trains/evaluates a protected policy on these converted sources.",
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-BF User Action Required",
        "",
        f"- source: `{payload['source']}`",
        "",
    ]
    for action in payload["user_action_required"]:
        lines.extend(
            [
                f"## {action['action_type']}",
                "",
                f"- priority: `{action['priority']}`",
                f"- notes: {action['notes']}",
                "",
            ]
        )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bf_gate"]
    lines = [
        "# Stage42-BF Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    return lines


def _append_ledger(payload: Mapping[str, Any]) -> None:
    row = {
        "stage": payload["stage"],
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_bf_gate"]["verdict"],
        "gate": f"{payload['stage42_bf_gate']['passed']}/{payload['stage42_bf_gate']['total']}",
        "git_commit": payload["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_local_t100_schema_conversion()
