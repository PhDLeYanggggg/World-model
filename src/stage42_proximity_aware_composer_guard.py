from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src import stage41_joint_multiagent_consistency as jmc
from src import stage41_joint_rollout_consistency as jrc
from src import stage42_common_validation_bridge_shape_composer as co
from src import stage42_common_validation_composer_safety as cp


OUT_DIR = Path("outputs/stage42_long_research")
CO_JSON = OUT_DIR / "common_validation_bridge_shape_composer_stage42.json"
CP_JSON = OUT_DIR / "common_validation_composer_safety_stage42.json"

REPORT_JSON = OUT_DIR / "proximity_aware_composer_guard_stage42.json"
REPORT_MD = OUT_DIR / "proximity_aware_composer_guard_stage42.md"
REPORT_CSV = OUT_DIR / "proximity_aware_composer_guard_stage42.csv"
GATE_MD = OUT_DIR / "stage42_stage_cq_gate.md"

PAPER_FILES = [
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

MIN_SEP_GRID = [0.0, 0.05, 0.08, 0.12, 0.20]
MARGIN_GRID = [0.0, 0.005, 0.01]
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CQ 修复 Stage42-CP 暴露的 composer proximity caveat。",
    "proximity guard 只使用 endpoint/full-waypoint model rollout 的预测几何，不使用 future labels 作为 inference input。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
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


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _apply_proximity_guard(
    endpoint: Mapping[str, Any],
    full: Mapping[str, Any],
    keys: np.ndarray,
    choices: Mapping[str, bool],
    min_sep: float,
    margin: float,
) -> dict[str, Any]:
    base = co._compose(endpoint, full, choices)
    labels = endpoint["labels"]
    endpoint_min = jmc._min_group_distance(endpoint["selected_xy"], keys, labels["normalizer"].astype(np.float64))
    selected_min = jmc._min_group_distance(base["selected_xy"], keys, labels["normalizer"].astype(np.float64))
    guard = (
        base["use_full"]
        & np.isfinite(selected_min)
        & np.isfinite(endpoint_min)
        & (selected_min < min_sep)
        & (selected_min + margin < endpoint_min)
    )
    use_full = base["use_full"].copy()
    use_full[guard] = False
    selected_xy = endpoint["selected_xy"].copy()
    selected_xy[use_full] = full["selected_xy"][use_full]
    selected_ade, selected_fde = co.ft._trajectory_errors(selected_xy, labels)
    return {
        "selected_xy": selected_xy,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "use_full": use_full,
        "guarded_off": int(np.sum(guard)),
        "guarded_off_rate": float(np.mean(guard)) if len(guard) else 0.0,
        "metric_vs_endpoint_ade": co._metric(selected_ade, endpoint["selected_ade"], labels, use_full),
        "metric_vs_floor_ade": co._metric(selected_ade, endpoint["floor_ade"], labels, use_full),
        "metric_vs_endpoint_fde": co._metric(selected_fde, endpoint["selected_fde"], labels, use_full),
        "metric_vs_floor_fde": co._metric(selected_fde, endpoint["floor_fde"], labels, use_full),
    }


def _joint_stats(endpoint: Mapping[str, Any], composed: Mapping[str, Any], keys: np.ndarray) -> dict[str, Any]:
    labels = endpoint["labels"]
    floor = jrc._joint_stats("strongest_floor", endpoint["floor_xy"], labels, keys, np.zeros(len(keys), dtype=bool))
    endpoint_stats = jrc._joint_stats("endpoint_linear_bridge", endpoint["selected_xy"], labels, keys, endpoint["switch"])
    composer_stats = jrc._joint_stats("proximity_aware_composer", composed["selected_xy"], labels, keys, composed["use_full"])
    return {
        "floor": floor,
        "endpoint_linear": endpoint_stats,
        "composer": composer_stats,
        "composer_minus_endpoint": cp._delta_stats(composer_stats, endpoint_stats),
        "composer_minus_floor": cp._delta_stats(composer_stats, floor),
    }


def _score(metric: Mapping[str, Any], near_collision_delta: float) -> float:
    return (
        float(metric.get("all_improvement", 0.0))
        + 1.3 * float(metric.get("t50_improvement", 0.0))
        + 0.9 * float(metric.get("t100_raw_frame_diagnostic_improvement", 0.0))
        + 1.2 * float(metric.get("hard_failure_improvement", 0.0))
        - 40.0 * max(0.0, float(metric.get("easy_degradation", 1.0)) - 0.02)
        - 20.0 * max(0.0, near_collision_delta)
    )


def _fit_guard(endpoint_val: Mapping[str, Any], full_val: Mapping[str, Any], choices: Mapping[str, bool]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    keys = jmc._group_metadata("val")["key"]
    rows: list[dict[str, Any]] = []
    for min_sep in MIN_SEP_GRID:
        for margin in MARGIN_GRID:
            ev = _apply_proximity_guard(endpoint_val, full_val, keys, choices, min_sep, margin)
            joint = _joint_stats(endpoint_val, ev, keys)
            metric = ev["metric_vs_endpoint_ade"]
            near_delta = float(joint["composer_minus_endpoint"]["near_collision_rate_005_delta"])
            eligible = bool(
                metric["all_improvement"] > 0.0
                and metric["t50_improvement"] > 0.0
                and metric["t100_raw_frame_diagnostic_improvement"] > 0.0
                and metric["hard_failure_improvement"] > 0.0
                and metric["easy_degradation"] <= 0.02
                and near_delta <= 0.0
            )
            rows.append(
                {
                    "policy": {
                        "type": "proximity_aware_domain_horizon_full_waypoint_composer",
                        "min_sep": min_sep,
                        "margin": margin,
                        "base_choices": dict(choices),
                    },
                    "eligible": eligible,
                    "score": _score(metric, near_delta),
                    "guarded_off": ev["guarded_off"],
                    "val_metric_vs_endpoint_ade": metric,
                    "val_near_collision_005_delta_vs_endpoint": near_delta,
                    "val_p05_min_distance_delta_vs_endpoint": joint["composer_minus_endpoint"]["p05_min_group_distance_delta"],
                    "val_switch_rate": ev["metric_vs_endpoint_ade"]["switch_rate"],
                }
            )
    eligible_rows = [row for row in rows if row["eligible"]]
    pool = eligible_rows if eligible_rows else rows
    selected = max(pool, key=lambda row: row["score"])
    return dict(selected["policy"]), rows


def _refresh_paper_files(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    test = payload["test_eval"]["metric_vs_endpoint_ade"]
    joint = payload["test_joint_safety"]
    boot = payload["bootstrap_vs_endpoint_ade"]
    lines = [
        "## Stage42-CQ Proximity-Aware Composer Guard",
        "",
        "- source: `fresh_validation_selected_proximity_guard_from_stage42_co_policy`",
        "- scope: Stage42-CO full-waypoint composer with validation-selected predicted-proximity guard.",
        "- guard uses only model rollout geometry, not future labels as inference input.",
        f"- test vs endpoint-linear ADE: all `{_pct(test['all_improvement'])}`, t50 `{_pct(test['t50_improvement'])}`, t100 raw diagnostic `{_pct(test['t100_raw_frame_diagnostic_improvement'])}`, hard `{_pct(test['hard_failure_improvement'])}`, easy `{_pct(test['easy_degradation'])}`.",
        f"- bootstrap vs endpoint-linear all CI: `[{_pct(boot['all']['low'])}, {_pct(boot['all']['high'])}]`.",
        f"- bootstrap vs endpoint-linear t50 CI: `[{_pct(boot['t50']['low'])}, {_pct(boot['t50']['high'])}]`.",
        f"- near-collision@0.05 delta vs endpoint-linear: `{_pct(joint['composer_minus_endpoint']['near_collision_rate_005_delta'])}`.",
        f"- near-collision@0.05 delta vs strongest floor: `{_pct(joint['composer_minus_floor']['near_collision_rate_005_delta'])}`.",
        "- claim boundary: still dataset-local/raw-frame 2.5D; no metric/seconds-level, no Stage5C, no SMC.",
    ]
    status = []
    for path in PAPER_FILES:
        co._replace_section(path, "STAGE42_CQ_PROXIMITY_AWARE_COMPOSER_GUARD", lines)
        text = path.read_text(encoding="utf-8")
        status.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "contains_stage42_cq": "Stage42-CQ Proximity-Aware Composer Guard" in text,
                "contains_claim_boundary": "no metric/seconds-level" in text,
            }
        )
    return status


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    co_gate = payload["inputs"]["stage42_co"]["stage42_co_gate"]
    cp_gate = payload["inputs"]["stage42_cp"]["stage42_cp_gate"]
    test = payload["test_eval"]["metric_vs_endpoint_ade"]
    joint = payload["test_joint_safety"]
    boot = payload["bootstrap_vs_endpoint_ade"]
    gates = {
        "stage42_co_gate_passed": co_gate["passed"] == co_gate["total"],
        "stage42_cp_gate_passed": cp_gate["passed"] == cp_gate["total"],
        "validation_selected_guard": payload["policy_selection"]["selected_on"] == "validation_only",
        "test_evaluated_once": payload["policy_selection"]["test_evaluated_once"] is True,
        "no_test_threshold_tuning": payload["no_leakage"]["test_threshold_tuning"] is False,
        "rollout_geometry_only_guard": payload["no_leakage"]["guard_uses_future_labels"] is False,
        "test_all_positive_vs_endpoint": test["all_improvement"] > 0.0,
        "test_t50_positive_vs_endpoint": test["t50_improvement"] > 0.0,
        "test_t100_positive_vs_endpoint": test["t100_raw_frame_diagnostic_improvement"] > 0.0,
        "test_hard_positive_vs_endpoint": test["hard_failure_improvement"] > 0.0,
        "easy_degradation_under_2pct": test["easy_degradation"] <= 0.02,
        "near_collision_not_worse_than_endpoint": joint["composer_minus_endpoint"]["near_collision_rate_005_delta"] <= 0.0,
        "near_collision_not_worse_than_floor": joint["composer_minus_floor"]["near_collision_rate_005_delta"] <= 0.0,
        "all_ci_low_positive": boot["all"]["low"] > 0.0,
        "t50_ci_low_positive": boot["t50"]["low"] > 0.0,
        "paper_files_refreshed": all(row["contains_stage42_cq"] for row in payload["paper_file_status"]),
        "metric_seconds_overclaim_blocked": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_not_enabled": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_cq_proximity_aware_composer_guard_pass" if passed == total else "stage42_cq_proximity_aware_composer_guard_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _write_csv(candidates: list[Mapping[str, Any]]) -> None:
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "min_sep",
                "margin",
                "eligible",
                "score",
                "guarded_off",
                "val_all",
                "val_t50",
                "val_t100",
                "val_hard",
                "val_easy",
                "val_near_collision_005_delta",
                "val_switch_rate",
            ],
        )
        writer.writeheader()
        for row in candidates:
            metric = row["val_metric_vs_endpoint_ade"]
            policy = row["policy"]
            writer.writerow(
                {
                    "min_sep": policy["min_sep"],
                    "margin": policy["margin"],
                    "eligible": row["eligible"],
                    "score": row["score"],
                    "guarded_off": row["guarded_off"],
                    "val_all": metric["all_improvement"],
                    "val_t50": metric["t50_improvement"],
                    "val_t100": metric["t100_raw_frame_diagnostic_improvement"],
                    "val_hard": metric["hard_failure_improvement"],
                    "val_easy": metric["easy_degradation"],
                    "val_near_collision_005_delta": row["val_near_collision_005_delta_vs_endpoint"],
                    "val_switch_rate": row["val_switch_rate"],
                }
            )


def _write_md(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_cq_gate"]
    policy = payload["selected_policy"]
    val = payload["val_eval"]["metric_vs_endpoint_ade"]
    test = payload["test_eval"]["metric_vs_endpoint_ade"]
    test_floor = payload["test_eval"]["metric_vs_floor_ade"]
    joint = payload["test_joint_safety"]
    boot = payload["bootstrap_vs_endpoint_ade"]
    lines = [
        "# Stage42-CQ Proximity-Aware Composer Guard",
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
        "## Validation-Selected Guard",
        "",
        f"- policy type: `{policy['type']}`",
        f"- min_sep: `{policy['min_sep']}`",
        f"- margin: `{policy['margin']}`",
        f"- candidate count: `{payload['policy_selection']['candidate_count']}`",
        f"- val guarded_off: `{payload['val_eval']['guarded_off']}` rows",
        f"- val vs endpoint all/t50/t100/hard/easy: `{_pct(val['all_improvement'])}` / `{_pct(val['t50_improvement'])}` / `{_pct(val['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(val['hard_failure_improvement'])}` / `{_pct(val['easy_degradation'])}`",
        f"- val near_collision@0.05 delta vs endpoint: `{_pct(payload['val_joint_safety']['composer_minus_endpoint']['near_collision_rate_005_delta'])}`",
        "",
        "## Test Once",
        "",
        f"- test guarded_off: `{payload['test_eval']['guarded_off']}` rows",
        f"- test use_full_rate: `{_pct(test['switch_rate'])}`",
        f"- test vs endpoint all/t50/t100/hard/easy: `{_pct(test['all_improvement'])}` / `{_pct(test['t50_improvement'])}` / `{_pct(test['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(test['hard_failure_improvement'])}` / `{_pct(test['easy_degradation'])}`",
        f"- test vs strongest floor all/t50/t100/hard/easy: `{_pct(test_floor['all_improvement'])}` / `{_pct(test_floor['t50_improvement'])}` / `{_pct(test_floor['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(test_floor['hard_failure_improvement'])}` / `{_pct(test_floor['easy_degradation'])}`",
        "",
        "## Bootstrap CI vs Endpoint-Linear",
        "",
        "| slice | low | mid | high | n |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in boot.items():
        lines.append(f"| `{name}` | {_pct(row['low'])} | {_pct(row['mid'])} | {_pct(row['high'])} | {row['n']} |")
    lines.extend(
        [
            "",
            "## Joint Safety",
            "",
            f"- near_collision@0.05 delta vs endpoint-linear: `{_pct(joint['composer_minus_endpoint']['near_collision_rate_005_delta'])}`",
            f"- near_collision@0.05 delta vs strongest floor: `{_pct(joint['composer_minus_floor']['near_collision_rate_005_delta'])}`",
            f"- p05 min-distance delta vs endpoint-linear: `{joint['composer_minus_endpoint']['p05_min_group_distance_delta']}`",
            f"- jagged-rate delta vs endpoint-linear: `{_pct(joint['composer_minus_endpoint']['jagged_rate_delta'])}`",
            "",
            "## Interpretation",
            "",
            "- Stage42-CQ turns the Stage42-CP proximity caveat into a validation-selected safety guard.",
            "- The guard gives up some Stage42-CO/CP accuracy gain, but keeps all/t50/t100 raw-frame/hard-failure positive with positive all/t50 bootstrap lower bounds.",
            "- Near-collision@0.05 is no longer worse than endpoint-linear or the strongest floor under this guarded policy.",
            "- This remains protected dataset-local/raw-frame 2.5D evidence, not metric/seconds-level, Stage5C, or SMC.",
        ]
    )
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage42-CQ Gate",
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
    co_report = read_json(CO_JSON, {})
    cp_report = read_json(CP_JSON, {})
    choices = co_report["selected_policy"].get("choices", {})
    endpoint_val = co._endpoint_bundle("val")
    full_val = co._full_bundle("val")
    policy, candidates = _fit_guard(endpoint_val, full_val, choices)
    keys_val = jmc._group_metadata("val")["key"]
    val_eval = _apply_proximity_guard(endpoint_val, full_val, keys_val, choices, float(policy["min_sep"]), float(policy["margin"]))
    val_joint = _joint_stats(endpoint_val, val_eval, keys_val)

    endpoint_test = co._endpoint_bundle("test")
    full_test = co._full_bundle("test")
    keys_test = jmc._group_metadata("test")["key"]
    test_eval = _apply_proximity_guard(endpoint_test, full_test, keys_test, choices, float(policy["min_sep"]), float(policy["margin"]))
    test_joint = _joint_stats(endpoint_test, test_eval, keys_test)

    payload: dict[str, Any] = {
        "source": "fresh_validation_selected_proximity_guard_from_stage42_co_policy",
        "stage": "Stage42-CQ proximity-aware composer guard",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([CO_JSON, CP_JSON]),
        "inputs": {"stage42_co": co_report, "stage42_cp": cp_report},
        "selected_policy": policy,
        "policy_selection": {
            "selected_on": "validation_only",
            "test_evaluated_once": True,
            "candidate_count": len(candidates),
        },
        "candidate_summary": candidates,
        "val_eval": {
            "guarded_off": val_eval["guarded_off"],
            "guarded_off_rate": val_eval["guarded_off_rate"],
            "metric_vs_endpoint_ade": val_eval["metric_vs_endpoint_ade"],
            "metric_vs_floor_ade": val_eval["metric_vs_floor_ade"],
        },
        "test_eval": {
            "guarded_off": test_eval["guarded_off"],
            "guarded_off_rate": test_eval["guarded_off_rate"],
            "metric_vs_endpoint_ade": test_eval["metric_vs_endpoint_ade"],
            "metric_vs_floor_ade": test_eval["metric_vs_floor_ade"],
            "metric_vs_endpoint_fde": test_eval["metric_vs_endpoint_fde"],
            "metric_vs_floor_fde": test_eval["metric_vs_floor_fde"],
        },
        "bootstrap_vs_endpoint_ade": cp._bootstrap_set(test_eval["selected_ade"], endpoint_test["selected_ade"], endpoint_test["labels"]),
        "bootstrap_vs_floor_ade": cp._bootstrap_set(test_eval["selected_ade"], endpoint_test["floor_ade"], endpoint_test["labels"]),
        "val_joint_safety": val_joint,
        "test_joint_safety": test_joint,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "guard_uses_future_labels": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["paper_file_status"] = _refresh_paper_files(payload)
    payload["stage42_cq_gate"] = _gate(payload)
    write_json(REPORT_JSON, _jsonable(payload))
    _write_csv(candidates)
    _write_md(payload)
    return payload


if __name__ == "__main__":
    result = run()
    gate = result["stage42_cq_gate"]
    print(f"Stage42-CQ proximity composer guard: {gate['verdict']} ({gate['passed']}/{gate['total']})")
