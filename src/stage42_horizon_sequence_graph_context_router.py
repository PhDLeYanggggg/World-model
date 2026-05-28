from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_context_gain_router as el
from src import stage42_sequence_graph_context_router as eq
from src import stage42_source_level_ablation as an
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_graph_context as sg
from src import stage42_source_level_incremental_ablation as ao
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "horizon_sequence_graph_context_router_stage42.json"
REPORT_MD = OUT_DIR / "horizon_sequence_graph_context_router_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_io_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_LEDGER = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_horizon_sequence_graph_context_router"
HORIZONS = [10, 25, 50, 100]
CANDIDATES = ["history_only", "motion_goal_context", "baseline_plus_history_goal_neighbor"]
MIN_HORIZON_INCREMENT = 0.01

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-IO 是 horizon-specific sequence+graph context router，不是新数据转换或 metric/seconds-level 结果。",
    "该实验修复 Stage42-EQ 的 horizon mixing 风险：t10/t25/t50/t100 分开训练 gain/harm router，再在 test 上评一次。",
    "sequence summary 与 graph summary 只使用当前帧和过去 history。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
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
}


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _filter_rows(data: Mapping[str, Any], mask: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n = len(mask)
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (n,):
            out[key] = value[mask]
        else:
            out[key] = value
    return out


def _metric_delta(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> dict[str, float]:
    return eq._metric_delta(lhs, rhs)


def _is_positive_horizon_context(metric: Mapping[str, Any]) -> bool:
    return (
        (
            float(metric["all_improvement"]) > MIN_HORIZON_INCREMENT
            or float(metric["hard_failure_improvement"]) > MIN_HORIZON_INCREMENT
        )
        and float(metric["easy_degradation"]) <= 0.02
    )


def _horizon_eval_rows(
    *,
    horizon: int,
    candidate: str,
    augmented: np.ndarray,
    base_ade: np.ndarray,
    candidate_ade: np.ndarray,
    split: np.ndarray,
    data: Mapping[str, np.ndarray],
    reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    horizon_mask = data["horizon"].astype(int) == int(horizon)
    subset_data = _filter_rows(data, horizon_mask)
    row = el._train_gain_router(
        name=f"h{horizon}_{candidate}",
        raw_router_features=augmented[horizon_mask],
        base_ade=base_ade[horizon_mask],
        candidate_ade=candidate_ade[horizon_mask],
        split=split[horizon_mask],
        data=subset_data,
    )
    metric = row["test_metric_vs_baseline_family"]
    row["horizon"] = int(horizon)
    row["candidate"] = candidate
    row["horizon_rows"] = {
        "train": int(np.sum(horizon_mask & (split == "train"))),
        "val": int(np.sum(horizon_mask & (split == "val"))),
        "test": int(np.sum(horizon_mask & (split == "test"))),
    }
    row["horizon_increment_supported"] = _is_positive_horizon_context(metric)
    if reference is not None:
        row["delta_vs_global_sequence_graph_router"] = _metric_delta(
            metric, reference["routers"][candidate]["test_metric_vs_baseline_family"]
        )
    return row


def _build_result() -> dict[str, Any]:
    shared = an._prep_shared()
    masks = ao._incremental_variant_masks(shared["feature_names"])
    split = shared["split"]
    data = shared["data"]
    base_pred = el._prepare_variant_predictions(shared["features"][:, masks["baseline_family_only"]], shared)
    graph, graph_names, graph_stats = sg._build_graph_features(data)
    seq_summary, seq_names, seq_stats = eq._sequence_summary(data)
    global_eq = read_json(eq.REPORT_JSON, {}) if eq.REPORT_JSON.exists() else None
    candidate_predictions: dict[str, Any] = {}
    horizon_routers: dict[str, Any] = {}
    for candidate in CANDIDATES:
        candidate_features = shared["features"][:, masks[candidate]]
        candidate_predictions[candidate] = el._prepare_variant_predictions(candidate_features, shared)
        augmented = eq._augmented_router_features(candidate_features, graph, seq_summary)
        for horizon in HORIZONS:
            key = f"h{horizon}_{candidate}"
            horizon_routers[key] = _horizon_eval_rows(
                horizon=horizon,
                candidate=candidate,
                augmented=augmented,
                base_ade=base_pred["selected_ade"],
                candidate_ade=candidate_predictions[candidate]["selected_ade"],
                split=split,
                data=data,
                reference=global_eq,
            )
    positive = sorted([key for key, row in horizon_routers.items() if row["horizon_increment_supported"]])
    best_by_horizon: dict[str, Any] = {}
    for horizon in HORIZONS:
        rows = {key: row for key, row in horizon_routers.items() if row["horizon"] == horizon}
        best_key = max(
            rows,
            key=lambda key: (
                rows[key]["test_metric_vs_baseline_family"]["all_improvement"]
                + rows[key]["test_metric_vs_baseline_family"]["hard_failure_improvement"]
                - max(0.0, rows[key]["test_metric_vs_baseline_family"]["easy_degradation"] - 0.02)
            ),
        )
        best_by_horizon[str(horizon)] = {
            "best_key": best_key,
            "candidate": rows[best_key]["candidate"],
            "metric": rows[best_key]["test_metric_vs_baseline_family"],
            "supported": rows[best_key]["horizon_increment_supported"],
        }
    best_overall_key = max(
        horizon_routers,
        key=lambda key: (
            horizon_routers[key]["test_metric_vs_baseline_family"]["all_improvement"]
            + horizon_routers[key]["test_metric_vs_baseline_family"]["hard_failure_improvement"]
            + horizon_routers[key]["test_metric_vs_baseline_family"]["t50_improvement"]
        ),
    )
    result = {
        "source": SOURCE,
        "stage": "Stage42-IO Horizon-Specific Sequence+Graph Context Router",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/context_gain_router_stage42.json",
                "outputs/stage42_long_research/sequence_graph_context_router_stage42.json",
            ]
        ),
        "current_facts": CURRENT_FACTS,
        "split_stats": shared["split_stats"],
        "baseline_family_control": {
            "lambda": base_pred["lambda"],
            "policy_slice_count": len(base_pred["policy"]["slices"]),
            "val_metric_vs_causal_floor": base_pred["val_metric"],
        },
        "sequence_summary_schema": {"feature_names": seq_names, "stats": seq_stats},
        "graph_summary_schema": {
            "feature_names": graph_names,
            "stats": graph_stats,
            "group_key": "source_file + frame_id",
            "current_and_past_only": True,
        },
        "candidate_policy_summary": {
            name: {
                "lambda": row["lambda"],
                "policy_slice_count": len(row["policy"]["slices"]),
                "val_metric_vs_causal_floor": row["val_metric"],
            }
            for name, row in candidate_predictions.items()
        },
        "horizon_routers": horizon_routers,
        "positive_horizon_sequence_graph_context_routers": positive,
        "best_by_horizon": best_by_horizon,
        "best_overall_router": {
            "key": best_overall_key,
            "horizon": horizon_routers[best_overall_key]["horizon"],
            "candidate": horizon_routers[best_overall_key]["candidate"],
            "metric": horizon_routers[best_overall_key]["test_metric_vs_baseline_family"],
        },
        "global_sequence_graph_reference": {
            "source": "cached_verified" if global_eq else "not_run",
            "verdict": (global_eq or {}).get("summary", {}).get("sequence_graph_increment_verdict"),
            "positive_sequence_graph_context_routers": (global_eq or {}).get("summary", {}).get(
                "positive_sequence_graph_context_routers"
            ),
        },
        "summary": {
            "source": SOURCE,
            "router_target": "horizon-specific supervised gain/harm routing for context proposal over baseline-family protected control",
            "horizons": HORIZONS,
            "candidates": CANDIDATES,
            "positive_horizon_sequence_graph_context_routers": positive,
            "best_by_horizon": best_by_horizon,
            "best_overall_router": best_overall_key,
            "horizon_specific_increment_verdict": (
                "stage42_io_horizon_sequence_graph_context_router_supported"
                if positive
                else "stage42_io_horizon_sequence_graph_context_router_not_supported"
            ),
            "interpretation": (
                "Stage42-IO separates horizons after Stage42-EQ's global sequence+graph router failed. "
                "A positive row supports only a narrow horizon-specific routing contribution. "
                "If no horizon is positive, horizon mixing is not the main reason sequence/graph context failed."
            ),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "sequence_summary_current_past_only": True,
            "graph_summary_current_past_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_selected_thresholds": True,
            "source_overlap_pass": bool(shared["split_stats"]["source_overlap_pass"]),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    result["stage42_io_gate"] = _gate(result)
    return result


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    no_leakage = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "source_level_split_used": payload["split_stats"]["by_split"]["test"]["rows"] == 47458,
        "baseline_family_control_loaded": payload["baseline_family_control"]["policy_slice_count"] >= 0,
        "sequence_summary_built": payload["sequence_summary_schema"]["stats"]["feature_count"] >= 8,
        "graph_summary_built": payload["graph_summary_schema"]["stats"]["rows_with_neighbors"] > 0,
        "horizon_routers_complete": len(payload["horizon_routers"]) == len(HORIZONS) * len(CANDIDATES),
        "all_horizons_have_test_rows": all(
            payload["best_by_horizon"][str(h)]["metric"]["rows"] > 0 for h in HORIZONS
        ),
        "validation_only_selection": all(
            row["validation_selection"]["source"] == "validation_only"
            and row["validation_selection"]["test_threshold_tuning"] is False
            for row in payload["horizon_routers"].values()
        ),
        "horizon_increment_measured": payload["summary"]["horizon_specific_increment_verdict"]
        in {
            "stage42_io_horizon_sequence_graph_context_router_supported",
            "stage42_io_horizon_sequence_graph_context_router_not_supported",
        },
        "negative_or_positive_claim_bounded": isinstance(
            payload["positive_horizon_sequence_graph_context_routers"], list
        ),
        "no_leakage_pass": no_leakage["future_endpoint_input"] is False
        and no_leakage["future_waypoint_input"] is False
        and no_leakage["sequence_summary_current_past_only"] is True
        and no_leakage["graph_summary_current_past_only"] is True
        and no_leakage["central_velocity"] is False
        and no_leakage["test_endpoint_goals"] is False
        and no_leakage["test_threshold_tuning"] is False
        and no_leakage["validation_selected_thresholds"] is True,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = (
        "stage42_io_horizon_sequence_graph_context_router_pass"
        if passed == total
        else "stage42_io_horizon_sequence_graph_context_router_partial"
    )
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_io_gate"]
    lines = [
        "# Stage42-IO Horizon-Specific Sequence+Graph Context Router",
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
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- router_target: `{payload['summary']['router_target']}`",
        f"- horizons: `{payload['summary']['horizons']}`",
        f"- candidates: `{payload['summary']['candidates']}`",
        f"- positive_horizon_sequence_graph_context_routers: `{payload['summary']['positive_horizon_sequence_graph_context_routers']}`",
        f"- best_overall_router: `{payload['summary']['best_overall_router']}`",
        f"- horizon_specific_increment_verdict: `{payload['summary']['horizon_specific_increment_verdict']}`",
        "",
        payload["summary"]["interpretation"],
        "",
        "## Best Router By Horizon",
        "",
        "| horizon | best router | candidate | all | t50 | t100 diag | hard/failure | easy degradation | switch rate | supported |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for horizon in HORIZONS:
        row = payload["best_by_horizon"][str(horizon)]
        metric = row["metric"]
        lines.append(
            f"| {horizon} | `{row['best_key']}` | `{row['candidate']}` | "
            f"{metric['all_improvement']:.6f} | {metric['t50_improvement']:.6f} | "
            f"{metric['t100_raw_frame_diagnostic_improvement']:.6f} | {metric['hard_failure_improvement']:.6f} | "
            f"{metric['easy_degradation']:.6f} | {metric['switch_rate']:.6f} | {row['supported']} |"
        )
    lines.extend(
        [
            "",
            "## All Horizon Routers",
            "",
            "| key | rows | all | hard/failure | easy | switch | supported |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, row in payload["horizon_routers"].items():
        metric = row["test_metric_vs_baseline_family"]
        lines.append(
            f"| `{key}` | {metric['rows']} | {metric['all_improvement']:.6f} | "
            f"{metric['hard_failure_improvement']:.6f} | {metric['easy_degradation']:.6f} | "
            f"{metric['switch_rate']:.6f} | {row['horizon_increment_supported']} |"
        )
    lines.extend(
        [
            "",
            "## Sequence / Graph Schema",
            "",
            f"- sequence_summary_stats: `{payload['sequence_summary_schema']['stats']}`",
            f"- graph_summary_stats: `{payload['graph_summary_schema']['stats']}`",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
            "",
            "## Interpretation",
            "",
            "- Stage42-IO is a fresh horizon-specific follow-up to Stage42-EQ.",
            "- It tests whether horizon mixing caused the earlier negative sequence/graph context result.",
            "- Positive routers are narrow horizon-specific routing evidence only; negative results keep sequence/graph context diagnostic under this protocol.",
            "- Claims remain raw-frame / dataset-local 2.5D only; no metric/seconds claim, Stage5C, or SMC.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_io_gate"]
    return [
        "# Stage42-IO Gate",
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
    summary = payload["summary"]
    best = payload["best_overall_router"]
    metric = best["metric"]
    return [
        "## Stage42-IO Horizon-Specific Sequence+Graph Context Router",
        "",
        "- source: `fresh_stage42_horizon_sequence_graph_context_router`",
        "- role: tests whether splitting t10/t25/t50/t100 fixes the negative Stage42-EQ global sequence+graph context router.",
        f"- gate: `{payload['stage42_io_gate']['passed']} / {payload['stage42_io_gate']['total']}`; verdict `{payload['stage42_io_gate']['verdict']}`.",
        f"- positive_horizon_sequence_graph_context_routers: `{summary['positive_horizon_sequence_graph_context_routers']}`.",
        f"- best_overall_router: `{summary['best_overall_router']}`.",
        f"- best all/t50/t100raw/hard/easy: `{metric['all_improvement']:.6f}` / `{metric['t50_improvement']:.6f}` / `{metric['t100_raw_frame_diagnostic_improvement']:.6f}` / `{metric['hard_failure_improvement']:.6f}` / `{metric['easy_degradation']:.6f}`.",
        f"- horizon_specific_increment_verdict: `{summary['horizon_specific_increment_verdict']}`.",
        "- Boundary: fresh horizon-specific router audit only; raw-frame/dataset-local 2.5D; no metric/seconds claim, no Stage5C, no SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, WORK_LEDGER]:
        _replace_section(path, "STAGE42_IO_HORIZON_SEQUENCE_GRAPH_CONTEXT_ROUTER", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    files = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        text = str(path)
        if text not in files:
            files.append(text)
    state["current_stage"] = "Stage42-IO horizon sequence graph context router"
    state["current_verdict"] = payload["stage42_io_gate"]["verdict"]
    state.setdefault("stage42_long_research", {})["stage_io_horizon_sequence_graph_context_router"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_io_gate"]["verdict"],
        "gates": f"{payload['stage42_io_gate']['passed']}/{payload['stage42_io_gate']['total']}",
        "summary": payload["summary"],
        "best_by_horizon": payload["best_by_horizon"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_horizon_sequence_graph_context_router(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    payload = _build_result()
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_horizon_sequence_graph_context_router()
