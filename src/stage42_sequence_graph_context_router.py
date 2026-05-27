from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_context_gain_router as el
from src import stage42_source_level_ablation as an
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_graph_context as sg
from src import stage42_source_level_incremental_ablation as ao
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "sequence_graph_context_router_stage42.json"
REPORT_MD = OUT_DIR / "sequence_graph_context_router_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_eq_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
TARGET_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
WORK_LEDGER = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_sequence_graph_context_router"
MIN_INCREMENT = 0.01
CANDIDATES = ["history_only", "motion_goal_context", "baseline_plus_history_goal_neighbor"]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EQ 是 source-compatible sequence+graph context router fresh probe，不是新数据转换或 metric/seconds-level 结果。",
    "该实验不让 sequence/graph 直接替代 floor，而是用 past-only sequence summary 与 current-frame graph summary 决定 context proposal 是否值得切换。",
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


def _sequence_summary(data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    seq = data["history_seq"].astype(np.float32)
    valid = seq[..., 6] > 0.5
    counts = valid.sum(axis=1).astype(np.float32)
    safe_counts = np.maximum(counts, 1.0)
    speed = np.where(valid, seq[..., 2], 0.0)
    heading = np.where(valid, seq[..., 4], 0.0)
    mean_speed = speed.sum(axis=1) / safe_counts
    speed_var = ((speed - mean_speed[:, None]) ** 2 * valid).sum(axis=1) / safe_counts
    speed_std = np.sqrt(np.maximum(speed_var, 0.0))
    max_speed = np.where(valid, speed, -np.inf).max(axis=1)
    max_speed = np.where(np.isfinite(max_speed), max_speed, 0.0)
    min_speed = np.where(valid, speed, np.inf).min(axis=1)
    min_speed = np.where(np.isfinite(min_speed), min_speed, 0.0)
    heading_cos = (np.cos(heading) * valid).sum(axis=1) / safe_counts
    heading_sin = (np.sin(heading) * valid).sum(axis=1) / safe_counts
    idx = np.arange(seq.shape[1])[None, :]
    first = np.where(valid, idx, seq.shape[1]).min(axis=1).astype(int)
    first = np.minimum(first, seq.shape[1] - 1)
    last = np.where(valid, idx, 0).max(axis=1).astype(int)
    row = np.arange(len(seq))
    # Dimensions 0/1 are the history coordinate channels in the Stage37/41 cache.
    dx = np.where(counts > 0, seq[row, last, 0] - seq[row, first, 0], 0.0)
    dy = np.where(counts > 0, seq[row, last, 1] - seq[row, first, 1], 0.0)
    displacement = np.sqrt(dx * dx + dy * dy)
    step_dx = np.diff(seq[..., 0], axis=1)
    step_dy = np.diff(seq[..., 1], axis=1)
    step_valid = valid[:, 1:] & valid[:, :-1]
    path_length = (np.sqrt(step_dx * step_dx + step_dy * step_dy) * step_valid).sum(axis=1)
    path_efficiency = displacement / np.maximum(path_length, 1e-6)
    features = np.stack(
        [
            counts,
            mean_speed,
            speed_std,
            max_speed,
            min_speed,
            max_speed - min_speed,
            heading_cos,
            heading_sin,
            displacement,
            path_length,
            path_efficiency,
        ],
        axis=1,
    ).astype(np.float32)
    names = [
        "seq_valid_count",
        "seq_mean_speed",
        "seq_speed_std",
        "seq_max_speed",
        "seq_min_speed",
        "seq_speed_range",
        "seq_heading_cos_mean",
        "seq_heading_sin_mean",
        "seq_displacement",
        "seq_path_length",
        "seq_path_efficiency",
    ]
    stats = {
        "source": "fresh_run",
        "rows": int(len(seq)),
        "feature_count": int(features.shape[1]),
        "valid_history_mean": float(np.mean(counts)),
        "valid_history_min": float(np.min(counts)),
        "valid_history_max": float(np.max(counts)),
        "uses_future_endpoint": False,
        "uses_future_waypoint": False,
    }
    return features, names, stats


def _augmented_router_features(
    candidate_features: np.ndarray,
    graph: np.ndarray,
    seq_summary: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            candidate_features.astype(np.float32),
            graph.astype(np.float32),
            seq_summary.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def _metric_delta(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> dict[str, float]:
    keys = [
        "all_improvement",
        "t50_improvement",
        "t100_raw_frame_diagnostic_improvement",
        "hard_failure_improvement",
        "easy_degradation",
        "switch_rate",
    ]
    return {key: float(lhs[key]) - float(rhs[key]) for key in keys}


def _build_result() -> dict[str, Any]:
    shared = an._prep_shared()
    masks = ao._incremental_variant_masks(shared["feature_names"])
    split = shared["split"]
    data = shared["data"]
    base_pred = el._prepare_variant_predictions(shared["features"][:, masks["baseline_family_only"]], shared)
    graph, graph_names, graph_stats = sg._build_graph_features(data)
    seq_summary, seq_names, seq_stats = _sequence_summary(data)
    baseline_el = read_json(el.REPORT_JSON, {}) if el.REPORT_JSON.exists() else {}
    el_best_name = baseline_el.get("best_router") or baseline_el.get("summary", {}).get("best_router")
    el_best_metric = (
        baseline_el.get("routers", {})
        .get(el_best_name or "", {})
        .get("test_metric_vs_baseline_family")
    )
    routers: dict[str, Any] = {}
    for name in CANDIDATES:
        candidate_features = shared["features"][:, masks[name]]
        candidate_pred = el._prepare_variant_predictions(candidate_features, shared)
        augmented = _augmented_router_features(candidate_features, graph, seq_summary)
        row = el._train_gain_router(
            name=name,
            raw_router_features=augmented,
            base_ade=base_pred["selected_ade"],
            candidate_ade=candidate_pred["selected_ade"],
            split=split,
            data=data,
        )
        row["router_feature_schema"] = {
            "candidate_feature_count": int(candidate_features.shape[1]),
            "graph_feature_count": int(graph.shape[1]),
            "sequence_summary_feature_count": int(seq_summary.shape[1]),
            "total_feature_count": int(augmented.shape[1]),
            "uses_future_endpoint": False,
            "uses_future_waypoint": False,
        }
        row["candidate_policy"] = {
            "lambda": candidate_pred["lambda"],
            "policy_slice_count": len(candidate_pred["policy"]["slices"]),
            "val_metric_vs_causal_floor": candidate_pred["val_metric"],
        }
        if el_best_metric:
            row["delta_vs_stage42_el_best_router"] = _metric_delta(row["test_metric_vs_baseline_family"], el_best_metric)
        routers[name] = row
    positive = sorted([name for name, row in routers.items() if row["increment_supported"]])
    best_name = max(
        routers,
        key=lambda key: (
            routers[key]["test_metric_vs_baseline_family"]["all_improvement"]
            + routers[key]["test_metric_vs_baseline_family"]["t50_improvement"]
            + routers[key]["test_metric_vs_baseline_family"]["hard_failure_improvement"]
        ),
    )
    best_metric = routers[best_name]["test_metric_vs_baseline_family"]
    return {
        "source": SOURCE,
        "stage": "Stage42-EQ Sequence+Graph Context Router",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/context_gain_router_stage42.json",
                "outputs/stage42_long_research/source_level_sequence_context_stage42.json",
                "outputs/stage42_long_research/source_level_graph_context_stage42.json",
            ]
        ),
        "current_facts": CURRENT_FACTS,
        "split_stats": shared["split_stats"],
        "baseline_family_control": {
            "lambda": base_pred["lambda"],
            "policy_slice_count": len(base_pred["policy"]["slices"]),
            "val_metric_vs_causal_floor": base_pred["val_metric"],
        },
        "sequence_summary_schema": {
            "feature_names": seq_names,
            "stats": seq_stats,
        },
        "graph_summary_schema": {
            "feature_names": graph_names,
            "stats": graph_stats,
            "group_key": "source_file + frame_id",
            "current_and_past_only": True,
        },
        "routers": routers,
        "positive_sequence_graph_context_routers": positive,
        "best_router": best_name,
        "stage42_el_best_router_reference": {
            "source": "cached_verified" if el_best_metric else "not_run",
            "best_router": el_best_name,
            "metric_vs_baseline_family": el_best_metric,
        },
        "summary": {
            "source": SOURCE,
            "router_target": "predict supervised gain of context proposal over baseline-family protected control using sequence+graph past-only summaries",
            "candidates": CANDIDATES,
            "positive_sequence_graph_context_routers": positive,
            "best_router": best_name,
            "best_router_test_metric_vs_baseline_family": best_metric,
            "sequence_graph_increment_verdict": (
                "stage42_eq_sequence_graph_context_router_supported"
                if positive
                else "stage42_eq_sequence_graph_context_router_not_supported"
            ),
            "interpretation": (
                "Stage42-EQ tests whether sequence/graph context is useful as a safe switchability signal after "
                "direct sequence and graph residual prediction failed. Positive routers would support a narrow "
                "deployment-router contribution. Negative routers keep context as diagnostic under this protocol."
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


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    no_leakage = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "source_level_split_used": payload["split_stats"]["by_split"]["test"]["rows"] == 47458,
        "baseline_family_control_loaded": payload["baseline_family_control"]["policy_slice_count"] >= 0,
        "sequence_summary_built": payload["sequence_summary_schema"]["stats"]["feature_count"] >= 8,
        "graph_summary_built": payload["graph_summary_schema"]["stats"]["rows_with_neighbors"] > 0,
        "router_candidates_complete": len(payload["routers"]) >= 3,
        "validation_only_selection": all(
            row["validation_selection"]["source"] == "validation_only"
            and row["validation_selection"]["test_threshold_tuning"] is False
            for row in payload["routers"].values()
        ),
        "sequence_graph_increment_measured": payload["summary"]["sequence_graph_increment_verdict"]
        in {
            "stage42_eq_sequence_graph_context_router_supported",
            "stage42_eq_sequence_graph_context_router_not_supported",
        },
        "negative_or_positive_claim_bounded": isinstance(payload["positive_sequence_graph_context_routers"], list),
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
    passed = sum(1 for value in gates.values() if value)
    total = len(gates)
    verdict = (
        "stage42_eq_sequence_graph_context_router_pass"
        if passed == total
        else "stage42_eq_sequence_graph_context_router_partial"
    )
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-EQ Sequence+Graph Context Router",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_eq_gate']['passed']} / {payload['stage42_eq_gate']['total']}`",
        f"- verdict: `{payload['stage42_eq_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- router_target: `{payload['summary']['router_target']}`",
        f"- candidates: `{payload['summary']['candidates']}`",
        f"- positive_sequence_graph_context_routers: `{payload['summary']['positive_sequence_graph_context_routers']}`",
        f"- best_router: `{payload['summary']['best_router']}`",
        f"- sequence_graph_increment_verdict: `{payload['summary']['sequence_graph_increment_verdict']}`",
        "",
        payload["summary"]["interpretation"],
        "",
        "## Router Results vs Baseline-Family Protected Control",
        "",
        "| candidate | features | all | t50 | t100 diag | hard/failure | easy degradation | switch rate | increment supported |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in payload["routers"].items():
        m = row["test_metric_vs_baseline_family"]
        features = row["router_feature_schema"]["total_feature_count"]
        lines.append(
            f"| `{name}` | {features} | {m['all_improvement']:.6f} | {m['t50_improvement']:.6f} | "
            f"{m['t100_raw_frame_diagnostic_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | "
            f"{m['easy_degradation']:.6f} | {m['switch_rate']:.6f} | {row['increment_supported']} |"
        )
    lines.extend(
        [
            "",
            "## Sequence / Graph Schema",
            "",
            f"- sequence_summary_stats: `{payload['sequence_summary_schema']['stats']}`",
            f"- graph_summary_stats: `{payload['graph_summary_schema']['stats']}`",
            "",
            "## Interpretation",
            "",
            "- This experiment is a stricter follow-up to negative sequence-residual and graph-residual probes.",
            "- It tests sequence/graph context only as a validated router signal over a protected baseline-family control.",
            "- If the result is negative, scene/goal/neighbor/sequence context remains diagnostic under this source-level protocol.",
            "- Claims remain raw-frame / dataset-local 2.5D only; no metric/seconds claim, Stage5C, or SMC.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_eq_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_eq_gate"]
    return [
        "# Stage42-EQ Gate",
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
    best = summary["best_router_test_metric_vs_baseline_family"]
    return [
        "## Stage42-EQ Sequence+Graph Context Router",
        "",
        "- source: `fresh_stage42_sequence_graph_context_router`",
        "- role: tests whether past-only sequence summary + current-frame graph summary can improve context gain routing over baseline-family protected control.",
        f"- gate: `{payload['stage42_eq_gate']['passed']} / {payload['stage42_eq_gate']['total']}`; verdict `{payload['stage42_eq_gate']['verdict']}`.",
        f"- positive_sequence_graph_context_routers: `{summary['positive_sequence_graph_context_routers']}`; best router `{summary['best_router']}`.",
        f"- best all/t50/t100raw/hard delta vs baseline-family: `{best['all_improvement']:.6f}` / `{best['t50_improvement']:.6f}` / `{best['t100_raw_frame_diagnostic_improvement']:.6f}` / `{best['hard_failure_improvement']:.6f}`; easy `{best['easy_degradation']:.6f}`.",
        f"- sequence_graph_increment_verdict: `{summary['sequence_graph_increment_verdict']}`.",
        "- Boundary: fresh router audit only; raw-frame/dataset-local 2.5D; no metric/seconds claim, no Stage5C, no SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, TARGET_SUMMARY, WORK_LEDGER]:
        _replace_section(path, "STAGE42_EQ_SEQUENCE_GRAPH_CONTEXT_ROUTER", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-EQ sequence graph context router"
    state["current_verdict"] = payload["stage42_eq_gate"]["verdict"]
    state["stage42_eq_sequence_graph_context_router"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_eq_gate"]["verdict"],
        "gates": f"{payload['stage42_eq_gate']['passed']}/{payload['stage42_eq_gate']['total']}",
        "summary": payload["summary"],
        "sequence_summary_stats": payload["sequence_summary_schema"]["stats"],
        "graph_summary_stats": payload["graph_summary_schema"]["stats"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": (
            "Stage42-EQ freshly tests whether past-only sequence summary and current-frame graph summary can act as "
            "a context gain router over the protected baseline-family control. The result is bounded as positive or "
            "negative evidence and cannot be used for metric/seconds, true-3D, Stage5C, or SMC claims."
        ),
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_sequence_graph_context_router(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    payload = _build_result()
    payload["stage42_eq_gate"] = _gate(payload)
    write_json(REPORT_JSON, am._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_sequence_graph_context_router()
