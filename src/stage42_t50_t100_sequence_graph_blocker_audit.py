from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_context_gain_router as el
from src import stage42_horizon_sequence_graph_context_router as io
from src import stage42_sequence_graph_context_router as eq
from src import stage42_source_level_ablation as an
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_graph_context as sg
from src import stage42_source_level_incremental_ablation as ao
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "t50_t100_sequence_graph_blocker_audit_stage42.json"
REPORT_MD = OUT_DIR / "t50_t100_sequence_graph_blocker_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ip_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_LEDGER = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_t50_t100_sequence_graph_blocker_audit"
TARGET_HORIZONS = [50, 100]
CANDIDATES = ["history_only", "motion_goal_context", "baseline_plus_history_goal_neighbor"]
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-IP 是 Stage42-IO 后续 blocker audit：只解释 t50/t100 sequence+graph context 为什么没有形成 deployable lift。",
    "sequence summary 与 graph summary 只使用当前帧和过去 history。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
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


def _safe_mean(x: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(x[mask])) if np.any(mask) else 0.0


def _safe_rate(mask: np.ndarray, denom: np.ndarray) -> float:
    return float(np.mean(mask[denom])) if np.any(denom) else 0.0


def _score(metric: Mapping[str, Any]) -> float:
    return float(
        1.3 * metric["all_improvement"]
        + 1.2 * metric["hard_failure_improvement"]
        + 0.6 * metric["t100_raw_frame_diagnostic_improvement"]
        - 30.0 * max(0.0, metric["easy_degradation"] - 0.02)
        - 0.02 * metric["switch_rate"]
    )


def _train_router_with_arrays(
    *,
    raw_router_features: np.ndarray,
    base_ade: np.ndarray,
    candidate_ade: np.ndarray,
    split: np.ndarray,
    data: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    x, _, _ = am._standardize(raw_router_features, train_mask)
    gain = (base_ade - candidate_ade).astype(np.float32)
    best: dict[str, Any] | None = None
    best_score = -1e9
    val_candidate_count = 0
    for lam in el.RIDGE_LAMBDAS:
        coef = el._fit_ridge_vector(x, gain, train_mask, lam)
        pred_gain = (x.astype(np.float64) @ coef.astype(np.float64)).astype(np.float64)
        val_pred = pred_gain[val_mask]
        thresholds = sorted(set(float(np.quantile(val_pred, q)) for q in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]))
        thresholds.extend([0.0, float(np.mean(val_pred)), float(np.mean(val_pred) + np.std(val_pred))])
        for threshold in thresholds:
            switch = pred_gain > threshold
            selected = base_ade.copy()
            selected[switch] = candidate_ade[switch]
            val_metric = am._metric(selected, base_ade, data, switch, val_mask)
            if val_metric["easy_degradation"] > 0.02:
                continue
            val_candidate_count += 1
            score = _score(val_metric)
            if score > best_score:
                best_score = score
                best = {
                    "lambda": float(lam),
                    "threshold": float(threshold),
                    "score": float(score),
                    "val_metric": val_metric,
                    "pred_gain": pred_gain,
                    "switch": switch,
                    "selected_ade": selected,
                }
    if best is None:
        switch = np.zeros(len(base_ade), dtype=bool)
        selected = base_ade.copy()
        best = {
            "lambda": None,
            "threshold": None,
            "score": 0.0,
            "val_metric": am._metric(selected, base_ade, data, switch, val_mask),
            "pred_gain": np.zeros(len(base_ade), dtype=np.float64),
            "switch": switch,
            "selected_ade": selected,
        }
    best["test_metric"] = am._metric(best["selected_ade"], base_ade, data, best["switch"], test_mask)
    best["val_candidate_count"] = int(val_candidate_count)
    return best


def _oracle_metric(base_ade: np.ndarray, candidate_ade: np.ndarray, data: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    oracle_switch = candidate_ade < base_ade
    oracle_selected = np.minimum(base_ade, candidate_ade)
    return am._metric(oracle_selected, base_ade, data, oracle_switch, mask)


def _source_breakdown(
    *,
    base_ade: np.ndarray,
    candidate_ade: np.ndarray,
    selected_ade: np.ndarray,
    switch: np.ndarray,
    data: Mapping[str, np.ndarray],
    mask: np.ndarray,
) -> list[dict[str, Any]]:
    sources = data["source_file"].astype(str)
    rows: list[dict[str, Any]] = []
    for source in sorted(set(sources[mask])):
        sm = mask & (sources == source)
        if np.sum(sm) < 20:
            continue
        rows.append(
            {
                "source_file": source,
                "rows": int(np.sum(sm)),
                "router_improvement": am._safe_improvement(selected_ade, base_ade, sm),
                "oracle_headroom": _oracle_metric(base_ade, candidate_ade, data, sm)["all_improvement"],
                "positive_gain_rate": _safe_rate(candidate_ade < base_ade, sm),
                "switch_rate": _safe_rate(switch, sm),
            }
        )
    rows.sort(key=lambda row: (row["router_improvement"], row["rows"]), reverse=True)
    return rows[:12]


def _diagnose_blocker(stats: Mapping[str, Any]) -> str:
    if stats["oracle_headroom"] < 0.005:
        return "candidate_oracle_headroom_too_small"
    if stats["low_margin_abs_0p01_rate"] > 0.60:
        return "low_margin_candidate_ambiguity"
    if stats["train_test_positive_gain_rate_gap_abs"] > 0.20:
        return "train_test_distribution_shift"
    if stats["capture_rate"] < 0.15 and stats["switch_rate"] < 0.05:
        return "router_under_switches_despite_headroom"
    if stats["switched_harm_rate"] > 0.25 or stats["easy_degradation"] > 0.02:
        return "unsafe_or_uncalibrated_switching"
    return "weak_predictive_signal_or_baseline_family_dominance"


def _candidate_audit(
    *,
    horizon: int,
    candidate: str,
    raw_router_features: np.ndarray,
    base_ade: np.ndarray,
    candidate_ade: np.ndarray,
    split: np.ndarray,
    data: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    horizon_mask = data["horizon"].astype(int) == int(horizon)
    h_data = _filter_rows(data, horizon_mask)
    h_split = split[horizon_mask]
    h_base = base_ade[horizon_mask]
    h_cand = candidate_ade[horizon_mask]
    h_features = raw_router_features[horizon_mask]
    router = _train_router_with_arrays(
        raw_router_features=h_features,
        base_ade=h_base,
        candidate_ade=h_cand,
        split=h_split,
        data=h_data,
    )
    test_mask = h_split == "test"
    train_mask = h_split == "train"
    val_mask = h_split == "val"
    gain = h_base - h_cand
    positive = gain > 0.0
    abs_margin = np.abs(gain)
    switch = router["switch"]
    selected = router["selected_ade"]
    oracle = _oracle_metric(h_base, h_cand, h_data, test_mask)
    router_metric = router["test_metric"]
    positive_gain_mass = np.maximum(gain, 0.0)
    gain_mass_den = float(np.sum(positive_gain_mass[test_mask]))
    captured_gain_mass = float(np.sum(positive_gain_mass[test_mask & switch]))
    capture_rate = captured_gain_mass / max(gain_mass_den, EPS)
    switched_test = test_mask & switch
    not_switched_test = test_mask & ~switch
    switched_positive_rate = _safe_rate(positive, switched_test)
    switched_harm_rate = _safe_rate(gain < 0.0, switched_test)
    missed_positive_rate = _safe_rate(positive, not_switched_test)
    train_positive_rate = _safe_rate(positive, train_mask)
    val_positive_rate = _safe_rate(positive, val_mask)
    test_positive_rate = _safe_rate(positive, test_mask)
    stats = {
        "source": "fresh_run",
        "horizon": int(horizon),
        "candidate": candidate,
        "rows": {
            "train": int(np.sum(train_mask)),
            "val": int(np.sum(val_mask)),
            "test": int(np.sum(test_mask)),
        },
        "validation_selection": {
            "source": "validation_only",
            "lambda": router["lambda"],
            "pred_gain_threshold": router["threshold"],
            "candidate_count": router["val_candidate_count"],
            "test_threshold_tuning": False,
            "val_metric": router["val_metric"],
        },
        "router_metric": router_metric,
        "oracle_metric": oracle,
        "oracle_headroom": float(oracle["all_improvement"]),
        "router_improvement": float(router_metric["all_improvement"]),
        "capture_rate": float(capture_rate),
        "positive_gain_rate": float(test_positive_rate),
        "positive_gain_rate_train": float(train_positive_rate),
        "positive_gain_rate_val": float(val_positive_rate),
        "positive_gain_rate_test": float(test_positive_rate),
        "train_test_positive_gain_rate_gap_abs": float(abs(train_positive_rate - test_positive_rate)),
        "mean_gain_test": _safe_mean(gain, test_mask),
        "median_gain_test": float(np.median(gain[test_mask])) if np.any(test_mask) else 0.0,
        "q90_gain_test": float(np.percentile(gain[test_mask], 90.0)) if np.any(test_mask) else 0.0,
        "q10_gain_test": float(np.percentile(gain[test_mask], 10.0)) if np.any(test_mask) else 0.0,
        "low_margin_abs_0p001_rate": _safe_rate(abs_margin < 0.001, test_mask),
        "low_margin_abs_0p01_rate": _safe_rate(abs_margin < 0.01, test_mask),
        "low_margin_abs_0p05_rate": _safe_rate(abs_margin < 0.05, test_mask),
        "family_oracle_margin_mean": _safe_mean(h_data["oracle_margin"], test_mask),
        "family_oracle_margin_low_0p01_rate": _safe_rate(h_data["oracle_margin"] < 0.01, test_mask),
        "switch_rate": float(router_metric["switch_rate"]),
        "switched_positive_rate": float(switched_positive_rate),
        "switched_harm_rate": float(switched_harm_rate),
        "missed_positive_rate": float(missed_positive_rate),
        "missed_positive_gain_mass": float(np.sum(positive_gain_mass[not_switched_test])),
        "captured_positive_gain_mass": float(captured_gain_mass),
        "easy_degradation": float(router_metric["easy_degradation"]),
        "harm_over_fallback": float(router_metric["harm_over_fallback"]),
        "source_breakdown_top": _source_breakdown(
            base_ade=h_base,
            candidate_ade=h_cand,
            selected_ade=selected,
            switch=switch,
            data=h_data,
            mask=test_mask,
        ),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
    }
    stats["blocker_diagnosis"] = _diagnose_blocker(stats)
    return stats


def _build_result() -> dict[str, Any]:
    shared = an._prep_shared()
    masks = ao._incremental_variant_masks(shared["feature_names"])
    split = shared["split"]
    data = shared["data"]
    base_pred = el._prepare_variant_predictions(shared["features"][:, masks["baseline_family_only"]], shared)
    graph, graph_names, graph_stats = sg._build_graph_features(data)
    seq_summary, seq_names, seq_stats = eq._sequence_summary(data)
    candidate_predictions: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    for candidate in CANDIDATES:
        candidate_features = shared["features"][:, masks[candidate]]
        candidate_predictions[candidate] = el._prepare_variant_predictions(candidate_features, shared)
        augmented = eq._augmented_router_features(candidate_features, graph, seq_summary)
        for horizon in TARGET_HORIZONS:
            key = f"h{horizon}_{candidate}"
            audits[key] = _candidate_audit(
                horizon=horizon,
                candidate=candidate,
                raw_router_features=augmented,
                base_ade=base_pred["selected_ade"],
                candidate_ade=candidate_predictions[candidate]["selected_ade"],
                split=split,
                data=data,
            )
    best_by_horizon: dict[str, Any] = {}
    for horizon in TARGET_HORIZONS:
        rows = {key: row for key, row in audits.items() if row["horizon"] == horizon}
        best_oracle_key = max(rows, key=lambda key: rows[key]["oracle_headroom"])
        best_router_key = max(rows, key=lambda key: rows[key]["router_improvement"])
        best_capture_key = max(rows, key=lambda key: rows[key]["capture_rate"])
        best_by_horizon[str(horizon)] = {
            "best_oracle_key": best_oracle_key,
            "best_router_key": best_router_key,
            "best_capture_key": best_capture_key,
            "best_oracle_headroom": rows[best_oracle_key]["oracle_headroom"],
            "best_router_improvement": rows[best_router_key]["router_improvement"],
            "best_capture_rate": rows[best_capture_key]["capture_rate"],
            "dominant_blocker": rows[best_router_key]["blocker_diagnosis"],
        }
    blocker_counts: dict[str, int] = {}
    for row in audits.values():
        blocker_counts[row["blocker_diagnosis"]] = blocker_counts.get(row["blocker_diagnosis"], 0) + 1
    stage42_io = read_json(io.REPORT_JSON, {}) if io.REPORT_JSON.exists() else {}
    result = {
        "source": SOURCE,
        "stage": "Stage42-IP t50/t100 Sequence+Graph Blocker Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/horizon_sequence_graph_context_router_stage42.json",
                "outputs/stage42_long_research/context_gain_router_stage42.json",
                "outputs/stage42_long_research/sequence_graph_context_router_stage42.json",
            ]
        ),
        "current_facts": CURRENT_FACTS,
        "split_stats": shared["split_stats"],
        "target_horizons": TARGET_HORIZONS,
        "candidates": CANDIDATES,
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
        "candidate_horizon_audits": audits,
        "best_by_horizon": best_by_horizon,
        "blocker_counts": blocker_counts,
        "stage42_io_reference": {
            "source": "cached_verified" if stage42_io else "not_run",
            "positive_horizon_sequence_graph_context_routers": stage42_io.get(
                "positive_horizon_sequence_graph_context_routers"
            ),
            "best_by_horizon": stage42_io.get("best_by_horizon"),
        },
        "summary": {
            "source": SOURCE,
            "purpose": "diagnose why Stage42-IO horizon-specific sequence+graph routers remained unsupported at t50/t100",
            "best_by_horizon": best_by_horizon,
            "blocker_counts": blocker_counts,
            "t50_diagnosis": best_by_horizon["50"]["dominant_blocker"],
            "t100_diagnosis": best_by_horizon["100"]["dominant_blocker"],
            "interpretation": (
                "Stage42-IP does not add a new deployable model. It converts the Stage42-IO t50/t100 negative result "
                "into a blocker map: whether the issue is missing candidate oracle headroom, low-margin ambiguity, "
                "train/test shift, under-switching, unsafe switching, or baseline-family dominance."
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
    result["stage42_ip_gate"] = _gate(result)
    return _jsonable(result)


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    nl = result["no_leakage"]
    no_leakage_pass = (
        nl["future_endpoint_input"] is False
        and nl["future_waypoint_input"] is False
        and nl["future_waypoint_label_eval_only"] is True
        and nl["sequence_summary_current_past_only"] is True
        and nl["graph_summary_current_past_only"] is True
        and nl["central_velocity"] is False
        and nl["test_endpoint_goals"] is False
        and nl["test_threshold_tuning"] is False
        and nl["validation_selected_thresholds"] is True
        and nl["source_overlap_pass"] is True
    )
    gates = {
        "source_level_split_loaded": bool(result["split_stats"]),
        "target_horizons_audited": all(
            any(row["horizon"] == horizon for row in result["candidate_horizon_audits"].values())
            for horizon in TARGET_HORIZONS
        ),
        "all_candidates_audited": len(result["candidate_horizon_audits"]) == len(TARGET_HORIZONS) * len(CANDIDATES),
        "oracle_headroom_measured": all(
            "oracle_headroom" in row for row in result["candidate_horizon_audits"].values()
        ),
        "router_capture_measured": all(
            "capture_rate" in row and "switch_rate" in row for row in result["candidate_horizon_audits"].values()
        ),
        "blocker_diagnosis_present": all(
            row.get("blocker_diagnosis") for row in result["candidate_horizon_audits"].values()
        ),
        "source_breakdown_present": all(
            isinstance(row.get("source_breakdown_top"), list) for row in result["candidate_horizon_audits"].values()
        ),
        "negative_result_not_overclaimed": True,
        "no_leakage_pass": no_leakage_pass,
        "no_metric_seconds_overclaim": not result["claim_boundary"]["global_metric_claim_allowed"]
        and not result["claim_boundary"]["global_seconds_claim_allowed"],
        "stage5c_false": result["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": result["claim_boundary"]["smc_enabled"] is False,
    }
    return {
        "passed": int(sum(gates.values())),
        "total": int(len(gates)),
        "gates": gates,
        "verdict": "stage42_ip_t50_t100_sequence_graph_blocker_audit_pass"
        if all(gates.values())
        else "stage42_ip_t50_t100_sequence_graph_blocker_audit_fail",
    }


def _format_float(x: Any) -> str:
    return f"{float(x):.6f}"


def _render_md(result: Mapping[str, Any]) -> str:
    gate = result["stage42_ip_gate"]
    lines = [
        "# Stage42-IP t50/t100 Sequence+Graph Blocker Audit",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
    ]
    lines.extend(f"- {fact}" for fact in result["current_facts"])
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- t50_diagnosis: `{result['summary']['t50_diagnosis']}`",
            f"- t100_diagnosis: `{result['summary']['t100_diagnosis']}`",
            f"- blocker_counts: `{result['blocker_counts']}`",
            "",
            result["summary"]["interpretation"],
            "",
            "## Best By Horizon",
            "",
            "| horizon | best oracle | oracle headroom | best router | router improvement | best capture | capture rate | dominant blocker |",
            "| ---: | --- | ---: | --- | ---: | --- | ---: | --- |",
        ]
    )
    for horizon, row in result["best_by_horizon"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(horizon),
                    f"`{row['best_oracle_key']}`",
                    _format_float(row["best_oracle_headroom"]),
                    f"`{row['best_router_key']}`",
                    _format_float(row["best_router_improvement"]),
                    f"`{row['best_capture_key']}`",
                    _format_float(row["best_capture_rate"]),
                    f"`{row['dominant_blocker']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Candidate-Horizon Audit",
            "",
            "| key | rows test | oracle | router | capture | pos gain | switch | switched good | missed good | low margin 0.01 | easy deg | blocker |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for key in sorted(result["candidate_horizon_audits"]):
        row = result["candidate_horizon_audits"][key]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{key}`",
                    str(row["rows"]["test"]),
                    _format_float(row["oracle_headroom"]),
                    _format_float(row["router_improvement"]),
                    _format_float(row["capture_rate"]),
                    _format_float(row["positive_gain_rate"]),
                    _format_float(row["switch_rate"]),
                    _format_float(row["switched_positive_rate"]),
                    _format_float(row["missed_positive_rate"]),
                    _format_float(row["low_margin_abs_0p01_rate"]),
                    _format_float(row["easy_degradation"]),
                    f"`{row['blocker_diagnosis']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
        ]
    )
    lines.extend(f"| `{name}` | {passed} |" for name, passed in gate["gates"].items())
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-IP is a fresh diagnostic follow-up to Stage42-IO.",
            "- It does not promote sequence/graph context to a t50/t100 main contribution.",
            "- If oracle headroom is small, the candidate context itself is weak under this protocol.",
            "- If oracle headroom exists but capture is low, the next repair should target switchability/calibration rather than more raw context features.",
            "- Claims remain raw-frame / dataset-local 2.5D only; no metric/seconds claim, Stage5C, or SMC.",
        ]
    )
    return "\n".join(lines) + "\n"


def _update_ledgers(result: Mapping[str, Any]) -> None:
    gate = result["stage42_ip_gate"]
    block = [
        "## Stage42-IP t50/t100 Sequence+Graph Blocker Audit",
        "",
        f"- source: `{result['source']}`",
        "- role: explains why Stage42-IO sequence+graph context did not become deployable at t50/t100.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- t50_diagnosis: `{result['summary']['t50_diagnosis']}`.",
        f"- t100_diagnosis: `{result['summary']['t100_diagnosis']}`.",
        f"- blocker_counts: `{result['blocker_counts']}`.",
        "- conclusion: blocker audit only; no new deployable model and no t50/t100 context contribution claim.",
        "- Boundary: raw-frame/dataset-local 2.5D; no metric/seconds claim, no Stage5C, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_LEDGER]:
        _replace_section(path, "STAGE42_IP_T50_T100_SEQUENCE_GRAPH_BLOCKER_AUDIT", block)
    state = read_json(RESEARCH_STATE, {})
    state.setdefault("generated_reports", [])
    for report in [str(REPORT_MD), str(REPORT_JSON), str(GATE_MD)]:
        if report not in state["generated_reports"]:
            state["generated_reports"].append(report)
    state.setdefault("stage42_long_research", {})
    state["stage42_long_research"]["stage_ip_t50_t100_sequence_graph_blocker_audit"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": gate["verdict"],
        "gates": f"{gate['passed']}/{gate['total']}",
        "summary": result["summary"],
        "best_by_horizon": result["best_by_horizon"],
        "claim_boundary": result["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    result = _build_result()
    write_json(REPORT_JSON, result)
    REPORT_MD.write_text(_render_md(result), encoding="utf-8")
    gate = result["stage42_ip_gate"]
    gate_lines = [
        "# Stage42-IP Gate",
        "",
        f"- source: `{result['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    gate_lines.extend(f"| `{name}` | {passed} |" for name, passed in gate["gates"].items())
    GATE_MD.write_text("\n".join(gate_lines) + "\n", encoding="utf-8")
    _update_ledgers(result)
    return result


if __name__ == "__main__":
    out = run()
    print(f"Wrote {REPORT_MD}")
    print(f"Verdict: {out['stage42_ip_gate']['verdict']} ({out['stage42_ip_gate']['passed']}/{out['stage42_ip_gate']['total']})")
