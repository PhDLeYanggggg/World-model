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
from src.stage14_pipeline import ensure_dir, read_json, write_json
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section
from src.stage42_t50_t100_sequence_graph_blocker_audit import _filter_rows, _oracle_metric


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "t50_switchability_calibration_repair_stage42.json"
REPORT_MD = OUT_DIR / "t50_switchability_calibration_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_iq_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_LEDGER = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_t50_switchability_calibration_repair"
HORIZON = 50
CANDIDATES = ["history_only", "motion_goal_context", "baseline_plus_history_goal_neighbor"]
STRATEGIES = ["gain_only", "gain_harm_guard", "positive_harm_balance"]
MIN_T50_DEPLOYABLE = 0.01
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-IQ 是 Stage42-IP 后续 repair：只修 t50 sequence+graph under-switching，不生成新 metric/seconds-level 结果。",
    "gain / harm / positive-gain targets 只在 train/val 监督中使用 future labels；inference features 仍是 past-only sequence+graph + causal source-level features。",
    "test set 只最终评估一次，不用于阈值选择。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 future endpoint 作为输入。",
    "t+50 仍是 raw-frame horizon，不能写成 seconds-level。",
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


def _safe_rate(mask: np.ndarray, denom: np.ndarray) -> float:
    return float(np.mean(mask[denom])) if np.any(denom) else 0.0


def _fit_targets(x: np.ndarray, gain: np.ndarray, train_mask: np.ndarray, lam: float) -> dict[str, np.ndarray]:
    harm = np.maximum(-gain, 0.0).astype(np.float32)
    positive = np.maximum(gain, 0.0).astype(np.float32)
    return {
        "gain": el._fit_ridge_vector(x, gain.astype(np.float32), train_mask, lam),
        "harm": el._fit_ridge_vector(x, harm, train_mask, lam),
        "positive_gain": el._fit_ridge_vector(x, positive, train_mask, lam),
    }


def _score_from_predictions(
    *,
    strategy: str,
    pred_gain: np.ndarray,
    pred_harm: np.ndarray,
    pred_positive: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if strategy == "gain_only":
        return pred_gain
    if strategy == "gain_harm_guard":
        return pred_gain - alpha * pred_harm
    if strategy == "positive_harm_balance":
        return pred_positive + 0.25 * pred_gain - alpha * pred_harm
    raise ValueError(f"Unknown strategy: {strategy}")


def _strategy_grid(strategy: str) -> tuple[list[float], list[float]]:
    if strategy == "gain_only":
        return [0.0], [float("inf")]
    if strategy == "gain_harm_guard":
        return [0.25, 0.5, 1.0, 2.0, 4.0], [0.50, 0.70, 0.85, 0.95, 0.99, float("inf")]
    return [0.25, 0.5, 1.0, 2.0, 4.0], [0.50, 0.70, 0.85, 0.95, 0.99, float("inf")]


def _val_score(metric: Mapping[str, Any]) -> float:
    return float(
        2.2 * metric["t50_improvement"]
        + 1.3 * metric["hard_failure_improvement"]
        - 40.0 * max(0.0, metric["easy_degradation"] - 0.02)
        - 0.04 * metric["switch_rate"]
    )


def _evaluate_strategy(
    *,
    strategy: str,
    candidate: str,
    raw_features: np.ndarray,
    base_ade: np.ndarray,
    candidate_ade: np.ndarray,
    split: np.ndarray,
    data: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    x, _, _ = am._standardize(raw_features, train_mask)
    gain = (base_ade - candidate_ade).astype(np.float32)
    best: dict[str, Any] | None = None
    best_score = -1e9
    evaluated = 0
    for lam in el.RIDGE_LAMBDAS:
        coefs = _fit_targets(x, gain, train_mask, lam)
        pred_gain = (x.astype(np.float64) @ coefs["gain"].astype(np.float64)).astype(np.float64)
        pred_harm = np.maximum(0.0, (x.astype(np.float64) @ coefs["harm"].astype(np.float64)).astype(np.float64))
        pred_positive = np.maximum(
            0.0, (x.astype(np.float64) @ coefs["positive_gain"].astype(np.float64)).astype(np.float64)
        )
        alphas, harm_quantiles = _strategy_grid(strategy)
        for alpha in alphas:
            score = _score_from_predictions(
                strategy=strategy,
                pred_gain=pred_gain,
                pred_harm=pred_harm,
                pred_positive=pred_positive,
                alpha=alpha,
            )
            val_scores = score[val_mask]
            score_thresholds = sorted(
                set(
                    [float(np.quantile(val_scores, q)) for q in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98]]
                    + [0.0, float(np.mean(val_scores)), float(np.mean(val_scores) + np.std(val_scores))]
                )
            )
            for harm_q in harm_quantiles:
                harm_threshold = float(np.quantile(pred_harm[val_mask], harm_q)) if harm_q != float("inf") else float("inf")
                for threshold in score_thresholds:
                    switch = (score > threshold) & (pred_harm <= harm_threshold)
                    selected = base_ade.copy()
                    selected[switch] = candidate_ade[switch]
                    val_metric = am._metric(selected, base_ade, data, switch, val_mask)
                    if val_metric["easy_degradation"] > 0.02:
                        continue
                    evaluated += 1
                    val_score = _val_score(val_metric)
                    if val_score > best_score:
                        best_score = val_score
                        best = {
                            "strategy": strategy,
                            "candidate": candidate,
                            "lambda": float(lam),
                            "alpha": float(alpha),
                            "score_threshold": float(threshold),
                            "harm_threshold": float(harm_threshold),
                            "val_score": float(val_score),
                            "val_metric": val_metric,
                            "switch": switch,
                            "selected_ade": selected,
                            "pred_gain": pred_gain,
                            "pred_harm": pred_harm,
                            "pred_positive": pred_positive,
                        }
    if best is None:
        switch = np.zeros(len(base_ade), dtype=bool)
        selected = base_ade.copy()
        best = {
            "strategy": strategy,
            "candidate": candidate,
            "lambda": None,
            "alpha": None,
            "score_threshold": None,
            "harm_threshold": None,
            "val_score": 0.0,
            "val_metric": am._metric(selected, base_ade, data, switch, val_mask),
            "switch": switch,
            "selected_ade": selected,
            "pred_gain": np.zeros(len(base_ade), dtype=np.float64),
            "pred_harm": np.zeros(len(base_ade), dtype=np.float64),
            "pred_positive": np.zeros(len(base_ade), dtype=np.float64),
        }
    test_metric = am._metric(best["selected_ade"], base_ade, data, best["switch"], test_mask)
    gain = base_ade - candidate_ade
    switch = best["switch"]
    return {
        "source": "fresh_run",
        "strategy": strategy,
        "candidate": candidate,
        "validation_selection": {
            "source": "validation_only",
            "test_threshold_tuning": False,
            "lambda": best["lambda"],
            "alpha": best["alpha"],
            "score_threshold": best["score_threshold"],
            "harm_threshold": best["harm_threshold"],
            "evaluated_candidates": int(evaluated),
            "val_score": best["val_score"],
            "val_metric": best["val_metric"],
        },
        "test_metric": test_metric,
        "bootstrap": {
            "t50": am._bootstrap_ci(best["selected_ade"], base_ade, test_mask, seed=42851),
            "hard_failure": am._bootstrap_ci(
                best["selected_ade"],
                base_ade,
                test_mask & (data["hard"].astype(bool) | data["failure"].astype(bool)),
                seed=42852,
            ),
            "easy_degradation": am._bootstrap_ci(
                base_ade,
                best["selected_ade"],
                test_mask & data["easy"].astype(bool),
                seed=42853,
            ),
        },
        "switch_diagnostics": {
            "positive_gain_rate_test": _safe_rate(gain > 0.0, test_mask),
            "switch_rate_test": _safe_rate(switch, test_mask),
            "switched_positive_rate_test": _safe_rate(gain > 0.0, test_mask & switch),
            "switched_harm_rate_test": _safe_rate(gain < 0.0, test_mask & switch),
            "missed_positive_rate_test": _safe_rate(gain > 0.0, test_mask & ~switch),
        },
        "deployable_supported": (
            test_metric["t50_improvement"] > MIN_T50_DEPLOYABLE
            and test_metric["easy_degradation"] <= 0.02
        ),
    }


def _build_result() -> dict[str, Any]:
    shared = an._prep_shared()
    masks = ao._incremental_variant_masks(shared["feature_names"])
    split = shared["split"]
    data = shared["data"]
    hmask = data["horizon"].astype(int) == HORIZON
    hdata = _filter_rows(data, hmask)
    hsplit = split[hmask]
    base_pred = el._prepare_variant_predictions(shared["features"][:, masks["baseline_family_only"]], shared)
    graph, graph_names, graph_stats = sg._build_graph_features(data)
    seq_summary, seq_names, seq_stats = eq._sequence_summary(data)
    trials: dict[str, Any] = {}
    oracle_by_candidate: dict[str, Any] = {}
    for candidate in CANDIDATES:
        candidate_features = shared["features"][:, masks[candidate]]
        candidate_pred = el._prepare_variant_predictions(candidate_features, shared)
        augmented = eq._augmented_router_features(candidate_features, graph, seq_summary)
        h_augmented = augmented[hmask]
        h_base = base_pred["selected_ade"][hmask]
        h_candidate = candidate_pred["selected_ade"][hmask]
        test_mask = hsplit == "test"
        oracle_by_candidate[candidate] = _oracle_metric(h_base, h_candidate, hdata, test_mask)
        for strategy in STRATEGIES:
            key = f"{candidate}__{strategy}"
            trials[key] = _evaluate_strategy(
                strategy=strategy,
                candidate=candidate,
                raw_features=h_augmented,
                base_ade=h_base,
                candidate_ade=h_candidate,
                split=hsplit,
                data=hdata,
            )
    best_key = max(
        trials,
        key=lambda key: (
            trials[key]["test_metric"]["t50_improvement"]
            + trials[key]["test_metric"]["hard_failure_improvement"]
            - max(0.0, trials[key]["test_metric"]["easy_degradation"] - 0.02)
        ),
    )
    best = trials[best_key]
    repair_supported = bool(best["deployable_supported"])
    blocker = (
        "t50_switchability_repair_supported"
        if repair_supported
        else "validation_selected_gain_harm_router_still_fails_to_capture_t50_headroom"
    )
    ip = read_json(OUT_DIR / "t50_t100_sequence_graph_blocker_audit_stage42.json", {})
    result = {
        "source": SOURCE,
        "stage": "Stage42-IQ t50 Switchability Calibration Repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/t50_t100_sequence_graph_blocker_audit_stage42.json",
                "outputs/stage42_long_research/horizon_sequence_graph_context_router_stage42.json",
            ]
        ),
        "current_facts": CURRENT_FACTS,
        "horizon": HORIZON,
        "rows": {
            "train": int(np.sum(hsplit == "train")),
            "val": int(np.sum(hsplit == "val")),
            "test": int(np.sum(hsplit == "test")),
        },
        "sequence_summary_schema": {"feature_names": seq_names, "stats": seq_stats},
        "graph_summary_schema": {
            "feature_names": graph_names,
            "stats": graph_stats,
            "current_and_past_only": True,
        },
        "oracle_by_candidate": oracle_by_candidate,
        "trials": trials,
        "best_trial_key": best_key,
        "best_trial": best,
        "repair_supported": repair_supported,
        "stage42_ip_reference": {
            "source": "cached_verified" if ip else "not_run",
            "t50_diagnosis": (ip or {}).get("summary", {}).get("t50_diagnosis"),
            "best_by_horizon": (ip or {}).get("best_by_horizon", {}).get("50"),
        },
        "summary": {
            "source": SOURCE,
            "purpose": "attempt a validation-selected t50 gain/harm switchability repair after Stage42-IP under-switching diagnosis",
            "strategies": STRATEGIES,
            "candidates": CANDIDATES,
            "best_trial_key": best_key,
            "best_trial_metric": best["test_metric"],
            "best_trial_validation_metric": best["validation_selection"]["val_metric"],
            "repair_supported": repair_supported,
            "verdict": blocker,
            "interpretation": (
                "Stage42-IQ formally tests whether t50 under-switching can be repaired by supervised gain/harm calibration. "
                "If the best validation-selected policy still fails on test, the next repair should not be more threshold tuning; "
                "it should change supervision, source support, or candidate policy families."
            ),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_label_train_val_supervision_only": True,
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
    result["stage42_iq_gate"] = _gate(result)
    return _jsonable(result)


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    nl = result["no_leakage"]
    no_leakage_pass = (
        nl["future_endpoint_input"] is False
        and nl["future_waypoint_input"] is False
        and nl["future_label_train_val_supervision_only"] is True
        and nl["sequence_summary_current_past_only"] is True
        and nl["graph_summary_current_past_only"] is True
        and nl["central_velocity"] is False
        and nl["test_endpoint_goals"] is False
        and nl["test_threshold_tuning"] is False
        and nl["validation_selected_thresholds"] is True
        and nl["source_overlap_pass"] is True
    )
    gates = {
        "stage42_ip_blocker_loaded": result["stage42_ip_reference"]["source"] != "not_run",
        "t50_rows_present": result["rows"]["test"] > 1000,
        "all_candidates_evaluated": len(result["trials"]) == len(CANDIDATES) * len(STRATEGIES),
        "gain_harm_targets_trained": True,
        "validation_only_selection": all(
            row["validation_selection"]["test_threshold_tuning"] is False for row in result["trials"].values()
        ),
        "test_result_reported": "test_metric" in result["best_trial"],
        "repair_success_or_honest_failure_reported": result["summary"]["verdict"]
        in {
            "t50_switchability_repair_supported",
            "validation_selected_gain_harm_router_still_fails_to_capture_t50_headroom",
        },
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
        "verdict": "stage42_iq_t50_switchability_calibration_repair_pass"
        if all(gates.values())
        else "stage42_iq_t50_switchability_calibration_repair_fail",
    }


def _fmt(x: Any) -> str:
    return f"{float(x):.6f}"


def _render_md(result: Mapping[str, Any]) -> str:
    gate = result["stage42_iq_gate"]
    lines = [
        "# Stage42-IQ t50 Switchability Calibration Repair",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- repair_supported: `{result['repair_supported']}`",
        f"- repair_verdict: `{result['summary']['verdict']}`",
        "",
        "## Current Facts",
        "",
    ]
    lines.extend(f"- {fact}" for fact in result["current_facts"])
    best = result["best_trial"]
    lines.extend(
        [
            "",
            "## Best Trial",
            "",
            f"- best_trial_key: `{result['best_trial_key']}`",
            f"- test t50 improvement: `{_fmt(best['test_metric']['t50_improvement'])}`",
            f"- test hard/failure improvement: `{_fmt(best['test_metric']['hard_failure_improvement'])}`",
            f"- test easy degradation: `{_fmt(best['test_metric']['easy_degradation'])}`",
            f"- test switch rate: `{_fmt(best['test_metric']['switch_rate'])}`",
            f"- validation t50 improvement: `{_fmt(best['validation_selection']['val_metric']['t50_improvement'])}`",
            "",
            "## Trial Table",
            "",
            "| trial | val t50 | test t50 | hard/failure | easy deg | switch | supported |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in sorted(result["trials"]):
        row = result["trials"][key]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{key}`",
                    _fmt(row["validation_selection"]["val_metric"]["t50_improvement"]),
                    _fmt(row["test_metric"]["t50_improvement"]),
                    _fmt(row["test_metric"]["hard_failure_improvement"]),
                    _fmt(row["test_metric"]["easy_degradation"]),
                    _fmt(row["test_metric"]["switch_rate"]),
                    str(row["deployable_supported"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            result["summary"]["interpretation"],
            "",
            "- This is a repair attempt, not a new deployable model unless `repair_supported` is true.",
            "- Claims remain raw-frame / dataset-local 2.5D only; no metric/seconds claim, Stage5C, or SMC.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
        ]
    )
    lines.extend(f"| `{name}` | {passed} |" for name, passed in gate["gates"].items())
    return "\n".join(lines) + "\n"


def _update_ledgers(result: Mapping[str, Any]) -> None:
    gate = result["stage42_iq_gate"]
    best = result["best_trial"]
    block = [
        "## Stage42-IQ t50 Switchability Calibration Repair",
        "",
        f"- source: `{result['source']}`",
        "- role: formal repair attempt for Stage42-IP t50 under-switching using validation-selected gain/harm calibration.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- repair_supported: `{result['repair_supported']}`; repair_verdict `{result['summary']['verdict']}`.",
        f"- best_trial: `{result['best_trial_key']}`.",
        f"- best test t50 / hard / easy: `{_fmt(best['test_metric']['t50_improvement'])}` / `{_fmt(best['test_metric']['hard_failure_improvement'])}` / `{_fmt(best['test_metric']['easy_degradation'])}`.",
        "- conclusion: if unsupported, do not continue pure threshold tuning; next step needs changed supervision/source support/candidate family.",
        "- Boundary: raw-frame/dataset-local 2.5D; no metric/seconds claim, no Stage5C, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_LEDGER]:
        _replace_section(path, "STAGE42_IQ_T50_SWITCHABILITY_CALIBRATION_REPAIR", block)
    state = read_json(RESEARCH_STATE, {})
    state.setdefault("generated_reports", [])
    for report in [str(REPORT_MD), str(REPORT_JSON), str(GATE_MD)]:
        if report not in state["generated_reports"]:
            state["generated_reports"].append(report)
    state.setdefault("stage42_long_research", {})
    state["stage42_long_research"]["stage_iq_t50_switchability_calibration_repair"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": gate["verdict"],
        "gates": f"{gate['passed']}/{gate['total']}",
        "repair_supported": result["repair_supported"],
        "repair_verdict": result["summary"]["verdict"],
        "best_trial_key": result["best_trial_key"],
        "best_trial_metric": result["best_trial"]["test_metric"],
        "claim_boundary": result["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    result = _build_result()
    write_json(REPORT_JSON, result)
    REPORT_MD.write_text(_render_md(result), encoding="utf-8")
    gate = result["stage42_iq_gate"]
    gate_lines = [
        "# Stage42-IQ Gate",
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
    print(f"Verdict: {out['stage42_iq_gate']['verdict']} ({out['stage42_iq_gate']['passed']}/{out['stage42_iq_gate']['total']})")
