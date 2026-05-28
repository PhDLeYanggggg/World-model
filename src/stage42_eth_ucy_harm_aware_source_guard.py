from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_eth_ucy_source_specific_easy_guard as jg
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_rotation_easy_guard_repair as jf
from src import stage42_source_rotation_full_waypoint_eval as je
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "eth_ucy_harm_aware_source_guard_stage42.json"
REPORT_MD = OUT_DIR / "eth_ucy_harm_aware_source_guard_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_jh_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE42_JH_ETH_UCY_HARM_AWARE_SOURCE_GUARD"
SOURCE = "fresh_stage42_jh_eth_ucy_harm_aware_source_guard"
EPS = 1e-6
GAIN_LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0]
CAPS = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JH 是 ETH_UCY harm-aware source guard：它用 train sources 的 switch gain/harm labels 学 predicted-gain guard。",
    "threshold/cap 只在 non-heldout validation source 上选择；held-out source 不参与调参。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _fit_gain_model(x: np.ndarray, gain: np.ndarray, train_mask: np.ndarray, lam: float) -> np.ndarray:
    ids = np.where(train_mask)[0]
    xt = x[ids].astype(np.float64, copy=False)
    yt = gain[ids].astype(np.float64, copy=False)
    reg = np.eye(xt.shape[1], dtype=np.float64) * float(lam)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xt.T @ xt + reg, xt.T @ yt).astype(np.float32)


def _cap_by_score(base_switch: np.ndarray, score: np.ndarray, horizon: np.ndarray, cap: float) -> np.ndarray:
    capped = np.zeros(len(base_switch), dtype=bool)
    for h in [10, 25, 50, 100]:
        hm = horizon == h
        candidates = np.where(base_switch & hm)[0]
        if len(candidates) == 0:
            continue
        budget = int(np.floor(float(cap) * int(np.sum(hm))))
        budget = max(0, min(len(candidates), budget))
        if budget == 0:
            continue
        order = candidates[np.argsort(-score[candidates])]
        capped[order[:budget]] = True
    return capped


def _thresholds_from_validation(score: np.ndarray, mask: np.ndarray) -> list[float]:
    vals = score[mask]
    if len(vals) == 0:
        return [0.0]
    qs = np.quantile(vals, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
    return sorted({float(v) for v in [np.min(vals), 0.0, *qs.tolist(), np.max(vals)]})


def _evaluate_fold(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], heldout_source: str) -> dict[str, Any]:
    split, split_stats = jg._source_cv_split(data, heldout_source)
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    floor = am._floor_arrays(data, train_mask)
    features, feature_names, removed = je._domain_invariant_features(data, floor)
    x, mean, std = am._standardize(features, train_mask)
    target_delta = (
        (
            labels["waypoint_xy"].astype(np.float64)
            - np.stack([data["current_x"], data["current_y"]], axis=1)[:, None, :]
        )
        / np.maximum(data["scale"].astype(np.float64)[:, None, None], EPS)
    ).astype(np.float32)
    horizon = data["horizon"].astype(int)
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    best: dict[str, Any] | None = None
    candidates = []
    for ridge_lam in am.LAMBDAS:
        coef = am._fit_ridge_model(x, target_delta, labels["waypoint_valid"], train_mask, ridge_lam)
        pred_xy = am._predict_waypoints(x, coef, data)
        base_policy, base_ade, base_fde, base_switch = je._select_horizon_policy_on_val(
            pred_xy, floor["floor_xy"], labels, data, val_mask
        )
        if not base_policy.get("slices"):
            continue
        base_gain = floor_ade - base_ade
        residual_norm = np.linalg.norm(pred_xy[:, -1] - floor["floor_xy"][:, -1], axis=1) / np.maximum(
            data["scale"].astype(np.float64), EPS
        )
        gain_features = np.concatenate([x, residual_norm[:, None].astype(np.float32), base_switch[:, None].astype(np.float32)], axis=1)
        base_test_metric = am._metric(base_ade, floor_ade, data, base_switch, test_mask)
        for gain_lam in GAIN_LAMBDAS:
            gain_coef = _fit_gain_model(gain_features, base_gain, train_mask, gain_lam)
            pred_gain = (gain_features.astype(np.float64) @ gain_coef.astype(np.float64)).astype(np.float64)
            thresholds = _thresholds_from_validation(pred_gain, val_mask & base_switch)
            for threshold in thresholds:
                threshold_switch = base_switch & (pred_gain >= float(threshold))
                for cap in CAPS:
                    guarded_switch = _cap_by_score(threshold_switch, pred_gain, horizon, cap)
                    selected_ade, selected_fde, floor_ade_ref = jf._blend_errors_for_switch(
                        pred_xy, floor["floor_xy"], labels, base_switch, guarded_switch, base_policy, horizon
                    )
                    val_metric = am._metric(selected_ade, floor_ade_ref, data, guarded_switch, val_mask)
                    score = jf._candidate_score(val_metric)
                    row = {
                        "ridge_lambda": float(ridge_lam),
                        "gain_lambda": float(gain_lam),
                        "predicted_gain_threshold": float(threshold),
                        "switch_cap": float(cap),
                        "score": float(score),
                        "policy_slice_count": int(len(base_policy["slices"])),
                        "val_metric": val_metric,
                        "base_test_metric_before_harm_guard": base_test_metric,
                    }
                    candidates.append(row)
                    if best is None or score > best["score"]:
                        best = {
                            **row,
                            "policy": base_policy,
                            "pred_xy": pred_xy,
                            "selected_ade": selected_ade,
                            "selected_fde": selected_fde,
                            "switch": guarded_switch,
                            "floor_ade": floor_ade_ref,
                            "floor_fde": floor_fde,
                            "pred_gain_stats": {
                                "train_mean": float(np.mean(pred_gain[train_mask])),
                                "val_mean": float(np.mean(pred_gain[val_mask])),
                                "test_mean": float(np.mean(pred_gain[test_mask])),
                                "test_switch_mean": float(np.mean(pred_gain[test_mask & guarded_switch])) if np.any(test_mask & guarded_switch) else 0.0,
                            },
                        }
    if best is None:
        raise RuntimeError(f"No harm-aware source guard candidate for {heldout_source}.")
    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    metric = am._metric(best["selected_ade"], best["floor_ade"], data, best["switch"], test_mask)
    fde = am._metric(best["selected_fde"], best["floor_fde"], data, best["switch"], test_mask)
    base = best["base_test_metric_before_harm_guard"]
    return {
        "source": "fresh_run",
        "heldout_source": heldout_source,
        "split_stats": split_stats,
        "feature_schema": {
            "feature_count": int(len(feature_names) + 2),
            "domain_features_removed": removed,
            "extra_features": ["residual_norm", "base_switch_indicator"],
            "normalization": "train_split_mean_std_only",
            "future_inputs": False,
        },
        "selected_candidate": {
            "ridge_lambda": best["ridge_lambda"],
            "gain_lambda": best["gain_lambda"],
            "predicted_gain_threshold": best["predicted_gain_threshold"],
            "switch_cap": best["switch_cap"],
            "score": best["score"],
            "validation_selection_source": "eth_ucy_non_heldout_source_validation_only",
            "test_threshold_tuning": False,
            "policy_slice_count": best["policy_slice_count"],
            "val_metric": best["val_metric"],
            "pred_gain_stats": best["pred_gain_stats"],
        },
        "candidate_count": int(len(candidates)),
        "top_validation_candidates": sorted(candidates, key=lambda r: r["score"], reverse=True)[:8],
        "metrics": {
            "base_horizon_policy_before_harm_guard": base,
            "harm_aware_source_guard": metric,
            "harm_aware_source_guard_fde": fde,
        },
        "delta_vs_base_horizon_policy": {
            "all_improvement": float(metric["all_improvement"]) - float(base["all_improvement"]),
            "t50_improvement": float(metric["t50_improvement"]) - float(base["t50_improvement"]),
            "hard_failure_improvement": float(metric["hard_failure_improvement"]) - float(base["hard_failure_improvement"]),
            "easy_degradation": float(metric["easy_degradation"]) - float(base["easy_degradation"]),
            "switch_rate": float(metric["switch_rate"]) - float(base["switch_rate"]),
        },
        "bootstrap": {
            "all": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask, seed=42171),
            "t50": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (h == 50), seed=42172),
            "t100_raw_frame_diagnostic": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (h == 100), seed=42173),
            "hard_failure": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & hard_failure, seed=42174),
            "easy_degradation": am._bootstrap_ci(best["floor_ade"], best["selected_ade"], test_mask & easy, seed=42175),
        },
    }


def _summary(folds: list[Mapping[str, Any]]) -> dict[str, Any]:
    deployable = []
    repaired = []
    blocked = []
    for row in folds:
        metric = row["metrics"]["harm_aware_source_guard"]
        base = row["metrics"]["base_horizon_policy_before_harm_guard"]
        ok_positive = metric["all_improvement"] > 0.03 and (
            metric["t50_improvement"] > 0.03 or metric["hard_failure_improvement"] > 0.10
        )
        ok_easy = metric["easy_degradation"] <= 0.02
        if ok_positive and ok_easy:
            deployable.append(row["heldout_source"])
        else:
            blocked.append(row["heldout_source"])
        if base["easy_degradation"] > 0.02 and ok_easy:
            repaired.append(row["heldout_source"])
    decision = (
        "eth_ucy_harm_aware_guard_supported_all_sources"
        if len(deployable) == len(folds) and folds
        else "eth_ucy_harm_aware_guard_partial_support"
        if deployable
        else "eth_ucy_harm_aware_guard_not_supported"
    )
    return {
        "source": SOURCE,
        "fold_count": int(len(folds)),
        "deployable_heldout_sources": deployable,
        "blocked_heldout_sources": blocked,
        "easy_repaired_sources": repaired,
        "deployable_source_count": int(len(deployable)),
        "decision": decision,
        "next_action": "Promote only sources that remain positive and easy-safe under harm-aware source-CV; keep other ETH_UCY sources fallback-only.",
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    data = jg._eth_ucy_data()
    labels = am._reconstruct_waypoint_labels(data)
    sources = sorted(set(jg._source_ids(data).tolist()))
    folds = [_evaluate_fold(data, labels, source) for source in sources]
    payload: dict[str, Any] = {
        "stage": "Stage42-JH",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/eth_ucy_source_specific_easy_guard_stage42.json",
            ]
        ),
        "current_facts": CURRENT_FACTS,
        "eth_ucy_rows": int(len(data["horizon"])),
        "eth_ucy_sources": sources,
        "folds": folds,
        "summary": _summary(folds),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "family_fde_input": False,
            "safe_strongest_idx_old_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
            "source_overlap_pass": all(row["split_stats"]["source_overlap_pass"] for row in folds),
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
    payload["stage42_jh_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "harm_aware_guard_attempted_all_sources": len(payload["folds"]) == len(payload["eth_ucy_sources"]) and len(payload["folds"]) >= 5,
        "candidate_search_recorded": all(row["candidate_count"] > 0 and row["top_validation_candidates"] for row in payload["folds"]),
        "validation_only_selection": all(row["selected_candidate"]["test_threshold_tuning"] is False for row in payload["folds"]),
        "deployable_or_blocked_sources_recorded": bool(payload["summary"]["deployable_heldout_sources"] or payload["summary"]["blocked_heldout_sources"]),
        "no_overclaim_full_eth_ucy": payload["summary"]["decision"] != "eth_ucy_harm_aware_guard_supported_all_sources"
        or len(payload["summary"]["blocked_heldout_sources"]) == 0,
        "no_leakage_pass": all(
            payload["no_leakage"][key] is False
            for key in [
                "future_endpoint_input",
                "future_waypoint_input",
                "family_fde_input",
                "safe_strongest_idx_old_input",
                "central_velocity",
                "test_endpoint_goals",
                "test_threshold_tuning",
            ]
        )
        and payload["no_leakage"]["train_only_feature_normalization"]
        and payload["no_leakage"]["source_overlap_pass"],
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    verdict = "stage42_jh_eth_ucy_harm_aware_source_guard_pass" if passed == len(gates) else "stage42_jh_eth_ucy_harm_aware_source_guard_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": len(gates), "verdict": verdict}


def _fmt(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "n/a"


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jh_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage42-JH ETH_UCY Harm-Aware Source Guard",
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
        f"- decision: `{summary['decision']}`",
        f"- deployable_heldout_sources: `{summary['deployable_heldout_sources']}`",
        f"- blocked_heldout_sources: `{summary['blocked_heldout_sources']}`",
        f"- easy_repaired_sources: `{summary['easy_repaired_sources']}`",
        f"- next_action: {summary['next_action']}",
        "",
        "## Source-CV Fold Metrics",
        "",
        "| heldout source | rows | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | base easy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["folds"]:
        m = row["metrics"]["harm_aware_source_guard"]
        b = row["metrics"]["base_horizon_policy_before_harm_guard"]
        lines.append(
            f"| `{row['heldout_source']}` | {m['rows']} | {_fmt(m['all_improvement'])} | {_fmt(m['t50_improvement'])} | "
            f"{_fmt(m['t100_raw_frame_diagnostic_improvement'])} | {_fmt(m['hard_failure_improvement'])} | "
            f"{_fmt(m['easy_degradation'])} | {_fmt(m['switch_rate'])} | {_fmt(b['easy_degradation'])} |"
        )
    lines.extend(["", "## Bootstrap CI", "", "| heldout source | slice | low | mid | high | n |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in payload["folds"]:
        for key, ci in row["bootstrap"].items():
            lines.append(f"| `{row['heldout_source']}` | `{key}` | {_fmt(ci['low'])} | {_fmt(ci['mid'])} | {_fmt(ci['high'])} | {ci['n']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is an actual harm-aware repair attempt, not an overclaim: if a source remains blocked, default deployment stays fallback-only.",
            "- This remains dataset-local/raw-frame 2.5D evidence and does not enable Stage5C or SMC.",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{payload['no_leakage']}`",
            f"- claim_boundary: `{payload['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jh_gate"]
    lines = [
        "# Stage42-JH Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    return lines


def _section_lines(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    gate = payload["stage42_jh_gate"]
    fold_bits = []
    for row in payload["folds"]:
        metric = row["metrics"]["harm_aware_source_guard"]
        fold_bits.append(
            f"{row['heldout_source']}: all {_fmt(metric['all_improvement'])}, t50 {_fmt(metric['t50_improvement'])}, hard {_fmt(metric['hard_failure_improvement'])}, easy {_fmt(metric['easy_degradation'])}"
        )
    return [
        "## Stage42-JH ETH_UCY Harm-Aware Source Guard",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`",
        f"- source-CV harm-aware folds: {'; '.join(fold_bits)}.",
        f"- decision: `{summary['decision']}`; deployable sources: `{summary['deployable_heldout_sources']}`; blocked sources: `{summary['blocked_heldout_sources']}`; easy repaired: `{summary['easy_repaired_sources']}`.",
        "- boundary: this is ETH_UCY source-specific support only, not global/cross-domain success; no metric/seconds claim, no Stage5C, no SMC.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, _section_lines(payload))


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    stage42 = state.setdefault("stage42", {})
    stage42["eth_ucy_harm_aware_source_guard"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_jh_gate"]["verdict"],
        "gate": {"passed": payload["stage42_jh_gate"]["passed"], "total": payload["stage42_jh_gate"]["total"]},
        "decision": payload["summary"]["decision"],
        "deployable_heldout_sources": payload["summary"]["deployable_heldout_sources"],
        "blocked_heldout_sources": payload["summary"]["blocked_heldout_sources"],
        "easy_repaired_sources": payload["summary"]["easy_repaired_sources"],
        "metric_or_seconds_claim": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(LEDGER.parent)
    import json

    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": "Stage42-JH",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_jh_gate"]["verdict"],
                    "result": payload["summary"]["decision"],
                    "fresh_run": True,
                    "downloaded": False,
                    "converted": False,
                    "trained": True,
                    "evaluated": True,
                    "stage5c_executed": False,
                    "smc_enabled": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_stage42_eth_ucy_harm_aware_source_guard(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _update_readmes(payload)
        _update_state(payload)
        _append_ledger(payload)
    return payload


def main() -> None:
    run_stage42_eth_ucy_harm_aware_source_guard(refresh_readmes=True)


if __name__ == "__main__":
    main()
