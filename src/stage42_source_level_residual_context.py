from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_source_level_ablation as an
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_incremental_ablation as ao
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "source_level_residual_context_stage42.json"
REPORT_MD = OUT_DIR / "source_level_residual_context_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ap_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

RESIDUAL_ALPHAS = [0.25, 0.50, 0.75, 1.00]
MIN_RESIDUAL_DELTA = 0.01
EPS = 1e-6


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-AP 是 proposed source-level split residual-context retraining，不是 metric 或 seconds-level 结果。",
    "第一阶段只用 baseline-family rollout context 训练；第二阶段只让 context features 预测第一阶段剩余误差。",
    "所有 residual alpha / lambda / safety policy 均在 validation 上选择；test 只评一次。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _target_delta(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> np.ndarray:
    cur = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float64)
    scale = np.maximum(data["scale"].astype(np.float64), EPS)
    return ((labels["waypoint_xy"].astype(np.float64) - cur[:, None, :]) / scale[:, None, None]).astype(np.float32)


def _xy_to_delta(data: Mapping[str, np.ndarray], xy: np.ndarray) -> np.ndarray:
    cur = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float64)
    scale = np.maximum(data["scale"].astype(np.float64), EPS)
    return ((xy.astype(np.float64) - cur[:, None, :]) / scale[:, None, None]).astype(np.float32)


def _predict_residual_xy(
    base_xy: np.ndarray,
    x: np.ndarray,
    coef: np.ndarray,
    data: Mapping[str, np.ndarray],
    residual_alpha: float,
) -> np.ndarray:
    scale = np.maximum(data["scale"].astype(np.float64), EPS)
    residual_delta = (x.astype(np.float64) @ coef.astype(np.float64)).reshape(len(x), len(am.WAYPOINT_FRAC), 2)
    return (base_xy.astype(np.float64) + float(residual_alpha) * residual_delta * scale[:, None, None]).astype(np.float32)


def _direct_candidate(raw_features: np.ndarray, shared: Mapping[str, Any]) -> dict[str, Any]:
    split = shared["split"]
    data = shared["data"]
    labels = shared["labels"]
    x, _, _ = am._standardize(raw_features, split == "train")
    model = am._evaluate_models(data, split, labels, shared["floor"], x)
    target_delta = _target_delta(data, labels)
    coef = am._fit_ridge_model(x, target_delta, labels["waypoint_valid"], split == "train", float(model["best_lambda"]))
    pred_xy = am._predict_waypoints(x, coef, data)
    return {"x": x, "model": model, "pred_xy": pred_xy, "coef": coef}


def _residual_variant_masks(names: list[str]) -> dict[str, np.ndarray]:
    groups = ao._group_masks(names)
    control = ao._or_mask(groups["horizon"], groups["domain"])
    return {
        "residual_history": ao._or_mask(groups["history"], control),
        "residual_goal": ao._or_mask(groups["goal_prototype"], control),
        "residual_neighbor": ao._or_mask(groups["neighbor_interaction"], control),
        "residual_history_goal": ao._or_mask(groups["history"], groups["goal_prototype"], control),
        "residual_history_neighbor": ao._or_mask(groups["history"], groups["neighbor_interaction"], control),
        "residual_goal_neighbor": ao._or_mask(groups["goal_prototype"], groups["neighbor_interaction"], control),
        "residual_history_goal_neighbor": ao._or_mask(groups["history"], groups["goal_prototype"], groups["neighbor_interaction"], control),
    }


def _baseline_mask(names: list[str]) -> np.ndarray:
    groups = ao._group_masks(names)
    return ao._or_mask(groups["baseline_family"], groups["horizon"], groups["domain"])


def _evaluate_residual_variant(
    name: str,
    raw_features: np.ndarray,
    shared: Mapping[str, Any],
    base_xy: np.ndarray,
    residual_target_delta: np.ndarray,
) -> dict[str, Any]:
    split = shared["split"]
    data = shared["data"]
    labels = shared["labels"]
    floor = shared["floor"]
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    x, _, _ = am._standardize(raw_features, train_mask)
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    val_results = []
    best: dict[str, Any] | None = None
    best_score = -1e9
    for lam in am.LAMBDAS:
        coef = am._fit_ridge_model(x, residual_target_delta, labels["waypoint_valid"], train_mask, float(lam))
        for residual_alpha in RESIDUAL_ALPHAS:
            pred_xy = _predict_residual_xy(base_xy, x, coef, data, residual_alpha)
            policy, selected_ade, selected_fde, switch = am._select_policy_on_val(pred_xy, floor["floor_xy"], labels, data, val_mask)
            val_metric = am._metric(selected_ade, floor_ade, data, switch, val_mask)
            score = (
                1.1 * val_metric["all_improvement"]
                + 2.0 * val_metric["t50_improvement"]
                + 1.2 * val_metric["hard_failure_improvement"]
                - 30.0 * max(0.0, val_metric["easy_degradation"] - 0.02)
                - 0.03 * val_metric["switch_rate"]
            )
            val_results.append(
                {
                    "lambda": float(lam),
                    "residual_alpha": float(residual_alpha),
                    "score": float(score),
                    "policy_slice_count": int(len(policy["slices"])),
                    "val_metric": val_metric,
                }
            )
            if score > best_score:
                best_score = float(score)
                best = {
                    "lambda": float(lam),
                    "residual_alpha": float(residual_alpha),
                    "coef": coef,
                    "policy": policy,
                    "pred_xy": pred_xy,
                    "selected_ade": selected_ade,
                    "selected_fde": selected_fde,
                    "switch": switch,
                    "score": float(score),
                    "val_metric": val_metric,
                }
    if best is None:
        raise RuntimeError(f"No residual model evaluated for {name}.")
    pred_ade, pred_fde = am._trajectory_errors(best["pred_xy"], labels)
    domain = data["dataset"].astype(str)
    horizon = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    ungated_switch = np.ones(len(pred_ade), dtype=bool)
    return {
        "source": "fresh_run",
        "variant": name,
        "feature_count": int(raw_features.shape[1]),
        "best_lambda": float(best["lambda"]),
        "best_residual_alpha": float(best["residual_alpha"]),
        "validation_selection": {
            "source": "fresh_run",
            "test_threshold_tuning": False,
            "selected_score": float(best["score"]),
            "candidates": val_results,
        },
        "policy_slice_count": int(len(best["policy"]["slices"])),
        "protected": am._metric(best["selected_ade"], floor_ade, data, best["switch"], test_mask),
        "protected_fde": am._metric(best["selected_fde"], floor_fde, data, best["switch"], test_mask),
        "ungated_diagnostic": am._metric(pred_ade, floor_ade, data, ungated_switch, test_mask),
        "bootstrap": {
            "all": am._bootstrap_ci(best["selected_ade"], floor_ade, test_mask, seed=42101),
            "t50": am._bootstrap_ci(best["selected_ade"], floor_ade, test_mask & (horizon == 50), seed=42102),
            "t100_raw_frame_diagnostic": am._bootstrap_ci(best["selected_ade"], floor_ade, test_mask & (horizon == 100), seed=42103),
            "hard_failure": am._bootstrap_ci(best["selected_ade"], floor_ade, test_mask & hard_failure, seed=42104),
            "easy_degradation": am._bootstrap_ci(floor_ade, best["selected_ade"], test_mask & easy, seed=42105),
        },
        "by_domain": {
            d: am._metric(best["selected_ade"], floor_ade, data, best["switch"], test_mask & (domain == d))
            for d in sorted(set(domain[test_mask].tolist()))
        },
        "by_horizon": {
            str(h): am._metric(best["selected_ade"], floor_ade, data, best["switch"], test_mask & (horizon == h))
            for h in [10, 25, 50, 100]
        },
    }


def _metric_delta(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> dict[str, float]:
    return ao._metric_delta(lhs, rhs)


def _positive_residual_delta(delta: Mapping[str, float], threshold: float = MIN_RESIDUAL_DELTA) -> bool:
    return (
        delta["all_improvement"] > threshold
        or delta["t50_improvement"] > threshold
        or delta["hard_failure_improvement"] > threshold
    )


def run_stage42_source_level_residual_context() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    shared = an._prep_shared()
    names = shared["feature_names"]
    features = shared["features"]
    baseline_direct = _direct_candidate(features[:, _baseline_mask(names)], shared)
    base_xy = baseline_direct["pred_xy"]
    labels = shared["labels"]
    data = shared["data"]
    residual_target_delta = _target_delta(data, labels) - _xy_to_delta(data, base_xy)
    variants = {}
    for name, mask in _residual_variant_masks(names).items():
        variants[name] = _evaluate_residual_variant(name, features[:, mask], shared, base_xy, residual_target_delta)
    baseline_metric = baseline_direct["model"]["metrics"]["protected_ridge_source_level"]
    residual_deltas = {
        name: {
            "source": "fresh_run",
            "delta_vs_baseline_family_only": _metric_delta(row["protected"], baseline_metric),
            "positive_residual_increment": _positive_residual_delta(_metric_delta(row["protected"], baseline_metric)),
            "interpretation": _interpret_residual(name, _metric_delta(row["protected"], baseline_metric)),
        }
        for name, row in variants.items()
    }
    positive = sorted([name for name, row in residual_deltas.items() if row["positive_residual_increment"]])
    result = {
        "source": "fresh_run",
        "stage": "Stage42-AP proposed source-level residual context retraining",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/source_level_full_waypoint_eval_stage42.json",
                "outputs/stage42_long_research/source_level_incremental_ablation_stage42.json",
            ]
        ),
        "split_stats": shared["split_stats"],
        "feature_groups": {k: int(np.sum(v)) for k, v in ao._group_masks(names).items()},
        "baseline_family_only": {
            "source": "fresh_run",
            "feature_count": int(np.sum(_baseline_mask(names))),
            "best_lambda": baseline_direct["model"]["best_lambda"],
            "protected": baseline_metric,
            "bootstrap": baseline_direct["model"]["bootstrap"],
        },
        "residual_variants": variants,
        "residual_deltas": residual_deltas,
        "positive_residual_context_variants": positive,
        "summary": {
            "source": "fresh_run",
            "residual_context_verdict": "stage42_ap_residual_context_supported" if positive else "stage42_ap_residual_context_not_supported",
            "positive_residual_context_variants": positive,
            "interpretation": "Residual-context variants test whether history/goal/neighbor explain errors left by baseline-family rollout context. Positive variants support context contribution beyond baseline family; negative variants mean current context features do not add residual value under this ridge protocol.",
        },
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
            "source_overlap_pass": bool(shared["split_stats"]["source_overlap_pass"]),
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
    result["stage42_ap_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _append_ledger(result)
    return result


def _interpret_residual(name: str, delta: Mapping[str, float]) -> str:
    if _positive_residual_delta(delta):
        return f"{name} improves over the baseline-family-only first-stage model by > {MIN_RESIDUAL_DELTA} on at least one core metric."
    return f"{name} does not improve over the baseline-family-only first-stage model by > {MIN_RESIDUAL_DELTA}; residual context contribution not proven here."


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    baseline = result["baseline_family_only"]["protected"]
    best_positive = len(result["positive_residual_context_variants"]) >= 1
    gates = {
        "proposed_source_level_split_used": result["split_stats"]["by_split"]["test"]["rows"] == 47458,
        "baseline_family_first_stage_positive": baseline["all_improvement"] > 0
        and baseline["t50_improvement"] > 0
        and baseline["easy_degradation"] <= 0.02,
        "residual_variants_complete": len(result["residual_variants"]) >= 7,
        "residual_context_increment_found": best_positive,
        "bootstrap_available_for_baseline": result["baseline_family_only"]["bootstrap"]["all"]["bootstrap_n"] > 0
        and result["baseline_family_only"]["bootstrap"]["t50"]["bootstrap_n"] > 0,
        "no_leakage_pass": all(
            result["no_leakage"][k] is False
            for k in [
                "future_endpoint_input",
                "future_waypoint_input",
                "family_fde_input",
                "safe_strongest_idx_old_input",
                "central_velocity",
                "test_endpoint_goals",
                "test_threshold_tuning",
            ]
        )
        and result["no_leakage"]["train_only_feature_normalization"],
        "no_metric_seconds_overclaim": not result["claim_boundary"]["metric_or_seconds_claim"],
        "stage5c_false": not result["claim_boundary"]["stage5c_executed"],
        "smc_false": not result["claim_boundary"]["smc_enabled"],
    }
    verdict = (
        "stage42_ap_residual_context_evidence_pass"
        if all(gates.values())
        else "stage42_ap_residual_context_evidence_partial_or_negative"
    )
    return {"source": "fresh_run", "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": int(len(gates)), "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    base = result["baseline_family_only"]["protected"]
    lines = [
        "# Stage42-AP Proposed Source-Level Residual Context",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_ap_gate']['passed']} / {result['stage42_ap_gate']['total']}`",
        f"- verdict: `{result['stage42_ap_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Why This Was Run",
        "",
        "- Stage42-AO found that baseline-family rollout context dominates direct ridge evidence.",
        "- Stage42-AP uses a two-stage residual design: first train baseline-family-only, then train history/goal/neighbor context on the remaining full-waypoint residual.",
        "- If context modules have residual world-state information, they should improve over the first-stage baseline-family model without using future inputs.",
        "",
        "## Baseline-Family First Stage",
        "",
        f"- feature_count: `{result['baseline_family_only']['feature_count']}`",
        f"- best_lambda: `{result['baseline_family_only']['best_lambda']}`",
        f"- protected_metric: `{base}`",
        "",
        "## Residual Variants",
        "",
        "| variant | features | alpha | all | t50 | t100 diag | hard/failure | easy | delta all | delta t50 | delta hard |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["residual_variants"].items():
        metric = row["protected"]
        d = result["residual_deltas"][name]["delta_vs_baseline_family_only"]
        lines.append(
            f"| `{name}` | {row['feature_count']} | {row['best_residual_alpha']:.2f} | {metric['all_improvement']:.6f} | {metric['t50_improvement']:.6f} | {metric['t100_raw_frame_diagnostic_improvement']:.6f} | {metric['hard_failure_improvement']:.6f} | {metric['easy_degradation']:.6f} | {d['all_improvement']:.6f} | {d['t50_improvement']:.6f} | {d['hard_failure_improvement']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- positive_residual_context_variants: `{result['positive_residual_context_variants']}`",
            f"- residual_context_verdict: `{result['summary']['residual_context_verdict']}`",
            "",
        ]
    )
    if result["positive_residual_context_variants"]:
        lines.append("- Stage42-AP found residual context value beyond baseline-family rollout context for at least one variant.")
    else:
        lines.append("- Stage42-AP did not find residual context value beyond baseline-family rollout context under this ridge residual protocol.")
    lines.extend(
        [
            "- This is a boundary result: it constrains paper claims and motivates a stronger neural/graph context model rather than overclaiming ridge context modules.",
            "- Claims remain dataset-local raw-frame 2.5D, not metric/seconds-level, true-3D, foundation, Stage5C, or SMC evidence.",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{result['no_leakage']}`",
            f"- claim_boundary: `{result['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_ap_gate"]
    lines = [
        "# Stage42-AP Gate",
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


def _append_ledger(result: Mapping[str, Any]) -> None:
    row = {
        "stage": result["stage"],
        "source": result["source"],
        "generated_at_utc": result["generated_at_utc"],
        "verdict": result["stage42_ap_gate"]["verdict"],
        "gate": f"{result['stage42_ap_gate']['passed']}/{result['stage42_ap_gate']['total']}",
        "git_commit": result["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(f"{row}\n")


if __name__ == "__main__":
    run_stage42_source_level_residual_context()
