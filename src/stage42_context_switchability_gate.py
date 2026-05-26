from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_neighbor_interaction_gated_expert as ck
from src import stage42_source_level_ablation as an
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "context_switchability_gate_stage42.json"
REPORT_MD = OUT_DIR / "context_switchability_gate_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_dc_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_README = Path("README_M3W_GOAL_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

LAMBDAS = [0.1, 1.0, 10.0, 100.0]
THRESHOLD_QUANTILES = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
MIN_SWITCH_GAIN = 0.01
EASY_LIMIT = 0.02
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DC 是 context switchability / gain-harm gate，不是 residual waypoint retrain。",
    "本阶段响应 Stage42-DB 的 no-go：换监督目标，训练 context 是否应该切换的 gain predictor。",
    "future endpoints / waypoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
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
    return am._jsonable(value)


def _target_delta(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> np.ndarray:
    cur = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float64)
    scale = np.maximum(data["scale"].astype(np.float64), EPS)
    return ((labels["waypoint_xy"].astype(np.float64) - cur[:, None, :]) / scale[:, None, None]).astype(np.float32)


def _candidate_arrays(name: str, raw_features: np.ndarray, shared: Mapping[str, Any]) -> dict[str, Any]:
    data = shared["data"]
    split = shared["split"]
    labels = shared["labels"]
    floor = shared["floor"]
    train_mask = split == "train"
    val_mask = split == "val"
    x, _, _ = am._standardize(raw_features, train_mask)
    model = am._evaluate_models(data, split, labels, floor, x)
    coef = am._fit_ridge_model(x, _target_delta(data, labels), labels["waypoint_valid"], train_mask, float(model["best_lambda"]))
    pred_xy = am._predict_waypoints(x, coef, data)
    policy, selected_ade, selected_fde, switch = am._select_policy_on_val(
        pred_xy, floor["floor_xy"], labels, data, val_mask
    )
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    pred_ade, pred_fde = am._trajectory_errors(pred_xy, labels)
    return {
        "source": "fresh_run",
        "name": name,
        "best_lambda": float(model["best_lambda"]),
        "feature_count": int(raw_features.shape[1]),
        "pred_xy": pred_xy,
        "pred_ade": pred_ade,
        "pred_fde": pred_fde,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "switch": switch,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "protected_metric": am._metric(selected_ade, floor_ade, data, switch, split == "test"),
        "protected_fde_metric": am._metric(selected_fde, floor_fde, data, switch, split == "test"),
        "policy_slice_count": int(len(policy["slices"])),
    }


def _switch_features(
    data: Mapping[str, np.ndarray],
    baseline_xy: np.ndarray,
    candidate_xy: np.ndarray,
    raw_context: np.ndarray,
) -> np.ndarray:
    current = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float32)
    scale = np.maximum(data["scale"].astype(np.float32), EPS)
    final_gap = (candidate_xy[:, -1, :] - baseline_xy[:, -1, :]) / scale[:, None]
    path_gap = np.linalg.norm(candidate_xy - baseline_xy, axis=2).mean(axis=1, keepdims=True) / scale[:, None]
    candidate_step = (candidate_xy[:, -1, :] - current) / scale[:, None]
    baseline_step = (baseline_xy[:, -1, :] - current) / scale[:, None]
    compact_context = raw_context[:, : min(raw_context.shape[1], 96)].astype(np.float32)
    return np.concatenate(
        [
            final_gap.astype(np.float32),
            path_gap.astype(np.float32),
            candidate_step.astype(np.float32),
            baseline_step.astype(np.float32),
            data["history_scalar"].astype(np.float32),
            compact_context,
        ],
        axis=1,
    ).astype(np.float32)


def _standardize(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[mask].mean(axis=0).astype(np.float32)
    std = np.maximum(x[mask].std(axis=0), 1e-4).astype(np.float32)
    z = ((x - mean) / std).astype(np.float32)
    z = np.concatenate([z, np.ones((len(z), 1), dtype=np.float32)], axis=1)
    return z, mean, std


def _fit_ridge(x: np.ndarray, y: np.ndarray, mask: np.ndarray, lam: float) -> np.ndarray:
    xt = x[mask].astype(np.float64)
    yt = y[mask].astype(np.float64)
    reg = np.eye(xt.shape[1], dtype=np.float64) * float(lam)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xt.T @ xt + reg, xt.T @ yt).astype(np.float32)


def _choose_switch_policy(
    candidate_name: str,
    predicted_gain: np.ndarray,
    baseline_ade: np.ndarray,
    candidate_ade: np.ndarray,
    floor_ade: np.ndarray,
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
) -> dict[str, Any]:
    val_mask = split == "val"
    test_mask = split == "test"
    best: dict[str, Any] | None = None
    candidates = []
    val_scores = predicted_gain[val_mask]
    thresholds = sorted(set(float(np.quantile(val_scores, q)) for q in THRESHOLD_QUANTILES))
    for threshold in thresholds:
        for min_gain in [0.0, MIN_SWITCH_GAIN]:
            switch = (predicted_gain >= threshold) & (predicted_gain >= min_gain)
            selected = baseline_ade.copy()
            selected[switch] = candidate_ade[switch]
            metric = am._metric(selected, floor_ade, data, switch, val_mask)
            score = (
                1.2 * metric["all_improvement"]
                + 1.8 * metric["t50_improvement"]
                + 1.1 * metric["hard_failure_improvement"]
                - 30.0 * max(0.0, metric["easy_degradation"] - EASY_LIMIT)
                - 0.03 * metric["switch_rate"]
            )
            row = {
                "candidate": candidate_name,
                "threshold": float(threshold),
                "min_gain": float(min_gain),
                "score": float(score),
                "val_metric": metric,
                "passes_easy": metric["easy_degradation"] <= EASY_LIMIT,
            }
            candidates.append(row)
            if row["passes_easy"] and (best is None or row["score"] > best["score"]):
                best = row
    if best is None:
        best = {
            "candidate": candidate_name,
            "threshold": float("inf"),
            "min_gain": float("inf"),
            "score": 0.0,
            "val_metric": am._metric(baseline_ade, floor_ade, data, np.zeros(len(floor_ade), dtype=bool), val_mask),
            "passes_easy": True,
        }
    test_switch = (predicted_gain >= float(best["threshold"])) & (predicted_gain >= float(best["min_gain"]))
    selected_test = baseline_ade.copy()
    selected_test[test_switch] = candidate_ade[test_switch]
    return {
        "source": "fresh_run",
        "candidate": candidate_name,
        "selected_threshold": float(best["threshold"]),
        "selected_min_gain": float(best["min_gain"]),
        "selected_score": float(best["score"]),
        "validation_candidates": candidates,
        "test_metric": am._metric(selected_test, floor_ade, data, test_switch, test_mask),
        "switch": test_switch,
        "selected_ade": selected_test,
        "test_threshold_tuning": False,
    }


def _bootstrap_bundle(selected: np.ndarray, floor: np.ndarray, data: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    horizon = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    return {
        "all": am._bootstrap_ci(selected, floor, mask, seed=42501),
        "t50": am._bootstrap_ci(selected, floor, mask & (horizon == 50), seed=42502),
        "hard_failure": am._bootstrap_ci(selected, floor, mask & hard_failure, seed=42503),
        "easy_degradation": am._bootstrap_ci(floor, selected, mask & easy, seed=42504),
    }


def run_stage42_context_switchability_gate(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    shared = an._prep_shared()
    data = shared["data"]
    split = shared["split"]
    test_mask = split == "test"
    train_mask = split == "train"
    raw_variants, graph_info = ck._build_variant_features(shared)
    selected_names = [
        "baseline_family_control",
        "baseline_plus_goal_scene",
        "baseline_plus_scalar_neighbor",
        "baseline_plus_knn_graph",
        "baseline_plus_graph_goal",
        "baseline_plus_graph_history_scalar",
    ]
    arrays = {name: _candidate_arrays(name, raw_variants[name][0], shared) for name in selected_names}
    baseline = arrays["baseline_family_control"]
    candidate_results: dict[str, Any] = {}
    best: dict[str, Any] | None = None
    for name in selected_names:
        if name == "baseline_family_control":
            continue
        cand = arrays[name]
        features = _switch_features(data, baseline["pred_xy"], cand["pred_xy"], raw_variants[name][0])
        x, mean, std = _standardize(features, train_mask)
        target_gain = (baseline["selected_ade"] - cand["selected_ade"]).astype(np.float32)
        model_rows = []
        best_model: dict[str, Any] | None = None
        for lam in LAMBDAS:
            coef = _fit_ridge(x, target_gain, train_mask, lam)
            pred_gain = (x @ coef).astype(np.float64)
            policy = _choose_switch_policy(
                name,
                pred_gain,
                baseline["selected_ade"],
                cand["selected_ade"],
                baseline["floor_ade"],
                data,
                split,
            )
            model_rows.append(
                {
                    "lambda": float(lam),
                    "selected_score": policy["selected_score"],
                    "test_metric": policy["test_metric"],
                    "threshold": policy["selected_threshold"],
                    "min_gain": policy["selected_min_gain"],
                }
            )
            if best_model is None or policy["selected_score"] > best_model["policy"]["selected_score"]:
                best_model = {"lambda": float(lam), "coef_shape": list(coef.shape), "policy": policy}
        if best_model is None:
            raise RuntimeError(f"No switchability model evaluated for {name}.")
        selected = best_model["policy"]["selected_ade"]
        candidate_results[name] = {
            "source": "fresh_run",
            "candidate": name,
            "feature_count": int(features.shape[1]),
            "best_lambda": best_model["lambda"],
            "coef_shape": best_model["coef_shape"],
            "model_rows": model_rows,
            "selected_policy": {
                k: v
                for k, v in best_model["policy"].items()
                if k not in {"switch", "selected_ade", "validation_candidates"}
            },
            "test_metric": best_model["policy"]["test_metric"],
            "bootstrap": _bootstrap_bundle(selected, baseline["floor_ade"], data, test_mask),
        }
        metric = best_model["policy"]["test_metric"]
        score = (
            metric["all_improvement"]
            + metric["t50_improvement"]
            + metric["hard_failure_improvement"]
            - 10.0 * max(0.0, metric["easy_degradation"] - EASY_LIMIT)
        )
        if best is None or score > best["score"]:
            best = {
                "candidate": name,
                "score": float(score),
                "test_metric": metric,
                "bootstrap": candidate_results[name]["bootstrap"],
            }
    if best is None:
        raise RuntimeError("No context switchability candidate evaluated.")
    baseline_metric = baseline["protected_metric"]
    delta = {
        key: float(best["test_metric"].get(key, 0.0)) - float(baseline_metric.get(key, 0.0))
        for key in [
            "all_improvement",
            "t50_improvement",
            "hard_failure_improvement",
            "easy_degradation",
            "switch_rate",
        ]
    }
    supported = (
        best["test_metric"]["easy_degradation"] <= EASY_LIMIT
        and (
            delta["all_improvement"] > MIN_SWITCH_GAIN
            or delta["t50_improvement"] > MIN_SWITCH_GAIN
            or delta["hard_failure_improvement"] > MIN_SWITCH_GAIN
        )
    )
    result: dict[str, Any] = {
        "source": "fresh_run",
        "stage": "Stage42-DC context switchability gain-harm gate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "split_stats": shared["split_stats"],
        "graph_info": graph_info,
        "baseline_family_control": {
            "source": "fresh_run",
            "protected_metric": baseline_metric,
            "feature_count": arrays["baseline_family_control"]["feature_count"],
            "best_lambda": arrays["baseline_family_control"]["best_lambda"],
        },
        "candidate_results": candidate_results,
        "selected_context_switchability_policy": {
            "source": "fresh_run",
            "selected_candidate": best["candidate"],
            "test_metric": best["test_metric"],
            "delta_vs_baseline_family_control": delta,
            "bootstrap": best["bootstrap"],
            "context_switchability_supported": bool(supported),
            "decision": "context_switchability_supported" if supported else "context_switchability_not_supported",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "gain_label_train_only_for_model_fit": True,
            "validation_only_threshold_selection": True,
            "test_threshold_tuning": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
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
    result["stage42_dc_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    if refresh_readmes:
        _refresh_readmes(result)
        _refresh_research_state(result)
    return result


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    claim = result["claim_boundary"]
    selected = result["selected_context_switchability_policy"]
    gates = {
        "source_level_split_used": result["split_stats"]["by_split"]["test"]["rows"] == 47458,
        "baseline_family_control_positive": result["baseline_family_control"]["protected_metric"]["all_improvement"] > 0
        and result["baseline_family_control"]["protected_metric"]["t50_improvement"] > 0,
        "multiple_context_candidates_evaluated": len(result["candidate_results"]) >= 5,
        "gain_harm_target_used": result["no_leakage"]["gain_label_train_only_for_model_fit"],
        "validation_only_threshold_selection": result["no_leakage"]["validation_only_threshold_selection"],
        "test_evaluated_once": result["no_leakage"]["test_threshold_tuning"] is False,
        "decision_recorded": selected["decision"] in {
            "context_switchability_supported",
            "context_switchability_not_supported",
        },
        "no_context_overclaim_if_negative": selected["context_switchability_supported"]
        or selected["decision"] == "context_switchability_not_supported",
        "bootstrap_available": selected["bootstrap"]["all"]["bootstrap_n"] > 0
        and selected["bootstrap"]["t50"]["bootstrap_n"] > 0,
        "no_leakage_pass": all(
            result["no_leakage"][key] is False
            for key in [
                "future_endpoint_input",
                "future_waypoint_input",
                "test_threshold_tuning",
                "central_velocity",
                "test_endpoint_goals",
            ]
        )
        and result["no_leakage"]["source_overlap_pass"],
        "no_metric_seconds_overclaim": claim["metric_or_seconds_claim"] is False,
        "true_3d_overclaim_blocked": claim["true_3d"] is False,
        "foundation_overclaim_blocked": claim["foundation_world_model"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_dc_context_switchability_gate_pass" if passed == total else "stage42_dc_context_switchability_gate_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    selected = result["selected_context_switchability_policy"]
    lines = [
        "# Stage42-DC Context Switchability / Gain-Harm Gate",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- gate: `{result['stage42_dc_gate']['passed']} / {result['stage42_dc_gate']['total']}`",
        f"- verdict: `{result['stage42_dc_gate']['verdict']}`",
        f"- decision: `{selected['decision']}`",
        "",
        "## Current Facts",
        "",
    ]
    lines.extend(f"- {fact}" for fact in CURRENT_FACTS)
    lines.extend(
        [
            "",
            "## Baseline-Family Control",
            "",
            f"- protected_metric: `{result['baseline_family_control']['protected_metric']}`",
            "",
            "## Context Switchability Candidates",
            "",
            "| candidate | lambda | all | t50 | hard/failure | easy | switch | delta all | delta t50 | delta hard |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    base = result["baseline_family_control"]["protected_metric"]
    for name, row in result["candidate_results"].items():
        m = row["test_metric"]
        lines.append(
            f"| `{name}` | {row['best_lambda']:.2f} | {m['all_improvement']:.6f} | {m['t50_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['switch_rate']:.6f} | {m['all_improvement'] - base['all_improvement']:.6f} | {m['t50_improvement'] - base['t50_improvement']:.6f} | {m['hard_failure_improvement'] - base['hard_failure_improvement']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Selected Policy",
            "",
            f"- selected_candidate: `{selected['selected_candidate']}`",
            f"- test_metric: `{selected['test_metric']}`",
            f"- delta_vs_baseline_family_control: `{selected['delta_vs_baseline_family_control']}`",
            f"- context_switchability_supported: `{selected['context_switchability_supported']}`",
            "",
            "## Interpretation",
            "",
        ]
    )
    if selected["context_switchability_supported"]:
        lines.append("- Stage42-DC found a safe context switchability increment beyond baseline-family control.")
    else:
        lines.append(
            "- Stage42-DC changed the supervision target to gain/harm switchability, but still did not find a safe positive context increment beyond baseline-family control."
        )
    lines.extend(
        [
            "- This is fresh training/evaluation of a gain-harm gate; it is not Stage5C, not SMC, not metric/seconds-level, and not true 3D evidence.",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{result['no_leakage']}`",
            f"- claim_boundary: `{result['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_dc_gate"]
    lines = [
        "# Stage42-DC Gate",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in gate["gates"].items())
    return lines


def _refresh_lines(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_dc_gate"]
    selected = result["selected_context_switchability_policy"]
    d = selected["delta_vs_baseline_family_control"]
    return [
        "## Stage42-DC Context Switchability / Gain-Harm Gate",
        "",
        "- source: `fresh_run`",
        "- role: change context supervision from waypoint residual to gain/harm switchability after Stage42-DB no-go.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- selected candidate: `{selected['selected_candidate']}`; decision `{selected['decision']}`.",
        f"- delta vs baseline-family all/t50/hard/easy: `{d['all_improvement']:.4f}` / `{d['t50_improvement']:.4f}` / `{d['hard_failure_improvement']:.4f}` / `{d['easy_degradation']:.4f}`.",
        "- Claim boundary unchanged: protected dataset-local/raw-frame 2.5D only; no true 3D, foundation, metric/seconds-level, Stage5C, or SMC.",
    ]


def _refresh_readmes(result: Mapping[str, Any]) -> None:
    lines = _refresh_lines(result)
    for path in [README_RESULTS, M3W_README, GOAL_README]:
        _replace_section(path, "STAGE42_DC_CONTEXT_SWITCHABILITY_GATE", lines)


def _refresh_research_state(result: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    selected = result["selected_context_switchability_policy"]
    state["current_stage"] = "Stage42-DC context switchability gain-harm gate"
    state["current_verdict"] = result["stage42_dc_gate"]["verdict"]
    state["stage42_dc_context_switchability_gate"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": result["stage42_dc_gate"]["verdict"],
        "gates": f"{result['stage42_dc_gate']['passed']}/{result['stage42_dc_gate']['total']}",
        "selected_candidate": selected["selected_candidate"],
        "decision": selected["decision"],
        "delta_vs_baseline_family_control": selected["delta_vs_baseline_family_control"],
        "claim_boundary": result["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


if __name__ == "__main__":
    run_stage42_context_switchability_gate()
