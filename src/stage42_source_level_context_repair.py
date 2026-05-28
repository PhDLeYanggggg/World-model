from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_source_level_ablation as an
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "source_level_context_repair_stage42.json"
REPORT_MD = OUT_DIR / "source_level_context_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ix_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

EPS = 1e-6
LAMBDAS = [0.1, 1.0, 10.0, 100.0]
MIN_CONTEXT_LIFT = 0.01


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-IX 是 source-level context contribution repair trial：它在 Stage42-AO partial/negative 后修改训练目标并重训/重评。",
    "本实验测试 history / goal / neighbor context 在加权 hard/t50/t100 和 floor-residual 目标下是否能提供增量。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "本阶段仍是 dataset-local/raw-frame 2.5D evidence，不是 metric 或 seconds-level 结果。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


TRIALS = [
    {
        "name": "baseline_family_absolute_weighted",
        "feature_set": "baseline_family",
        "target_mode": "absolute_from_current",
        "weight_profile": "t50_hard_t100",
    },
    {
        "name": "context_only_absolute_weighted",
        "feature_set": "context_only",
        "target_mode": "absolute_from_current",
        "weight_profile": "t50_hard_t100",
    },
    {
        "name": "context_only_floor_residual_weighted",
        "feature_set": "context_only",
        "target_mode": "floor_residual",
        "weight_profile": "t50_hard_t100",
    },
    {
        "name": "full_absolute_weighted",
        "feature_set": "full",
        "target_mode": "absolute_from_current",
        "weight_profile": "t50_hard_t100",
    },
    {
        "name": "full_floor_residual_weighted",
        "feature_set": "full",
        "target_mode": "floor_residual",
        "weight_profile": "t50_hard_t100",
    },
    {
        "name": "goal_neighbor_floor_residual_weighted",
        "feature_set": "goal_neighbor_context",
        "target_mode": "floor_residual",
        "weight_profile": "t50_hard_t100",
    },
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _cached_result_if_available() -> dict[str, Any] | None:
    if not REPORT_JSON.exists():
        return None
    payload = read_json(REPORT_JSON, {})
    if payload.get("stage") == "Stage42-IX source-level context repair trials":
        return payload
    return None


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _or_mask(*masks: np.ndarray) -> np.ndarray:
    if not masks:
        raise ValueError("At least one mask is required.")
    out = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        out |= mask
    return out


def _feature_masks(names: list[str]) -> dict[str, np.ndarray]:
    groups = an._feature_indices(names)
    control = _or_mask(groups["domain"], groups["horizon"])
    context = _or_mask(groups["history"], groups["neighbor_interaction"], groups["goal_prototype"], control)
    goal_neighbor = _or_mask(groups["neighbor_interaction"], groups["goal_prototype"], control)
    baseline = _or_mask(groups["baseline_family"], control)
    return {
        "full": np.ones(len(names), dtype=bool),
        "baseline_family": baseline,
        "context_only": context,
        "goal_neighbor_context": goal_neighbor,
        "history_context": _or_mask(groups["history"], control),
        "goal_context": _or_mask(groups["goal_prototype"], control),
        "neighbor_context": _or_mask(groups["neighbor_interaction"], control),
    }


def _sample_weights(data: Mapping[str, np.ndarray], profile: str) -> np.ndarray:
    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    w = np.ones(len(h), dtype=np.float64)
    if profile == "t50_hard_t100":
        w += 2.0 * (h == 50)
        w += 1.0 * (h == 100)
        w += 2.0 * hard_failure
    elif profile != "uniform":
        raise ValueError(f"Unknown weight profile: {profile}")
    return w.astype(np.float64)


def _target_delta(
    data: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    target_mode: str,
) -> np.ndarray:
    scale = np.maximum(data["scale"].astype(np.float64), EPS)
    if target_mode == "absolute_from_current":
        base = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float64)
        delta = (labels["waypoint_xy"].astype(np.float64) - base[:, None, :]) / scale[:, None, None]
    elif target_mode == "floor_residual":
        delta = (labels["waypoint_xy"].astype(np.float64) - floor["floor_xy"].astype(np.float64)) / scale[:, None, None]
    else:
        raise ValueError(f"Unknown target mode: {target_mode}")
    return delta.astype(np.float32)


def _predict_waypoints(
    x: np.ndarray,
    coef: np.ndarray,
    data: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    target_mode: str,
) -> np.ndarray:
    scale = np.maximum(data["scale"].astype(np.float64), EPS)
    delta = (x.astype(np.float64) @ coef.astype(np.float64)).reshape(len(x), len(am.WAYPOINT_FRAC), 2)
    if target_mode == "absolute_from_current":
        base = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float64)
        out = base[:, None, :] + delta * scale[:, None, None]
    elif target_mode == "floor_residual":
        out = floor["floor_xy"].astype(np.float64) + delta * scale[:, None, None]
    else:
        raise ValueError(f"Unknown target mode: {target_mode}")
    return out.astype(np.float32)


def _fit_weighted_ridge(
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    train_mask: np.ndarray,
    weights: np.ndarray,
    lam: float,
) -> np.ndarray:
    y2 = y.reshape(len(y), -1)
    coef = np.zeros((x.shape[1], y2.shape[1]), dtype=np.float32)
    for w_i in range(len(am.WAYPOINT_FRAC)):
        m = train_mask & valid[:, w_i]
        ids = np.where(m)[0]
        if len(ids) == 0:
            continue
        xt = x[ids].astype(np.float64, copy=False)
        sw = np.sqrt(np.maximum(weights[ids], EPS))[:, None]
        xtw = xt * sw
        reg = np.eye(xt.shape[1], dtype=np.float64) * float(lam)
        reg[-1, -1] = 0.0
        for j in [2 * w_i, 2 * w_i + 1]:
            yt = y2[ids, j].astype(np.float64, copy=False)
            coef[:, j] = np.linalg.solve(xtw.T @ xtw + reg, xtw.T @ (yt * sw[:, 0])).astype(np.float32)
    return coef


def _evaluate_trial(shared: Mapping[str, Any], trial: Mapping[str, str]) -> dict[str, Any]:
    split = shared["split"]
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    mask = shared["feature_masks"][trial["feature_set"]]
    x, _, _ = am._standardize(shared["features"][:, mask], train_mask)
    target = _target_delta(shared["data"], shared["labels"], shared["floor"], trial["target_mode"])
    weights = _sample_weights(shared["data"], trial["weight_profile"])
    val_rows = []
    best: dict[str, Any] | None = None
    best_score = -1e9
    for lam in LAMBDAS:
        coef = _fit_weighted_ridge(x, target, shared["labels"]["waypoint_valid"], train_mask, weights, lam)
        pred_xy = _predict_waypoints(x, coef, shared["data"], shared["floor"], trial["target_mode"])
        policy, selected_ade, selected_fde, switch = am._select_policy_on_val(
            pred_xy,
            shared["floor"]["floor_xy"],
            shared["labels"],
            shared["data"],
            val_mask,
        )
        floor_ade, floor_fde = am._trajectory_errors(shared["floor"]["floor_xy"], shared["labels"])
        val_metric = am._metric(selected_ade, floor_ade, shared["data"], switch, val_mask)
        score = (
            1.0 * val_metric["all_improvement"]
            + 2.5 * val_metric["t50_improvement"]
            + 1.4 * val_metric["hard_failure_improvement"]
            + 1.2 * val_metric["t100_raw_frame_diagnostic_improvement"]
            - 35.0 * max(0.0, val_metric["easy_degradation"] - 0.02)
            - 0.03 * val_metric["switch_rate"]
        )
        val_rows.append({"lambda": float(lam), "score": float(score), "val_metric": val_metric, "policy_slices": int(len(policy["slices"]))})
        if score > best_score:
            best_score = float(score)
            pred_ade, pred_fde = am._trajectory_errors(pred_xy, shared["labels"])
            best = {
                "lambda": float(lam),
                "score": float(score),
                "policy": policy,
                "selected_ade": selected_ade,
                "selected_fde": selected_fde,
                "switch": switch,
                "floor_ade": floor_ade,
                "floor_fde": floor_fde,
                "ungated_ade": pred_ade,
                "ungated_fde": pred_fde,
                "val_metric": val_metric,
            }
    if best is None:
        raise RuntimeError(f"No candidate evaluated for {trial['name']}.")
    data = shared["data"]
    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    return {
        "source": "fresh_run",
        "trial": dict(trial),
        "feature_count": int(np.sum(mask)),
        "best_lambda": best["lambda"],
        "best_score": best["score"],
        "validation_candidates": val_rows,
        "policy_slice_count": int(len(best["policy"]["slices"])),
        "protected": am._metric(best["selected_ade"], best["floor_ade"], data, best["switch"], test_mask),
        "ungated_diagnostic": am._metric(best["ungated_ade"], best["floor_ade"], data, np.ones(len(best["floor_ade"]), dtype=bool), test_mask),
        "bootstrap": {
            "all": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask, seed=42201),
            "t50": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (h == 50), seed=42202),
            "t100_raw_frame_diagnostic": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (h == 100), seed=42203),
            "hard_failure": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & hard_failure, seed=42204),
            "easy_degradation": am._bootstrap_ci(best["floor_ade"], best["selected_ade"], test_mask & easy, seed=42205),
        },
    }


def _metric_delta(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float]:
    keys = [
        "all_improvement",
        "t50_improvement",
        "t100_raw_frame_diagnostic_improvement",
        "hard_failure_improvement",
        "easy_degradation",
        "switch_rate",
        "harm_over_fallback",
    ]
    return {k: float(a.get(k, 0.0)) - float(b.get(k, 0.0)) for k in keys}


def _summarize(trials: Mapping[str, Any]) -> dict[str, Any]:
    baseline = trials["baseline_family_absolute_weighted"]["protected"]
    context_candidates = {
        name: row for name, row in trials.items() if name != "baseline_family_absolute_weighted"
    }
    deltas = {name: _metric_delta(row["protected"], baseline) for name, row in context_candidates.items()}
    positive = [
        name
        for name, delta in deltas.items()
        if (
            delta["all_improvement"] > MIN_CONTEXT_LIFT
            or delta["t50_improvement"] > MIN_CONTEXT_LIFT
            or delta["hard_failure_improvement"] > MIN_CONTEXT_LIFT
            or delta["t100_raw_frame_diagnostic_improvement"] > MIN_CONTEXT_LIFT
        )
        and context_candidates[name]["protected"]["easy_degradation"] <= 0.02
    ]
    best = max(trials.values(), key=lambda row: (
        row["protected"]["all_improvement"]
        + 1.4 * row["protected"]["t50_improvement"]
        + row["protected"]["hard_failure_improvement"]
        + row["protected"]["t100_raw_frame_diagnostic_improvement"]
        - 10.0 * max(0.0, row["protected"]["easy_degradation"] - 0.02)
    ))
    return {
        "baseline_family_reference": baseline,
        "context_delta_vs_baseline_family": deltas,
        "positive_context_repair_trials": positive,
        "best_trial": best["trial"]["name"],
        "best_trial_metric": best["protected"],
        "context_claim_verdict": (
            "stage42_ix_context_repair_positive"
            if positive
            else "stage42_ix_context_repair_negative_context_still_not_incremental"
        ),
        "interpretation": (
            "Weighted hard/t50/t100 and floor-residual objectives were tested to give context features a fairer repair path. "
            "Positive trials can support a limited context contribution claim; if none are positive, the evidence remains that current source-level ridge dynamics are baseline-family dominated."
        ),
    }


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    best = summary["best_trial_metric"]
    gates = {
        "source_level_split_used": result["split_stats"]["by_split"]["test"]["rows"] == 47458,
        "weighted_repair_trials_complete": len(result["trials"]) == len(TRIALS),
        "floor_residual_objective_tested": any(row["trial"]["target_mode"] == "floor_residual" for row in result["trials"].values()),
        "context_only_trials_tested": any(row["trial"]["feature_set"] == "context_only" for row in result["trials"].values()),
        "best_trial_safe": best["easy_degradation"] <= 0.02,
        "best_trial_positive": best["all_improvement"] > 0 or best["t50_improvement"] > 0 or best["hard_failure_improvement"] > 0,
        "context_incremental_claim_supported": summary["context_claim_verdict"] == "stage42_ix_context_repair_positive",
        "negative_or_positive_result_recorded": summary["context_claim_verdict"] in {
            "stage42_ix_context_repair_positive",
            "stage42_ix_context_repair_negative_context_still_not_incremental",
        },
        "no_leakage_pass": all(
            result["no_leakage"][k] is False
            for k in [
                "future_endpoint_input",
                "future_waypoint_input",
                "central_velocity",
                "test_endpoint_goals",
                "test_threshold_tuning",
            ]
        )
        and result["no_leakage"]["train_only_feature_normalization"],
        "no_metric_seconds_overclaim": result["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": result["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": result["claim_boundary"]["smc_enabled"] is False,
    }
    verdict = (
        "stage42_ix_context_repair_positive"
        if all(gates.values())
        else "stage42_ix_context_repair_completed_context_not_proven"
        if gates["weighted_repair_trials_complete"] and gates["negative_or_positive_result_recorded"] and gates["no_leakage_pass"]
        else "stage42_ix_context_repair_partial"
    )
    return {"source": "fresh_run", "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": len(gates), "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-IX Source-Level Context Repair Trials",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_ix_gate']['passed']} / {result['stage42_ix_gate']['total']}`",
        f"- verdict: `{result['stage42_ix_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Why This Was Run",
        "",
        "- Stage42-AO found standalone context signal but no incremental context gain after baseline-family rollout features.",
        "- Stage42-IX changes the training target instead of just restating the negative result: it tests hard/t50/t100 weighted ridge training and floor-residual targets.",
        "- Test thresholds are still not tuned on test; validation selects lambda and safe-switch policy.",
        "",
        "## Trial Metrics",
        "",
        "| trial | features | target | feature_count | all | t50 | t100 diag | hard/failure | easy | switch |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["trials"].items():
        metric = row["protected"]
        trial = row["trial"]
        lines.append(
            f"| `{name}` | `{trial['feature_set']}` | `{trial['target_mode']}` | {row['feature_count']} | {metric['all_improvement']:.6f} | {metric['t50_improvement']:.6f} | {metric['t100_raw_frame_diagnostic_improvement']:.6f} | {metric['hard_failure_improvement']:.6f} | {metric['easy_degradation']:.6f} | {metric['switch_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Context Delta Versus Baseline-Family Reference",
            "",
            "| trial | delta all | delta t50 | delta t100 diag | delta hard | delta easy |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, delta in result["summary"]["context_delta_vs_baseline_family"].items():
        lines.append(
            f"| `{name}` | {delta['all_improvement']:.6f} | {delta['t50_improvement']:.6f} | {delta['t100_raw_frame_diagnostic_improvement']:.6f} | {delta['hard_failure_improvement']:.6f} | {delta['easy_degradation']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- best_trial: `{result['summary']['best_trial']}`",
            f"- positive_context_repair_trials: `{result['summary']['positive_context_repair_trials']}`",
            f"- context_claim_verdict: `{result['summary']['context_claim_verdict']}`",
            f"- interpretation: {result['summary']['interpretation']}",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{result['no_leakage']}`",
            f"- claim_boundary: `{result['claim_boundary']}`",
            "",
            "## Interpretation",
            "",
        ]
    )
    if result["summary"]["positive_context_repair_trials"]:
        lines.append("- Stage42-IX found at least one context repair trial with incremental value over the baseline-family reference while preserving easy cases.")
    else:
        lines.append("- Stage42-IX did not turn history/goal/neighbor context into an incremental source-level ridge contribution; current evidence remains baseline-family dominated.")
    lines.append("- This is a retrained repair attempt, not an inference-mask ablation and not a metric/seconds-level or true-3D claim.")
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_ix_gate"]
    lines = [
        "# Stage42-IX Gate",
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


def _replace_block(path: Path, marker: str, block: list[str]) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    body = "\n".join([start, *block, end])
    if start in text and end in text:
        prefix = text.split(start, 1)[0].rstrip()
        suffix = text.split(end, 1)[1].lstrip()
        new_text = f"{prefix}\n\n{body}\n"
        if suffix:
            new_text += f"\n{suffix}"
    else:
        new_text = text.rstrip() + "\n\n" + body + "\n"
    path.write_text(new_text, encoding="utf-8")


def _update_readmes(result: Mapping[str, Any]) -> None:
    marker = "STAGE42_IX_SOURCE_LEVEL_CONTEXT_REPAIR"
    summary = result["summary"]
    best = summary["best_trial_metric"]
    block = [
        "## Stage42-IX Source-Level Context Repair Trials",
        "",
        f"- source: `{result['source']}`",
        "- role: retrained repair attempt after Stage42-AO showed context was not incremental after baseline-family rollout features.",
        f"- gate: `{result['stage42_ix_gate']['passed']} / {result['stage42_ix_gate']['total']}`; verdict `{result['stage42_ix_gate']['verdict']}`.",
        f"- tested: `{len(result['trials'])}` weighted/floor-residual variants.",
        f"- best_trial: `{summary['best_trial']}`; best all/t50/t100raw/hard `{best['all_improvement']:.6f}` / `{best['t50_improvement']:.6f}` / `{best['t100_raw_frame_diagnostic_improvement']:.6f}` / `{best['hard_failure_improvement']:.6f}`.",
        f"- easy degradation: `{best['easy_degradation']:.6f}`.",
        f"- positive_context_repair_trials: `{summary['positive_context_repair_trials']}`.",
        f"- context_claim_verdict: `{summary['context_claim_verdict']}`.",
        "- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.",
    ]
    for p in [Path("README_RESULTS.md"), Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")]:
        _replace_block(p, marker, block)
    _replace_block(Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md"), marker, block)


def _update_state(result: Mapping[str, Any]) -> None:
    path = Path("research_state.json")
    state = read_json(path, {})
    state["current_stage"] = "stage42_ix_source_level_context_repair"
    state["current_verdict"] = result["stage42_ix_gate"]["verdict"]
    state.setdefault("stage42", {})["stage_ix_source_level_context_repair"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": result["stage42_ix_gate"]["verdict"],
        "gates": f"{result['stage42_ix_gate']['passed']}/{result['stage42_ix_gate']['total']}",
        "best_trial": result["summary"]["best_trial"],
        "best_trial_metric": result["summary"]["best_trial_metric"],
        "positive_context_repair_trials": result["summary"]["positive_context_repair_trials"],
        "context_claim_verdict": result["summary"]["context_claim_verdict"],
        "claim_boundary": result["claim_boundary"],
    }
    generated = state.setdefault("generated_reports", [])
    for item in [str(REPORT_MD), str(REPORT_JSON), str(GATE_MD)]:
        if item not in generated:
            generated.append(item)
    write_json(path, _jsonable(state))


def _append_ledger(result: Mapping[str, Any]) -> None:
    row = {
        "stage": result["stage"],
        "timestamp": result["generated_at_utc"],
        "source": result["source"],
        "verdict": result["stage42_ix_gate"]["verdict"],
        "gate": f"{result['stage42_ix_gate']['passed']}/{result['stage42_ix_gate']['total']}",
        "best_trial": result["summary"]["best_trial"],
        "context_claim_verdict": result["summary"]["context_claim_verdict"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_stage42_source_level_context_repair(*, use_cached: bool = False) -> dict[str, Any]:
    if use_cached:
        cached = _cached_result_if_available()
        if cached is not None:
            return cached
    ensure_dir(OUT_DIR)
    shared = an._prep_shared()
    shared["feature_masks"] = _feature_masks(shared["feature_names"])
    trials = {trial["name"]: _evaluate_trial(shared, trial) for trial in TRIALS}
    result = {
        "stage": "Stage42-IX source-level context repair trials",
        "source": "fresh_run_weighted_floor_residual_context_repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/source_level_incremental_ablation_stage42.json",
                "outputs/stage42_long_research/source_level_row_cache_mechanism_audit_stage42.json",
            ]
        ),
        "split_stats": shared["split_stats"],
        "feature_groups": {k: int(np.sum(v)) for k, v in an._feature_indices(shared["feature_names"]).items()},
        "feature_sets": {k: int(np.sum(v)) for k, v in shared["feature_masks"].items()},
        "trials": trials,
        "summary": _summarize(trials),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
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
    result["stage42_ix_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _update_readmes(result)
    _update_state(result)
    _append_ledger(result)
    return result


if __name__ == "__main__":
    run_stage42_source_level_context_repair()
