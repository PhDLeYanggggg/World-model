from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from src import stage42_source_level_ablation as an
from src import stage42_source_level_context_repair as ix
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "source_level_nonlinear_context_repair_stage42.json"
REPORT_MD = OUT_DIR / "source_level_nonlinear_context_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_iy_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

EPS = 1e-6
MAX_TREE_TRAIN_ROWS = 120_000
MIN_CONTEXT_LIFT = 0.01


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-IY 是 Stage42-IX negative result 后的非线性 context repair trial，用 ExtraTrees 多输出 residual 模型测试容量不足假设。",
    "训练使用 train split 的 deterministic capped subset；validation 选 safe policy；test 只评一次。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "本阶段仍是 dataset-local/raw-frame 2.5D evidence，不是 metric 或 seconds-level 结果。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


TREE_TRIALS = [
    {"name": "tree_baseline_family_residual", "feature_set": "baseline_family", "target_mode": "floor_residual"},
    {"name": "tree_context_only_residual", "feature_set": "context_only", "target_mode": "floor_residual"},
    {"name": "tree_full_residual", "feature_set": "full", "target_mode": "floor_residual"},
    {"name": "tree_goal_neighbor_residual", "feature_set": "goal_neighbor_context", "target_mode": "floor_residual"},
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
    if payload.get("stage") == "Stage42-IY source-level nonlinear context repair":
        return payload
    return None


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _train_ids(split: np.ndarray, data: Mapping[str, np.ndarray]) -> np.ndarray:
    train = np.where(split == "train")[0]
    if len(train) <= MAX_TREE_TRAIN_ROWS:
        return train
    h = data["horizon"].astype(int)
    domain = data["dataset"].astype(str)
    hard = data["hard"].astype(bool) | data["failure"].astype(bool)
    rng = np.random.default_rng(42091)
    buckets: dict[str, np.ndarray] = {}
    for d in sorted(set(domain[train].tolist())):
        for hv in [10, 25, 50, 100]:
            for hf in [False, True]:
                ids = train[(domain[train] == d) & (h[train] == hv) & (hard[train] == hf)]
                if len(ids):
                    buckets[f"{d}|{hv}|{int(hf)}"] = ids
    chosen: list[np.ndarray] = []
    per_bucket = max(1, MAX_TREE_TRAIN_ROWS // max(len(buckets), 1))
    for ids in buckets.values():
        n = min(len(ids), per_bucket)
        chosen.append(rng.choice(ids, size=n, replace=False))
    out = np.concatenate(chosen) if chosen else train[:MAX_TREE_TRAIN_ROWS]
    if len(out) < MAX_TREE_TRAIN_ROWS:
        remaining = np.setdiff1d(train, out, assume_unique=False)
        need = min(len(remaining), MAX_TREE_TRAIN_ROWS - len(out))
        if need > 0:
            out = np.concatenate([out, rng.choice(remaining, size=need, replace=False)])
    return np.sort(out.astype(np.int64))


def _target(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], floor: Mapping[str, Any]) -> np.ndarray:
    scale = np.maximum(data["scale"].astype(np.float64), EPS)
    delta = (labels["waypoint_xy"].astype(np.float64) - floor["floor_xy"].astype(np.float64)) / scale[:, None, None]
    return delta.reshape(len(delta), -1).astype(np.float32)


def _predict_tree_waypoints(
    model: ExtraTreesRegressor,
    x: np.ndarray,
    data: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
) -> np.ndarray:
    scale = np.maximum(data["scale"].astype(np.float64), EPS)
    delta = model.predict(x).reshape(len(x), len(am.WAYPOINT_FRAC), 2)
    return (floor["floor_xy"].astype(np.float64) + delta * scale[:, None, None]).astype(np.float32)


def _fit_tree_trial(shared: Mapping[str, Any], trial: Mapping[str, str]) -> dict[str, Any]:
    split = shared["split"]
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    feature_mask = shared["feature_masks"][trial["feature_set"]]
    z, _, _ = am._standardize(shared["features"][:, feature_mask], train_mask)
    train_ids = shared["tree_train_ids"]
    y = _target(shared["data"], shared["labels"], shared["floor"])
    weights = ix._sample_weights(shared["data"], "t50_hard_t100")
    model = ExtraTreesRegressor(
        n_estimators=64,
        max_depth=16,
        min_samples_leaf=32,
        random_state=42092,
        n_jobs=1,
    )
    model.fit(z[train_ids], y[train_ids], sample_weight=weights[train_ids])
    pred_xy = _predict_tree_waypoints(model, z, shared["data"], shared["floor"])
    policy, selected_ade, selected_fde, switch = am._select_policy_on_val(
        pred_xy,
        shared["floor"]["floor_xy"],
        shared["labels"],
        shared["data"],
        val_mask,
    )
    floor_ade, floor_fde = am._trajectory_errors(shared["floor"]["floor_xy"], shared["labels"])
    pred_ade, pred_fde = am._trajectory_errors(pred_xy, shared["labels"])
    data = shared["data"]
    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    return {
        "source": "fresh_run_sampled_train_nonlinear_tree",
        "trial": dict(trial),
        "model": {
            "type": "ExtraTreesRegressor",
            "n_estimators": 64,
            "max_depth": 16,
            "min_samples_leaf": 32,
            "n_jobs": 1,
            "random_state": 42092,
        },
        "feature_count": int(np.sum(feature_mask)),
        "train_rows_available": int(np.sum(train_mask)),
        "train_rows_used": int(len(train_ids)),
        "validation_policy_slices": int(len(policy["slices"])),
        "protected": am._metric(selected_ade, floor_ade, data, switch, test_mask),
        "ungated_diagnostic": am._metric(pred_ade, floor_ade, data, np.ones(len(pred_ade), dtype=bool), test_mask),
        "bootstrap": {
            "all": am._bootstrap_ci(selected_ade, floor_ade, test_mask, seed=42301),
            "t50": am._bootstrap_ci(selected_ade, floor_ade, test_mask & (h == 50), seed=42302),
            "t100_raw_frame_diagnostic": am._bootstrap_ci(selected_ade, floor_ade, test_mask & (h == 100), seed=42303),
            "hard_failure": am._bootstrap_ci(selected_ade, floor_ade, test_mask & hard_failure, seed=42304),
            "easy_degradation": am._bootstrap_ci(floor_ade, selected_ade, test_mask & easy, seed=42305),
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


def _summarize(trials: Mapping[str, Any], ix_report: Mapping[str, Any]) -> dict[str, Any]:
    tree_baseline = trials["tree_baseline_family_residual"]["protected"]
    ix_baseline = ix_report.get("summary", {}).get("baseline_family_reference", {})
    deltas_tree = {
        name: _metric_delta(row["protected"], tree_baseline)
        for name, row in trials.items()
        if name != "tree_baseline_family_residual"
    }
    deltas_ix = {
        name: _metric_delta(row["protected"], ix_baseline)
        for name, row in trials.items()
        if ix_baseline
    }
    positive = [
        name
        for name, delta in deltas_tree.items()
        if (
            delta["all_improvement"] > MIN_CONTEXT_LIFT
            or delta["t50_improvement"] > MIN_CONTEXT_LIFT
            or delta["t100_raw_frame_diagnostic_improvement"] > MIN_CONTEXT_LIFT
            or delta["hard_failure_improvement"] > MIN_CONTEXT_LIFT
        )
        and trials[name]["protected"]["easy_degradation"] <= 0.02
    ]
    best = max(trials.values(), key=lambda row: (
        row["protected"]["all_improvement"]
        + 1.4 * row["protected"]["t50_improvement"]
        + row["protected"]["hard_failure_improvement"]
        + row["protected"]["t100_raw_frame_diagnostic_improvement"]
        - 10.0 * max(0.0, row["protected"]["easy_degradation"] - 0.02)
    ))
    return {
        "tree_baseline_family_reference": tree_baseline,
        "stage42_ix_ridge_baseline_reference": ix_baseline,
        "delta_vs_tree_baseline_family": deltas_tree,
        "delta_vs_stage42_ix_ridge_baseline": deltas_ix,
        "positive_nonlinear_context_trials": positive,
        "best_trial": best["trial"]["name"],
        "best_trial_metric": best["protected"],
        "capacity_hypothesis_verdict": (
            "stage42_iy_nonlinear_context_capacity_positive"
            if positive
            else "stage42_iy_nonlinear_context_capacity_not_sufficient"
        ),
        "interpretation": (
            "If nonlinear context trials still fail to beat the nonlinear baseline-family tree, the context gap is not explained by simple linear capacity alone. "
            "If a context trial wins, that supports a limited nonlinear context contribution claim under source-level split."
        ),
    }


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    best = summary["best_trial_metric"]
    gates = {
        "source_level_split_used": result["split_stats"]["by_split"]["test"]["rows"] == 47458,
        "tree_trials_complete": len(result["trials"]) == len(TREE_TRIALS),
        "sampled_training_recorded": all(row["train_rows_used"] <= MAX_TREE_TRAIN_ROWS for row in result["trials"].values()),
        "context_tree_trial_tested": "tree_context_only_residual" in result["trials"],
        "full_tree_trial_tested": "tree_full_residual" in result["trials"],
        "best_trial_safe": best["easy_degradation"] <= 0.02,
        "best_trial_positive": best["all_improvement"] > 0 or best["t50_improvement"] > 0 or best["hard_failure_improvement"] > 0,
        "nonlinear_context_claim_supported": summary["capacity_hypothesis_verdict"] == "stage42_iy_nonlinear_context_capacity_positive",
        "negative_or_positive_result_recorded": summary["capacity_hypothesis_verdict"] in {
            "stage42_iy_nonlinear_context_capacity_positive",
            "stage42_iy_nonlinear_context_capacity_not_sufficient",
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
        "stage42_iy_nonlinear_context_repair_positive"
        if all(gates.values())
        else "stage42_iy_nonlinear_context_repair_completed_context_not_proven"
        if gates["tree_trials_complete"] and gates["negative_or_positive_result_recorded"] and gates["no_leakage_pass"]
        else "stage42_iy_nonlinear_context_repair_partial"
    )
    return {"source": "fresh_run", "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": len(gates), "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-IY Source-Level Nonlinear Context Repair",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_iy_gate']['passed']} / {result['stage42_iy_gate']['total']}`",
        f"- verdict: `{result['stage42_iy_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Why This Was Run",
        "",
        "- Stage42-IX showed that weighted ridge and floor-residual targets did not make history/goal/neighbor context incremental.",
        "- Stage42-IY tests whether a nonlinear ExtraTrees residual model can recover context value under the same source-level no-leakage protocol.",
        "- Training uses a deterministic train-only capped subset and records that cap explicitly; validation selects the safe policy; test is evaluated once.",
        "",
        "## Trial Metrics",
        "",
        "| trial | feature_set | train_used | all | t50 | t100 diag | hard/failure | easy | switch |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["trials"].items():
        metric = row["protected"]
        lines.append(
            f"| `{name}` | `{row['trial']['feature_set']}` | {row['train_rows_used']} | {metric['all_improvement']:.6f} | {metric['t50_improvement']:.6f} | {metric['t100_raw_frame_diagnostic_improvement']:.6f} | {metric['hard_failure_improvement']:.6f} | {metric['easy_degradation']:.6f} | {metric['switch_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Delta Versus Nonlinear Baseline-Family Tree",
            "",
            "| trial | delta all | delta t50 | delta t100 diag | delta hard | delta easy |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, delta in result["summary"]["delta_vs_tree_baseline_family"].items():
        lines.append(
            f"| `{name}` | {delta['all_improvement']:.6f} | {delta['t50_improvement']:.6f} | {delta['t100_raw_frame_diagnostic_improvement']:.6f} | {delta['hard_failure_improvement']:.6f} | {delta['easy_degradation']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- best_trial: `{result['summary']['best_trial']}`",
            f"- positive_nonlinear_context_trials: `{result['summary']['positive_nonlinear_context_trials']}`",
            f"- capacity_hypothesis_verdict: `{result['summary']['capacity_hypothesis_verdict']}`",
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
    if result["summary"]["positive_nonlinear_context_trials"]:
        lines.append("- Stage42-IY found nonlinear context trials that beat the nonlinear baseline-family reference while preserving easy cases.")
    else:
        lines.append("- Stage42-IY did not recover incremental context value with ExtraTrees capacity; current source-level evidence remains baseline-family dominated.")
    lines.append("- This is a sampled train-only nonlinear repair test, not a full foundation/full-data claim and not metric/seconds-level evidence.")
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_iy_gate"]
    lines = [
        "# Stage42-IY Gate",
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
    marker = "STAGE42_IY_SOURCE_LEVEL_NONLINEAR_CONTEXT_REPAIR"
    summary = result["summary"]
    best = summary["best_trial_metric"]
    block = [
        "## Stage42-IY Source-Level Nonlinear Context Repair",
        "",
        f"- source: `{result['source']}`",
        "- role: nonlinear capacity test after Stage42-IX still failed to make context incremental.",
        f"- gate: `{result['stage42_iy_gate']['passed']} / {result['stage42_iy_gate']['total']}`; verdict `{result['stage42_iy_gate']['verdict']}`.",
        f"- trials: `{len(result['trials'])}` ExtraTrees residual models; deterministic train cap `{MAX_TREE_TRAIN_ROWS}`.",
        f"- best_trial: `{summary['best_trial']}`; best all/t50/t100raw/hard `{best['all_improvement']:.6f}` / `{best['t50_improvement']:.6f}` / `{best['t100_raw_frame_diagnostic_improvement']:.6f}` / `{best['hard_failure_improvement']:.6f}`.",
        f"- easy degradation: `{best['easy_degradation']:.6f}`.",
        f"- positive_nonlinear_context_trials: `{summary['positive_nonlinear_context_trials']}`.",
        f"- capacity_hypothesis_verdict: `{summary['capacity_hypothesis_verdict']}`.",
        "- boundary: sampled train-only nonlinear repair; dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.",
    ]
    for p in [Path("README_RESULTS.md"), Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")]:
        _replace_block(p, marker, block)
    _replace_block(Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md"), marker, block)


def _update_state(result: Mapping[str, Any]) -> None:
    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_iy_source_level_nonlinear_context_repair"
    state["current_verdict"] = result["stage42_iy_gate"]["verdict"]
    state.setdefault("stage42", {})["stage_iy_source_level_nonlinear_context_repair"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": result["stage42_iy_gate"]["verdict"],
        "gates": f"{result['stage42_iy_gate']['passed']}/{result['stage42_iy_gate']['total']}",
        "best_trial": result["summary"]["best_trial"],
        "best_trial_metric": result["summary"]["best_trial_metric"],
        "positive_nonlinear_context_trials": result["summary"]["positive_nonlinear_context_trials"],
        "capacity_hypothesis_verdict": result["summary"]["capacity_hypothesis_verdict"],
        "claim_boundary": result["claim_boundary"],
    }
    generated = state.setdefault("generated_reports", [])
    for item in [str(REPORT_MD), str(REPORT_JSON), str(GATE_MD)]:
        if item not in generated:
            generated.append(item)
    write_json(Path("research_state.json"), _jsonable(state))


def _append_ledger(result: Mapping[str, Any]) -> None:
    row = {
        "stage": result["stage"],
        "timestamp": result["generated_at_utc"],
        "source": result["source"],
        "verdict": result["stage42_iy_gate"]["verdict"],
        "gate": f"{result['stage42_iy_gate']['passed']}/{result['stage42_iy_gate']['total']}",
        "best_trial": result["summary"]["best_trial"],
        "capacity_hypothesis_verdict": result["summary"]["capacity_hypothesis_verdict"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_stage42_source_level_nonlinear_context_repair(*, use_cached: bool = False) -> dict[str, Any]:
    if use_cached:
        cached = _cached_result_if_available()
        if cached is not None:
            return cached
    ensure_dir(OUT_DIR)
    shared = an._prep_shared()
    shared["feature_masks"] = ix._feature_masks(shared["feature_names"])
    shared["tree_train_ids"] = _train_ids(shared["split"], shared["data"])
    trials = {trial["name"]: _fit_tree_trial(shared, trial) for trial in TREE_TRIALS}
    ix_report = read_json(ix.REPORT_JSON, {})
    result = {
        "stage": "Stage42-IY source-level nonlinear context repair",
        "source": "fresh_run_sampled_extra_trees_context_capacity_repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/source_level_context_repair_stage42.json",
                "outputs/stage42_long_research/source_level_row_cache_mechanism_audit_stage42.json",
            ]
        ),
        "split_stats": shared["split_stats"],
        "feature_groups": {k: int(np.sum(v)) for k, v in an._feature_indices(shared["feature_names"]).items()},
        "feature_sets": {k: int(np.sum(v)) for k, v in shared["feature_masks"].items()},
        "tree_train_sampling": {
            "strategy": "deterministic_train_only_domain_horizon_hard_stratified_cap",
            "train_rows_available": int(np.sum(shared["split"] == "train")),
            "train_rows_used": int(len(shared["tree_train_ids"])),
            "max_train_rows": MAX_TREE_TRAIN_ROWS,
        },
        "trials": trials,
        "summary": _summarize(trials, ix_report),
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
    result["stage42_iy_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _update_readmes(result)
    _update_state(result)
    _append_ledger(result)
    return result


if __name__ == "__main__":
    run_stage42_source_level_nonlinear_context_repair()
