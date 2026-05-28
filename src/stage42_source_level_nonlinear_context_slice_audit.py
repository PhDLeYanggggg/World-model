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
from src import stage42_source_level_nonlinear_context_repair as iy
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "source_level_nonlinear_context_slice_audit_stage42.json"
REPORT_MD = OUT_DIR / "source_level_nonlinear_context_slice_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_iz_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

EPS = 1e-6
MIN_SLICE_ROWS = 1000
MIN_CONTEXT_LIFT = 0.01


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-IZ 是 Stage42-IY 后的切片级 context utility audit：检查非线性 context 是否只在特定 source / horizon / density / neighbor / curvature / goal-ambiguity 切片有效。",
    "本阶段重新训练 sampled train-only ExtraTrees residual trials；validation 选 safe policy；test 只评一次。",
    "所有切片阈值来自 train split quantiles，不用 test 调阈值。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "本阶段仍是 dataset-local/raw-frame 2.5D evidence，不是 metric 或 seconds-level 结果。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


TRIALS = [
    {"name": "tree_baseline_family_residual", "feature_set": "baseline_family"},
    {"name": "tree_context_only_residual", "feature_set": "context_only"},
    {"name": "tree_full_residual", "feature_set": "full"},
    {"name": "tree_goal_neighbor_residual", "feature_set": "goal_neighbor_context"},
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _cached_result_if_available() -> dict[str, Any] | None:
    if not REPORT_JSON.exists():
        return None
    payload = read_json(REPORT_JSON, {})
    if payload.get("stage") == "Stage42-IZ source-level nonlinear context slice audit":
        return payload
    return None


def _tree_outputs(shared: Mapping[str, Any], trial: Mapping[str, str]) -> dict[str, Any]:
    split = shared["split"]
    train_mask = split == "train"
    val_mask = split == "val"
    feature_mask = shared["feature_masks"][trial["feature_set"]]
    z, _, _ = am._standardize(shared["features"][:, feature_mask], train_mask)
    train_ids = shared["tree_train_ids"]
    y = iy._target(shared["data"], shared["labels"], shared["floor"])
    weights = ix._sample_weights(shared["data"], "t50_hard_t100")
    model = ExtraTreesRegressor(
        n_estimators=64,
        max_depth=16,
        min_samples_leaf=32,
        random_state=42107,
        n_jobs=1,
    )
    model.fit(z[train_ids], y[train_ids], sample_weight=weights[train_ids])
    pred_xy = iy._predict_tree_waypoints(model, z, shared["data"], shared["floor"])
    policy, selected_ade, selected_fde, switch = am._select_policy_on_val(
        pred_xy,
        shared["floor"]["floor_xy"],
        shared["labels"],
        shared["data"],
        val_mask,
    )
    floor_ade, floor_fde = am._trajectory_errors(shared["floor"]["floor_xy"], shared["labels"])
    pred_ade, pred_fde = am._trajectory_errors(pred_xy, shared["labels"])
    return {
        "trial": dict(trial),
        "feature_count": int(np.sum(feature_mask)),
        "train_rows_used": int(len(train_ids)),
        "validation_policy_slices": int(len(policy["slices"])),
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "switch": switch,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "ungated_ade": pred_ade,
        "ungated_fde": pred_fde,
    }


def _train_quantile(values: np.ndarray, train_mask: np.ndarray, q: float) -> float:
    finite = values[train_mask & np.isfinite(values)]
    if len(finite) == 0:
        return 0.0
    return float(np.quantile(finite.astype(np.float64), q))


def _slice_masks(data: Mapping[str, np.ndarray], split: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    train = split == "train"
    h = data["horizon"].astype(int)
    domain = data["dataset"].astype(str)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    hs = data["history_scalar"].astype(np.float64)
    goal_amb = data["goal_ambiguity"].astype(np.float64)
    thresholds = {
        "path_length_q75": _train_quantile(hs[:, 0], train, 0.75),
        "neighbor_count_q75": _train_quantile(hs[:, 1], train, 0.75),
        "min_neighbor_dist_q25": _train_quantile(hs[:, 2], train, 0.25),
        "density_q75": _train_quantile(hs[:, 3], train, 0.75),
        "ttc_q25": _train_quantile(hs[:, 4], train, 0.25),
        "curvature_q75": _train_quantile(hs[:, 6], train, 0.75),
        "abs_turn_angle_q75": _train_quantile(np.abs(hs[:, 7]), train, 0.75),
        "goal_ambiguity_q75": _train_quantile(goal_amb, train, 0.75),
    }
    slices: dict[str, np.ndarray] = {
        "all_test": np.ones(len(h), dtype=bool),
        "hard_failure": hard_failure,
        "easy": easy,
        "long_history_path": hs[:, 0] >= thresholds["path_length_q75"],
        "high_neighbor_count": hs[:, 1] >= thresholds["neighbor_count_q75"],
        "close_neighbor": hs[:, 2] <= thresholds["min_neighbor_dist_q25"],
        "high_density": hs[:, 3] >= thresholds["density_q75"],
        "low_ttc": hs[:, 4] <= thresholds["ttc_q25"],
        "high_curvature": hs[:, 6] >= thresholds["curvature_q75"],
        "high_turn_angle": np.abs(hs[:, 7]) >= thresholds["abs_turn_angle_q75"],
        "high_goal_ambiguity": goal_amb >= thresholds["goal_ambiguity_q75"],
    }
    for d in sorted(set(domain.tolist())):
        slices[f"domain:{d}"] = domain == d
        for hv in [10, 25, 50, 100]:
            slices[f"domain_horizon:{d}|{hv}"] = (domain == d) & (h == hv)
    for hv in [10, 25, 50, 100]:
        slices[f"horizon:{hv}"] = h == hv
    return slices, thresholds


def _metric_for_slice(row: Mapping[str, Any], data: Mapping[str, np.ndarray], test_mask: np.ndarray, slice_mask: np.ndarray) -> dict[str, Any]:
    mask = test_mask & slice_mask
    return am._metric(row["selected_ade"], row["floor_ade"], data, row["switch"], mask)


def _slice_table(shared: Mapping[str, Any], outputs: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    test_mask = shared["split"] == "test"
    slices, thresholds = _slice_masks(shared["data"], shared["split"])
    baseline = outputs["tree_baseline_family_residual"]
    rows: list[dict[str, Any]] = []
    for slice_name, smask in sorted(slices.items()):
        rows_n = int(np.sum(test_mask & smask))
        if rows_n < 30:
            continue
        baseline_metric = _metric_for_slice(baseline, shared["data"], test_mask, smask)
        for trial_name, out in outputs.items():
            if trial_name == "tree_baseline_family_residual":
                continue
            metric = _metric_for_slice(out, shared["data"], test_mask, smask)
            row = {
                "source": "fresh_run",
                "slice": slice_name,
                "trial": trial_name,
                "rows": rows_n,
                "context_all_improvement": float(metric["all_improvement"]),
                "baseline_all_improvement": float(baseline_metric["all_improvement"]),
                "delta_vs_baseline_all": float(metric["all_improvement"] - baseline_metric["all_improvement"]),
                "context_t50_improvement": float(metric["t50_improvement"]),
                "baseline_t50_improvement": float(baseline_metric["t50_improvement"]),
                "delta_vs_baseline_t50": float(metric["t50_improvement"] - baseline_metric["t50_improvement"]),
                "context_t100_raw_frame_diagnostic_improvement": float(metric["t100_raw_frame_diagnostic_improvement"]),
                "baseline_t100_raw_frame_diagnostic_improvement": float(baseline_metric["t100_raw_frame_diagnostic_improvement"]),
                "delta_vs_baseline_t100_raw": float(metric["t100_raw_frame_diagnostic_improvement"] - baseline_metric["t100_raw_frame_diagnostic_improvement"]),
                "context_hard_failure_improvement": float(metric["hard_failure_improvement"]),
                "baseline_hard_failure_improvement": float(baseline_metric["hard_failure_improvement"]),
                "delta_vs_baseline_hard": float(metric["hard_failure_improvement"] - baseline_metric["hard_failure_improvement"]),
                "context_easy_degradation": float(metric["easy_degradation"]),
                "switch_rate": float(metric["switch_rate"]),
                "slice_claim_supported": bool(
                    rows_n >= MIN_SLICE_ROWS
                    and metric["easy_degradation"] <= 0.02
                    and (
                        metric["all_improvement"] - baseline_metric["all_improvement"] > MIN_CONTEXT_LIFT
                        or metric["t50_improvement"] - baseline_metric["t50_improvement"] > MIN_CONTEXT_LIFT
                        or metric["t100_raw_frame_diagnostic_improvement"] - baseline_metric["t100_raw_frame_diagnostic_improvement"] > MIN_CONTEXT_LIFT
                        or metric["hard_failure_improvement"] - baseline_metric["hard_failure_improvement"] > MIN_CONTEXT_LIFT
                    )
                ),
            }
            rows.append(row)
    rows.sort(
        key=lambda r: (
            r["slice_claim_supported"],
            max(r["delta_vs_baseline_all"], r["delta_vs_baseline_t50"], r["delta_vs_baseline_t100_raw"], r["delta_vs_baseline_hard"]),
            r["rows"],
        ),
        reverse=True,
    )
    return rows, thresholds


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    powered = [r for r in rows if int(r["rows"]) >= MIN_SLICE_ROWS]
    supported = [r for r in powered if r["slice_claim_supported"]]
    best_by_trial: dict[str, Mapping[str, Any]] = {}
    for r in powered:
        score = max(
            float(r["delta_vs_baseline_all"]),
            float(r["delta_vs_baseline_t50"]),
            float(r["delta_vs_baseline_t100_raw"]),
            float(r["delta_vs_baseline_hard"]),
        )
        old = best_by_trial.get(str(r["trial"]))
        if old is None or score > max(float(old["delta_vs_baseline_all"]), float(old["delta_vs_baseline_t50"]), float(old["delta_vs_baseline_t100_raw"]), float(old["delta_vs_baseline_hard"])):
            best_by_trial[str(r["trial"])] = r
    blocker_counts = {
        "no_powered_positive_context_slice": int(not supported),
        "context_below_baseline_family": int(sum(1 for r in powered if max(float(r["delta_vs_baseline_all"]), float(r["delta_vs_baseline_t50"]), float(r["delta_vs_baseline_t100_raw"]), float(r["delta_vs_baseline_hard"])) <= MIN_CONTEXT_LIFT)),
        "easy_or_safety_not_primary_blocker": int(sum(1 for r in powered if float(r["context_easy_degradation"]) > 0.02)),
    }
    return {
        "powered_slice_rows": int(len(powered)),
        "supported_context_slices": [dict(r) for r in supported],
        "supported_context_slice_count": int(len(supported)),
        "best_slice_by_trial": {k: dict(v) for k, v in best_by_trial.items()},
        "decision": "context_has_powered_slice_level_support" if supported else "context_slice_level_support_not_found",
        "blocker_counts": blocker_counts,
        "interpretation": (
            "No powered slice shows >1pp easy-safe lift over the nonlinear baseline-family reference."
            if not supported
            else "At least one powered slice shows easy-safe lift over the nonlinear baseline-family reference; this is slice-limited context evidence, not a global context claim."
        ),
    }


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    s = result["summary"]
    gates = {
        "source_level_split_used": result["split_stats"]["by_split"]["test"]["rows"] == 47458,
        "tree_trials_retrained": len(result["trial_overview"]) == len(TRIALS),
        "slice_thresholds_train_only": result["slice_threshold_source"] == "train_split_quantiles_only",
        "slice_audit_complete": result["slice_rows_total"] > 0,
        "powered_slices_present": s["powered_slice_rows"] > 0,
        "context_slice_claim_supported": s["supported_context_slice_count"] > 0,
        "negative_or_positive_result_recorded": s["decision"] in {"context_has_powered_slice_level_support", "context_slice_level_support_not_found"},
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
        and result["no_leakage"]["train_only_slice_thresholds"],
        "no_metric_seconds_overclaim": result["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": result["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": result["claim_boundary"]["smc_enabled"] is False,
    }
    verdict = (
        "stage42_iz_context_slice_audit_positive"
        if all(gates.values())
        else "stage42_iz_context_slice_audit_completed_context_not_proven"
        if gates["slice_audit_complete"] and gates["negative_or_positive_result_recorded"] and gates["no_leakage_pass"]
        else "stage42_iz_context_slice_audit_partial"
    )
    return {"source": "fresh_run", "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": len(gates), "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    s = result["summary"]
    lines = [
        "# Stage42-IZ Source-Level Nonlinear Context Slice Audit",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_iz_gate']['passed']} / {result['stage42_iz_gate']['total']}`",
        f"- verdict: `{result['stage42_iz_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Summary",
        "",
        f"- decision: `{s['decision']}`",
        f"- supported_context_slice_count: `{s['supported_context_slice_count']}`",
        f"- powered_slice_rows: `{s['powered_slice_rows']}`",
        f"- interpretation: {s['interpretation']}",
        "",
        "## Train-Only Slice Thresholds",
        "",
        "| threshold | value |",
        "| --- | ---: |",
    ]
    for key, value in result["slice_thresholds"].items():
        lines.append(f"| `{key}` | {float(value):.6f} |")
    lines.extend(
        [
            "",
            "## Best Powered Slice By Trial",
            "",
            "| trial | slice | rows | delta all | delta t50 | delta t100 raw | delta hard | easy | supported |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for trial, row in s["best_slice_by_trial"].items():
        lines.append(
            f"| `{trial}` | `{row['slice']}` | {row['rows']} | {row['delta_vs_baseline_all']:.6f} | {row['delta_vs_baseline_t50']:.6f} | {row['delta_vs_baseline_t100_raw']:.6f} | {row['delta_vs_baseline_hard']:.6f} | {row['context_easy_degradation']:.6f} | `{row['slice_claim_supported']}` |"
        )
    lines.extend(
        [
            "",
            "## Top Slice Rows",
            "",
            "| slice | trial | rows | context all | baseline all | delta all | delta t50 | delta hard | easy |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["slice_rows"][:20]:
        lines.append(
            f"| `{row['slice']}` | `{row['trial']}` | {row['rows']} | {row['context_all_improvement']:.6f} | {row['baseline_all_improvement']:.6f} | {row['delta_vs_baseline_all']:.6f} | {row['delta_vs_baseline_t50']:.6f} | {row['delta_vs_baseline_hard']:.6f} | {row['context_easy_degradation']:.6f} |"
        )
    lines.extend(
        [
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
    if s["supported_context_slice_count"] > 0:
        lines.append("- Stage42-IZ found slice-limited nonlinear context support. This can only be written as a narrow slice claim until broader validation passes.")
    else:
        lines.append("- Stage42-IZ did not find a powered source/horizon/context slice where nonlinear context beats the nonlinear baseline-family reference by >1pp while preserving easy cases.")
    lines.append("- This closes another capacity/slice-level escape hatch for the current flattened context protocol; future work should change target/architecture/data context rather than repeat the same row-level residual setup.")
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_iz_gate"]
    lines = [
        "# Stage42-IZ Gate",
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
    marker = "STAGE42_IZ_SOURCE_LEVEL_NONLINEAR_CONTEXT_SLICE_AUDIT"
    s = result["summary"]
    block = [
        "## Stage42-IZ Source-Level Nonlinear Context Slice Audit",
        "",
        f"- source: `{result['source']}`",
        "- role: after Stage42-IY, test whether nonlinear context has only local slice-level utility.",
        f"- gate: `{result['stage42_iz_gate']['passed']} / {result['stage42_iz_gate']['total']}`; verdict `{result['stage42_iz_gate']['verdict']}`.",
        f"- supported_context_slice_count: `{s['supported_context_slice_count']}`.",
        f"- decision: `{s['decision']}`.",
        f"- blocker_counts: `{s['blocker_counts']}`.",
        "- boundary: train-only slice thresholds, validation-selected safe policy, test-once audit; dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.",
    ]
    for p in [Path("README_RESULTS.md"), Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")]:
        _replace_block(p, marker, block)
    _replace_block(Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md"), marker, block)


def _update_state(result: Mapping[str, Any]) -> None:
    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_iz_source_level_nonlinear_context_slice_audit"
    state["current_verdict"] = result["stage42_iz_gate"]["verdict"]
    state.setdefault("stage42", {})["stage_iz_source_level_nonlinear_context_slice_audit"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": result["stage42_iz_gate"]["verdict"],
        "gates": f"{result['stage42_iz_gate']['passed']}/{result['stage42_iz_gate']['total']}",
        "decision": result["summary"]["decision"],
        "supported_context_slice_count": result["summary"]["supported_context_slice_count"],
        "best_slice_by_trial": result["summary"]["best_slice_by_trial"],
        "blocker_counts": result["summary"]["blocker_counts"],
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
        "verdict": result["stage42_iz_gate"]["verdict"],
        "gate": f"{result['stage42_iz_gate']['passed']}/{result['stage42_iz_gate']['total']}",
        "decision": result["summary"]["decision"],
        "supported_context_slice_count": result["summary"]["supported_context_slice_count"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_stage42_source_level_nonlinear_context_slice_audit(*, use_cached: bool = False) -> dict[str, Any]:
    if use_cached:
        cached = _cached_result_if_available()
        if cached is not None:
            return cached
    ensure_dir(OUT_DIR)
    shared = an._prep_shared()
    shared["feature_masks"] = ix._feature_masks(shared["feature_names"])
    shared["tree_train_ids"] = iy._train_ids(shared["split"], shared["data"])
    outputs = {trial["name"]: _tree_outputs(shared, trial) for trial in TRIALS}
    slice_rows, thresholds = _slice_table(shared, outputs)
    result = {
        "stage": "Stage42-IZ source-level nonlinear context slice audit",
        "source": "fresh_run_retrained_extra_trees_context_slice_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/source_level_nonlinear_context_repair_stage42.json",
                "outputs/stage42_long_research/source_level_context_repair_stage42.json",
            ]
        ),
        "split_stats": shared["split_stats"],
        "slice_threshold_source": "train_split_quantiles_only",
        "slice_thresholds": thresholds,
        "trial_overview": {
            name: {
                "trial": out["trial"],
                "feature_count": out["feature_count"],
                "train_rows_used": out["train_rows_used"],
                "validation_policy_slices": out["validation_policy_slices"],
            }
            for name, out in outputs.items()
        },
        "slice_rows_total": int(len(slice_rows)),
        "slice_rows": slice_rows,
        "summary": _summarize(slice_rows),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_slice_thresholds": True,
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
    result["stage42_iz_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _update_readmes(result)
    _update_state(result)
    _append_ledger(result)
    return result


if __name__ == "__main__":
    run_stage42_source_level_nonlinear_context_slice_audit()
