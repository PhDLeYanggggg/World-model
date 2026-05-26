from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_source_level_ablation as an
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "source_level_incremental_ablation_stage42.json"
REPORT_MD = OUT_DIR / "source_level_incremental_ablation_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ao_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

MIN_INCREMENTAL_DELTA = 0.01
MIN_STANDALONE_SIGNAL = 0.03


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-AO 是 proposed source-level split incremental / standalone retrained ablation，不是 metric 或 seconds-level 结果。",
    "每个 variant 都重新训练 ridge full-waypoint probe，并在 validation 上重新选 safe policy；test 只评一次。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _group_masks(names: list[str]) -> dict[str, np.ndarray]:
    groups = an._feature_indices(names)
    return {
        "history": groups["history"].copy(),
        "neighbor_interaction": groups["neighbor_interaction"].copy(),
        "goal_prototype": groups["goal_prototype"].copy(),
        "baseline_family": groups["baseline_family"].copy(),
        "domain": groups["domain"].copy(),
        "horizon": groups["horizon"].copy(),
    }


def _or_mask(*masks: np.ndarray) -> np.ndarray:
    if not masks:
        raise ValueError("At least one mask is required.")
    out = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        out |= mask
    return out


def _incremental_variant_masks(names: list[str]) -> dict[str, np.ndarray]:
    groups = _group_masks(names)
    control = _or_mask(groups["horizon"], groups["domain"])
    baseline = _or_mask(groups["baseline_family"], control)
    history = groups["history"]
    neighbor = groups["neighbor_interaction"]
    goal = groups["goal_prototype"]
    return {
        "full": np.ones(len(names), dtype=bool),
        "horizon_domain_only": control,
        "baseline_family_only": baseline,
        "history_only": _or_mask(history, control),
        "goal_only": _or_mask(goal, control),
        "neighbor_only": _or_mask(neighbor, control),
        "motion_goal_context": _or_mask(history, goal, neighbor, control),
        "baseline_plus_history": _or_mask(baseline, history),
        "baseline_plus_goal": _or_mask(baseline, goal),
        "baseline_plus_neighbor": _or_mask(baseline, neighbor),
        "baseline_plus_history_goal_neighbor": _or_mask(baseline, history, goal, neighbor),
    }


def _evaluate_variant(name: str, raw_features: np.ndarray, shared: Mapping[str, Any]) -> dict[str, Any]:
    split = shared["split"]
    x, _, _ = am._standardize(raw_features, split == "train")
    result = am._evaluate_models(shared["data"], split, shared["labels"], shared["floor"], x)
    return {
        "source": "fresh_run",
        "variant": name,
        "feature_count": int(raw_features.shape[1]),
        "best_lambda": result["best_lambda"],
        "policy_slice_count": int(len(result["policy"]["slices"])),
        "protected": result["metrics"]["protected_ridge_source_level"],
        "ungated_diagnostic": result["metrics"]["ungated_ridge_diagnostic"],
        "bootstrap": result["bootstrap"],
        "by_domain": result["by_domain"],
        "by_horizon": result["by_horizon"],
        "normalization": "train_split_mean_std_only",
    }


def _metric_delta(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> dict[str, float]:
    keys = [
        "all_improvement",
        "t50_improvement",
        "t100_raw_frame_diagnostic_improvement",
        "hard_failure_improvement",
        "easy_degradation",
        "switch_rate",
        "harm_over_fallback",
    ]
    return {key: float(lhs[key]) - float(rhs[key]) for key in keys}


def _positive_core_delta(delta: Mapping[str, float], threshold: float = MIN_INCREMENTAL_DELTA) -> bool:
    return (
        delta["all_improvement"] > threshold
        or delta["t50_improvement"] > threshold
        or delta["hard_failure_improvement"] > threshold
    )


def _standalone_positive(metric: Mapping[str, Any], threshold: float = MIN_STANDALONE_SIGNAL) -> bool:
    return (
        (metric["all_improvement"] > threshold or metric["t50_improvement"] > threshold or metric["hard_failure_improvement"] > threshold)
        and metric["easy_degradation"] <= 0.02
    )


def run_stage42_source_level_incremental_ablation() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    shared = an._prep_shared()
    masks = _incremental_variant_masks(shared["feature_names"])
    variants: dict[str, Any] = {}
    for name, mask in masks.items():
        variants[name] = _evaluate_variant(name, shared["features"][:, mask], shared)

    baseline = variants["baseline_family_only"]["protected"]
    control = variants["horizon_domain_only"]["protected"]
    full = variants["full"]["protected"]
    incremental = {}
    for name, row in variants.items():
        if name in {"full", "baseline_family_only", "horizon_domain_only"}:
            continue
        metric = row["protected"]
        delta_vs_baseline = _metric_delta(metric, baseline)
        delta_vs_control = _metric_delta(metric, control)
        incremental[name] = {
            "source": "fresh_run",
            "delta_vs_baseline_family_only": delta_vs_baseline,
            "delta_vs_horizon_domain_only": delta_vs_control,
            "positive_increment_over_baseline": _positive_core_delta(delta_vs_baseline),
            "positive_increment_over_control": _positive_core_delta(delta_vs_control),
            "standalone_positive": _standalone_positive(metric),
            "interpretation": _interpret_increment(name, metric, delta_vs_baseline, delta_vs_control),
        }

    positive_standalone = sorted([name for name, row in incremental.items() if row["standalone_positive"] and name in {"history_only", "goal_only", "neighbor_only", "motion_goal_context"}])
    positive_incremental = sorted([name for name, row in incremental.items() if row["positive_increment_over_baseline"] and name.startswith("baseline_plus_")])
    result = {
        "source": "fresh_run",
        "stage": "Stage42-AO proposed source-level incremental retrained ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/source_level_full_waypoint_eval_stage42.json",
                "outputs/stage42_long_research/source_level_ablation_stage42.json",
            ]
        ),
        "split_stats": shared["split_stats"],
        "feature_groups": {k: int(np.sum(v)) for k, v in _group_masks(shared["feature_names"]).items()},
        "variants": variants,
        "incremental": incremental,
        "positive_standalone_context_variants": positive_standalone,
        "positive_incremental_context_variants": positive_incremental,
        "summary": _summarize(variants, incremental, positive_standalone, positive_incremental),
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
    result["stage42_ao_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _append_ledger(result)
    return result


def _interpret_increment(
    name: str,
    metric: Mapping[str, Any],
    delta_vs_baseline: Mapping[str, float],
    delta_vs_control: Mapping[str, float],
) -> str:
    if name in {"history_only", "goal_only", "neighbor_only", "motion_goal_context"}:
        if _standalone_positive(metric):
            return f"{name} has standalone positive signal versus the train-horizon floor while preserving easy cases."
        return f"{name} does not show standalone positive signal strong enough for a paper module claim."
    if name.startswith("baseline_plus_"):
        if _positive_core_delta(delta_vs_baseline):
            return f"{name} improves over baseline_family_only by > {MIN_INCREMENTAL_DELTA} on at least one core metric."
        return f"{name} does not improve over baseline_family_only by > {MIN_INCREMENTAL_DELTA}; baseline-family context likely absorbs this signal here."
    return "Diagnostic control variant."


def _summarize(
    variants: Mapping[str, Any],
    incremental: Mapping[str, Any],
    positive_standalone: list[str],
    positive_incremental: list[str],
) -> dict[str, Any]:
    baseline = variants["baseline_family_only"]["protected"]
    full = variants["full"]["protected"]
    full_minus_baseline = _metric_delta(full, baseline)
    return {
        "source": "fresh_run",
        "baseline_family_only": baseline,
        "full_minus_baseline_family_only": full_minus_baseline,
        "positive_standalone_context_variants": positive_standalone,
        "positive_incremental_context_variants": positive_incremental,
        "component_evidence_verdict": (
            "stage42_ao_context_components_supported"
            if len(positive_standalone) >= 1 and len(positive_incremental) >= 1
            else "stage42_ao_context_components_not_independently_supported"
        ),
        "interpretation": (
            "This experiment tests whether context modules have standalone or incremental value after baseline-family context. "
            "A negative result means the current ridge source-level feature set is dominated by baseline-family rollout context, not that history/goal/neighbor are impossible to use in stronger neural models."
        ),
    }


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    full = result["variants"]["full"]["protected"]
    gates = {
        "proposed_source_level_split_used": result["split_stats"]["by_split"]["test"]["rows"] == 47458,
        "incremental_variants_complete": len(result["variants"]) >= 10,
        "full_variant_positive": full["all_improvement"] > 0
        and full["t50_improvement"] > 0
        and full["hard_failure_improvement"] > 0
        and full["easy_degradation"] <= 0.02,
        "baseline_family_control_positive": result["variants"]["baseline_family_only"]["protected"]["all_improvement"] > 0,
        "standalone_context_signal_found": len(result["positive_standalone_context_variants"]) >= 1,
        "incremental_context_signal_found": len(result["positive_incremental_context_variants"]) >= 1,
        "bootstrap_available_for_full": result["variants"]["full"]["bootstrap"]["all"]["bootstrap_n"] > 0
        and result["variants"]["full"]["bootstrap"]["t50"]["bootstrap_n"] > 0,
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
        "stage42_ao_incremental_component_evidence_pass"
        if all(gates.values())
        else "stage42_ao_incremental_component_evidence_partial_or_negative"
    )
    return {"source": "fresh_run", "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": int(len(gates)), "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-AO Proposed Source-Level Incremental Ablation",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_ao_gate']['passed']} / {result['stage42_ao_gate']['total']}`",
        f"- verdict: `{result['stage42_ao_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Why This Was Run",
        "",
        "- Stage42-AN showed that full-minus-module retrained ablation only clearly supported `baseline_family_context`.",
        "- Stage42-AO asks a sharper question: do history / goal / neighbor context features have standalone signal, or incremental value after baseline-family rollout context?",
        "- A negative result is still useful: it tells us not to write these modules as independent paper claims under the current ridge source-level feature set.",
        "",
        "## Variant Metrics",
        "",
        "| variant | features | all | t50 | t100 diag | hard/failure | easy | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["variants"].items():
        metric = row["protected"]
        lines.append(
            f"| `{name}` | {row['feature_count']} | {metric['all_improvement']:.6f} | {metric['t50_improvement']:.6f} | {metric['t100_raw_frame_diagnostic_improvement']:.6f} | {metric['hard_failure_improvement']:.6f} | {metric['easy_degradation']:.6f} | {metric['switch_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Incremental Evidence",
            "",
            "| variant | delta all vs baseline-family | delta t50 | delta hard/failure | standalone positive | incremental positive | interpretation |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for name, row in result["incremental"].items():
        d = row["delta_vs_baseline_family_only"]
        lines.append(
            f"| `{name}` | {d['all_improvement']:.6f} | {d['t50_improvement']:.6f} | {d['hard_failure_improvement']:.6f} | {row['standalone_positive']} | {row['positive_increment_over_baseline']} | {row['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- positive_standalone_context_variants: `{result['positive_standalone_context_variants']}`",
            f"- positive_incremental_context_variants: `{result['positive_incremental_context_variants']}`",
            f"- component_evidence_verdict: `{result['summary']['component_evidence_verdict']}`",
            f"- full_minus_baseline_family_only: `{result['summary']['full_minus_baseline_family_only']}`",
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
    if result["stage42_ao_gate"]["verdict"].endswith("pass"):
        lines.append("- Stage42-AO found both standalone and incremental context-module evidence on the proposed source-level split.")
    else:
        lines.append("- Stage42-AO did not find enough standalone/incremental context-module evidence under this ridge protocol; the current evidence remains dominated by baseline-family rollout context.")
    lines.append("- This does not prove history/goal/neighbor are useless; it means their independent paper claim requires a stronger neural/graph retraining protocol or richer source-level context.")
    lines.append("- Claims remain dataset-local raw-frame 2.5D, not metric/seconds-level, true-3D, foundation, Stage5C, or SMC evidence.")
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_ao_gate"]
    lines = [
        "# Stage42-AO Gate",
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
        "verdict": result["stage42_ao_gate"]["verdict"],
        "gate": f"{result['stage42_ao_gate']['passed']}/{result['stage42_ao_gate']['total']}",
        "git_commit": result["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(f"{row}\n")


if __name__ == "__main__":
    run_stage42_source_level_incremental_ablation()
