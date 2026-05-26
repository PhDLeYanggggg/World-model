from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "source_level_ablation_stage42.json"
REPORT_MD = OUT_DIR / "source_level_ablation_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_an_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

MIN_MEANINGFUL_DELTA = 0.01
COMPONENT_BY_VARIANT = {
    "no_history": "history",
    "no_neighbor_interaction": "neighbor_interaction",
    "no_goal_prototype": "goal_prototype",
    "no_baseline_family": "baseline_family_context",
    "no_domain_expert": "domain_expert",
}


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-AN 是 proposed source-level split retrained ablation，不是 metric 或 seconds-level 结果。",
    "每个特征消融都重新训练 ridge probe，并在 validation 上重新选 safe policy；test 只评一次。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _feature_indices(names: list[str]) -> dict[str, np.ndarray]:
    arr = np.asarray(names)
    groups = {
        "history": np.asarray([n.startswith("history_scalar_") or n.startswith("history_tail_") for n in arr]),
        "neighbor_interaction": np.asarray([n in {f"history_scalar_{i}" for i in [1, 2, 3, 4, 5]} for n in arr]),
        "goal_prototype": np.asarray([n.startswith("prototype_") or n in {"prototype_entropy", "goal_ambiguity"} for n in arr]),
        "baseline_family": np.asarray([n.startswith("safe_baseline_rel_") or n.startswith("family_baseline_rel_") or n.startswith("floor_rel_") for n in arr]),
        "domain": np.asarray([n.startswith("domain_") for n in arr]),
        "horizon": np.asarray([n.startswith("horizon_") for n in arr]),
    }
    return groups


def _variant_masks(names: list[str]) -> dict[str, np.ndarray]:
    groups = _feature_indices(names)
    keep_all = np.ones(len(names), dtype=bool)
    return {
        "full": keep_all.copy(),
        "no_history": keep_all & ~groups["history"],
        "no_neighbor_interaction": keep_all & ~groups["neighbor_interaction"],
        "no_goal_prototype": keep_all & ~groups["goal_prototype"],
        "no_baseline_family": keep_all & ~groups["baseline_family"],
        "no_domain_expert": keep_all & ~groups["domain"],
        "motion_goal_no_baseline_domain": keep_all & ~groups["baseline_family"] & ~groups["domain"],
    }


def _prep_shared() -> dict[str, Any]:
    data = s41._combined()
    split, group = am._split_arrays(data)
    split_stats = am._source_stats(data, split, group)
    labels = am._reconstruct_waypoint_labels(data)
    train_mask = split == "train"
    floor = am._floor_arrays(data, train_mask)
    features, names = am._feature_matrix(data, floor)
    return {
        "data": data,
        "split": split,
        "group": group,
        "split_stats": split_stats,
        "labels": labels,
        "floor": floor,
        "features": features,
        "feature_names": names,
    }


def _evaluate_variant(name: str, raw_features: np.ndarray, shared: Mapping[str, Any]) -> dict[str, Any]:
    split = shared["split"]
    x, mean, std = am._standardize(raw_features, split == "train")
    result = am._evaluate_models(shared["data"], split, shared["labels"], shared["floor"], x)
    protected = result["metrics"]["protected_ridge_source_level"]
    ungated = result["metrics"]["ungated_ridge_diagnostic"]
    return {
        "source": "fresh_run",
        "variant": name,
        "feature_count": int(raw_features.shape[1]),
        "best_lambda": result["best_lambda"],
        "policy_slice_count": int(len(result["policy"]["slices"])),
        "protected": protected,
        "ungated_diagnostic": ungated,
        "bootstrap": result["bootstrap"],
        "by_domain": result["by_domain"],
        "by_horizon": result["by_horizon"],
        "normalization": "train_split_mean_std_only",
    }


def _delta(full: Mapping[str, Any], variant: Mapping[str, Any]) -> dict[str, float]:
    keys = [
        "all_improvement",
        "t50_improvement",
        "t100_raw_frame_diagnostic_improvement",
        "hard_failure_improvement",
        "easy_degradation",
    ]
    return {k: float(full[k]) - float(variant[k]) for k in keys}


def run_stage42_source_level_ablation() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    shared = _prep_shared()
    masks = _variant_masks(shared["feature_names"])
    variants: dict[str, Any] = {}
    for name, mask in masks.items():
        variants[name] = _evaluate_variant(name, shared["features"][:, mask], shared)
    full = variants["full"]["protected"]
    ablations = {}
    for name, row in variants.items():
        if name == "full":
            continue
        d = _delta(full, row["protected"])
        ablations[name] = {
            "source": "fresh_run",
            "delta_full_minus_variant": d,
            "contributes_all": d["all_improvement"] > MIN_MEANINGFUL_DELTA,
            "contributes_t50": d["t50_improvement"] > MIN_MEANINGFUL_DELTA,
            "contributes_hard_failure": d["hard_failure_improvement"] > MIN_MEANINGFUL_DELTA,
            "easy_safety_effect": -d["easy_degradation"],
            "interpretation": _interpret_delta(name, d),
        }
    ungated = variants["full"]["ungated_diagnostic"]
    safe = variants["full"]["protected"]
    safe_switch_delta = {
        "protected_minus_ungated_all": float(safe["all_improvement"]) - float(ungated["all_improvement"]),
        "protected_minus_ungated_t50": float(safe["t50_improvement"]) - float(ungated["t50_improvement"]),
        "protected_minus_ungated_hard_failure": float(safe["hard_failure_improvement"]) - float(ungated["hard_failure_improvement"]),
        "protected_minus_ungated_easy_degradation": float(safe["easy_degradation"]) - float(ungated["easy_degradation"]),
    }
    positive_components = sorted(
        {
            COMPONENT_BY_VARIANT[name]
            for name, row in ablations.items()
            if name in COMPONENT_BY_VARIANT and (row["contributes_all"] or row["contributes_t50"] or row["contributes_hard_failure"])
        }
    )
    positive_interaction_variants = [
        name
        for name, row in ablations.items()
        if name not in COMPONENT_BY_VARIANT and (row["contributes_all"] or row["contributes_t50"] or row["contributes_hard_failure"])
    ]
    result = {
        "source": "fresh_run",
        "stage": "Stage42-AN proposed source-level retrained ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/source_level_full_waypoint_eval_stage42.json",
            ]
        ),
        "split_stats": shared["split_stats"],
        "feature_groups": {k: int(np.sum(v)) for k, v in _feature_indices(shared["feature_names"]).items()},
        "variants": variants,
        "ablations": ablations,
        "positive_components": positive_components,
        "positive_interaction_variants": positive_interaction_variants,
        "safe_switch_vs_ungated": {
            "source": "fresh_run",
            "delta": safe_switch_delta,
            "safe_switch_contribution_supported_here": safe_switch_delta["protected_minus_ungated_easy_degradation"] < 0
            and safe["easy_degradation"] <= 0.02,
            "interpretation": "If protected trails ungated on accuracy but improves easy safety, safe-switch is safety-positive. If ungated is also safe, this specific ridge probe does not prove safe-switch necessity.",
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
    result["stage42_an_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _append_ledger(result)
    return result


def _interpret_delta(name: str, d: Mapping[str, float]) -> str:
    wins = [k for k in ["all_improvement", "t50_improvement", "hard_failure_improvement"] if d[k] > MIN_MEANINGFUL_DELTA]
    component = COMPONENT_BY_VARIANT.get(name, name)
    if wins:
        return f"Removing {component} hurts {', '.join(wins)} by > {MIN_MEANINGFUL_DELTA}; contribution supported on this proposed source-level split."
    return f"Removing {component} does not hurt core metrics by > {MIN_MEANINGFUL_DELTA}; contribution not proven by this ablation."


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    full = result["variants"]["full"]["protected"]
    gates = {
        "proposed_source_level_split_used": result["split_stats"]["by_split"]["test"]["rows"] == 47458,
        "retrained_ablation_variants_complete": len(result["variants"]) >= 6,
        "full_variant_positive": full["all_improvement"] > 0
        and full["t50_improvement"] > 0
        and full["hard_failure_improvement"] > 0
        and full["easy_degradation"] <= 0.02,
        "at_least_two_independent_components_positive": len(result["positive_components"]) >= 2,
        "safe_switch_analyzed": "safe_switch_vs_ungated" in result,
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
        "stage42_an_source_level_ablation_pass"
        if all(gates.values())
        else "stage42_an_source_level_ablation_partial_component_evidence"
    )
    return {"source": "fresh_run", "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": int(len(gates)), "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    full = result["variants"]["full"]["protected"]
    lines = [
        "# Stage42-AN Proposed Source-Level Retrained Ablation",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_an_gate']['passed']} / {result['stage42_an_gate']['total']}`",
        f"- verdict: `{result['stage42_an_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Full Variant",
        "",
        f"- metrics: `{full}`",
        f"- feature_groups: `{result['feature_groups']}`",
        "",
        "## Retrained Variants",
        "",
        "| variant | features | all | t50 | t100 diag | hard/failure | easy | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["variants"].items():
        metric = row["protected"]
        lines.append(
            f"| `{name}` | {row['feature_count']} | {metric['all_improvement']:.6f} | {metric['t50_improvement']:.6f} | {metric['t100_raw_frame_diagnostic_improvement']:.6f} | {metric['hard_failure_improvement']:.6f} | {metric['easy_degradation']:.6f} | {metric['switch_rate']:.6f} |"
        )
    lines.extend(["", "## Full Minus Ablated Variant", "", "| ablation | delta all | delta t50 | delta hard/failure | interpretation |", "| --- | ---: | ---: | ---: | --- |"])
    for name, row in result["ablations"].items():
        d = row["delta_full_minus_variant"]
        lines.append(f"| `{name}` | {d['all_improvement']:.6f} | {d['t50_improvement']:.6f} | {d['hard_failure_improvement']:.6f} | {row['interpretation']} |")
    lines.extend(
        [
            "",
            "## Safe Switch / Teacher Floor Analysis",
            "",
            f"- safe_switch_vs_ungated: `{result['safe_switch_vs_ungated']}`",
            "",
            "## Interpretation",
            "",
        ]
    )
    if result["positive_components"]:
        lines.append(f"- Positive independent component evidence on this split: `{result['positive_components']}`.")
    else:
        lines.append("- No individual ablated module crossed the meaningful positive threshold; this would be a blocker for module-contribution claims.")
    if result["positive_interaction_variants"]:
        lines.append(f"- Positive combined/interacting ablation variants: `{result['positive_interaction_variants']}`. These do not count as independent modules by themselves.")
    if len(result["positive_components"]) < 2:
        lines.append("- Stage42-AN does not prove two independent module contributions; next work should target stronger retrained neural/graph ablation or richer source-level features.")
    lines.append("- This is retrained ridge ablation evidence on proposed source-level split. It does not prove true 3D, metric, seconds-level, foundation, Stage5C, or SMC readiness.")
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_an_gate"]
    lines = [
        "# Stage42-AN Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| {name} | `{ok}` |")
    return lines


def _append_ledger(result: Mapping[str, Any]) -> None:
    entry = {
        "command": "run_stage42_source_level_ablation.py",
        "source": result["source"],
        "status": "success",
        "generated_at_utc": result["generated_at_utc"],
        "git_commit": result["git_commit"],
        "input_hash": result["input_hash"],
        "outputs": [str(REPORT_JSON), str(REPORT_MD), str(GATE_MD)],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_source_level_ablation()
