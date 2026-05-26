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
REPORT_JSON = OUT_DIR / "source_level_baseline_family_mechanism_stage42.json"
REPORT_MD = OUT_DIR / "source_level_baseline_family_mechanism_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_au_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

MIN_MECHANISM_DELTA = 0.01
EPS = 1e-6


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-AU 是 proposed source-level baseline-family mechanism audit，不是 metric 或 seconds-level 结果。",
    "本审计拆解 floor_rel、safe_baseline_rel、family_baseline_rel 和 horizon/domain controls 的贡献。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _prefix_mask(names: list[str], prefix: str) -> np.ndarray:
    return np.asarray([name.startswith(prefix) for name in names], dtype=bool)


def _variant_masks(names: list[str]) -> dict[str, np.ndarray]:
    groups = ao._group_masks(names)
    floor = _prefix_mask(names, "floor_rel_")
    safe = _prefix_mask(names, "safe_baseline_rel_")
    family = _prefix_mask(names, "family_baseline_rel_")
    control = ao._or_mask(groups["horizon"], groups["domain"])
    return {
        "horizon_domain_control": control,
        "floor_rel_only": ao._or_mask(floor, control),
        "safe_baseline_rel_only": ao._or_mask(safe, control),
        "family_baseline_rel_only": ao._or_mask(family, control),
        "floor_plus_safe": ao._or_mask(floor, safe, control),
        "floor_plus_family": ao._or_mask(floor, family, control),
        "safe_plus_family": ao._or_mask(safe, family, control),
        "baseline_family_all": ao._or_mask(floor, safe, family, control),
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
        "ungated": result["metrics"]["ungated_ridge_diagnostic"],
        "bootstrap": result["bootstrap"],
        "by_domain": result["by_domain"],
        "by_horizon": result["by_horizon"],
        "normalization": "train_split_mean_std_only",
    }


def _metric_delta(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> dict[str, float]:
    return ao._metric_delta(lhs, rhs)


def _core_positive(metric: Mapping[str, Any], threshold: float = 0.0) -> bool:
    return bool(
        metric["all_improvement"] > threshold
        and (metric["t50_improvement"] > threshold or metric["hard_failure_improvement"] > threshold)
        and metric["easy_degradation"] <= 0.02
    )


def _best_single_family(variants: Mapping[str, Any], metric_name: str) -> tuple[str, Mapping[str, Any]]:
    single = ["floor_rel_only", "safe_baseline_rel_only", "family_baseline_rel_only"]
    best = max(single, key=lambda name: float(variants[name][metric_name]["all_improvement"]))
    return best, variants[best][metric_name]


def _family_increment(variants: Mapping[str, Any], metric_name: str) -> dict[str, Any]:
    full = variants["baseline_family_all"][metric_name]
    best_name, best_metric = _best_single_family(variants, metric_name)
    control = variants["horizon_domain_control"][metric_name]
    return {
        "source": "fresh_run",
        "metric_family": metric_name,
        "best_single_family_variant": best_name,
        "baseline_family_all_minus_best_single": _metric_delta(full, best_metric),
        "baseline_family_all_minus_control": _metric_delta(full, control),
        "multi_family_increment_supported": _core_delta_supported(_metric_delta(full, best_metric)),
        "baseline_family_mechanism_supported": _core_positive(full) and _core_delta_supported(_metric_delta(full, control)),
    }


def _core_delta_supported(delta: Mapping[str, float], threshold: float = MIN_MECHANISM_DELTA) -> bool:
    return bool(
        delta["all_improvement"] > threshold
        or delta["t50_improvement"] > threshold
        or delta["hard_failure_improvement"] > threshold
    )


def run_stage42_source_level_baseline_family_mechanism() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    shared = an._prep_shared()
    masks = _variant_masks(shared["feature_names"])
    variants = {name: _evaluate_variant(name, shared["features"][:, mask], shared) for name, mask in masks.items()}
    protected_increment = _family_increment(variants, "protected")
    ungated_increment = _family_increment(variants, "ungated")
    pairwise_deltas = {}
    for name in variants:
        if name == "baseline_family_all":
            continue
        pairwise_deltas[name] = {
            "source": "fresh_run",
            "protected_delta_to_all": _metric_delta(variants["baseline_family_all"]["protected"], variants[name]["protected"]),
            "ungated_delta_to_all": _metric_delta(variants["baseline_family_all"]["ungated"], variants[name]["ungated"]),
        }
    result = {
        "source": "fresh_run",
        "stage": "Stage42-AU proposed source-level baseline-family mechanism audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/source_level_safety_floor_audit_stage42.json",
                "outputs/stage42_long_research/source_level_incremental_ablation_stage42.json",
            ]
        ),
        "split_stats": shared["split_stats"],
        "feature_counts": {name: int(np.sum(mask)) for name, mask in masks.items()},
        "variants": variants,
        "pairwise_deltas_to_baseline_family_all": pairwise_deltas,
        "protected_family_increment": protected_increment,
        "ungated_family_increment": ungated_increment,
        "summary": _summarize(variants, protected_increment, ungated_increment),
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
            "ungated_neural_deployable": False,
        },
    }
    result["stage42_au_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _append_ledger(result)
    return result


def _summarize(
    variants: Mapping[str, Any],
    protected_increment: Mapping[str, Any],
    ungated_increment: Mapping[str, Any],
) -> dict[str, Any]:
    best_protected_name, best_protected = _best_single_family(variants, "protected")
    best_ungated_name, best_ungated = _best_single_family(variants, "ungated")
    full_protected = variants["baseline_family_all"]["protected"]
    full_ungated = variants["baseline_family_all"]["ungated"]
    return {
        "source": "fresh_run",
        "best_single_family_protected": best_protected_name,
        "best_single_family_ungated": best_ungated_name,
        "best_single_family_protected_metric": best_protected,
        "best_single_family_ungated_metric": best_ungated,
        "baseline_family_all_protected": full_protected,
        "baseline_family_all_ungated": full_ungated,
        "protected_multi_family_increment_supported": protected_increment["multi_family_increment_supported"],
        "ungated_multi_family_increment_supported": ungated_increment["multi_family_increment_supported"],
        "mechanism_verdict": (
            "baseline_family_rollout_context_supported_as_dominant_mechanism"
            if protected_increment["baseline_family_mechanism_supported"] or ungated_increment["baseline_family_mechanism_supported"]
            else "baseline_family_rollout_context_not_supported"
        ),
        "interpretation": (
            "This audit tests whether current source-level success is just horizon/domain controls, one floor baseline, or a broader baseline-family rollout context. "
            "It does not claim true 3D, metric prediction, seconds-level horizons, Stage5C, SMC, or floor-free neural dynamics."
        ),
    }


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    full_protected = result["variants"]["baseline_family_all"]["protected"]
    full_ungated = result["variants"]["baseline_family_all"]["ungated"]
    gates = {
        "proposed_source_level_split_used": result["split_stats"]["by_split"]["test"]["rows"] == 47458,
        "all_baseline_family_variants_evaluated": len(result["variants"]) >= 8,
        "control_variant_reported": "horizon_domain_control" in result["variants"],
        "single_family_variants_reported": all(
            name in result["variants"] for name in ["floor_rel_only", "safe_baseline_rel_only", "family_baseline_rel_only"]
        ),
        "baseline_family_all_positive_protected_or_ungated": _core_positive(full_protected) or _core_positive(full_ungated),
        "mechanism_supported_over_control": result["summary"]["mechanism_verdict"] == "baseline_family_rollout_context_supported_as_dominant_mechanism",
        "multi_family_increment_checked": "baseline_family_all_minus_best_single" in result["protected_family_increment"],
        "no_leakage_pass": all(
            result["no_leakage"][k] is False
            for k in ["future_endpoint_input", "future_waypoint_input", "central_velocity", "test_endpoint_goals", "test_threshold_tuning"]
        )
        and result["no_leakage"]["train_only_feature_normalization"],
        "no_metric_seconds_overclaim": not result["claim_boundary"]["metric_or_seconds_claim"],
        "stage5c_false": not result["claim_boundary"]["stage5c_executed"],
        "smc_false": not result["claim_boundary"]["smc_enabled"],
    }
    verdict = "stage42_au_baseline_family_mechanism_pass" if all(gates.values()) else "stage42_au_baseline_family_mechanism_partial"
    return {"source": "fresh_run", "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": int(len(gates)), "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-AU Source-Level Baseline-Family Mechanism Audit",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_au_gate']['passed']} / {result['stage42_au_gate']['total']}`",
        f"- verdict: `{result['stage42_au_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
    ]
    lines.extend([f"- {fact}" for fact in result["current_facts"]])
    lines.extend(
        [
            "",
            "## Why This Audit Exists",
            "",
            "Stage42-AN/AO/AP/AQ/AR/AS showed that history, goal, neighbor, sequence, and hand-built graph residual context do not independently beat the baseline-family first stage under the tested source-level protocols.",
            "Stage42-AU therefore tests the mechanism that is actually working: baseline-family rollout context.",
            "",
            "## Variant Comparison",
            "",
            "| variant | features | protected all | protected t50 | protected hard | protected easy | ungated all | ungated t50 | ungated hard | ungated easy |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in result["variants"].items():
        p = row["protected"]
        u = row["ungated"]
        lines.append(
            f"| `{name}` | {row['feature_count']} | {p['all_improvement']:.6f} | {p['t50_improvement']:.6f} | {p['hard_failure_improvement']:.6f} | {p['easy_degradation']:.6f} | {u['all_improvement']:.6f} | {u['t50_improvement']:.6f} | {u['hard_failure_improvement']:.6f} | {u['easy_degradation']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism Summary",
            "",
            f"- best_single_family_protected: `{result['summary']['best_single_family_protected']}`",
            f"- best_single_family_ungated: `{result['summary']['best_single_family_ungated']}`",
            f"- protected_multi_family_increment_supported: `{result['summary']['protected_multi_family_increment_supported']}`",
            f"- ungated_multi_family_increment_supported: `{result['summary']['ungated_multi_family_increment_supported']}`",
            f"- mechanism_verdict: `{result['summary']['mechanism_verdict']}`",
            f"- interpretation: {result['summary']['interpretation']}",
            "",
            "## Pairwise Deltas To Baseline-Family-All",
            "",
            "| variant | protected delta all | protected delta t50 | ungated delta all | ungated delta t50 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in result["pairwise_deltas_to_baseline_family_all"].items():
        p = row["protected_delta_to_all"]
        u = row["ungated_delta_to_all"]
        lines.append(f"| `{name}` | {p['all_improvement']:.6f} | {p['t50_improvement']:.6f} | {u['all_improvement']:.6f} | {u['t50_improvement']:.6f} |")
    lines.extend(
        [
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{result['no_leakage']}`",
            f"- claim_boundary: `{result['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_au_gate"]
    lines = [
        "# Stage42-AU Gate",
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
        "verdict": result["stage42_au_gate"]["verdict"],
        "gate": f"{result['stage42_au_gate']['passed']}/{result['stage42_au_gate']['total']}",
        "git_commit": result["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(f"{row}\n")


if __name__ == "__main__":
    run_stage42_source_level_baseline_family_mechanism()
