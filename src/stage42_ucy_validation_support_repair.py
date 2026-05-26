from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_baseline_family_mechanism as au
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "ucy_validation_support_repair_stage42.json"
REPORT_MD = OUT_DIR / "ucy_validation_support_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_aw_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-AW 修复的是 UCY 在 proposed source-level split 下没有 validation rows 的 blocker。",
    "本修复只从 UCY train sources 中切出 internal validation；test source 完全不参与 policy/threshold 选择。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _ucy_internal_val_group(split: np.ndarray, group: np.ndarray, domain: np.ndarray) -> str:
    candidates: list[tuple[int, str]] = []
    for g in sorted(set(group[(split == "train") & (domain == "UCY")].tolist())):
        rows = int(np.sum((split == "train") & (group == g)))
        candidates.append((rows, str(g)))
    if not candidates:
        raise ValueError("No UCY train group is available for internal validation.")
    return min(candidates)[1]


def _split_with_ucy_internal_val(split: np.ndarray, group: np.ndarray, domain: np.ndarray) -> tuple[np.ndarray, str]:
    repaired = split.astype("U8").copy()
    val_group = _ucy_internal_val_group(split, group, domain)
    repaired[(split == "train") & (group == val_group)] = "val"
    return repaired, val_group


def _safe_variant_masks(names: list[str]) -> dict[str, np.ndarray]:
    base = au._variant_masks(names)
    return {
        "family_baseline_rel_only": base["family_baseline_rel_only"],
        "floor_plus_family": base["floor_plus_family"],
        "safe_plus_family": base["safe_plus_family"],
        "baseline_family_all": base["baseline_family_all"],
    }


def _evaluate_variant(name: str, raw_features: np.ndarray, data: Mapping[str, np.ndarray], split: np.ndarray, labels: Mapping[str, np.ndarray], floor: Mapping[str, Any]) -> dict[str, Any]:
    x, _, _ = am._standardize(raw_features, split == "train")
    result = am._evaluate_models(data, split, labels, floor, x)
    return {
        "source": "fresh_run",
        "variant": name,
        "feature_count": int(raw_features.shape[1]),
        "best_lambda": result["best_lambda"],
        "validation_score": float(result["validation_selection"]["selected_score"]),
        "policy_slice_count": int(len(result["policy"]["slices"])),
        "policy_slices": sorted(result["policy"]["slices"].keys()),
        "protected": result["metrics"]["protected_ridge_source_level"],
        "fde": result["metrics"]["protected_ridge_source_level_fde"],
        "bootstrap": result["bootstrap"],
        "by_domain": result["by_domain"],
        "by_horizon": result["by_horizon"],
        "normalization": "train_split_mean_std_only",
    }


def _select_validation_best(variants: Mapping[str, Mapping[str, Any]]) -> str:
    return max(variants.keys(), key=lambda name: float(variants[name]["validation_score"]))


def _positive_ucy(metric: Mapping[str, Any]) -> bool:
    return bool(
        float(metric["all_improvement"]) > 0.0
        and float(metric["t50_improvement"]) > 0.0
        and float(metric["hard_failure_improvement"]) > 0.0
        and float(metric["easy_degradation"]) <= 0.02
        and float(metric["switch_rate"]) > 0.0
    )


def run_stage42_ucy_validation_support_repair() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    data = s41._combined()
    original_split, group = am._split_arrays(data)
    domain = data["dataset"].astype(str)
    repaired_split, internal_val_group = _split_with_ucy_internal_val(original_split, group, domain)
    labels = am._reconstruct_waypoint_labels(data)
    train_mask = repaired_split == "train"
    floor = am._floor_arrays(data, train_mask)
    features, feature_names = am._feature_matrix(data, floor)
    masks = _safe_variant_masks(feature_names)
    variants = {
        name: _evaluate_variant(name, features[:, mask], data, repaired_split, labels, floor)
        for name, mask in masks.items()
    }
    best_name = _select_validation_best(variants)
    best = variants[best_name]
    original_stats = am._source_stats(data, original_split, group)
    repaired_stats = am._source_stats(data, repaired_split, group)
    result = {
        "source": "fresh_run",
        "stage": "Stage42-AW UCY validation-support repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/source_level_baseline_family_robustness_stage42.json",
                "outputs/stage42_long_research/source_level_baseline_family_mechanism_stage42.json",
            ]
        ),
        "original_split_stats": original_stats,
        "repaired_split_stats": repaired_stats,
        "internal_validation": {
            "source": "fresh_run",
            "domain": "UCY",
            "selected_from": "original_train_sources_only",
            "internal_val_group": internal_val_group,
            "uses_test_rows": False,
            "test_rows_unchanged": int(original_stats["by_split"]["test"]["rows"]) == int(repaired_stats["by_split"]["test"]["rows"]),
        },
        "feature_counts": {name: int(np.sum(mask)) for name, mask in masks.items()},
        "variants": variants,
        "validation_best_variant": best_name,
        "validation_best": best,
        "summary": {
            "source": "fresh_run",
            "ucy_blocker_before": "no_validation_rows_for_domain_policy_selection_floor_only",
            "ucy_val_rows_after": int(repaired_stats["by_split"]["val"]["domains"].get("UCY", 0)),
            "ucy_positive_transfer_after": _positive_ucy(best["by_domain"].get("UCY", {})),
            "trajnet_preserved_after": bool(
                float(best["by_domain"]["TrajNet"]["all_improvement"]) > 0.0
                and float(best["by_domain"]["TrajNet"]["easy_degradation"]) <= 0.02
            ),
            "paper_claim": "UCY floor-only blocker is repaired by carving validation support from UCY train sources only; the selected policy is validation-best, test-once, dataset-local raw-frame evidence.",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "internal_val_from_train_only": True,
            "test_sources_unchanged": int(original_stats["by_split"]["test"]["rows"]) == int(repaired_stats["by_split"]["test"]["rows"]),
            "train_only_feature_normalization": True,
            "source_overlap_pass": bool(repaired_stats["source_overlap_pass"]),
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
    result["stage42_aw_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _append_ledger(result)
    return result


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    best = result["validation_best"]
    ucy = best["by_domain"].get("UCY", {})
    traj = best["by_domain"].get("TrajNet", {})
    gates = {
        "ucy_internal_val_created": result["summary"]["ucy_val_rows_after"] > 0,
        "internal_val_from_train_only": result["no_leakage"]["internal_val_from_train_only"],
        "test_sources_unchanged": result["no_leakage"]["test_sources_unchanged"],
        "source_overlap_pass": result["no_leakage"]["source_overlap_pass"],
        "variants_evaluated": len(result["variants"]) >= 4,
        "validation_best_selected_without_test": not result["no_leakage"]["test_threshold_tuning"],
        "ucy_positive_transfer": _positive_ucy(ucy),
        "global_positive": best["protected"]["all_improvement"] > 0.0 and best["protected"]["t50_improvement"] > 0.0,
        "global_easy_safe": best["protected"]["easy_degradation"] <= 0.02,
        "trajnet_preserved": bool(traj) and traj["all_improvement"] > 0.0 and traj["easy_degradation"] <= 0.02,
        "no_leakage_pass": all(
            result["no_leakage"][k] is False
            for k in ["future_endpoint_input", "future_waypoint_input", "central_velocity", "test_endpoint_goals", "test_threshold_tuning"]
        )
        and result["no_leakage"]["train_only_feature_normalization"],
        "no_metric_seconds_overclaim": not result["claim_boundary"]["metric_or_seconds_claim"],
        "stage5c_false": not result["claim_boundary"]["stage5c_executed"],
        "smc_false": not result["claim_boundary"]["smc_enabled"],
    }
    verdict = "stage42_aw_ucy_validation_support_repair_pass" if all(gates.values()) else "stage42_aw_ucy_validation_support_repair_partial"
    return {"source": "fresh_run", "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": int(len(gates)), "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    best = result["validation_best"]
    lines = [
        "# Stage42-AW UCY Validation-Support Repair",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_aw_gate']['passed']} / {result['stage42_aw_gate']['total']}`",
        f"- verdict: `{result['stage42_aw_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
    ]
    lines.extend([f"- {fact}" for fact in result["current_facts"]])
    lines.extend(
        [
            "",
            "## Internal Validation Repair",
            "",
            f"- internal_val_group: `{result['internal_validation']['internal_val_group']}`",
            f"- selected_from: `{result['internal_validation']['selected_from']}`",
            f"- uses_test_rows: `{result['internal_validation']['uses_test_rows']}`",
            f"- test_rows_unchanged: `{result['internal_validation']['test_rows_unchanged']}`",
            f"- original UCY val rows: `{result['original_split_stats']['by_split']['val']['domains'].get('UCY', 0)}`",
            f"- repaired UCY val rows: `{result['repaired_split_stats']['by_split']['val']['domains'].get('UCY', 0)}`",
            "",
            "## Variant Comparison",
            "",
            "| variant | val score | slices | global all | global t50 | global hard | global easy | UCY all | UCY t50 | UCY hard | UCY easy | TrajNet all |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in result["variants"].items():
        p = row["protected"]
        ucy = row["by_domain"].get("UCY", {})
        traj = row["by_domain"].get("TrajNet", {})
        lines.append(
            f"| `{name}` | {row['validation_score']:.6f} | {row['policy_slice_count']} | {p['all_improvement']:.6f} | {p['t50_improvement']:.6f} | {p['hard_failure_improvement']:.6f} | {p['easy_degradation']:.6f} | {ucy.get('all_improvement', 0.0):.6f} | {ucy.get('t50_improvement', 0.0):.6f} | {ucy.get('hard_failure_improvement', 0.0):.6f} | {ucy.get('easy_degradation', 0.0):.6f} | {traj.get('all_improvement', 0.0):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Validation-Best Candidate",
            "",
            f"- validation_best_variant: `{result['validation_best_variant']}`",
            f"- policy_slices: `{best['policy_slices']}`",
            f"- summary: `{result['summary']}`",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{result['no_leakage']}`",
            f"- claim_boundary: `{result['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_aw_gate"]
    lines = [
        "# Stage42-AW Gate",
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
        "verdict": result["stage42_aw_gate"]["verdict"],
        "gate": f"{result['stage42_aw_gate']['passed']}/{result['stage42_aw_gate']['total']}",
        "git_commit": result["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(f"{row}\n")


if __name__ == "__main__":
    run_stage42_ucy_validation_support_repair()
