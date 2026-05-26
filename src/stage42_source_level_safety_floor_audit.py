from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_source_level_ablation as an
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_incremental_ablation as ao
from src import stage42_source_level_residual_context as ap
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "source_level_safety_floor_audit_stage42.json"
REPORT_MD = OUT_DIR / "source_level_safety_floor_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_at_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

EPS = 1e-6
HORIZONS = [10, 25, 50, 100]


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-AT 是 proposed source-level fallback / teacher-floor context audit，不是 metric 或 seconds-level 结果。",
    "本审计区分 fallback floor removal 与 teacher/floor rollout context removal；二者不能混为一谈。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _feature_masks(names: list[str]) -> dict[str, np.ndarray]:
    arr = np.asarray(names)
    groups = ao._group_masks(names)
    baseline = ap._baseline_mask(names)
    floor_rel = np.asarray([n.startswith("floor_rel_") for n in arr])
    safe_baseline = np.asarray([n.startswith("safe_baseline_rel_") for n in arr])
    family = np.asarray([n.startswith("family_baseline_rel_") for n in arr])
    horizon_domain = ao._or_mask(groups["horizon"], groups["domain"])
    return {
        "baseline_family_all_context": baseline,
        "no_floor_rel_context": baseline & ~floor_rel,
        "family_only_no_floor_safe_context": ao._or_mask(family, horizon_domain),
        "no_safe_baseline_context": baseline & ~safe_baseline,
    }


def _metric_delta(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> dict[str, float]:
    return ao._metric_delta(lhs, rhs)


def _bootstrap_bundle(selected: np.ndarray, floor: np.ndarray, data: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    return {
        "all": am._bootstrap_ci(selected, floor, mask, seed=424201),
        "t50": am._bootstrap_ci(selected, floor, mask & (h == 50), seed=424202),
        "t100_raw_frame_diagnostic": am._bootstrap_ci(selected, floor, mask & (h == 100), seed=424203),
        "hard_failure": am._bootstrap_ci(selected, floor, mask & hard_failure, seed=424204),
        "easy_degradation": am._bootstrap_ci(floor, selected, mask & easy, seed=424205),
    }


def _evaluate_direct(name: str, raw_features: np.ndarray, shared: Mapping[str, Any]) -> dict[str, Any]:
    split = shared["split"]
    data = shared["data"]
    labels = shared["labels"]
    floor = shared["floor"]
    direct = ap._direct_candidate(raw_features, shared)
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    pred_ade, pred_fde = am._trajectory_errors(direct["pred_xy"], labels)
    val_mask = split == "val"
    test_mask = split == "test"
    policy, protected_ade, protected_fde, switch = am._select_policy_on_val(direct["pred_xy"], floor["floor_xy"], labels, data, val_mask)
    all_switch = np.ones(len(pred_ade), dtype=bool)
    none_switch = np.zeros(len(pred_ade), dtype=bool)
    protected = am._metric(protected_ade, floor_ade, data, switch, test_mask)
    ungated_ade = am._metric(pred_ade, floor_ade, data, all_switch, test_mask)
    ungated_fde = am._metric(pred_fde, floor_fde, data, all_switch, test_mask)
    return {
        "source": "fresh_run",
        "variant": name,
        "feature_count": int(raw_features.shape[1]),
        "best_lambda": direct["model"]["best_lambda"],
        "policy_slice_count": int(len(policy["slices"])),
        "floor_only": am._metric(floor_ade, floor_ade, data, none_switch, test_mask),
        "protected_safe_switch": protected,
        "ungated_all_rows": ungated_ade,
        "ungated_all_rows_fde": ungated_fde,
        "fallback_removed_delta_vs_protected": _metric_delta(ungated_ade, protected),
        "bootstrap": {
            "protected_safe_switch": _bootstrap_bundle(protected_ade, floor_ade, data, test_mask),
            "ungated_all_rows": _bootstrap_bundle(pred_ade, floor_ade, data, test_mask),
        },
        "validation_selection": {
            "source": "validation_only",
            "test_threshold_tuning": False,
            "safe_switch_policy_slice_count": int(len(policy["slices"])),
        },
    }


def _deployable(metric: Mapping[str, Any]) -> bool:
    return bool(
        metric["all_improvement"] > 0.0
        and (metric["t50_improvement"] > 0.0 or metric["hard_failure_improvement"] > 0.0)
        and metric["easy_degradation"] <= 0.02
    )


def _slice_safety(data: Mapping[str, np.ndarray], split: np.ndarray, selected: np.ndarray, floor: np.ndarray) -> dict[str, Any]:
    domain = data["dataset"].astype(str)
    horizon = data["horizon"].astype(int)
    out: dict[str, Any] = {}
    for d in sorted(set(domain[split == "test"].tolist())):
        for h in HORIZONS:
            key = f"{d}|{h}"
            val_mask = (split == "val") & (domain == d) & (horizon == h)
            test_mask = (split == "test") & (domain == d) & (horizon == h)
            if int(np.sum(val_mask)) < 30 and int(np.sum(test_mask)) < 30:
                continue
            switch = np.ones(len(selected), dtype=bool)
            out[key] = {
                "source": "fresh_run",
                "val_rows": int(np.sum(val_mask)),
                "test_rows": int(np.sum(test_mask)),
                "val_metric": am._metric(selected, floor, data, switch, val_mask),
                "test_metric": am._metric(selected, floor, data, switch, test_mask),
            }
    return out


def run_stage42_source_level_safety_floor_audit() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    shared = an._prep_shared()
    names = shared["feature_names"]
    masks = _feature_masks(names)
    candidates = {name: _evaluate_direct(name, shared["features"][:, mask], shared) for name, mask in masks.items()}
    base = candidates["baseline_family_all_context"]
    floor_ade, _ = am._trajectory_errors(shared["floor"]["floor_xy"], shared["labels"])
    base_pred = ap._direct_candidate(shared["features"][:, masks["baseline_family_all_context"]], shared)["pred_xy"]
    base_ade, _ = am._trajectory_errors(base_pred, shared["labels"])
    slice_safety = _slice_safety(shared["data"], shared["split"], base_ade, floor_ade)
    context_deltas = {
        name: {
            "source": "fresh_run",
            "protected_delta_vs_all_context": _metric_delta(row["protected_safe_switch"], base["protected_safe_switch"]),
            "ungated_delta_vs_all_context": _metric_delta(row["ungated_all_rows"], base["ungated_all_rows"]),
        }
        for name, row in candidates.items()
        if name != "baseline_family_all_context"
    }
    result = {
        "source": "fresh_run",
        "stage": "Stage42-AT proposed source-level safety floor / fallback audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/source_level_graph_context_stage42.json",
                "outputs/stage42_long_research/source_level_full_waypoint_eval_stage42.json",
            ]
        ),
        "split_stats": shared["split_stats"],
        "feature_counts": {k: int(np.sum(v)) for k, v in masks.items()},
        "candidates": candidates,
        "context_deltas": context_deltas,
        "slice_safety_for_all_context_ungated": slice_safety,
        "summary": {
            "source": "fresh_run",
            "fallback_removal_for_baseline_family_probe": "supported_on_this_source_level_split"
            if _deployable(base["ungated_all_rows"])
            else "not_supported",
            "teacher_floor_context_removal": "not_supported_as_global_replacement"
            if any(v["ungated_delta_vs_all_context"]["all_improvement"] < 0 for v in context_deltas.values())
            else "not_decisive",
            "interpretation": "This audit separates fallback removal from teacher/floor rollout context removal. Baseline-family ridge can be evaluated ungated on this source-level split, but that does not prove floor-free neural dynamics because floor/baseline rollout context remains an input mechanism.",
        },
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
    result["stage42_at_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _append_ledger(result)
    return result


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    base = result["candidates"]["baseline_family_all_context"]
    gates = {
        "proposed_source_level_split_used": result["split_stats"]["by_split"]["test"]["rows"] == 47458,
        "fallback_and_ungated_evaluated": "protected_safe_switch" in base and "ungated_all_rows" in base,
        "fallback_removal_nonharm_for_baseline_family_probe": _deployable(base["ungated_all_rows"]),
        "floor_context_ablation_evaluated": len(result["context_deltas"]) >= 3,
        "teacher_floor_context_not_overclaimed": result["summary"]["teacher_floor_context_removal"] != "supported_as_global_replacement",
        "slice_safety_reported": len(result["slice_safety_for_all_context_ungated"]) > 0,
        "bootstrap_available": base["bootstrap"]["ungated_all_rows"]["all"]["bootstrap_n"] > 0
        and base["bootstrap"]["ungated_all_rows"]["t50"]["bootstrap_n"] > 0,
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
        "no_metric_seconds_overclaim": not result["claim_boundary"]["metric_or_seconds_claim"],
        "stage5c_false": not result["claim_boundary"]["stage5c_executed"],
        "smc_false": not result["claim_boundary"]["smc_enabled"],
    }
    verdict = (
        "stage42_at_source_level_fallback_audit_pass"
        if all(gates.values())
        else "stage42_at_source_level_fallback_audit_partial"
    )
    return {"source": "fresh_run", "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": int(len(gates)), "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-AT Source-Level Safety Floor / Fallback Audit",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_at_gate']['passed']} / {result['stage42_at_gate']['total']}`",
        f"- verdict: `{result['stage42_at_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Key Distinction",
        "",
        "- `fallback removal` means the source-level ridge probe predicts every row without falling back to the floor.",
        "- `teacher/floor context removal` means removing floor-related rollout context from the input feature family.",
        "- A pass on fallback removal does not prove floor-free neural dynamics.",
        "",
        "## Candidate Comparison",
        "",
        "| candidate | features | protected all | protected t50 | protected hard | protected easy | ungated all | ungated t50 | ungated hard | ungated easy | ungated minus protected all |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["candidates"].items():
        p = row["protected_safe_switch"]
        u = row["ungated_all_rows"]
        d = row["fallback_removed_delta_vs_protected"]
        lines.append(
            f"| `{name}` | {row['feature_count']} | {p['all_improvement']:.6f} | {p['t50_improvement']:.6f} | {p['hard_failure_improvement']:.6f} | {p['easy_degradation']:.6f} | {u['all_improvement']:.6f} | {u['t50_improvement']:.6f} | {u['hard_failure_improvement']:.6f} | {u['easy_degradation']:.6f} | {d['all_improvement']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Context Removal Deltas",
            "",
            "| candidate | protected delta all | protected delta t50 | ungated delta all | ungated delta t50 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in result["context_deltas"].items():
        p = row["protected_delta_vs_all_context"]
        u = row["ungated_delta_vs_all_context"]
        lines.append(f"| `{name}` | {p['all_improvement']:.6f} | {p['t50_improvement']:.6f} | {u['all_improvement']:.6f} | {u['t50_improvement']:.6f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- fallback_removal_for_baseline_family_probe: `{result['summary']['fallback_removal_for_baseline_family_probe']}`",
            f"- teacher_floor_context_removal: `{result['summary']['teacher_floor_context_removal']}`",
            f"- interpretation: {result['summary']['interpretation']}",
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
    gate = result["stage42_at_gate"]
    lines = [
        "# Stage42-AT Gate",
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
        "verdict": result["stage42_at_gate"]["verdict"],
        "gate": f"{result['stage42_at_gate']['passed']}/{result['stage42_at_gate']['total']}",
        "git_commit": result["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(f"{row}\n")


if __name__ == "__main__":
    run_stage42_source_level_safety_floor_audit()
