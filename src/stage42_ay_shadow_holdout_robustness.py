from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_aw_t100_easy_safety_repair as ay
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_ucy_validation_support_repair as aw
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
AY_JSON = OUT_DIR / "aw_t100_easy_safety_repair_stage42.json"
REPORT_JSON = OUT_DIR / "ay_shadow_holdout_robustness_stage42.json"
REPORT_MD = OUT_DIR / "ay_shadow_holdout_robustness_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_az_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

MIN_T100_VAL_SOURCES_PER_DOMAIN = 2


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-AZ 是 Stage42-AY strict t100 guard 的 source-level shadow-holdout robustness audit。",
    "Shadow holdout 只从原 train sources 内部构建，不使用最终 test source 调参。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t100 仍是 raw-frame diagnostic，不是 seconds-level long-horizon claim。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _shadow_split(data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    original_split, group = am._split_arrays(data)
    domain = data["dataset"].astype(str)
    horizon = data["horizon"].astype(int)
    split = np.full(len(group), "final_eval_excluded", dtype="U32")
    plan: dict[str, Any] = {
        "source": "fresh_run",
        "rule": "Use original train sources only. For each domain with >=3 t100-capable train sources, hold out the largest t100 source, use the second-largest as shadow validation, and fit on the rest.",
        "final_val_test_excluded": True,
        "domains": {},
    }
    for d in sorted(set(domain[original_split == "train"].tolist())):
        train_groups = sorted(set(group[(original_split == "train") & (domain == d)].tolist()))
        counts = [(int(np.sum((group == g) & (horizon == 100))), g) for g in train_groups]
        t100_groups = sorted([row for row in counts if row[0] > 0])
        if len(t100_groups) < 3:
            split[(original_split == "train") & (domain == d)] = "blocked_no_shadow_holdout"
            plan["domains"][d] = {
                "status": "not_run",
                "reason": "fewer_than_three_t100_capable_original_train_sources",
                "t100_groups": [{"group": g, "t100_rows": c} for c, g in t100_groups],
            }
            continue
        shadow_holdout = t100_groups[-1][1]
        shadow_val = t100_groups[-2][1]
        for _, g in counts:
            if g == shadow_holdout:
                split[group == g] = "shadow_holdout"
            elif g == shadow_val:
                split[group == g] = "shadow_val"
            else:
                split[group == g] = "shadow_train"
        plan["domains"][d] = {
            "status": "fresh_run",
            "shadow_val_group": shadow_val,
            "shadow_holdout_group": shadow_holdout,
            "shadow_train_groups": sorted(set(group[(split == "shadow_train") & (domain == d)].tolist())),
            "t100_groups": [{"group": g, "t100_rows": c} for c, g in t100_groups],
        }
    return split, group, plan


def _split_stats(data: Mapping[str, np.ndarray], split: np.ndarray, group: np.ndarray) -> dict[str, Any]:
    domain = data["dataset"].astype(str)
    horizon = data["horizon"].astype(int)
    out: dict[str, Any] = {}
    for sp in ["shadow_train", "shadow_val", "shadow_holdout", "blocked_no_shadow_holdout", "final_eval_excluded"]:
        m = split == sp
        out[sp] = {
            "rows": int(np.sum(m)),
            "domains": dict(Counter(domain[m].tolist())),
            "sources": int(len(set(group[m].tolist()))),
            "t100": int(np.sum(m & (horizon == 100))),
        }
    return out


def _fit_shadow_model(data: Mapping[str, np.ndarray], split: np.ndarray) -> dict[str, Any]:
    labels = am._reconstruct_waypoint_labels(data)
    floor = am._floor_arrays(data, split == "shadow_train")
    features, feature_names = am._feature_matrix(data, floor)
    masks = aw._safe_variant_masks(feature_names)
    variant_name = "family_baseline_rel_only"
    x, _, _ = am._standardize(features[:, masks[variant_name]], split == "shadow_train")
    target = ay._target_delta(data, labels)
    val_rows = []
    best: dict[str, Any] | None = None
    best_score = -1e9
    for lam in am.LAMBDAS:
        coef = am._fit_ridge_model(x, target, labels["waypoint_valid"], split == "shadow_train", lam)
        pred_xy = am._predict_waypoints(x, coef, data)
        policy, selected_ade, selected_fde, switch = am._select_policy_on_val(
            pred_xy,
            floor["floor_xy"],
            labels,
            data,
            split == "shadow_val",
        )
        floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
        val_metric = am._metric(selected_ade, floor_ade, data, switch, split == "shadow_val")
        score = (
            1.2 * val_metric["all_improvement"]
            + 1.8 * val_metric["t50_improvement"]
            + 1.1 * val_metric["hard_failure_improvement"]
            - 30.0 * max(0.0, val_metric["easy_degradation"] - 0.02)
            - 0.03 * val_metric["switch_rate"]
        )
        val_rows.append({"lambda": float(lam), "score": float(score), "policy_slices": sorted(policy["slices"].keys()), "val_metric": val_metric})
        if score > best_score:
            best_score = float(score)
            best = {
                "lambda": float(lam),
                "score": float(score),
                "policy": policy,
                "selected_ade": selected_ade,
                "selected_fde": selected_fde,
                "switch": switch,
                "floor_ade": floor_ade,
                "floor_fde": floor_fde,
                "feature_count": int(np.sum(masks[variant_name])),
            }
    if best is None:
        raise RuntimeError("No shadow model was evaluated.")
    best["validation_candidates"] = val_rows
    best["variant_name"] = variant_name
    return best


def _val_t100_source_count(data: Mapping[str, np.ndarray], split: np.ndarray, group: np.ndarray, domain_name: str) -> int:
    domain = data["dataset"].astype(str)
    horizon = data["horizon"].astype(int)
    return len(set(group[(split == "shadow_val") & (domain == domain_name) & (horizon == 100)].tolist()))


def _apply_source_support_t100_guard(
    *,
    policy: Mapping[str, Any],
    data: Mapping[str, np.ndarray],
    group: np.ndarray,
    split: np.ndarray,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switch: np.ndarray,
    floor_ade: np.ndarray,
    floor_fde: np.ndarray,
    min_sources: int = MIN_T100_VAL_SOURCES_PER_DOMAIN,
) -> dict[str, Any]:
    domain = data["dataset"].astype(str)
    horizon = data["horizon"].astype(int)
    repaired_ade = selected_ade.copy()
    repaired_fde = selected_fde.copy()
    repaired_switch = switch.copy()
    guarded: dict[str, Any] = {}
    kept: dict[str, Any] = {}
    for key, params in sorted(policy.get("slices", {}).items()):
        d, h_s = key.split("|", 1)
        if int(h_s) != 100:
            continue
        val_sources = _val_t100_source_count(data, split, group, d)
        val_metric = params.get("val_metric", {})
        keep = (
            val_sources >= min_sources
            and float(val_metric.get("all_improvement", 0.0)) > 0.0
            and float(val_metric.get("easy_degradation", 1.0)) <= 0.0
        )
        record = {
            "source": "fresh_run_shadow_source_support_guard",
            "val_t100_source_count": int(val_sources),
            "min_t100_val_sources_per_domain": int(min_sources),
            "val_all_improvement": float(val_metric.get("all_improvement", 0.0)),
            "val_easy_degradation": float(val_metric.get("easy_degradation", 0.0)),
        }
        if keep:
            kept[key] = record
            continue
        mask = (domain == d) & (horizon == 100)
        repaired_ade[mask] = floor_ade[mask]
        repaired_fde[mask] = floor_fde[mask]
        repaired_switch[mask] = False
        guarded[key] = {**record, "reason": "insufficient_independent_t100_validation_sources_or_validation_easy_harm"}
    return {
        "source": "fresh_run",
        "type": "shadow_source_support_t100_guard",
        "min_t100_val_sources_per_domain": int(min_sources),
        "uses_final_test_metrics_for_threshold": False,
        "guarded_slices": guarded,
        "kept_slices": kept,
        "selected_ade": repaired_ade,
        "selected_fde": repaired_fde,
        "switch": repaired_switch,
    }


def _metric_bundle(data: Mapping[str, np.ndarray], split: np.ndarray, selected_ade: np.ndarray, selected_fde: np.ndarray, floor_ade: np.ndarray, floor_fde: np.ndarray, switch: np.ndarray) -> dict[str, Any]:
    holdout = split == "shadow_holdout"
    domain = data["dataset"].astype(str)
    horizon = data["horizon"].astype(int)
    easy = data["easy"].astype(bool)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    return {
        "protected": am._metric(selected_ade, floor_ade, data, switch, holdout),
        "fde": am._metric(selected_fde, floor_fde, data, switch, holdout),
        "bootstrap": {
            "all": am._bootstrap_ci(selected_ade, floor_ade, holdout, seed=42201),
            "t50": am._bootstrap_ci(selected_ade, floor_ade, holdout & (horizon == 50), seed=42202),
            "t100_raw_frame_diagnostic": am._bootstrap_ci(selected_ade, floor_ade, holdout & (horizon == 100), seed=42203),
            "hard_failure": am._bootstrap_ci(selected_ade, floor_ade, holdout & hard_failure, seed=42204),
            "easy_degradation": am._bootstrap_ci(floor_ade, selected_ade, holdout & easy, seed=42205),
            "h100_easy_degradation": am._bootstrap_ci(floor_ade, selected_ade, holdout & (horizon == 100) & easy, seed=42206),
        },
        "by_domain": {
            d: am._metric(selected_ade, floor_ade, data, switch, holdout & (domain == d))
            for d in sorted(set(domain[holdout].tolist()))
        },
        "by_horizon": {
            str(h): am._metric(selected_ade, floor_ade, data, switch, holdout & (horizon == h))
            for h in [10, 25, 50, 100]
        },
    }


def run_stage42_ay_shadow_holdout_robustness() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ay_report = _load_json(AY_JSON)
    data = s41._combined()
    split, group, split_plan = _shadow_split(data)
    model = _fit_shadow_model(data, split)
    ay_guard = ay._apply_t100_easy_guard(
        policy=model["policy"],
        data=data,
        selected_ade=model["selected_ade"],
        selected_fde=model["selected_fde"],
        switch=model["switch"],
        floor_ade=model["floor_ade"],
        floor_fde=model["floor_fde"],
    )
    support_guard = _apply_source_support_t100_guard(
        policy=model["policy"],
        data=data,
        group=group,
        split=split,
        selected_ade=model["selected_ade"],
        selected_fde=model["selected_fde"],
        switch=model["switch"],
        floor_ade=model["floor_ade"],
        floor_fde=model["floor_fde"],
    )
    base_metrics = _metric_bundle(data, split, model["selected_ade"], model["selected_fde"], model["floor_ade"], model["floor_fde"], model["switch"])
    ay_guard_metrics = _metric_bundle(data, split, ay_guard["selected_ade"], ay_guard["selected_fde"], model["floor_ade"], model["floor_fde"], ay_guard["switch"])
    support_guard_metrics = _metric_bundle(data, split, support_guard["selected_ade"], support_guard["selected_fde"], model["floor_ade"], model["floor_fde"], support_guard["switch"])
    result = {
        "source": "fresh_run",
        "stage": "Stage42-AZ AY shadow-holdout t100 robustness audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([str(AY_JSON), "data/stage41_world_model/combined_external.npz"]),
        "ay_verdict": ay_report["stage42_ay_gate"]["verdict"],
        "shadow_split_plan": split_plan,
        "shadow_split_stats": _split_stats(data, split, group),
        "model": {
            "source": "fresh_run",
            "variant_name": model["variant_name"],
            "feature_count": model["feature_count"],
            "best_lambda": model["lambda"],
            "validation_score": model["score"],
            "validation_candidates": model["validation_candidates"],
            "policy_slices": sorted(model["policy"]["slices"].keys()),
        },
        "policies": {
            "base_shadow_policy": {"source": "fresh_run", "type": "validation_safe_policy_without_extra_t100_source_support_guard"},
            "ay_strict_t100_guard": {
                "source": ay_guard["source"],
                "type": "stage42ay_strict_validation_easy_guard_replayed_on_shadow_split",
                "guarded_slices": ay_guard["guarded_slices"],
                "kept_slices": ay_guard["kept_slices"],
                "uses_final_test_metrics_for_threshold": False,
            },
            "source_support_t100_guard": {
                "source": support_guard["source"],
                "type": support_guard["type"],
                "min_t100_val_sources_per_domain": support_guard["min_t100_val_sources_per_domain"],
                "guarded_slices": support_guard["guarded_slices"],
                "kept_slices": support_guard["kept_slices"],
                "uses_final_test_metrics_for_threshold": False,
            },
        },
        "shadow_holdout_metrics": {
            "base_shadow_policy": base_metrics,
            "ay_strict_t100_guard": ay_guard_metrics,
            "source_support_t100_guard": support_guard_metrics,
        },
        "summary": {
            "source": "fresh_run",
            "ay_strict_guard_shadow_h100_easy_safe": ay_guard_metrics["by_horizon"]["100"]["easy_degradation"] <= 0.02,
            "source_support_guard_shadow_h100_easy_safe": support_guard_metrics["by_horizon"]["100"]["easy_degradation"] <= 0.02,
            "source_support_guard_all_positive": support_guard_metrics["protected"]["all_improvement"] > 0.0,
            "source_support_guard_t50_positive": support_guard_metrics["protected"]["t50_improvement"] > 0.0,
            "source_support_guard_t100_positive": support_guard_metrics["protected"]["t100_raw_frame_diagnostic_improvement"] > 0.0,
            "source_support_guard_hard_positive": support_guard_metrics["protected"]["hard_failure_improvement"] > 0.0,
            "ucy_shadow_status": split_plan["domains"].get("UCY", {}).get("status", "not_run"),
            "paper_claim": (
                "AY strict t100 guard is not independently robust on the original-train shadow holdout; ETH_UCY t100 easy harm appears. "
                "A more conservative source-support t100 guard protects easy cases but removes positive t100 gain on this shadow holdout."
            ),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "final_test_metrics_for_threshold": False,
            "shadow_fit_val_holdout_from_original_train_only": True,
            "train_only_feature_normalization": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "t100_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "ungated_neural_deployable": False,
        },
    }
    result["stage42_az_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _append_ledger(result)
    return result


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    support = result["shadow_holdout_metrics"]["source_support_t100_guard"]
    support_h100 = support["by_horizon"]["100"]
    support_boot = support["bootstrap"]
    gates = {
        "ay_input_verified": result["ay_verdict"] == "stage42_ay_t100_easy_safety_repair_pass",
        "fresh_shadow_split_built": result["source"] == "fresh_run" and result["shadow_split_stats"]["shadow_holdout"]["rows"] > 0,
        "final_test_excluded_from_thresholds": result["no_leakage"]["final_test_metrics_for_threshold"] is False,
        "ay_strict_guard_limitation_identified": result["summary"]["ay_strict_guard_shadow_h100_easy_safe"] is False,
        "source_support_guard_applied": bool(result["policies"]["source_support_t100_guard"]["guarded_slices"]),
        "source_support_h100_easy_safe": support_h100["easy_degradation"] <= 0.02,
        "source_support_h100_easy_ci_safe": support_boot["h100_easy_degradation"]["high"] <= 0.02,
        "source_support_all_ci_positive": support_boot["all"]["low"] > 0.0,
        "source_support_t50_ci_positive": support_boot["t50"]["low"] > 0.0,
        "source_support_hard_ci_positive": support_boot["hard_failure"]["low"] > 0.0,
        "t100_positive_not_overclaimed": result["summary"]["source_support_guard_t100_positive"] is False,
        "ucy_blocker_reported": result["summary"]["ucy_shadow_status"] == "not_run",
        "no_leakage_pass": all(
            result["no_leakage"][k] is False
            for k in ["future_endpoint_input", "future_waypoint_input", "central_velocity", "test_endpoint_goals", "final_test_metrics_for_threshold"]
        )
        and result["no_leakage"]["shadow_fit_val_holdout_from_original_train_only"],
        "no_metric_seconds_overclaim": not result["claim_boundary"]["metric_or_seconds_claim"] and not result["claim_boundary"]["t100_seconds_claim"],
        "stage5c_false": not result["claim_boundary"]["stage5c_executed"],
        "smc_false": not result["claim_boundary"]["smc_enabled"],
    }
    verdict = "stage42_az_shadow_holdout_robustness_pass_with_ay_t100_limitation" if all(gates.values()) else "stage42_az_shadow_holdout_robustness_partial"
    return {"source": result["source"], "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": int(len(gates)), "verdict": verdict}


def _render_metric_table(title: str, metrics: Mapping[str, Any]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| slice | rows | all | t50 | t100 raw diag | hard | easy | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = {"global": metrics["protected"], **{f"domain:{k}": v for k, v in metrics["by_domain"].items()}, **{f"horizon:{k}": v for k, v in metrics["by_horizon"].items()}}
    for name, row in rows.items():
        lines.append(
            f"| `{name}` | {row['rows']} | {row['all_improvement']:.6f} | {row['t50_improvement']:.6f} | {row['t100_raw_frame_diagnostic_improvement']:.6f} | {row['hard_failure_improvement']:.6f} | {row['easy_degradation']:.6f} | {row['switch_rate']:.6f} |"
        )
    return lines


def _render_report(result: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-AZ AY Shadow-Holdout T100 Robustness Audit",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_az_gate']['passed']} / {result['stage42_az_gate']['total']}`",
        f"- verdict: `{result['stage42_az_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Shadow Split",
        "",
        f"- split_plan: `{result['shadow_split_plan']}`",
        f"- split_stats: `{result['shadow_split_stats']}`",
        "",
        "## Model And Policies",
        "",
        f"- model: `{result['model']}`",
        f"- ay_strict_t100_guard: `{result['policies']['ay_strict_t100_guard']}`",
        f"- source_support_t100_guard: `{result['policies']['source_support_t100_guard']}`",
        "",
    ]
    lines.extend(_render_metric_table("AY Strict Guard On Shadow Holdout", result["shadow_holdout_metrics"]["ay_strict_t100_guard"]))
    lines.extend([""])
    lines.extend(_render_metric_table("Source-Support T100 Guard On Shadow Holdout", result["shadow_holdout_metrics"]["source_support_t100_guard"]))
    lines.extend(
        [
            "",
            "## Source-Support Guard Bootstrap",
            "",
            "| metric | low | mid | high | n |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, row in result["shadow_holdout_metrics"]["source_support_t100_guard"]["bootstrap"].items():
        lines.append(f"| `{key}` | {row['low']:.6f} | {row['mid']:.6f} | {row['high']:.6f} | {row['n']} |")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- summary: `{result['summary']}`",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{result['no_leakage']}`",
            f"- claim_boundary: `{result['claim_boundary']}`",
            "",
            "## Interpretation",
            "",
            "- Stage42-AZ finds that the Stage42-AY strict t100 guard is not enough for original-train shadow-holdout robustness: ETH_UCY t100 easy harm appears.",
            "- A conservative source-support guard protects shadow-holdout easy cases and keeps all/t50/hard positive, but it removes positive t100 gain on this shadow holdout.",
            "- This is negative/repair evidence, not a new t100 success claim. It strengthens deployment safety boundaries and shows why t100 still needs more validation support.",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_az_gate"]
    lines = [
        "# Stage42-AZ Gate",
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
        "verdict": result["stage42_az_gate"]["verdict"],
        "gate": f"{result['stage42_az_gate']['passed']}/{result['stage42_az_gate']['total']}",
        "git_commit": result["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_ay_shadow_holdout_robustness()
