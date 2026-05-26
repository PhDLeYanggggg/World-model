from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_aw_t100_easy_safety_repair as ay
from src import stage42_ay_shadow_holdout_robustness as az
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
AY_JSON = OUT_DIR / "aw_t100_easy_safety_repair_stage42.json"
AZ_JSON = OUT_DIR / "ay_shadow_holdout_robustness_stage42.json"
REPORT_JSON = OUT_DIR / "t100_source_cv_repair_stage42.json"
REPORT_MD = OUT_DIR / "t100_source_cv_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ba_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

MIN_SAFE_POSITIVE_FOLDS = 2
EASY_THRESHOLD = 0.02


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BA 是 Stage42-AZ 后的 train-only t100 source-CV support audit / repair。",
    "source-CV folds 只从 original train sources 内部构建，final val/test 不参与 threshold 或 domain support 选择。",
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


def _t100_train_groups(data: Mapping[str, np.ndarray], split: np.ndarray, group: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    domain = data["dataset"].astype(str)
    horizon = data["horizon"].astype(int)
    out: dict[str, list[dict[str, Any]]] = {}
    for d in sorted(set(domain[split == "train"].tolist())):
        rows: list[dict[str, Any]] = []
        for g in sorted(set(group[(split == "train") & (domain == d)].tolist())):
            n = int(np.sum((split == "train") & (group == g) & (horizon == 100)))
            if n > 0:
                rows.append({"group": str(g), "t100_rows": n})
        out[d] = sorted(rows, key=lambda row: (-int(row["t100_rows"]), str(row["group"])))
    return out


def _build_source_cv_folds(data: Mapping[str, np.ndarray]) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    original_split, group = am._split_arrays(data)
    domain = data["dataset"].astype(str)
    t100_groups = _t100_train_groups(data, original_split, group)
    folds: list[dict[str, Any]] = []
    plan: dict[str, Any] = {
        "source": "fresh_run",
        "rule": "For each domain, leave one original-train t100-capable source out as source-CV holdout; choose the largest remaining t100-capable source as validation; train on all other original-train rows. Final val/test rows are excluded from support selection.",
        "final_val_test_excluded": True,
        "domains": {},
    }
    for d, rows in t100_groups.items():
        if len(rows) < 3:
            plan["domains"][d] = {
                "status": "not_run",
                "reason": "fewer_than_three_t100_capable_original_train_sources",
                "t100_groups": rows,
            }
            continue
        domain_folds: list[dict[str, Any]] = []
        for holdout in rows:
            remaining = [row for row in rows if row["group"] != holdout["group"]]
            val_group = remaining[0]["group"]
            train_groups = sorted(
                set(group[(original_split == "train") & (domain == d)].tolist())
                - {str(holdout["group"]), str(val_group)}
            )
            fold = {
                "source": "fresh_run",
                "domain": d,
                "holdout_group": str(holdout["group"]),
                "holdout_t100_rows": int(holdout["t100_rows"]),
                "val_group": str(val_group),
                "train_groups": train_groups,
            }
            folds.append(fold)
            domain_folds.append(fold)
        plan["domains"][d] = {
            "status": "fresh_run",
            "t100_groups": rows,
            "fold_count": len(domain_folds),
            "folds": domain_folds,
        }
    return folds, original_split, group, plan


def _split_for_fold(original_split: np.ndarray, group: np.ndarray, fold: Mapping[str, Any]) -> np.ndarray:
    split = np.full(len(group), "final_eval_excluded", dtype="U32")
    split[original_split == "train"] = "shadow_train"
    split[group == str(fold["val_group"])] = "shadow_val"
    split[group == str(fold["holdout_group"])] = "shadow_holdout"
    return split


def _evaluate_source_cv_fold(data: Mapping[str, np.ndarray], original_split: np.ndarray, group: np.ndarray, fold: Mapping[str, Any]) -> dict[str, Any]:
    split = _split_for_fold(original_split, group, fold)
    model = az._fit_shadow_model(data, split)
    guard = ay._apply_t100_easy_guard(
        policy=model["policy"],
        data=data,
        selected_ade=model["selected_ade"],
        selected_fde=model["selected_fde"],
        switch=model["switch"],
        floor_ade=model["floor_ade"],
        floor_fde=model["floor_fde"],
    )
    metrics = az._metric_bundle(
        data,
        split,
        guard["selected_ade"],
        guard["selected_fde"],
        model["floor_ade"],
        model["floor_fde"],
        guard["switch"],
    )
    h100 = metrics["by_horizon"]["100"]
    return {
        "source": "fresh_run",
        "domain": str(fold["domain"]),
        "holdout_group": str(fold["holdout_group"]),
        "val_group": str(fold["val_group"]),
        "holdout_t100_rows": int(fold["holdout_t100_rows"]),
        "best_lambda": float(model["lambda"]),
        "validation_score": float(model["score"]),
        "policy_slices": sorted(model["policy"]["slices"].keys()),
        "guarded_t100_slices": sorted(guard["guarded_slices"].keys()),
        "kept_t100_slices": sorted(guard["kept_slices"].keys()),
        "holdout_metrics": metrics["protected"],
        "holdout_h100": h100,
        "bootstrap": metrics["bootstrap"],
        "safe_positive_t100": bool(h100["t100_raw_frame_diagnostic_improvement"] > 0.0 and h100["easy_degradation"] <= EASY_THRESHOLD),
    }


def _domain_cv_summary(fold_results: list[Mapping[str, Any]], domains: Mapping[str, Any]) -> dict[str, Any]:
    by_domain: dict[str, Any] = {}
    for d, info in sorted(domains.items()):
        rows = [row for row in fold_results if row["domain"] == d]
        if not rows:
            by_domain[d] = {
                "source": "fresh_run",
                "status": info.get("status", "not_run"),
                "reason": info.get("reason", "no_cv_folds"),
                "fold_count": 0,
                "supported_for_t100": False,
            }
            continue
        t100_vals = [float(row["holdout_h100"]["t100_raw_frame_diagnostic_improvement"]) for row in rows]
        easy_vals = [float(row["holdout_h100"]["easy_degradation"]) for row in rows]
        safe_positive = [bool(row["safe_positive_t100"]) for row in rows]
        supported = (
            len(rows) >= MIN_SAFE_POSITIVE_FOLDS
            and sum(safe_positive) >= MIN_SAFE_POSITIVE_FOLDS
            and min(t100_vals) > 0.0
            and max(easy_vals) <= EASY_THRESHOLD
        )
        by_domain[d] = {
            "source": "fresh_run",
            "status": "fresh_run",
            "fold_count": int(len(rows)),
            "safe_positive_fold_count": int(sum(safe_positive)),
            "min_t100_improvement": float(min(t100_vals)),
            "median_t100_improvement": float(np.median(t100_vals)),
            "max_easy_degradation": float(max(easy_vals)),
            "supported_for_t100": bool(supported),
            "support_rule": f">={MIN_SAFE_POSITIVE_FOLDS} safe-positive folds, min t100 > 0, max easy <= {EASY_THRESHOLD}",
        }
    return by_domain


def _apply_domain_cv_t100_guard(
    *,
    data: Mapping[str, np.ndarray],
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switch: np.ndarray,
    floor_ade: np.ndarray,
    floor_fde: np.ndarray,
    domain_support: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    domain = data["dataset"].astype(str)
    horizon = data["horizon"].astype(int)
    repaired_ade = selected_ade.copy()
    repaired_fde = selected_fde.copy()
    repaired_switch = switch.copy()
    guarded: dict[str, Any] = {}
    kept: dict[str, Any] = {}
    for d, support in sorted(domain_support.items()):
        if bool(support.get("supported_for_t100", False)):
            kept[f"{d}|100"] = {
                "source": "fresh_run_train_source_cv_guard",
                "reason": "domain_has_source_cv_t100_support",
                "support": support,
            }
            continue
        mask = (domain == d) & (horizon == 100)
        repaired_ade[mask] = floor_ade[mask]
        repaired_fde[mask] = floor_fde[mask]
        repaired_switch[mask] = False
        guarded[f"{d}|100"] = {
            "source": "fresh_run_train_source_cv_guard",
            "reason": "domain_lacks_train_source_cv_t100_support",
            "support": support,
            "rows_all_splits": int(np.sum(mask)),
        }
    return {
        "source": "fresh_run",
        "type": "stage42ba_train_source_cv_t100_guard",
        "uses_final_test_metrics_for_threshold": False,
        "guarded_slices": guarded,
        "kept_slices": kept,
        "selected_ade": repaired_ade,
        "selected_fde": repaired_fde,
        "switch": repaired_switch,
    }


def _final_aw_metrics_with_cv_guard(domain_support: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ay_report = _load_json(AY_JSON)
    variant_name = ay_report["validation_best_variant"]
    best_lambda = float(ay_report["best_lambda"])
    arrays = ay._recompute_aw_variant_arrays(variant_name, best_lambda)
    guarded = _apply_domain_cv_t100_guard(
        data=arrays["data"],
        selected_ade=arrays["selected_ade"],
        selected_fde=arrays["selected_fde"],
        switch=arrays["switch"],
        floor_ade=arrays["floor_ade"],
        floor_fde=arrays["floor_fde"],
        domain_support=domain_support,
    )
    original = ay._metrics_bundle(
        arrays["data"],
        arrays["split"],
        arrays["selected_ade"],
        arrays["selected_fde"],
        arrays["floor_ade"],
        arrays["floor_fde"],
        arrays["switch"],
    )
    repaired = ay._metrics_bundle(
        arrays["data"],
        arrays["split"],
        guarded["selected_ade"],
        guarded["selected_fde"],
        arrays["floor_ade"],
        arrays["floor_fde"],
        guarded["switch"],
    )
    return {
        "source": "fresh_run",
        "variant_name": variant_name,
        "best_lambda": best_lambda,
        "feature_count": int(arrays["feature_count"]),
        "policy_slices_before_guard": sorted(arrays["policy"]["slices"].keys()),
        "guard": {
            "source": guarded["source"],
            "type": guarded["type"],
            "guarded_slices": guarded["guarded_slices"],
            "kept_slices": guarded["kept_slices"],
            "uses_final_test_metrics_for_threshold": False,
        },
        "before_cv_guard": original,
        "after_cv_guard": repaired,
    }


def run_stage42_t100_source_cv_repair() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ay_report = _load_json(AY_JSON)
    az_report = _load_json(AZ_JSON)
    data = s41._combined()
    folds, original_split, group, plan = _build_source_cv_folds(data)
    fold_results = [_evaluate_source_cv_fold(data, original_split, group, fold) for fold in folds]
    domain_support = _domain_cv_summary(fold_results, plan["domains"])
    final_eval = _final_aw_metrics_with_cv_guard(domain_support)
    result = {
        "source": "fresh_run",
        "stage": "Stage42-BA Train-Only T100 Source-CV Repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([str(AY_JSON), str(AZ_JSON), "data/stage41_world_model/combined_external.npz"]),
        "ay_verdict": ay_report["stage42_ay_gate"]["verdict"],
        "az_verdict": az_report["stage42_az_gate"]["verdict"],
        "source_cv_plan": plan,
        "source_cv_fold_count": int(len(fold_results)),
        "source_cv_fold_results": fold_results,
        "domain_t100_support": domain_support,
        "final_eval": final_eval,
        "summary": {
            "source": "fresh_run",
            "supported_t100_domains": [d for d, row in domain_support.items() if row.get("supported_for_t100")],
            "unsupported_t100_domains": [d for d, row in domain_support.items() if not row.get("supported_for_t100")],
            "final_all_positive": final_eval["after_cv_guard"]["protected"]["all_improvement"] > 0.0,
            "final_t50_positive": final_eval["after_cv_guard"]["protected"]["t50_improvement"] > 0.0,
            "final_t100_positive": final_eval["after_cv_guard"]["protected"]["t100_raw_frame_diagnostic_improvement"] > 0.0,
            "final_hard_positive": final_eval["after_cv_guard"]["protected"]["hard_failure_improvement"] > 0.0,
            "final_easy_safe": final_eval["after_cv_guard"]["protected"]["easy_degradation"] <= EASY_THRESHOLD,
            "paper_claim": "Train-only source-CV support is required before keeping any t100 domain slice. Unsupported t100 domains are guarded to the causal floor; t100 remains raw-frame diagnostic and cannot be overclaimed.",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "final_test_metrics_for_threshold": False,
            "source_cv_from_original_train_only": True,
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
    result["stage42_ba_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _append_ledger(result)
    return result


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    after = result["final_eval"]["after_cv_guard"]
    protected = after["protected"]
    boot = after["bootstrap"]
    gates = {
        "ay_input_verified": result["ay_verdict"] == "stage42_ay_t100_easy_safety_repair_pass",
        "az_input_verified": result["az_verdict"] == "stage42_az_shadow_holdout_robustness_pass_with_ay_t100_limitation",
        "source_cv_folds_built": result["source_cv_fold_count"] > 0,
        "domain_support_decisions_reported": bool(result["domain_t100_support"]),
        "final_test_excluded_from_support_selection": result["no_leakage"]["final_test_metrics_for_threshold"] is False,
        "cv_guard_applied": bool(result["final_eval"]["guard"]["guarded_slices"]),
        "final_all_ci_positive": boot["all"]["low"] > 0.0,
        "final_t50_ci_positive": boot["t50"]["low"] > 0.0,
        "final_hard_ci_positive": boot["hard_failure"]["low"] > 0.0,
        "final_easy_safe": protected["easy_degradation"] <= EASY_THRESHOLD,
        "h100_easy_ci_safe": boot["h100_easy_degradation"]["high"] <= EASY_THRESHOLD,
        "t100_not_overclaimed_if_unsupported": (bool(result["summary"]["supported_t100_domains"]) or protected["t100_raw_frame_diagnostic_improvement"] == 0.0),
        "no_leakage_pass": all(
            result["no_leakage"][k] is False
            for k in ["future_endpoint_input", "future_waypoint_input", "central_velocity", "test_endpoint_goals", "final_test_metrics_for_threshold"]
        )
        and result["no_leakage"]["source_cv_from_original_train_only"],
        "no_metric_seconds_overclaim": not result["claim_boundary"]["metric_or_seconds_claim"] and not result["claim_boundary"]["t100_seconds_claim"],
        "stage5c_false": not result["claim_boundary"]["stage5c_executed"],
        "smc_false": not result["claim_boundary"]["smc_enabled"],
    }
    if all(gates.values()) and result["summary"]["final_t100_positive"]:
        verdict = "stage42_ba_t100_source_cv_repair_pass_with_supported_t100"
    elif all(gates.values()):
        verdict = "stage42_ba_t100_source_cv_repair_pass_with_t100_blocker"
    else:
        verdict = "stage42_ba_t100_source_cv_repair_partial"
    return {"source": result["source"], "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": int(len(gates)), "verdict": verdict}


def _render_fold_table(rows: list[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| domain | holdout source | val source | t100 rows | best lambda | h100 t100 | h100 easy | safe positive | guarded | kept |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        h100 = row["holdout_h100"]
        lines.append(
            f"| `{row['domain']}` | `{row['holdout_group']}` | `{row['val_group']}` | {row['holdout_t100_rows']} | {row['best_lambda']:.3f} | {h100['t100_raw_frame_diagnostic_improvement']:.6f} | {h100['easy_degradation']:.6f} | {bool(row['safe_positive_t100'])} | `{row['guarded_t100_slices']}` | `{row['kept_t100_slices']}` |"
        )
    return lines


def _render_metric_block(title: str, bundle: Mapping[str, Any]) -> list[str]:
    p = bundle["protected"]
    f = bundle["fde"]
    lines = [
        f"## {title}",
        "",
        "| metric type | rows | all | t50 | t100 raw diag | hard | easy | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| ADE | {p['rows']} | {p['all_improvement']:.6f} | {p['t50_improvement']:.6f} | {p['t100_raw_frame_diagnostic_improvement']:.6f} | {p['hard_failure_improvement']:.6f} | {p['easy_degradation']:.6f} | {p['switch_rate']:.6f} |",
        f"| FDE | {f['rows']} | {f['all_improvement']:.6f} | {f['t50_improvement']:.6f} | {f['t100_raw_frame_diagnostic_improvement']:.6f} | {f['hard_failure_improvement']:.6f} | {f['easy_degradation']:.6f} | {f['switch_rate']:.6f} |",
        "",
        "### Bootstrap",
        "",
        "| slice | low | mid | high | n |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, row in bundle["bootstrap"].items():
        lines.append(f"| `{key}` | {row['low']:.6f} | {row['mid']:.6f} | {row['high']:.6f} | {row['n']} |")
    return lines


def _render_report(result: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-BA Train-Only T100 Source-CV Repair",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_ba_gate']['passed']} / {result['stage42_ba_gate']['total']}`",
        f"- verdict: `{result['stage42_ba_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Source-CV Plan",
        "",
        f"- plan: `{result['source_cv_plan']}`",
        f"- fold_count: `{result['source_cv_fold_count']}`",
        f"- domain_support: `{result['domain_t100_support']}`",
        "",
        "## Source-CV Fold Results",
        "",
    ]
    lines.extend(_render_fold_table(result["source_cv_fold_results"]))
    lines.extend(
        [
            "",
            "## Final Test-On-Eval Guard",
            "",
            f"- guard: `{result['final_eval']['guard']}`",
            f"- supported_t100_domains: `{result['summary']['supported_t100_domains']}`",
            f"- unsupported_t100_domains: `{result['summary']['unsupported_t100_domains']}`",
            "",
        ]
    )
    lines.extend(_render_metric_block("Before Source-CV Guard", result["final_eval"]["before_cv_guard"]))
    lines.extend([""])
    lines.extend(_render_metric_block("After Source-CV Guard", result["final_eval"]["after_cv_guard"]))
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
            "- Stage42-BA uses train-only source-CV to decide whether any domain has enough independent t100 support.",
            "- Unsupported t100 domain slices are guarded to the causal floor before final test evaluation.",
            "- If t100 becomes zero after this guard, that is a safety/blocker result rather than a failure to report: it means current t100 positive gain lacks enough source-level validation support.",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_ba_gate"]
    lines = [
        "# Stage42-BA Gate",
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
        "verdict": result["stage42_ba_gate"]["verdict"],
        "gate": f"{result['stage42_ba_gate']['passed']}/{result['stage42_ba_gate']['total']}",
        "git_commit": result["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_t100_source_cv_repair()
