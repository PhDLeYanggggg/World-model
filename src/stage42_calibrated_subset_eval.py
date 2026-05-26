from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_external_validation as s42b
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
BN_JSON = OUT_DIR / "source_time_geometry_calibration_stage42.json"
REPORT_JSON = OUT_DIR / "calibrated_subset_eval_stage42.json"
REPORT_MD = OUT_DIR / "calibrated_subset_eval_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bo_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_calibrated_subset_stage42.md"

SOURCE_TO_REL = {
    "ETH_seq_eth": "ETH/seq_eth/obsmat.txt",
    "ETH_seq_hotel": "ETH/seq_hotel/obsmat.txt",
    "UCY_zara01": "UCY/zara01/obsmat.txt",
    "UCY_zara02": "UCY/zara02/obsmat.txt",
    "UCY_zara03": "UCY/zara03/crowds_zara03.txt",
    "UCY_students03": "UCY/students03/obsmat.txt",
}

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BO 是 calibrated-subset-only source-CV evaluation，不是全局 metric 或 seconds-level claim。",
    "只评估 Stage42-BN 审计出的 source-specific calibration candidates。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _source_rel_array(data: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.asarray([s42b._rel_source(str(path)) for path in data["source_file"].astype(str)], dtype="U256")


def _calibrated_source_ids(calibration: Mapping[str, Any]) -> list[str]:
    ids = []
    for raw in calibration.get("summary", {}).get("source_specific_metric_time_sources", []):
        if raw in SOURCE_TO_REL:
            ids.append(str(raw))
    return sorted(ids)


def _build_source_cv_folds(source_ids: list[str]) -> list[dict[str, Any]]:
    """Build deterministic source-CV folds without using test metrics.

    Each source is held out once. The next source in sorted order is validation,
    all other calibrated sources are training. This is deliberately simple and
    deterministic so the split can be audited and rerun.
    """

    folds: list[dict[str, Any]] = []
    n = len(source_ids)
    for idx, holdout in enumerate(source_ids):
        val = source_ids[(idx + 1) % n]
        train = [s for s in source_ids if s not in {holdout, val}]
        folds.append(
            {
                "fold_id": idx,
                "holdout_source": holdout,
                "validation_source": val,
                "train_sources": train,
                "test_sources": [holdout],
            }
        )
    return folds


def _split_for_fold(rel_source: np.ndarray, fold: Mapping[str, Any]) -> np.ndarray:
    split = np.full(len(rel_source), "ignore", dtype="U8")
    train_rels = {SOURCE_TO_REL[s] for s in fold["train_sources"]}
    val_rels = {SOURCE_TO_REL[fold["validation_source"]]}
    test_rels = {SOURCE_TO_REL[s] for s in fold["test_sources"]}
    for rels, name in [(train_rels, "train"), (val_rels, "val"), (test_rels, "test")]:
        mask = np.isin(rel_source, list(rels))
        split[mask] = name
    return split


def _fold_stats(data: Mapping[str, np.ndarray], rel_source: np.ndarray, split: np.ndarray) -> dict[str, Any]:
    h = data["horizon"].astype(int)
    domain = data["dataset"].astype(str)
    stats: dict[str, Any] = {}
    for sp in ["train", "val", "test"]:
        m = split == sp
        stats[sp] = {
            "rows": int(np.sum(m)),
            "domains": dict(Counter(domain[m].tolist())),
            "sources": dict(Counter(rel_source[m].tolist())),
            "t10": int(np.sum(m & (h == 10))),
            "t25": int(np.sum(m & (h == 25))),
            "t50": int(np.sum(m & (h == 50))),
            "t100": int(np.sum(m & (h == 100))),
            "hard": int(np.sum(data["hard"].astype(bool)[m])),
            "failure": int(np.sum(data["failure"].astype(bool)[m])),
            "easy": int(np.sum(data["easy"].astype(bool)[m])),
        }
    sets = {sp: set(rel_source[split == sp].tolist()) for sp in ["train", "val", "test"]}
    overlap = {
        "train_val": sorted(sets["train"] & sets["val"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "val_test": sorted(sets["val"] & sets["test"]),
    }
    stats["source_overlap"] = overlap
    stats["source_overlap_pass"] = not any(overlap.values())
    return stats


def _evaluate_fold(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], rel_source: np.ndarray, fold: Mapping[str, Any]) -> dict[str, Any]:
    split = _split_for_fold(rel_source, fold)
    train_mask = split == "train"
    if int(np.sum(train_mask)) == 0 or int(np.sum(split == "val")) == 0 or int(np.sum(split == "test")) == 0:
        raise ValueError(f"Invalid empty fold split: {fold}")
    floor = am._floor_arrays(data, train_mask)
    features, feature_names = am._feature_matrix(data, floor)
    x, _, _ = am._standardize(features, train_mask)
    model = am._evaluate_models(data, split, labels, floor, x)
    protected = model["metrics"]["protected_ridge_source_level"]
    ungated = model["metrics"]["ungated_ridge_diagnostic"]
    fde = model["metrics"]["protected_ridge_source_level_fde"]
    return {
        "source": "fresh_calibrated_subset_source_cv_fold",
        "fold": fold,
        "fold_stats": _fold_stats(data, rel_source, split),
        "feature_count": len(feature_names),
        "floor": {
            "type": "train_horizon_selected_safe_causal_baseline",
            "strongest_by_horizon": floor["strongest_by_horizon"],
            "geometry_diagnostics": floor["geometry_diagnostics"],
        },
        "selected_lambda": model["best_lambda"],
        "policy_slice_count": len(model["policy"]["slices"]),
        "protected_ade": protected,
        "protected_fde": fde,
        "ungated_ade_diagnostic": ungated,
        "bootstrap": model["bootstrap"],
        "by_domain": model["by_domain"],
        "by_horizon": model["by_horizon"],
    }


def _aggregate_folds(folds: list[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [fold["protected_ade"] for fold in folds]
    def vals(key: str) -> list[float]:
        return [float(m[key]) for m in metrics]

    easy = vals("easy_degradation")
    t50 = vals("t50_improvement")
    all_imp = vals("all_improvement")
    hard = vals("hard_failure_improvement")
    t100 = vals("t100_raw_frame_diagnostic_improvement")
    rows = [int(m["rows"]) for m in metrics]
    return {
        "folds": len(folds),
        "rows_total": int(sum(rows)),
        "all_improvement_macro_mean": float(np.mean(all_imp)),
        "all_improvement_min": float(np.min(all_imp)),
        "t50_improvement_macro_mean": float(np.mean(t50)),
        "t50_improvement_min": float(np.min(t50)),
        "t100_raw_frame_diagnostic_macro_mean": float(np.mean(t100)),
        "t100_raw_frame_diagnostic_min": float(np.min(t100)),
        "hard_failure_improvement_macro_mean": float(np.mean(hard)),
        "hard_failure_improvement_min": float(np.min(hard)),
        "easy_degradation_macro_mean": float(np.mean(easy)),
        "easy_degradation_max": float(np.max(easy)),
        "positive_all_folds": bool(all(v > 0.0 for v in all_imp)),
        "positive_t50_folds": bool(all(v > 0.0 for v in t50)),
        "easy_safe_all_folds": bool(all(v <= 0.02 for v in easy)),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "bn_input_verified": payload["bn_verdict"] == "stage42_bn_source_time_geometry_calibration_pass_with_global_claim_blocked",
        "calibrated_sources_present": summary["calibrated_sources_evaluated"] >= 6,
        "source_cv_completed": summary["source_cv_folds"] == summary["calibrated_sources_evaluated"],
        "source_overlap_pass": summary["source_overlap_pass"] is True,
        "all_fold_positive": summary["positive_all_folds"] is True,
        "t50_fold_positive": summary["positive_t50_folds"] is True,
        "easy_safe": summary["easy_degradation_max"] <= 0.02,
        "no_future_inputs": no_leak["future_endpoint_input"] is False and no_leak["future_waypoint_input"] is False,
        "validation_only_selection": no_leak["test_threshold_tuning"] is False,
        "global_metric_blocked": claim["global_metric_claim_allowed"] is False,
        "global_seconds_blocked": claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    positive = gates["all_fold_positive"] and gates["t50_fold_positive"] and gates["easy_safe"]
    verdict = (
        "stage42_bo_calibrated_subset_eval_pass_positive_claim_limited"
        if passed == total and positive
        else "stage42_bo_calibrated_subset_eval_pass_diagnostic_claim_limited"
        if passed == total
        else "stage42_bo_calibrated_subset_eval_partial"
    )
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "positive_transfer": bool(positive), "verdict": verdict}


def run_stage42_calibrated_subset_eval() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    calibration = _load_json(BN_JSON)
    data = s41._combined()
    labels = am._reconstruct_waypoint_labels(data)
    rel_source = _source_rel_array(data)
    source_ids = _calibrated_source_ids(calibration)
    present_ids = [sid for sid in source_ids if np.any(rel_source == SOURCE_TO_REL[sid])]
    folds = _build_source_cv_folds(present_ids)
    fold_results = [_evaluate_fold(data, labels, rel_source, fold) for fold in folds]
    aggregate = _aggregate_folds(fold_results)
    source_rows = {
        sid: int(np.sum(rel_source == SOURCE_TO_REL[sid]))
        for sid in present_ids
    }
    no_leakage = {
        "future_endpoint_input": False,
        "future_waypoint_input": False,
        "future_waypoint_label_eval_only": True,
        "central_velocity": False,
        "test_endpoint_goals": False,
        "test_threshold_tuning": False,
        "train_only_feature_normalization": True,
        "source_overlap_pass": all(fold["fold_stats"]["source_overlap_pass"] for fold in fold_results),
    }
    claim_boundary = {
        "true_3d": False,
        "foundation_world_model": False,
        "source_specific_annotation_step_subset_claim_allowed": True,
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "m3w_official_metric_seconds_claim_allowed": False,
        "raw_frame_dataset_local_global_claim_required": True,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    summary = {
        "source": "fresh_calibrated_subset_source_cv",
        "calibrated_sources_evaluated": len(present_ids),
        "calibrated_source_ids": present_ids,
        "source_rows": source_rows,
        "source_cv_folds": len(fold_results),
        "source_overlap_pass": no_leakage["source_overlap_pass"],
        **aggregate,
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "training_run": True,
        "auto_download_executed": False,
    }
    payload: dict[str, Any] = {
        "source": "fresh_calibrated_subset_source_cv",
        "stage": "Stage42-BO calibrated subset source-CV evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(BN_JSON), "data/stage41_world_model/combined_external.npz"]),
        "current_facts": CURRENT_FACTS,
        "bn_verdict": calibration.get("stage42_bn_gate", {}).get("verdict"),
        "calibration_summary": calibration.get("summary", {}),
        "summary": summary,
        "fold_results": fold_results,
        "no_leakage": no_leakage,
        "claim_boundary": claim_boundary,
    }
    payload["stage42_bo_gate"] = _gate(payload)
    write_json(REPORT_JSON, am._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    lines = [
        "# Stage42-BO Calibrated-Subset Source-CV Evaluation",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bo_gate']['passed']} / {payload['stage42_bo_gate']['total']}`",
        f"- verdict: `{payload['stage42_bo_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- calibrated_sources_evaluated: `{summary['calibrated_sources_evaluated']}`",
        f"- source_cv_folds: `{summary['source_cv_folds']}`",
        f"- rows_total: `{summary['rows_total']}`",
        f"- all_improvement_macro_mean: `{summary['all_improvement_macro_mean']}`",
        f"- all_improvement_min: `{summary['all_improvement_min']}`",
        f"- t50_improvement_macro_mean: `{summary['t50_improvement_macro_mean']}`",
        f"- t50_improvement_min: `{summary['t50_improvement_min']}`",
        f"- t100_raw_frame_diagnostic_macro_mean: `{summary['t100_raw_frame_diagnostic_macro_mean']}`",
        f"- hard_failure_improvement_macro_mean: `{summary['hard_failure_improvement_macro_mean']}`",
        f"- easy_degradation_max: `{summary['easy_degradation_max']}`",
        f"- positive_all_folds: `{summary['positive_all_folds']}`",
        f"- positive_t50_folds: `{summary['positive_t50_folds']}`",
        f"- easy_safe_all_folds: `{summary['easy_safe_all_folds']}`",
        "",
        "## Fold Results",
        "",
        "| holdout | val | rows | all | t50 | t100 raw diag | hard/failure | easy degradation | switch |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in payload["fold_results"]:
        m = fold["protected_ade"]
        lines.append(
            f"| `{fold['fold']['holdout_source']}` | `{fold['fold']['validation_source']}` | {m['rows']} | {m['all_improvement']:.6f} | {m['t50_improvement']:.6f} | {m['t100_raw_frame_diagnostic_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['switch_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Source Rows",
            "",
            "| source | rows |",
            "| --- | ---: |",
        ]
    )
    for sid, rows in summary["source_rows"].items():
        lines.append(f"| `{sid}` | {rows} |")
    lines.extend(
        [
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{payload['no_leakage']}`",
            f"- claim_boundary: `{payload['claim_boundary']}`",
            "",
            "## Interpretation",
            "",
            "- This is a source-specific calibrated-subset evaluation candidate, not a global M3W metric/seconds-level claim.",
            "- Fold splits are rebuilt at source level from Stage42-BN calibrated candidates; each source is held out once.",
            "- Future waypoints/endpoints remain labels only, and threshold selection is validation-only.",
            "- If a future paper uses this result, wording must restrict it to source-specific annotation-step calibrated subsets.",
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    return [
        "# Stage42-BO User Action Required: Calibrated Subset",
        "",
        f"- source: `{payload['source']}`",
        "",
        "## Before Stronger Metric / Seconds Claims",
        "",
        "- Manually verify homography direction and coordinate convention for each calibrated source in the fold table.",
        "- Confirm whether each horizon is an annotation-step horizon at 2.5fps / 0.4s, not a raw video-frame horizon.",
        "- Keep global M3W reports as raw-frame / dataset-local unless evaluation is explicitly restricted to these calibrated sources.",
        "- Do not use this as a true-3D, foundation, Stage5C, or SMC claim.",
    ]


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bo_gate"]
    lines = [
        "# Stage42-BO Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- positive_transfer: `{gate['positive_transfer']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    return lines


if __name__ == "__main__":
    run_stage42_calibrated_subset_eval()
