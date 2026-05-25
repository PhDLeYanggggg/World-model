from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src import stage41_joint_policy_distillation as jpd
from src import stage41_joint_distiller_evidence as jde


OUT_DIR = jpd.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_ucy_fallback_repair.json"
REPORT_MD = OUT_DIR / "stage41_ucy_fallback_repair.md"
CALIBRATION_MODULUS = 5


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _domain_counts(data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    domain = data["domain"].astype(str)
    horizon = data["horizon"].astype(int)
    out: dict[str, Any] = {}
    for name in sorted(set(domain.tolist())):
        mask = domain == name
        out[name] = {
            "rows": int(mask.sum()),
            "horizons": {str(h): int((mask & (horizon == h)).sum()) for h in [10, 25, 50, 100]},
        }
    return out


def _calibration_mask(data: Mapping[str, np.ndarray], domain_name: str = "UCY") -> np.ndarray:
    domain = data["domain"].astype(str)
    ids = np.arange(len(domain), dtype=np.int64)
    return (domain == domain_name) & ((ids % CALIBRATION_MODULUS) == 0)


def _slice_arrays(data: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[mask] for k, v in data.items() if isinstance(v, np.ndarray) and len(v) == len(mask)}


def _gain_forensics(scores: Mapping[str, np.ndarray], data: Mapping[str, np.ndarray], domain_name: str) -> dict[str, Any]:
    domain = data["domain"].astype(str)
    horizon = data["horizon"].astype(int)
    mask = domain == domain_name
    out: dict[str, Any] = {
        "rows": int(mask.sum()),
        "mean_gain": float(np.mean(data["gain"][mask])) if np.any(mask) else 0.0,
        "positive_gain_rate": float(np.mean(data["gain"][mask] > 0)) if np.any(mask) else 0.0,
        "score_quantiles": {},
        "by_horizon": {},
    }
    for key in ["switch_prob", "gain_pred", "harm_prob"]:
        vals = scores[key][mask]
        out["score_quantiles"][key] = [float(v) for v in np.quantile(vals, [0.1, 0.5, 0.9])] if len(vals) else []
    for h in [10, 25, 50, 100]:
        hm = mask & (horizon == h)
        out["by_horizon"][str(h)] = {
            "rows": int(hm.sum()),
            "mean_gain": float(np.mean(data["gain"][hm])) if np.any(hm) else 0.0,
            "positive_gain_rate": float(np.mean(data["gain"][hm] > 0)) if np.any(hm) else 0.0,
            "floor_ade": float(np.mean(data["floor_ade"][hm])) if np.any(hm) else 0.0,
            "neural_ade": float(np.mean(data["neural_ade"][hm])) if np.any(hm) else 0.0,
        }
    return out


def run_ucy_fallback_repair() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    base_result = read_json(OUT_DIR / "stage41_joint_policy_distillation.json", {})
    if not base_result:
        raise FileNotFoundError("Run stage41_joint_policy_distillation before UCY repair.")
    if base_result.get("no_leakage", {}).get("base_switch_input", True):
        raise RuntimeError("Refusing to repair a base-switch-leaking policy.")
    checkpoint = base_result["best_checkpoint"]
    base_policy = base_result["best_policy"]

    split_counts: dict[str, Any] = {}
    split_domains: dict[str, set[str]] = {}
    for split in ["train", "val", "test"]:
        data = jpd._feature_bundle(split)
        split_counts[split] = _domain_counts(data)
        split_domains[split] = set(split_counts[split].keys())
    missing_val_domains = sorted((split_domains["train"] | split_domains["test"]) - split_domains["val"])

    scores_train, data_train = jpd._predict_checkpoint(checkpoint, "train")
    scores_test, data_test = jpd._predict_checkpoint(checkpoint, "test")
    base_selected, _base_fde, base_switch = jpd._apply_policy(scores_test, data_test, base_policy)
    base_metrics = jpd._metric(base_selected, data_test["floor_ade"], data_test, base_switch)

    cal_mask = _calibration_mask(data_train, "UCY")
    cal_data = _slice_arrays(data_train, cal_mask)
    cal_scores = _slice_arrays(scores_train, cal_mask)
    ucy_policy, calibration_metrics = jpd._fit_policy(cal_scores, cal_data, "distiller_only")
    repaired_policy = json.loads(json.dumps(base_policy))
    repaired_policy.setdefault("slices", {}).update(ucy_policy.get("slices", {}))

    repaired_selected, _repaired_fde, repaired_switch = jpd._apply_policy(scores_test, data_test, repaired_policy)
    repaired_metrics = jpd._metric(repaired_selected, data_test["floor_ade"], data_test, repaired_switch)
    bootstrap = jde._bootstrap_report(repaired_selected, data_test["floor_ade"], data_test)
    lift_over_base = {
        "all_delta": float(repaired_metrics.get("all_improvement", 0.0) - base_metrics.get("all_improvement", 0.0)),
        "t50_delta": float(repaired_metrics.get("t50_improvement", 0.0) - base_metrics.get("t50_improvement", 0.0)),
        "t100_delta": float(repaired_metrics.get("t100_improvement", 0.0) - base_metrics.get("t100_improvement", 0.0)),
        "hard_delta": float(repaired_metrics.get("hard_failure_improvement", 0.0) - base_metrics.get("hard_failure_improvement", 0.0)),
        "easy_delta": float(repaired_metrics.get("easy_degradation", 0.0) - base_metrics.get("easy_degradation", 0.0)),
    }
    ucy_metrics = (repaired_metrics.get("by_domain") or {}).get("UCY", {})
    contributes = bool(
        ucy_metrics.get("all_improvement", 0.0) > 0
        and ucy_metrics.get("t50_improvement", 0.0) > 0
        and repaired_metrics.get("all_improvement", 0.0) > base_metrics.get("all_improvement", 0.0)
        and repaired_metrics.get("easy_degradation", 1.0) <= 0.02
    )
    result = {
        "source": "fresh_run",
        "protocol": "ucy_train_calibrated_repair",
        "split_counts": split_counts,
        "missing_val_domains": missing_val_domains,
        "forensics": {
            "ucy_test_gain": _gain_forensics(scores_test, data_test, "UCY"),
            "ucy_val_absent": "UCY" in missing_val_domains,
            "reason_current_policy_falls_back": "UCY has train/test rows but no validation rows, so the val-selected slice policy never created UCY thresholds.",
        },
        "calibration": {
            "domain": "UCY",
            "source_split": "train",
            "calibration_rule": f"train row index modulo {CALIBRATION_MODULUS} == 0",
            "rows": int(cal_mask.sum()),
            "metrics": calibration_metrics,
            "policy_slices": ucy_policy.get("slices", {}),
        },
        "base_metrics": base_metrics,
        "repaired_metrics": repaired_metrics,
        "bootstrap": bootstrap,
        "lift_over_base_policy": lift_over_base,
        "repaired_policy": repaired_policy,
        "ucy_repair_contributes": contributes,
        "no_leakage": {
            "future_waypoints_input": False,
            "future_labels_eval_only": True,
            "train_only_ucy_threshold_calibration": True,
            "test_threshold_tuning": False,
            "base_switch_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "caveat": "UCY repair is calibrated from train-only UCY rows because the external validation split has no UCY rows. It should be independently replicated with a true UCY validation split before final deployment.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    write_md(
        REPORT_MD,
        [
            "# Stage41 UCY Fallback Repair",
            "",
            "- source: `fresh_run`",
            f"- contributes: `{contributes}`",
            f"- missing validation domains: `{missing_val_domains}`",
            f"- calibration rows: `{int(cal_mask.sum())}`",
            f"- base metrics: `{base_metrics}`",
            f"- repaired metrics: `{repaired_metrics}`",
            f"- lift over base policy: `{lift_over_base}`",
            f"- UCY test forensics: `{result['forensics']['ucy_test_gain']}`",
            f"- bootstrap: `{bootstrap}`",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "This repairs the UCY fallback-only behavior using train-only UCY threshold calibration. It is not metric, not seconds-level, not true 3D, and not a foundation-model claim.",
        ],
    )
    return result


def main_ucy_fallback_repair() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_ucy_fallback_repair()
        status = "ok"
    finally:
        jpd._append_ledger(
            "stage41_ucy_fallback_repair",
            status,
            started,
            [OUT_DIR / "stage41_joint_policy_distillation.json"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_ucy_fallback_repair()
