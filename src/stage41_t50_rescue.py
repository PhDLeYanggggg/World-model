from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_intervention_calibrator as cal
from src import stage41_all_agent as s41a


OUT_DIR = s41.OUT_DIR
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
EPS = 1e-6


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


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    write_json(path, _jsonable(dict(payload)))


def _append_ledger(step: str, status: str, started: float, inputs: list[str], outputs: list[str]) -> None:
    ensure_dir(OUT_DIR)
    entry = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "step": step,
        "source": "fresh_run",
        "status": status,
        "wall_time_s": time.perf_counter() - started,
        "input_hash": _combined_hash(inputs),
        "output_hash": _combined_hash(outputs),
        "git_commit": _git_commit(),
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _policy_without_t50(policy: Mapping[str, Any]) -> Dict[str, Any]:
    out = deepcopy(dict(policy))
    out["slices"] = {k: v for k, v in dict(policy.get("slices", {})).items() if not k.endswith("|50")}
    return out


def _domain_source_counts(split_report: Mapping[str, Any], split: str = "val") -> Dict[str, int]:
    result: Dict[str, int] = {}
    for domain, item in split_report.get("by_domain", {}).items():
        result[domain] = int(item.get(split, {}).get("source_files", 0))
    return result


def _domain_history_mean(split_report: Mapping[str, Any], split: str = "val") -> Dict[str, float]:
    result: Dict[str, float] = {}
    for domain, item in split_report.get("by_domain", {}).items():
        result[domain] = float(item.get(split, {}).get("history_len_mean", 0.0))
    return result


def _slice_metrics(selected: np.ndarray, fallback: np.ndarray, ds: Mapping[str, np.ndarray], switch: np.ndarray, domain: str, horizon: int) -> Dict[str, float]:
    mask = (ds["domain"].astype(str) == domain) & (ds["horizon"].astype(int) == horizon)
    if not np.any(mask):
        return {"rows": 0, "improvement": 0.0, "switch_rate": 0.0, "easy_degradation": 0.0}
    easy = ds["easy"].astype(bool) & mask
    imp = 1.0 - float(selected[mask].mean()) / max(float(fallback[mask].mean()), EPS)
    easy_deg = 0.0
    if np.any(easy):
        easy_deg = max(0.0, float(selected[easy].mean()) / max(float(fallback[easy].mean()), EPS) - 1.0)
    return {"rows": int(np.sum(mask)), "improvement": imp, "switch_rate": float(np.mean(switch[mask])), "easy_degradation": easy_deg}


def _evaluate_variant(name: str, calib_path: str | Path, policy: Mapping[str, Any], split: str) -> Dict[str, Any]:
    calib_pred, base_pred, payload = cal._calib_predict(calib_path, split)
    ds = s41a._ds(split)
    selected, switch = cal._apply_policy(calib_pred, base_pred, ds, policy)
    metrics = s41a._metrics(selected, ds["floor_fde"].astype(np.float64), ds, switch)
    metrics["variant"] = name
    metrics["base_name"] = payload["base_name"]
    metrics["t50_slices"] = {
        domain: _slice_metrics(selected, ds["floor_fde"].astype(np.float64), ds, switch, domain, 50)
        for domain in sorted(set(ds["domain"].astype(str).tolist()))
    }
    return metrics


def _score_for_rescue(metrics: Mapping[str, Any]) -> float:
    # This score is for validation-only policy selection. It strongly penalizes
    # t50 regression while still preserving all/hard/t100 gains and easy safety.
    return (
        1.8 * float(metrics.get("t50_improvement", 0.0))
        + float(metrics.get("all_improvement", 0.0))
        + float(metrics.get("hard_failure_improvement", 0.0))
        + 0.25 * float(metrics.get("t100_improvement", 0.0))
        - 18.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 0.1 * max(0.0, float(metrics.get("harm_over_fallback", 0.0)))
    )


def build_t50_rescue_variants() -> Dict[str, Any]:
    started = time.perf_counter()
    eval_report = read_json(OUT_DIR / "stage41_intervention_calibrator_eval.json", {})
    if not eval_report:
        cal.eval_intervention_calibrators()
        eval_report = read_json(OUT_DIR / "stage41_intervention_calibrator_eval.json", {})
    trials = eval_report.get("trials", {})
    if not trials:
        result = {"source": "not_run", "reason": "no intervention calibrator trials"}
        _write_json(OUT_DIR / "stage41_t50_rescue.json", result)
        return result
    split_report = read_json("outputs/stage41_external_split/report.json", {})
    source_counts = _domain_source_counts(split_report, "val")
    history_mean = _domain_history_mean(split_report, "val")
    variants: Dict[str, Any] = {}
    for trial_name, item in trials.items():
        base_policy = item.get("policy", {})
        calib_path = item.get("train", {}).get("checkpoint")
        if not calib_path:
            continue
        policies: Dict[str, Dict[str, Any]] = {}
        policies["non_t50_only"] = _policy_without_t50(base_policy)
        stable_multi_source = _policy_without_t50(base_policy)
        stable_short_history = _policy_without_t50(base_policy)
        for key, params in dict(base_policy.get("slices", {})).items():
            if not key.endswith("|50"):
                continue
            domain = key.split("|")[0]
            if source_counts.get(domain, 0) >= 2:
                stable_multi_source["slices"][key] = params
            # TrajNet/UCY style external rows have short histories; Stage41
            # validation showed these domains are less prone to t50 overreach.
            # This guard is based only on split metadata, not test endpoint error.
            if history_mean.get(domain, 999.0) <= 12.0:
                stable_short_history["slices"][key] = params
        policies["t50_multi_source_guard"] = stable_multi_source
        policies["t50_short_history_guard"] = stable_short_history
        trial_variants: Dict[str, Any] = {}
        best_name = ""
        best_score = -1e18
        for variant_name, policy in policies.items():
            val_metrics = _evaluate_variant(variant_name, calib_path, policy, "val")
            test_metrics = _evaluate_variant(variant_name, calib_path, policy, "test")
            score = _score_for_rescue(val_metrics)
            trial_variants[variant_name] = {"policy": policy, "val_metrics": val_metrics, "test_metrics": test_metrics, "val_score": score}
            if val_metrics.get("easy_degradation", 1.0) <= 0.02 and score > best_score:
                best_name, best_score = variant_name, score
        variants[trial_name] = {"source": "fresh_run", "base_trial": trial_name, "base_checkpoint": calib_path, "variants": trial_variants, "selected_by_val": best_name, "selected_val_score": best_score}
    best_trial = None
    best_variant = None
    best_score = -1e18
    for trial_name, row in variants.items():
        selected = row.get("selected_by_val")
        if not selected:
            continue
        v = row["variants"][selected]
        score = _score_for_rescue(v["val_metrics"])
        if score > best_score:
            best_trial, best_variant, best_score = trial_name, selected, score
    if best_trial is None:
        result = {"source": "not_run", "reason": "no val-safe t50 rescue variant", "variants": variants}
    else:
        selected_item = variants[best_trial]["variants"][best_variant]
        # Bootstrap only the val-selected best test policy.
        calib_path = variants[best_trial]["base_checkpoint"]
        calib_pred, base_pred, _payload = cal._calib_predict(calib_path, "test")
        ds = s41a._ds("test")
        selected, switch = cal._apply_policy(calib_pred, base_pred, ds, selected_item["policy"])
        test_metrics = s41a._metrics(selected, ds["floor_fde"].astype(np.float64), ds, switch)
        test_metrics["variant"] = best_variant
        test_metrics["base_trial"] = best_trial
        test_metrics["t50_slices"] = {
            domain: _slice_metrics(selected, ds["floor_fde"].astype(np.float64), ds, switch, domain, 50)
            for domain in sorted(set(ds["domain"].astype(str).tolist()))
        }
        test_metrics["t50_ci"] = s41._bootstrap_ci(selected, ds["floor_fde"].astype(np.float64), ds, "t50", n=2000)
        test_metrics["hard_failure_ci"] = s41._bootstrap_ci(selected, ds["floor_fde"].astype(np.float64), ds, "hard_failure", n=1000)
        positive_domains = sum(
            1
            for row in test_metrics.get("by_domain", {}).values()
            if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0
        )
        exceeds_stage37 = bool(
            test_metrics.get("easy_degradation", 1.0) <= 0.02
            and (
                test_metrics.get("all_improvement", 0.0) >= s41.STAGE37_REFERENCE["all_improvement"] + 0.02
                or test_metrics.get("t50_improvement", 0.0) >= s41.STAGE37_REFERENCE["t50_improvement"] + 0.02
                or test_metrics.get("hard_failure_improvement", 0.0) >= s41.STAGE37_REFERENCE["hard_failure_improvement"] + 0.02
            )
        )
        result = {
            "source": "fresh_run",
            "best_stage41_t50_rescue": f"{best_trial}::{best_variant}",
            "best_trial": best_trial,
            "best_variant": best_variant,
            "selection_rule": "validation score prioritizing t50/all/hard/t100 and easy<=2%; no test threshold tuning",
            "source_counts_val": source_counts,
            "history_mean_val": history_mean,
            "best_metrics": test_metrics,
            "positive_external_domains": positive_domains,
            "neural_exceeds_stage37_by_gate_margin": exceeds_stage37,
            "deployment_decision": "deploy_stage41_t50_rescue" if exceeds_stage37 and positive_domains >= 2 else "keep_stage37_selector",
            "variants": variants,
        }
    _write_json(OUT_DIR / "stage41_t50_rescue.json", result)
    write_md(OUT_DIR / "stage41_t50_rescue.md", ["# Stage41 t50 Rescue", "", "- source: `fresh_run`", f"- result: `{result}`"])
    _append_ledger("stage41_t50_rescue", "ok", started, [str(OUT_DIR / "stage41_intervention_calibrator_eval.json")], [str(OUT_DIR / "stage41_t50_rescue.md")])
    return result


def main_t50_rescue() -> None:
    build_t50_rescue_variants()
