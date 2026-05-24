from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_all_agent as s41a
from src import stage41_breakthrough as s41
from src import stage41_intervention_calibrator as cal
from src import stage41_t50_rescue as rescue


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


def _slice_metrics(selected: np.ndarray, fallback: np.ndarray, ds: Mapping[str, np.ndarray], switch: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if not np.any(mask):
        return {"rows": 0, "improvement": 0.0, "hard_failure_improvement": 0.0, "easy_degradation": 0.0, "switch_rate": 0.0}
    hard = (ds["hard"].astype(bool) | ds["failure"].astype(bool)) & mask
    easy = ds["easy"].astype(bool) & mask
    improvement = 1.0 - float(selected[mask].mean()) / max(float(fallback[mask].mean()), EPS)
    hard_imp = 0.0
    if np.any(hard):
        hard_imp = 1.0 - float(selected[hard].mean()) / max(float(fallback[hard].mean()), EPS)
    easy_deg = 0.0
    if np.any(easy):
        easy_deg = max(0.0, float(selected[easy].mean()) / max(float(fallback[easy].mean()), EPS) - 1.0)
    return {
        "rows": int(np.sum(mask)),
        "improvement": float(improvement),
        "hard_failure_improvement": float(hard_imp),
        "easy_degradation": float(easy_deg),
        "switch_rate": float(np.mean(switch[mask])),
    }


def _score_for_mode(metrics: Mapping[str, float], horizon: int, mode: str) -> float:
    h_bonus = 0.0
    if mode in {"long_horizon", "stage37_gap_hunting"} and horizon in {50, 100}:
        h_bonus = 0.6 * float(metrics.get("improvement", 0.0))
    hard_weight = 0.5 if mode in {"hard_aware", "stage37_gap_hunting"} else 0.2
    switch_bonus = 0.02 * float(metrics.get("switch_rate", 0.0))
    return (
        float(metrics.get("improvement", 0.0))
        + hard_weight * float(metrics.get("hard_failure_improvement", 0.0))
        + h_bonus
        + switch_bonus
        - 15.0 * max(0.0, float(metrics.get("easy_degradation", 0.0)) - 0.02)
    )


def _metadata_by_domain(split: str = "val") -> Dict[str, Dict[str, float]]:
    split_report = read_json("outputs/stage41_external_split/report.json", {})
    result: Dict[str, Dict[str, float]] = {}
    for domain, item in split_report.get("by_domain", {}).items():
        row = item.get(split, {})
        result[domain] = {
            "source_files": float(row.get("source_files", 0)),
            "history_len_mean": float(row.get("history_len_mean", 0.0)),
            "history_ge_32": float(row.get("history_ge_32", 0)),
            "rows": float(row.get("rows", 0)),
        }
    return result


def _overall_score(metrics: Mapping[str, Any], risk_penalty: float = 0.0) -> float:
    positive_domains = sum(
        1
        for row in metrics.get("by_domain", {}).values()
        if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0
    )
    return (
        float(metrics.get("all_improvement", 0.0))
        + 1.5 * float(metrics.get("t50_improvement", 0.0))
        + float(metrics.get("hard_failure_improvement", 0.0))
        + 0.5 * float(metrics.get("t100_improvement", 0.0))
        + 0.03 * positive_domains
        - 20.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 0.2 * max(0.0, float(metrics.get("harm_over_fallback", 0.0)))
        - risk_penalty
    )


def _action_library() -> Dict[str, Dict[str, Any]]:
    eval_report = read_json(OUT_DIR / "stage41_intervention_calibrator_eval.json", {})
    if not eval_report:
        cal.eval_intervention_calibrators()
        eval_report = read_json(OUT_DIR / "stage41_intervention_calibrator_eval.json", {})
    t50_report = read_json(OUT_DIR / "stage41_t50_rescue.json", {})
    if not t50_report:
        rescue.build_t50_rescue_variants()
        t50_report = read_json(OUT_DIR / "stage41_t50_rescue.json", {})

    actions: Dict[str, Dict[str, Any]] = {"fallback": {"type": "fallback"}}
    for trial_name, item in eval_report.get("trials", {}).items():
        checkpoint = item.get("train", {}).get("checkpoint")
        policy = item.get("policy", {})
        if not checkpoint or not policy:
            continue
        actions[f"{trial_name}::original"] = {"type": "calibrator", "checkpoint": checkpoint, "policy": policy}
        actions[f"{trial_name}::non_t50_only"] = {"type": "calibrator", "checkpoint": checkpoint, "policy": rescue._policy_without_t50(policy)}
    for trial_name, item in t50_report.get("variants", {}).items():
        checkpoint = item.get("base_checkpoint")
        if not checkpoint:
            continue
        for variant_name, row in item.get("variants", {}).items():
            policy = row.get("policy", {})
            if policy:
                actions[f"{trial_name}::{variant_name}"] = {"type": "calibrator", "checkpoint": checkpoint, "policy": policy}
    return actions


def _arrays_for_action(action: Mapping[str, Any], split: str, cache: Dict[Tuple[str, str], Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]) -> Tuple[np.ndarray, np.ndarray]:
    ds = s41a._ds(split)
    fallback = ds["floor_fde"].astype(np.float64)
    if action.get("type") == "fallback":
        return fallback.copy(), np.zeros(len(fallback), dtype=bool)
    checkpoint = str(action["checkpoint"])
    key = (checkpoint, split)
    if key not in cache:
        calib_pred, base_pred, _payload = cal._calib_predict(checkpoint, split)
        cache[key] = (calib_pred, base_pred)
    calib_pred, base_pred = cache[key]
    return cal._apply_policy(calib_pred, base_pred, ds, action["policy"])


def _blend(actions: Mapping[str, Mapping[str, Any]], selected_slices: Mapping[str, str], split: str, cache: Dict[Tuple[str, str], Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]) -> Tuple[np.ndarray, np.ndarray]:
    ds = s41a._ds(split)
    fallback = ds["floor_fde"].astype(np.float64)
    selected = fallback.copy()
    switch = np.zeros(len(fallback), dtype=bool)
    domains = ds["domain"].astype(str)
    horizons = ds["horizon"].astype(int)
    action_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for slice_key, action_name in selected_slices.items():
        if action_name == "fallback":
            continue
        domain, horizon_s = slice_key.split("|")
        mask = (domains == domain) & (horizons == int(horizon_s))
        if not np.any(mask):
            continue
        if action_name not in action_arrays:
            action_arrays[action_name] = _arrays_for_action(actions[action_name], split, cache)
        action_selected, action_switch = action_arrays[action_name]
        selected[mask] = action_selected[mask]
        switch[mask] = action_switch[mask]
    return selected, switch


def _select_slices(actions: Mapping[str, Mapping[str, Any]], mode: str) -> Tuple[Dict[str, str], Dict[str, Any], float]:
    ds = s41a._ds("val")
    fallback = ds["floor_fde"].astype(np.float64)
    domains = ds["domain"].astype(str)
    horizons = ds["horizon"].astype(int)
    meta = _metadata_by_domain("val")
    cache: Dict[Tuple[str, str], Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = {}
    action_arrays = {name: _arrays_for_action(action, "val", cache) for name, action in actions.items()}
    train_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    train_ds: Mapping[str, np.ndarray] | None = None
    train_fallback: np.ndarray | None = None
    train_domains: np.ndarray | None = None
    train_horizons: np.ndarray | None = None
    if mode in {"train_val_consistency", "robust_short_history"}:
        train_ds = s41a._ds("train")
        train_fallback = train_ds["floor_fde"].astype(np.float64)
        train_domains = train_ds["domain"].astype(str)
        train_horizons = train_ds["horizon"].astype(int)
        train_cache: Dict[Tuple[str, str], Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = {}
        train_arrays = {name: _arrays_for_action(action, "train", train_cache) for name, action in actions.items()}
    selected_slices: Dict[str, str] = {}
    diagnostics: Dict[str, Any] = {}
    risk_penalty = 0.0
    for domain in sorted(set(domains.tolist())):
        for horizon in [10, 25, 50, 100]:
            mask = (domains == domain) & (horizons == horizon)
            if int(np.sum(mask)) < 100:
                continue
            best_name = "fallback"
            best_score = 0.0
            best_metrics = _slice_metrics(fallback, fallback, ds, np.zeros(len(fallback), dtype=bool), mask)
            for action_name, (action_selected, action_switch) in action_arrays.items():
                metrics = _slice_metrics(action_selected, fallback, ds, action_switch, mask)
                min_imp = 0.002 if mode in {"conservative", "long_horizon", "metadata_guarded", "robust_short_history"} else 0.0
                max_easy = 0.004 if mode in {"conservative", "metadata_guarded", "robust_short_history"} else 0.02
                if metrics["improvement"] <= min_imp or metrics["easy_degradation"] > max_easy:
                    continue
                risky_t50 = horizon == 50 and meta.get(domain, {}).get("history_len_mean", 0.0) > 12.0
                if mode in {"metadata_guarded", "robust_short_history"} and risky_t50:
                    continue
                if mode in {"train_val_consistency", "robust_short_history"} and train_ds is not None and train_domains is not None and train_horizons is not None and train_fallback is not None:
                    train_mask = (train_domains == domain) & (train_horizons == horizon)
                    if int(np.sum(train_mask)) < 100:
                        continue
                    train_selected, train_switch = train_arrays[action_name]
                    train_metrics = _slice_metrics(train_selected, train_fallback, train_ds, train_switch, train_mask)
                    if train_metrics["improvement"] <= 0.002 or train_metrics["easy_degradation"] > 0.01:
                        continue
                score = _score_for_mode(metrics, horizon, mode)
                if score > best_score:
                    best_name = action_name
                    best_score = score
                    best_metrics = metrics
            selected_slices[f"{domain}|{horizon}"] = best_name
            if horizon == 50 and best_name != "fallback" and meta.get(domain, {}).get("history_len_mean", 0.0) > 12.0:
                risk_penalty += 0.30
            diagnostics[f"{domain}|{horizon}"] = {"selected_action": best_name, "val_score": best_score, "val_metrics": best_metrics}
    return selected_slices, diagnostics, risk_penalty


def build_policy_blender() -> Dict[str, Any]:
    started = time.perf_counter()
    actions = _action_library()
    modes = ["conservative", "balanced", "long_horizon", "hard_aware", "stage37_gap_hunting", "metadata_guarded", "train_val_consistency", "robust_short_history"]
    mode_reports: Dict[str, Any] = {}
    best_mode = ""
    best_val_score = -1e18
    best_slices: Dict[str, str] = {}
    for mode in modes:
        selected_slices, diagnostics, risk_penalty = _select_slices(actions, mode)
        cache: Dict[Tuple[str, str], Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = {}
        val_selected, val_switch = _blend(actions, selected_slices, "val", cache)
        val_ds = s41a._ds("val")
        val_metrics = s41a._metrics(val_selected, val_ds["floor_fde"].astype(np.float64), val_ds, val_switch)
        score = _overall_score(val_metrics, risk_penalty=risk_penalty)
        mode_reports[mode] = {
            "source": "fresh_run",
            "selected_slices": selected_slices,
            "slice_diagnostics": diagnostics,
            "val_metrics": val_metrics,
            "val_score": score,
            "metadata_risk_penalty": risk_penalty,
        }
        if val_metrics.get("easy_degradation", 1.0) <= 0.02 and score > best_val_score:
            best_mode = mode
            best_val_score = score
            best_slices = selected_slices
    if not best_mode:
        result = {"source": "not_run", "reason": "no val-safe policy blender mode", "modes": mode_reports}
    else:
        cache = {}
        test_selected, test_switch = _blend(actions, best_slices, "test", cache)
        test_ds = s41a._ds("test")
        test_metrics = s41a._metrics(test_selected, test_ds["floor_fde"].astype(np.float64), test_ds, test_switch)
        test_metrics["t50_ci"] = s41._bootstrap_ci(test_selected, test_ds["floor_fde"].astype(np.float64), test_ds, "t50", n=2000)
        test_metrics["hard_failure_ci"] = s41._bootstrap_ci(test_selected, test_ds["floor_fde"].astype(np.float64), test_ds, "hard_failure", n=1000)
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
            "best_stage41_policy_blender": best_mode,
            "selection_rule": "domain+horizon slice policy selected on validation only; test evaluated once",
            "guard_note": "metadata_guarded modes forbid t50 switching for validation domains with mean history > 12 frames; train_val modes require positive train and val slice metrics.",
            "action_count": len(actions),
            "best_selected_slices": best_slices,
            "best_val_score": best_val_score,
            "best_metrics": test_metrics,
            "positive_external_domains": positive_domains,
            "neural_exceeds_stage37_by_gate_margin": exceeds_stage37,
            "deployment_decision": "deploy_stage41_policy_blender" if exceeds_stage37 and positive_domains >= 2 else "keep_stage37_selector",
            "modes": mode_reports,
        }
    _write_json(OUT_DIR / "stage41_policy_blender.json", result)
    write_md(
        OUT_DIR / "stage41_policy_blender.md",
        [
            "# Stage41 Policy Blender",
            "",
            "- source: `fresh_run`",
            "- selection: validation-only domain+horizon neural policy mixture",
            f"- result: `{result}`",
        ],
    )
    _append_ledger("stage41_policy_blender", "ok", started, [str(OUT_DIR / "stage41_intervention_calibrator_eval.json"), str(OUT_DIR / "stage41_t50_rescue.json")], [str(OUT_DIR / "stage41_policy_blender.md")])
    return result


def main_policy_blender() -> None:
    build_policy_blender()
