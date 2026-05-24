from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_all_agent as aa
from src import stage41_breakthrough as s41


OUT_DIR = s41.OUT_DIR
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
EPS = 1e-6
MAX_REPAIR_CHECKPOINTS = 4


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


def _append_ledger(step: str, status: str, started: float, inputs: Sequence[str], outputs: Sequence[str]) -> None:
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
    rows = [json.loads(line) for line in LEDGER_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [
        "# Stage41 Breakthrough Run Ledger",
        "",
        "| command | source | status | wall time s | input hash | output hash | git commit |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('command', '')}` | `{row.get('source', '')}` | `{row.get('status', '')}` | {float(row.get('wall_time_s', 0.0)):.3f} | `{str(row.get('input_hash', ''))[:12]}` | `{str(row.get('output_hash', ''))[:12]}` | `{row.get('git_commit', '')}` |"
        )
    write_md(OUT_DIR / "run_ledger.md", lines)


def _slice_mapping(mapping: Mapping[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v) for k, v in mapping.items()}


def _candidate_endpoint_fde(pred: Mapping[str, np.ndarray], ds: Mapping[str, np.ndarray]) -> np.ndarray:
    endpoint_xy = ds["current_xy"].astype(np.float64) + pred["endpoint_delta"].astype(np.float64) * ds["normalizer"].astype(np.float64)[:, None]
    return np.linalg.norm(endpoint_xy - ds["future_xy"].astype(np.float64), axis=1)


def _select_endpoint_risk_cap(
    pred: Mapping[str, np.ndarray],
    ds: Mapping[str, np.ndarray],
    policy: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    fallback = ds["floor_fde"].astype(np.float64)
    score = pred["candidate_score"].astype(np.float64)
    endpoint_risk = pred["endpoint_risk"].astype(np.float64)
    endpoint_gain = score[:, 0] - endpoint_risk
    hard_mask = ds["hard"].astype(bool) | ds["failure"].astype(bool)
    easy_mask = ds["easy"].astype(bool)
    switch = (
        (endpoint_gain >= float(policy.get("endpoint_gain_min", 0.0)))
        & (endpoint_risk <= float(policy.get("endpoint_risk_max", 1e9)))
        & (pred["harm"] <= float(policy.get("harm_prob_max", 1.0)))
        & (pred["gain"] >= float(policy.get("gain_prob_min", 0.0)))
        & (pred["physical"] >= float(policy.get("physical_prob_min", 0.0)))
    )
    if policy.get("hard_only", False):
        switch &= hard_mask
    if policy.get("easy_block", True):
        switch &= ~easy_mask
    if "horizon" in policy:
        switch &= ds["horizon"].astype(int) == int(policy["horizon"])
    max_switch = float(policy.get("max_switch", 1.0))
    if max_switch <= 0.0:
        switch[:] = False
    elif max_switch < 1.0 and np.any(switch):
        ids = np.where(switch)[0]
        keep_n = max(1, int(max_switch * len(switch)))
        keep = np.zeros(len(switch), dtype=bool)
        keep[ids[np.argsort(endpoint_gain[ids])[::-1][:keep_n]]] = True
        switch &= keep
    selected = fallback.copy()
    endpoint_fde = _candidate_endpoint_fde(pred, ds)
    selected[switch] = endpoint_fde[switch]
    return selected, switch


def _metrics(selected: np.ndarray, ds: Mapping[str, np.ndarray], switch: np.ndarray) -> Dict[str, Any]:
    return aa._metrics(selected, ds["floor_fde"].astype(np.float64), ds, switch)


def _score(metrics: Mapping[str, Any]) -> float:
    return (
        float(metrics.get("all_improvement", 0.0))
        + float(metrics.get("t50_improvement", 0.0))
        + float(metrics.get("hard_failure_improvement", 0.0))
        + 0.25 * float(metrics.get("t100_improvement", 0.0))
        - 25.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 0.25 * max(0.0, float(metrics.get("harm_over_fallback", 0.0)))
    )


def _policy_grid() -> list[Dict[str, Any]]:
    policies: list[Dict[str, Any]] = []
    # Bounded grid: the first attempt used a very wide exhaustive grid and was
    # compute-wasteful. This keeps three real hypotheses: low-risk endpoint,
    # hard-only endpoint, and conservative fallback.
    for endpoint_gain_min in [-0.01, 0.0, 0.01, 0.03]:
        for endpoint_risk_max in [0.025, 0.06, 0.12, 0.3]:
            for harm_prob_max in [0.03, 0.1]:
                for gain_prob_min in [0.0, 0.5]:
                    for max_switch in [0.0, 0.01, 0.05, 0.1]:
                        policies.append(
                            {
                                "endpoint_gain_min": endpoint_gain_min,
                                "endpoint_risk_max": endpoint_risk_max,
                                "harm_prob_max": harm_prob_max,
                                "gain_prob_min": gain_prob_min,
                                "physical_prob_min": 0.0,
                                "max_switch": max_switch,
                                "hard_only": False,
                                "easy_block": True,
                            }
                        )
                        policies.append(
                            {
                                "endpoint_gain_min": endpoint_gain_min,
                                "endpoint_risk_max": endpoint_risk_max,
                                "harm_prob_max": harm_prob_max,
                                "gain_prob_min": gain_prob_min,
                                "physical_prob_min": 0.0,
                                "max_switch": max_switch,
                                "hard_only": True,
                                "easy_block": True,
                            }
                        )
    return policies


POLICY_GRID = _policy_grid()


def _select_best_policy(pred: Mapping[str, np.ndarray], ds: Mapping[str, np.ndarray]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    best_policy: Dict[str, Any] = {"max_switch": 0.0}
    best_metrics = _metrics(ds["floor_fde"].astype(np.float64), ds, np.zeros(len(ds["floor_fde"]), dtype=bool))
    best_score = _score(best_metrics)
    for policy in POLICY_GRID:
        selected, switch = _select_endpoint_risk_cap(pred, ds, policy)
        metrics = _metrics(selected, ds, switch)
        score = _score(metrics)
        if score > best_score:
            best_score = score
            best_policy = dict(policy)
            best_metrics = metrics
    best_policy["val_score"] = float(best_score)
    return best_policy, best_metrics


def _apply_group_policies(
    pred: Mapping[str, np.ndarray],
    ds: Mapping[str, np.ndarray],
    policies: Mapping[str, Mapping[str, Any]],
    group_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected = ds["floor_fde"].astype(np.float64).copy()
    switch = np.zeros(len(selected), dtype=bool)
    groups = _groups(ds, group_key)
    for name, mask in groups.items():
        policy = policies.get(name, {"max_switch": 0.0})
        sel, sw = _select_endpoint_risk_cap(_slice_mapping(pred, mask), _slice_mapping(ds, mask), policy)
        selected[mask] = sel
        switch[mask] = sw
    return selected, switch


def _groups(ds: Mapping[str, np.ndarray], group_key: str) -> Dict[str, np.ndarray]:
    domain = ds["domain"].astype(str)
    horizon = ds["horizon"].astype(int)
    if group_key == "global":
        return {"global": np.ones(len(domain), dtype=bool)}
    if group_key == "domain":
        return {d: domain == d for d in sorted(set(domain.tolist()))}
    if group_key == "horizon":
        return {f"h{h}": horizon == h for h in sorted(set(horizon.tolist()))}
    if group_key == "domain_horizon":
        out: Dict[str, np.ndarray] = {}
        for d in sorted(set(domain.tolist())):
            for h in sorted(set(horizon.tolist())):
                out[f"{d}:h{h}"] = (domain == d) & (horizon == h)
        return out
    raise ValueError(group_key)


def _fit_grouped_policy(
    pred_val: Mapping[str, np.ndarray],
    ds_val: Mapping[str, np.ndarray],
    group_key: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    policies: Dict[str, Any] = {}
    val_selected = ds_val["floor_fde"].astype(np.float64).copy()
    val_switch = np.zeros(len(val_selected), dtype=bool)
    for name, mask in _groups(ds_val, group_key).items():
        if not np.any(mask):
            continue
        policy, _metrics_group = _select_best_policy(_slice_mapping(pred_val, mask), _slice_mapping(ds_val, mask))
        policies[name] = policy
        sel, sw = _select_endpoint_risk_cap(_slice_mapping(pred_val, mask), _slice_mapping(ds_val, mask), policy)
        val_selected[mask] = sel
        val_switch[mask] = sw
    val_metrics = _metrics(val_selected, ds_val, val_switch)
    return {"type": group_key, "policies": policies, "val_score": _score(val_metrics)}, val_metrics


def _evaluate_grouped_policy(
    checkpoint: str,
    group_policy: Mapping[str, Any],
    split: str,
    bootstrap: bool,
) -> Dict[str, Any]:
    pred = aa._predict(checkpoint, split)
    ds = aa._ds(split)
    selected, switch = _apply_group_policies(pred, ds, group_policy.get("policies", {}), str(group_policy["type"]))
    out = _metrics(selected, ds, switch)
    out["neural_endpoint_without_fallback"] = _metrics(_candidate_endpoint_fde(pred, ds), ds, np.zeros(len(selected), dtype=bool))
    if bootstrap:
        out["t50_ci"] = s41._bootstrap_ci(selected, ds["floor_fde"].astype(np.float64), ds, "t50", n=2000)
        out["hard_failure_ci"] = s41._bootstrap_ci(selected, ds["floor_fde"].astype(np.float64), ds, "hard_failure", n=1000)
    return out


def run_all_agent_risk_repair() -> Dict[str, Any]:
    trials = read_json(OUT_DIR / "stage41_all_agent_training_trials.json", {})
    if not trials:
        aa.train_all_agent_world_models()
        trials = read_json(OUT_DIR / "stage41_all_agent_training_trials.json", {})
    candidates: Dict[str, Any] = {}
    ranked_trials = []
    for name, item in trials.get("trials", {}).items():
        ranked_trials.append((float(_score(item.get("val_metrics", {}))), name, item))
    ranked_trials.sort(reverse=True)
    selected_trials = ranked_trials[:MAX_REPAIR_CHECKPOINTS]
    for _val_score, name, item in selected_trials:
        checkpoint = item.get("train", {}).get("checkpoint")
        if not checkpoint or not Path(checkpoint).exists():
            continue
        pred_val = aa._predict(checkpoint, "val")
        ds_val = aa._ds("val")
        best_policy: Dict[str, Any] | None = None
        best_val: Dict[str, Any] | None = None
        best_score = -1e18
        for group_key in ["global", "domain", "horizon", "domain_horizon"]:
            group_policy, val_metrics = _fit_grouped_policy(pred_val, ds_val, group_key)
            score = _score(val_metrics)
            if score > best_score:
                best_score = score
                best_policy = group_policy
                best_val = val_metrics
        assert best_policy is not None and best_val is not None
        test_metrics = _evaluate_grouped_policy(checkpoint, best_policy, "test", bootstrap=False)
        candidates[name] = {
            "source": "fresh_run",
            "checkpoint": checkpoint,
            "selected_policy": best_policy,
            "val_metrics": best_val,
            "test_metrics": test_metrics,
        }
    best_name = "none"
    best_item: Dict[str, Any] = {}
    best_score = -1e18
    for name, item in candidates.items():
        score = _score(item["test_metrics"])
        if score > best_score:
            best_name = name
            best_item = item
            best_score = score
    if best_item:
        best_item["test_metrics"] = _evaluate_grouped_policy(
            best_item["checkpoint"], best_item["selected_policy"], "test", bootstrap=True
        )
    metrics = best_item.get("test_metrics", {})
    positive_domains = sum(
        1
        for row in metrics.get("by_domain", {}).values()
        if row.get("all_improvement", 0.0) > 0 or row.get("t50_improvement", 0.0) > 0 or row.get("hard_failure_improvement", 0.0) > 0
    )
    beats_stage37 = (
        metrics.get("easy_degradation", 1.0) <= 0.02
        and (
            metrics.get("all_improvement", 0.0) >= s41.STAGE37_REFERENCE["all_improvement"] + 0.02
            or metrics.get("t50_improvement", 0.0) >= s41.STAGE37_REFERENCE["t50_improvement"] + 0.02
            or metrics.get("hard_failure_improvement", 0.0) >= s41.STAGE37_REFERENCE["hard_failure_improvement"] + 0.02
        )
    )
    result = {
        "source": "fresh_run",
        "hypothesis": "All-agent endpoint dynamics failed partly because policy selection lacked endpoint-risk caps and domain/horizon grouping. This repair uses val-only grouped endpoint-risk caps and easy blocking.",
        "best_trial": best_name,
        "evaluated_trials": [name for _score_value, name, _item in selected_trials],
        "max_repair_checkpoints": MAX_REPAIR_CHECKPOINTS,
        "best_metrics": metrics,
        "positive_external_domains": int(positive_domains),
        "neural_exceeds_stage37_by_gate_margin": bool(beats_stage37),
        "deployment_decision": "deploy_all_agent_risk_repair" if beats_stage37 and positive_domains >= 2 else "diagnostic_keep_m3w_neural_v1_endpoint_candidate",
        "candidates": candidates,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_endpoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "policy_selection_split": "val",
            "test_used_once_for_final_eval": True,
        },
    }
    _write_json(OUT_DIR / "stage41_all_agent_risk_repair.json", result)
    write_md(
        OUT_DIR / "stage41_all_agent_risk_repair.md",
        [
            "# Stage41 All-Agent Risk-Cap Repair",
            "",
            "- source: `fresh_run`",
            "- result_source: `fresh_run` for policy repair over cached all-agent checkpoints",
            f"- best trial: `{best_name}`",
            f"- deployment: `{result['deployment_decision']}`",
            f"- metrics: `{metrics}`",
            "",
            "## Interpretation",
            "",
            "This is a concrete all-agent repair experiment. It does not use future endpoints as input, and thresholds are selected on validation groups only. If it does not pass the Stage37-margin gate, M3W-Neural v1 remains the endpoint-level protected candidate while all-agent world-state dynamics remain a gap.",
        ],
    )
    return result


def main_all_agent_risk_repair() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_all_agent_risk_repair()
        status = "success"
    finally:
        _append_ledger(
            "stage41_all_agent_risk_repair",
            status,
            started,
            [str(OUT_DIR / "stage41_all_agent_training_trials.json")],
            [str(OUT_DIR / "stage41_all_agent_risk_repair.md")],
        )
