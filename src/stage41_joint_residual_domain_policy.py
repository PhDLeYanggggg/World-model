from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_full_trajectory_world_state as ft
from src import stage41_joint_residual_rollout as jrr
from src import stage41_joint_policy_distillation as jpd


OUT_DIR = jpd.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_joint_residual_domain_policy.json"
REPORT_MD = OUT_DIR / "stage41_joint_residual_domain_policy.md"


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


def _append_ledger(step: str, status: str, started: float, inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
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
    with (OUT_DIR / "run_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _slice_arrays(pred: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[mask] for k, v in pred.items()}


def _apply_sliced_policy(pred: Mapping[str, np.ndarray], data: Mapping[str, Any], policy: Mapping[str, Any]) -> np.ndarray:
    switch = np.zeros(len(data["x"]), dtype=bool)
    domain = data["domain"].astype(str)
    horizon = data["horizon"].astype(int)
    for key, params in (policy.get("slices") or {}).items():
        d, h_text = key.split("|")
        mask = (domain == d) & (horizon == int(h_text))
        if not np.any(mask):
            continue
        local = jrr._policy_switch(_slice_arrays(pred, mask), params)
        idx = np.where(mask)[0]
        switch[idx] = local
    return switch


def _fit_sliced_policy(pred: Mapping[str, np.ndarray], data: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    domain = data["domain"].astype(str)
    horizon = data["horizon"].astype(int)
    labels = jrr._labels_for_eval(data)
    neural_xy = jrr._pred_waypoints(pred, data)
    neural_ade, _neural_fde = ft._trajectory_errors(neural_xy, labels)
    floor_ade = data["floor_ade"].astype(np.float64)
    metric_ds = {
        "horizon": data["horizon"],
        "hard": data["hard"],
        "failure": data["failure"],
        "easy": data["easy"],
        "domain": data["domain"],
        "candidate_fde": data["candidate_fde"],
    }
    policy: dict[str, Any] = {"type": "joint_residual_domain_horizon_policy", "slices": {}, "missing_validation_domains": []}
    diagnostics: list[dict[str, Any]] = []
    for d in sorted(set(domain.tolist())):
        for h in [10, 25, 50, 100]:
            mask = (domain == d) & (horizon == h)
            rows = int(np.sum(mask))
            if rows < 120:
                continue
            candidates = []
            for params in jrr._policy_grid(_slice_arrays(pred, mask)):
                switch = np.zeros(len(data["x"]), dtype=bool)
                local = jrr._policy_switch(_slice_arrays(pred, mask), params)
                switch[np.where(mask)[0]] = local
                selected = floor_ade.copy()
                selected[switch] = neural_ade[switch]
                metrics = s41._metrics(selected, floor_ade, metric_ds, switch)
                eligible = bool(
                    metrics.get("all_improvement", 0.0) > 0
                    and metrics.get("easy_degradation", 1.0) <= 0.02
                    and float(np.mean(switch[mask])) > 0.0
                )
                score = (
                    float(metrics.get("all_improvement", 0.0))
                    + 1.2 * float(metrics.get("t50_improvement", 0.0))
                    + 1.0 * float(metrics.get("hard_failure_improvement", 0.0))
                    - 35.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
                )
                candidates.append(
                    {
                        "params": dict(params),
                        "metrics": metrics,
                        "local_switch_rate": float(np.mean(switch[mask])),
                        "eligible": eligible,
                        "score": score,
                    }
                )
            eligible = [row for row in candidates if row["eligible"]]
            best = max(eligible or candidates, key=lambda row: row["score"])
            selected = bool(eligible and best["score"] > 0)
            if selected:
                policy["slices"][f"{d}|{h}"] = best["params"]
            diagnostics.append({"slice": f"{d}|{h}", "rows": rows, "selected": selected, "best": best})
    return policy, diagnostics


def _trial_paths() -> list[dict[str, Any]]:
    report = read_json(OUT_DIR / "stage41_joint_residual_rollout.json", {})
    trials = report.get("trained_trials") or {}
    out = []
    for name, payload in trials.items():
        train = payload.get("train") or {}
        trial = train.get("trial") or {}
        ckpt = train.get("checkpoint")
        if ckpt and trial:
            out.append({"name": name, "checkpoint": ckpt, "trial": trial})
    return out


def run_joint_residual_domain_policy() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    trial_reports: dict[str, Any] = {}
    val_rank: list[dict[str, Any]] = []
    for item in _trial_paths():
        clip = float((item.get("trial") or {}).get("clip", 1.0))
        val = jrr._residual_bundle("val", clip)
        test = jrr._residual_bundle("test", clip)
        pred_val = jrr._predict(item["checkpoint"], val)
        policy, diagnostics = _fit_sliced_policy(pred_val, val)
        val_switch = _apply_sliced_policy(pred_val, val, policy)
        val_eval = jrr._rollout_eval(pred_val, val, val_switch, "val_domain_horizon_residual")
        pred_test = jrr._predict(item["checkpoint"], test)
        test_switch = _apply_sliced_policy(pred_test, test, policy)
        test_eval = jrr._rollout_eval(pred_test, test, test_switch, "test_domain_horizon_residual")
        score = jrr._score(val_eval["selected_metrics"], val_eval["collision_delta_005"])
        trial_reports[item["name"]] = {
            "checkpoint": item["checkpoint"],
            "trial": item["trial"],
            "policy": policy,
            "slice_diagnostics": diagnostics,
            "val_metrics": val_eval["selected_metrics"],
            "val_collision_delta_005": val_eval["collision_delta_005"],
            "val_switch_rate": float(np.mean(val_switch)),
            "test_metrics": test_eval["selected_metrics"],
            "test_collision_delta_005": test_eval["collision_delta_005"],
            "test_switch_rate": float(np.mean(test_switch)),
        }
        val_rank.append({"trial": item["name"], "score": score})
    selected = max(val_rank, key=lambda row: row["score"]) if val_rank else {"trial": None, "score": None}
    best = trial_reports.get(str(selected.get("trial")), {})
    metrics = best.get("test_metrics") or {}
    deployable = bool(
        metrics.get("all_improvement", 0.0) > 0
        and metrics.get("t50_improvement", 0.0) >= 0
        and metrics.get("hard_failure_improvement", 0.0) >= 0
        and metrics.get("easy_degradation", 1.0) <= 0.02
        and best.get("test_collision_delta_005", 1.0) <= 0.01
        and best.get("test_switch_rate", 0.0) > 0.0
    )
    result = {
        "source": "fresh_run",
        "protocol_status": "domain_horizon_residual_policy_repair",
        "hypothesis": "Validation-only domain/horizon slicing prevents a global residual gate from switching domains without validation support.",
        "selected_trial": selected.get("trial"),
        "selected_val_score": selected.get("score"),
        "trial_reports": trial_reports,
        "test_metrics": metrics,
        "test_switch_rate": best.get("test_switch_rate"),
        "test_collision_delta_005": best.get("test_collision_delta_005"),
        "domain_horizon_policy_deployable": deployable,
        "no_leakage": {
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "policy_selected_on_val": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "caveat": "This is a validation-only policy repair over bounded residual neural predictions. It is not Stage5C/SMC and remains dataset-local raw-frame 2.5D.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    write_md(
        REPORT_MD,
        [
            "# Stage41 Joint Residual Domain-Horizon Policy Repair",
            "",
            "- source: `fresh_run`",
            f"- selected trial: `{selected.get('trial')}`",
            f"- deployable: `{deployable}`",
            f"- test metrics: `{metrics}`",
            f"- test switch rate: `{best.get('test_switch_rate')}`",
            f"- test collision delta @0.05 normalized: `{best.get('test_collision_delta_005')}`",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "The policy is selected only on validation slices. Domains/horizons without validation support fall back to the floor policy.",
        ],
    )
    return result


def main_joint_residual_domain_policy() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_joint_residual_domain_policy()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_joint_residual_domain_policy",
            status,
            started,
            [OUT_DIR / "stage41_joint_residual_rollout.json"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_joint_residual_domain_policy()
