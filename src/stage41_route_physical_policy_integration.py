from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_fresh_confirmation as fresh
from src import stage41_full_trajectory_world_state as ft
from src import stage41_goal_route_physical_repair as gr


OUT_DIR = fresh.OUT_DIR
DATA_DIR = fresh.DATA_DIR
LEDGER_JSONL = fresh.LEDGER_JSONL
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
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _checkpoint_paths(report_path: str | Path, prefix: str) -> list[str]:
    report = read_json(report_path, {})
    out: list[str] = []
    for item in (report.get("trials") or {}).values():
        path = item.get("train", {}).get("checkpoint")
        if path and Path(path).exists():
            out.append(str(path))
    if out:
        return out
    paths = sorted(str(p) for p in fresh.CHECKPOINT_DIR.glob(f"stage41_{prefix}*.pt"))
    if not paths:
        raise FileNotFoundError(f"no checkpoints found for {prefix}")
    return paths


def _route_features(route_pred: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    prob = gr._softmax(route_pred["route_logits"].astype(np.float64))
    route_idx = np.argmax(prob, axis=1).astype(np.int16)
    route_conf = np.max(prob, axis=1).astype(np.float64)
    non_straight = route_idx != gr.ROUTE_NAMES.index("straight")
    hard_route_ids = {
        gr.ROUTE_NAMES.index("left_turn"),
        gr.ROUTE_NAMES.index("right_turn"),
        gr.ROUTE_NAMES.index("reverse_or_uturn"),
        gr.ROUTE_NAMES.index("interaction_detour"),
    }
    hard_route = np.asarray([int(r) in hard_route_ids for r in route_idx], dtype=bool)
    return {
        "route_idx": route_idx,
        "route_conf": route_conf,
        "non_straight": non_straight,
        "hard_route": hard_route,
        "physical_challenge": route_pred["physical"].astype(np.float64),
    }


def _slice_mapping(mapping: Mapping[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v) for k, v in mapping.items()}


def _floor_ade(labels: Mapping[str, np.ndarray]) -> np.ndarray:
    return ft._trajectory_errors(ft._floor_waypoints(labels), labels)[0]


def _neural_ade(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> np.ndarray:
    return ft._trajectory_errors(ft._pred_waypoints(pred, labels), labels)[0]


def _metric(selected: np.ndarray, fallback: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray) -> Dict[str, Any]:
    base = {
        "horizon": labels["horizon"],
        "hard": labels["hard"],
        "failure": labels["failure"],
        "easy": labels["easy"],
        "domain": labels["domain"],
        "candidate_fde": labels["candidate_fde"],
    }
    return s41._metrics(selected, fallback, base, switch)


def _route_condition(route_feat: Mapping[str, np.ndarray], params: Mapping[str, Any]) -> np.ndarray:
    mode = params.get("route_mode", "any")
    conf = route_feat["route_conf"] >= float(params.get("route_conf_min", 0.0))
    if mode == "any":
        return np.ones_like(conf, dtype=bool)
    if mode == "confident":
        return conf
    if mode == "non_straight":
        return conf & route_feat["non_straight"]
    if mode == "hard_route":
        return conf & route_feat["hard_route"]
    if mode == "straight_or_stop":
        idx = route_feat["route_idx"]
        allowed = (idx == gr.ROUTE_NAMES.index("straight")) | (idx == gr.ROUTE_NAMES.index("stop"))
        return conf & allowed
    raise ValueError(f"unknown route mode {mode}")


def _apply_policy(
    traj_pred: Mapping[str, np.ndarray],
    route_pred: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    policy: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    floor = _floor_ade(labels)
    neural = _neural_ade(traj_pred, labels)
    selected = floor.copy()
    switch = np.zeros(len(floor), dtype=bool)
    source = np.zeros(len(floor), dtype=np.int16)
    route_feat = _route_features(route_pred)
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    easy = labels["easy"].astype(bool)
    for key, params in policy.get("slices", {}).items():
        d, h_s = key.split("|")
        mask = (domain == d) & (horizon == int(h_s))
        if not np.any(mask):
            continue
        local = (
            mask
            & (traj_pred["traj_risk"] <= float(params.get("traj_risk_max", 1e9)))
            & _route_condition(route_feat, params)
        )
        if params.get("use_physical", False):
            lo = float(params.get("physical_min", 0.0))
            hi = float(params.get("physical_max", 1.0))
            local &= (route_feat["physical_challenge"] >= lo) & (route_feat["physical_challenge"] <= hi)
        if params.get("hard_only", False):
            local &= hard
        if params.get("easy_block", True):
            local &= ~easy
        max_switch = float(params.get("max_switch", 1.0))
        if max_switch <= 0.0:
            local[:] = False
        elif max_switch < 1.0 and np.any(local):
            ids = np.where(local)[0]
            priority = -traj_pred["traj_risk"][ids] + 0.05 * route_feat["route_conf"][ids] + 0.05 * route_feat["physical_challenge"][ids]
            keep_n = max(1, int(max_switch * int(np.sum(mask))))
            keep = np.zeros(len(local), dtype=bool)
            keep[ids[np.argsort(priority)[::-1][:keep_n]]] = True
            local &= keep
        selected[local] = neural[local]
        switch |= local
        source[local] = 1
    return selected, switch, source


def _policy_grid(
    traj_pred: Mapping[str, np.ndarray],
    route_pred: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    mask: np.ndarray,
    mode: str,
) -> list[Dict[str, Any]]:
    route_feat = _route_features(route_pred)
    risk = traj_pred["traj_risk"][mask]
    phys = route_feat["physical_challenge"][mask]
    thresholds = [float(v) for v in np.quantile(risk, [0.05, 0.10, 0.20, 0.35, 0.50, 0.70])] if len(risk) else [0.0]
    phys_thresholds = [float(v) for v in np.quantile(phys, [0.25, 0.50, 0.75])] if len(phys) else [0.0]
    route_modes = ["any"] if mode in {"no_route_physical", "physical_only"} else ["confident", "non_straight", "hard_route", "straight_or_stop"]
    use_physical = mode in {"physical_only", "route_physical"}
    out: list[Dict[str, Any]] = []
    for traj_risk_max in thresholds:
        for max_switch in [0.0, 0.05, 0.10, 0.20, 0.40, 0.70]:
            for hard_only in [False, True]:
                for route_mode in route_modes:
                    route_conf_values = [0.0] if route_mode == "any" else [0.35, 0.50, 0.65, 0.80]
                    for route_conf_min in route_conf_values:
                        if not use_physical:
                            out.append(
                                {
                                    "traj_risk_max": traj_risk_max,
                                    "max_switch": max_switch,
                                    "hard_only": hard_only,
                                    "easy_block": True,
                                    "route_mode": route_mode,
                                    "route_conf_min": route_conf_min,
                                    "use_physical": False,
                                }
                            )
                        else:
                            for physical_min in [0.0, *phys_thresholds]:
                                out.append(
                                    {
                                        "traj_risk_max": traj_risk_max,
                                        "max_switch": max_switch,
                                        "hard_only": hard_only,
                                        "easy_block": True,
                                        "route_mode": route_mode,
                                        "route_conf_min": route_conf_min,
                                        "use_physical": True,
                                        "physical_min": physical_min,
                                        "physical_max": 1.0,
                                    }
                                )
    return out


def _score(metrics: Mapping[str, Any]) -> float:
    max_domain_easy = max([float(row.get("easy_degradation", 0.0)) for row in (metrics.get("by_domain") or {}).values()] or [0.0])
    return (
        1.3 * float(metrics.get("all_improvement", 0.0))
        + 1.5 * float(metrics.get("t50_improvement", 0.0))
        + 1.0 * float(metrics.get("t100_improvement", 0.0))
        + 1.2 * float(metrics.get("hard_failure_improvement", 0.0))
        - 35.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 40.0 * max(0.0, max_domain_easy - 0.02)
    )


def _fit_policy(
    traj_pred: Mapping[str, np.ndarray],
    route_pred: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    mode: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    floor = _floor_ade(labels)
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    policy = {"type": f"route_physical_aware_{mode}", "mode": mode, "slices": {}}
    selected = floor.copy()
    switch = np.zeros(len(floor), dtype=bool)
    diagnostics: Dict[str, Any] = {}
    for d in sorted(set(domain.tolist())):
        for h in [10, 25, 50, 100]:
            mask = (domain == d) & (horizon == h)
            if int(np.sum(mask)) < 80:
                continue
            masked_labels = _slice_mapping(labels, mask)
            masked_traj = _slice_mapping(traj_pred, mask)
            masked_route = _slice_mapping(route_pred, mask)
            best_params: Dict[str, Any] | None = None
            best_metrics: Dict[str, Any] | None = None
            best_score = 0.0
            for params in _policy_grid(traj_pred, route_pred, labels, mask, mode):
                sel, sw, _src = _apply_policy(masked_traj, masked_route, masked_labels, {"slices": {f"{d}|{h}": params}})
                metrics = _metric(sel, _floor_ade(masked_labels), masked_labels, sw)
                if metrics.get("all_improvement", 0.0) <= 0.0 or metrics.get("easy_degradation", 0.0) > 0.02:
                    continue
                score = _score(metrics)
                if score > best_score:
                    best_score = score
                    best_params = dict(params)
                    best_metrics = metrics
            if best_params is not None:
                policy["slices"][f"{d}|{h}"] = best_params
                sel, sw, _src = _apply_policy(masked_traj, masked_route, masked_labels, {"slices": {f"{d}|{h}": best_params}})
                selected[mask] = sel
                switch[mask] = sw
            diagnostics[f"{d}|{h}"] = {"selected": bool(best_params), "val_score": float(best_score), "val_metrics": best_metrics or {"rows": int(np.sum(mask)), "all_improvement": 0.0}}
    metrics = _metric(selected, floor, labels, switch)
    metrics["slice_diagnostics"] = diagnostics
    return policy, metrics


def _eval_policy(
    traj_pred: Mapping[str, np.ndarray],
    route_pred: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    policy: Mapping[str, Any],
    bootstrap: bool = False,
) -> Dict[str, Any]:
    selected, switch, source = _apply_policy(traj_pred, route_pred, labels, policy)
    floor = _floor_ade(labels)
    metrics = _metric(selected, floor, labels, switch)
    metrics["selected_source_distribution"] = {str(k): int(v) for k, v in zip(*np.unique(source, return_counts=True))}
    route_feat = _route_features(route_pred)
    metrics["route_switch_rate"] = {
        name: float(np.mean(switch[route_feat["route_idx"] == i])) if np.any(route_feat["route_idx"] == i) else 0.0
        for i, name in enumerate(gr.ROUTE_NAMES)
    }
    if bootstrap:
        base = {
            "horizon": labels["horizon"],
            "hard": labels["hard"],
            "failure": labels["failure"],
            "easy": labels["easy"],
            "domain": labels["domain"],
            "candidate_fde": labels["candidate_fde"],
        }
        metrics["all_ci"] = s41._bootstrap_ci(selected, floor, base, "all", n=2000)
        metrics["t50_ci"] = s41._bootstrap_ci(selected, floor, base, "t50", n=2000)
        metrics["hard_failure_ci"] = s41._bootstrap_ci(selected, floor, base, "hard_failure", n=1000)
    return metrics


def _improvement_delta(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "all_delta": float(a.get("all_improvement", 0.0) - b.get("all_improvement", 0.0)),
        "t50_delta": float(a.get("t50_improvement", 0.0) - b.get("t50_improvement", 0.0)),
        "t100_delta": float(a.get("t100_improvement", 0.0) - b.get("t100_improvement", 0.0)),
        "hard_delta": float(a.get("hard_failure_improvement", 0.0) - b.get("hard_failure_improvement", 0.0)),
        "easy_delta": float(a.get("easy_degradation", 0.0) - b.get("easy_degradation", 0.0)),
    }


def run_route_physical_policy_integration() -> Dict[str, Any]:
    full_report = read_json(OUT_DIR / "stage41_full_trajectory_world_state.json", {})
    route_report = read_json(OUT_DIR / "stage41_goal_route_physical_repair.json", {})
    if not full_report:
        from src.stage41_full_trajectory_world_state import train_full_trajectory_world_state

        full_report = train_full_trajectory_world_state()
    if not route_report:
        from src.stage41_goal_route_physical_repair import train_goal_route_physical_repair

        route_report = train_goal_route_physical_repair()
    traj_paths = _checkpoint_paths(OUT_DIR / "stage41_full_trajectory_world_state.json", "full_traj")
    route_paths = _checkpoint_paths(OUT_DIR / "stage41_goal_route_physical_repair.json", "goal_route")
    val_traj, val_labels = ft._predict_ensemble(traj_paths, "val")
    test_traj, test_labels = ft._predict_ensemble(traj_paths, "test")
    val_route, route_val_labels = gr._predict_ensemble(route_paths, "val")
    test_route, route_test_labels = gr._predict_ensemble(route_paths, "test")
    if len(route_val_labels["route"]) != len(val_labels["horizon"]) or len(route_test_labels["route"]) != len(test_labels["horizon"]):
        raise ValueError("route/physical predictions are not row-aligned with trajectory predictions")

    modes = ["no_route_physical", "physical_only", "route_only", "route_physical"]
    ablations: Dict[str, Any] = {}
    best_mode = ""
    best_score = -1e18
    best_policy: Dict[str, Any] = {}
    for mode in modes:
        policy, val_metrics = _fit_policy(val_traj, val_route, val_labels, mode)
        test_metrics = _eval_policy(test_traj, test_route, test_labels, policy, bootstrap=False)
        score = _score(val_metrics)
        ablations[mode] = {"source": "fresh_run", "policy": policy, "val_metrics": val_metrics, "test_metrics": test_metrics, "val_score": score}
        if score > best_score:
            best_score = score
            best_mode = mode
            best_policy = policy
    best_metrics = _eval_policy(test_traj, test_route, test_labels, best_policy, bootstrap=True)
    no_aux = ablations["no_route_physical"]["test_metrics"]
    full_ref = full_report.get("best_metrics", {})
    lift_over_no_aux = _improvement_delta(best_metrics, no_aux)
    lift_over_full_ref = _improvement_delta(best_metrics, full_ref)
    contributes = bool(
        best_mode == "route_physical"
        and best_metrics.get("easy_degradation", 1.0) <= 0.02
        and (
            lift_over_no_aux["all_delta"] > 0.002
            or lift_over_no_aux["t50_delta"] > 0.002
            or lift_over_no_aux["hard_delta"] > 0.002
        )
    )
    result = {
        "source": "fresh_run",
        "protocol_status": "route_physical_policy_integration",
        "best_mode": best_mode,
        "best_policy": best_policy,
        "best_metrics": best_metrics,
        "ablations": ablations,
        "lift_over_no_route_physical": lift_over_no_aux,
        "lift_over_full_trajectory_reference": lift_over_full_ref,
        "route_physical_policy_contributes": contributes,
        "no_leakage": {
            "route_physical_predictions_from_past_only_models": True,
            "future_route_label_input": False,
            "future_physical_label_input": False,
            "future_waypoints_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "caveat": "This evaluates whether auxiliary route/physical heads improve deployment gating. It remains dataset-local raw-frame 2.5D and does not execute Stage5C or SMC.",
    }
    _write_json(OUT_DIR / "stage41_route_physical_policy_integration.json", result)
    lines = [
        "# Stage41 Route/Physical-Aware Policy Integration",
        "",
        "- source: `fresh_run`",
        f"- best mode: `{best_mode}`",
        f"- route/physical contributes: `{contributes}`",
        f"- best metrics: `{best_metrics}`",
        f"- lift over no-route/physical: `{lift_over_no_aux}`",
        f"- lift over full trajectory reference: `{lift_over_full_ref}`",
        "",
        "| ablation | all | t50 | t100 | hard | easy | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, row in ablations.items():
        m = row["test_metrics"]
        lines.append(
            f"| {mode} | {m.get('all_improvement', 0.0):.6f} | {m.get('t50_improvement', 0.0):.6f} | {m.get('t100_improvement', 0.0):.6f} | {m.get('hard_failure_improvement', 0.0):.6f} | {m.get('easy_degradation', 0.0):.6f} | {m.get('switch_rate', 0.0):.6f} |"
        )
    lines.extend(["", f"- no leakage: `{result['no_leakage']}`"])
    write_md(OUT_DIR / "stage41_route_physical_policy_integration.md", lines)
    return result


def main_route_physical_policy_integration() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_route_physical_policy_integration()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_route_physical_policy_integration",
            status,
            started,
            [OUT_DIR / "stage41_full_trajectory_world_state.json", OUT_DIR / "stage41_goal_route_physical_repair.json"],
            [OUT_DIR / "stage41_route_physical_policy_integration.md", OUT_DIR / "stage41_route_physical_policy_integration.json"],
        )

