from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_domain_local_neural_retrain as dl
from src import stage41_endpoint_to_full_trajectory_repair as bridge
from src import stage41_full_trajectory_world_state as ft
from src import stage41_learned_shape_gain_gate as gain_gate
from src import stage41_learned_waypoint_shape_bridge as shape
from src import stage41_shape_policy_composer as composer


OUT_DIR = dl.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_dynamic_shape_meta_policy.json"
REPORT_MD = OUT_DIR / "stage41_dynamic_shape_meta_policy.md"
SOURCES = ("bridge", "old_shape", "gain_gate")
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


def _fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float64)
    std = np.maximum(x.std(axis=0), 1e-4).astype(np.float64)
    return mean, std


def _standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float64) - mean) / std).astype(np.float64)


def _ridge_fit(x: np.ndarray, y: np.ndarray, lam: float = 0.08) -> np.ndarray:
    xb = np.concatenate([np.ones((len(x), 1), dtype=np.float64), x.astype(np.float64)], axis=1)
    reg = lam * np.eye(xb.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ y.astype(np.float64))


def _ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xb = np.concatenate([np.ones((len(x), 1), dtype=np.float64), x.astype(np.float64)], axis=1)
    return xb @ w


def _source_xy(pack: Mapping[str, Any], source: str) -> np.ndarray:
    if source == "bridge":
        return pack["bridge_xy"]
    return pack[source]["xy"]


def _source_switch(pack: Mapping[str, Any], source: str) -> np.ndarray:
    if source == "bridge":
        return np.zeros(len(pack["horizon"]), dtype=bool)
    return pack[source]["shape_switch"].astype(bool)


def _source_feature_matrix(pack: Mapping[str, Any], source: str) -> np.ndarray:
    data = pack["data"]
    xy = _source_xy(pack, source).astype(np.float64)
    bridge_xy = pack["bridge_xy"].astype(np.float64)
    norm = np.maximum(data["normalizer"].astype(np.float64), EPS)
    current = data["current_xy"].astype(np.float64)
    endpoint_delta = (xy[:, -1, :] - current) / norm[:, None]
    bridge_endpoint_delta = (bridge_xy[:, -1, :] - current) / norm[:, None]
    endpoint_gap = endpoint_delta - bridge_endpoint_delta
    source_step = np.linalg.norm(np.diff(xy, axis=1), axis=2) / norm[:, None]
    bridge_step = np.linalg.norm(np.diff(bridge_xy, axis=1), axis=2) / norm[:, None]
    waypoint_gap = np.linalg.norm(xy - bridge_xy, axis=2) / norm[:, None]
    horizon = data["horizon"].astype(np.float64)
    source_onehot = np.zeros((len(horizon), len(SOURCES)), dtype=np.float64)
    source_onehot[:, SOURCES.index(source)] = 1.0
    geometric = np.column_stack(
        [
            endpoint_delta,
            endpoint_gap,
            np.linalg.norm(endpoint_delta, axis=1),
            np.linalg.norm(endpoint_gap, axis=1),
            source_step.sum(axis=1),
            source_step.max(axis=1),
            source_step.std(axis=1),
            (source_step - bridge_step).sum(axis=1),
            waypoint_gap.mean(axis=1),
            waypoint_gap.max(axis=1),
            _source_switch(pack, source).astype(np.float64),
            horizon / 100.0,
        ]
    )
    return np.nan_to_num(
        np.concatenate([shape._feature_matrix(data, pack["endpoint_pred"]), source_onehot, geometric], axis=1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float64)


def _source_costs(pack: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {source: ft._trajectory_errors(_source_xy(pack, source), pack["labels"])[0] for source in SOURCES}


def _fit_cost_model(pack: Mapping[str, Any]) -> dict[str, Any]:
    x_parts = []
    y_parts = []
    for source, cost in _source_costs(pack).items():
        x_parts.append(_source_feature_matrix(pack, source))
        y_parts.append(np.log1p(np.maximum(cost, 0.0)))
    x_raw = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    mean, std = _fit_standardizer(x_raw)
    w = _ridge_fit(_standardize(x_raw, mean, std), y)
    return {"mean": mean, "std": std, "w": w, "feature_dim": int(x_raw.shape[1])}


def _predict_source_costs(model: Mapping[str, Any], pack: Mapping[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for source in SOURCES:
        x = _standardize(_source_feature_matrix(pack, source), np.asarray(model["mean"]), np.asarray(model["std"]))
        raw = np.clip(_ridge_predict(x, np.asarray(model["w"])), -12.0, 12.0)
        pred = np.maximum(np.expm1(raw), 0.0)
        out[source] = np.nan_to_num(pred, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)
    return out


def _choose_dynamic_sources(pack: Mapping[str, Any], pred: Mapping[str, np.ndarray], policy: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_stack = np.nan_to_num(np.column_stack([pred[source] for source in SOURCES]), nan=1.0e6, posinf=1.0e6, neginf=1.0e6)
    best_idx = np.argmin(pred_stack, axis=1)
    ordered = np.sort(pred_stack, axis=1)
    margin = ordered[:, 1] - ordered[:, 0]
    gain = pred["bridge"] - pred_stack[np.arange(len(best_idx)), best_idx]
    chosen = best_idx.copy()
    non_bridge = best_idx != SOURCES.index("bridge")
    allowed = non_bridge & (gain >= float(policy["gain_min"])) & (margin >= float(policy["margin_min"]))
    if np.any(allowed):
        for horizon_value in [10, 25, 50, 100]:
            active = allowed & (pack["horizon"].astype(int) == horizon_value)
            max_rate = float(policy[f"max_rate_h{horizon_value}"])
            max_rows = int(max_rate * max(1, int(np.sum(pack["horizon"].astype(int) == horizon_value))))
            if max_rows <= 0:
                allowed[active] = False
                continue
            ids = np.where(active)[0]
            keep = np.zeros(len(active), dtype=bool)
            keep[ids[np.argsort(gain[ids])[::-1][:max_rows]]] = True
            allowed[active & ~keep] = False
    chosen[~allowed] = SOURCES.index("bridge")
    selected = pack["bridge_xy"].copy()
    shape_switch = np.zeros(len(chosen), dtype=bool)
    for i, source in enumerate(SOURCES):
        mask = chosen == i
        if source == "bridge" or not np.any(mask):
            continue
        selected[mask] = _source_xy(pack, source)[mask]
        shape_switch[mask] = _source_switch(pack, source)[mask]
    return selected, shape_switch, chosen


def _ranking_accuracy(pack: Mapping[str, Any], pred: Mapping[str, np.ndarray]) -> float:
    true_cost = np.column_stack([_source_costs(pack)[source] for source in SOURCES])
    pred_cost = np.nan_to_num(np.column_stack([pred[source] for source in SOURCES]), nan=1.0e6, posinf=1.0e6, neginf=1.0e6)
    return float(np.mean(np.argmin(true_cost, axis=1) == np.argmin(pred_cost, axis=1)))


def _fast_eval_policy(pack: Mapping[str, Any], xy: np.ndarray, shape_switch: np.ndarray) -> dict[str, Any]:
    labels = pack["labels"]
    floor_ade, _floor_fde = ft._trajectory_errors(pack["floor_xy"], labels)
    bridge_ade, _bridge_fde = ft._trajectory_errors(pack["bridge_xy"], labels)
    selected_ade, _selected_fde = ft._trajectory_errors(xy, labels)
    horizon = labels["horizon"].astype(int)
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    return {
        "ade_metrics_vs_floor": s41._metrics(selected_ade, floor_ade, shape._metric_ds(labels), pack["bridge_switch"]),
        "shape_gain_vs_bridge": {
            "all": shape._gain(selected_ade, bridge_ade, np.ones(len(selected_ade), dtype=bool)),
            "t50": shape._gain(selected_ade, bridge_ade, horizon == 50),
            "t100": shape._gain(selected_ade, bridge_ade, horizon == 100),
            "hard_failure": shape._gain(selected_ade, bridge_ade, hard),
            "shape_switch_rate": float(np.mean(shape_switch)) if len(shape_switch) else 0.0,
        },
    }


def _select_policy_on_val(model: Mapping[str, Any], pack: Mapping[str, Any]) -> dict[str, Any]:
    pred = _predict_source_costs(model, pack)
    pred_stack = np.nan_to_num(np.column_stack([pred[source] for source in SOURCES]), nan=1.0e6, posinf=1.0e6, neginf=1.0e6)
    margin = np.sort(pred_stack, axis=1)[:, 1] - np.sort(pred_stack, axis=1)[:, 0]
    gain = pred["bridge"] - pred_stack.min(axis=1)
    gain_grid = [0.0]
    margin_grid = [0.0]
    positive_gain = gain[gain > 0]
    positive_margin = margin[margin > 0]
    if len(positive_gain):
        gain_grid.extend(float(v) for v in np.quantile(positive_gain, [0.45, 0.65, 0.80]))
    if len(positive_margin):
        margin_grid.extend(float(v) for v in np.quantile(positive_margin, [0.35, 0.55, 0.75]))
    rate_grid = [0.002, 0.005, 0.01, 0.02, 0.05]
    fast_rows: list[dict[str, Any]] = []
    for gain_min in gain_grid:
        for margin_min in margin_grid:
            for short_rate in [0.0, 0.002, 0.005, 0.01]:
                for t50_rate in rate_grid:
                    for t100_rate in rate_grid:
                        policy = {
                            "gain_min": gain_min,
                            "margin_min": margin_min,
                            "max_rate_h10": short_rate,
                            "max_rate_h25": short_rate,
                            "max_rate_h50": t50_rate,
                            "max_rate_h100": t100_rate,
                        }
                        xy, shape_switch, chosen = _choose_dynamic_sources(pack, pred, policy)
                        ev = _fast_eval_policy(pack, xy, shape_switch)
                        m = ev["ade_metrics_vs_floor"]
                        g = ev["shape_gain_vs_bridge"]
                        non_bridge = chosen != SOURCES.index("bridge")
                        eligible = (
                            np.any(non_bridge)
                            and m.get("all_improvement", 0.0) > 0.0
                            and m.get("t50_improvement", 0.0) > 0.0
                            and m.get("hard_failure_improvement", 0.0) > 0.0
                            and m.get("easy_degradation", 1.0) <= 0.02
                            and max(g["all"], g["t50"], g["t100"], g["hard_failure"]) >= 0.0
                        )
                        score = float(
                            m.get("all_improvement", 0.0)
                            + 1.6 * m.get("t50_improvement", 0.0)
                            + 1.4 * m.get("t100_improvement", 0.0)
                            + 1.2 * m.get("hard_failure_improvement", 0.0)
                            + 4.0 * max(g["all"], g["t50"], g["t100"], g["hard_failure"])
                            - 45.0 * max(0.0, m.get("easy_degradation", 1.0) - 0.02)
                        )
                        fast_rows.append(
                            {
                                "policy": policy,
                                "eligible": bool(eligible),
                                "score": score,
                                "val_metrics": {
                                    "ade": m,
                                    "shape_gain_vs_bridge": g,
                                    "collision_delta_005": None,
                                    "source_distribution": _source_distribution(chosen),
                                },
                            }
                        )
    rows: list[dict[str, Any]] = []
    preselected = sorted(fast_rows, key=lambda row: row["score"], reverse=True)[:48]
    for row in preselected:
        xy, shape_switch, chosen = _choose_dynamic_sources(pack, pred, row["policy"])
        ev = composer._eval_selected(pack, xy, shape_switch)
        m = ev["ade_metrics_vs_floor"]
        g = ev["shape_gain_vs_bridge"]
        non_bridge = chosen != SOURCES.index("bridge")
        eligible = (
            np.any(non_bridge)
            and m.get("all_improvement", 0.0) > 0.0
            and m.get("t50_improvement", 0.0) > 0.0
            and m.get("hard_failure_improvement", 0.0) > 0.0
            and m.get("easy_degradation", 1.0) <= 0.02
            and ev["collision_delta_vs_floor_005"] <= 0.01
            and max(g["all"], g["t50"], g["t100"], g["hard_failure"]) >= 0.0
        )
        rows.append(
            {
                "policy": row["policy"],
                "eligible": bool(eligible),
                "score": composer._score_eval(ev),
                "val_metrics": {
                    "ade": m,
                    "shape_gain_vs_bridge": g,
                    "collision_delta_005": ev["collision_delta_vs_floor_005"],
                    "source_distribution": _source_distribution(chosen),
                },
            }
        )
    pool = [row for row in rows if row["eligible"]] or rows or fast_rows
    selected = max(pool, key=lambda row: (bool(row["eligible"]), row["score"]))
    return {
        "selected": selected,
        "candidate_count": len(rows),
        "fast_candidate_count": len(fast_rows),
        "eligible_count": int(sum(row["eligible"] for row in rows)),
        "top_candidates": sorted(rows, key=lambda row: row["score"], reverse=True)[:10],
        "ranking_accuracy": _ranking_accuracy(pack, pred),
    }


def _source_distribution(chosen: np.ndarray) -> dict[str, float]:
    return {source: float(np.mean(chosen == i)) if len(chosen) else 0.0 for i, source in enumerate(SOURCES)}


def _compact_eval(ev: Mapping[str, Any], chosen: np.ndarray) -> dict[str, Any]:
    m = ev["ade_metrics_vs_floor"]
    g = ev["shape_gain_vs_bridge"]
    return {
        "all": float(m.get("all_improvement", 0.0)),
        "t50": float(m.get("t50_improvement", 0.0)),
        "t100": float(m.get("t100_improvement", 0.0)),
        "hard_failure": float(m.get("hard_failure_improvement", 0.0)),
        "easy_degradation": float(m.get("easy_degradation", 0.0)),
        "switch_rate": float(m.get("switch_rate", 0.0)),
        "shape_gain_all": float(g.get("all", 0.0)),
        "shape_gain_t50": float(g.get("t50", 0.0)),
        "shape_gain_t100": float(g.get("t100", 0.0)),
        "shape_gain_hard_failure": float(g.get("hard_failure", 0.0)),
        "shape_switch_rate": float(g.get("shape_switch_rate", 0.0)),
        "collision_delta_005": float(ev["collision_delta_vs_floor_005"]),
        "source_distribution": _source_distribution(chosen),
    }


def _chosen_from_fixed_policy(pack: Mapping[str, Any], policy: Mapping[str, str]) -> np.ndarray:
    names = composer._source_name(policy, pack["horizon"])
    out = np.zeros(len(names), dtype=np.int64)
    for i, source in enumerate(SOURCES):
        out[names == source] = i
    return out


def _source_policy_summary(selection: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = selection["selected"]
    return {
        "policy": selected.get("policy", selected),
        "eligible": bool(selected.get("eligible", False)),
        "score": float(selected.get("score", 0.0)),
        "candidate_count": int(selection.get("candidate_count", 0)),
        "eligible_count": int(selection.get("eligible_count", 0)),
    }


def _domain_data(split: str, domain: str) -> dict[str, np.ndarray]:
    data = dl._load_split(split)
    return bridge._subset(data, dl._domain_mask(data, domain))


def _make_meta_pack(
    data: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    endpoint_pred: Mapping[str, np.ndarray],
    gate_pred: Mapping[str, np.ndarray],
    shape_pred: Mapping[str, np.ndarray],
    bridge_selection: Mapping[str, Any],
    old_policy: Mapping[str, Any],
    gain_policy: Mapping[str, Any],
) -> dict[str, Any]:
    pack = composer._make_source_pack(data, labels, endpoint_pred, gate_pred, shape_pred, bridge_selection, old_policy, gain_policy)
    pack["endpoint_pred"] = endpoint_pred
    return pack


def _evaluate_domain(domain: str) -> dict[str, Any]:
    train = _domain_data("train", domain)
    val = _domain_data("val", domain)
    test = _domain_data("test", domain)
    if min(len(train["horizon"]), len(val["horizon"]), len(test["horizon"])) < 500:
        return {"domain": domain, "status": "not_run", "reason": "not enough domain rows"}
    endpoint_training = dl._train_endpoint(domain, train, val)
    pred_train = dl._predict_endpoint(endpoint_training["checkpoint"], train)
    pred_val = dl._predict_endpoint(endpoint_training["checkpoint"], val)
    pred_test = dl._predict_endpoint(endpoint_training["checkpoint"], test)
    fde_train = dl._endpoint_fde(pred_train["delta"], train)
    fde_val = dl._endpoint_fde(pred_val["delta"], val)
    fde_test = dl._endpoint_fde(pred_test["delta"], test)
    gate = dl._train_gate(train, pred_train, fde_train)
    gate_train = dl._predict_gate(gate, train, pred_train, fde_train)
    gate_val = dl._predict_gate(gate, val, pred_val, fde_val)
    gate_test = dl._predict_gate(gate, test, pred_test, fde_test)
    labels_train = bridge._align_full_labels("train", train)
    labels_val = bridge._align_full_labels("val", val)
    labels_test = bridge._align_full_labels("test", test)
    bridge_selection = bridge._select_policy_on_val(val, labels_val, pred_val, gate_val)
    shape_training = shape._train_shape_head(domain, train, val, pred_train, pred_val, labels_train, labels_val)
    shape_train = shape._predict_shape(shape_training["checkpoint"], train, pred_train)
    shape_val = shape._predict_shape(shape_training["checkpoint"], val, pred_val)
    shape_test = shape._predict_shape(shape_training["checkpoint"], test, pred_test)
    old_selection = shape._select_shape_policy_on_val(val, labels_val, pred_val, gate_val, shape_val, bridge_selection)
    train_gain_pack = gain_gate._make_pack(train, labels_train, pred_train, gate_train, shape_train, bridge_selection)
    val_gain_pack = gain_gate._make_pack(val, labels_val, pred_val, gate_val, shape_val, bridge_selection)
    gain_selection = gain_gate._select_policy_on_val(train_gain_pack, val_gain_pack)
    train_pack = _make_meta_pack(train, labels_train, pred_train, gate_train, shape_train, bridge_selection, old_selection["selected"], gain_selection["selected"])
    val_pack = _make_meta_pack(val, labels_val, pred_val, gate_val, shape_val, bridge_selection, old_selection["selected"], gain_selection["selected"])
    test_pack = _make_meta_pack(test, labels_test, pred_test, gate_test, shape_test, bridge_selection, old_selection["selected"], gain_selection["selected"])
    cost_model = _fit_cost_model(train_pack)
    val_selection = _select_policy_on_val(cost_model, val_pack)
    pred_test_cost = _predict_source_costs(cost_model, test_pack)
    selected_xy, shape_switch, chosen = _choose_dynamic_sources(test_pack, pred_test_cost, val_selection["selected"]["policy"])
    ev = composer._eval_selected(test_pack, selected_xy, shape_switch)
    dynamic_compact = _compact_eval(ev, chosen)
    fixed_selection = composer._select_composer_on_val(val_pack)
    fixed_xy, fixed_shape_switch = composer._compose_sources(test_pack, fixed_selection["selected"]["policy"])
    fixed_eval = composer._eval_selected(test_pack, fixed_xy, fixed_shape_switch)
    fixed_chosen = _chosen_from_fixed_policy(test_pack, fixed_selection["selected"]["policy"])
    m = ev["ade_metrics_vs_floor"]
    g = ev["shape_gain_vs_bridge"]
    pass_gate = bool(
        m.get("all_improvement", 0.0) > 0.0
        and m.get("t50_improvement", 0.0) > 0.0
        and m.get("hard_failure_improvement", 0.0) > 0.0
        and m.get("easy_degradation", 1.0) <= 0.02
        and ev["collision_delta_vs_floor_005"] <= 0.01
        and dynamic_compact["shape_switch_rate"] > 0.0
        and max(g["all"], g["t50"], g["t100"], g["hard_failure"]) >= 0.0
    )
    return {
        "domain": domain,
        "source": "fresh_run",
        "status": "ok",
        "rows": {"train": int(len(train["horizon"])), "val": int(len(val["horizon"])), "test": int(len(test["horizon"]))},
        "cost_model": {"type": "ridge_log_ade_per_source", "feature_dim": int(cost_model["feature_dim"])},
        "endpoint_training": endpoint_training,
        "shape_training": shape_training,
        "bridge_selection": _source_policy_summary(bridge_selection),
        "old_shape_selection": _source_policy_summary(old_selection),
        "gain_gate_selection": _source_policy_summary(gain_selection),
        "dynamic_meta_selection": val_selection,
        "dynamic_meta_metrics": ev,
        "dynamic_meta_compact": dynamic_compact,
        "fixed_horizon_composer_compact": _compact_eval(fixed_eval, fixed_chosen),
        "test_ranking_accuracy": _ranking_accuracy(test_pack, pred_test_cost),
        "dynamic_meta_pass": pass_gate,
        "caveat": "Dynamic meta-policy predicts per-source ADE from past-only features and candidate rollout geometry. Train future waypoints are labels only; validation selects safety thresholds; test is evaluated once. Dataset-local raw-frame 2.5D only; no Stage5C/SMC/metric/seconds/true-3D/foundation claim.",
    }


def run_dynamic_shape_meta_policy() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ft.build_full_trajectory_labels()
    domains = ["ETH_UCY", "TrajNet"]
    results = {domain: _evaluate_domain(domain) for domain in domains}
    positive = [domain for domain, row in results.items() if row.get("dynamic_meta_pass")]
    result = {
        "source": "fresh_run",
        "protocol": "dynamic_shape_source_meta_policy",
        "stage5c_executed": False,
        "smc_enabled": False,
        "metric_or_seconds_claim": False,
        "positive_domains": positive,
        "positive_domain_count": len(positive),
        "two_domain_dynamic_meta_gate": len(positive) >= 2,
        "domain_results": results,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_endpoint_label_eval_only": True,
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "val_selected_policy": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "claim_boundary": {
            "dynamic_source_meta_policy": True,
            "learned_waypoint_shape_residual": True,
            "protected_by_endpoint_bridge_or_floor": True,
            "latent_generative_rollout": False,
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
        },
    }
    write_json(REPORT_JSON, _jsonable(result))
    lines = [
        "# Stage41 Dynamic Shape Source Meta-Policy",
        "",
        "- source: `fresh_run`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        f"- positive domains: `{positive}`",
        f"- two-domain dynamic meta gate: `{result['two_domain_dynamic_meta_gate']}`",
        "",
        "| domain | all ADE | t50 ADE | t100 ADE | hard ADE | easy | shape gain all/t50/t100/hard | source distribution | rank acc | pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for domain, row in results.items():
        if row.get("status") != "ok":
            lines.append(f"| `{domain}` | 0 | 0 | 0 | 0 | 0 | 0/0/0/0 | not_run | 0 | `False` |")
            continue
        c = row["dynamic_meta_compact"]
        lines.append(
            f"| `{domain}` | {c['all']:.4f} | {c['t50']:.4f} | {c['t100']:.4f} | {c['hard_failure']:.4f} | {c['easy_degradation']:.4f} | "
            f"{c['shape_gain_all']:.6f}/{c['shape_gain_t50']:.6f}/{c['shape_gain_t100']:.6f}/{c['shape_gain_hard_failure']:.6f} | "
            f"`{c['source_distribution']}` | {row['test_ranking_accuracy']:.4f} | `{row.get('dynamic_meta_pass')}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This experiment trains a per-row source cost model over bridge, previous learned-shape, and gain-gate shape sources.",
            "- The model is trained with train future-waypoint labels only; validation chooses gain/margin/source-rate thresholds; test is evaluated once.",
            "- This is a stronger dynamic policy than the fixed horizon composer if it preserves easy cases while allowing source switches with positive shape gain.",
            "- It is still protected 2.5D world-state evidence, not Stage5C, SMC, metric prediction, seconds-level prediction, true 3D, or foundation evidence.",
            f"- no leakage: `{result['no_leakage']}`",
            f"- claim boundary: `{result['claim_boundary']}`",
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_dynamic_shape_meta_policy() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_dynamic_shape_meta_policy()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_dynamic_shape_meta_policy",
            status,
            started,
            [dl.DATA_DIR / "seq2seq_train.npz", dl.DATA_DIR / "seq2seq_val.npz", dl.DATA_DIR / "seq2seq_test.npz", ft.DATA_DIR / "full_trajectory_test.npz"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_dynamic_shape_meta_policy()
