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
from src import stage41_learned_waypoint_shape_bridge as shape


OUT_DIR = dl.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_learned_shape_gain_gate.json"
REPORT_MD = OUT_DIR / "stage41_learned_shape_gain_gate.md"
SEED = 41691
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


def _ridge_fit(x: np.ndarray, y: np.ndarray, lam: float = 0.05) -> np.ndarray:
    xb = np.concatenate([np.ones((len(x), 1), dtype=np.float64), x.astype(np.float64)], axis=1)
    reg = lam * np.eye(xb.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ y.astype(np.float64))


def _ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xb = np.concatenate([np.ones((len(x), 1), dtype=np.float64), x.astype(np.float64)], axis=1)
    return xb @ w


def _gate_feature_matrix(data: Mapping[str, np.ndarray], endpoint_pred: Mapping[str, np.ndarray], shape_pred: Mapping[str, np.ndarray], shape_xy: np.ndarray, bridge_xy: np.ndarray, mode: str, residual_scale: float, residual_clip: float) -> np.ndarray:
    residual = shape_pred["residual"].astype(np.float64)
    residual_norm = np.linalg.norm(residual, axis=2)
    seg = np.linalg.norm(np.diff(shape_xy, axis=1), axis=2) / np.maximum(data["normalizer"].astype(np.float64)[:, None], EPS)
    bridge_seg = np.linalg.norm(np.diff(bridge_xy, axis=1), axis=2) / np.maximum(data["normalizer"].astype(np.float64)[:, None], EPS)
    horizon = data["horizon"].astype(np.float64)
    horizon_onehot = np.stack([(horizon == h).astype(np.float64) for h in [10, 25, 50, 100]], axis=1)
    mode_vec = np.asarray(
        [
            mode == "intermediate_only",
            mode == "endpoint_half",
            mode == "all_waypoints",
            residual_scale,
            residual_clip,
        ],
        dtype=np.float64,
    )
    core = np.column_stack(
        [
            shape_pred["risk"].astype(np.float64),
            endpoint_pred["uncertainty"].astype(np.float64),
            residual_norm.mean(axis=1),
            residual_norm.max(axis=1),
            residual_norm[:, -1],
            seg.sum(axis=1),
            seg.max(axis=1),
            np.std(seg, axis=1),
            (seg - bridge_seg).sum(axis=1),
            horizon / 100.0,
        ]
    )
    return np.nan_to_num(np.concatenate([shape._feature_matrix(data, endpoint_pred), core, horizon_onehot, np.tile(mode_vec[None, :], (len(horizon), 1))], axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def _fit_gain_gate(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], endpoint_pred: Mapping[str, np.ndarray], shape_pred: Mapping[str, np.ndarray], bridge_xy: np.ndarray, shape_xy: np.ndarray, mode: str, residual_scale: float, residual_clip: float) -> dict[str, Any]:
    bridge_ade, _bridge_fde = ft._trajectory_errors(bridge_xy, labels)
    shape_ade, _shape_fde = ft._trajectory_errors(shape_xy, labels)
    gain = bridge_ade - shape_ade
    harm = shape_ade - bridge_ade
    x_raw = _gate_feature_matrix(data, endpoint_pred, shape_pred, shape_xy, bridge_xy, mode, residual_scale, residual_clip)
    mean, std = _fit_standardizer(x_raw)
    x = _standardize(x_raw, mean, std)
    return {
        "mean": mean,
        "std": std,
        "gain_w": _ridge_fit(x, gain),
        "harm_w": _ridge_fit(x, harm),
        "gain_positive_rate": float(np.mean(gain > 0.0)),
        "mean_gain": float(np.mean(gain)),
    }


def _predict_gain_gate(gate: Mapping[str, Any], data: Mapping[str, np.ndarray], endpoint_pred: Mapping[str, np.ndarray], shape_pred: Mapping[str, np.ndarray], bridge_xy: np.ndarray, shape_xy: np.ndarray, mode: str, residual_scale: float, residual_clip: float) -> dict[str, np.ndarray]:
    x = _standardize(_gate_feature_matrix(data, endpoint_pred, shape_pred, shape_xy, bridge_xy, mode, residual_scale, residual_clip), np.asarray(gate["mean"]), np.asarray(gate["std"]))
    return {"pred_gain": _ridge_predict(x, np.asarray(gate["gain_w"])), "pred_harm": _ridge_predict(x, np.asarray(gate["harm_w"]))}


def _bridge_bundle(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], endpoint_pred: Mapping[str, np.ndarray], gate_pred: Mapping[str, np.ndarray], bridge_selection: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "labels": labels,
        **dict(zip(["floor_xy", "bridge_xy", "selected_xy", "bridge_switch", "shape_switch"], shape._apply_shape_policy(data, labels, endpoint_pred, gate_pred, {"residual": np.zeros((len(data["horizon"]), len(ft.WAYPOINT_FRAC), 2), dtype=np.float32), "risk": np.zeros(len(data["horizon"]), dtype=np.float32)}, bridge_selection, {"reason": "bridge_only"}))),
    }


def _score_val(ev: Mapping[str, Any]) -> float:
    m = ev["ade_metrics_vs_floor"]
    g = ev["learned_shape_gain_vs_bridge"]
    return float(
        m.get("all_improvement", 0.0)
        + 1.5 * m.get("t50_improvement", 0.0)
        + 1.2 * m.get("t100_improvement", 0.0)
        + 1.2 * m.get("hard_failure_improvement", 0.0)
        + 5.0 * max(g["all"], g["t50"], g["t100"], g["hard_failure"])
        - 40.0 * max(0.0, m.get("easy_degradation", 1.0) - 0.02)
        - 10.0 * max(0.0, ev["collision_delta_vs_floor_005"] - 0.01)
    )


def _candidate_modes() -> list[tuple[str, float, float]]:
    return [
        ("intermediate_only", 0.25, 0.04),
        ("intermediate_only", 0.50, 0.08),
        ("endpoint_half", 0.25, 0.04),
        ("endpoint_half", 0.50, 0.08),
        ("all_waypoints", 0.25, 0.04),
        ("all_waypoints", 0.50, 0.08),
    ]


def _select_policy_on_val(train_pack: Mapping[str, Any], val_pack: Mapping[str, Any]) -> dict[str, Any]:
    floor_ade, _floor_fde = ft._trajectory_errors(val_pack["floor_xy"], val_pack["labels"])
    bridge_ade, _bridge_fde = ft._trajectory_errors(val_pack["bridge_xy"], val_pack["labels"])
    fast_rows: list[dict[str, Any]] = []
    for mode, residual_scale, residual_clip in _candidate_modes():
        train_shape_xy = shape._shape_xy(train_pack["data"], train_pack["endpoint_pred"], train_pack["shape_pred"], mode, residual_scale, residual_clip)
        val_shape_xy = shape._shape_xy(val_pack["data"], val_pack["endpoint_pred"], val_pack["shape_pred"], mode, residual_scale, residual_clip)
        gate = _fit_gain_gate(train_pack["data"], train_pack["labels"], train_pack["endpoint_pred"], train_pack["shape_pred"], train_pack["bridge_xy"], train_shape_xy, mode, residual_scale, residual_clip)
        pred = _predict_gain_gate(gate, val_pack["data"], val_pack["endpoint_pred"], val_pack["shape_pred"], val_pack["bridge_xy"], val_shape_xy, mode, residual_scale, residual_clip)
        for horizons_name, horizons in {"all_horizons": None, "t50_only": {50}, "t100_only": {100}, "long_horizon": {50, 100}}.items():
            allowed = np.ones(len(val_pack["bridge_switch"]), dtype=bool) if horizons is None else np.isin(val_pack["data"]["horizon"].astype(int), sorted(horizons))
            active = val_pack["bridge_switch"] & allowed
            if not np.any(active):
                continue
            gain_grid = [0.0]
            gain_grid.extend(float(v) for v in np.quantile(pred["pred_gain"][active], [0.55, 0.70, 0.85]))
            harm_grid = [0.0]
            harm_grid.extend(float(v) for v in np.quantile(pred["pred_harm"][active], [0.25, 0.50]))
            for gain_min in gain_grid:
                for harm_max in harm_grid:
                    for max_shape_rate in [0.002, 0.005, 0.01, 0.02, 0.05]:
                        local = active & (pred["pred_gain"] >= gain_min) & (pred["pred_harm"] <= harm_max)
                        if not np.any(local):
                            continue
                        max_rows = max(1, int(max_shape_rate * int(np.sum(active))))
                        ids = np.where(local)[0]
                        keep = np.zeros(len(local), dtype=bool)
                        keep[ids[np.argsort(pred["pred_gain"][ids])[::-1][:max_rows]]] = True
                        local &= keep
                        if not np.any(local):
                            continue
                        selected_xy = val_pack["bridge_xy"].copy()
                        selected_xy[local] = val_shape_xy[local]
                        selected_ade, _selected_fde = ft._trajectory_errors(selected_xy, val_pack["labels"])
                        m = s41._metrics(selected_ade, floor_ade, shape._metric_ds(val_pack["labels"]), val_pack["bridge_switch"])
                        horizon = val_pack["labels"]["horizon"].astype(int)
                        hard = val_pack["labels"]["hard"].astype(bool) | val_pack["labels"]["failure"].astype(bool)
                        g = {
                            "all": shape._gain(selected_ade, bridge_ade, np.ones(len(local), dtype=bool)),
                            "t50": shape._gain(selected_ade, bridge_ade, horizon == 50),
                            "t100": shape._gain(selected_ade, bridge_ade, horizon == 100),
                            "hard_failure": shape._gain(selected_ade, bridge_ade, hard),
                            "shape_switch_rate": float(np.mean(local)) if len(local) else 0.0,
                        }
                        score = (
                            m.get("all_improvement", 0.0)
                            + 1.5 * m.get("t50_improvement", 0.0)
                            + 1.2 * m.get("t100_improvement", 0.0)
                            + 1.2 * m.get("hard_failure_improvement", 0.0)
                            + 5.0 * max(g["all"], g["t50"], g["t100"], g["hard_failure"])
                            - 40.0 * max(0.0, m.get("easy_degradation", 1.0) - 0.02)
                        )
                        fast_rows.append(
                            {
                                "mode": mode,
                                "residual_scale": residual_scale,
                                "residual_clip": residual_clip,
                                "horizons": horizons_name,
                                "gain_min": gain_min,
                                "harm_max": harm_max,
                                "max_shape_rate": max_shape_rate,
                                "min_sep": 0.05,
                                "shape_rows": int(np.sum(local)),
                                "score": float(score),
                                "gate": _jsonable(gate),
                                "fast_metrics": {"ade": m, "shape_gain_vs_bridge": g},
                            }
                        )
    rows: list[dict[str, Any]] = []
    for candidate in sorted(fast_rows, key=lambda r: r["score"], reverse=True)[:32]:
        mode = str(candidate["mode"])
        residual_scale = float(candidate["residual_scale"])
        residual_clip = float(candidate["residual_clip"])
        val_shape_xy = shape._shape_xy(val_pack["data"], val_pack["endpoint_pred"], val_pack["shape_pred"], mode, residual_scale, residual_clip)
        pred = _predict_gain_gate(candidate["gate"], val_pack["data"], val_pack["endpoint_pred"], val_pack["shape_pred"], val_pack["bridge_xy"], val_shape_xy, mode, residual_scale, residual_clip)
        horizons = {"all_horizons": None, "t50_only": {50}, "t100_only": {100}, "long_horizon": {50, 100}}[str(candidate["horizons"])]
        allowed = np.ones(len(val_pack["bridge_switch"]), dtype=bool) if horizons is None else np.isin(val_pack["data"]["horizon"].astype(int), sorted(horizons))
        local = val_pack["bridge_switch"] & allowed & (pred["pred_gain"] >= float(candidate["gain_min"])) & (pred["pred_harm"] <= float(candidate["harm_max"]))
        if np.any(local):
            active = val_pack["bridge_switch"] & allowed
            max_rows = max(1, int(float(candidate["max_shape_rate"]) * int(np.sum(active))))
            ids = np.where(local)[0]
            keep = np.zeros(len(local), dtype=bool)
            keep[ids[np.argsort(pred["pred_gain"][ids])[::-1][:max_rows]]] = True
            local &= keep
        if not np.any(local):
            continue
        selected_xy = val_pack["bridge_xy"].copy()
        selected_xy[local] = val_shape_xy[local]
        guarded_xy, guarded_switch, guarded = shape._guard(val_pack["floor_xy"], selected_xy, val_pack["bridge_xy"], val_pack["labels"], val_pack["data"], local, 0.05)
        ev = shape._eval_xy(val_pack["floor_xy"], val_pack["bridge_xy"], guarded_xy, val_shape_xy, val_pack["labels"], val_pack["data"], val_pack["bridge_switch"], guarded_switch)
        m = ev["ade_metrics_vs_floor"]
        g = ev["learned_shape_gain_vs_bridge"]
        eligible = (
            int(np.sum(guarded_switch)) > 0
            and m.get("all_improvement", 0.0) > 0.0
            and m.get("t50_improvement", 0.0) > 0.0
            and m.get("hard_failure_improvement", 0.0) > 0.0
            and m.get("easy_degradation", 1.0) <= 0.02
            and ev["collision_delta_vs_floor_005"] <= 0.01
            and max(g["all"], g["t50"], g["t100"], g["hard_failure"]) > 0.0
        )
        rows.append(
            {
                **candidate,
                "guarded_shape_rows": guarded,
                "shape_rows": int(np.sum(guarded_switch)),
                "eligible": bool(eligible),
                "score": _score_val(ev),
                "val_metrics": {"ade": m, "shape_gain_vs_bridge": g, "collision_delta_005": ev["collision_delta_vs_floor_005"]},
            }
        )
    pool = [r for r in rows if r["eligible"]] or rows
    if not pool:
        return {"selected": {"reason": "no_gain_gate_candidate", "eligible": False}, "candidate_count": len(rows), "fast_candidate_count": len(fast_rows), "eligible_count": 0, "top_candidates": []}
    selected = max(pool, key=lambda r: (bool(r["eligible"]), r["score"]))
    return {"selected": selected, "candidate_count": len(rows), "fast_candidate_count": len(fast_rows), "eligible_count": int(sum(r["eligible"] for r in rows)), "top_candidates": sorted(rows, key=lambda r: r["score"], reverse=True)[:10]}


def _apply_policy(pack: Mapping[str, Any], selected: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if selected.get("reason"):
        return pack["bridge_xy"], pack["bridge_switch"], np.zeros(len(pack["bridge_switch"]), dtype=bool)
    mode = str(selected["mode"])
    scale = float(selected["residual_scale"])
    clip = float(selected["residual_clip"])
    shape_xy = shape._shape_xy(pack["data"], pack["endpoint_pred"], pack["shape_pred"], mode, scale, clip)
    pred = _predict_gain_gate(selected["gate"], pack["data"], pack["endpoint_pred"], pack["shape_pred"], pack["bridge_xy"], shape_xy, mode, scale, clip)
    horizons = {"all_horizons": None, "t50_only": {50}, "t100_only": {100}, "long_horizon": {50, 100}}[str(selected["horizons"])]
    allowed = np.ones(len(pack["bridge_switch"]), dtype=bool) if horizons is None else np.isin(pack["data"]["horizon"].astype(int), sorted(horizons))
    local = pack["bridge_switch"] & allowed & (pred["pred_gain"] >= float(selected["gain_min"])) & (pred["pred_harm"] <= float(selected["harm_max"]))
    if np.any(local):
        active = pack["bridge_switch"] & allowed
        max_rows = max(1, int(float(selected["max_shape_rate"]) * int(np.sum(active))))
        ids = np.where(local)[0]
        keep = np.zeros(len(local), dtype=bool)
        keep[ids[np.argsort(pred["pred_gain"][ids])[::-1][:max_rows]]] = True
        local &= keep
    selected_xy = pack["bridge_xy"].copy()
    selected_xy[local] = shape_xy[local]
    selected_xy, shape_switch, _guarded = shape._guard(pack["floor_xy"], selected_xy, pack["bridge_xy"], pack["labels"], pack["data"], local, float(selected.get("min_sep", 0.05)))
    return selected_xy, pack["bridge_switch"], shape_switch


def _make_pack(split_data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], endpoint_pred: Mapping[str, np.ndarray], gate_pred: Mapping[str, np.ndarray], shape_pred: Mapping[str, np.ndarray], bridge_selection: Mapping[str, Any]) -> dict[str, Any]:
    floor_xy, bridge_xy, _selected_xy, bridge_switch, _shape_switch = shape._apply_shape_policy(split_data, labels, endpoint_pred, gate_pred, shape_pred, bridge_selection, {"reason": "bridge_only"})
    return {"data": split_data, "labels": labels, "endpoint_pred": endpoint_pred, "gate_pred": gate_pred, "shape_pred": shape_pred, "floor_xy": floor_xy, "bridge_xy": bridge_xy, "bridge_switch": bridge_switch}


def _domain_data(split: str, domain: str) -> dict[str, np.ndarray]:
    data = dl._load_split(split)
    return bridge._subset(data, dl._domain_mask(data, domain))


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
    train_pack = _make_pack(train, labels_train, pred_train, dl._predict_gate(gate, train, pred_train, fde_train), shape_train, bridge_selection)
    val_pack = _make_pack(val, labels_val, pred_val, gate_val, shape_val, bridge_selection)
    test_pack = _make_pack(test, labels_test, pred_test, gate_test, shape_test, bridge_selection)
    selection = _select_policy_on_val(train_pack, val_pack)
    selected_xy, bridge_switch, shape_switch = _apply_policy(test_pack, selection["selected"])
    mode = str(selection["selected"].get("mode", "intermediate_only"))
    scale = float(selection["selected"].get("residual_scale", 0.25))
    clip = float(selection["selected"].get("residual_clip", 0.04))
    neural_shape_xy = shape._shape_xy(test, pred_test, shape_test, mode, scale, clip)
    ev = shape._eval_xy(test_pack["floor_xy"], test_pack["bridge_xy"], selected_xy, neural_shape_xy, labels_test, test, bridge_switch, shape_switch)
    m = ev["ade_metrics_vs_floor"]
    g = ev["learned_shape_gain_vs_bridge"]
    pass_gate = bool(
        m.get("all_improvement", 0.0) > 0.0
        and m.get("t50_improvement", 0.0) > 0.0
        and m.get("hard_failure_improvement", 0.0) > 0.0
        and m.get("easy_degradation", 1.0) <= 0.02
        and ev["collision_delta_vs_floor_005"] <= 0.01
        and g["shape_switch_rate"] > 0.0
        and max(g["all"], g["t50"], g["t100"], g["hard_failure"]) > 0.0
    )
    return {
        "domain": domain,
        "source": "fresh_run",
        "status": "ok",
        "rows": {"train": int(len(train["horizon"])), "val": int(len(val["horizon"])), "test": int(len(test["horizon"]))},
        "endpoint_training": endpoint_training,
        "shape_training": shape_training,
        "bridge_selection": bridge_selection,
        "gain_gate_selection": selection,
        "bridge_direct_endpoint_without_fallback_test": dl._metrics(fde_test, test["floor_fde"], test, np.ones(len(test["horizon"]), dtype=bool)),
        **ev,
        "learned_shape_gain_gate_pass": pass_gate,
        "caveat": "Train-fitted shape gain/harm gate uses future waypoint labels only during training/validation. Inference remains past-only and protected by endpoint bridge/floor fallback. Dataset-local raw-frame 2.5D only; no Stage5C/SMC/metric/seconds claim.",
    }


def run_learned_shape_gain_gate() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ft.build_full_trajectory_labels()
    domains = ["ETH_UCY", "TrajNet"]
    results = {domain: _evaluate_domain(domain) for domain in domains}
    positive = [d for d, r in results.items() if r.get("learned_shape_gain_gate_pass")]
    previous_path = OUT_DIR / "stage41_learned_waypoint_shape_bridge.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else {}
    deltas: dict[str, Any] = {}
    for domain, row in results.items():
        if row.get("status") != "ok":
            continue
        old = (previous.get("domain_results") or {}).get(domain, {})
        old_g = old.get("learned_shape_gain_vs_bridge") or {}
        g = row["learned_shape_gain_vs_bridge"]
        deltas[domain] = {k: float(g.get(k, 0.0) - old_g.get(k, 0.0)) for k in ["all", "t50", "t100", "hard_failure", "shape_switch_rate"]}
    result = {
        "source": "fresh_run",
        "protocol": "learned_shape_gain_harm_gate",
        "stage5c_executed": False,
        "smc_enabled": False,
        "metric_or_seconds_claim": False,
        "positive_domains": positive,
        "positive_domain_count": len(positive),
        "two_domain_gain_gate": len(positive) >= 2,
        "delta_vs_previous_learned_shape": deltas,
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
            "learned_shape_gain_harm_gate": True,
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
        "# Stage41 Learned Shape Gain/Harm Gate",
        "",
        "- source: `fresh_run`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        f"- positive domains: `{positive}`",
        f"- two-domain gain gate: `{result['two_domain_gain_gate']}`",
        "",
        "| domain | all ADE | t50 ADE | t100 ADE | hard ADE | easy | shape gain all/t50/t100/hard | shape switch | delta shape all/t100 | pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for domain, row in results.items():
        if row.get("status") != "ok":
            lines.append(f"| `{domain}` | 0 | 0 | 0 | 0 | 0 | 0/0/0/0 | 0 | `{row.get('reason')}` | `False` |")
            continue
        m = row["ade_metrics_vs_floor"]
        g = row["learned_shape_gain_vs_bridge"]
        d = deltas.get(domain, {})
        lines.append(
            f"| `{domain}` | {m.get('all_improvement', 0.0):.4f} | {m.get('t50_improvement', 0.0):.4f} | {m.get('t100_improvement', 0.0):.4f} | "
            f"{m.get('hard_failure_improvement', 0.0):.4f} | {m.get('easy_degradation', 0.0):.4f} | "
            f"{g.get('all', 0.0):.6f}/{g.get('t50', 0.0):.6f}/{g.get('t100', 0.0):.6f}/{g.get('hard_failure', 0.0):.6f} | "
            f"{g.get('shape_switch_rate', 0.0):.6f} | {d.get('all', 0.0):.6f}/{d.get('t100', 0.0):.6f} | `{row.get('learned_shape_gain_gate_pass')}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This experiment replaces the previous residual-norm heuristic with a train-fitted gain/harm gate for learned waypoint-shape interventions.",
            "- Future waypoint labels are used only to train/evaluate the gain/harm gate; inference inputs remain past-only neural predictions and causal features.",
            "- A positive delta versus the previous learned-shape bridge is needed before calling this a stronger shape-dynamics contribution.",
            f"- no leakage: `{result['no_leakage']}`",
            f"- claim boundary: `{result['claim_boundary']}`",
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_learned_shape_gain_gate() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_learned_shape_gain_gate()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_learned_shape_gain_gate",
            status,
            started,
            [dl.DATA_DIR / "seq2seq_train.npz", dl.DATA_DIR / "seq2seq_val.npz", dl.DATA_DIR / "seq2seq_test.npz", ft.DATA_DIR / "full_trajectory_test.npz"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_learned_shape_gain_gate()
