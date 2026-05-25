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
from src import stage41_dynamic_shape_meta_policy as meta


OUT_DIR = dl.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_pairwise_shape_switch_policy.json"
REPORT_MD = OUT_DIR / "stage41_pairwise_shape_switch_policy.md"
NON_BRIDGE_SOURCES = ("old_shape", "gain_gate")
SOURCES = meta.SOURCES
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


def _domain_data(split: str, domain: str) -> dict[str, np.ndarray]:
    data = dl._load_split(split)
    return bridge._subset(data, dl._domain_mask(data, domain))


def _gain_labels(pack: Mapping[str, Any], source: str) -> np.ndarray:
    bridge_ade = ft._trajectory_errors(pack["bridge_xy"], pack["labels"])[0]
    source_ade = ft._trajectory_errors(meta._source_xy(pack, source), pack["labels"])[0]
    return np.nan_to_num(bridge_ade - source_ade, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)


def _fit_pairwise_models(pack: Mapping[str, Any], lam: float = 0.12) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for source in NON_BRIDGE_SOURCES:
        x_raw = meta._source_feature_matrix(pack, source)
        gain = _gain_labels(pack, source)
        clip = float(np.quantile(np.abs(gain), 0.995)) if len(gain) else 0.0
        if clip > EPS:
            gain = np.clip(gain, -clip, clip)
        harm = np.maximum(-gain, 0.0)
        mean, std = meta._fit_standardizer(x_raw)
        x = meta._standardize(x_raw, mean, std)
        models[source] = {
            "mean": mean,
            "std": std,
            "w_gain": meta._ridge_fit(x, gain, lam=lam),
            "w_harm": meta._ridge_fit(x, harm, lam=lam),
            "feature_dim": int(x_raw.shape[1]),
            "positive_rate": float(np.mean(gain > 0.0)) if len(gain) else 0.0,
            "gain_clip": clip,
        }
    return {"type": "pairwise_bridge_switch_gain_harm_ridge", "sources": models}


def _predict_pairwise_models(model: Mapping[str, Any], pack: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    pred: dict[str, dict[str, np.ndarray]] = {}
    for source in NON_BRIDGE_SOURCES:
        m = model["sources"][source]
        x_raw = meta._source_feature_matrix(pack, source)
        x = meta._standardize(x_raw, np.asarray(m["mean"]), np.asarray(m["std"]))
        gain = meta._ridge_predict(x, np.asarray(m["w_gain"]))
        harm = np.maximum(meta._ridge_predict(x, np.asarray(m["w_harm"])), 0.0)
        pred[source] = {
            "gain": np.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64),
            "harm": np.nan_to_num(harm, nan=1.0e6, posinf=1.0e6, neginf=1.0e6).astype(np.float64),
        }
    return pred


def _pairwise_signal_metrics(pack: Mapping[str, Any], pred: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    per_source: dict[str, Any] = {}
    signed_hits = []
    for source in NON_BRIDGE_SOURCES:
        gain = _gain_labels(pack, source)
        predicted_gain = pred[source]["gain"]
        sign_acc = float(np.mean((gain > 0.0) == (predicted_gain > 0.0))) if len(gain) else 0.0
        corr = float(np.corrcoef(gain, predicted_gain)[0, 1]) if len(gain) > 2 and np.std(gain) > EPS and np.std(predicted_gain) > EPS else 0.0
        per_source[source] = {
            "true_positive_rate": float(np.mean(gain > 0.0)) if len(gain) else 0.0,
            "pred_positive_rate": float(np.mean(predicted_gain > 0.0)) if len(gain) else 0.0,
            "sign_accuracy": sign_acc,
            "gain_correlation": corr,
        }
        signed_hits.append(sign_acc)
    return {"per_source": per_source, "mean_sign_accuracy": float(np.mean(signed_hits)) if signed_hits else 0.0}


def _candidate_arrays(pack: Mapping[str, Any], pred: Mapping[str, Mapping[str, np.ndarray]], policy: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(pack["horizon"])
    score_stack = np.zeros((n, len(SOURCES)), dtype=np.float64)
    gain_stack = np.zeros_like(score_stack)
    harm_stack = np.zeros_like(score_stack)
    switch_stack = np.zeros((n, len(SOURCES)), dtype=bool)
    for source in NON_BRIDGE_SOURCES:
        i = SOURCES.index(source)
        gain = pred[source]["gain"]
        harm = pred[source]["harm"]
        score_stack[:, i] = gain - float(policy["harm_weight"]) * harm
        gain_stack[:, i] = gain
        harm_stack[:, i] = harm
        switch_stack[:, i] = meta._source_switch(pack, source)
    return score_stack, gain_stack, harm_stack, switch_stack


def _choose_pairwise_sources(pack: Mapping[str, Any], pred: Mapping[str, Mapping[str, np.ndarray]], policy: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_stack, gain_stack, harm_stack, switch_stack = _candidate_arrays(pack, pred, policy)
    best_idx = np.argmax(score_stack, axis=1)
    ordered = np.sort(score_stack, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    row = np.arange(len(best_idx))
    best_gain = gain_stack[row, best_idx]
    best_harm = harm_stack[row, best_idx]
    best_score = score_stack[row, best_idx]
    allowed = (
        (best_idx != SOURCES.index("bridge"))
        & switch_stack[row, best_idx]
        & (best_gain >= float(policy["gain_min"]))
        & (best_harm <= float(policy["harm_max"]))
        & (best_score >= float(policy["score_min"]))
        & (margin >= float(policy["margin_min"]))
    )
    horizon = pack["horizon"].astype(int)
    for horizon_value in [10, 25, 50, 100]:
        horizon_mask = horizon == horizon_value
        active = np.where(allowed & horizon_mask)[0]
        max_rows = int(float(policy[f"max_rate_h{horizon_value}"]) * max(1, int(np.sum(horizon_mask))))
        if len(active) == 0:
            continue
        if max_rows <= 0:
            allowed[active] = False
            continue
        if len(active) > max_rows:
            keep_ids = active[np.argsort(best_score[active])[::-1][:max_rows]]
            drop = np.setdiff1d(active, keep_ids, assume_unique=False)
            allowed[drop] = False
    best_idx[~allowed] = SOURCES.index("bridge")
    selected = pack["bridge_xy"].copy()
    shape_switch = np.zeros(len(best_idx), dtype=bool)
    for source in NON_BRIDGE_SOURCES:
        i = SOURCES.index(source)
        mask = best_idx == i
        if not np.any(mask):
            continue
        selected[mask] = meta._source_xy(pack, source)[mask]
        shape_switch[mask] = meta._source_switch(pack, source)[mask]
    return selected, shape_switch, best_idx


def _fast_eval_policy(pack: Mapping[str, Any], xy: np.ndarray, shape_switch: np.ndarray) -> dict[str, Any]:
    labels = pack["labels"]
    floor_ade = ft._trajectory_errors(pack["floor_xy"], labels)[0]
    bridge_ade = ft._trajectory_errors(pack["bridge_xy"], labels)[0]
    selected_ade = ft._trajectory_errors(xy, labels)[0]
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


def _policy_grid(pred: Mapping[str, Mapping[str, np.ndarray]]) -> list[dict[str, float]]:
    gains = np.concatenate([pred[source]["gain"] for source in NON_BRIDGE_SOURCES])
    harms = np.concatenate([pred[source]["harm"] for source in NON_BRIDGE_SOURCES])
    positive_gains = gains[gains > 0.0]
    finite_harms = harms[np.isfinite(harms) & (harms >= 0.0)]
    gain_grid = [0.0]
    score_grid = [0.0]
    margin_grid = [0.0]
    harm_grid = [1.0e6]
    if len(positive_gains):
        gain_grid.extend(float(v) for v in np.quantile(positive_gains, [0.55, 0.75]))
        score_grid.extend(float(v) for v in np.quantile(positive_gains, [0.55]))
        margin_grid.extend(float(v) for v in np.quantile(positive_gains, [0.50]))
    if len(finite_harms):
        harm_grid.extend(float(v) for v in np.quantile(finite_harms, [0.35]))
    rows = []
    for harm_weight in [0.0, 1.0]:
        for gain_min in gain_grid:
            for score_min in score_grid:
                for margin_min in margin_grid:
                    for harm_max in harm_grid:
                        for short_rate in [0.0, 0.005]:
                            for t50_rate in [0.005, 0.02, 0.05]:
                                for t100_rate in [0.005, 0.02, 0.05]:
                                    rows.append(
                                        {
                                            "harm_weight": harm_weight,
                                            "gain_min": gain_min,
                                            "score_min": score_min,
                                            "margin_min": margin_min,
                                            "harm_max": harm_max,
                                            "max_rate_h10": short_rate,
                                            "max_rate_h25": short_rate,
                                            "max_rate_h50": t50_rate,
                                            "max_rate_h100": t100_rate,
                                        }
                                    )
    return rows


def _select_pairwise_policy_on_val(model: Mapping[str, Any], pack: Mapping[str, Any]) -> dict[str, Any]:
    pred = _predict_pairwise_models(model, pack)
    fast_rows: list[dict[str, Any]] = []
    for policy in _policy_grid(pred):
        xy, shape_switch, chosen = _choose_pairwise_sources(pack, pred, policy)
        ev = _fast_eval_policy(pack, xy, shape_switch)
        m = ev["ade_metrics_vs_floor"]
        g = ev["shape_gain_vs_bridge"]
        non_bridge = chosen != SOURCES.index("bridge")
        score = float(
            m.get("all_improvement", 0.0)
            + 1.8 * m.get("t50_improvement", 0.0)
            + 1.3 * m.get("t100_improvement", 0.0)
            + 1.4 * m.get("hard_failure_improvement", 0.0)
            + 4.0 * max(g["all"], g["t50"], g["t100"], g["hard_failure"])
            - 45.0 * max(0.0, m.get("easy_degradation", 1.0) - 0.02)
        )
        fast_rows.append(
            {
                "policy": policy,
                "score": score,
                "has_switch": bool(np.any(non_bridge)),
                "fast_metrics": {
                    "ade": m,
                    "shape_gain_vs_bridge": g,
                    "source_distribution": meta._source_distribution(chosen),
                },
            }
        )
    rows: list[dict[str, Any]] = []
    for row in sorted(fast_rows, key=lambda r: r["score"], reverse=True)[:64]:
        xy, shape_switch, chosen = _choose_pairwise_sources(pack, pred, row["policy"])
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
                    "source_distribution": meta._source_distribution(chosen),
                },
            }
        )
    pool = [row for row in rows if row["eligible"]] or rows or fast_rows
    selected = max(pool, key=lambda r: (bool(r.get("eligible", False)), r["score"]))
    return {
        "selected": selected,
        "candidate_count": len(rows),
        "fast_candidate_count": len(fast_rows),
        "eligible_count": int(sum(bool(row.get("eligible", False)) for row in rows)),
        "top_candidates": sorted(rows, key=lambda r: r["score"], reverse=True)[:10],
        "pairwise_signal": _pairwise_signal_metrics(pack, pred),
    }


def _build_domain_packs(domain: str) -> dict[str, Any]:
    train = _domain_data("train", domain)
    val = _domain_data("val", domain)
    test = _domain_data("test", domain)
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
    return {
        "rows": {"train": int(len(train["horizon"])), "val": int(len(val["horizon"])), "test": int(len(test["horizon"]))},
        "endpoint_training": endpoint_training,
        "shape_training": shape_training,
        "bridge_selection": meta._source_policy_summary(bridge_selection),
        "old_shape_selection": meta._source_policy_summary(old_selection),
        "gain_gate_selection": meta._source_policy_summary(gain_selection),
        "train_pack": meta._make_meta_pack(train, labels_train, pred_train, gate_train, shape_train, bridge_selection, old_selection["selected"], gain_selection["selected"]),
        "val_pack": meta._make_meta_pack(val, labels_val, pred_val, gate_val, shape_val, bridge_selection, old_selection["selected"], gain_selection["selected"]),
        "test_pack": meta._make_meta_pack(test, labels_test, pred_test, gate_test, shape_test, bridge_selection, old_selection["selected"], gain_selection["selected"]),
    }


def _evaluate_domain(domain: str) -> dict[str, Any]:
    train = _domain_data("train", domain)
    val = _domain_data("val", domain)
    test = _domain_data("test", domain)
    if min(len(train["horizon"]), len(val["horizon"]), len(test["horizon"])) < 500:
        return {"domain": domain, "status": "not_run", "reason": "not enough domain rows"}
    packs = _build_domain_packs(domain)
    model = _fit_pairwise_models(packs["train_pack"])
    selection = _select_pairwise_policy_on_val(model, packs["val_pack"])
    pred_test = _predict_pairwise_models(model, packs["test_pack"])
    selected_xy, shape_switch, chosen = _choose_pairwise_sources(packs["test_pack"], pred_test, selection["selected"]["policy"])
    ev = composer._eval_selected(packs["test_pack"], selected_xy, shape_switch)
    compact = meta._compact_eval(ev, chosen)
    fixed_selection = composer._select_composer_on_val(packs["val_pack"])
    fixed_xy, fixed_shape_switch = composer._compose_sources(packs["test_pack"], fixed_selection["selected"]["policy"])
    fixed_eval = composer._eval_selected(packs["test_pack"], fixed_xy, fixed_shape_switch)
    fixed_chosen = meta._chosen_from_fixed_policy(packs["test_pack"], fixed_selection["selected"]["policy"])
    fixed_compact = meta._compact_eval(fixed_eval, fixed_chosen)
    m = ev["ade_metrics_vs_floor"]
    g = ev["shape_gain_vs_bridge"]
    pass_gate = bool(
        m.get("all_improvement", 0.0) > 0.0
        and m.get("t50_improvement", 0.0) > 0.0
        and m.get("hard_failure_improvement", 0.0) > 0.0
        and m.get("easy_degradation", 1.0) <= 0.02
        and ev["collision_delta_vs_floor_005"] <= 0.01
        and compact["shape_switch_rate"] > 0.0
        and max(g["all"], g["t50"], g["t100"], g["hard_failure"]) >= 0.0
    )
    return {
        "domain": domain,
        "source": "fresh_run",
        "status": "ok",
        "rows": packs["rows"],
        "endpoint_training": packs["endpoint_training"],
        "shape_training": packs["shape_training"],
        "bridge_selection": packs["bridge_selection"],
        "old_shape_selection": packs["old_shape_selection"],
        "gain_gate_selection": packs["gain_gate_selection"],
        "pairwise_model": {
            "type": model["type"],
            "sources": {
                source: {
                    "feature_dim": int(model["sources"][source]["feature_dim"]),
                    "positive_rate": float(model["sources"][source]["positive_rate"]),
                    "gain_clip": float(model["sources"][source]["gain_clip"]),
                }
                for source in NON_BRIDGE_SOURCES
            },
        },
        "pairwise_selection": selection,
        "pairwise_metrics": ev,
        "pairwise_compact": compact,
        "fixed_horizon_composer_compact": fixed_compact,
        "delta_vs_fixed": {
            key: float(compact[key] - fixed_compact[key])
            for key in ["all", "t50", "t100", "hard_failure", "easy_degradation", "shape_gain_all", "shape_gain_t50", "shape_gain_t100", "shape_gain_hard_failure"]
        },
        "test_pairwise_signal": _pairwise_signal_metrics(packs["test_pack"], pred_test),
        "pairwise_pass": pass_gate,
        "caveat": "Pairwise switch policy learns source gain/harm relative to the Stage37/bridge floor from past-only features and candidate rollout geometry. Future waypoints are labels/eval only; validation selects safety thresholds; test is evaluated once. Dataset-local raw-frame 2.5D only; no Stage5C/SMC/metric/seconds/true-3D/foundation claim.",
    }


def run_pairwise_shape_switch_policy() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ft.build_full_trajectory_labels()
    domains = ["ETH_UCY", "TrajNet"]
    results = {domain: _evaluate_domain(domain) for domain in domains}
    positive = [domain for domain, row in results.items() if row.get("pairwise_pass")]
    better_than_fixed = [
        domain
        for domain, row in results.items()
        if row.get("status") == "ok"
        and (
            row["delta_vs_fixed"]["all"] > 0.0
            or row["delta_vs_fixed"]["t50"] > 0.0
            or row["delta_vs_fixed"]["hard_failure"] > 0.0
        )
    ]
    result = {
        "source": "fresh_run",
        "protocol": "pairwise_gain_harm_shape_switch_policy",
        "stage5c_executed": False,
        "smc_enabled": False,
        "metric_or_seconds_claim": False,
        "positive_domains": positive,
        "positive_domain_count": len(positive),
        "two_domain_pairwise_gate": len(positive) >= 2,
        "domains_better_than_fixed_on_any_core_metric": better_than_fixed,
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
            "pairwise_gain_harm_source_switch": True,
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
        "# Stage41 Pairwise Gain/Harm Shape Switch Policy",
        "",
        "- source: `fresh_run`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        f"- positive domains: `{positive}`",
        f"- two-domain pairwise gate: `{result['two_domain_pairwise_gate']}`",
        f"- better than fixed composer on any core metric: `{better_than_fixed}`",
        "",
        "| domain | all ADE | t50 ADE | t100 ADE | hard ADE | easy | shape gain all/t50/t100/hard | source distribution | sign acc | fixed delta all/t50/t100/hard | pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- |",
    ]
    for domain, row in results.items():
        if row.get("status") != "ok":
            lines.append(f"| `{domain}` | 0 | 0 | 0 | 0 | 0 | 0/0/0/0 | not_run | 0 | 0/0/0/0 | `False` |")
            continue
        c = row["pairwise_compact"]
        d = row["delta_vs_fixed"]
        lines.append(
            f"| `{domain}` | {c['all']:.4f} | {c['t50']:.4f} | {c['t100']:.4f} | {c['hard_failure']:.4f} | {c['easy_degradation']:.4f} | "
            f"{c['shape_gain_all']:.6f}/{c['shape_gain_t50']:.6f}/{c['shape_gain_t100']:.6f}/{c['shape_gain_hard_failure']:.6f} | "
            f"`{c['source_distribution']}` | {row['test_pairwise_signal']['mean_sign_accuracy']:.4f} | "
            f"{d['all']:.6f}/{d['t50']:.6f}/{d['t100']:.6f}/{d['hard_failure']:.6f} | `{row.get('pairwise_pass')}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This experiment is the direct repair attempt after absolute-ADE dynamic source ranking and simple calibration failed on ETH_UCY.",
            "- It predicts pairwise gain and harm for switching from the protected bridge/Stage37 floor into each learned shape source.",
            "- Validation selects conservative gain, harm, margin, and per-horizon switch-rate thresholds; test is evaluated once.",
            "- If pairwise switching does not beat the fixed horizon composer, the fixed composer remains the safer deployable policy.",
            "- This is still protected 2.5D dataset-local raw-frame evidence, not Stage5C/SMC/metric/seconds/true-3D/foundation evidence.",
            f"- no leakage: `{result['no_leakage']}`",
            f"- claim boundary: `{result['claim_boundary']}`",
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_pairwise_shape_switch_policy() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_pairwise_shape_switch_policy()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_pairwise_shape_switch_policy",
            status,
            started,
            [dl.DATA_DIR / "seq2seq_train.npz", dl.DATA_DIR / "seq2seq_val.npz", dl.DATA_DIR / "seq2seq_test.npz", ft.DATA_DIR / "full_trajectory_test.npz"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_pairwise_shape_switch_policy()
