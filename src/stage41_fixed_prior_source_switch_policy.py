from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_domain_local_neural_retrain as dl
from src import stage41_full_trajectory_world_state as ft
from src import stage41_shape_policy_composer as composer
from src import stage41_dynamic_shape_meta_policy as meta
from src import stage41_pairwise_shape_switch_policy as pairwise
from src import stage41_weighted_pairwise_shape_switch_policy as weighted


OUT_DIR = dl.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_fixed_prior_source_switch_policy.json"
REPORT_MD = OUT_DIR / "stage41_fixed_prior_source_switch_policy.md"
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


def _fixed_chosen(pack: Mapping[str, Any], fixed_policy: Mapping[str, str]) -> np.ndarray:
    return meta._chosen_from_fixed_policy(pack, fixed_policy)


def _xy_from_chosen(pack: Mapping[str, Any], chosen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = pack["bridge_xy"].copy()
    shape_switch = np.zeros(len(chosen), dtype=bool)
    for i, source in enumerate(SOURCES):
        mask = chosen == i
        if source == "bridge" or not np.any(mask):
            continue
        selected[mask] = meta._source_xy(pack, source)[mask]
        shape_switch[mask] = meta._source_switch(pack, source)[mask]
    return selected, shape_switch


def _source_cost(pack: Mapping[str, Any], source: str) -> np.ndarray:
    return ft._trajectory_errors(meta._source_xy(pack, source), pack["labels"])[0]


def _fixed_cost(pack: Mapping[str, Any], fixed_policy: Mapping[str, str]) -> np.ndarray:
    fixed_xy, _fixed_switch = composer._compose_sources(pack, fixed_policy)
    return ft._trajectory_errors(fixed_xy, pack["labels"])[0]


def _fixed_prior_feature_matrix(pack: Mapping[str, Any], source: str, fixed_policy: Mapping[str, str]) -> np.ndarray:
    data = pack["data"]
    fixed_chosen = _fixed_chosen(pack, fixed_policy)
    fixed_xy, fixed_switch = _xy_from_chosen(pack, fixed_chosen)
    candidate_xy = meta._source_xy(pack, source).astype(np.float64)
    norm = np.maximum(data["normalizer"].astype(np.float64), EPS)
    current = data["current_xy"].astype(np.float64)
    candidate_delta = (candidate_xy[:, -1, :] - current) / norm[:, None]
    fixed_delta = (fixed_xy[:, -1, :] - current) / norm[:, None]
    delta_gap = candidate_delta - fixed_delta
    waypoint_gap = np.linalg.norm(candidate_xy - fixed_xy, axis=2) / norm[:, None]
    candidate_step = np.linalg.norm(np.diff(candidate_xy, axis=1), axis=2) / norm[:, None]
    fixed_step = np.linalg.norm(np.diff(fixed_xy, axis=1), axis=2) / norm[:, None]
    fixed_onehot = np.zeros((len(fixed_chosen), len(SOURCES)), dtype=np.float64)
    fixed_onehot[np.arange(len(fixed_chosen)), fixed_chosen] = 1.0
    candidate_onehot = np.zeros((len(fixed_chosen), len(SOURCES)), dtype=np.float64)
    candidate_onehot[:, SOURCES.index(source)] = 1.0
    extra = np.column_stack(
        [
            fixed_delta,
            candidate_delta,
            delta_gap,
            np.linalg.norm(delta_gap, axis=1),
            waypoint_gap.mean(axis=1),
            waypoint_gap.max(axis=1),
            (candidate_step - fixed_step).sum(axis=1),
            (candidate_step - fixed_step).max(axis=1),
            meta._source_switch(pack, source).astype(np.float64),
            fixed_switch.astype(np.float64),
            (fixed_chosen == SOURCES.index(source)).astype(np.float64),
        ]
    )
    return np.nan_to_num(
        np.concatenate([meta._source_feature_matrix(pack, source), fixed_onehot, candidate_onehot, extra], axis=1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float64)


def _fixed_prior_gain_labels(pack: Mapping[str, Any], source: str, fixed_policy: Mapping[str, str]) -> np.ndarray:
    fixed = _fixed_cost(pack, fixed_policy)
    candidate = _source_cost(pack, source)
    return np.nan_to_num(fixed - candidate, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)


def _training_weight(pack: Mapping[str, Any], source: str, fixed_policy: Mapping[str, str], gain: np.ndarray) -> np.ndarray:
    labels = pack["labels"]
    horizon = labels["horizon"].astype(int)
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    positive = gain > 0.0
    pos = max(1, int(np.sum(positive)))
    neg = max(1, int(np.sum(~positive)))
    positive_boost = min(60.0, max(3.0, neg / pos))
    fixed_chosen = _fixed_chosen(pack, fixed_policy)
    candidate_is_fixed = fixed_chosen == SOURCES.index(source)
    weight = np.ones(len(gain), dtype=np.float64)
    weight += 2.0 * hard.astype(np.float64)
    weight += 2.0 * (horizon == 50).astype(np.float64)
    weight += 1.5 * (horizon == 100).astype(np.float64)
    weight += positive_boost * positive.astype(np.float64)
    weight += 0.5 * (~candidate_is_fixed).astype(np.float64)
    return np.nan_to_num(weight, nan=1.0, posinf=1.0, neginf=1.0)


def _fit_fixed_prior_model(pack: Mapping[str, Any], fixed_policy: Mapping[str, str], lam: float = 0.12) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for source in SOURCES:
        x_raw = _fixed_prior_feature_matrix(pack, source, fixed_policy)
        gain = _fixed_prior_gain_labels(pack, source, fixed_policy)
        clip = float(np.quantile(np.abs(gain), 0.995)) if len(gain) else 0.0
        if clip > EPS:
            gain = np.clip(gain, -clip, clip)
        harm = np.maximum(-gain, 0.0)
        weight = _training_weight(pack, source, fixed_policy, gain)
        mean, std = meta._fit_standardizer(x_raw)
        x = meta._standardize(x_raw, mean, std)
        models[source] = {
            "mean": mean,
            "std": std,
            "w_gain": weighted._weighted_ridge_fit(x, gain, weight, lam=lam),
            "w_harm": weighted._weighted_ridge_fit(x, harm, weight, lam=lam),
            "feature_dim": int(x_raw.shape[1]),
            "positive_rate": float(np.mean(gain > 0.0)) if len(gain) else 0.0,
            "mean_weight": float(np.mean(weight)) if len(weight) else 0.0,
            "gain_clip": clip,
        }
    return {"type": "fixed_composer_prior_pairwise_switch", "fixed_policy": dict(fixed_policy), "sources": models}


def _predict_fixed_prior_model(model: Mapping[str, Any], pack: Mapping[str, Any], fixed_policy: Mapping[str, str]) -> dict[str, dict[str, np.ndarray]]:
    pred: dict[str, dict[str, np.ndarray]] = {}
    for source in SOURCES:
        m = model["sources"][source]
        x_raw = _fixed_prior_feature_matrix(pack, source, fixed_policy)
        x = meta._standardize(x_raw, np.asarray(m["mean"]), np.asarray(m["std"]))
        pred[source] = {
            "gain": np.nan_to_num(meta._ridge_predict(x, np.asarray(m["w_gain"])), nan=0.0, posinf=0.0, neginf=0.0),
            "harm": np.maximum(np.nan_to_num(meta._ridge_predict(x, np.asarray(m["w_harm"])), nan=1.0e6, posinf=1.0e6, neginf=1.0e6), 0.0),
        }
    return pred


def _choose_sources(pack: Mapping[str, Any], pred: Mapping[str, Mapping[str, np.ndarray]], fixed_policy: Mapping[str, str], policy: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fixed_chosen = _fixed_chosen(pack, fixed_policy)
    score_stack = np.column_stack([pred[source]["gain"] - float(policy["harm_weight"]) * pred[source]["harm"] for source in SOURCES])
    gain_stack = np.column_stack([pred[source]["gain"] for source in SOURCES])
    harm_stack = np.column_stack([pred[source]["harm"] for source in SOURCES])
    row = np.arange(len(fixed_chosen))
    score_stack[row, fixed_chosen] = 0.0
    best_idx = np.argmax(score_stack, axis=1)
    best_score = score_stack[row, best_idx]
    best_gain = gain_stack[row, best_idx]
    best_harm = harm_stack[row, best_idx]
    sorted_score = np.sort(score_stack, axis=1)
    margin = sorted_score[:, -1] - sorted_score[:, -2]
    allowed = (
        (best_idx != fixed_chosen)
        & (best_gain >= float(policy["gain_min"]))
        & (best_harm <= float(policy["harm_max"]))
        & (best_score >= float(policy["score_min"]))
        & (margin >= float(policy["margin_min"]))
    )
    for source in ["old_shape", "gain_gate"]:
        source_idx = SOURCES.index(source)
        source_mask = best_idx == source_idx
        allowed[source_mask] &= meta._source_switch(pack, source)[source_mask]
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
    chosen = fixed_chosen.copy()
    chosen[allowed] = best_idx[allowed]
    xy, shape_switch = _xy_from_chosen(pack, chosen)
    return xy, shape_switch, chosen


def _policy_grid(pred: Mapping[str, Mapping[str, np.ndarray]]) -> list[dict[str, float]]:
    gains = np.concatenate([pred[source]["gain"] for source in SOURCES])
    harms = np.concatenate([pred[source]["harm"] for source in SOURCES])
    positive = gains[gains > 0.0]
    finite_harm = harms[np.isfinite(harms) & (harms >= 0.0)]
    gain_grid = [0.0]
    score_grid = [0.0]
    margin_grid = [0.0]
    harm_grid = [1.0e6]
    if len(positive):
        gain_grid.extend(float(v) for v in np.quantile(positive, [0.55, 0.75]))
        score_grid.extend(float(v) for v in np.quantile(positive, [0.55]))
        margin_grid.extend(float(v) for v in np.quantile(positive, [0.50]))
    if len(finite_harm):
        harm_grid.extend(float(v) for v in np.quantile(finite_harm, [0.35]))
    rows = []
    for harm_weight in [0.0, 1.0]:
        for gain_min in gain_grid:
            for score_min in score_grid:
                for margin_min in margin_grid:
                    for harm_max in harm_grid:
                        for short_rate in [0.0, 0.002, 0.005]:
                            for t50_rate in [0.0, 0.005, 0.02, 0.05]:
                                for t100_rate in [0.0, 0.005, 0.02, 0.05]:
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


def _source_distribution(chosen: np.ndarray) -> dict[str, float]:
    return {source: float(np.mean(chosen == i)) if len(chosen) else 0.0 for i, source in enumerate(SOURCES)}


def _compact(ev: Mapping[str, Any], chosen: np.ndarray, fixed_chosen: np.ndarray) -> dict[str, Any]:
    c = meta._compact_eval(ev, chosen)
    c["source_switch_rate"] = float(np.mean(chosen != fixed_chosen)) if len(chosen) else 0.0
    c["source_distribution"] = _source_distribution(chosen)
    return c


def _select_policy_on_val(model: Mapping[str, Any], pack: Mapping[str, Any], fixed_policy: Mapping[str, str]) -> dict[str, Any]:
    pred = _predict_fixed_prior_model(model, pack, fixed_policy)
    fixed_chosen = _fixed_chosen(pack, fixed_policy)
    fixed_xy, fixed_shape_switch = composer._compose_sources(pack, fixed_policy)
    fixed_fast = pairwise._fast_eval_policy(pack, fixed_xy, fixed_shape_switch)
    fixed_full = composer._eval_selected(pack, fixed_xy, fixed_shape_switch)
    fixed_full_compact = _compact(fixed_full, fixed_chosen, fixed_chosen)
    fast_rows = []
    for policy in _policy_grid(pred):
        xy, shape_switch, chosen = _choose_sources(pack, pred, fixed_policy, policy)
        ev = pairwise._fast_eval_policy(pack, xy, shape_switch)
        m = ev["ade_metrics_vs_floor"]
        g = ev["shape_gain_vs_bridge"]
        fm = fixed_fast["ade_metrics_vs_floor"]
        source_switch_rate = float(np.mean(chosen != fixed_chosen)) if len(chosen) else 0.0
        delta_all = m.get("all_improvement", 0.0) - fm.get("all_improvement", 0.0)
        delta_t50 = m.get("t50_improvement", 0.0) - fm.get("t50_improvement", 0.0)
        delta_t100 = m.get("t100_improvement", 0.0) - fm.get("t100_improvement", 0.0)
        delta_hard = m.get("hard_failure_improvement", 0.0) - fm.get("hard_failure_improvement", 0.0)
        score = float(
            m.get("all_improvement", 0.0)
            + 1.8 * m.get("t50_improvement", 0.0)
            + 1.3 * m.get("t100_improvement", 0.0)
            + 1.4 * m.get("hard_failure_improvement", 0.0)
            + 4.0 * max(g["all"], g["t50"], g["t100"], g["hard_failure"])
            + 10.0 * max(delta_all, delta_hard, delta_t100, 0.0)
            + 2.0 * source_switch_rate
            - 20.0 * max(0.0, -delta_t50)
            - 45.0 * max(0.0, m.get("easy_degradation", 1.0) - 0.02)
        )
        fast_rows.append({"policy": policy, "score": score, "source_switch_rate": source_switch_rate})
    rows = []
    for row in sorted(fast_rows, key=lambda r: r["score"], reverse=True)[:64]:
        xy, shape_switch, chosen = _choose_sources(pack, pred, fixed_policy, row["policy"])
        ev = composer._eval_selected(pack, xy, shape_switch)
        m = ev["ade_metrics_vs_floor"]
        g = ev["shape_gain_vs_bridge"]
        compact = _compact(ev, chosen, fixed_chosen)
        delta = {key: float(compact[key] - fixed_full_compact[key]) for key in ["all", "t50", "t100", "hard_failure", "easy_degradation", "shape_gain_all", "shape_gain_t50", "shape_gain_t100", "shape_gain_hard_failure"]}
        eligible = (
            compact["source_switch_rate"] > 0.0
            and m.get("all_improvement", 0.0) > 0.0
            and m.get("t50_improvement", 0.0) > 0.0
            and m.get("hard_failure_improvement", 0.0) > 0.0
            and m.get("easy_degradation", 1.0) <= 0.02
            and ev["collision_delta_vs_floor_005"] <= 0.01
            and max(g["all"], g["t50"], g["t100"], g["hard_failure"]) >= -0.0005
            and delta["t50"] >= -1.0e-12
            and (delta["all"] > 0.0 or delta["hard_failure"] > 0.0 or delta["t100"] > 0.0)
        )
        rows.append(
            {
                "policy": row["policy"],
                "eligible": bool(eligible),
                "score": composer._score_eval(ev)
                + 20.0 * max(delta["all"], delta["hard_failure"], delta["t100"], 0.0)
                + 2.0 * compact["source_switch_rate"]
                - 40.0 * max(0.0, -delta["t50"]),
                "val_metrics": {
                    "ade": m,
                    "shape_gain_vs_bridge": g,
                    "collision_delta_005": ev["collision_delta_vs_floor_005"],
                    "source_switch_rate": compact["source_switch_rate"],
                    "source_distribution": compact["source_distribution"],
                    "delta_vs_fixed": delta,
                },
            }
        )
    eligible_pool = [row for row in rows if row["eligible"]]
    if eligible_pool:
        selected = max(eligible_pool, key=lambda r: r["score"])
    else:
        selected = {
            "policy": {
                "harm_weight": 0.0,
                "gain_min": 1.0e12,
                "score_min": 1.0e12,
                "margin_min": 1.0e12,
                "harm_max": 0.0,
                "max_rate_h10": 0.0,
                "max_rate_h25": 0.0,
                "max_rate_h50": 0.0,
                "max_rate_h100": 0.0,
            },
            "eligible": False,
            "score": composer._score_eval(fixed_full),
            "val_metrics": {
                "ade": fixed_full["ade_metrics_vs_floor"],
                "shape_gain_vs_bridge": fixed_full["shape_gain_vs_bridge"],
                "collision_delta_005": fixed_full["collision_delta_vs_floor_005"],
                "source_switch_rate": 0.0,
                "source_distribution": fixed_full_compact["source_distribution"],
                "delta_vs_fixed": {key: 0.0 for key in ["all", "t50", "t100", "hard_failure", "easy_degradation", "shape_gain_all", "shape_gain_t50", "shape_gain_t100", "shape_gain_hard_failure"]},
            },
        }
    return {
        "selected": selected,
        "candidate_count": len(rows),
        "fast_candidate_count": len(fast_rows),
        "eligible_count": int(sum(bool(row.get("eligible", False)) for row in rows)),
        "top_candidates": sorted(rows, key=lambda r: r["score"], reverse=True)[:10],
    }


def _signal_metrics(pack: Mapping[str, Any], pred: Mapping[str, Mapping[str, np.ndarray]], fixed_policy: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    acc = []
    for source in SOURCES:
        gain = _fixed_prior_gain_labels(pack, source, fixed_policy)
        pg = pred[source]["gain"]
        sign_acc = float(np.mean((gain > 0.0) == (pg > 0.0))) if len(gain) else 0.0
        corr = float(np.corrcoef(gain, pg)[0, 1]) if len(gain) > 2 and np.std(gain) > EPS and np.std(pg) > EPS else 0.0
        out[source] = {
            "true_positive_rate": float(np.mean(gain > 0.0)) if len(gain) else 0.0,
            "pred_positive_rate": float(np.mean(pg > 0.0)) if len(gain) else 0.0,
            "sign_accuracy": sign_acc,
            "gain_correlation": corr,
        }
        acc.append(sign_acc)
    return {"per_source": out, "mean_sign_accuracy": float(np.mean(acc)) if acc else 0.0}


def _evaluate_domain(domain: str) -> dict[str, Any]:
    train = pairwise._domain_data("train", domain)
    val = pairwise._domain_data("val", domain)
    test = pairwise._domain_data("test", domain)
    if min(len(train["horizon"]), len(val["horizon"]), len(test["horizon"])) < 500:
        return {"domain": domain, "status": "not_run", "reason": "not enough domain rows"}
    packs = pairwise._build_domain_packs(domain)
    fixed_selection = composer._select_composer_on_val(packs["val_pack"])
    fixed_policy = fixed_selection["selected"]["policy"]
    model = _fit_fixed_prior_model(packs["train_pack"], fixed_policy)
    selection = _select_policy_on_val(model, packs["val_pack"], fixed_policy)
    pred_test = _predict_fixed_prior_model(model, packs["test_pack"], fixed_policy)
    selected_xy, shape_switch, chosen = _choose_sources(packs["test_pack"], pred_test, fixed_policy, selection["selected"]["policy"])
    ev = composer._eval_selected(packs["test_pack"], selected_xy, shape_switch)
    fixed_xy, fixed_shape_switch = composer._compose_sources(packs["test_pack"], fixed_policy)
    fixed_eval = composer._eval_selected(packs["test_pack"], fixed_xy, fixed_shape_switch)
    fixed_chosen = _fixed_chosen(packs["test_pack"], fixed_policy)
    compact = _compact(ev, chosen, fixed_chosen)
    fixed_compact = _compact(fixed_eval, fixed_chosen, fixed_chosen)
    delta_vs_fixed = {key: float(compact[key] - fixed_compact[key]) for key in ["all", "t50", "t100", "hard_failure", "easy_degradation", "shape_gain_all", "shape_gain_t50", "shape_gain_t100", "shape_gain_hard_failure"]}
    m = ev["ade_metrics_vs_floor"]
    pass_gate = bool(
        m.get("all_improvement", 0.0) > 0.0
        and m.get("t50_improvement", 0.0) > 0.0
        and m.get("hard_failure_improvement", 0.0) > 0.0
        and m.get("easy_degradation", 1.0) <= 0.02
        and ev["collision_delta_vs_floor_005"] <= 0.01
        and compact["source_switch_rate"] > 0.0
    )
    improves_fixed_core = bool(delta_vs_fixed["all"] > 0.0 or delta_vs_fixed["t50"] > 0.0 or delta_vs_fixed["hard_failure"] > 0.0)
    return {
        "domain": domain,
        "source": "fresh_run",
        "status": "ok",
        "rows": packs["rows"],
        "fixed_policy": fixed_policy,
        "fixed_prior_selection": selection,
        "fixed_prior_compact": compact,
        "fixed_horizon_composer_compact": fixed_compact,
        "delta_vs_fixed": delta_vs_fixed,
        "fixed_prior_metrics": ev,
        "test_signal": _signal_metrics(packs["test_pack"], pred_test, fixed_policy),
        "fixed_prior_pass": pass_gate,
        "improves_fixed_core": improves_fixed_core,
        "caveat": "This policy learns switches relative to the validation-selected fixed composer. Train future waypoints supervise gain/harm, validation selects thresholds, and test is evaluated once. Inference remains past-only and protected; dataset-local raw-frame 2.5D only; no Stage5C/SMC/metric/seconds/true-3D/foundation claim.",
    }


def run_fixed_prior_source_switch_policy() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ft.build_full_trajectory_labels()
    domains = ["ETH_UCY", "TrajNet"]
    results = {domain: _evaluate_domain(domain) for domain in domains}
    positive = [domain for domain, row in results.items() if row.get("fixed_prior_pass")]
    better = [domain for domain, row in results.items() if row.get("improves_fixed_core")]
    result = {
        "source": "fresh_run",
        "protocol": "fixed_composer_prior_source_switch_policy",
        "stage5c_executed": False,
        "smc_enabled": False,
        "metric_or_seconds_claim": False,
        "positive_domains": positive,
        "positive_domain_count": len(positive),
        "two_domain_fixed_prior_gate": len(positive) >= 2,
        "domains_better_than_fixed_on_any_core_metric": better,
        "two_domain_fixed_prior_beats_fixed_gate": len(better) >= 2,
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
            "fixed_composer_prior_switch": True,
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
        "# Stage41 Fixed-Composer Prior Source Switch Policy",
        "",
        "- source: `fresh_run`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        f"- positive domains: `{positive}`",
        f"- two-domain fixed-prior gate: `{result['two_domain_fixed_prior_gate']}`",
        f"- domains better than fixed composer on any core metric: `{better}`",
        f"- two-domain beats-fixed gate: `{result['two_domain_fixed_prior_beats_fixed_gate']}`",
        "",
        "| domain | fixed policy | all ADE | t50 ADE | t100 ADE | hard ADE | easy | source switch | fixed delta all/t50/t100/hard | pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for domain, row in results.items():
        if row.get("status") != "ok":
            lines.append(f"| `{domain}` | not_run | 0 | 0 | 0 | 0 | 0 | 0 | 0/0/0/0 | `False` |")
            continue
        c = row["fixed_prior_compact"]
        d = row["delta_vs_fixed"]
        pol = row["fixed_policy"]
        lines.append(
            f"| `{domain}` | `{pol['short']}/{pol['t50']}/{pol['t100']}` | {c['all']:.4f} | {c['t50']:.4f} | {c['t100']:.4f} | "
            f"{c['hard_failure']:.4f} | {c['easy_degradation']:.4f} | {c['source_switch_rate']:.6f} | "
            f"{d['all']:.6f}/{d['t50']:.6f}/{d['t100']:.6f}/{d['hard_failure']:.6f} | `{row.get('fixed_prior_pass')}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This experiment treats the fixed horizon composer as the safety prior and learns only residual source switches around that prior.",
            "- It directly asks whether a learned source-switch model can improve the current deployable composer instead of merely staying positive versus the floor.",
            "- Validation selects conservative thresholds; test is evaluated once.",
            "- This remains protected dataset-local raw-frame 2.5D evidence, not Stage5C/SMC/metric/seconds/true-3D/foundation evidence.",
            f"- no leakage: `{result['no_leakage']}`",
            f"- claim boundary: `{result['claim_boundary']}`",
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_fixed_prior_source_switch_policy() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_fixed_prior_source_switch_policy()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_fixed_prior_source_switch_policy",
            status,
            started,
            [dl.DATA_DIR / "seq2seq_train.npz", dl.DATA_DIR / "seq2seq_val.npz", dl.DATA_DIR / "seq2seq_test.npz", ft.DATA_DIR / "full_trajectory_test.npz"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_fixed_prior_source_switch_policy()
