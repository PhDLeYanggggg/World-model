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


OUT_DIR = dl.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_weighted_pairwise_shape_switch_policy.json"
REPORT_MD = OUT_DIR / "stage41_weighted_pairwise_shape_switch_policy.md"
NON_BRIDGE_SOURCES = pairwise.NON_BRIDGE_SOURCES
SOURCES = pairwise.SOURCES
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


def _weighted_ridge_fit(x: np.ndarray, y: np.ndarray, weight: np.ndarray, lam: float = 0.12) -> np.ndarray:
    xb = np.concatenate([np.ones((len(x), 1), dtype=np.float64), x.astype(np.float64)], axis=1)
    w = np.sqrt(np.maximum(weight.astype(np.float64), EPS))[:, None]
    xw = xb * w
    yw = y.astype(np.float64) * w[:, 0]
    reg = lam * np.eye(xb.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    return np.linalg.solve(xw.T @ xw + reg, xw.T @ yw)


def _training_weight(pack: Mapping[str, Any], source: str, gain: np.ndarray) -> np.ndarray:
    labels = pack["labels"]
    horizon = labels["horizon"].astype(int)
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    positive = gain > 0.0
    pos = max(1, int(np.sum(positive)))
    neg = max(1, int(np.sum(~positive)))
    positive_boost = min(80.0, max(4.0, neg / pos))
    source_switch = meta._source_switch(pack, source)
    weight = np.ones(len(gain), dtype=np.float64)
    weight += 3.0 * hard.astype(np.float64)
    weight += 2.5 * (horizon == 50).astype(np.float64)
    weight += 2.0 * (horizon == 100).astype(np.float64)
    weight += positive_boost * positive.astype(np.float64)
    weight += 1.5 * source_switch.astype(np.float64)
    return np.nan_to_num(weight, nan=1.0, posinf=1.0, neginf=1.0)


def _fit_weighted_pairwise_models(pack: Mapping[str, Any], lam: float = 0.12) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for source in NON_BRIDGE_SOURCES:
        x_raw = meta._source_feature_matrix(pack, source)
        gain = pairwise._gain_labels(pack, source)
        clip = float(np.quantile(np.abs(gain), 0.995)) if len(gain) else 0.0
        if clip > EPS:
            gain = np.clip(gain, -clip, clip)
        harm = np.maximum(-gain, 0.0)
        weight = _training_weight(pack, source, gain)
        mean, std = meta._fit_standardizer(x_raw)
        x = meta._standardize(x_raw, mean, std)
        models[source] = {
            "mean": mean,
            "std": std,
            "w_gain": _weighted_ridge_fit(x, gain, weight, lam=lam),
            "w_harm": _weighted_ridge_fit(x, harm, weight, lam=lam),
            "feature_dim": int(x_raw.shape[1]),
            "positive_rate": float(np.mean(gain > 0.0)) if len(gain) else 0.0,
            "mean_weight": float(np.mean(weight)) if len(weight) else 0.0,
            "gain_clip": clip,
        }
    return {"type": "hard_tail_positive_weighted_pairwise_gain_harm_ridge", "sources": models}


def _compact_mode_eval(ev: Mapping[str, Any], chosen: np.ndarray) -> dict[str, Any]:
    compact = meta._compact_eval(ev, chosen)
    return {k: compact[k] for k in ["all", "t50", "t100", "hard_failure", "easy_degradation", "shape_gain_all", "shape_gain_t50", "shape_gain_t100", "shape_gain_hard_failure", "shape_switch_rate", "collision_delta_005", "source_distribution"]}


def _evaluate_domain(domain: str) -> dict[str, Any]:
    base_train = pairwise._domain_data("train", domain)
    base_val = pairwise._domain_data("val", domain)
    base_test = pairwise._domain_data("test", domain)
    if min(len(base_train["horizon"]), len(base_val["horizon"]), len(base_test["horizon"])) < 500:
        return {"domain": domain, "status": "not_run", "reason": "not enough domain rows"}
    packs = pairwise._build_domain_packs(domain)
    model = _fit_weighted_pairwise_models(packs["train_pack"])
    selection = pairwise._select_pairwise_policy_on_val(model, packs["val_pack"])
    pred_test = pairwise._predict_pairwise_models(model, packs["test_pack"])
    selected_xy, shape_switch, chosen = pairwise._choose_pairwise_sources(packs["test_pack"], pred_test, selection["selected"]["policy"])
    ev = composer._eval_selected(packs["test_pack"], selected_xy, shape_switch)
    fixed_selection = composer._select_composer_on_val(packs["val_pack"])
    fixed_xy, fixed_shape_switch = composer._compose_sources(packs["test_pack"], fixed_selection["selected"]["policy"])
    fixed_eval = composer._eval_selected(packs["test_pack"], fixed_xy, fixed_shape_switch)
    fixed_chosen = meta._chosen_from_fixed_policy(packs["test_pack"], fixed_selection["selected"]["policy"])
    compact = _compact_mode_eval(ev, chosen)
    fixed_compact = _compact_mode_eval(fixed_eval, fixed_chosen)
    delta_vs_fixed = {key: float(compact[key] - fixed_compact[key]) for key in ["all", "t50", "t100", "hard_failure", "easy_degradation", "shape_gain_all", "shape_gain_t50", "shape_gain_t100", "shape_gain_hard_failure"]}
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
        "weighted_pairwise_model": {
            "type": model["type"],
            "sources": {
                source: {
                    "feature_dim": int(model["sources"][source]["feature_dim"]),
                    "positive_rate": float(model["sources"][source]["positive_rate"]),
                    "mean_weight": float(model["sources"][source]["mean_weight"]),
                    "gain_clip": float(model["sources"][source]["gain_clip"]),
                }
                for source in NON_BRIDGE_SOURCES
            },
        },
        "weighted_pairwise_selection": selection,
        "weighted_pairwise_metrics": ev,
        "weighted_pairwise_compact": compact,
        "fixed_horizon_composer_compact": fixed_compact,
        "delta_vs_fixed": delta_vs_fixed,
        "test_pairwise_signal": pairwise._pairwise_signal_metrics(packs["test_pack"], pred_test),
        "weighted_pairwise_pass": pass_gate,
        "caveat": "Hard/tail/positive-gain labels are used only as training weights, never inference inputs. Inference uses past-only features and candidate rollout geometry; validation selects safety thresholds and test is evaluated once. Dataset-local raw-frame 2.5D only; no Stage5C/SMC/metric/seconds/true-3D/foundation claim.",
    }


def run_weighted_pairwise_shape_switch_policy() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ft.build_full_trajectory_labels()
    domains = ["ETH_UCY", "TrajNet"]
    results = {domain: _evaluate_domain(domain) for domain in domains}
    positive = [domain for domain, row in results.items() if row.get("weighted_pairwise_pass")]
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
        "protocol": "hard_tail_positive_weighted_pairwise_shape_switch_policy",
        "stage5c_executed": False,
        "smc_enabled": False,
        "metric_or_seconds_claim": False,
        "positive_domains": positive,
        "positive_domain_count": len(positive),
        "two_domain_weighted_pairwise_gate": len(positive) >= 2,
        "domains_better_than_fixed_on_any_core_metric": better_than_fixed,
        "domain_results": results,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_endpoint_label_eval_only": True,
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "hard_failure_labels_inference_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "val_selected_policy": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "claim_boundary": {
            "weighted_pairwise_gain_harm_source_switch": True,
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
        "# Stage41 Weighted Pairwise Shape Switch Policy",
        "",
        "- source: `fresh_run`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        f"- positive domains: `{positive}`",
        f"- two-domain weighted pairwise gate: `{result['two_domain_weighted_pairwise_gate']}`",
        f"- better than fixed composer on any core metric: `{better_than_fixed}`",
        "",
        "| domain | all ADE | t50 ADE | t100 ADE | hard ADE | easy | shape gain all/t50/t100/hard | sign acc | fixed delta all/t50/t100/hard | pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for domain, row in results.items():
        if row.get("status") != "ok":
            lines.append(f"| `{domain}` | 0 | 0 | 0 | 0 | 0 | 0/0/0/0 | 0 | 0/0/0/0 | `False` |")
            continue
        c = row["weighted_pairwise_compact"]
        d = row["delta_vs_fixed"]
        lines.append(
            f"| `{domain}` | {c['all']:.4f} | {c['t50']:.4f} | {c['t100']:.4f} | {c['hard_failure']:.4f} | {c['easy_degradation']:.4f} | "
            f"{c['shape_gain_all']:.6f}/{c['shape_gain_t50']:.6f}/{c['shape_gain_t100']:.6f}/{c['shape_gain_hard_failure']:.6f} | "
            f"{row['test_pairwise_signal']['mean_sign_accuracy']:.4f} | "
            f"{d['all']:.6f}/{d['t50']:.6f}/{d['t100']:.6f}/{d['hard_failure']:.6f} | `{row.get('weighted_pairwise_pass')}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This experiment addresses the previous pairwise model's rare-positive switch labels by upweighting hard/failure, t50/t100, source-switch, and positive-gain rows during training.",
            "- The hard/failure labels are training weights only and are not available as inference inputs.",
            "- Validation selects conservative deployment thresholds; test is evaluated once.",
            "- This remains protected dataset-local raw-frame 2.5D evidence, not Stage5C/SMC/metric/seconds/true-3D/foundation evidence.",
            f"- no leakage: `{result['no_leakage']}`",
            f"- claim boundary: `{result['claim_boundary']}`",
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_weighted_pairwise_shape_switch_policy() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_weighted_pairwise_shape_switch_policy()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_weighted_pairwise_shape_switch_policy",
            status,
            started,
            [dl.DATA_DIR / "seq2seq_train.npz", dl.DATA_DIR / "seq2seq_val.npz", dl.DATA_DIR / "seq2seq_test.npz", ft.DATA_DIR / "full_trajectory_test.npz"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_weighted_pairwise_shape_switch_policy()
