from __future__ import annotations

import itertools
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
from src import stage41_joint_multiagent_consistency as jmc
from src import stage41_joint_rollout_consistency as jrc
from src import stage41_learned_shape_gain_gate as gain_gate
from src import stage41_learned_waypoint_shape_bridge as shape


OUT_DIR = dl.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_shape_policy_composer.json"
REPORT_MD = OUT_DIR / "stage41_shape_policy_composer.md"
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


def _source_name(policy: Mapping[str, str], horizon: np.ndarray) -> np.ndarray:
    out = np.empty(len(horizon), dtype=object)
    out[np.isin(horizon.astype(int), [10, 25])] = policy["short"]
    out[horizon.astype(int) == 50] = policy["t50"]
    out[horizon.astype(int) == 100] = policy["t100"]
    return out


def _compose_sources(pack: Mapping[str, Any], policy: Mapping[str, str]) -> tuple[np.ndarray, np.ndarray]:
    selected = pack["bridge_xy"].copy()
    shape_switch = np.zeros(len(pack["horizon"]), dtype=bool)
    sources = _source_name(policy, pack["horizon"])
    for name in ["old_shape", "gain_gate"]:
        mask = sources == name
        if not np.any(mask):
            continue
        selected[mask] = pack[name]["xy"][mask]
        shape_switch[mask] = pack[name]["shape_switch"][mask]
    return selected, shape_switch


def _shape_gain(selected_ade: np.ndarray, bridge_ade: np.ndarray, labels: Mapping[str, np.ndarray]) -> dict[str, float]:
    horizon = labels["horizon"].astype(int)
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    return {
        "all": shape._gain(selected_ade, bridge_ade, np.ones(len(selected_ade), dtype=bool)),
        "t50": shape._gain(selected_ade, bridge_ade, horizon == 50),
        "t100": shape._gain(selected_ade, bridge_ade, horizon == 100),
        "hard_failure": shape._gain(selected_ade, bridge_ade, hard),
    }


def _eval_selected(pack: Mapping[str, Any], selected_xy: np.ndarray, shape_switch: np.ndarray) -> dict[str, Any]:
    labels = pack["labels"]
    floor_ade, floor_fde = ft._trajectory_errors(pack["floor_xy"], labels)
    bridge_ade, _bridge_fde = ft._trajectory_errors(pack["bridge_xy"], labels)
    selected_ade, selected_fde = ft._trajectory_errors(selected_xy, labels)
    keys = shape._group_keys(pack["data"])
    floor_stats = jrc._joint_stats("floor", pack["floor_xy"], labels, keys, np.zeros(len(shape_switch), dtype=bool))
    selected_stats = jrc._joint_stats("composer_selected", selected_xy, labels, keys, pack["bridge_switch"])
    selected_min_dist = jmc._min_group_distance(selected_xy, keys, labels["normalizer"])
    floor_min_dist = jmc._min_group_distance(pack["floor_xy"], keys, labels["normalizer"])
    finite_group_dist = np.isfinite(selected_min_dist) & np.isfinite(floor_min_dist)
    group_dist_delta = selected_min_dist[finite_group_dist] - floor_min_dist[finite_group_dist]
    return {
        "ade_metrics_vs_floor": s41._metrics(selected_ade, floor_ade, shape._metric_ds(labels), pack["bridge_switch"]),
        "fde_metrics_vs_floor": s41._metrics(selected_fde, floor_fde, shape._metric_ds(labels), pack["bridge_switch"]),
        "shape_gain_vs_bridge": {
            **_shape_gain(selected_ade, bridge_ade, labels),
            "shape_switch_rate": float(np.mean(shape_switch)) if len(shape_switch) else 0.0,
        },
        "collision_delta_vs_floor_005": float(selected_stats["near_collision_rate_005"] - floor_stats["near_collision_rate_005"]),
        "smoothness_jagged_delta": float(selected_stats["smoothness"]["jagged_rate"] - floor_stats["smoothness"]["jagged_rate"]),
        "mean_group_distance_delta": float(np.mean(group_dist_delta)) if len(group_dist_delta) else 0.0,
    }


def _candidate_policies() -> list[dict[str, str]]:
    sources = ["bridge", "old_shape", "gain_gate"]
    return [{"short": short, "t50": t50, "t100": t100} for short, t50, t100 in itertools.product(sources, repeat=3)]


def _score_eval(ev: Mapping[str, Any]) -> float:
    m = ev["ade_metrics_vs_floor"]
    g = ev["shape_gain_vs_bridge"]
    return float(
        m.get("all_improvement", 0.0)
        + 1.6 * m.get("t50_improvement", 0.0)
        + 1.4 * m.get("t100_improvement", 0.0)
        + 1.2 * m.get("hard_failure_improvement", 0.0)
        + 4.0 * max(g["all"], g["t50"], g["t100"], g["hard_failure"])
        - 45.0 * max(0.0, m.get("easy_degradation", 1.0) - 0.02)
        - 8.0 * max(0.0, ev["collision_delta_vs_floor_005"] - 0.01)
    )


def _select_composer_on_val(pack: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for policy in _candidate_policies():
        selected_xy, shape_switch = _compose_sources(pack, policy)
        ev = _eval_selected(pack, selected_xy, shape_switch)
        m = ev["ade_metrics_vs_floor"]
        g = ev["shape_gain_vs_bridge"]
        eligible = (
            m.get("all_improvement", 0.0) > 0.0
            and m.get("t50_improvement", 0.0) > 0.0
            and m.get("hard_failure_improvement", 0.0) > 0.0
            and m.get("easy_degradation", 1.0) <= 0.02
            and ev["collision_delta_vs_floor_005"] <= 0.01
            and any(source != "bridge" for source in policy.values())
            and max(g["all"], g["t50"], g["t100"], g["hard_failure"]) >= 0.0
        )
        rows.append({"policy": policy, "eligible": bool(eligible), "score": _score_eval(ev), "val_metrics": ev})
    pool = [row for row in rows if row["eligible"]] or rows
    selected = max(pool, key=lambda row: (bool(row["eligible"]), row["score"]))
    return {
        "selected": selected,
        "candidate_count": len(rows),
        "eligible_count": int(sum(row["eligible"] for row in rows)),
        "top_candidates": sorted(rows, key=lambda row: row["score"], reverse=True)[:10],
    }


def _make_source_pack(
    data: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    endpoint_pred: Mapping[str, np.ndarray],
    gate_pred: Mapping[str, np.ndarray],
    shape_pred: Mapping[str, np.ndarray],
    bridge_selection: Mapping[str, Any],
    old_policy: Mapping[str, Any],
    gain_policy: Mapping[str, Any],
) -> dict[str, Any]:
    floor_xy, bridge_xy, old_xy, bridge_switch, old_shape_switch = shape._apply_shape_policy(
        data, labels, endpoint_pred, gate_pred, shape_pred, bridge_selection, old_policy
    )
    gain_pack = gain_gate._make_pack(data, labels, endpoint_pred, gate_pred, shape_pred, bridge_selection)
    gain_xy, _gain_bridge_switch, gain_shape_switch = gain_gate._apply_policy(gain_pack, gain_policy)
    return {
        "data": data,
        "labels": labels,
        "horizon": data["horizon"].astype(int),
        "floor_xy": floor_xy,
        "bridge_xy": bridge_xy,
        "bridge_switch": bridge_switch,
        "old_shape": {"xy": old_xy, "shape_switch": old_shape_switch},
        "gain_gate": {"xy": gain_xy, "shape_switch": gain_shape_switch},
    }


def _compact_metrics(ev: Mapping[str, Any]) -> dict[str, Any]:
    m = ev["ade_metrics_vs_floor"]
    return {
        "all": float(m.get("all_improvement", 0.0)),
        "t50": float(m.get("t50_improvement", 0.0)),
        "t100": float(m.get("t100_improvement", 0.0)),
        "hard_failure": float(m.get("hard_failure_improvement", 0.0)),
        "easy_degradation": float(m.get("easy_degradation", 0.0)),
        "switch_rate": float(m.get("switch_rate", 0.0)),
        "shape_gain_vs_bridge": ev["shape_gain_vs_bridge"],
        "collision_delta_005": float(ev["collision_delta_vs_floor_005"]),
    }


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
    val_pack = _make_source_pack(
        val,
        labels_val,
        pred_val,
        gate_val,
        shape_val,
        bridge_selection,
        old_selection["selected"],
        gain_selection["selected"],
    )
    test_pack = _make_source_pack(
        test,
        labels_test,
        pred_test,
        gate_test,
        shape_test,
        bridge_selection,
        old_selection["selected"],
        gain_selection["selected"],
    )
    composer = _select_composer_on_val(val_pack)
    selected_xy, shape_switch = _compose_sources(test_pack, composer["selected"]["policy"])
    test_eval = _eval_selected(test_pack, selected_xy, shape_switch)
    old_eval = _eval_selected(test_pack, test_pack["old_shape"]["xy"], test_pack["old_shape"]["shape_switch"])
    gain_eval = _eval_selected(test_pack, test_pack["gain_gate"]["xy"], test_pack["gain_gate"]["shape_switch"])
    pass_gate = bool(
        test_eval["ade_metrics_vs_floor"].get("all_improvement", 0.0) > 0.0
        and test_eval["ade_metrics_vs_floor"].get("t50_improvement", 0.0) > 0.0
        and test_eval["ade_metrics_vs_floor"].get("hard_failure_improvement", 0.0) > 0.0
        and test_eval["ade_metrics_vs_floor"].get("easy_degradation", 1.0) <= 0.02
        and test_eval["collision_delta_vs_floor_005"] <= 0.01
        and test_eval["shape_gain_vs_bridge"]["shape_switch_rate"] > 0.0
    )
    return {
        "domain": domain,
        "source": "fresh_run",
        "status": "ok",
        "rows": {"train": int(len(train["horizon"])), "val": int(len(val["horizon"])), "test": int(len(test["horizon"]))},
        "endpoint_training": endpoint_training,
        "shape_training": shape_training,
        "bridge_selection_summary": bridge_selection["selected"],
        "old_shape_selection_summary": old_selection["selected"],
        "gain_gate_selection_summary": gain_selection["selected"],
        "composer_selection": composer,
        "composer_metrics": test_eval,
        "baseline_source_metrics": {
            "old_shape": _compact_metrics(old_eval),
            "gain_gate": _compact_metrics(gain_eval),
        },
        "composer_compact": _compact_metrics(test_eval),
        "composer_pass": pass_gate,
        "caveat": "Composer is val-selected by horizon family over bridge/old-shape/gain-gate sources and evaluated once on test. Future waypoints remain labels/eval only; inference is past-only and protected by the Stage37/floor bridge. Dataset-local raw-frame 2.5D only; no Stage5C/SMC/metric/seconds/true-3D/foundation claim.",
    }


def run_shape_policy_composer() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ft.build_full_trajectory_labels()
    domains = ["ETH_UCY", "TrajNet"]
    results = {domain: _evaluate_domain(domain) for domain in domains}
    positive = [domain for domain, row in results.items() if row.get("composer_pass")]
    result = {
        "source": "fresh_run",
        "protocol": "domain_horizon_shape_policy_composer",
        "stage5c_executed": False,
        "smc_enabled": False,
        "metric_or_seconds_claim": False,
        "positive_domains": positive,
        "positive_domain_count": len(positive),
        "two_domain_composer_gate": len(positive) >= 2,
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
            "domain_horizon_composer": True,
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
        "# Stage41 Domain/Horizon Shape-Policy Composer",
        "",
        "- source: `fresh_run`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        f"- positive domains: `{positive}`",
        f"- two-domain composer gate: `{result['two_domain_composer_gate']}`",
        "",
        "| domain | selected short/t50/t100 | all ADE | t50 ADE | t100 ADE | hard ADE | easy | shape gain all/t50/t100/hard | shape switch | pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for domain, row in results.items():
        if row.get("status") != "ok":
            lines.append(f"| `{domain}` | `not_run` | 0 | 0 | 0 | 0 | 0 | 0/0/0/0 | 0 | `False` |")
            continue
        m = row["composer_metrics"]["ade_metrics_vs_floor"]
        g = row["composer_metrics"]["shape_gain_vs_bridge"]
        policy = row["composer_selection"]["selected"]["policy"]
        lines.append(
            f"| `{domain}` | `{policy['short']}/{policy['t50']}/{policy['t100']}` | {m.get('all_improvement', 0.0):.4f} | "
            f"{m.get('t50_improvement', 0.0):.4f} | {m.get('t100_improvement', 0.0):.4f} | "
            f"{m.get('hard_failure_improvement', 0.0):.4f} | {m.get('easy_degradation', 0.0):.4f} | "
            f"{g.get('all', 0.0):.6f}/{g.get('t50', 0.0):.6f}/{g.get('t100', 0.0):.6f}/{g.get('hard_failure', 0.0):.6f} | "
            f"{g.get('shape_switch_rate', 0.0):.6f} | `{row.get('composer_pass')}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The composer addresses the previous mixed result by selecting the best source per horizon family on validation only.",
            "- It can choose the pure bridge, the residual-norm learned-shape policy, or the train-fitted gain/harm shape policy for short, t50, and t100 rows separately.",
            "- This is still a protected full-waypoint bridge, not an unprotected latent generative rollout and not a metric/seconds/true-3D/foundation result.",
            f"- no leakage: `{result['no_leakage']}`",
            f"- claim boundary: `{result['claim_boundary']}`",
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_shape_policy_composer() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_shape_policy_composer()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_shape_policy_composer",
            status,
            started,
            [dl.DATA_DIR / "seq2seq_train.npz", dl.DATA_DIR / "seq2seq_val.npz", dl.DATA_DIR / "seq2seq_test.npz", ft.DATA_DIR / "full_trajectory_test.npz"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_shape_policy_composer()
