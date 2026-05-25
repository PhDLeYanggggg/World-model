from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_domain_local_neural_retrain as dl
from src import stage41_full_trajectory_world_state as ft
from src import stage41_joint_multiagent_consistency as jmc
from src import stage41_joint_rollout_consistency as jrc


OUT_DIR = dl.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_domain_local_all_agent_world_state.json"
REPORT_MD = OUT_DIR / "stage41_domain_local_all_agent_world_state.md"
BOOTSTRAP_N = 1000
SEED = 41543
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


def _linear_waypoints(current_xy: np.ndarray, endpoint_xy: np.ndarray) -> np.ndarray:
    current = current_xy.astype(np.float64)
    endpoint = endpoint_xy.astype(np.float64)
    return (current[:, None, :] + ft.WAYPOINT_FRAC[None, :, None].astype(np.float64) * (endpoint[:, None, :] - current[:, None, :])).astype(np.float64)


def _endpoint_from_delta(data: Mapping[str, np.ndarray], delta: np.ndarray) -> np.ndarray:
    return data["current_xy"].astype(np.float64) + delta.astype(np.float64) * data["normalizer"].astype(np.float64)[:, None]


def _metadata(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if "agent_id" in data and "frame_id" in data:
        return {"agent_id": data["agent_id"].astype(np.int64), "frame_id": data["frame_id"].astype(np.float64)}
    if "ids" not in data:
        return {"agent_id": np.arange(len(data["horizon"]), dtype=np.int64), "frame_id": np.arange(len(data["horizon"]), dtype=np.float64)}
    combined = dl._load_combined()
    ids = data["ids"].astype(np.int64)
    return {"agent_id": combined["agent_id"].astype(np.int64)[ids], "frame_id": combined["frame_id"].astype(np.float64)[ids]}


def _group_keys(data: Mapping[str, np.ndarray]) -> np.ndarray:
    meta = _metadata(data)
    source = data["source_file"].astype(str)
    scene = data["scene_id"].astype(str)
    frame = np.rint(meta["frame_id"]).astype(np.int64)
    horizon = data["horizon"].astype(int)
    return np.asarray([f"{source[i]}|{scene[i]}|{frame[i]}|{horizon[i]}" for i in range(len(horizon))], dtype=object)


def _group_count(keys: np.ndarray) -> np.ndarray:
    counts = Counter(map(str, keys.tolist()))
    return np.asarray([counts[str(k)] for k in keys], dtype=np.int32)


def _labels_from_endpoint_dataset(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    waypoint_xy = _linear_waypoints(data["current_xy"], data["future_xy"])
    return {
        "waypoint_xy": waypoint_xy,
        "waypoint_valid": np.ones((len(waypoint_xy), len(ft.WAYPOINT_FRAC)), dtype=bool),
        "current_xy": data["current_xy"].astype(np.float64),
        "normalizer": data["normalizer"].astype(np.float64),
        "horizon": data["horizon"].astype(np.int16),
        "hard": data["hard"].astype(bool),
        "failure": data["failure"].astype(bool),
        "easy": data["easy"].astype(bool),
        "domain": data["domain"].astype("U32"),
        "candidate_fde": data["candidate_fde"].astype(np.float64),
    }


def _metric_ds(labels: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "horizon": labels["horizon"],
        "hard": labels["hard"],
        "failure": labels["failure"],
        "easy": labels["easy"],
        "domain": labels["domain"],
        "candidate_fde": labels["candidate_fde"],
    }


def _safe_metrics(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if not np.any(mask):
        return {
            "rows": 0,
            "all_improvement": 0.0,
            "t50_improvement": 0.0,
            "t100_improvement": 0.0,
            "hard_failure_improvement": 0.0,
            "easy_degradation": 0.0,
            "switch_rate": 0.0,
        }
    ds = {k: v[mask] for k, v in _metric_ds(labels).items()}
    return s41._metrics(selected[mask], floor[mask], ds, switch[mask])


def _bootstrap(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray, seed: int) -> dict[str, Any]:
    ids = np.where(mask)[0]
    if len(ids) < 20:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(len(ids)), "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(BOOTSTRAP_N):
        sample = rng.choice(ids, size=len(ids), replace=True)
        vals.append(1.0 - float(selected[sample].mean()) / max(float(floor[sample].mean()), EPS))
    return {
        "low": float(np.percentile(vals, 2.5)),
        "mid": float(np.percentile(vals, 50.0)),
        "high": float(np.percentile(vals, 97.5)),
        "n": int(len(ids)),
        "bootstrap_n": BOOTSTRAP_N,
    }


def _bootstrap_report(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray], multi: np.ndarray) -> dict[str, Any]:
    horizon = labels["horizon"].astype(int)
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    masks = {
        "all": np.ones(len(selected), dtype=bool),
        "t50": horizon == 50,
        "t100": horizon == 100,
        "hard_failure": hard,
        "multi_agent": multi,
        "multi_agent_t50": multi & (horizon == 50),
        "multi_agent_hard_failure": multi & hard,
    }
    return {name: _bootstrap(selected, floor, mask, SEED + i) for i, (name, mask) in enumerate(masks.items())}


def _world_state_from_selection(data: Mapping[str, np.ndarray], selected_delta: np.ndarray, switch: np.ndarray) -> dict[str, Any]:
    labels = _labels_from_endpoint_dataset(data)
    keys = _group_keys(data)
    floor_delta = data["cand_delta"][:, 0, :].astype(np.float64)
    floor_endpoint = _endpoint_from_delta(data, floor_delta)
    selected_endpoint = _endpoint_from_delta(data, selected_delta)
    floor_xy = _linear_waypoints(data["current_xy"], floor_endpoint)
    selected_xy = _linear_waypoints(data["current_xy"], selected_endpoint)
    floor_ade, floor_fde = ft._trajectory_errors(floor_xy, labels)
    selected_ade, selected_fde = ft._trajectory_errors(selected_xy, labels)
    group_counts = _group_count(keys)
    multi = group_counts >= 2
    floor_stats = jrc._joint_stats("floor", floor_xy, labels, keys, np.zeros(len(switch), dtype=bool))
    selected_stats = jrc._joint_stats("domain_local_selected_endpoint_linear_rollout", selected_xy, labels, keys, switch)
    collision_delta = float(selected_stats["near_collision_rate_005"] - floor_stats["near_collision_rate_005"])
    smoothness_delta = float(selected_stats["smoothness"]["jagged_rate"] - floor_stats["smoothness"]["jagged_rate"])
    return {
        "labels": labels,
        "keys": keys,
        "floor_xy": floor_xy,
        "selected_xy": selected_xy,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "multi": multi,
        "floor_stats": floor_stats,
        "selected_stats": selected_stats,
        "collision_delta_005": collision_delta,
        "smoothness_jagged_delta": smoothness_delta,
    }


def _apply_proximity_guard(
    data: Mapping[str, np.ndarray],
    selected_delta: np.ndarray,
    switch: np.ndarray,
    min_sep: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if min_sep <= 0:
        return selected_delta.copy(), switch.copy(), 0
    labels = _labels_from_endpoint_dataset(data)
    keys = _group_keys(data)
    floor_delta = data["cand_delta"][:, 0, :].astype(np.float64)
    floor_endpoint = _endpoint_from_delta(data, floor_delta)
    selected_endpoint = _endpoint_from_delta(data, selected_delta)
    floor_xy = _linear_waypoints(data["current_xy"], floor_endpoint)
    selected_xy = _linear_waypoints(data["current_xy"], selected_endpoint)
    normalizer = labels["normalizer"].astype(np.float64)
    floor_min = jmc._min_group_distance(floor_xy, keys, normalizer)
    selected_min = jmc._min_group_distance(selected_xy, keys, normalizer)
    guard = switch.astype(bool) & np.isfinite(selected_min) & (selected_min < min_sep) & (selected_min < floor_min)
    out_switch = switch.astype(bool).copy()
    out_delta = selected_delta.astype(np.float64).copy()
    out_switch[guard] = False
    out_delta[guard] = floor_delta[guard]
    return out_delta, out_switch, int(np.sum(guard))


def _gate_pass(ade_metrics: Mapping[str, Any], fde_metrics: Mapping[str, Any], multi_ade: Mapping[str, Any], collision_delta: float, smoothness_delta: float, bootstrap: Mapping[str, Any] | None = None) -> bool:
    bootstrap = bootstrap or {}
    all_low = (bootstrap.get("all") or {"low": 1.0}).get("low", 1.0)
    multi_low = (bootstrap.get("multi_agent") or {"low": 1.0}).get("low", 1.0)
    return bool(
        ade_metrics.get("all_improvement", 0.0) > 0
        and ade_metrics.get("t50_improvement", 0.0) > 0
        and ade_metrics.get("hard_failure_improvement", 0.0) > 0
        and ade_metrics.get("easy_degradation", 1.0) <= 0.02
        and multi_ade.get("all_improvement", 0.0) > 0
        and multi_ade.get("t50_improvement", 0.0) > 0
        and multi_ade.get("hard_failure_improvement", 0.0) > 0
        and fde_metrics.get("all_improvement", 0.0) > 0
        and fde_metrics.get("t50_improvement", 0.0) > 0
        and collision_delta <= 0.01
        and smoothness_delta <= 0.01
        and all_low > 0
        and multi_low > 0
    )


def _guard_score(ade_metrics: Mapping[str, Any], collision_delta: float, multi_ade: Mapping[str, Any]) -> float:
    return (
        float(ade_metrics.get("all_improvement", 0.0))
        + 1.4 * float(ade_metrics.get("t50_improvement", 0.0))
        + 1.0 * float(ade_metrics.get("t100_improvement", 0.0))
        + 1.2 * float(ade_metrics.get("hard_failure_improvement", 0.0))
        + 0.6 * float(multi_ade.get("all_improvement", 0.0))
        - 40.0 * max(0.0, float(ade_metrics.get("easy_degradation", 1.0)) - 0.02)
        - 12.0 * max(0.0, collision_delta - 0.01)
    )


def _evaluate_guard_candidate(data: Mapping[str, np.ndarray], selected_delta: np.ndarray, switch: np.ndarray, min_sep: float) -> dict[str, Any]:
    guarded_delta, guarded_switch, guarded_off = _apply_proximity_guard(data, selected_delta, switch, min_sep)
    ws = _world_state_from_selection(data, guarded_delta, guarded_switch)
    labels = ws["labels"]
    ade_metrics = s41._metrics(ws["selected_ade"], ws["floor_ade"], _metric_ds(labels), guarded_switch)
    fde_metrics = s41._metrics(ws["selected_fde"], ws["floor_fde"], _metric_ds(labels), guarded_switch)
    multi_ade = _safe_metrics(ws["selected_ade"], ws["floor_ade"], labels, guarded_switch, ws["multi"])
    return {
        "min_sep": float(min_sep),
        "guarded_off": guarded_off,
        "eligible": _gate_pass(ade_metrics, fde_metrics, multi_ade, ws["collision_delta_005"], ws["smoothness_jagged_delta"]),
        "score": _guard_score(ade_metrics, ws["collision_delta_005"], multi_ade),
        "ade_metrics": ade_metrics,
        "multi_agent_ade_metrics": multi_ade,
        "collision_delta_005": ws["collision_delta_005"],
        "smoothness_jagged_delta": ws["smoothness_jagged_delta"],
    }


def _select_proximity_guard(data: Mapping[str, np.ndarray], selected_delta: np.ndarray, switch: np.ndarray) -> dict[str, Any]:
    rows = [_evaluate_guard_candidate(data, selected_delta, switch, min_sep) for min_sep in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]]
    pool = [row for row in rows if row["eligible"]] or rows
    selected = max(pool, key=lambda row: row["score"])
    return {"selected": selected, "candidates": rows}


def _select_for_domain(
    domain: str,
    train_all: Mapping[str, np.ndarray],
    val_all: Mapping[str, np.ndarray],
    test_all: Mapping[str, np.ndarray],
    *,
    prefiltered: bool = False,
) -> dict[str, Any]:
    if prefiltered:
        train, val, test = dict(train_all), dict(val_all), dict(test_all)
    else:
        train = dl._subset(train_all, dl._domain_mask(train_all, domain))
        val = dl._subset(val_all, dl._domain_mask(val_all, domain))
        test = dl._subset(test_all, dl._domain_mask(test_all, domain))
    if min(len(train["horizon"]), len(val["horizon"]), len(test["horizon"])) < 100:
        return {"domain": domain, "status": "not_run", "reason": "not enough train/val/test rows for domain-local all-agent audit"}

    training = dl._train_endpoint(domain, train, val)
    pred_train = dl._predict_endpoint(training["checkpoint"], train)
    pred_val = dl._predict_endpoint(training["checkpoint"], val)
    pred_test = dl._predict_endpoint(training["checkpoint"], test)
    fde_train = dl._endpoint_fde(pred_train["delta"], train)
    fde_val = dl._endpoint_fde(pred_val["delta"], val)
    fde_test = dl._endpoint_fde(pred_test["delta"], test)
    gate = dl._train_gate(train, pred_train, fde_train)
    gate_val = dl._predict_gate(gate, val, pred_val, fde_val)
    selection = dl._select_gate_policy(val, pred_val, gate_val)
    selected_delta_val, switch_val = dl._apply_policy(val, pred_val, gate_val, selection["selected"]["policy"])
    proximity_guard = _select_proximity_guard(val, selected_delta_val, switch_val.astype(bool))
    gate_test = dl._predict_gate(gate, test, pred_test, fde_test)
    selected_delta, switch = dl._apply_policy(test, pred_test, gate_test, selection["selected"]["policy"])
    selected_delta, switch, guarded_off = _apply_proximity_guard(test, selected_delta, switch.astype(bool), float(proximity_guard["selected"]["min_sep"]))
    ws = _world_state_from_selection(test, selected_delta, switch.astype(bool))
    labels = ws["labels"]
    multi = ws["multi"]
    ade_metrics = s41._metrics(ws["selected_ade"], ws["floor_ade"], _metric_ds(labels), switch)
    fde_metrics = s41._metrics(ws["selected_fde"], ws["floor_fde"], _metric_ds(labels), switch)
    multi_ade = _safe_metrics(ws["selected_ade"], ws["floor_ade"], labels, switch, multi)
    multi_fde = _safe_metrics(ws["selected_fde"], ws["floor_fde"], labels, switch, multi)
    bootstrap = _bootstrap_report(ws["selected_ade"], ws["floor_ade"], labels, multi)
    pass_gate = _gate_pass(ade_metrics, fde_metrics, multi_ade, ws["collision_delta_005"], ws["smoothness_jagged_delta"], bootstrap)
    return {
        "domain": domain,
        "source": "fresh_run",
        "status": "ok",
        "rows": {"train": int(len(train["horizon"])), "val": int(len(val["horizon"])), "test": int(len(test["horizon"]))},
        "coverage": {
            "test_rows": int(len(test["horizon"])),
            "t50_rows": int(np.sum(test["horizon"] == 50)),
            "t100_rows": int(np.sum(test["horizon"] == 100)),
            "multi_agent_rows": int(np.sum(multi)),
            "multi_agent_t50_rows": int(np.sum(multi & (labels["horizon"].astype(int) == 50))),
            "groups": int(len(set(map(str, ws["keys"].tolist())))),
        },
        "training": training,
        "gate_selection": selection,
        "proximity_guard_selection": proximity_guard,
        "test_guarded_off": guarded_off,
        "ade_metrics_vs_floor": ade_metrics,
        "fde_metrics_vs_floor": fde_metrics,
        "multi_agent_ade_metrics": multi_ade,
        "multi_agent_fde_metrics": multi_fde,
        "rollout_stats": {"floor": ws["floor_stats"], "selected": ws["selected_stats"]},
        "collision_delta_vs_floor_005": ws["collision_delta_005"],
        "smoothness_jagged_delta": ws["smoothness_jagged_delta"],
        "bootstrap_ade": bootstrap,
        "domain_local_all_agent_world_state_gate": pass_gate,
        "caveat": "Endpoint neural dynamics are audited as a linear waypoint rollout proxy from current point to selected endpoint. This checks same-frame multi-agent proximity and waypoint ADE/FDE, but it is not a learned full-waypoint trajectory model, not Stage5C, not SMC, not metric, and not seconds-level.",
    }


def run_domain_local_all_agent_world_state() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    train = dl._load_split("train")
    val = dl._load_split("val")
    test = dl._load_split("test")
    domains = [str(d) for d in sorted(set(train["domain"].astype(str)) & set(val["domain"].astype(str)) & set(test["domain"].astype(str)))]
    results = {domain: _select_for_domain(domain, train, val, test) for domain in domains}
    pure_train, pure_val, pure_test = dl._pure_ucy_expanded_datasets()
    pure_ucy = _select_for_domain("UCY_expanded", pure_train, pure_val, pure_test, prefiltered=True)
    positive_domains = [domain for domain, row in results.items() if row.get("domain_local_all_agent_world_state_gate")]
    if pure_ucy.get("domain_local_all_agent_world_state_gate"):
        positive_domains.append("UCY_expanded")
    result = {
        "source": "fresh_run",
        "protocol": "domain_local_neural_endpoint_linear_all_agent_world_state_audit",
        "domains": domains,
        "positive_domains": positive_domains,
        "positive_domain_count": int(len(positive_domains)),
        "two_domain_all_agent_world_state_gate": bool(len(positive_domains) >= 2),
        "domain_results": results,
        "pure_ucy_expanded_all_agent_world_state": pure_ucy,
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
            "domain_local_neural_endpoint_retrain": True,
            "linear_endpoint_waypoint_proxy": True,
            "learned_full_waypoint_rollout": False,
            "latent_generative_rollout": False,
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
        },
        "caveat": "This extends domain-local endpoint-FDE evidence to same-frame all-agent endpoint-linear waypoint safety. It strengthens safety evidence but does not replace a true learned full-trajectory neural dynamics audit.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    lines = [
        "# Stage41 Domain-Local All-Agent World-State Audit",
        "",
        "- source: `fresh_run`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        f"- positive domains: `{positive_domains}`",
        f"- two-domain all-agent world-state gate: `{result['two_domain_all_agent_world_state_gate']}`",
        "",
        "| domain | rows train/val/test | ADE all | ADE t50 | ADE t100 | ADE hard | easy | multi all | multi t50 | collision d005 | pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    table_rows = {**results, "UCY_expanded": pure_ucy}
    for domain, row in table_rows.items():
        if row.get("status") != "ok":
            lines.append(f"| `{domain}` | `{row.get('reason')}` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `False` |")
            continue
        m = row["ade_metrics_vs_floor"]
        mm = row["multi_agent_ade_metrics"]
        lines.append(
            f"| `{domain}` | `{row['rows']['train']}/{row['rows']['val']}/{row['rows']['test']}` | "
            f"{float(m.get('all_improvement', 0.0)):.4f} | {float(m.get('t50_improvement', 0.0)):.4f} | "
            f"{float(m.get('t100_improvement', 0.0)):.4f} | {float(m.get('hard_failure_improvement', 0.0)):.4f} | "
            f"{float(m.get('easy_degradation', 0.0)):.4f} | {float(mm.get('all_improvement', 0.0)):.4f} | "
            f"{float(mm.get('t50_improvement', 0.0)):.4f} | {float(row.get('collision_delta_vs_floor_005', 0.0)):.4f} | "
            f"`{row['domain_local_all_agent_world_state_gate']}` |"
        )
    lines.extend(
        [
            "",
            "This audit deliberately uses endpoint-linear waypoints so that a domain-local endpoint neural model can be checked against same-frame all-agent proximity and waypoint ADE/FDE. It is not claimed as a learned full-waypoint neural rollout.",
            "",
            f"- no leakage: `{result['no_leakage']}`",
            f"- claim boundary: `{result['claim_boundary']}`",
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_domain_local_all_agent_world_state() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_domain_local_all_agent_world_state()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_domain_local_all_agent_world_state",
            status,
            started,
            [dl.DATA_DIR / "seq2seq_train.npz", dl.DATA_DIR / "seq2seq_val.npz", dl.DATA_DIR / "seq2seq_test.npz", dl.DATA_DIR / "combined_external.npz"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_domain_local_all_agent_world_state()
