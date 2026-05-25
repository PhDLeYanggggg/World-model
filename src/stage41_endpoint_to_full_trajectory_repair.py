from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_domain_local_all_agent_world_state as dlaa
from src import stage41_domain_local_neural_retrain as dl
from src import stage41_full_trajectory_world_state as ft
from src import stage41_joint_multiagent_consistency as jmc
from src import stage41_joint_rollout_consistency as jrc


OUT_DIR = dl.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_endpoint_to_full_trajectory_repair.json"
REPORT_MD = OUT_DIR / "stage41_endpoint_to_full_trajectory_repair.md"
SEED = 41643
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


def _subset(data: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.shape[:1] == mask.shape else v) for k, v in data.items()}


def _align_full_labels(split: str, endpoint_data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    del split
    tracks = ft._track_map(endpoint_data["source_file"].astype(str).tolist())
    n = len(endpoint_data["horizon"])
    if "agent_id" in endpoint_data and "frame_id" in endpoint_data:
        agent_id = endpoint_data["agent_id"].astype(np.int64)
        frame_id = endpoint_data["frame_id"].astype(np.float64)
    else:
        combined = dl._load_combined()
        ids = endpoint_data["ids"].astype(np.int64)
        agent_id = combined["agent_id"].astype(np.int64)[ids]
        frame_id = combined["frame_id"].astype(np.float64)[ids]
    waypoints = np.zeros((n, len(ft.WAYPOINT_FRAC), 2), dtype=np.float64)
    valid = np.zeros((n, len(ft.WAYPOINT_FRAC)), dtype=bool)
    for i in range(n):
        source = str(endpoint_data["source_file"][i])
        agent = int(agent_id[i])
        track = tracks.get((source, agent))
        if track is None:
            waypoints[i, -1] = endpoint_data["future_xy"][i].astype(np.float64)
            valid[i, -1] = True
            continue
        pts, mask = ft._lookup_waypoints(
            track,
            float(frame_id[i]),
            int(endpoint_data["horizon"][i]),
            endpoint_data["future_xy"][i].astype(np.float32),
        )
        waypoints[i] = pts.astype(np.float64)
        valid[i] = mask
    return {
        "waypoint_xy": waypoints,
        "waypoint_valid": valid,
        "current_xy": endpoint_data["current_xy"].astype(np.float64),
        "normalizer": endpoint_data["normalizer"].astype(np.float64),
        "horizon": endpoint_data["horizon"].astype(np.int16),
        "hard": endpoint_data["hard"].astype(bool),
        "failure": endpoint_data["failure"].astype(bool),
        "easy": endpoint_data["easy"].astype(bool),
        "domain": endpoint_data["domain"].astype("U32"),
        "candidate_fde": endpoint_data["candidate_fde"].astype(np.float64),
        "ids": endpoint_data["ids"].astype(np.int64),
        "source_file": endpoint_data["source_file"].astype(str),
        "scene_id": endpoint_data["scene_id"].astype(str),
    }


def _linear_waypoints_from_delta(data: Mapping[str, np.ndarray], delta: np.ndarray) -> np.ndarray:
    endpoint = data["current_xy"].astype(np.float64) + delta.astype(np.float64) * data["normalizer"].astype(np.float64)[:, None]
    return dlaa._linear_waypoints(data["current_xy"], endpoint)


def _metric_ds(labels: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "horizon": labels["horizon"],
        "hard": labels["hard"],
        "failure": labels["failure"],
        "easy": labels["easy"],
        "domain": labels["domain"],
        "candidate_fde": labels["candidate_fde"],
    }


def _metrics(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray) -> dict[str, Any]:
    return s41._metrics(selected.astype(np.float64), floor.astype(np.float64), _metric_ds(labels), switch.astype(bool))


def _group_keys(data: Mapping[str, np.ndarray]) -> np.ndarray:
    return dlaa._group_keys(data)


def _eval_world_state(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], selected_delta: np.ndarray, switch: np.ndarray) -> dict[str, Any]:
    floor_delta = data["cand_delta"][:, 0, :].astype(np.float64)
    floor_xy = _linear_waypoints_from_delta(data, floor_delta)
    selected_xy = _linear_waypoints_from_delta(data, selected_delta)
    floor_ade, floor_fde = ft._trajectory_errors(floor_xy, labels)
    selected_ade, selected_fde = ft._trajectory_errors(selected_xy, labels)
    keys = _group_keys(data)
    counts = Counter(map(str, keys.tolist()))
    multi = np.asarray([counts[str(k)] >= 2 for k in keys], dtype=bool)
    floor_stats = jrc._joint_stats("floor", floor_xy, labels, keys, np.zeros(len(switch), dtype=bool))
    selected_stats = jrc._joint_stats("endpoint_to_full_selected", selected_xy, labels, keys, switch.astype(bool))
    return {
        "floor_xy": floor_xy,
        "selected_xy": selected_xy,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "keys": keys,
        "multi": multi,
        "ade_metrics": _metrics(selected_ade, floor_ade, labels, switch),
        "fde_metrics": _metrics(selected_fde, floor_fde, labels, switch),
        "multi_ade_metrics": dlaa._safe_metrics(selected_ade, floor_ade, labels, switch, multi),
        "floor_stats": floor_stats,
        "selected_stats": selected_stats,
        "collision_delta_005": float(selected_stats["near_collision_rate_005"] - floor_stats["near_collision_rate_005"]),
        "smoothness_jagged_delta": float(selected_stats["smoothness"]["jagged_rate"] - floor_stats["smoothness"]["jagged_rate"]),
    }


def _apply_endpoint_policy(data: Mapping[str, np.ndarray], pred: Mapping[str, np.ndarray], gate_pred: Mapping[str, np.ndarray], policy: Mapping[str, Any], horizons: set[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    selected_delta, switch = dl._apply_policy(data, pred, gate_pred, policy)
    if horizons is not None:
        allowed = np.isin(data["horizon"].astype(int), sorted(horizons))
        floor_delta = data["cand_delta"][:, 0, :].astype(np.float64)
        selected_delta = selected_delta.copy()
        switch = switch.astype(bool) & allowed
        selected_delta[~allowed] = floor_delta[~allowed]
    return selected_delta, switch.astype(bool)


def _guard(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], selected_delta: np.ndarray, switch: np.ndarray, min_sep: float) -> tuple[np.ndarray, np.ndarray, int]:
    if min_sep <= 0:
        return selected_delta.copy(), switch.copy(), 0
    floor_delta = data["cand_delta"][:, 0, :].astype(np.float64)
    floor_xy = _linear_waypoints_from_delta(data, floor_delta)
    selected_xy = _linear_waypoints_from_delta(data, selected_delta)
    keys = _group_keys(data)
    normalizer = labels["normalizer"].astype(np.float64)
    floor_min = jmc._min_group_distance(floor_xy, keys, normalizer)
    selected_min = jmc._min_group_distance(selected_xy, keys, normalizer)
    guard = switch & np.isfinite(selected_min) & (selected_min < min_sep) & (selected_min < floor_min)
    out_delta = selected_delta.copy()
    out_switch = switch.copy()
    out_delta[guard] = floor_delta[guard]
    out_switch[guard] = False
    return out_delta, out_switch, int(np.sum(guard))


def _score(ev: Mapping[str, Any]) -> float:
    m = ev["ade_metrics"]
    mm = ev["multi_ade_metrics"]
    return float(
        m.get("all_improvement", 0.0)
        + 2.0 * m.get("t50_improvement", 0.0)
        + m.get("t100_improvement", 0.0)
        + 1.2 * m.get("hard_failure_improvement", 0.0)
        + 0.5 * mm.get("all_improvement", 0.0)
        - 35.0 * max(0.0, m.get("easy_degradation", 1.0) - 0.02)
        - 8.0 * max(0.0, ev["collision_delta_005"] - 0.01)
    )


def _eligible(ev: Mapping[str, Any]) -> bool:
    m = ev["ade_metrics"]
    mm = ev["multi_ade_metrics"]
    f = ev["fde_metrics"]
    return bool(
        m.get("all_improvement", 0.0) > 0.0
        and m.get("t50_improvement", 0.0) > 0.0
        and m.get("hard_failure_improvement", 0.0) > 0.0
        and m.get("easy_degradation", 1.0) <= 0.02
        and mm.get("all_improvement", 0.0) > 0.0
        and f.get("all_improvement", 0.0) > 0.0
        and f.get("t50_improvement", 0.0) > 0.0
        and ev["collision_delta_005"] <= 0.01
        and ev["smoothness_jagged_delta"] <= 0.01
    )


def _select_policy_on_val(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], pred: Mapping[str, np.ndarray], gate_pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    endpoint_selection = dl._select_gate_policy(data, pred, gate_pred)
    seed_policies = [endpoint_selection["selected"]["policy"]]
    seed_policies.extend(row["policy"] for row in endpoint_selection.get("top_candidates", [])[:8])
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for policy in seed_policies:
        key = json.dumps(_jsonable(policy), sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(policy)
    rows = []
    for base in unique:
        for variant, horizons in {"all_horizons": None, "t50_only": {50}, "long_horizon": {50, 100}}.items():
            selected_delta, switch = _apply_endpoint_policy(data, pred, gate_pred, base, horizons)
            for min_sep in [0.0, 0.05, 0.12]:
                gd, gs, guarded = _guard(data, labels, selected_delta, switch, min_sep)
                ev = _eval_world_state(data, labels, gd, gs)
                rows.append(
                    {
                        "policy": _jsonable(base),
                        "variant": variant,
                        "min_sep": min_sep,
                        "guarded_off": guarded,
                        "eligible": _eligible(ev),
                        "score": _score(ev),
                        "metrics": {
                            "ade": ev["ade_metrics"],
                            "fde": ev["fde_metrics"],
                            "multi_ade": ev["multi_ade_metrics"],
                            "collision_delta_005": ev["collision_delta_005"],
                            "smoothness_jagged_delta": ev["smoothness_jagged_delta"],
                        },
                    }
                )
    pool = [r for r in rows if r["eligible"]] or rows
    selected = max(
        pool,
        key=lambda r: (
            bool(r["eligible"]),
            bool(float(r["min_sep"]) >= 0.05),
            bool(r["metrics"]["ade"].get("t50_improvement", 0.0) > 0),
            r["score"],
        ),
    )
    return {
        "selected": selected,
        "candidate_count": len(rows),
        "eligible_count": int(sum(r["eligible"] for r in rows)),
        "endpoint_seed_candidate_count": len(unique),
        "endpoint_selection": endpoint_selection["selected"],
        "top_candidates": sorted(rows, key=lambda r: r["score"], reverse=True)[:10],
    }


def _domain_data(split: str, domain: str) -> dict[str, np.ndarray]:
    data = dl._load_split(split)
    return _subset(data, dl._domain_mask(data, domain))


def _evaluate_domain(domain: str) -> dict[str, Any]:
    train = _domain_data("train", domain)
    val = _domain_data("val", domain)
    test = _domain_data("test", domain)
    if min(len(train["horizon"]), len(val["horizon"]), len(test["horizon"])) < 500:
        return {"domain": domain, "status": "not_run", "reason": "not enough domain rows"}
    training = dl._train_endpoint(domain, train, val)
    pred_train = dl._predict_endpoint(training["checkpoint"], train)
    pred_val = dl._predict_endpoint(training["checkpoint"], val)
    pred_test = dl._predict_endpoint(training["checkpoint"], test)
    fde_train = dl._endpoint_fde(pred_train["delta"], train)
    fde_val = dl._endpoint_fde(pred_val["delta"], val)
    fde_test = dl._endpoint_fde(pred_test["delta"], test)
    gate = dl._train_gate(train, pred_train, fde_train)
    gate_val = dl._predict_gate(gate, val, pred_val, fde_val)
    labels_val = _align_full_labels("val", val)
    selection = _select_policy_on_val(val, labels_val, pred_val, gate_val)
    gate_test = dl._predict_gate(gate, test, pred_test, fde_test)
    selected_delta, switch = _apply_endpoint_policy(
        test,
        pred_test,
        gate_test,
        selection["selected"]["policy"],
        {"all_horizons": None, "t50_only": {50}, "long_horizon": {50, 100}}[selection["selected"]["variant"]],
    )
    labels_test = _align_full_labels("test", test)
    selected_delta, switch, guarded_off = _guard(test, labels_test, selected_delta, switch, float(selection["selected"]["min_sep"]))
    ev = _eval_world_state(test, labels_test, selected_delta, switch)
    return {
        "domain": domain,
        "source": "fresh_run",
        "status": "ok",
        "rows": {"train": int(len(train["horizon"])), "val": int(len(val["horizon"])), "test": int(len(test["horizon"]))},
        "t50_rows": {"train": int(np.sum(train["horizon"] == 50)), "val": int(np.sum(val["horizon"] == 50)), "test": int(np.sum(test["horizon"] == 50))},
        "t100_rows": {"train": int(np.sum(train["horizon"] == 100)), "val": int(np.sum(val["horizon"] == 100)), "test": int(np.sum(test["horizon"] == 100))},
        "training": training,
        "direct_endpoint_without_fallback_test": dl._metrics(fde_test, test["floor_fde"], test, np.ones(len(test["horizon"]), dtype=bool)),
        "selection": selection,
        "test_guarded_off": guarded_off,
        "ade_metrics_vs_floor": ev["ade_metrics"],
        "fde_metrics_vs_floor": ev["fde_metrics"],
        "multi_agent_ade_metrics": ev["multi_ade_metrics"],
        "collision_delta_vs_floor_005": ev["collision_delta_005"],
        "smoothness_jagged_delta": ev["smoothness_jagged_delta"],
        "endpoint_to_full_trajectory_gate": _eligible(ev),
        "caveat": "Endpoint neural prediction is projected to linear waypoints and scored against reconstructed real waypoint labels. This tests endpoint dynamics as a full-trajectory bridge; it is not learned waypoint-shape dynamics, not Stage5C, not SMC, not metric, and not seconds-level.",
    }


def run_endpoint_to_full_trajectory_repair() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ft.build_full_trajectory_labels()
    domains = ["ETH_UCY", "TrajNet"]
    results = {domain: _evaluate_domain(domain) for domain in domains}
    positive = [d for d, r in results.items() if r.get("endpoint_to_full_trajectory_gate")]
    result = {
        "source": "fresh_run",
        "protocol": "endpoint_neural_to_actual_full_waypoint_repair",
        "stage5c_executed": False,
        "smc_enabled": False,
        "metric_or_seconds_claim": False,
        "positive_domains": positive,
        "positive_domain_count": len(positive),
        "two_domain_endpoint_to_full_gate": len(positive) >= 2,
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
            "endpoint_neural_dynamics": True,
            "linear_waypoint_bridge": True,
            "learned_full_waypoint_shape": False,
            "latent_generative_rollout": False,
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
        },
    }
    write_json(REPORT_JSON, _jsonable(result))
    lines = [
        "# Stage41 Endpoint-To-Full-Trajectory Repair",
        "",
        "- source: `fresh_run`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        f"- positive domains: `{positive}`",
        f"- two-domain endpoint-to-full gate: `{result['two_domain_endpoint_to_full_gate']}`",
        "",
        "| domain | variant | all ADE | t50 ADE | t100 ADE | hard ADE | easy | multi all | collision d005 | pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for domain, row in results.items():
        if row.get("status") != "ok":
            lines.append(f"| `{domain}` | `{row.get('reason')}` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `False` |")
            continue
        m = row["ade_metrics_vs_floor"]
        mm = row["multi_agent_ade_metrics"]
        lines.append(
            f"| `{domain}` | `{row['selection']['selected']['variant']}` | "
            f"{m.get('all_improvement', 0.0):.4f} | {m.get('t50_improvement', 0.0):.4f} | {m.get('t100_improvement', 0.0):.4f} | "
            f"{m.get('hard_failure_improvement', 0.0):.4f} | {m.get('easy_degradation', 0.0):.4f} | {mm.get('all_improvement', 0.0):.4f} | "
            f"{row.get('collision_delta_vs_floor_005', 0.0):.4f} | `{row.get('endpoint_to_full_trajectory_gate')}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This repair checks whether domain-local endpoint neural dynamics can bridge the ETH_UCY t50 full-waypoint failure when scored against reconstructed actual waypoint labels.",
            "- It does not claim learned full-waypoint shape dynamics; the rollout between current point and predicted endpoint is linear.",
            f"- no leakage: `{result['no_leakage']}`",
            f"- claim boundary: `{result['claim_boundary']}`",
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_endpoint_to_full_trajectory_repair() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_endpoint_to_full_trajectory_repair()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_endpoint_to_full_trajectory_repair",
            status,
            started,
            [dl.DATA_DIR / "seq2seq_train.npz", dl.DATA_DIR / "seq2seq_val.npz", dl.DATA_DIR / "seq2seq_test.npz", ft.DATA_DIR / "full_trajectory_test.npz"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_endpoint_to_full_trajectory_repair()
