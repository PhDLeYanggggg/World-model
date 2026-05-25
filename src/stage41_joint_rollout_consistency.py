from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src import stage41_breakthrough as s41
from src import stage41_full_trajectory_world_state as ft
from src import stage41_joint_multiagent_consistency as jmc
from src import stage41_joint_policy_distillation as jpd


OUT_DIR = jpd.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_joint_rollout_consistency.json"
REPORT_MD = OUT_DIR / "stage41_joint_rollout_consistency.md"
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


def _group_counts(keys: np.ndarray) -> np.ndarray:
    counts = Counter(map(str, keys.tolist()))
    return np.asarray([counts[str(k)] for k in keys], dtype=np.int32)


def _subset_metrics(selected_ade: np.ndarray, floor_ade: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
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
    ds = {
        "horizon": labels["horizon"][mask],
        "hard": labels["hard"][mask],
        "failure": labels["failure"][mask],
        "easy": labels["easy"][mask],
        "domain": labels["domain"][mask],
        "candidate_fde": labels["candidate_fde"][mask],
    }
    return s41._metrics(selected_ade[mask], floor_ade[mask], ds, switch[mask])


def _rollout_smoothness(xy: np.ndarray, labels: Mapping[str, np.ndarray]) -> dict[str, float]:
    current = labels["current_xy"].astype(np.float64)
    pts = np.concatenate([current[:, None, :], xy.astype(np.float64)], axis=1)
    seg = np.linalg.norm(np.diff(pts, axis=1), axis=2)
    normalizer = np.maximum(labels["normalizer"].astype(np.float64), EPS)
    norm_seg = seg / normalizer[:, None]
    max_step = np.max(norm_seg, axis=1)
    median_step = np.median(norm_seg, axis=1)
    jagged = max_step > np.maximum(4.0 * median_step + EPS, 0.50)
    return {
        "mean_max_normalized_step": float(np.mean(max_step)),
        "jagged_rate": float(np.mean(jagged)),
    }


def _joint_stats(name: str, xy: np.ndarray, labels: Mapping[str, np.ndarray], keys: np.ndarray, switch: np.ndarray) -> dict[str, Any]:
    group_count = _group_counts(keys)
    multi = group_count >= 2
    min_dist = jmc._min_group_distance(xy, keys, labels["normalizer"].astype(np.float64))
    finite = np.isfinite(min_dist)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        groups[str(key)].append(i)
    group_switch = []
    mixed_switch = []
    all_switch = []
    no_switch = []
    for members in groups.values():
        sw = switch[np.asarray(members, dtype=np.int64)]
        group_switch.append(bool(np.any(sw)))
        mixed_switch.append(bool(np.any(sw) and not np.all(sw)))
        all_switch.append(bool(np.all(sw)))
        no_switch.append(bool(not np.any(sw)))
    return {
        "name": name,
        "rows": int(len(keys)),
        "groups": int(len(groups)),
        "multi_agent_rows": int(np.sum(multi)),
        "multi_agent_groups": int(sum(1 for members in groups.values() if len(members) >= 2)),
        "mean_group_size": float(np.mean(list(Counter(map(str, keys.tolist())).values()))) if len(groups) else 0.0,
        "switch_rate": float(np.mean(switch)) if len(switch) else 0.0,
        "group_switch_rate": float(np.mean(group_switch)) if group_switch else 0.0,
        "mixed_group_switch_rate": float(np.mean(mixed_switch)) if mixed_switch else 0.0,
        "all_switch_group_rate": float(np.mean(all_switch)) if all_switch else 0.0,
        "no_switch_group_rate": float(np.mean(no_switch)) if no_switch else 0.0,
        "mean_min_group_distance": float(np.mean(min_dist[finite])) if np.any(finite) else None,
        "p05_min_group_distance": float(np.percentile(min_dist[finite], 5)) if np.any(finite) else None,
        "near_collision_rate_002": float(np.mean(min_dist[finite] < 0.02)) if np.any(finite) else 0.0,
        "near_collision_rate_005": float(np.mean(min_dist[finite] < 0.05)) if np.any(finite) else 0.0,
        "smoothness": _rollout_smoothness(xy, labels),
    }


def _by_domain_joint_stats(xy: np.ndarray, labels: Mapping[str, np.ndarray], keys: np.ndarray, switch: np.ndarray) -> dict[str, Any]:
    domain = labels["domain"].astype(str)
    out: dict[str, Any] = {}
    for name in sorted(set(domain.tolist())):
        mask = domain == name
        out[name] = _joint_stats(name, xy[mask], {k: v[mask] for k, v in labels.items() if isinstance(v, np.ndarray) and len(v) == len(mask)}, keys[mask], switch[mask])
    return out


def _rollout_inputs(checkpoint: str | Path, policy: Mapping[str, Any], split: str) -> dict[str, Any]:
    pred, labels, _ref = jmc._split_predictions(split)
    scores, distill_data = jpd._predict_checkpoint(checkpoint, split)
    selected_ade, _selected_fde, switch = jpd._apply_policy(scores, distill_data, policy)
    meta = jmc._group_metadata(split)
    keys = meta["key"]
    floor_xy = ft._floor_waypoints(labels)
    neural_xy = ft._pred_waypoints(pred, labels)
    floor_ade, _floor_fde = ft._trajectory_errors(floor_xy, labels)
    return {
        "pred": pred,
        "labels": labels,
        "keys": keys,
        "floor_xy": floor_xy,
        "neural_xy": neural_xy,
        "floor_ade": floor_ade,
        "policy_selected_ade": selected_ade,
        "policy_switch": switch.astype(bool),
    }


def _apply_proximity_guard(
    floor_xy: np.ndarray,
    neural_xy: np.ndarray,
    labels: Mapping[str, np.ndarray],
    keys: np.ndarray,
    switch: np.ndarray,
    min_sep: float,
    margin: float = 0.0,
) -> tuple[np.ndarray, int]:
    if min_sep <= 0:
        return switch.copy(), 0
    selected_xy = floor_xy.copy()
    selected_xy[switch] = neural_xy[switch]
    normalizer = labels["normalizer"].astype(np.float64)
    floor_min = jmc._min_group_distance(floor_xy, keys, normalizer)
    selected_min = jmc._min_group_distance(selected_xy, keys, normalizer)
    guard = switch & np.isfinite(selected_min) & (selected_min < min_sep) & (selected_min + margin < floor_min)
    out = switch.copy()
    out[guard] = False
    return out, int(np.sum(guard))


def _evaluate_split_rollout(bundle: Mapping[str, Any], switch: np.ndarray, name: str) -> dict[str, Any]:
    labels = bundle["labels"]
    keys = bundle["keys"]
    floor_xy = bundle["floor_xy"]
    neural_xy = bundle["neural_xy"]
    selected_xy = floor_xy.copy()
    selected_xy[switch] = neural_xy[switch]
    selected_ade_from_xy, _selected_fde_from_xy = ft._trajectory_errors(selected_xy, labels)
    floor_ade = bundle["floor_ade"]
    neural_ade, _neural_fde = ft._trajectory_errors(neural_xy, labels)
    group_count = _group_counts(keys)
    multi = group_count >= 2

    floor_stats = _joint_stats("floor", floor_xy, labels, keys, np.zeros(len(keys), dtype=bool))
    neural_stats = _joint_stats("neural_without_fallback", neural_xy, labels, keys, np.ones(len(keys), dtype=bool))
    selected_stats = _joint_stats(name, selected_xy, labels, keys, switch)
    selected_metrics = _subset_metrics(selected_ade_from_xy, floor_ade, labels, switch, np.ones(len(keys), dtype=bool))
    multi_metrics = _subset_metrics(selected_ade_from_xy, floor_ade, labels, switch, multi)
    neural_metrics = _subset_metrics(neural_ade, floor_ade, labels, np.ones(len(keys), dtype=bool), np.ones(len(keys), dtype=bool))
    by_domain_multi = {}
    domain = labels["domain"].astype(str)
    for d in sorted(set(domain.tolist())):
        mask = (domain == d) & multi
        by_domain_multi[d] = _subset_metrics(selected_ade_from_xy, floor_ade, labels, switch, mask)
    collision_delta_005 = selected_stats["near_collision_rate_005"] - floor_stats["near_collision_rate_005"]
    return {
        "selected_xy": selected_xy,
        "selected_ade": selected_ade_from_xy,
        "selected_metrics": selected_metrics,
        "multi_agent_metrics": multi_metrics,
        "neural_without_fallback_metrics": neural_metrics,
        "by_domain_multi_agent_metrics": by_domain_multi,
        "floor_stats": floor_stats,
        "neural_stats": neural_stats,
        "selected_stats": selected_stats,
        "collision_delta_005": float(collision_delta_005),
    }


def _guard_score(metrics: Mapping[str, Any], collision_delta_005: float) -> float:
    return (
        1.0 * float(metrics.get("all_improvement", 0.0))
        + 1.4 * float(metrics.get("t50_improvement", 0.0))
        + 1.1 * float(metrics.get("hard_failure_improvement", 0.0))
        - 30.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 8.0 * max(0.0, collision_delta_005 - 0.01)
    )


def _select_guard(checkpoint: str | Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    val = _rollout_inputs(checkpoint, policy, "val")
    candidates: list[dict[str, Any]] = []
    for min_sep in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
        guarded, guarded_off = _apply_proximity_guard(
            val["floor_xy"],
            val["neural_xy"],
            val["labels"],
            val["keys"],
            val["policy_switch"],
            min_sep,
        )
        ev = _evaluate_split_rollout(val, guarded, f"val_guard_{min_sep}")
        m = ev["selected_metrics"]
        eligible = bool(
            m.get("all_improvement", 0.0) > 0
            and m.get("t50_improvement", 0.0) > 0
            and m.get("hard_failure_improvement", 0.0) > 0
            and m.get("easy_degradation", 1.0) <= 0.02
            and ev["collision_delta_005"] <= 0.01
        )
        candidates.append(
            {
                "min_sep": min_sep,
                "guarded_off": guarded_off,
                "eligible": eligible,
                "score": _guard_score(m, ev["collision_delta_005"]),
                "val_metrics": m,
                "val_collision_delta_005": ev["collision_delta_005"],
            }
        )
    eligible = [row for row in candidates if row["eligible"]]
    pool = eligible if eligible else candidates
    best = max(pool, key=lambda row: row["score"])
    return {
        "source": "validation_selected",
        "selected": {"min_sep": best["min_sep"], "eligible": best["eligible"], "score": best["score"], "guarded_off": best["guarded_off"]},
        "candidates": candidates,
    }


def run_joint_rollout_consistency() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    distiller = read_json(OUT_DIR / "stage41_joint_policy_distillation.json", {})
    repair = read_json(OUT_DIR / "stage41_ucy_fallback_repair.json", {})
    if not distiller:
        raise FileNotFoundError("Run stage41_joint_policy_distillation first.")
    checkpoint = distiller["best_checkpoint"]
    policy = repair.get("repaired_policy") or distiller.get("best_policy") or {}
    policy_source = "ucy_repaired_policy" if repair.get("repaired_policy") else "joint_distiller_policy"

    guard = _select_guard(checkpoint, policy)
    test = _rollout_inputs(checkpoint, policy, "test")
    guarded_switch, guarded_off = _apply_proximity_guard(
        test["floor_xy"],
        test["neural_xy"],
        test["labels"],
        test["keys"],
        test["policy_switch"],
        float((guard.get("selected") or {}).get("min_sep", 0.0)),
    )
    raw_eval = _evaluate_split_rollout(test, test["policy_switch"], "raw_policy_joint_rollout")
    guarded_eval = _evaluate_split_rollout(test, guarded_switch, "selected_joint_rollout")
    labels = test["labels"]
    keys = test["keys"]
    selected_metrics = guarded_eval["selected_metrics"]
    multi_metrics = guarded_eval["multi_agent_metrics"]
    neural_metrics = guarded_eval["neural_without_fallback_metrics"]
    selected_stats = guarded_eval["selected_stats"]
    floor_stats = guarded_eval["floor_stats"]
    neural_stats = guarded_eval["neural_stats"]
    selected_ade_from_xy = guarded_eval["selected_ade"]
    joint_rollout_pass = bool(
        selected_metrics.get("all_improvement", 0.0) > 0
        and selected_metrics.get("t50_improvement", 0.0) > 0
        and multi_metrics.get("all_improvement", 0.0) > 0
        and multi_metrics.get("hard_failure_improvement", 0.0) > 0
        and selected_metrics.get("easy_degradation", 1.0) <= 0.02
        and guarded_eval["collision_delta_005"] <= 0.01
    )
    result = {
        "source": "fresh_run",
        "policy_source": policy_source,
        "selected_policy": policy,
        "validation_selected_proximity_guard": guard,
        "test_guarded_off": guarded_off,
        "rows": int(len(keys)),
        "multi_agent_rows": int(guarded_eval["multi_agent_metrics"].get("rows", 0)),
        "ade_alignment_max_abs_error_raw_policy": float(np.max(np.abs(raw_eval["selected_ade"] - test["policy_selected_ade"]))) if len(keys) else 0.0,
        "raw_policy_metrics": raw_eval["selected_metrics"],
        "raw_policy_collision_delta_vs_floor_005": raw_eval["collision_delta_005"],
        "selected_metrics": selected_metrics,
        "multi_agent_metrics": multi_metrics,
        "neural_without_fallback_metrics": neural_metrics,
        "by_domain_multi_agent_metrics": guarded_eval["by_domain_multi_agent_metrics"],
        "group_switch_summary": {
            "selected_group_switch_rate": selected_stats["group_switch_rate"],
            "selected_mixed_group_switch_rate": selected_stats["mixed_group_switch_rate"],
            "selected_all_switch_group_rate": selected_stats["all_switch_group_rate"],
            "selected_no_switch_group_rate": selected_stats["no_switch_group_rate"],
        },
        "rollout_stats": {
            "floor": floor_stats,
            "neural_without_fallback": neural_stats,
            "selected": selected_stats,
            "by_domain_selected": _by_domain_joint_stats(guarded_eval["selected_xy"], labels, keys, guarded_switch),
        },
        "collision_delta_vs_floor_005": guarded_eval["collision_delta_005"],
        "joint_rollout_consistency_pass": joint_rollout_pass,
        "no_leakage": {
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "test_threshold_tuning": False,
            "proximity_guard_selected_on_val": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "caveat": "This is an all-agent grouped rollout consistency audit over the repaired policy. It is still not a latent generative rollout and does not execute Stage5C or SMC.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    write_md(
        REPORT_MD,
        [
            "# Stage41 Joint Rollout Consistency Audit",
            "",
            "- source: `fresh_run`",
            f"- policy source: `{policy_source}`",
            f"- joint rollout consistency pass: `{joint_rollout_pass}`",
            f"- validation-selected proximity guard: `{guard.get('selected')}`",
            f"- test guarded-off rows: `{guarded_off}`",
            f"- raw policy metrics: `{raw_eval['selected_metrics']}`",
            f"- raw policy collision delta @0.05 normalized: `{raw_eval['collision_delta_005']}`",
            f"- selected metrics: `{selected_metrics}`",
            f"- multi-agent metrics: `{multi_metrics}`",
            f"- neural without fallback metrics: `{neural_metrics}`",
            f"- collision delta vs floor @0.05 normalized: `{guarded_eval['collision_delta_005']}`",
            f"- rollout stats: `{result['rollout_stats']}`",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "The selected policy predicts grouped all-agent future waypoint rollouts under the Stage37 safety floor. It remains dataset-local raw-frame 2.5D and is not Stage5C/SMC.",
        ],
    )
    return result


def main_joint_rollout_consistency() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_joint_rollout_consistency()
        status = "ok"
    finally:
        jpd._append_ledger(
            "stage41_joint_rollout_consistency",
            status,
            started,
            [OUT_DIR / "stage41_ucy_fallback_repair.json", OUT_DIR / "stage41_joint_policy_distillation.json"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_joint_rollout_consistency()
