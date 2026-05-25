from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_bounded_neural_blend_dynamics as blend
from src import stage41_composite_tail_evidence as cte
from src import stage41_full_trajectory_world_state as ft
from src import stage41_joint_rollout_consistency as jrc
from src import stage41_teacher_guided_proposal as tgp


OUT_DIR = tgp.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_all_agent_composite_world_state.json"
REPORT_MD = OUT_DIR / "stage41_all_agent_composite_world_state.md"
BOOTSTRAP_N = 1000
SEED = 414141
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


def _metric_ds(labels: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "horizon": labels["horizon"],
        "hard": labels["hard"],
        "failure": labels["failure"],
        "easy": labels["easy"],
        "domain": labels["domain"],
        "candidate_fde": labels["candidate_fde"],
    }


def _group_count(keys: np.ndarray) -> np.ndarray:
    counts = Counter(map(str, keys.tolist()))
    return np.asarray([counts[str(k)] for k in keys], dtype=np.int32)


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


def _by_domain_metrics(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    domain = labels["domain"].astype(str)
    for name in sorted(set(domain.tolist())):
        local = mask & (domain == name)
        out[name] = _safe_metrics(selected, floor, labels, switch, local)
    return out


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


def _selected_world_state(data: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, np.ndarray]:
    alpha = blend._alpha_vector(data, policy)
    floor_xy = data["floor_xy"].astype(np.float64)
    neural_xy = data["neural_xy"].astype(np.float64)
    selected_xy = floor_xy + alpha[:, None, None] * (neural_xy - floor_xy)
    floor_ade, floor_fde = ft._trajectory_errors(floor_xy, data["labels"])
    selected_ade, selected_fde = ft._trajectory_errors(selected_xy, data["labels"])
    neural_ade, neural_fde = ft._trajectory_errors(neural_xy, data["labels"])
    return {
        "alpha": alpha.astype(np.float64),
        "switch": (alpha > EPS),
        "floor_xy": floor_xy,
        "neural_xy": neural_xy,
        "selected_xy": selected_xy,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "neural_ade": neural_ade,
        "neural_fde": neural_fde,
    }


def _group_summary(keys: np.ndarray, switch: np.ndarray) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        groups[str(key)].append(i)
    sizes = np.asarray([len(v) for v in groups.values()], dtype=np.int32)
    group_switch = np.asarray([np.any(switch[np.asarray(v, dtype=np.int64)]) for v in groups.values()], dtype=bool) if groups else np.asarray([], dtype=bool)
    mixed_switch = np.asarray(
        [np.any(switch[np.asarray(v, dtype=np.int64)]) and not np.all(switch[np.asarray(v, dtype=np.int64)]) for v in groups.values()],
        dtype=bool,
    ) if groups else np.asarray([], dtype=bool)
    return {
        "groups": int(len(groups)),
        "multi_agent_groups": int(np.sum(sizes >= 2)) if len(sizes) else 0,
        "mean_group_size": float(np.mean(sizes)) if len(sizes) else 0.0,
        "p95_group_size": float(np.percentile(sizes, 95)) if len(sizes) else 0.0,
        "group_switch_rate": float(np.mean(group_switch)) if len(group_switch) else 0.0,
        "mixed_group_switch_rate": float(np.mean(mixed_switch)) if len(mixed_switch) else 0.0,
    }


def run_all_agent_composite_world_state() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    evidence = read_json(cte.REPORT_JSON, {})
    policy = evidence.get("policy") or {}
    if not policy:
        cte.run_composite_tail_evidence()
        evidence = read_json(cte.REPORT_JSON, {})
        policy = evidence.get("policy") or {}
    checkpoint, teacher_policy, min_sep = blend._load_frozen_model()
    test = blend._bundle("test", checkpoint, teacher_policy, min_sep)
    selected = _selected_world_state(test, policy)
    labels = test["labels"]
    keys = test["keys"]
    switch = selected["switch"]
    group_counts = _group_count(keys)
    multi = group_counts >= 2
    full_waypoint = np.all(labels["waypoint_valid"].astype(bool), axis=1)
    any_waypoint = np.any(labels["waypoint_valid"].astype(bool), axis=1)

    ade_metrics = s41._metrics(selected["selected_ade"], selected["floor_ade"], _metric_ds(labels), switch)
    fde_metrics = s41._metrics(selected["selected_fde"], selected["floor_fde"], _metric_ds(labels), switch)
    multi_ade = _safe_metrics(selected["selected_ade"], selected["floor_ade"], labels, switch, multi)
    multi_fde = _safe_metrics(selected["selected_fde"], selected["floor_fde"], labels, switch, multi)
    full_waypoint_ade = _safe_metrics(selected["selected_ade"], selected["floor_ade"], labels, switch, full_waypoint)
    floor_stats = jrc._joint_stats("floor", selected["floor_xy"], labels, keys, np.zeros(len(switch), dtype=bool))
    neural_stats = jrc._joint_stats("neural_without_fallback", selected["neural_xy"], labels, keys, np.ones(len(switch), dtype=bool))
    selected_stats = jrc._joint_stats("composite_tail_world_state", selected["selected_xy"], labels, keys, switch)
    collision_delta = float(selected_stats["near_collision_rate_005"] - floor_stats["near_collision_rate_005"])
    smoothness_delta = float(selected_stats["smoothness"]["jagged_rate"] - floor_stats["smoothness"]["jagged_rate"])
    bootstrap = _bootstrap_report(selected["selected_ade"], selected["floor_ade"], labels, multi)
    by_domain_multi = _by_domain_metrics(selected["selected_ade"], selected["floor_ade"], labels, switch, multi)

    pass_gate = bool(
        ade_metrics.get("all_improvement", 0.0) > 0
        and ade_metrics.get("t50_improvement", 0.0) > 0
        and ade_metrics.get("t100_improvement", 0.0) > 0
        and ade_metrics.get("hard_failure_improvement", 0.0) > 0
        and ade_metrics.get("easy_degradation", 1.0) <= 0.02
        and multi_ade.get("all_improvement", 0.0) > 0
        and multi_ade.get("t50_improvement", 0.0) > 0
        and multi_ade.get("hard_failure_improvement", 0.0) > 0
        and fde_metrics.get("all_improvement", 0.0) > 0
        and fde_metrics.get("t50_improvement", 0.0) > 0
        and collision_delta <= blend.TEST_COLLISION_CEILING
        and smoothness_delta <= 0.01
        and (bootstrap.get("all") or {}).get("low", 0.0) > 0
        and (bootstrap.get("multi_agent") or {}).get("low", 0.0) > 0
    )

    result = {
        "source": "fresh_run",
        "protocol": "composite_tail_all_agent_full_world_state_audit",
        "policy": policy,
        "checkpoint": checkpoint,
        "rows": int(len(switch)),
        "waypoint_fractions": ft.WAYPOINT_FRAC.tolist(),
        "coverage": {
            "any_waypoint_rows": int(np.sum(any_waypoint)),
            "full_waypoint_rows": int(np.sum(full_waypoint)),
            "multi_agent_rows": int(np.sum(multi)),
            "multi_agent_full_waypoint_rows": int(np.sum(multi & full_waypoint)),
            "t50_rows": int(np.sum(labels["horizon"].astype(int) == 50)),
            "t100_rows": int(np.sum(labels["horizon"].astype(int) == 100)),
        },
        "group_summary": _group_summary(keys, switch),
        "ade_metrics_vs_floor": ade_metrics,
        "fde_metrics_vs_floor": fde_metrics,
        "multi_agent_ade_metrics": multi_ade,
        "multi_agent_fde_metrics": multi_fde,
        "full_waypoint_ade_metrics": full_waypoint_ade,
        "by_domain_multi_agent_ade_metrics": by_domain_multi,
        "neural_without_fallback_ade_metrics": s41._metrics(selected["neural_ade"], selected["floor_ade"], _metric_ds(labels), np.ones(len(switch), dtype=bool)),
        "rollout_stats": {
            "floor": floor_stats,
            "neural_without_fallback": neural_stats,
            "composite_tail_selected": selected_stats,
        },
        "collision_delta_vs_floor_005": collision_delta,
        "smoothness_jagged_delta": smoothness_delta,
        "alpha_stats": {
            "mean": float(np.mean(selected["alpha"])),
            "positive_rate": float(np.mean(switch)),
            "full_switch_alpha_rate": float(np.mean(selected["alpha"] >= float(policy.get("switch_alpha", 1.0)) - EPS)),
            "tail_alpha_rate": float(np.mean((selected["alpha"] > EPS) & (selected["alpha"] < float(policy.get("switch_alpha", 1.0)) - EPS))),
        },
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_ade": bootstrap,
        "all_agent_composite_world_state_pass": pass_gate,
        "claim_boundary": {
            "full_active_agent_future_world_state_under_safety_floor": pass_gate,
            "ungated_no_fallback_neural_rollout_safe": False,
            "latent_generative_rollout": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "metric_or_seconds_claim": False,
        },
        "no_leakage": {
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "policy_frozen_before_test": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "caveat": "This audits frozen composite-tail bounded full-waypoint rollout for all active rows grouped by current frame. It is protected 2.5D dataset-local raw-frame world-state evidence, not ungated neural rollout, Stage5C, SMC, metric prediction, or true 3D.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    write_md(
        REPORT_MD,
        [
            "# Stage41 All-Agent Composite-Tail World-State Audit",
            "",
            "- source: `fresh_run`",
            f"- pass: `{pass_gate}`",
            f"- rows: `{result['rows']}`",
            f"- coverage: `{result['coverage']}`",
            f"- group summary: `{result['group_summary']}`",
            f"- ADE metrics vs floor: `{ade_metrics}`",
            f"- FDE metrics vs floor: `{fde_metrics}`",
            f"- multi-agent ADE metrics: `{multi_ade}`",
            f"- multi-agent FDE metrics: `{multi_fde}`",
            f"- collision delta @0.05 normalized: `{collision_delta}`",
            f"- smoothness jagged delta: `{smoothness_delta}`",
            f"- bootstrap ADE: `{bootstrap}`",
            f"- alpha stats: `{result['alpha_stats']}`",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "The audit applies the already frozen composite-tail policy to full future waypoint rollouts for every active row in same-frame multi-agent groups. It does not reselect thresholds on test and does not execute Stage5C or SMC.",
        ],
    )
    return result


def main_all_agent_composite_world_state() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_all_agent_composite_world_state()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_all_agent_composite_world_state",
            status,
            started,
            [cte.REPORT_JSON, blend.REPORT_JSON, OUT_DIR / "stage41_full_trajectory_world_state.json"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_all_agent_composite_world_state()
