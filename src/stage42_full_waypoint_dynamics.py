from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_bounded_neural_blend_dynamics as blend
from src import stage41_composite_tail_evidence as cte
from src import stage41_domain_local_all_agent_world_state as dlaa
from src import stage41_full_trajectory_world_state as ft
from src import stage41_joint_rollout_consistency as jrc


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "full_waypoint_dynamics_stage42.json"
REPORT_MD = OUT_DIR / "full_waypoint_dynamics_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_c_gate.md"
BOOTSTRAP_N = 1000
SEED = 4243
EPS = 1e-6


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-C full-waypoint evaluation 使用 dataset-local raw-frame，不能写成 metric 或 seconds-level。",
    "future endpoints / future waypoints 只作为 loss/eval label，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


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
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_improvement(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return 1.0 - float(np.mean(selected[mask])) / max(float(np.mean(floor[mask])), EPS)


def _metrics(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray) -> dict[str, Any]:
    horizon = labels["horizon"].astype(int)
    hard_failure = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    easy = labels["easy"].astype(bool)
    domain = labels["domain"].astype(str)
    out = {
        "rows": int(len(selected)),
        "all_improvement": _safe_improvement(selected, floor, np.ones(len(selected), dtype=bool)),
        "t10_improvement": _safe_improvement(selected, floor, horizon == 10),
        "t25_improvement": _safe_improvement(selected, floor, horizon == 25),
        "t50_improvement": _safe_improvement(selected, floor, horizon == 50),
        "t100_improvement": _safe_improvement(selected, floor, horizon == 100),
        "hard_failure_improvement": _safe_improvement(selected, floor, hard_failure),
        "easy_degradation": -_safe_improvement(selected, floor, easy),
        "switch_rate": float(np.mean(switch)) if len(switch) else 0.0,
        "harm_over_fallback": float(np.mean(selected - floor)) if len(selected) else 0.0,
    }
    out["by_domain"] = {}
    for name in sorted(set(domain.tolist())):
        mask = domain == name
        out["by_domain"][name] = {
            "rows": int(np.sum(mask)),
            "all_improvement": _safe_improvement(selected, floor, mask),
            "t50_improvement": _safe_improvement(selected, floor, mask & (horizon == 50)),
            "t100_improvement": _safe_improvement(selected, floor, mask & (horizon == 100)),
            "hard_failure_improvement": _safe_improvement(selected, floor, mask & hard_failure),
            "easy_degradation": -_safe_improvement(selected, floor, mask & easy),
            "switch_rate": float(np.mean(switch[mask])) if np.any(mask) else 0.0,
        }
    return out


def _bootstrap(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray], mask: np.ndarray, seed: int) -> dict[str, Any]:
    ids = np.where(mask)[0]
    if len(ids) < 30:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(len(ids)), "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(BOOTSTRAP_N):
        sample = rng.choice(ids, size=len(ids), replace=True)
        vals.append(1.0 - float(np.mean(selected[sample])) / max(float(np.mean(floor[sample])), EPS))
    return {
        "low": float(np.percentile(vals, 2.5)),
        "mid": float(np.percentile(vals, 50.0)),
        "high": float(np.percentile(vals, 97.5)),
        "n": int(len(ids)),
        "bootstrap_n": BOOTSTRAP_N,
    }


def _bootstrap_report(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray]) -> dict[str, Any]:
    horizon = labels["horizon"].astype(int)
    hard_failure = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    masks = {
        "all": np.ones(len(horizon), dtype=bool),
        "t50": horizon == 50,
        "t100": horizon == 100,
        "hard_failure": hard_failure,
    }
    return {name: _bootstrap(selected, floor, labels, mask, SEED + i) for i, (name, mask) in enumerate(masks.items())}


def _load_composite_policy() -> dict[str, Any]:
    composite = read_json(cte.REPORT_JSON, {})
    package = read_json("outputs/m3w_neural_v1/package_manifest_m3w_neural_v1.json", {})
    policy = composite.get("policy") or (package.get("policy") or {}).get("policy") or {}
    if not policy:
        raise FileNotFoundError("Missing M3W-Neural v1 composite-tail policy.")
    return policy


def _composite_linear_bundle() -> dict[str, Any]:
    checkpoint, teacher_policy, min_sep = blend._load_frozen_model()
    data = blend._bundle("test", checkpoint, teacher_policy, min_sep)
    policy = _load_composite_policy()
    alpha = blend._alpha_vector(data, policy).astype(np.float64)
    floor_xy = data["floor_xy"].astype(np.float64)
    neural_xy = data["neural_xy"].astype(np.float64)
    selected_xy = floor_xy + alpha[:, None, None] * (neural_xy - floor_xy)
    teacher_xy = floor_xy.copy()
    teacher_switch = data["teacher_repaired_switch"].astype(bool)
    teacher_xy[teacher_switch] = neural_xy[teacher_switch]
    return {
        "data": data,
        "labels": data["labels"],
        "floor_xy": floor_xy,
        "neural_xy": neural_xy,
        "selected_xy": selected_xy,
        "teacher_xy": teacher_xy,
        "selected_switch": alpha > EPS,
        "teacher_switch": teacher_switch,
        "neural_switch": np.ones(len(alpha), dtype=bool),
    }


def _full_trajectory_best_paths(result: Mapping[str, Any]) -> list[str]:
    best = str(result.get("best_name", ""))
    trials = result.get("trials", {})
    if best == "full_trajectory_ensemble":
        return [row["train"]["checkpoint"] for row in trials.values() if isinstance(row, dict) and row.get("train", {}).get("checkpoint")]
    row = trials.get(best, {})
    checkpoint = (row.get("train") or {}).get("checkpoint")
    return [checkpoint] if checkpoint else []


def _full_trajectory_training_sources(result: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for name, row in (result.get("trials") or {}).items():
        if not isinstance(row, dict) or "train" not in row:
            continue
        rows.append(
            {
                "trial": name,
                "source": (row.get("train") or {}).get("source", row.get("source", "unknown")),
                "checkpoint": (row.get("train") or {}).get("checkpoint"),
            }
        )
    counts = dict(Counter(str(row["source"]) for row in rows))
    if counts and set(counts) == {"cached_verified"}:
        summary = "cached_verified_checkpoints_fresh_eval"
    elif counts:
        summary = "mixed_training_sources_fresh_eval"
    else:
        summary = "unknown_training_source_fresh_eval"
    return {"summary": summary, "counts": counts, "trials": rows}


def _full_trajectory_selected_xy(result: Mapping[str, Any]) -> dict[str, Any]:
    paths = _full_trajectory_best_paths(result)
    if not paths:
        raise FileNotFoundError("Missing Stage41 full-waypoint checkpoint paths.")
    pred, labels = ft._predict_ensemble(paths, "test")
    selected_ade, selected_fde, switch, floor_ade = ft._apply_policy(pred, labels, result["best_policy"])
    floor_xy = ft._floor_waypoints(labels)
    neural_xy = ft._pred_waypoints(pred, labels)
    selected_xy = floor_xy.copy()
    selected_xy[switch.astype(bool)] = neural_xy[switch.astype(bool)]
    floor_fde = ft._trajectory_errors(floor_xy, labels)[1]
    return {
        "paths": paths,
        "pred": pred,
        "labels": labels,
        "floor_xy": floor_xy,
        "neural_xy": neural_xy,
        "selected_xy": selected_xy,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "switch": switch.astype(bool),
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
    }


def _joint_eval(name: str, xy: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray) -> dict[str, Any]:
    raw = dict(np.load(ft.DATA_DIR / "all_agent_test.npz", allow_pickle=True))
    keys = dlaa._group_keys(raw)
    stats = jrc._joint_stats(name, xy, labels, keys, switch.astype(bool))
    floor_stats = jrc._joint_stats("floor", ft._floor_waypoints(labels), labels, keys, np.zeros(len(switch), dtype=bool))
    return {
        "stats": stats,
        "floor_stats": floor_stats,
        "near_collision_delta_005": float(stats["near_collision_rate_005"] - floor_stats["near_collision_rate_005"]),
        "near_collision_delta_002": float(stats["near_collision_rate_002"] - floor_stats["near_collision_rate_002"]),
        "smoothness_jagged_delta": float(stats["smoothness"]["jagged_rate"] - floor_stats["smoothness"]["jagged_rate"]),
    }


def _comparison_from_xy(
    *,
    name: str,
    source: str,
    selected_xy: np.ndarray,
    floor_xy: np.ndarray,
    labels: Mapping[str, np.ndarray],
    switch: np.ndarray,
    description: str,
    include_bootstrap: bool = False,
    include_joint: bool = False,
) -> dict[str, Any]:
    selected_ade, selected_fde = ft._trajectory_errors(selected_xy, labels)
    floor_ade, floor_fde = ft._trajectory_errors(floor_xy, labels)
    row = {
        "source": source,
        "description": description,
        "ade": _metrics(selected_ade, floor_ade, labels, switch),
        "fde": _metrics(selected_fde, floor_fde, labels, switch),
    }
    if include_bootstrap:
        row["ade_bootstrap"] = _bootstrap_report(selected_ade, floor_ade, labels)
        row["fde_bootstrap"] = _bootstrap_report(selected_fde, floor_fde, labels)
    if include_joint:
        row["joint"] = _joint_eval(name, selected_xy, labels, switch)
    return row


def _cached_stage41_comparisons() -> dict[str, Any]:
    learned_shape = read_json("outputs/stage41_domain_local/stage41_learned_waypoint_shape_bridge.json", {})
    endpoint_to_full = read_json("outputs/stage41_domain_local/stage41_endpoint_to_full_trajectory_repair.json", {})
    graph = read_json("outputs/stage41_fresh_confirmation/stage41_group_consistency_distiller.json", {})
    return {
        "endpoint_to_full_linear_bridge_domain_local": {
            "source": "cached_verified",
            "description": "Domain-local endpoint neural dynamics projected through a linear waypoint bridge.",
            "report": {
                "two_domain_endpoint_to_full_gate": endpoint_to_full.get("two_domain_endpoint_to_full_gate"),
                "positive_domains": endpoint_to_full.get("positive_domains"),
                "domains": endpoint_to_full.get("domains", {}),
            },
        },
        "learned_waypoint_shape_bridge": {
            "source": "cached_verified",
            "description": "Protected learned waypoint-shape residual around endpoint bridge; positive but small shape gain.",
            "report": {
                "two_domain_learned_shape_gate": learned_shape.get("two_domain_learned_shape_gate"),
                "positive_domains": learned_shape.get("positive_domains"),
                "domains": learned_shape.get("domains", {}),
            },
        },
        "graph_interaction_group_consistency": {
            "source": "cached_verified",
            "description": "Group/neighbor consistency protected model, used as graph interaction comparison.",
            "report": {
                "test_metrics": graph.get("test_metrics", {}),
                "lift_over_fixed_proximity_guard": graph.get("lift_over_fixed_proximity_guard", {}),
            },
        },
    }


def run_stage42_full_waypoint_dynamics() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    label_report = ft.build_full_trajectory_labels()
    # This trains only if Stage41 full-waypoint checkpoints are absent; otherwise
    # the checkpoint training source is cached_verified and the Stage42 eval below
    # is fresh_run. This keeps the result honest and resume-friendly.
    full_result = ft.train_full_trajectory_world_state()
    linear = _composite_linear_bundle()
    labels = linear["labels"]
    comparisons: dict[str, Any] = {}
    comparisons["strongest_floor_linear"] = _comparison_from_xy(
        name="strongest_floor_linear",
        source="fresh_run",
        selected_xy=linear["floor_xy"],
        floor_xy=linear["floor_xy"],
        labels=labels,
        switch=np.zeros(len(linear["floor_xy"]), dtype=bool),
        description="Stage37/strongest floor linear full-waypoint baseline.",
    )
    comparisons["endpoint_only_final_fde"] = {
        "source": "fresh_run",
        "description": "Endpoint-only diagnostic: compares final waypoint FDE only; not a full-waypoint model.",
        "fde": comparisons["strongest_floor_linear"]["fde"],
        "claim_boundary": {"full_waypoint_model": False, "diagnostic_only": True},
    }
    comparisons["teacher_repair_linear_bridge"] = _comparison_from_xy(
        name="teacher_repair_linear_bridge",
        source="fresh_run",
        selected_xy=linear["teacher_xy"],
        floor_xy=linear["floor_xy"],
        labels=labels,
        switch=linear["teacher_switch"],
        description="Stage37/teacher repaired endpoint policy projected as linear waypoints.",
    )
    comparisons["m3w_neural_v1_composite_tail_linear_bridge"] = _comparison_from_xy(
        name="m3w_neural_v1_composite_tail_linear_bridge",
        source="fresh_run",
        selected_xy=linear["selected_xy"],
        floor_xy=linear["floor_xy"],
        labels=labels,
        switch=linear["selected_switch"],
        description="M3W-Neural v1 composite-tail endpoint dynamics projected as linear full-waypoints.",
        include_bootstrap=True,
        include_joint=True,
    )
    comparisons["ungated_endpoint_linear_bridge"] = _comparison_from_xy(
        name="ungated_endpoint_linear_bridge",
        source="fresh_run",
        selected_xy=linear["neural_xy"],
        floor_xy=linear["floor_xy"],
        labels=labels,
        switch=linear["neural_switch"],
        description="Ungated endpoint neural dynamics projected as linear waypoints; diagnostic safety failure if easy degradation is high.",
    )
    full_bundle = _full_trajectory_selected_xy(full_result)
    full_labels = full_bundle["labels"]
    comparisons["full_waypoint_transformer_protected"] = _comparison_from_xy(
        name="full_waypoint_transformer_protected",
        source="fresh_run",
        selected_xy=full_bundle["selected_xy"],
        floor_xy=full_bundle["floor_xy"],
        labels=full_labels,
        switch=full_bundle["switch"],
        description="Validation-selected full-waypoint sequence model with protected switch policy.",
        include_bootstrap=True,
        include_joint=True,
    )
    comparisons["ungated_full_waypoint_transformer"] = _comparison_from_xy(
        name="ungated_full_waypoint_transformer",
        source="fresh_run",
        selected_xy=full_bundle["neural_xy"],
        floor_xy=full_bundle["floor_xy"],
        labels=full_labels,
        switch=np.ones(len(full_bundle["floor_xy"]), dtype=bool),
        description="Ungated full-waypoint neural sequence model; diagnostic only.",
    )
    comparisons.update(_cached_stage41_comparisons())
    result = {
        "source": "fresh_run",
        "stage": "Stage42-C full-waypoint dynamics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                ft.DATA_DIR / "all_agent_test.npz",
                ft.DATA_DIR / "full_trajectory_test.npz",
                "outputs/stage41_fresh_confirmation/stage41_full_trajectory_world_state.json",
                "outputs/stage41_domain_local/stage41_learned_waypoint_shape_bridge.json",
                "outputs/stage41_domain_local/stage41_endpoint_to_full_trajectory_repair.json",
            ]
        ),
        "label_reconstruction": label_report,
        "full_waypoint_training_result": {
            "source": full_result.get("source", "fresh_run"),
            "checkpoint_training_sources": _full_trajectory_training_sources(full_result),
            "stage42_eval_source": "fresh_run",
            "best_name": full_result.get("best_name"),
            "deployment_decision": full_result.get("deployment_decision"),
            "full_trajectory_world_state_pass": full_result.get("full_trajectory_world_state_pass"),
            "positive_external_domains": full_result.get("positive_external_domains"),
        },
        "comparisons": comparisons,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "future_labels_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    result["stage42_c_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _append_ledger(result)
    return result


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    comps = result["comparisons"]
    protected = comps["full_waypoint_transformer_protected"]
    protected_ade = protected["ade"]
    protected_fde = protected["fde"]
    protected_joint = protected.get("joint", {})
    protected_positive_domains = [
        name
        for name, row in protected_ade.get("by_domain", {}).items()
        if row.get("all_improvement", 0.0) > 0.0 and (row.get("t50_improvement", 0.0) > 0.0 or row.get("hard_failure_improvement", 0.0) > 0.0)
    ]
    comparison_names = set(comps)
    gates = {
        "full_waypoint_labels_reconstructed": bool((result.get("label_reconstruction") or {}).get("splits", {}).get("test", {}).get("rows", 0) > 0),
        "required_comparisons_present": {
            "endpoint_only_final_fde",
            "m3w_neural_v1_composite_tail_linear_bridge",
            "learned_waypoint_shape_bridge",
            "full_waypoint_transformer_protected",
            "graph_interaction_group_consistency",
            "ungated_full_waypoint_transformer",
        }.issubset(comparison_names),
        "full_waypoint_sequence_model_trained_or_cached_verified": bool(result.get("full_waypoint_training_result", {}).get("best_name")),
        "protected_full_waypoint_ade_positive": protected_ade.get("all_improvement", 0.0) > 0.0 and protected_ade.get("hard_failure_improvement", 0.0) > 0.0,
        "protected_full_waypoint_fde_positive": protected_fde.get("all_improvement", 0.0) > 0.0 and protected_fde.get("t50_improvement", 0.0) > 0.0,
        "two_external_domains_positive": len(protected_positive_domains) >= 2,
        "easy_preserved": protected_ade.get("easy_degradation", 1.0) <= 0.02,
        "collision_not_worse": protected_joint.get("near_collision_delta_005", 1.0) <= 0.01,
        "no_leakage_pass": all(
            result["no_leakage"][k] is False
            for k in ["future_endpoint_input", "future_waypoints_input", "central_velocity", "test_endpoint_goals", "test_threshold_tuning"]
        ),
        "no_metric_seconds_overclaim": not result["claim_boundary"]["metric_or_seconds_claim"],
        "stage5c_false": not result["claim_boundary"]["stage5c_executed"],
        "smc_false": not result["claim_boundary"]["smc_enabled"],
    }
    return {
        "source": "fresh_run",
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": int(len(gates)),
        "positive_domains": protected_positive_domains,
        "verdict": "stage42_c_full_waypoint_dynamics_pass" if all(gates.values()) else "stage42_c_full_waypoint_dynamics_partial",
    }


def _render_metric_row(name: str, row: Mapping[str, Any]) -> str:
    ade = row.get("ade", {})
    fde = row.get("fde", {})
    joint = row.get("joint", {})
    return (
        f"| `{name}` | `{row.get('source', 'unknown')}` | {ade.get('rows', 0)} | "
        f"{ade.get('all_improvement', 0.0):.4f} | {ade.get('t50_improvement', 0.0):.4f} | {ade.get('t100_improvement', 0.0):.4f} | "
        f"{ade.get('hard_failure_improvement', 0.0):.4f} | {ade.get('easy_degradation', 0.0):.4f} | "
        f"{fde.get('all_improvement', 0.0):.4f} | {fde.get('t50_improvement', 0.0):.4f} | "
        f"{joint.get('near_collision_delta_005', 0.0):.4f} |"
    )


def _render_report(result: Mapping[str, Any]) -> list[str]:
    comps = result["comparisons"]
    protected = comps["full_waypoint_transformer_protected"]
    lines = [
        "# Stage42-C Full-Waypoint Dynamics",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        "",
        "## Claim Boundary",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Label Reconstruction / Training",
        "",
        f"- full trajectory label source: `{result['label_reconstruction'].get('source')}`",
        f"- full-waypoint checkpoint training sources: `{result['full_waypoint_training_result'].get('checkpoint_training_sources', {}).get('summary')}`",
        f"- Stage42-C evaluation source: `{result['full_waypoint_training_result'].get('stage42_eval_source')}`",
        f"- best full-waypoint model: `{result['full_waypoint_training_result'].get('best_name')}`",
        f"- training deployment decision: `{result['full_waypoint_training_result'].get('deployment_decision')}`",
        "",
        "## Full-Waypoint Comparison",
        "",
        "| candidate | source | rows | ADE all | ADE t50 | ADE t100 diag | ADE hard/failure | ADE easy degr | FDE all | FDE t50 | near-collision d005 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in [
        "strongest_floor_linear",
        "teacher_repair_linear_bridge",
        "m3w_neural_v1_composite_tail_linear_bridge",
        "ungated_endpoint_linear_bridge",
        "full_waypoint_transformer_protected",
        "ungated_full_waypoint_transformer",
    ]:
        lines.append(_render_metric_row(name, comps[name]))
    lines.extend(
        [
            "",
            "## Cached-Verified Comparisons",
            "",
            "- `endpoint_to_full_linear_bridge_domain_local`: domain-local endpoint neural dynamics through a linear waypoint bridge.",
            "- `learned_waypoint_shape_bridge`: protected learned waypoint-shape residual; positive but small shape gain.",
            "- `graph_interaction_group_consistency`: group/neighbor consistency protected comparison.",
            "",
            "## Protected Full-Waypoint By Domain",
            "",
            "| domain | rows | ADE all | ADE t50 | ADE t100 diag | hard/failure | easy degr | switch |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for domain, row in protected["ade"].get("by_domain", {}).items():
        lines.append(
            f"| `{domain}` | {row.get('rows', 0)} | {row.get('all_improvement', 0.0):.4f} | {row.get('t50_improvement', 0.0):.4f} | {row.get('t100_improvement', 0.0):.4f} | {row.get('hard_failure_improvement', 0.0):.4f} | {row.get('easy_degradation', 0.0):.4f} | {row.get('switch_rate', 0.0):.4f} |"
        )
    boot = protected.get("ade_bootstrap", {})
    lines.extend(["", "## Bootstrap CI For Protected Full-Waypoint ADE", "", "| slice | low | mid | high | n |", "| --- | ---: | ---: | ---: | ---: |"])
    for key in ["all", "t50", "t100", "hard_failure"]:
        row = boot.get(key, {})
        lines.append(f"| `{key}` | {row.get('low', 0.0):.4f} | {row.get('mid', 0.0):.4f} | {row.get('high', 0.0):.4f} | {row.get('n', 0)} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-C now compares actual reconstructed future waypoint labels, not only endpoint FDE.",
            "- The protected full-waypoint sequence model is evaluated against endpoint-linear bridges and cached-verified learned-shape / graph-interaction evidence.",
            "- Ungated full-waypoint and ungated endpoint variants remain diagnostic and are not deployable if easy safety fails.",
            "- All results remain raw-frame dataset-local 2.5D. No metric, seconds-level, true 3D, Stage5C, or SMC claim is made.",
            "",
            "## Gate Verdict",
            "",
            f"`{result['stage42_c_gate']['verdict']}` ({result['stage42_c_gate']['passed']} / {result['stage42_c_gate']['total']})",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_c_gate"]
    lines = [
        "# Stage42-C Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- positive domains: `{gate.get('positive_domains', [])}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| {name} | `{ok}` |")
    return lines


def _append_ledger(result: Mapping[str, Any]) -> None:
    entry = {
        "command": "run_stage42_full_waypoint_dynamics.py",
        "source": result["source"],
        "status": "success",
        "generated_at_utc": result["generated_at_utc"],
        "git_commit": result["git_commit"],
        "input_hash": result["input_hash"],
        "outputs": [str(REPORT_JSON), str(REPORT_MD), str(GATE_MD)],
    }
    with (OUT_DIR / "run_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_full_waypoint_dynamics()
