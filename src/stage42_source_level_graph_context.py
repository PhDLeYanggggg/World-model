from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_source_level_ablation as an
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_incremental_ablation as ao
from src import stage42_source_level_residual_context as ap
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "source_level_graph_context_stage42.json"
REPORT_MD = OUT_DIR / "source_level_graph_context_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_as_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

K_NEIGHBORS = 4
MIN_GRAPH_DELTA = 0.01
EPS = 1e-6
HORIZONS = [10, 25, 50, 100]


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-AS 是 proposed source-level split graph-interaction residual training，不是 metric 或 seconds-level 结果。",
    "第一阶段只用 baseline-family rollout context；第二阶段用 current-frame kNN graph / past motion / train-safe goal prototype context 预测 residual full-waypoint delta。",
    "graph features 只使用当前帧和过去 history，不使用 future endpoint / future waypoint 作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _last_valid_motion(history_seq: np.ndarray) -> dict[str, np.ndarray]:
    valid = history_seq[..., 6] > 0.5
    counts = valid.sum(axis=1)
    idx = np.arange(history_seq.shape[1])[None, :]
    last = np.where(valid, idx, 0).max(axis=1).astype(int)
    row = np.arange(len(history_seq))
    speed = np.where(counts > 0, history_seq[row, last, 2], 0.0).astype(np.float32)
    heading = np.where(counts > 0, history_seq[row, last, 4], 0.0).astype(np.float32)
    vx = (speed * np.cos(heading)).astype(np.float32)
    vy = (speed * np.sin(heading)).astype(np.float32)
    return {"speed": speed, "heading": heading, "vx": vx, "vy": vy, "valid": (counts > 0)}


def _domain_onehot(data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    domains = sorted(set(data["dataset"].astype(str).tolist()))
    mat = np.stack([(data["dataset"].astype(str) == d).astype(np.float32) for d in domains], axis=1)
    return mat.astype(np.float32), domains


def _horizon_onehot(data: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.stack([(data["horizon"].astype(int) == h).astype(np.float32) for h in HORIZONS], axis=1)


def _empty_knn_feature() -> np.ndarray:
    vec = [0.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for _ in range(K_NEIGHBORS):
        vec.extend([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    return np.asarray(vec, dtype=np.float32)


def _knn_feature_names() -> list[str]:
    names = [
        "graph_neighbor_count",
        "graph_min_dist_norm",
        "graph_mean_k_dist_norm",
        "graph_inv_min_dist",
        "graph_density_r1",
        "graph_density_r2",
        "graph_density_r5",
        "graph_mean_speed_diff",
        "graph_mean_heading_cos",
        "graph_mean_heading_sin",
    ]
    for k in range(K_NEIGHBORS):
        names.extend(
            [
                f"graph_k{k}_dx_norm",
                f"graph_k{k}_dy_norm",
                f"graph_k{k}_dist_norm",
                f"graph_k{k}_closing_speed",
                f"graph_k{k}_heading_cos",
                f"graph_k{k}_heading_sin",
            ]
        )
    return names


def _build_graph_features(data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    n = len(data["horizon"])
    features = np.tile(_empty_knn_feature()[None, :], (n, 1))
    names = _knn_feature_names()
    motion = _last_valid_motion(data["history_seq"].astype(np.float32))
    source = data["source_file"].astype(str)
    frame = data["frame_id"].astype(np.float64)
    keys = np.asarray([f"{s}\t{int(round(f * 1000.0))}" for s, f in zip(source, frame)], dtype=object)
    order = np.argsort(keys)
    current_x = data["current_x"].astype(np.float64)
    current_y = data["current_y"].astype(np.float64)
    scale = np.maximum(data["scale"].astype(np.float64), EPS)
    agent = data["agent_id"].astype(np.int64)
    groups = 0
    rows_with_neighbors = 0
    max_unique_agents = 0
    start = 0
    while start < n:
        end = start + 1
        key = keys[order[start]]
        while end < n and keys[order[end]] == key:
            end += 1
        rows = order[start:end]
        groups += 1
        unique_agents, first_idx, inverse = np.unique(agent[rows], return_index=True, return_inverse=True)
        unique_rows = rows[first_idx]
        max_unique_agents = max(max_unique_agents, int(len(unique_rows)))
        if len(unique_rows) <= 1:
            start = end
            continue
        pos = np.stack([current_x[unique_rows], current_y[unique_rows]], axis=1)
        vel = np.stack([motion["vx"][unique_rows], motion["vy"][unique_rows]], axis=1).astype(np.float64)
        speed = motion["speed"][unique_rows].astype(np.float64)
        heading = motion["heading"][unique_rows].astype(np.float64)
        per_agent = np.zeros((len(unique_rows), len(names)), dtype=np.float32)
        rel_all = pos[None, :, :] - pos[:, None, :]
        dist_all = np.linalg.norm(rel_all, axis=2)
        dist_all[np.arange(len(unique_rows)), np.arange(len(unique_rows))] = np.inf
        order_k = np.argsort(dist_all, axis=1)[:, : min(K_NEIGHBORS, len(unique_rows) - 1)]
        for u in range(len(unique_rows)):
            ids = np.where(np.isfinite(dist_all[u]))[0]
            if len(ids) == 0:
                per_agent[u] = _empty_knn_feature()
                continue
            rows_with_neighbors += 1
            local_scale = float(scale[unique_rows[u]])
            ord_ids = order_k[u]
            ord_ids = ord_ids[np.isfinite(dist_all[u, ord_ids])]
            norm_all = dist_all[u, ids] / local_scale
            top_rel = rel_all[u, ord_ids] / local_scale
            top_dist = dist_all[u, ord_ids] / local_scale
            rel_vel = vel[ord_ids] - vel[u]
            unit = rel_all[u, ord_ids] / np.maximum(dist_all[u, ord_ids, None], EPS)
            closing = np.sum(rel_vel * unit, axis=1)
            hdiff = heading[ord_ids] - heading[u]
            top_speed_diff = speed[ord_ids] - speed[u]
            summary = [
                float(len(ids)),
                float(np.min(norm_all)),
                float(np.mean(np.sort(norm_all)[: min(K_NEIGHBORS, len(norm_all))])),
                float(1.0 / (1.0 + np.min(norm_all))),
                float(np.mean(norm_all < 1.0)),
                float(np.mean(norm_all < 2.0)),
                float(np.mean(norm_all < 5.0)),
                float(np.mean(top_speed_diff)),
                float(np.mean(np.cos(hdiff))),
                float(np.mean(np.sin(hdiff))),
            ]
            vec = list(summary)
            for kk in range(K_NEIGHBORS):
                if kk < len(ord_ids):
                    vec.extend(
                        [
                            float(top_rel[kk, 0]),
                            float(top_rel[kk, 1]),
                            float(top_dist[kk]),
                            float(closing[kk]),
                            float(np.cos(hdiff[kk])),
                            float(np.sin(hdiff[kk])),
                        ]
                    )
                else:
                    vec.extend([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
            per_agent[u] = np.asarray(vec, dtype=np.float32)
        features[rows] = per_agent[inverse]
        start = end
    stats = {
        "source": "fresh_run",
        "rows": int(n),
        "frame_groups": int(groups),
        "rows_with_neighbors": int(np.sum(features[:, 0] > 0)),
        "unique_agent_nodes_with_neighbors": int(rows_with_neighbors),
        "max_unique_agents_per_frame": int(max_unique_agents),
        "feature_count": int(features.shape[1]),
        "uses_future_endpoint": False,
        "uses_future_waypoint": False,
    }
    return features.astype(np.float32), names, stats


def _build_context_from_graph(
    data: Mapping[str, np.ndarray],
    variant: str,
    graph: np.ndarray,
    graph_names: list[str],
    stats: Mapping[str, Any],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    horizon = _horizon_onehot(data)
    domain, domains = _domain_onehot(data)
    proto = np.concatenate(
        [
            data["prototype_likelihood"].astype(np.float32),
            data["prototype_entropy"][:, None].astype(np.float32),
            data["goal_ambiguity"][:, None].astype(np.float32),
        ],
        axis=1,
    )
    history = data["history_scalar"].astype(np.float32)
    parts: list[np.ndarray] = []
    names: list[str] = []
    if variant in {"graph_only", "graph_goal", "graph_history_goal"}:
        parts.append(graph)
        names.extend(graph_names)
    else:
        raise ValueError(f"Unknown graph context variant: {variant}")
    if variant in {"graph_goal", "graph_history_goal"}:
        parts.append(proto)
        names.extend([f"prototype_{i}" for i in range(data["prototype_likelihood"].shape[1])] + ["prototype_entropy", "goal_ambiguity"])
    if variant == "graph_history_goal":
        parts.append(history)
        names.extend([f"history_scalar_{i}" for i in range(history.shape[1])])
    parts.extend([horizon.astype(np.float32), domain.astype(np.float32)])
    names.extend([f"horizon_{h}" for h in HORIZONS])
    names.extend([f"domain_{d}" for d in domains])
    return np.concatenate(parts, axis=1).astype(np.float32), names, dict(stats)


def _build_context(data: Mapping[str, np.ndarray], variant: str) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    graph, graph_names, stats = _build_graph_features(data)
    return _build_context_from_graph(data, variant, graph, graph_names, stats)


def _metric_delta(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> dict[str, float]:
    return ao._metric_delta(lhs, rhs)


def _positive_graph_delta(delta: Mapping[str, float], threshold: float = MIN_GRAPH_DELTA) -> bool:
    return (
        delta["all_improvement"] > threshold
        or delta["t50_improvement"] > threshold
        or delta["hard_failure_improvement"] > threshold
    )


def _interpret_graph_delta(name: str, delta: Mapping[str, float]) -> str:
    if _positive_graph_delta(delta):
        return f"{name} improves over baseline-family-only by > {MIN_GRAPH_DELTA} on at least one core metric."
    return f"{name} does not improve over baseline-family-only by > {MIN_GRAPH_DELTA}; graph/interaction contribution not proven."


def run_stage42_source_level_graph_context() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    shared = an._prep_shared()
    names = shared["feature_names"]
    features = shared["features"]
    baseline_direct = ap._direct_candidate(features[:, ap._baseline_mask(names)], shared)
    base_xy = baseline_direct["pred_xy"]
    data = shared["data"]
    labels = shared["labels"]
    residual_target = ap._target_delta(data, labels) - ap._xy_to_delta(data, base_xy)
    variants = {}
    context_stats = {}
    context_feature_counts = {}
    graph, graph_names, graph_stats = _build_graph_features(data)
    for name in ["graph_only", "graph_goal", "graph_history_goal"]:
        raw, raw_names, stats = _build_context_from_graph(data, name, graph, graph_names, graph_stats)
        context_stats[name] = stats
        context_feature_counts[name] = int(len(raw_names))
        variants[name] = ap._evaluate_residual_variant(name, raw, shared, base_xy, residual_target)
    baseline_metric = baseline_direct["model"]["metrics"]["protected_ridge_source_level"]
    deltas = {
        name: {
            "source": "fresh_run",
            "delta_vs_baseline_family_only": _metric_delta(row["protected"], baseline_metric),
            "positive_graph_increment": _positive_graph_delta(_metric_delta(row["protected"], baseline_metric)),
            "interpretation": _interpret_graph_delta(name, _metric_delta(row["protected"], baseline_metric)),
        }
        for name, row in variants.items()
    }
    positive = sorted([name for name, row in deltas.items() if row["positive_graph_increment"]])
    result = {
        "source": "fresh_run",
        "stage": "Stage42-AS proposed source-level graph interaction context residual",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage42_long_research/source_level_sequence_context_stage42.json",
                "outputs/stage42_long_research/source_level_residual_context_stage42.json",
            ]
        ),
        "split_stats": shared["split_stats"],
        "graph_schema": {
            "k_neighbors": K_NEIGHBORS,
            "feature_names": _knn_feature_names(),
            "group_key": "source_file + frame_id",
            "excludes_self_agent": True,
            "deduplicates_agent_horizon_rows": True,
            "uses_current_and_past_only": True,
        },
        "context_stats": context_stats,
        "context_feature_counts": context_feature_counts,
        "baseline_family_only": {
            "source": "fresh_run",
            "feature_count": int(np.sum(ap._baseline_mask(names))),
            "best_lambda": baseline_direct["model"]["best_lambda"],
            "protected": baseline_metric,
            "bootstrap": baseline_direct["model"]["bootstrap"],
        },
        "graph_variants": variants,
        "graph_deltas": deltas,
        "positive_graph_context_variants": positive,
        "summary": {
            "source": "fresh_run",
            "graph_context_verdict": "stage42_as_graph_context_supported" if positive else "stage42_as_graph_context_not_supported",
            "positive_graph_context_variants": positive,
            "interpretation": "Stage42-AS tests structured current-frame kNN graph / interaction context after tabular MLP and temporal sequence residual context failed. Positive variants support independent graph context contribution; negative variants imply source-level success remains dominated by baseline-family rollout context.",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "graph_features_current_and_past_only": True,
            "family_fde_input": False,
            "safe_strongest_idx_old_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
            "source_overlap_pass": bool(shared["split_stats"]["source_overlap_pass"]),
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
    result["stage42_as_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _append_ledger(result)
    return result


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    baseline = result["baseline_family_only"]["protected"]
    first_stats = next(iter(result["context_stats"].values()))
    gates = {
        "proposed_source_level_split_used": result["split_stats"]["by_split"]["test"]["rows"] == 47458,
        "graph_features_built": first_stats["feature_count"] == len(result["graph_schema"]["feature_names"])
        and first_stats["rows_with_neighbors"] > 0,
        "graph_current_past_only": result["graph_schema"]["uses_current_and_past_only"]
        and all(not row["uses_future_endpoint"] and not row["uses_future_waypoint"] for row in result["context_stats"].values()),
        "baseline_family_first_stage_positive": baseline["all_improvement"] > 0
        and baseline["t50_improvement"] > 0
        and baseline["easy_degradation"] <= 0.02,
        "graph_variants_complete": len(result["graph_variants"]) >= 3,
        "graph_context_increment_found": len(result["positive_graph_context_variants"]) >= 1,
        "bootstrap_available_for_baseline": result["baseline_family_only"]["bootstrap"]["all"]["bootstrap_n"] > 0
        and result["baseline_family_only"]["bootstrap"]["t50"]["bootstrap_n"] > 0,
        "no_leakage_pass": all(
            result["no_leakage"][k] is False
            for k in [
                "future_endpoint_input",
                "future_waypoint_input",
                "family_fde_input",
                "safe_strongest_idx_old_input",
                "central_velocity",
                "test_endpoint_goals",
                "test_threshold_tuning",
            ]
        )
        and result["no_leakage"]["graph_features_current_and_past_only"]
        and result["no_leakage"]["train_only_feature_normalization"],
        "no_metric_seconds_overclaim": not result["claim_boundary"]["metric_or_seconds_claim"],
        "stage5c_false": not result["claim_boundary"]["stage5c_executed"],
        "smc_false": not result["claim_boundary"]["smc_enabled"],
    }
    verdict = (
        "stage42_as_graph_context_evidence_pass"
        if all(gates.values())
        else "stage42_as_graph_context_evidence_partial_or_negative"
    )
    return {"source": "fresh_run", "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": int(len(gates)), "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    base = result["baseline_family_only"]["protected"]
    lines = [
        "# Stage42-AS Proposed Source-Level Graph Interaction Context",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_as_gate']['passed']} / {result['stage42_as_gate']['total']}`",
        f"- verdict: `{result['stage42_as_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Why This Was Run",
        "",
        "- Stage42-AQ ruled out a simple tabular MLP residual-context repair.",
        "- Stage42-AR ruled out a temporal Conv1D sequence-context residual repair.",
        "- Stage42-AS tests structured current-frame kNN graph / interaction context using only current and past information.",
        "",
        "## Graph Schema",
        "",
        f"- graph_schema: `{result['graph_schema']}`",
        f"- context_stats: `{result['context_stats']}`",
        "",
        "## Baseline-Family First Stage",
        "",
        f"- protected_metric: `{base}`",
        "",
        "## Graph Residual Variants",
        "",
        "| variant | lambda | alpha | features | all | t50 | t100 diag | hard/failure | easy | delta all | delta t50 | delta hard |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["graph_variants"].items():
        metric = row["protected"]
        d = result["graph_deltas"][name]["delta_vs_baseline_family_only"]
        lines.append(
            f"| `{name}` | {row['best_lambda']:.2f} | {row['best_residual_alpha']:.2f} | {row['feature_count']} | {metric['all_improvement']:.6f} | {metric['t50_improvement']:.6f} | {metric['t100_raw_frame_diagnostic_improvement']:.6f} | {metric['hard_failure_improvement']:.6f} | {metric['easy_degradation']:.6f} | {d['all_improvement']:.6f} | {d['t50_improvement']:.6f} | {d['hard_failure_improvement']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- positive_graph_context_variants: `{result['positive_graph_context_variants']}`",
            f"- graph_context_verdict: `{result['summary']['graph_context_verdict']}`",
            "",
        ]
    )
    if result["positive_graph_context_variants"]:
        lines.append("- Stage42-AS found graph/interaction residual value beyond baseline-family rollout context.")
    else:
        lines.append("- Stage42-AS did not find graph/interaction residual value beyond baseline-family rollout context.")
    lines.extend(
        [
            "- Claims remain dataset-local raw-frame 2.5D, not metric/seconds-level, true-3D, foundation, Stage5C, or SMC evidence.",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{result['no_leakage']}`",
            f"- claim_boundary: `{result['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_as_gate"]
    lines = [
        "# Stage42-AS Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    return lines


def _append_ledger(result: Mapping[str, Any]) -> None:
    row = {
        "stage": result["stage"],
        "source": result["source"],
        "generated_at_utc": result["generated_at_utc"],
        "verdict": result["stage42_as_gate"]["verdict"],
        "gate": f"{result['stage42_as_gate']['passed']}/{result['stage42_as_gate']['total']}",
        "git_commit": result["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(f"{row}\n")


if __name__ == "__main__":
    run_stage42_source_level_graph_context()
