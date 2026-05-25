from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_group_consistency_distiller as gcd
from src import stage41_joint_distiller_evidence as jde
from src import stage41_joint_policy_distillation as jpd


OUT_DIR = gcd.OUT_DIR
RESULT_JSON = OUT_DIR / "stage41_group_consistency_distiller.json"
REPORT_JSON = OUT_DIR / "stage41_group_consistency_evidence.json"
REPORT_MD = OUT_DIR / "stage41_group_consistency_evidence.md"
BOOTSTRAP_N = 2000
SEED = 4159


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


def _feature_slices(total_dim: int) -> dict[str, list[int]]:
    if total_dim <= 10:
        raise ValueError(f"Unexpected group distiller feature dimension: {total_dim}")
    base_dim = total_dim - 10
    base = {name: cols[:] for name, cols in jde._feature_slices(base_dim).items()}
    offset = base_dim
    base["group_consistency_features"] = list(range(offset, offset + 10))
    base["group_consistency_geometry"] = list(range(offset, offset + 6))
    base["proposal_score_features"] = list(range(offset + 6, offset + 10))
    return base


def _selected_from_policy(scores: Mapping[str, np.ndarray], data: Mapping[str, np.ndarray], policy: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    switch = (
        data["proposal_switch"].astype(bool)
        & (scores["safe_prob"] >= float(policy.get("safe_min", 0.5)))
        & (scores["gain_pred"] >= float(policy.get("gain_min", 0.0)))
        & (scores["unsafe_prob"] <= float(policy.get("unsafe_max", 1.0)))
    )
    selected = data["floor_ade"].copy()
    selected[switch] = data["neural_ade"][switch]
    return selected, switch


def _bootstrap_ci(selected: np.ndarray, fallback: np.ndarray, mask: np.ndarray, *, n: int = BOOTSTRAP_N, seed: int = SEED) -> dict[str, float]:
    ids = np.where(mask)[0]
    if len(ids) < 20:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(len(ids)), "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        boot = rng.choice(ids, size=len(ids), replace=True)
        vals[i] = 1.0 - float(selected[boot].mean()) / max(float(fallback[boot].mean()), jpd.EPS)
    return {
        "low": float(np.percentile(vals, 2.5)),
        "mid": float(np.percentile(vals, 50.0)),
        "high": float(np.percentile(vals, 97.5)),
        "n": int(len(ids)),
        "bootstrap_n": int(n),
    }


def _bootstrap_report(selected: np.ndarray, fallback: np.ndarray, data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    horizon = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    domain = data["domain"].astype(str)
    out = {
        "all": _bootstrap_ci(selected, fallback, np.ones(len(selected), dtype=bool), seed=SEED),
        "t50": _bootstrap_ci(selected, fallback, horizon == 50, seed=SEED + 1),
        "t100_raw_frame_diagnostic": _bootstrap_ci(selected, fallback, horizon == 100, seed=SEED + 2),
        "hard_failure": _bootstrap_ci(selected, fallback, hard_failure, seed=SEED + 3),
    }
    out["by_domain"] = {
        name: _bootstrap_ci(selected, fallback, domain == name, seed=SEED + 10 + i)
        for i, name in enumerate(sorted(set(domain.tolist())))
    }
    out["by_domain_t50"] = {
        name: _bootstrap_ci(selected, fallback, (domain == name) & (horizon == 50), seed=SEED + 20 + i)
        for i, name in enumerate(sorted(set(domain.tolist())))
    }
    return out


def _predict_ablation(checkpoint: str | Path, data: Mapping[str, np.ndarray], cols: Sequence[int]) -> dict[str, np.ndarray]:
    ablated = dict(data)
    x = data["x"].copy()
    x[:, list(cols)] = 0.0
    ablated["x"] = x
    return gcd._predict(checkpoint, ablated)


def run_group_consistency_evidence() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    result = read_json(RESULT_JSON, {})
    if not result:
        raise FileNotFoundError(f"Missing {RESULT_JSON}; run Stage41 group consistency distiller first.")
    if not result.get("group_consistency_distiller_deployable"):
        raise RuntimeError("Refusing to treat a non-deployable group consistency distiller as evidence.")
    checkpoint, repaired_policy, _policy_source = gcd._load_policy_and_checkpoint()
    data = gcd._bundle("test", checkpoint, repaired_policy)
    model_checkpoint = result["checkpoint"]
    policy = result["selected_policy"]
    scores = gcd._predict(model_checkpoint, data)
    selected, switch = _selected_from_policy(scores, data, policy)
    metrics, _metric_switch = gcd._policy_metrics(scores, data, policy)
    bootstrap = _bootstrap_report(selected, data["floor_ade"], data)
    groups = _feature_slices(int(data["x"].shape[1]))
    ablations: dict[str, Any] = {}
    for name in [
        "static_causal_features",
        "full_trajectory_prediction_signals",
        "all_group_geometry",
        "neighbor_count",
        "domain_embedding",
        "horizon_embedding",
        "group_consistency_features",
        "group_consistency_geometry",
        "proposal_score_features",
    ]:
        ablated_scores = _predict_ablation(model_checkpoint, data, groups[name])
        ablated_metrics, ablated_switch = gcd._policy_metrics(ablated_scores, data, policy)
        ablations[name] = {
            "metrics": ablated_metrics,
            "delta_vs_full": {
                "all_delta": float(ablated_metrics.get("all_improvement", 0.0) - metrics.get("all_improvement", 0.0)),
                "t50_delta": float(ablated_metrics.get("t50_improvement", 0.0) - metrics.get("t50_improvement", 0.0)),
                "t100_delta": float(ablated_metrics.get("t100_improvement", 0.0) - metrics.get("t100_improvement", 0.0)),
                "hard_delta": float(
                    ablated_metrics.get("hard_failure_improvement", 0.0)
                    - metrics.get("hard_failure_improvement", 0.0)
                ),
                "easy_delta": float(ablated_metrics.get("easy_degradation", 0.0) - metrics.get("easy_degradation", 0.0)),
                "switch_delta": float(ablated_metrics.get("switch_rate", 0.0) - metrics.get("switch_rate", 0.0)),
                "collision_delta_change": float(
                    ablated_metrics.get("collision_delta_vs_floor_005", 0.0)
                    - metrics.get("collision_delta_vs_floor_005", 0.0)
                ),
            },
            "switch_rate": float(np.mean(ablated_switch)),
        }
    stable = bool(
        bootstrap["all"]["low"] > 0
        and bootstrap["t50"]["low"] > 0
        and bootstrap["t100_raw_frame_diagnostic"]["low"] > 0
        and bootstrap["hard_failure"]["low"] > 0
        and metrics.get("easy_degradation", 1.0) <= 0.02
        and metrics.get("collision_delta_vs_floor_005", 1.0) <= 0.01
    )
    contribution_summary = {name: row["delta_vs_full"] for name, row in sorted(ablations.items())}
    evidence = {
        "source": "fresh_run",
        "checkpoint": model_checkpoint,
        "selected_policy": policy,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "bootstrap_n": BOOTSTRAP_N,
        "feature_groups": {name: len(cols) for name, cols in groups.items()},
        "ablations": ablations,
        "contribution_summary": contribution_summary,
        "statistically_stable_on_test": stable,
        "switch_rate": float(np.mean(switch)),
        "no_leakage": {
            "future_waypoints_input": False,
            "future_labels_eval_only": True,
            "train_gain_safe_unsafe_labels_only": True,
            "test_threshold_tuning": False,
            "policy_selected_on_val": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "caveat": "Evidence is for a frozen group-consistency distiller over proposed switches. It is still dataset-local raw-frame 2.5D, not true 3D, foundation-scale, Stage5C, or SMC.",
    }
    write_json(REPORT_JSON, _jsonable(evidence))
    write_md(
        REPORT_MD,
        [
            "# Stage41 Group Consistency Distiller Evidence",
            "",
            "- source: `fresh_run`",
            f"- statistically stable on test: `{stable}`",
            f"- metrics: `{metrics}`",
            f"- bootstrap: `{bootstrap}`",
            f"- ablation deltas: `{contribution_summary}`",
            f"- no leakage: `{evidence['no_leakage']}`",
            "",
            "This evidence tests the frozen neural group-consistency head. It keeps Stage5C and SMC disabled and does not make metric/seconds/foundation claims.",
        ],
    )
    return evidence


def main_group_consistency_evidence() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_group_consistency_evidence()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_group_consistency_evidence",
            status,
            started,
            [RESULT_JSON],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_group_consistency_evidence()
