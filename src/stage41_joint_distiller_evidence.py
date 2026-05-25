from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src import stage41_joint_policy_distillation as jpd


OUT_DIR = jpd.OUT_DIR
RESULT_JSON = OUT_DIR / "stage41_joint_policy_distillation.json"
REPORT_JSON = OUT_DIR / "stage41_joint_policy_distillation_evidence.json"
REPORT_MD = OUT_DIR / "stage41_joint_policy_distillation_evidence.md"
BOOTSTRAP_N = 2000
SEED = 4219


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


def _bootstrap_ci(
    selected: np.ndarray,
    fallback: np.ndarray,
    mask: np.ndarray,
    *,
    n: int = BOOTSTRAP_N,
    seed: int = SEED,
) -> dict[str, float]:
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


def _feature_slices(total_dim: int) -> dict[str, list[int]]:
    tail = 5 + 2 + 2 + 2 + 1 + 3 + 5
    static_dim = int(total_dim - tail)
    if static_dim <= 0:
        raise ValueError(f"Unexpected feature dimension {total_dim}; cannot infer static feature block.")
    start = 0
    groups: dict[str, list[int]] = {}

    def take(name: str, width: int) -> None:
        nonlocal start
        groups[name] = list(range(start, start + width))
        start += width

    take("static_causal_features", static_dim)
    take("full_trajectory_prediction_signals", 5)
    take("current_group_geometry", 2)
    take("floor_group_geometry", 2)
    take("neural_group_geometry", 2)
    take("neighbor_count", 1)
    take("domain_embedding", 3)
    take("horizon_embedding", 5)
    if start != total_dim:
        raise ValueError(f"Feature slice accounting mismatch: {start} != {total_dim}")
    groups["all_group_geometry"] = (
        groups["current_group_geometry"] + groups["floor_group_geometry"] + groups["neural_group_geometry"]
    )
    return groups


def _predict_from_matrix(checkpoint: str | Path, x_np: np.ndarray) -> dict[str, np.ndarray]:
    torch = jpd._torch()
    payload = torch.load(checkpoint, map_location="cpu")
    trial = payload["trial"]
    model = jpd._make_model(int(payload["dim"]), int(trial["width"]), float(trial["dropout"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    x = torch.tensor(x_np.astype(np.float32))
    outs = {"switch_prob": [], "gain_pred": [], "harm_prob": []}
    with torch.no_grad():
        for start in range(0, x.shape[0], 4096):
            out = model(x[start : start + 4096])
            outs["switch_prob"].append(torch.sigmoid(out["switch_logit"]).cpu().numpy())
            outs["gain_pred"].append(out["gain"].cpu().numpy())
            outs["harm_prob"].append(torch.sigmoid(out["harm_logit"]).cpu().numpy())
    return {k: np.concatenate(v).astype(np.float32) for k, v in outs.items()}


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


def run_joint_distiller_evidence() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    result = read_json(RESULT_JSON, {})
    if not result:
        raise FileNotFoundError(f"Missing {RESULT_JSON}; run Stage41 joint policy distillation first.")
    if result.get("no_leakage", {}).get("base_switch_input", True):
        raise RuntimeError("Refusing evidence pass for a distiller that used base_switch input.")

    checkpoint = result["best_checkpoint"]
    policy = result["best_policy"]
    scores, data = jpd._predict_checkpoint(checkpoint, "test")
    selected, _selected_fde, switch = jpd._apply_policy(scores, data, policy)
    metrics = jpd._metric(selected, data["floor_ade"], data, switch)
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
    ]:
        x = data["x"].copy()
        x[:, groups[name]] = 0.0
        ablated_scores = _predict_from_matrix(checkpoint, x)
        ablated_selected, _ablated_fde, ablated_switch = jpd._apply_policy(ablated_scores, data, policy)
        ablated_metrics = jpd._metric(ablated_selected, data["floor_ade"], data, ablated_switch)
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
                "switch_delta": float(ablated_metrics.get("switch_rate", 0.0) - metrics.get("switch_rate", 0.0)),
            },
        }

    stable = bool(
        bootstrap["all"]["low"] > 0
        and bootstrap["t50"]["low"] > 0
        and bootstrap["hard_failure"]["low"] > 0
        and metrics.get("easy_degradation", 1.0) <= 0.02
    )
    contribution_summary = {
        name: row["delta_vs_full"] for name, row in sorted(ablations.items())
    }
    evidence = {
        "source": "fresh_run",
        "best_name": result.get("best_name"),
        "checkpoint": checkpoint,
        "policy_mode": policy.get("mode"),
        "metrics": metrics,
        "bootstrap": bootstrap,
        "bootstrap_n": BOOTSTRAP_N,
        "feature_groups": {name: len(cols) for name, cols in groups.items()},
        "ablations": ablations,
        "contribution_summary": contribution_summary,
        "statistically_stable_on_test": stable,
        "no_leakage": {
            "base_switch_input": False,
            "future_waypoints_input": False,
            "future_labels_eval_only": True,
            "test_threshold_tuning": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "caveat": "Bootstrap and ablation evidence is computed on the frozen no-base-switch distiller. It is still dataset-local raw-frame 2.5D and UCY remains fallback-only.",
    }
    write_json(REPORT_JSON, _jsonable(evidence))
    write_md(
        REPORT_MD,
        [
            "# Stage41 Joint Distiller Evidence",
            "",
            "- source: `fresh_run`",
            f"- best: `{evidence['best_name']}`",
            f"- policy mode: `{evidence['policy_mode']}`",
            f"- statistically stable on test: `{stable}`",
            f"- all improvement: `{metrics.get('all_improvement')}`",
            f"- t50 improvement: `{metrics.get('t50_improvement')}`",
            f"- t100 raw-frame diagnostic improvement: `{metrics.get('t100_improvement')}`",
            f"- hard/failure improvement: `{metrics.get('hard_failure_improvement')}`",
            f"- easy degradation: `{metrics.get('easy_degradation')}`",
            f"- bootstrap: `{bootstrap}`",
            f"- ablation deltas: `{contribution_summary}`",
            f"- no leakage: `{evidence['no_leakage']}`",
            "",
            "The frozen no-base-switch distiller remains a candidate, not a final world-model completion: UCY is fallback-only, and the policy is still per-agent all-agent-context rather than a jointly consistent latent rollout.",
        ],
    )
    return evidence


def main_joint_distiller_evidence() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_joint_distiller_evidence()
        status = "ok"
    finally:
        jpd._append_ledger(
            "stage41_joint_distiller_evidence",
            status,
            started,
            [RESULT_JSON],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_joint_distiller_evidence()
