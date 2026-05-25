from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_bounded_neural_blend_dynamics as blend
from src import stage41_full_trajectory_world_state as ft
from src import stage41_joint_rollout_consistency as jrc
from src import stage41_pure_ucy_source_validation as pure_source
from src import stage41_source_level_validation_repair as slv


OUT_DIR = Path("outputs/stage41_external_split")
REPORT_JSON = OUT_DIR / "stage41_pure_ucy_retrain_protocol.json"
REPORT_MD = OUT_DIR / "stage41_pure_ucy_retrain_protocol.md"
EPS = 1e-6
RIDGE_LAMBDA = 1e-2


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


def _source_keys(data: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([slv._source_key(src) for src in data["labels"]["source_file"].astype(str)], dtype="U256")


def _pure_ucy_mask(data: Mapping[str, Any]) -> np.ndarray:
    keys = _source_keys(data)
    # OpenTraj keeps some UCY files under ETH_UCY metadata because ETH/UCY is
    # a benchmark family. For the pure-UCY protocol we trust the official UCY
    # source path and explicitly exclude TrajNet duplicate-like mirrors.
    return np.asarray([key.startswith("UCY/") for key in keys], dtype=bool)


def _subset(data: Mapping[str, Any], mask: np.ndarray) -> dict[str, Any]:
    return pure_source._subset_bundle(data, mask)


def _features(data: Mapping[str, Any]) -> np.ndarray:
    labels = data["labels"]
    horizon = labels["horizon"].astype(np.float32)
    horizon_onehot = np.stack([(horizon == h).astype(np.float32) for h in [10, 25, 50, 100]], axis=1)
    proposal = np.stack(
        [
            data["proposal_gain"].astype(np.float32),
            data["proposal_harm"].astype(np.float32),
            data["proposal_uncertainty"].astype(np.float32),
            data["proposal_teacher_prob"].astype(np.float32),
            data["teacher_raw_switch"].astype(np.float32),
            data["teacher_repaired_switch"].astype(np.float32),
        ],
        axis=1,
    )
    return np.concatenate([data["x_teacher"].astype(np.float32), proposal, horizon_onehot], axis=1).astype(np.float32)


def _fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    std = np.maximum(x.std(axis=0), 1e-3).astype(np.float32)
    return mean, std


def _standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) - mean) / std).astype(np.float32)


def _ridge_fit(x: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA) -> np.ndarray:
    xb = np.concatenate([x.astype(np.float64), np.ones((len(x), 1), dtype=np.float64)], axis=1)
    reg = lam * np.eye(xb.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ y.astype(np.float64))


def _ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x.astype(np.float64), np.ones((len(x), 1), dtype=np.float64)], axis=1)
    return (xb @ w).astype(np.float64)


def _train_ucy_heads(train: Mapping[str, Any]) -> dict[str, Any]:
    x_raw = _features(train)
    mean, std = _fit_standardizer(x_raw)
    x = _standardize(x_raw, mean, std)
    gain = (train["floor_ade"].astype(np.float64) - train["neural_ade"].astype(np.float64)).astype(np.float64)
    harm = (train["neural_ade"].astype(np.float64) - train["floor_ade"].astype(np.float64)).astype(np.float64)
    return {
        "mean": mean,
        "std": std,
        "gain_w": _ridge_fit(x, gain),
        "harm_w": _ridge_fit(x, harm),
        "train_gain_mean": float(np.mean(gain)),
        "train_positive_gain_rate": float(np.mean(gain > 0.0)),
    }


def _predict_heads(model: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    x = _standardize(_features(data), model["mean"], model["std"])
    return {
        "pred_gain": _ridge_predict(x, model["gain_w"]),
        "pred_harm": _ridge_predict(x, model["harm_w"]),
    }


def _metrics_for_alpha(data: Mapping[str, Any], alpha: np.ndarray, *, include_joint: bool = True) -> dict[str, Any]:
    labels = data["labels"]
    floor_xy = data["floor_xy"].astype(np.float64)
    neural_xy = data["neural_xy"].astype(np.float64)
    selected_xy = floor_xy + alpha[:, None, None] * (neural_xy - floor_xy)
    selected_ade, selected_fde = ft._trajectory_errors(selected_xy, labels)
    floor_ade = data["floor_ade"].astype(np.float64)
    floor_fde = data["floor_fde"].astype(np.float64)
    switch = alpha > EPS
    ds = {
        "horizon": labels["horizon"],
        "hard": labels["hard"],
        "failure": labels["failure"],
        "easy": labels["easy"],
        "domain": labels["domain"],
        "candidate_fde": labels["candidate_fde"],
    }
    metrics = s41._metrics(selected_ade, floor_ade, ds, switch)
    metrics["endpoint_fde_metrics"] = s41._metrics(selected_fde, floor_fde, ds, switch)
    if include_joint:
        floor_stats = jrc._joint_stats("floor", floor_xy, labels, data["keys"], np.zeros(len(alpha), dtype=bool))
        selected_stats = jrc._joint_stats("pure_ucy_selected", selected_xy, labels, data["keys"], switch)
        metrics["collision_delta_vs_floor_005"] = float(selected_stats["near_collision_rate_005"] - floor_stats["near_collision_rate_005"])
        metrics["smoothness_jagged_delta"] = float(selected_stats["smoothness"]["jagged_rate"] - floor_stats["smoothness"]["jagged_rate"])
    else:
        metrics["collision_delta_vs_floor_005"] = 0.0
        metrics["smoothness_jagged_delta"] = 0.0
    metrics["alpha_mean"] = float(np.mean(alpha))
    metrics["switch_rate"] = float(np.mean(switch))
    return metrics


def _alpha_for_policy(data: Mapping[str, Any], pred: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> np.ndarray:
    mode = str(policy.get("mode", "gain_harm"))
    base = np.ones(len(pred["pred_gain"]), dtype=bool)
    if mode == "teacher_repaired_gain_harm":
        base = data["teacher_repaired_switch"].astype(bool)
    elif mode == "teacher_raw_gain_harm":
        base = data["teacher_raw_switch"].astype(bool)
    elif mode == "proposal_teacher_gain_harm":
        base = data["proposal_teacher_prob"].astype(np.float64) >= float(policy.get("teacher_min", 0.5))
    elif mode != "gain_harm":
        raise ValueError(f"unknown policy mode: {mode}")
    switch = (
        base
        & (pred["pred_gain"] >= float(policy.get("gain_min", 0.0)))
        & (pred["pred_harm"] <= float(policy.get("harm_max", 0.0)))
        & (data["proposal_harm"].astype(np.float64) <= float(policy.get("proposal_harm_max", 1.0)))
        & (data["proposal_uncertainty"].astype(np.float64) <= float(policy.get("uncertainty_max", 1.0)))
    )
    alpha = np.zeros(len(switch), dtype=np.float64)
    alpha[switch] = float(policy.get("alpha", 1.0))
    return alpha


def _candidate_policies(val_pred: Mapping[str, np.ndarray], val: Mapping[str, Any]) -> list[dict[str, Any]]:
    gain_grid = [0.0]
    positive = val_pred["pred_gain"][np.isfinite(val_pred["pred_gain"])]
    if len(positive):
        gain_grid.extend(float(v) for v in np.quantile(positive, [0.40, 0.55, 0.70, 0.85]))
    harm_grid = [0.0]
    finite_harm = val_pred["pred_harm"][np.isfinite(val_pred["pred_harm"])]
    if len(finite_harm):
        harm_grid.extend(float(v) for v in np.quantile(finite_harm, [0.20, 0.40, 0.60]))
    proposal_harm_grid = [0.25, 0.40, 0.60, 0.80, 1.00]
    policies: list[dict[str, Any]] = []
    for mode in ["gain_harm", "proposal_teacher_gain_harm", "teacher_raw_gain_harm", "teacher_repaired_gain_harm"]:
        for alpha in [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]:
            for gain_min in gain_grid:
                for harm_max in harm_grid:
                    for proposal_harm_max in proposal_harm_grid:
                        policy = {
                            "type": "pure_ucy_ridge_gain_harm",
                            "mode": mode,
                            "alpha": alpha,
                            "gain_min": gain_min,
                            "harm_max": harm_max,
                            "proposal_harm_max": proposal_harm_max,
                            "uncertainty_max": proposal_harm_max,
                            "teacher_min": 0.50,
                        }
                        policies.append(policy)
    return policies


def _score(metrics: Mapping[str, Any]) -> float:
    return (
        float(metrics.get("all_improvement", 0.0))
        + 1.6 * float(metrics.get("t50_improvement", 0.0))
        + 1.0 * float(metrics.get("t100_improvement", 0.0))
        + 1.4 * float(metrics.get("hard_failure_improvement", 0.0))
        - 40.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 10.0 * max(0.0, float(metrics.get("collision_delta_vs_floor_005", 1.0)) - blend.TEST_COLLISION_CEILING)
    )


def _eligible(metrics: Mapping[str, Any]) -> bool:
    return bool(
        metrics.get("all_improvement", 0.0) > 0.0
        and metrics.get("t50_improvement", 0.0) > 0.0
        and metrics.get("hard_failure_improvement", 0.0) > 0.0
        and metrics.get("easy_degradation", 1.0) <= 0.02
        and metrics.get("collision_delta_vs_floor_005", 1.0) <= blend.TEST_COLLISION_CEILING
        and metrics.get("switch_rate", 0.0) > 0.0
    )


def _select_policy(model: Mapping[str, Any], val: Mapping[str, Any]) -> dict[str, Any]:
    pred = _predict_heads(model, val)
    rows: list[dict[str, Any]] = []
    for policy in _candidate_policies(pred, val):
        metrics = _metrics_for_alpha(val, _alpha_for_policy(val, pred, policy), include_joint=False)
        rows.append({"policy": _jsonable(policy), "metrics": metrics, "eligible": _eligible(metrics), "score": _score(metrics)})
    # Joint safety is expensive, so only the strongest cheap candidates are
    # re-ranked with collision/smoothness statistics before freezing the policy.
    top_rows = sorted(rows, key=lambda row: row["score"], reverse=True)[:80]
    conservative_rows = [
        row
        for row in rows
        if str(row["policy"].get("mode")) in {"teacher_repaired_gain_harm", "teacher_raw_gain_harm", "proposal_teacher_gain_harm"}
        and float(row["policy"].get("alpha", 1.0)) <= 0.4
    ]
    selected_keys = {json.dumps(row["policy"], sort_keys=True) for row in top_rows}
    for row in sorted(conservative_rows, key=lambda item: item["score"], reverse=True)[:160]:
        key = json.dumps(row["policy"], sort_keys=True)
        if key not in selected_keys:
            top_rows.append(row)
            selected_keys.add(key)
    joint_rows: list[dict[str, Any]] = []
    for row in top_rows:
        policy = dict(row["policy"])
        metrics = _metrics_for_alpha(val, _alpha_for_policy(val, pred, policy), include_joint=True)
        joint_rows.append(
            {
                "policy": policy,
                "metrics": metrics,
                "eligible": _eligible(metrics),
                "score": _score(metrics),
                "cheap_score": row["score"],
            }
        )
    pool = [row for row in joint_rows if row["eligible"]] or joint_rows
    selected = max(pool, key=lambda row: row["score"])
    return {
        "selected": selected,
        "candidate_count": int(len(rows)),
        "top_cheap_candidates": sorted(rows, key=lambda row: row["score"], reverse=True)[:20],
        "joint_reranked_candidates": joint_rows,
        "eligible_count": int(sum(row["eligible"] for row in rows)),
        "joint_eligible_count": int(sum(row["eligible"] for row in joint_rows)),
    }


def _source_inventory(*splits: tuple[str, Mapping[str, Any]]) -> dict[str, Any]:
    return pure_source._source_inventory(*splits)


def _evaluate_policy(data: Mapping[str, Any], model: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    pred = _predict_heads(model, data)
    alpha = _alpha_for_policy(data, pred, policy)
    metrics = _metrics_for_alpha(data, alpha)
    return {
        "metrics": metrics,
        "prediction_summary": {
            "pred_gain_mean": float(np.mean(pred["pred_gain"])),
            "pred_gain_positive_rate": float(np.mean(pred["pred_gain"] > 0.0)),
            "pred_harm_mean": float(np.mean(pred["pred_harm"])),
            "switch_rate": float(np.mean(alpha > EPS)),
            "alpha_mean": float(np.mean(alpha)),
        },
    }


def run_pure_ucy_retrain_protocol() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    checkpoint, teacher_policy, min_sep = blend._load_frozen_model()
    raw_train = blend._bundle("train", checkpoint, teacher_policy, min_sep)
    raw_val = blend._bundle("val", checkpoint, teacher_policy, min_sep)
    raw_test = blend._bundle("test", checkpoint, teacher_policy, min_sep)
    train = _subset(raw_train, _pure_ucy_mask(raw_train))
    val = _subset(raw_val, _pure_ucy_mask(raw_val))
    test = _subset(raw_test, _pure_ucy_mask(raw_test))
    if len(train["labels"]["horizon"]) == 0 or len(val["labels"]["horizon"]) == 0 or len(test["labels"]["horizon"]) == 0:
        raise RuntimeError("Pure UCY train/val/test rows are required for this protocol.")
    model = _train_ucy_heads(train)
    selection = _select_policy(model, val)
    policy = dict(selection["selected"]["policy"])
    val_eval = _evaluate_policy(val, model, policy)
    test_eval = _evaluate_policy(test, model, policy)
    frozen_report_policy = read_json(blend.REPORT_JSON, {}).get("safe_switch_test_policy") or {}
    frozen_ucy_test_eval = blend._evaluate_blend(test, frozen_report_policy) if frozen_report_policy else {}
    test_metrics = test_eval["metrics"]
    source_inventory = _source_inventory(("train", train), ("val", val), ("test", test))
    policy_gate = bool(_eligible(test_metrics))
    strict_neural_gate = False
    result = {
        "source": "fresh_run",
        "protocol": "pure_ucy_train_val_test_policy_head_calibration_over_frozen_neural_proposal",
        "checkpoint": checkpoint,
        "teacher_policy_source": "cached_verified_mixed_external_neural_floor",
        "train_rows": int(len(train["labels"]["horizon"])),
        "val_rows": int(len(val["labels"]["horizon"])),
        "test_rows": int(len(test["labels"]["horizon"])),
        "source_inventory": source_inventory,
        "ridge_model_summary": {
            "lambda": RIDGE_LAMBDA,
            "feature_dim": int(len(model["mean"])),
            "train_gain_mean": model["train_gain_mean"],
            "train_positive_gain_rate": model["train_positive_gain_rate"],
        },
        "validation_selection": selection,
        "selected_policy": policy,
        "val_eval": val_eval,
        "test_eval": test_eval,
        "frozen_mixed_policy_ucy_test_reference": frozen_ucy_test_eval,
        "pure_ucy_policy_train_val_test_gate": policy_gate,
        "strict_pure_ucy_only_neural_retrain_select_test_gate": strict_neural_gate,
        "remaining_blocker": (
            "The policy/head is trained and selected only on UCY train/val rows, but the underlying neural proposal, Stage37 floor, and teacher-repaired switch features are inherited from mixed external training. "
            "A full strict pure-UCY neural world-model retrain is still not complete."
        ),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_labels_eval_only": True,
            "gain_harm_labels_train_only": True,
            "validation_threshold_selection_only": True,
            "test_threshold_tuning": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_endpoints_for_goal_construction": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "caveat": "This is UCY-only policy/head calibration over frozen mixed-trained neural proposals. It is dataset-local raw-frame 2.5D, not metric, true 3D, foundation, Stage5C, or SMC.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    lines = [
        "# Stage41 Pure UCY Train/Val/Test Policy-Head Calibration",
        "",
        "- source: `fresh_run`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        f"- protocol: `{result['protocol']}`",
        f"- train/val/test rows: `{result['train_rows']}` / `{result['val_rows']}` / `{result['test_rows']}`",
        f"- selected policy: `{policy}`",
        f"- pure UCY policy train/val/test gate: `{policy_gate}`",
        f"- strict pure UCY-only neural retrain/select/test gate: `{strict_neural_gate}`",
        f"- remaining blocker: `{result['remaining_blocker']}`",
        "",
        "## Metrics",
        "",
        "| split | rows | all | t50 | t100 | hard/failure | easy degradation | switch | collision delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, rows, ev in [("val", result["val_rows"], val_eval), ("test", result["test_rows"], test_eval)]:
        m = ev["metrics"]
        lines.append(
            f"| `{split}` | {rows} | {float(m.get('all_improvement', 0.0)):.4f} | "
            f"{float(m.get('t50_improvement', 0.0)):.4f} | {float(m.get('t100_improvement', 0.0)):.4f} | "
            f"{float(m.get('hard_failure_improvement', 0.0)):.4f} | {float(m.get('easy_degradation', 0.0)):.4f} | "
            f"{float(m.get('switch_rate', 0.0)):.4f} | {float(m.get('collision_delta_vs_floor_005', 0.0)):.4f} |"
        )
    if frozen_ucy_test_eval:
        fm = frozen_ucy_test_eval["metrics"]
        lines.extend(
            [
                "",
                "## Frozen Mixed-Policy UCY Test Reference",
                "",
                f"- metrics: `{fm}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Source Inventory",
            "",
            f"`{source_inventory}`",
            "",
            "## No Leakage",
            "",
            f"`{result['no_leakage']}`",
            "",
            "Future endpoints are labels/evaluation only. The selected policy was chosen on UCY validation rows and evaluated once on UCY test rows.",
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_pure_ucy_retrain_protocol() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_pure_ucy_retrain_protocol()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_pure_ucy_retrain_protocol",
            status,
            started,
            [blend.REPORT_JSON, "outputs/stage41_external_split/stage41_pure_ucy_source_validation.json"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_pure_ucy_retrain_protocol()
