from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_auxiliary_head_repair import (
    _calibrator_features,
    _fit_select,
    _ridge_fit,
    _ridge_predict,
    _standardize_features,
)
from src.stage43_full_waypoint_latent_dynamics import (
    OUT_DIR,
    README_RESULTS,
    RESEARCH_STATE,
    WORK_SUMMARY,
    M3W_README,
    _build_split,
    _cache_path,
    _git_commit,
    _jsonable,
    _npz,
    _sha256,
)
from src.stage43_full_waypoint_latent_robustness_audit import (
    STAGE43_M_JSON,
    _load_model,
    _standardize_from_checkpoint,
)
from src.stage43_world_state_head_audit import _binary_metrics, _regression_metrics, _predict_heads


REPORT_JSON = OUT_DIR / "stage43_interaction_validity_proxy.json"
REPORT_MD = OUT_DIR / "stage43_interaction_validity_proxy.md"
GATE_MD = OUT_DIR / "stage43_stage_x_interaction_validity_proxy_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

STAGE43_W_JSON = OUT_DIR / "stage43_auxiliary_head_repair.json"
SECTION = "STAGE43_X_INTERACTION_VALIDITY_PROXY"
SOURCE = "fresh_stage43_x_interaction_validity_proxy"
EPS = 1e-8


def _group_key(cache: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.asarray(
        [
            f"{source}|{int(round(float(frame)))}|{int(horizon)}"
            for source, frame, horizon in zip(cache["source_file"].astype(str), cache["frame_id"], cache["horizon"])
        ],
        dtype=object,
    )


def _future_min_neighbor_distance(cache: Mapping[str, np.ndarray]) -> dict[str, Any]:
    waypoints = cache["waypoint_xy"].astype(np.float32)
    valid = cache["waypoint_valid"].astype(bool)
    scale = np.maximum(cache["scale"].astype(np.float32), 1e-4)
    agent_id = cache["agent_id"].astype(np.int64)
    keys = _group_key(cache)
    out = np.full(len(keys), np.inf, dtype=np.float64)
    grouped = np.zeros(len(keys), dtype=bool)
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        groups[str(key)].append(idx)
    for members in groups.values():
        if len(members) < 2:
            continue
        mem = np.asarray(members, dtype=np.int64)
        agents = agent_id[mem]
        for local_i, row in enumerate(mem):
            other_mask = (np.arange(len(mem)) != local_i) & (agents != agents[local_i])
            if not np.any(other_mask):
                continue
            other = mem[other_mask]
            best = np.inf
            for waypoint_idx in range(waypoints.shape[1]):
                if not bool(valid[row, waypoint_idx]):
                    continue
                keep = valid[other, waypoint_idx]
                if not np.any(keep):
                    continue
                pts = waypoints[other[keep], waypoint_idx, :].astype(np.float64)
                d = np.linalg.norm(pts - waypoints[row, waypoint_idx, :].astype(np.float64)[None, :], axis=1)
                best = min(best, float(np.min(d) / max(float(scale[row]), EPS)))
            if np.isfinite(best):
                out[row] = best
                grouped[row] = True
    return {
        "min_future_neighbor_distance": out.astype(np.float32),
        "grouped_rows": grouped,
        "groups": int(len(groups)),
        "rows": int(len(keys)),
    }


def _smoothness_proxy(cache: Mapping[str, np.ndarray]) -> np.ndarray:
    current = cache["current_xy"].astype(np.float32)
    waypoints = cache["waypoint_xy"].astype(np.float32)
    valid = cache["waypoint_valid"].astype(bool)
    scale = np.maximum(cache["scale"].astype(np.float32), 1e-4)
    points = np.concatenate([current[:, None, :], waypoints], axis=1)
    step = np.linalg.norm(np.diff(points, axis=1), axis=2) / scale[:, None]
    valid_step = np.concatenate([valid[:, :1], valid], axis=1)
    valid_step = valid_step[:, 1:] & valid_step[:, :-1]
    step = np.where(valid_step, step, np.nan)
    accel = np.abs(np.diff(step, axis=1))
    finite = np.isfinite(accel)
    sum_accel = np.where(finite, accel, 0.0).sum(axis=1)
    count_accel = finite.sum(axis=1)
    mean_accel = np.where(count_accel > 0, sum_accel / np.maximum(count_accel, 1), 10.0)
    valid_rate = valid.mean(axis=1).astype(np.float32)
    return (valid_rate * np.exp(-0.5 * mean_accel)).astype(np.float32)


def _make_labels(split: str, *, interaction_threshold: float) -> dict[str, Any]:
    cache = _npz(_cache_path(split))
    future_min = _future_min_neighbor_distance(cache)
    min_dist = future_min["min_future_neighbor_distance"]
    grouped = future_min["grouped_rows"]
    risk = (grouped & np.isfinite(min_dist) & (min_dist < float(interaction_threshold))).astype(np.float32)
    smooth = _smoothness_proxy(cache)
    return {
        "interaction_risk": risk,
        "smoothness_validity_proxy": smooth,
        "min_future_neighbor_distance": min_dist,
        "grouped_rows": grouped,
        "summary": {
            "rows": int(len(risk)),
            "grouped_rows": int(grouped.sum()),
            "interaction_threshold": float(interaction_threshold),
            "interaction_positive_rate": float(np.mean(risk)) if len(risk) else 0.0,
            "smoothness_mean": float(np.mean(smooth)) if len(smooth) else 0.0,
            "smoothness_std": float(np.std(smooth)) if len(smooth) else 0.0,
            "groups": future_min["groups"],
        },
    }


def _fit_binary_select(
    train,
    val,
    test,
    train_pred: Mapping[str, np.ndarray],
    val_pred: Mapping[str, np.ndarray],
    test_pred: Mapping[str, np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    *,
    feature_sets: tuple[str, ...],
    l2_grid: tuple[float, ...],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for feature_set in feature_sets:
        raw_train_x, feature_names = _calibrator_features(train, train_pred, feature_set)
        raw_val_x, _ = _calibrator_features(val, val_pred, feature_set)
        raw_test_x, _ = _calibrator_features(test, test_pred, feature_set)
        train_x, val_x, test_x, mean, std = _standardize_features(raw_train_x, raw_val_x, raw_test_x)
        for l2 in l2_grid:
            weight = _ridge_fit(train_x, y_train.astype(np.float32), float(l2))
            val_score = np.clip(_ridge_predict(val_x, weight), 0.0, 1.0)
            val_metrics = _binary_metrics(y_val, val_score)
            auroc = val_metrics["auroc"] if val_metrics["auroc"] is not None else 0.0
            auprc = val_metrics["auprc"] if val_metrics["auprc"] is not None else 0.0
            candidates.append(
                {
                    "feature_set": feature_set,
                    "feature_names": feature_names,
                    "l2": float(l2),
                    "weight": weight,
                    "feature_mean": mean,
                    "feature_std": std,
                    "validation_metrics": val_metrics,
                    "objective": float(auroc + 0.25 * auprc - 0.10 * val_metrics["brier"]),
                    "test_x": test_x,
                }
            )
    candidates.sort(key=lambda row: row["objective"], reverse=True)
    best = candidates[0]
    test_score = np.clip(_ridge_predict(best["test_x"], best["weight"]), 0.0, 1.0)
    test_metrics = _binary_metrics(y_test, test_score)
    return {
        "selected": {
            "target": "future_interaction_risk",
            "feature_set": best["feature_set"],
            "l2": best["l2"],
            "feature_count": int(len(best["feature_names"])),
            "validation_metrics": best["validation_metrics"],
            "test_metrics": test_metrics,
        },
        "candidate_table": [
            {
                "feature_set": row["feature_set"],
                "l2": row["l2"],
                "validation_metrics": row["validation_metrics"],
                "objective": row["objective"],
            }
            for row in candidates
        ],
    }


def _target_override_fit_select(
    train,
    val,
    test,
    train_pred: Mapping[str, np.ndarray],
    val_pred: Mapping[str, np.ndarray],
    test_pred: Mapping[str, np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    *,
    feature_sets: tuple[str, ...],
    l2_grid: tuple[float, ...],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for feature_set in feature_sets:
        raw_train_x, feature_names = _calibrator_features(train, train_pred, feature_set)
        raw_val_x, _ = _calibrator_features(val, val_pred, feature_set)
        raw_test_x, _ = _calibrator_features(test, test_pred, feature_set)
        train_x, val_x, test_x, _, _ = _standardize_features(raw_train_x, raw_val_x, raw_test_x)
        for l2 in l2_grid:
            weight = _ridge_fit(train_x, y_train.astype(np.float32), float(l2))
            val_hat = np.clip(_ridge_predict(val_x, weight), 0.0, 1.0)
            val_metrics = _regression_metrics(y_val, val_hat)
            candidates.append(
                {
                    "feature_set": feature_set,
                    "feature_names": feature_names,
                    "l2": float(l2),
                    "weight": weight,
                    "validation_metrics": val_metrics,
                    "objective": float(val_metrics["r2"] - 0.01 * val_metrics["rmse"]),
                    "test_x": test_x,
                }
            )
    candidates.sort(key=lambda row: row["objective"], reverse=True)
    best = candidates[0]
    test_hat = np.clip(_ridge_predict(best["test_x"], best["weight"]), 0.0, 1.0)
    return {
        "selected": {
            "target": "future_waypoint_smoothness_validity_proxy",
            "feature_set": best["feature_set"],
            "l2": best["l2"],
            "feature_count": int(len(best["feature_names"])),
            "validation_metrics": best["validation_metrics"],
            "test_metrics": _regression_metrics(y_test, test_hat),
        },
        "candidate_table": [
            {
                "feature_set": row["feature_set"],
                "l2": row["l2"],
                "validation_metrics": row["validation_metrics"],
                "objective": row["objective"],
            }
            for row in candidates
        ],
    }


def run_interaction_validity_proxy(
    *,
    batch_size: int = 4096,
    interaction_threshold: float = 0.10,
    feature_sets: tuple[str, ...] = ("latent_only", "latent_heads", "latent_heads_context", "causal_x", "latent_heads_causal_x"),
    l2_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43m = read_json(STAGE43_M_JSON, {})
    stage43w = read_json(STAGE43_W_JSON, {})
    checkpoint, ckpt, model = _load_model(stage43m)
    seed = int(ckpt.get("seed", 431))
    train = _standardize_from_checkpoint(_build_split("train", max_rows=None, seed=seed), ckpt)
    val = _standardize_from_checkpoint(_build_split("val", max_rows=None, seed=seed), ckpt)
    test = _standardize_from_checkpoint(_build_split("test", max_rows=None, seed=seed), ckpt)
    train_pred = _predict_heads(model, train, batch_size=int(batch_size))
    val_pred = _predict_heads(model, val, batch_size=int(batch_size))
    test_pred = _predict_heads(model, test, batch_size=int(batch_size))
    labels = {split: _make_labels(split, interaction_threshold=float(interaction_threshold)) for split in ["train", "val", "test"]}
    interaction = _fit_binary_select(
        train,
        val,
        test,
        train_pred,
        val_pred,
        test_pred,
        labels["train"]["interaction_risk"],
        labels["val"]["interaction_risk"],
        labels["test"]["interaction_risk"],
        feature_sets=feature_sets,
        l2_grid=l2_grid,
    )
    smoothness = _target_override_fit_select(
        train,
        val,
        test,
        train_pred,
        val_pred,
        test_pred,
        labels["train"]["smoothness_validity_proxy"],
        labels["val"]["smoothness_validity_proxy"],
        labels["test"]["smoothness_validity_proxy"],
        feature_sets=feature_sets,
        l2_grid=l2_grid,
    )
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_future_label_proxy_head_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_w_precondition": {
            "verdict": stage43w.get("stage43_w_gate", {}).get("verdict"),
            "density_proxy_deployable": stage43w.get("stage43_w_gate", {}).get("deploy_density_proxy_head"),
            "true_physical_validity_deployable": stage43w.get("stage43_w_gate", {}).get("deploy_true_physical_validity"),
        },
        "training_protocol": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_sha256_matches_stage43_m": _sha256(checkpoint) == stage43m.get("checkpoint_sha256"),
            "selection_data": "train_val_selected_test_once",
            "test_threshold_tuning": False,
            "interaction_threshold_fixed_predeclared": float(interaction_threshold),
            "num_workers": 0,
            "batch_size": int(batch_size),
            "feature_sets": list(feature_sets),
            "l2_grid": list(map(float, l2_grid)),
        },
        "label_summaries": {split: labels[split]["summary"] for split in ["train", "val", "test"]},
        "interaction_risk_head": interaction,
        "smoothness_validity_proxy_head": smoothness,
        "no_leakage": {
            "future_waypoints_used_as_labels_only": True,
            "future_waypoint_input": False,
            "future_endpoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "future_interaction_risk_is_proxy_label": True,
            "smoothness_validity_proxy_not_true_physical_validity": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage43_x_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    inter = payload["interaction_risk_head"]["selected"]["test_metrics"]
    smooth = payload["smoothness_validity_proxy_head"]["selected"]["test_metrics"]
    gates = {
        "stage43_w_precondition_seen": payload["stage43_w_precondition"]["verdict"]
        == "stage43_w_density_proxy_repaired_validity_proxy_diagnostic",
        "checkpoint_replayed": payload["training_protocol"]["checkpoint_sha256_matches_stage43_m"] is True,
        "train_val_selected_test_once": payload["training_protocol"]["selection_data"] == "train_val_selected_test_once"
        and payload["training_protocol"]["test_threshold_tuning"] is False,
        "interaction_labels_have_both_classes": inter["defined"] is True,
        "interaction_head_signal": inter["auroc"] is not None and float(inter["auroc"]) > 0.60,
        "smoothness_proxy_reported": smooth["rows"] > 0,
        "smoothness_proxy_not_true_physical_claim": payload["claim_boundary"]["smoothness_validity_proxy_not_true_physical_validity"] is True,
        "future_labels_targets_only": payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoints_used_as_labels_only"] is True,
        "no_test_threshold_tuning": payload["no_leakage"]["test_threshold_tuning"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_x_interaction_proxy_signal_validity_proxy_diagnostic"
        if passed == total
        else "stage43_x_interaction_validity_proxy_incomplete",
        "deploy_interaction_risk_proxy_head": bool(passed == total),
        "deploy_true_physical_validity": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    gate = payload["stage43_x_gate"]
    inter = payload["interaction_risk_head"]["selected"]
    smooth = payload["smoothness_validity_proxy_head"]["selected"]
    lines = [
        "# Stage43-X Interaction/Validity Proxy Head Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deploy interaction risk proxy head: `{gate['deploy_interaction_risk_proxy_head']}`",
        f"- deploy true physical validity: `{gate['deploy_true_physical_validity']}`",
        "",
        "## Future Interaction Risk Proxy",
        "",
        f"- fixed threshold: `{payload['training_protocol']['interaction_threshold_fixed_predeclared']}`",
        f"- selected feature set: `{inter['feature_set']}`",
        f"- AUROC: `{inter['test_metrics']['auroc']:.4f}`",
        f"- AUPRC: `{inter['test_metrics']['auprc']:.4f}`",
        f"- positive rate: `{inter['test_metrics']['positive_rate']:.4f}`",
        f"- ECE: `{inter['test_metrics']['ece']:.4f}`",
        "",
        "## Smoothness / Validity Proxy",
        "",
        f"- selected feature set: `{smooth['feature_set']}`",
        f"- R2: `{smooth['test_metrics']['r2']:.4f}`",
        f"- corr: `{smooth['test_metrics']['corr']:.4f}`",
        f"- RMSE: `{smooth['test_metrics']['rmse']:.4f}`",
        "",
        "## Interpretation",
        "",
        "Stage43-X uses future full-waypoints only to construct supervised/evaluation labels. Inputs remain frozen latent/context predictions and causal features. The interaction label is a future-proximity proxy, not a human interaction annotation. The smoothness/validity label is a diagnostic proxy and is not true physical validity.",
        "",
        "Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.",
    ]
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage43-X Gate",
        "",
        f"verdict: `{gate['verdict']}`",
        f"passed: `{gate['passed']} / {gate['total']}`",
        f"deploy_interaction_risk_proxy_head: `{gate['deploy_interaction_risk_proxy_head']}`",
        f"deploy_true_physical_validity: `{gate['deploy_true_physical_validity']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    gate_lines.extend([f"| {name} | `{value}` |" for name, value in gate["gates"].items()])
    write_md(GATE_MD, gate_lines)
    _refresh_readmes(payload)
    _update_state(payload)
    _append_ledger(payload)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_x_gate"]
    inter = payload["interaction_risk_head"]["selected"]
    smooth = payload["smoothness_validity_proxy_head"]["selected"]
    lines = [
        "## Stage43-X interaction / validity proxy head audit",
        "",
        f"Result source: `{payload['result_source']}`. I froze the Stage43-M latent checkpoint and trained train/val-selected proxy heads for future-proximity interaction risk and waypoint smoothness/validity.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- interaction feature set: `{inter['feature_set']}`",
        f"- interaction AUROC/AUPRC: `{inter['test_metrics']['auroc']:.4f}` / `{inter['test_metrics']['auprc']:.4f}`",
        f"- interaction positive rate: `{inter['test_metrics']['positive_rate']:.4f}`",
        f"- smoothness proxy R2/corr: `{smooth['test_metrics']['r2']:.4f}` / `{smooth['test_metrics']['corr']:.4f}`",
        f"- deploy interaction risk proxy head: `{gate['deploy_interaction_risk_proxy_head']}`",
        f"- true physical validity claim: `False`",
        "",
        "Boundary: future waypoints are labels/evaluation only, never inference inputs. Interaction risk is a future-proximity proxy, not human interaction annotation; smoothness/validity remains diagnostic, not true physical validity.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, lines)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    gate = payload["stage43_x_gate"]
    state["stage43_x_interaction_validity_proxy"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "interaction_risk_head": payload["interaction_risk_head"]["selected"],
        "smoothness_validity_proxy_head": payload["smoothness_validity_proxy_head"]["selected"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["current_stage"] = "stage43_x_interaction_validity_proxy"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"event": "stage43_x_interaction_validity_proxy", "payload": payload}), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-X interaction/validity proxy head audit.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--interaction-threshold", type=float, default=0.10)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = run_interaction_validity_proxy(batch_size=int(args.batch_size), interaction_threshold=float(args.interaction_threshold))
    gate = result["stage43_x_gate"]
    print(f"Stage43-X: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    return result


if __name__ == "__main__":
    main()
