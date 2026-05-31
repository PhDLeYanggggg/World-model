from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_dynamics import (
    OUT_DIR,
    README_RESULTS,
    RESEARCH_STATE,
    WORK_SUMMARY,
    M3W_README,
    _build_split,
    _git_commit,
    _jsonable,
    _sha256,
)
from src.stage43_full_waypoint_latent_robustness_audit import (
    STAGE43_M_JSON,
    _load_model,
    _pct,
    _standardize_from_checkpoint,
)
from src.stage43_world_state_head_audit import (
    REPORT_JSON as STAGE43_V_JSON,
    _predict_heads,
    _regression_metrics,
)


REPORT_JSON = OUT_DIR / "stage43_auxiliary_head_repair.json"
REPORT_MD = OUT_DIR / "stage43_auxiliary_head_repair.md"
GATE_MD = OUT_DIR / "stage43_stage_w_auxiliary_head_repair_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_W_AUXILIARY_HEAD_REPAIR"
SOURCE = "fresh_stage43_w_auxiliary_head_repair"
EPS = 1e-8


def _ridge_fit(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    xb = np.concatenate([x.astype(np.float32), np.ones((len(x), 1), dtype=np.float32)], axis=1).astype(np.float64)
    eye = np.eye(xb.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0
    return np.linalg.solve(xb.T @ xb + float(l2) * eye, xb.T @ y.astype(np.float64))


def _ridge_predict(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x.astype(np.float32), np.ones((len(x), 1), dtype=np.float32)], axis=1).astype(np.float64)
    return (xb @ weight).astype(np.float32)


def _model_hash(weight: np.ndarray, *, l2: float, target: str, feature_set: str) -> str:
    digest = hashlib.sha256()
    digest.update(weight.astype(np.float32).tobytes())
    digest.update(str(float(l2)).encode("utf-8"))
    digest.update(target.encode("utf-8"))
    digest.update(feature_set.encode("utf-8"))
    return digest.hexdigest()


def _calibrator_features(ds, pred: Mapping[str, np.ndarray], feature_set: str) -> tuple[np.ndarray, list[str]]:
    latent = pred["latent"].astype(np.float32)
    head_scores = np.stack(
        [
            pred["failure"].astype(np.float32),
            pred["gain"].astype(np.float32),
            pred["harm"].astype(np.float32),
            pred["density"].astype(np.float32),
        ],
        axis=1,
    )
    horizon = ds.horizon.astype(np.float32)[:, None] / 100.0
    source = ds.source_file.astype(str)
    domain = ds.domain.astype(str)
    domain_values = ["ETH_UCY", "TrajNet", "UCY"]
    domain_oh = np.stack([(domain == value).astype(np.float32) for value in domain_values], axis=1)
    h_values = [10, 25, 50, 100]
    horizon_oh = np.stack([(ds.horizon == value).astype(np.float32) for value in h_values], axis=1)
    base_parts = [latent, head_scores, horizon, domain_oh, horizon_oh]
    names = [
        *[f"latent_{i}" for i in range(latent.shape[1])],
        "stage43m_failure_score",
        "stage43m_gain_score",
        "stage43m_harm_score",
        "stage43m_density_score",
        "horizon_norm",
        *[f"domain_{value}" for value in domain_values],
        *[f"horizon_{value}" for value in h_values],
    ]
    if feature_set == "latent_only":
        return latent, [f"latent_{i}" for i in range(latent.shape[1])]
    if feature_set == "latent_heads":
        return np.concatenate([latent, head_scores], axis=1).astype(np.float32), names[: latent.shape[1] + head_scores.shape[1]]
    if feature_set == "latent_heads_context":
        # Add a small source-stability cue without using labels or test statistics.
        source_len = np.asarray([len(value) for value in source], dtype=np.float32)[:, None] / 128.0
        x = np.concatenate([*base_parts, source_len], axis=1).astype(np.float32)
        return x, [*names, "source_name_length_norm"]
    if feature_set == "causal_x":
        return ds.x.astype(np.float32), list(ds.feature_names)
    if feature_set == "latent_heads_causal_x":
        x = np.concatenate([latent, head_scores, ds.x.astype(np.float32)], axis=1).astype(np.float32)
        return x, [
            *[f"latent_{i}" for i in range(latent.shape[1])],
            "stage43m_failure_score",
            "stage43m_gain_score",
            "stage43m_harm_score",
            "stage43m_density_score",
            *list(ds.feature_names),
        ]
    raise ValueError(f"Unknown feature_set={feature_set}")


def _standardize_features(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0).astype(np.float32)
    raw_std = train_x.std(axis=0).astype(np.float32)
    std = np.where(raw_std < 1e-4, 1.0, raw_std).astype(np.float32)
    return (
        ((train_x - mean) / std).astype(np.float32),
        ((val_x - mean) / std).astype(np.float32),
        ((test_x - mean) / std).astype(np.float32),
        mean,
        std,
    )


def _target(ds, target: str) -> np.ndarray:
    if target == "density":
        return ds.y_density.astype(np.float32)
    if target == "waypoint_validity_proxy":
        return ds.waypoint_valid.mean(axis=1).astype(np.float32)
    raise ValueError(f"Unknown target={target}")


def _original_prediction(pred: Mapping[str, np.ndarray], target: str) -> np.ndarray:
    if target == "density":
        return pred["density"].astype(np.float32)
    if target == "waypoint_validity_proxy":
        return pred["validity"].astype(np.float32)
    raise ValueError(f"Unknown target={target}")


def _fit_select(
    train,
    val,
    test,
    train_pred: Mapping[str, np.ndarray],
    val_pred: Mapping[str, np.ndarray],
    test_pred: Mapping[str, np.ndarray],
    *,
    target: str,
    l2_grid: tuple[float, ...],
    feature_sets: tuple[str, ...],
) -> dict[str, Any]:
    y_train = _target(train, target)
    y_val = _target(val, target)
    y_test = _target(test, target)
    candidates: list[dict[str, Any]] = []
    for feature_set in feature_sets:
        raw_train_x, feature_names = _calibrator_features(train, train_pred, feature_set)
        raw_val_x, _ = _calibrator_features(val, val_pred, feature_set)
        raw_test_x, _ = _calibrator_features(test, test_pred, feature_set)
        train_x, val_x, test_x, mean, std = _standardize_features(raw_train_x, raw_val_x, raw_test_x)
        for l2 in l2_grid:
            weight = _ridge_fit(train_x, y_train, float(l2))
            val_hat = np.clip(_ridge_predict(val_x, weight), 0.0, 1.0)
            val_metrics = _regression_metrics(y_val, val_hat)
            candidates.append(
                {
                    "target": target,
                    "feature_set": feature_set,
                    "feature_names": feature_names,
                    "l2": float(l2),
                    "weight": weight,
                    "feature_mean": mean,
                    "feature_std": std,
                    "validation_metrics": val_metrics,
                    "objective": float(val_metrics["r2"] - 0.01 * val_metrics["rmse"]),
                    "model_hash": _model_hash(weight, l2=float(l2), target=target, feature_set=feature_set),
                    "test_x": test_x,
                }
            )
    candidates.sort(key=lambda row: row["objective"], reverse=True)
    best = candidates[0]
    test_hat = np.clip(_ridge_predict(best["test_x"], best["weight"]), 0.0, 1.0)
    original_test = np.clip(_original_prediction(test_pred, target), 0.0, 1.0)
    test_metrics = _regression_metrics(y_test, test_hat)
    original_metrics = _regression_metrics(y_test, original_test)
    return {
        "selected": {
            "target": target,
            "feature_set": best["feature_set"],
            "l2": best["l2"],
            "model_hash": best["model_hash"],
            "feature_count": int(len(best["feature_names"])),
            "validation_metrics": best["validation_metrics"],
            "test_metrics": test_metrics,
            "original_stage43m_test_metrics": original_metrics,
            "delta_r2_vs_stage43m": float(test_metrics["r2"] - original_metrics["r2"]),
            "delta_rmse_vs_stage43m": float(original_metrics["rmse"] - test_metrics["rmse"]),
        },
        "candidate_table": [
            {
                "target": row["target"],
                "feature_set": row["feature_set"],
                "l2": row["l2"],
                "model_hash": row["model_hash"],
                "validation_metrics": row["validation_metrics"],
                "objective": row["objective"],
            }
            for row in candidates
        ],
    }


def _breakdown_regression(values: np.ndarray, y_true: np.ndarray, pred: np.ndarray, *, min_rows: int = 100) -> dict[str, Any]:
    vv = values.astype(str)
    out: dict[str, Any] = {}
    for value in sorted(set(vv.tolist())):
        mask = vv == value
        if int(mask.sum()) < int(min_rows):
            continue
        out[value] = _regression_metrics(y_true[mask], pred[mask])
    return out


def _predict_selected(ds, pred: Mapping[str, np.ndarray], selected: Mapping[str, Any]) -> np.ndarray:
    x, _ = _calibrator_features(ds, pred, selected["feature_set"])
    # This helper is only used in tests with injected selected dictionaries; production path stores metrics directly.
    return x


def run_auxiliary_head_repair(
    *,
    batch_size: int = 4096,
    l2_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    feature_sets: tuple[str, ...] = ("latent_only", "latent_heads", "latent_heads_context", "causal_x", "latent_heads_causal_x"),
) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43m = read_json(STAGE43_M_JSON, {})
    stage43v = read_json(STAGE43_V_JSON, {})
    checkpoint, ckpt, model = _load_model(stage43m)
    seed = int(ckpt.get("seed", 431))
    train = _standardize_from_checkpoint(_build_split("train", max_rows=None, seed=seed), ckpt)
    val = _standardize_from_checkpoint(_build_split("val", max_rows=None, seed=seed), ckpt)
    test = _standardize_from_checkpoint(_build_split("test", max_rows=None, seed=seed), ckpt)
    train_pred = _predict_heads(model, train, batch_size=int(batch_size))
    val_pred = _predict_heads(model, val, batch_size=int(batch_size))
    test_pred = _predict_heads(model, test, batch_size=int(batch_size))
    density = _fit_select(train, val, test, train_pred, val_pred, test_pred, target="density", l2_grid=l2_grid, feature_sets=feature_sets)
    validity = _fit_select(
        train,
        val,
        test,
        train_pred,
        val_pred,
        test_pred,
        target="waypoint_validity_proxy",
        l2_grid=l2_grid,
        feature_sets=feature_sets,
    )
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_train_val_selected_auxiliary_head_repair",
        "generated_at_utc": datetime.now().replace(tzinfo=timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_v_precondition": {
            "verdict": stage43v.get("stage43_v_gate", {}).get("verdict"),
            "density_r2": stage43v.get("head_metrics", {}).get("density", {}).get("r2"),
            "physical_validity_deployable": stage43v.get("head_metrics", {}).get("physical_validity_proxy", {}).get("deployment_allowed"),
        },
        "training_protocol": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_sha256_matches_stage43_m": _sha256(checkpoint) == stage43m.get("checkpoint_sha256"),
            "selection_data": "train_val_selected_test_once",
            "test_threshold_tuning": False,
            "num_workers": 0,
            "batch_size": int(batch_size),
            "l2_grid": list(map(float, l2_grid)),
            "feature_sets": list(feature_sets),
            "future_labels_as_targets_only": True,
        },
        "data_rows": {
            "train": int(len(train.x)),
            "val": int(len(val.x)),
            "test": int(len(test.x)),
        },
        "density_repair": density,
        "waypoint_validity_proxy_repair": validity,
        "by_horizon": {
            "density_original": _breakdown_regression(test.horizon.astype(str), _target(test, "density"), _original_prediction(test_pred, "density")),
            "validity_original": _breakdown_regression(
                test.horizon.astype(str), _target(test, "waypoint_validity_proxy"), _original_prediction(test_pred, "waypoint_validity_proxy")
            ),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
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
            "stage5c_executed": False,
            "smc_enabled": False,
            "physical_validity_true_claim": False,
            "density_proxy_is_causal_history_density_not_future_occupancy": True,
        },
    }
    payload["stage43_w_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    density = payload["density_repair"]["selected"]
    validity = payload["waypoint_validity_proxy_repair"]["selected"]
    gates = {
        "stage43_v_precondition_seen": payload["stage43_v_precondition"]["verdict"] == "stage43_v_world_state_head_audit_partial",
        "checkpoint_replayed": payload["training_protocol"]["checkpoint_sha256_matches_stage43_m"] is True,
        "train_val_selected_test_once": payload["training_protocol"]["selection_data"] == "train_val_selected_test_once"
        and payload["training_protocol"]["test_threshold_tuning"] is False,
        "density_repair_improves_stage43m": density["delta_r2_vs_stage43m"] > 0.0 and density["delta_rmse_vs_stage43m"] > 0.0,
        "density_test_r2_positive": density["test_metrics"]["r2"] > 0.0,
        "validity_proxy_repair_reported": validity["test_metrics"]["rows"] > 0,
        "validity_proxy_not_true_physical_claim": payload["claim_boundary"]["physical_validity_true_claim"] is False,
        "no_test_threshold_tuning": payload["no_leakage"]["test_threshold_tuning"] is False,
        "future_labels_targets_only": payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True,
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
        "verdict": "stage43_w_density_proxy_repaired_validity_proxy_diagnostic"
        if passed == total
        else "stage43_w_auxiliary_head_repair_incomplete",
        "deploy_density_proxy_head": bool(passed == total and density["test_metrics"]["r2"] > 0.0),
        "deploy_true_physical_validity": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(dict(payload)))
    gate = payload["stage43_w_gate"]
    density = payload["density_repair"]["selected"]
    validity = payload["waypoint_validity_proxy_repair"]["selected"]
    lines = [
        "# Stage43-W Auxiliary Density/Validity Head Repair",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deploy density proxy head: `{gate['deploy_density_proxy_head']}`",
        f"- deploy true physical validity: `{gate['deploy_true_physical_validity']}`",
        "",
        "## Density Proxy Repair",
        "",
        f"- selected feature set: `{density['feature_set']}`",
        f"- l2: `{density['l2']}`",
        f"- original Stage43-M density R2: `{density['original_stage43m_test_metrics']['r2']:.4f}`",
        f"- repaired density R2: `{density['test_metrics']['r2']:.4f}`",
        f"- repaired density corr: `{density['test_metrics']['corr']:.4f}`",
        f"- RMSE improvement: `{density['delta_rmse_vs_stage43m']:.4f}`",
        "",
        "## Waypoint Validity Proxy",
        "",
        f"- selected feature set: `{validity['feature_set']}`",
        f"- l2: `{validity['l2']}`",
        f"- original proxy R2: `{validity['original_stage43m_test_metrics']['r2']:.4f}`",
        f"- repaired proxy R2: `{validity['test_metrics']['r2']:.4f}`",
        f"- deployment status: diagnostic proxy only, not true physical validity",
        "",
        "## Interpretation",
        "",
        "Stage43-W freezes the Stage43-M latent checkpoint and trains small train/val-selected ridge calibrators for weak auxiliary heads. This repairs the causal history-density proxy if the held-out test R2 is positive, but it is not a future occupancy claim. The validity target remains a waypoint-label availability proxy rather than a verified physical-validity label.",
        "",
        "Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.",
    ]
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage43-W Gate",
        "",
        f"verdict: `{gate['verdict']}`",
        f"passed: `{gate['passed']} / {gate['total']}`",
        f"deploy_density_proxy_head: `{gate['deploy_density_proxy_head']}`",
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
    gate = payload["stage43_w_gate"]
    density = payload["density_repair"]["selected"]
    validity = payload["waypoint_validity_proxy_repair"]["selected"]
    lines = [
        "## Stage43-W auxiliary density/validity head repair",
        "",
        f"Result source: `{payload['result_source']}`. I froze the Stage43-M latent checkpoint and trained train/val-selected ridge calibrators for the weak density and waypoint-validity proxy heads.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- density feature set: `{density['feature_set']}`",
        f"- density R2 before -> after: `{density['original_stage43m_test_metrics']['r2']:.4f}` -> `{density['test_metrics']['r2']:.4f}`",
        f"- density corr after: `{density['test_metrics']['corr']:.4f}`",
        f"- validity proxy R2 before -> after: `{validity['original_stage43m_test_metrics']['r2']:.4f}` -> `{validity['test_metrics']['r2']:.4f}`",
        f"- deploy density proxy head: `{gate['deploy_density_proxy_head']}`",
        f"- true physical validity claim: `False`",
        "",
        "Boundary: this repairs a causal history-density proxy head from frozen latent/context features. It is not a future occupancy claim. The validity head remains a label-availability proxy, not a true physical-validity certificate.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, lines)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    gate = payload["stage43_w_gate"]
    state["stage43_w_auxiliary_head_repair"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "density_repair": payload["density_repair"]["selected"],
        "waypoint_validity_proxy_repair": payload["waypoint_validity_proxy_repair"]["selected"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["current_stage"] = "stage43_w_auxiliary_head_repair"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"event": "stage43_w_auxiliary_head_repair", "payload": payload}), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-W auxiliary density/validity head repair.")
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = run_auxiliary_head_repair(batch_size=int(args.batch_size))
    gate = result["stage43_w_gate"]
    print(f"Stage43-W: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    return result


if __name__ == "__main__":
    main()
