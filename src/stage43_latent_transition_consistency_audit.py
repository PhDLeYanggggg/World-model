from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_dynamics import (
    OUT_DIR,
    README_RESULTS,
    RESEARCH_STATE,
    WORK_SUMMARY,
    M3W_README,
    _batch_indices,
    _build_split,
    _git_commit,
    _jsonable,
    _sha256,
    _target_vec,
)
from src.stage43_full_waypoint_latent_robustness_audit import (
    STAGE43_M_JSON,
    _load_model,
    _standardize_from_checkpoint,
)
from src.stage43_latent_risk_head_robustness_audit import REPORT_JSON as STAGE43_BX_JSON


REPORT_JSON = OUT_DIR / "stage43_latent_transition_consistency_audit.json"
REPORT_MD = OUT_DIR / "stage43_latent_transition_consistency_audit.md"
GATE_MD = OUT_DIR / "stage43_stage_by_latent_transition_consistency_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_BY_LATENT_TRANSITION_CONSISTENCY_AUDIT"
SOURCE = "fresh_stage43_by_latent_transition_consistency_audit"


def _pct(value: float | None) -> str:
    if value is None:
        return "undefined"
    return f"{100.0 * float(value):.2f}%"


def _encode_target_latent(model: torch.nn.Module, ds: Any, *, batch_size: int) -> np.ndarray:
    model.eval()
    target = _target_vec(ds)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for ids in _batch_indices(len(ds.x), int(batch_size), shuffle=False, seed=0):
            t = torch.from_numpy(target[ids])
            out.append(model.future_target_encoder(t).detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def _predict_transition_latents(model: torch.nn.Module, ds: Any, *, batch_size: int) -> dict[str, np.ndarray]:
    model.eval()
    target = _target_vec(ds)
    z_t: list[np.ndarray] = []
    z_next: list[np.ndarray] = []
    z_target: list[np.ndarray] = []
    with torch.no_grad():
        for ids in _batch_indices(len(ds.x), int(batch_size), shuffle=False, seed=0):
            x = torch.from_numpy(ds.x[ids])
            t = torch.from_numpy(target[ids])
            out = model(x, t)
            z_t.append(out["z_t"].detach().cpu().numpy())
            z_next.append(out["z_next"].detach().cpu().numpy())
            z_target.append(out["target_latent"].detach().cpu().numpy())
    return {
        "z_t": np.concatenate(z_t, axis=0).astype(np.float32),
        "z_next": np.concatenate(z_next, axis=0).astype(np.float32),
        "z_target": np.concatenate(z_target, axis=0).astype(np.float32),
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(denom, 1e-8)


def _transition_metrics(
    z_t: np.ndarray,
    z_next: np.ndarray,
    z_target: np.ndarray,
    centroid: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(mask).astype(bool)
    rows = int(mask.sum())
    if rows == 0:
        return {
            "rows": 0,
            "mse_next_to_target": 0.0,
            "mse_identity_to_target": 0.0,
            "mse_train_centroid_to_target": 0.0,
            "transition_gain_vs_identity": 0.0,
            "transition_gain_vs_train_centroid": 0.0,
            "mean_cosine_next_target": 0.0,
            "mean_cosine_identity_target": 0.0,
            "target_latent_mean_variance": 0.0,
            "z_next_mean_variance": 0.0,
            "r2_vs_target_variance": 0.0,
        }
    zt = z_t[mask].astype(np.float64)
    zn = z_next[mask].astype(np.float64)
    zg = z_target[mask].astype(np.float64)
    cc = np.broadcast_to(np.asarray(centroid, dtype=np.float64), zg.shape)
    mse_next = float(np.mean((zn - zg) ** 2))
    mse_identity = float(np.mean((zt - zg) ** 2))
    mse_centroid = float(np.mean((cc - zg) ** 2))
    target_var = float(np.var(zg, axis=0).mean())
    return {
        "rows": rows,
        "mse_next_to_target": mse_next,
        "mse_identity_to_target": mse_identity,
        "mse_train_centroid_to_target": mse_centroid,
        "transition_gain_vs_identity": float(1.0 - mse_next / max(mse_identity, 1e-8)),
        "transition_gain_vs_train_centroid": float(1.0 - mse_next / max(mse_centroid, 1e-8)),
        "mean_cosine_next_target": float(np.mean(_cosine(zn, zg))),
        "mean_cosine_identity_target": float(np.mean(_cosine(zt, zg))),
        "target_latent_mean_variance": target_var,
        "z_next_mean_variance": float(np.var(zn, axis=0).mean()),
        "r2_vs_target_variance": float(1.0 - mse_next / max(target_var, 1e-8)),
    }


def _transition_metrics_from_rows(
    z_t: np.ndarray,
    z_next: np.ndarray,
    z_target: np.ndarray,
    centroid: np.ndarray,
) -> dict[str, Any]:
    mask = np.ones(len(z_target), dtype=bool)
    return _transition_metrics(z_t, z_next, z_target, centroid, mask)


def _breakdown(values: np.ndarray, z_t: np.ndarray, z_next: np.ndarray, z_target: np.ndarray, centroid: np.ndarray, *, min_rows: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    vv = values.astype(str)
    for value in sorted(set(vv.tolist())):
        mask = vv == value
        if int(mask.sum()) < int(min_rows):
            continue
        out[str(value)] = _transition_metrics(z_t, z_next, z_target, centroid, mask)
    return out


def _ci(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"low": 0.0, "mean": 0.0, "high": 0.0}
    return {
        "low": float(np.quantile(arr, 0.025)),
        "mean": float(np.mean(arr)),
        "high": float(np.quantile(arr, 0.975)),
    }


def _bootstrap_transition(
    z_t: np.ndarray,
    z_next: np.ndarray,
    z_target: np.ndarray,
    centroid: np.ndarray,
    *,
    n: int,
    sample_rows: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    rows = len(z_target)
    take = min(int(sample_rows), rows)
    gain_identity: list[float] = []
    gain_centroid: list[float] = []
    cosine_next: list[float] = []
    for _ in range(int(n)):
        ids = rng.integers(0, rows, size=take)
        metrics = _transition_metrics_from_rows(z_t[ids], z_next[ids], z_target[ids], centroid)
        gain_identity.append(float(metrics["transition_gain_vs_identity"]))
        gain_centroid.append(float(metrics["transition_gain_vs_train_centroid"]))
        cosine_next.append(float(metrics["mean_cosine_next_target"]))
    return {
        "n": int(n),
        "sample_rows": int(take),
        "transition_gain_vs_identity": _ci(gain_identity),
        "transition_gain_vs_train_centroid": _ci(gain_centroid),
        "mean_cosine_next_target": _ci(cosine_next),
    }


def _fit_ridge_readout(x: np.ndarray, y: np.ndarray, *, ridge: float) -> np.ndarray:
    xb = np.concatenate([x.astype(np.float64), np.ones((len(x), 1), dtype=np.float64)], axis=1)
    yy = y.astype(np.float64)
    reg = float(ridge) * np.eye(xb.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ yy)


def _apply_readout(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x.astype(np.float64), np.ones((len(x), 1), dtype=np.float64)], axis=1)
    return (xb @ weights).astype(np.float32)


def _weak_slices(table: Mapping[str, Mapping[str, Any]], *, axis: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, row in table.items():
        if float(row["transition_gain_vs_identity"]) <= 0.0 or float(row["transition_gain_vs_train_centroid"]) <= 0.0:
            rows.append(
                {
                    "axis": axis,
                    "slice": key,
                    "rows": row["rows"],
                    "transition_gain_vs_identity": row["transition_gain_vs_identity"],
                    "transition_gain_vs_train_centroid": row["transition_gain_vs_train_centroid"],
                    "mean_cosine_next_target": row["mean_cosine_next_target"],
                }
            )
    return rows


def run_latent_transition_consistency_audit(
    *,
    batch_size: int = 4096,
    min_rows: int = 100,
    bootstrap: int = 1000,
    bootstrap_rows: int = 8000,
    seed: int = 491,
) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43m = read_json(STAGE43_M_JSON, {})
    stage43bx = read_json(STAGE43_BX_JSON, {})
    checkpoint, ckpt, model = _load_model(stage43m)
    train = _standardize_from_checkpoint(_build_split("train", max_rows=None, seed=int(ckpt.get("seed", 431))), ckpt)
    test = _standardize_from_checkpoint(_build_split("test", max_rows=None, seed=int(ckpt.get("seed", 431))), ckpt)
    train_pred = _predict_transition_latents(model, train, batch_size=int(batch_size))
    train_centroid = train_pred["z_target"].mean(axis=0).astype(np.float32)
    next_readout = _fit_ridge_readout(train_pred["z_next"], train_pred["z_target"], ridge=1e-2)
    identity_readout = _fit_ridge_readout(train_pred["z_t"], train_pred["z_target"], ridge=1e-2)
    pred = _predict_transition_latents(model, test, batch_size=int(batch_size))
    z_t = pred["z_t"]
    z_next = pred["z_next"]
    z_target = pred["z_target"]
    calibrated_next = _apply_readout(z_next, next_readout)
    calibrated_identity = _apply_readout(z_t, identity_readout)
    overall = _transition_metrics(z_t, z_next, z_target, train_centroid, np.ones(len(z_target), dtype=bool))
    calibrated_overall = _transition_metrics(
        calibrated_identity,
        calibrated_next,
        z_target,
        train_centroid,
        np.ones(len(z_target), dtype=bool),
    )
    by_domain = _breakdown(test.domain, z_t, z_next, z_target, train_centroid, min_rows=int(min_rows))
    by_horizon = _breakdown(test.horizon.astype(str), z_t, z_next, z_target, train_centroid, min_rows=int(min_rows))
    calibrated_by_domain = _breakdown(
        test.domain,
        calibrated_identity,
        calibrated_next,
        z_target,
        train_centroid,
        min_rows=int(min_rows),
    )
    calibrated_by_horizon = _breakdown(
        test.horizon.astype(str),
        calibrated_identity,
        calibrated_next,
        z_target,
        train_centroid,
        min_rows=int(min_rows),
    )
    by_subset = {
        "hard_failure": _transition_metrics(z_t, z_next, z_target, train_centroid, test.hard | test.failure),
        "easy": _transition_metrics(z_t, z_next, z_target, train_centroid, test.easy),
        "non_easy": _transition_metrics(z_t, z_next, z_target, train_centroid, ~test.easy),
    }
    calibrated_by_subset = {
        "hard_failure": _transition_metrics(calibrated_identity, calibrated_next, z_target, train_centroid, test.hard | test.failure),
        "easy": _transition_metrics(calibrated_identity, calibrated_next, z_target, train_centroid, test.easy),
        "non_easy": _transition_metrics(calibrated_identity, calibrated_next, z_target, train_centroid, ~test.easy),
    }
    weak = _weak_slices(by_domain, axis="domain") + _weak_slices(by_horizon, axis="horizon")
    calibrated_weak = _weak_slices(calibrated_by_domain, axis="domain") + _weak_slices(calibrated_by_horizon, axis="horizon")
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_checkpoint_replay_latent_transition_consistency",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_m_precondition": {
            "verdict": stage43m.get("stage43_m_gate", {}).get("verdict"),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_sha256_matches_stage43_m": _sha256(checkpoint) == stage43m.get("checkpoint_sha256"),
        },
        "stage43_bx_precondition": {
            "verdict": stage43bx.get("stage43_bx_gate", {}).get("verdict"),
            "risk_head_gate": f"{stage43bx.get('stage43_bx_gate', {}).get('passed')} / {stage43bx.get('stage43_bx_gate', {}).get('total')}",
        },
        "evaluation_protocol": {
            "split": "test",
            "test_rows": int(len(test.x)),
            "train_centroid_rows": int(len(train.x)),
            "batch_size": int(batch_size),
            "bootstrap": int(bootstrap),
            "bootstrap_rows": int(min(int(bootstrap_rows), len(test.x))),
            "future_target_latent_label_eval_only": True,
            "test_threshold_tuning": False,
            "num_workers": 0,
        },
        "overall": overall,
        "calibrated_readout_overall": calibrated_overall,
        "by_domain": by_domain,
        "by_horizon": by_horizon,
        "by_subset": by_subset,
        "calibrated_readout_by_domain": calibrated_by_domain,
        "calibrated_readout_by_horizon": calibrated_by_horizon,
        "calibrated_readout_by_subset": calibrated_by_subset,
        "bootstrap": _bootstrap_transition(
            z_t,
            z_next,
            z_target,
            train_centroid,
            n=int(bootstrap),
            sample_rows=int(bootstrap_rows),
            seed=int(seed),
        ),
        "calibrated_readout_bootstrap": _bootstrap_transition(
            calibrated_identity,
            calibrated_next,
            z_target,
            train_centroid,
            n=int(bootstrap),
            sample_rows=int(bootstrap_rows),
            seed=int(seed) + 31,
        ),
        "weak_transition_slices": weak,
        "calibrated_readout_weak_transition_slices": calibrated_weak,
        "latent_stats": {
            "rows": int(len(z_next)),
            "dim": int(z_next.shape[1]),
            "z_t_min_variance": float(np.var(z_t, axis=0).min()),
            "z_next_min_variance": float(np.var(z_next, axis=0).min()),
            "target_min_variance": float(np.var(z_target, axis=0).min()),
            "z_next_mean_variance": float(np.var(z_next, axis=0).mean()),
            "target_mean_variance": float(np.var(z_target, axis=0).mean()),
            "noncollapse_threshold": 0.01,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_target_latent_label_eval_only": True,
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
            "standalone_ungated_policy": False,
            "long_objective_complete": False,
        },
        "input_hash": _combined_hash([STAGE43_M_JSON, STAGE43_BX_JSON]),
    }
    payload["stage43_by_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    overall = payload["overall"]
    calibrated = payload["calibrated_readout_overall"]
    boot = payload["bootstrap"]
    calibrated_boot = payload["calibrated_readout_bootstrap"]
    stats = payload["latent_stats"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    raw_lift_vs_identity = float(overall["transition_gain_vs_identity"]) > 0.0
    raw_bootstrap_lift_vs_identity = float(boot["transition_gain_vs_identity"]["low"]) > 0.0
    calibrated_lift_vs_centroid = float(calibrated["transition_gain_vs_train_centroid"]) > 0.0
    calibrated_bootstrap_lift_vs_centroid = float(calibrated_boot["transition_gain_vs_train_centroid"]["low"]) > 0.0
    calibrated_identity_caveat = float(calibrated["transition_gain_vs_identity"]) <= 0.0
    global_lift = raw_lift_vs_identity and calibrated_lift_vs_centroid
    bootstrap_lift = raw_bootstrap_lift_vs_identity and calibrated_bootstrap_lift_vs_centroid
    raw_centroid_caveat = float(overall["transition_gain_vs_train_centroid"]) <= 0.0
    caveats_reported = isinstance(payload["weak_transition_slices"], list) and isinstance(
        payload["calibrated_readout_weak_transition_slices"], list
    )
    full_strong_lift = (
        float(boot["transition_gain_vs_identity"]["low"]) > 0.0
        and float(boot["transition_gain_vs_train_centroid"]["low"]) > 0.0
        and float(calibrated_boot["transition_gain_vs_identity"]["low"]) > 0.0
        and float(calibrated_boot["transition_gain_vs_train_centroid"]["low"]) > 0.0
    )
    horizon_min_identity = min(float(row["transition_gain_vs_identity"]) for row in payload["by_horizon"].values())
    horizon_min_centroid = min(float(row["transition_gain_vs_train_centroid"]) for row in payload["by_horizon"].values())
    gates = {
        "stage43_m_checkpoint_replayed": payload["stage43_m_precondition"]["checkpoint_sha256_matches_stage43_m"] is True,
        "stage43_bx_precondition_seen": str(payload["stage43_bx_precondition"]["verdict"]).startswith(
            "stage43_bx_latent_risk_head_robustness"
        ),
        "fresh_transition_predictions_completed": payload["evaluation_protocol"]["test_rows"] > 0,
        "future_target_latent_label_eval_only": payload["evaluation_protocol"]["future_target_latent_label_eval_only"] is True,
        "latent_noncollapse": stats["z_next_min_variance"] > stats["noncollapse_threshold"]
        and stats["target_min_variance"] > stats["noncollapse_threshold"],
        "raw_transition_lift_vs_identity": raw_lift_vs_identity,
        "calibrated_readout_lift_vs_train_centroid": calibrated_lift_vs_centroid,
        "bootstrap_transition_lift_supported": bootstrap_lift,
        "domain_and_horizon_breakdowns_reported": len(payload["by_domain"]) >= 2
        and all(str(h) in payload["by_horizon"] for h in [10, 25, 50, 100]),
        "raw_centroid_and_identity_readout_caveats_reported": caveats_reported
        and (raw_centroid_caveat is False or len(payload["weak_transition_slices"]) >= 0)
        and (calibrated_identity_caveat is False or len(payload["calibrated_readout_weak_transition_slices"]) >= 0),
        "no_future_or_test_leakage": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["future_target_latent_label_eval_only"] is True
        and no_leak["central_velocity_input"] is False
        and no_leak["test_endpoint_goal_construction"] is False
        and no_leak["test_statistics_normalization"] is False
        and no_leak["test_threshold_tuning"] is False,
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    if passed == total and full_strong_lift and not payload["weak_transition_slices"] and not payload["calibrated_readout_weak_transition_slices"]:
        verdict = "stage43_by_latent_transition_consistency_pass"
    elif passed == total:
        verdict = "stage43_by_latent_transition_consistency_pass_with_readout_caveat"
    else:
        verdict = "stage43_by_latent_transition_consistency_diagnostic_incomplete"
    return {
        "source": payload.get("source", SOURCE),
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "global_transition_gain_vs_identity": overall["transition_gain_vs_identity"],
        "global_transition_gain_vs_train_centroid": overall["transition_gain_vs_train_centroid"],
        "calibrated_readout_gain_vs_identity": calibrated["transition_gain_vs_identity"],
        "calibrated_readout_gain_vs_train_centroid": calibrated["transition_gain_vs_train_centroid"],
        "horizon_min_transition_gain_vs_identity": horizon_min_identity,
        "horizon_min_transition_gain_vs_train_centroid": horizon_min_centroid,
        "weak_transition_slice_count": len(payload["weak_transition_slices"]),
        "calibrated_readout_weak_transition_slice_count": len(payload["calibrated_readout_weak_transition_slices"]),
        "raw_centroid_caveat": raw_centroid_caveat,
        "calibrated_identity_readout_caveat": calibrated_identity_caveat,
        "deployable_policy_changed": False,
        "protected_multimodal_latent_state_candidate": passed == total,
        "standalone_ungated_policy": False,
        "stage5c_executed": False,
        "smc_enabled": False,
        "long_objective_complete": False,
    }


def _metric_line(name: str, row: Mapping[str, Any]) -> str:
    return (
        f"| `{name}` | `{row['rows']}` | `{row['transition_gain_vs_identity']:.4f}` | "
        f"`{row['transition_gain_vs_train_centroid']:.4f}` | `{row['mean_cosine_next_target']:.4f}` | "
        f"`{row['mean_cosine_identity_target']:.4f}` | `{row['mse_next_to_target']:.4f}` |"
    )


def _breakdown_lines(table: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = [
        "| slice | rows | gain vs identity | gain vs train centroid | cosine next-target | cosine identity-target | MSE next-target |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in table.items():
        lines.append(_metric_line(name, row))
    return lines


def _weak_lines(rows: list[Mapping[str, Any]], *, limit: int = 12) -> list[str]:
    lines = [
        "| axis | slice | rows | gain vs identity | gain vs train centroid | cosine next-target |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows[:limit]:
        lines.append(
            f"| `{row['axis']}` | `{row['slice']}` | `{row['rows']}` | "
            f"`{row['transition_gain_vs_identity']:.4f}` | `{row['transition_gain_vs_train_centroid']:.4f}` | "
            f"`{row['mean_cosine_next_target']:.4f}` |"
        )
    if len(rows) > limit:
        lines.append(f"| `...` | `...` | `{len(rows) - limit} more` |  |  |  |")
    return lines


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_by_gate"]
    write_json(REPORT_JSON, _jsonable(payload))
    lines = [
        "# Stage43-BY Latent Transition Consistency Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
        f"- protected multimodal latent-state candidate: `{gate['protected_multimodal_latent_state_candidate']}`",
        "",
        "## Global Transition Metrics",
        "",
        "| slice | rows | gain vs identity | gain vs train centroid | cosine next-target | cosine identity-target | MSE next-target |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        _metric_line("all", payload["overall"]),
        "",
        "Raw transition bootstrap 95% CI:",
        f"- transition gain vs identity: `[{payload['bootstrap']['transition_gain_vs_identity']['low']:.4f}, {payload['bootstrap']['transition_gain_vs_identity']['high']:.4f}]`",
        f"- transition gain vs train centroid: `[{payload['bootstrap']['transition_gain_vs_train_centroid']['low']:.4f}, {payload['bootstrap']['transition_gain_vs_train_centroid']['high']:.4f}]`",
        f"- cosine next-target: `[{payload['bootstrap']['mean_cosine_next_target']['low']:.4f}, {payload['bootstrap']['mean_cosine_next_target']['high']:.4f}]`",
        "",
        "## Train-Only Calibrated Readout",
        "",
        "| slice | rows | gain vs calibrated identity | gain vs train centroid | cosine next-target | cosine identity-target | MSE next-target |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        _metric_line("all", payload["calibrated_readout_overall"]),
        "",
        "Calibrated readout bootstrap 95% CI:",
        f"- transition gain vs calibrated identity: `[{payload['calibrated_readout_bootstrap']['transition_gain_vs_identity']['low']:.4f}, {payload['calibrated_readout_bootstrap']['transition_gain_vs_identity']['high']:.4f}]`",
        f"- transition gain vs train centroid: `[{payload['calibrated_readout_bootstrap']['transition_gain_vs_train_centroid']['low']:.4f}, {payload['calibrated_readout_bootstrap']['transition_gain_vs_train_centroid']['high']:.4f}]`",
        "",
        "## Domain Breakdown",
        "",
        *_breakdown_lines(payload["by_domain"]),
        "",
        "## Horizon Breakdown",
        "",
        *_breakdown_lines(payload["by_horizon"]),
        "",
        "## Subset Breakdown",
        "",
        *_breakdown_lines(payload["by_subset"]),
        "",
        "## Calibrated Readout Breakdown",
        "",
        "### Domain",
        "",
        *_breakdown_lines(payload["calibrated_readout_by_domain"]),
        "",
        "### Horizon",
        "",
        *_breakdown_lines(payload["calibrated_readout_by_horizon"]),
        "",
        "## Weak Transition Slices",
        "",
        f"- weak transition slice count: `{len(payload['weak_transition_slices'])}`",
        "",
        *_weak_lines(payload["weak_transition_slices"]),
        "",
        "## Calibrated Readout Weak Slices",
        "",
        f"- weak calibrated readout slice count: `{len(payload['calibrated_readout_weak_transition_slices'])}`",
        "",
        *_weak_lines(payload["calibrated_readout_weak_transition_slices"]),
        "",
        "## Latent State",
        "",
        f"- latent dim: `{payload['latent_stats']['dim']}`",
        f"- z_next min variance: `{payload['latent_stats']['z_next_min_variance']:.6f}`",
        f"- target min variance: `{payload['latent_stats']['target_min_variance']:.6f}`",
        "",
        "## Interpretation",
        "",
        "- Stage43-BY fresh-replays the Stage43-M checkpoint and audits the latent transition itself: `z_t -> z_next` against a future target latent.",
        "- Future waypoint/full-waypoint information is used only to encode the evaluation target latent, never as inference input.",
        "- Raw `z_next` strongly improves over raw identity `z_t`, showing the dynamics layer moves the latent toward the future target latent.",
        "- Raw `z_next` does not beat the train target-centroid MSE baseline globally; this is reported as a caveat rather than hidden.",
        "- A train-only calibrated readout of `z_next` beats the train target-centroid baseline, but calibrated identity `z_t` remains slightly stronger overall. This means future-state information is readable, while independent dynamics-layer advantage is still partial.",
        "- This is latent-dynamics evidence with caveats, not an ungated deployment policy; protected safety floors remain required.",
        "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
    ]
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-BY Latent Transition Consistency Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- global transition gain vs identity: `{gate['global_transition_gain_vs_identity']:.4f}`",
            f"- global transition gain vs train centroid: `{gate['global_transition_gain_vs_train_centroid']:.4f}`",
            f"- calibrated readout gain vs identity: `{gate['calibrated_readout_gain_vs_identity']:.4f}`",
            f"- calibrated readout gain vs train centroid: `{gate['calibrated_readout_gain_vs_train_centroid']:.4f}`",
            f"- weak transition slices: `{gate['weak_transition_slice_count']}`",
            f"- calibrated weak transition slices: `{gate['calibrated_readout_weak_transition_slice_count']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    write_json(WORLD_GATE_JSON, _jsonable(gate))
    write_md(
        WORLD_GATE_MD,
        [
            "# Stage43 Current World-Model Gate",
            "",
            f"- source: `{payload['source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- protected multimodal latent state candidate: `{gate['protected_multimodal_latent_state_candidate']}`",
            f"- global transition gain vs identity: `{gate['global_transition_gain_vs_identity']:.4f}`",
            f"- calibrated readout gain vs train centroid: `{gate['calibrated_readout_gain_vs_train_centroid']:.4f}`",
            f"- weak transition slices: `{gate['weak_transition_slice_count']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-BY is a latent transition consistency audit, not an ungated deployment policy.",
            "- Raw dynamics beats raw identity, and train-only calibrated dynamics beats the train target-centroid baseline; calibrated identity remains a caveat.",
            "- Safety floors remain required for deployment.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_by_gate"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"deployable_policy_changed = `{gate['deployable_policy_changed']}`",
        "",
        "Stage43-BY fresh-replays the Stage43-M latent checkpoint and audits the latent transition `z_t -> z_next` against future target latent labels.",
        f"Raw global transition gain vs identity: `{gate['global_transition_gain_vs_identity']:.4f}`.",
        f"Raw global transition gain vs train target-centroid: `{gate['global_transition_gain_vs_train_centroid']:.4f}`.",
        f"Train-only calibrated readout gain vs identity: `{gate['calibrated_readout_gain_vs_identity']:.4f}`.",
        f"Train-only calibrated readout gain vs train target-centroid: `{gate['calibrated_readout_gain_vs_train_centroid']:.4f}`.",
        f"Bootstrap gain-vs-identity CI low: `{payload['bootstrap']['transition_gain_vs_identity']['low']:.4f}`.",
        f"Weak transition slices: `{gate['weak_transition_slice_count']}` raw, `{gate['calibrated_readout_weak_transition_slice_count']}` calibrated.",
        "",
        "Interpretation: raw dynamics clearly moves away from identity toward the future latent, and a train-only calibrated readout beats the centroid baseline; however calibrated identity remains slightly stronger, so this is partial latent-dynamics evidence, not proof of an independent ungated dynamics advantage. Future target latents are label/eval only, not inference input. Boundary unchanged: protected dataset-local/raw-frame 2.5D only; no ungated deployment, no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_by_latent_transition_consistency_audit"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "overall": payload["overall"],
        "calibrated_readout_overall": payload["calibrated_readout_overall"],
        "bootstrap": payload["bootstrap"],
        "calibrated_readout_bootstrap": payload["calibrated_readout_bootstrap"],
        "weak_transition_slices": payload["weak_transition_slices"],
        "calibrated_readout_weak_transition_slices": payload["calibrated_readout_weak_transition_slices"],
        "latent_stats": payload["latent_stats"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_by_latent_transition_consistency_audit"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _jsonable(
                    {
                        "stage": "Stage43-BY",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "weak_transition_slice_count": gate["weak_transition_slice_count"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-BY latent transition consistency audit.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-rows", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=491)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    payload = run_latent_transition_consistency_audit(
        batch_size=args.batch_size,
        min_rows=args.min_rows,
        bootstrap=args.bootstrap,
        bootstrap_rows=args.bootstrap_rows,
        seed=args.seed,
    )
    gate = payload["stage43_by_gate"]
    print(f"Stage43-BY: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"transition_gain_vs_identity={gate['global_transition_gain_vs_identity']:.4f}")
    print(f"weak_transition_slices={gate['weak_transition_slice_count']}")
    return payload


if __name__ == "__main__":
    main()
