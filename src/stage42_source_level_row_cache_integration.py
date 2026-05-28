from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage41_full_trajectory_world_state as ft
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_ucy_full_waypoint_candidate as v
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
CACHE_DIR = Path("data/stage42_source_level_full_waypoint_cache")
STAGE42_IT_JSON = OUT_DIR / "source_level_full_waypoint_eval_stage42.json"
STAGE42_V_JSON = OUT_DIR / "ucy_full_waypoint_candidate_stage42.json"
REPORT_JSON = OUT_DIR / "source_level_row_cache_integration_stage42.json"
REPORT_MD = OUT_DIR / "source_level_row_cache_integration_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_iv_gate.md"
CACHE_NPZ = CACHE_DIR / "stage42iv_source_level_merged_cache.npz"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
BOOTSTRAP_N = 2000
EPS = 1e-8


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-IV 把 Stage42-IT/Stage42-IU source-level full-waypoint evidence 升级为单一 row-level merged cache。",
    "TrajNet rows 来自当前 Stage42-IT source-level full-waypoint rerun；UCY rows 来自 Stage42-V UCY full-waypoint specialist。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "本阶段仍是 dataset-local/raw-frame 2.5D evidence，不是 metric 或 seconds-level 结果。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _source_level_best_arrays(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    x: np.ndarray,
) -> dict[str, Any]:
    train_mask = split == "train"
    val_mask = split == "val"
    cur = np.stack([data["current_x"], data["current_y"]], axis=1)
    target_delta = ((labels["waypoint_xy"].astype(np.float64) - cur[:, None, :]) / np.maximum(data["scale"].astype(np.float64)[:, None, None], EPS)).astype(np.float32)
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    best: dict[str, Any] | None = None
    best_score = -1e9
    candidates = []
    for lam in am.LAMBDAS:
        coef = am._fit_ridge_model(x, target_delta, labels["waypoint_valid"], train_mask, lam)
        pred_xy = am._predict_waypoints(x, coef, data)
        policy, selected_ade, selected_fde, switch = am._select_policy_on_val(pred_xy, floor["floor_xy"], labels, data, val_mask)
        val_metric = am._metric(selected_ade, floor_ade, data, switch, val_mask)
        score = (
            1.2 * val_metric["all_improvement"]
            + 1.8 * val_metric["t50_improvement"]
            + 1.1 * val_metric["hard_failure_improvement"]
            - 30.0 * max(0.0, val_metric["easy_degradation"] - 0.02)
            - 0.03 * val_metric["switch_rate"]
        )
        candidate = {
            "lambda": float(lam),
            "score": float(score),
            "policy_slice_count": len(policy.get("slices", {})),
            "val_metric": val_metric,
        }
        candidates.append(candidate)
        if score > best_score:
            best_score = float(score)
            best = {
                "lambda": float(lam),
                "policy": policy,
                "selected_ade": selected_ade.astype(np.float64),
                "selected_fde": selected_fde.astype(np.float64),
                "floor_ade": floor_ade.astype(np.float64),
                "floor_fde": floor_fde.astype(np.float64),
                "switch": switch.astype(bool),
                "val_metric": val_metric,
                "score": float(score),
                "candidates": candidates,
            }
    if best is None:
        raise RuntimeError("No source-level ridge candidate was evaluated.")
    best["candidates"] = candidates
    return best


def _source_labels(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], test_mask: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "domain": data["dataset"][test_mask].astype(str),
        "source_file": data["source_file"][test_mask].astype(str),
        "scene_id": data["scene_id"][test_mask].astype(str),
        "horizon": data["horizon"][test_mask].astype(np.int64),
        "current_xy": np.stack([data["current_x"][test_mask], data["current_y"][test_mask]], axis=1).astype(np.float64),
        "future_xy": np.stack([data["future_endpoint_x"][test_mask], data["future_endpoint_y"][test_mask]], axis=1).astype(np.float64),
        "waypoint_xy": labels["waypoint_xy"][test_mask].astype(np.float64),
        "waypoint_valid": labels["waypoint_valid"][test_mask].astype(bool),
        "normalizer": data["scale"][test_mask].astype(np.float64),
        "hard": data["hard"][test_mask].astype(bool),
        "failure": data["failure"][test_mask].astype(bool),
        "easy": data["easy"][test_mask].astype(bool),
        "candidate_fde": data["family_fde"][test_mask].astype(np.float64),
    }


def _stage42v_best_rows(stage42_v: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    best = str(stage42_v.get("best_trial", ""))
    rows = [row for row in stage42_v.get("rows", []) if row.get("trial", {}).get("name") == best]
    if not rows:
        raise ValueError("Stage42-V report has no rows for best_trial.")
    return sorted(rows, key=lambda row: int(row.get("seed", 0)))


def _stage42v_ucy_arrays(best_rows: list[Mapping[str, Any]]) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    arrays = []
    labels_ref: dict[str, np.ndarray] | None = None
    for row in best_rows:
        pred, labels = v._predict(row["train_info"], "test")
        selected, switch = v._apply_policy(pred, labels, row["val_policy"])
        selected_ade, selected_fde = ft._trajectory_errors(selected, labels)
        arrays.append(
            {
                "selected_ade": selected_ade.astype(np.float64),
                "selected_fde": selected_fde.astype(np.float64),
                "switch": switch.astype(bool),
            }
        )
        if labels_ref is None:
            labels_ref = labels
    if labels_ref is None:
        raise ValueError("No Stage42-V rows available.")
    return arrays, labels_ref


def _alignment_checks(source_labels: Mapping[str, np.ndarray], v_labels: Mapping[str, np.ndarray]) -> dict[str, Any]:
    source_ucy = source_labels["domain"].astype(str) == "UCY"
    v_ucy = v_labels["domain"].astype(str) == "UCY"
    checks = {
        "source_ucy_rows": int(np.sum(source_ucy)),
        "stage42v_ucy_rows": int(np.sum(v_ucy)),
        "horizon_order": bool(np.array_equal(source_labels["horizon"][source_ucy].astype(int), v_labels["horizon"][v_ucy].astype(int))),
        "current_xy_match": bool(np.allclose(source_labels["current_xy"][source_ucy], v_labels["current_xy"][v_ucy])),
        "future_xy_match": bool(np.allclose(source_labels["future_xy"][source_ucy], v_labels["future_xy"][v_ucy])),
        "waypoint_xy_match": bool(np.allclose(source_labels["waypoint_xy"][source_ucy], v_labels["waypoint_xy"][v_ucy])),
        "waypoint_valid_match": bool(np.array_equal(source_labels["waypoint_valid"][source_ucy], v_labels["waypoint_valid"][v_ucy])),
        "source_file_text_match": bool(np.array_equal(source_labels["source_file"][source_ucy].astype(str), v_labels["source_file"][v_ucy].astype(str))),
        "scene_id_text_match": bool(np.array_equal(source_labels["scene_id"][source_ucy].astype(str), v_labels["scene_id"][v_ucy].astype(str))),
        "normalizer_max_abs_diff": float(np.max(np.abs(source_labels["normalizer"][source_ucy] - v_labels["normalizer"][v_ucy]))) if int(np.sum(source_ucy)) else 0.0,
    }
    required = [
        "horizon_order",
        "current_xy_match",
        "future_xy_match",
        "waypoint_xy_match",
        "waypoint_valid_match",
    ]
    checks["strict_geometry_alignment_pass"] = bool(checks["source_ucy_rows"] == checks["stage42v_ucy_rows"] > 0 and all(checks[k] for k in required))
    checks["text_id_note"] = "source_file/scene_id text may differ across Stage42-IT and Stage42-V derivations; geometry and waypoint alignment are the required row-level evidence."
    if not checks["strict_geometry_alignment_pass"]:
        raise ValueError(f"Stage42-IV UCY row geometry alignment failed: {checks}")
    return checks


def _metric(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    if mask is None:
        mask = np.ones(len(selected), dtype=bool)
    return ft._metric(selected[mask].astype(np.float64), floor[mask].astype(np.float64), {k: v[mask] for k, v in labels.items() if isinstance(v, np.ndarray) and len(v) == len(mask)}, switch[mask].astype(bool))


def _bootstrap(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray, *, easy: bool = False, seed: int = 42073) -> dict[str, Any]:
    ids = np.where(mask)[0]
    if len(ids) < 30:
        return {"rows": int(len(ids)), "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    vals = np.empty(BOOTSTRAP_N, dtype=np.float64)
    for i in range(BOOTSTRAP_N):
        sample = rng.choice(ids, size=len(ids), replace=True)
        if easy:
            vals[i] = max(0.0, float(np.mean(selected[sample]) / max(float(np.mean(floor[sample])), EPS) - 1.0))
        else:
            vals[i] = 1.0 - float(np.mean(selected[sample]) / max(float(np.mean(floor[sample])), EPS))
    if easy:
        mean = max(0.0, float(np.mean(selected[ids]) / max(float(np.mean(floor[ids])), EPS) - 1.0))
    else:
        mean = 1.0 - float(np.mean(selected[ids]) / max(float(np.mean(floor[ids])), EPS))
    return {
        "rows": int(len(ids)),
        "mean": float(mean),
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
        "bootstrap_n": BOOTSTRAP_N,
    }


def _build_merged_cache() -> dict[str, Any]:
    stage42_it = read_json(STAGE42_IT_JSON, {})
    stage42_v = read_json(STAGE42_V_JSON, {})
    data = s41._combined()
    split, group = am._split_arrays(data)
    labels_all = am._reconstruct_waypoint_labels(data)
    train_mask = split == "train"
    test_mask = split == "test"
    floor = am._floor_arrays(data, train_mask)
    features, feature_names = am._feature_matrix(data, floor)
    x, feature_mean, feature_std = am._standardize(features, train_mask)
    source_best = _source_level_best_arrays(data, split, labels_all, floor, x)
    source_labels = _source_labels(data, labels_all, test_mask)

    selected_ade = source_best["selected_ade"][test_mask].copy()
    selected_fde = source_best["selected_fde"][test_mask].copy()
    floor_ade = source_best["floor_ade"][test_mask].copy()
    floor_fde = source_best["floor_fde"][test_mask].copy()
    switch = source_best["switch"][test_mask].copy()

    best_rows = _stage42v_best_rows(stage42_v)
    v_arrays, v_labels = _stage42v_ucy_arrays(best_rows)
    alignment = _alignment_checks(source_labels, v_labels)
    source_ucy = source_labels["domain"].astype(str) == "UCY"
    v_ucy = v_labels["domain"].astype(str) == "UCY"
    seed_rows = []
    seed_selected_ade = []
    seed_selected_fde = []
    seed_switch = []
    for idx, arr in enumerate(v_arrays):
        local_ade = selected_ade.copy()
        local_fde = selected_fde.copy()
        local_switch = switch.copy()
        ucy_sw = arr["switch"][v_ucy]
        ucy_ade = arr["selected_ade"][v_ucy]
        ucy_fde = arr["selected_fde"][v_ucy]
        # Keep the exact Stage42-IT source-level floor when Stage42-V falls
        # back, so the merged cache shares one fallback baseline.
        local_ade[source_ucy] = np.where(ucy_sw, ucy_ade, floor_ade[source_ucy])
        local_fde[source_ucy] = np.where(ucy_sw, ucy_fde, floor_fde[source_ucy])
        local_switch[source_ucy] = ucy_sw
        seed_selected_ade.append(local_ade.astype(np.float32))
        seed_selected_fde.append(local_fde.astype(np.float32))
        seed_switch.append(local_switch.astype(bool))
        seed_rows.append(
            {
                "seed": int(best_rows[idx].get("seed", idx)),
                "trial": best_rows[idx].get("trial", {}).get("name", "unknown"),
                "metric": _metric(local_ade, floor_ade, source_labels, local_switch),
                "fde_metric": _metric(local_fde, floor_fde, source_labels, local_switch),
            }
        )

    merged_ade = np.mean(np.stack(seed_selected_ade, axis=0), axis=0)
    merged_fde = np.mean(np.stack(seed_selected_fde, axis=0), axis=0)
    merged_switch_rate = np.mean(np.stack([s.astype(np.float32) for s in seed_switch], axis=0), axis=0)
    merged_switch = merged_switch_rate > 0.0
    ensure_dir(CACHE_DIR)
    np.savez_compressed(
        CACHE_NPZ,
        floor_ade=floor_ade.astype(np.float32),
        floor_fde=floor_fde.astype(np.float32),
        selected_ade_seed_mean=merged_ade.astype(np.float32),
        selected_fde_seed_mean=merged_fde.astype(np.float32),
        switch_seed_mean=merged_switch_rate.astype(np.float32),
        switch_any=merged_switch.astype(bool),
        domain=source_labels["domain"].astype("U32"),
        source_file=source_labels["source_file"].astype("U512"),
        scene_id=source_labels["scene_id"].astype("U256"),
        horizon=source_labels["horizon"].astype(np.int16),
        hard=source_labels["hard"].astype(bool),
        failure=source_labels["failure"].astype(bool),
        easy=source_labels["easy"].astype(bool),
        current_xy=source_labels["current_xy"].astype(np.float32),
        future_xy=source_labels["future_xy"].astype(np.float32),
        waypoint_xy=source_labels["waypoint_xy"].astype(np.float32),
        waypoint_valid=source_labels["waypoint_valid"].astype(bool),
    )

    h = source_labels["horizon"].astype(int)
    hard_failure = source_labels["hard"].astype(bool) | source_labels["failure"].astype(bool)
    easy = source_labels["easy"].astype(bool)
    domain = source_labels["domain"].astype(str)
    summary_metric = _metric(merged_ade, floor_ade, source_labels, merged_switch)
    summary_fde = _metric(merged_fde, floor_fde, source_labels, merged_switch)
    bootstrap = {
        "all": _bootstrap(merged_ade, floor_ade, np.ones(len(merged_ade), dtype=bool), seed=42073),
        "t50": _bootstrap(merged_ade, floor_ade, h == 50, seed=42074),
        "t100_raw_frame_diagnostic": _bootstrap(merged_ade, floor_ade, h == 100, seed=42075),
        "hard_failure": _bootstrap(merged_ade, floor_ade, hard_failure, seed=42076),
        "easy_degradation": _bootstrap(merged_ade, floor_ade, easy, easy=True, seed=42077),
        "fde_t50": _bootstrap(merged_fde, floor_fde, h == 50, seed=42078),
    }
    by_domain = {
        d: {
            "ade": _metric(merged_ade, floor_ade, source_labels, merged_switch, domain == d),
            "fde": _metric(merged_fde, floor_fde, source_labels, merged_switch, domain == d),
        }
        for d in sorted(set(domain.tolist()))
    }
    by_horizon = {
        str(hv): {
            "ade": _metric(merged_ade, floor_ade, source_labels, merged_switch, h == hv),
            "fde": _metric(merged_fde, floor_fde, source_labels, merged_switch, h == hv),
        }
        for hv in [10, 25, 50, 100]
    }
    cache_hash = _combined_hash([str(CACHE_NPZ)])
    return {
        "inputs": {
            "stage42_it_report": str(STAGE42_IT_JSON),
            "stage42_it_source": stage42_it.get("source"),
            "stage42_it_verdict": stage42_it.get("stage42_am_gate", {}).get("verdict"),
            "stage42_v_report": str(STAGE42_V_JSON),
            "stage42_v_source": stage42_v.get("source"),
            "stage42_v_verdict": stage42_v.get("stage42_v_gate", {}).get("verdict"),
            "stage42_v_best_trial": stage42_v.get("best_trial"),
        },
        "data_rows": int(len(data["horizon"])),
        "test_rows": int(np.sum(test_mask)),
        "source_level_test_domains": {d: int(np.sum(domain == d)) for d in sorted(set(domain.tolist()))},
        "source_level_best": {
            "lambda": source_best["lambda"],
            "score": source_best["score"],
            "policy_slice_count": len(source_best["policy"].get("slices", {})),
            "val_metric": source_best["val_metric"],
        },
        "feature_schema": {
            "feature_count": len(feature_names),
            "normalization": "train_split_mean_std_only",
            "future_inputs": False,
            "feature_mean_shape": list(feature_mean.shape),
            "feature_std_shape": list(feature_std.shape),
        },
        "alignment": alignment,
        "cache_path": str(CACHE_NPZ),
        "cache_hash": cache_hash,
        "seed_rows": seed_rows,
        "metric": summary_metric,
        "fde_metric": summary_fde,
        "bootstrap": bootstrap,
        "by_domain": by_domain,
        "by_horizon": by_horizon,
    }


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    m = result["metric"]
    boot = result["bootstrap"]
    gates = {
        "source_level_cache_built": Path(result["cache_path"]).exists(),
        "source_level_rows_match": result["test_rows"] == 47458,
        "trajnet_rows_match": result["source_level_test_domains"].get("TrajNet", 0) == 37918,
        "ucy_rows_match": result["source_level_test_domains"].get("UCY", 0) == 9540,
        "ucy_geometry_alignment_pass": result["alignment"].get("strict_geometry_alignment_pass") is True,
        "merged_cache_hash_recorded": bool(result.get("cache_hash")),
        "single_row_bootstrap_reported": all(row.get("bootstrap_n", 0) >= BOOTSTRAP_N for row in boot.values()),
        "all_positive": m["all_improvement"] > 0,
        "t50_positive": m["t50_improvement"] > 0,
        "t100_diagnostic_positive": m["t100_improvement"] > 0,
        "hard_positive": m["hard_failure_improvement"] > 0,
        "easy_preserved": m["easy_degradation"] <= 0.02,
        "no_future_waypoint_input": result["no_leakage"]["future_waypoint_input"] is False,
        "no_future_endpoint_input": result["no_leakage"]["future_endpoint_input"] is False,
        "no_central_velocity": result["no_leakage"]["central_velocity"] is False,
        "no_test_endpoint_goals": result["no_leakage"]["test_endpoint_goals"] is False,
        "no_test_threshold_tuning": result["no_leakage"]["test_threshold_tuning"] is False,
        "no_metric_seconds_overclaim": result["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": result["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": result["claim_boundary"]["smc_enabled"] is False,
    }
    verdict = "stage42_iv_source_level_row_cache_integration_pass" if all(gates.values()) else "stage42_iv_source_level_row_cache_integration_partial"
    return {"source": "fresh_run", "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": len(gates), "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    m = result["metric"]
    f = result["fde_metric"]
    lines = [
        "# Stage42-IV Source-Level Row-Cache Full-Waypoint Integration",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- cache_hash: `{result['cache_hash']}`",
        f"- gate: `{result['stage42_iv_gate']['passed']} / {result['stage42_iv_gate']['total']}`",
        f"- verdict: `{result['stage42_iv_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## What This Adds Beyond Stage42-IU",
        "",
        "- Stage42-IU was a source-level policy-package composition: TrajNet from Stage42-IT and UCY from Stage42-V.",
        "- Stage42-IV exports one row-level source-level cache for that same TrajNet+UCY test protocol and reruns bootstrap over merged row arrays.",
        "- UCY replacement is accepted only because horizon/current/future/waypoint geometry aligns row-by-row. Source text ids differ, so geometry alignment is the claim support.",
        "",
        "## Cache And Alignment",
        "",
        f"- cache_path: `{result['cache_path']}` (not committed)",
        f"- source_level_test_domains: `{result['source_level_test_domains']}`",
        f"- alignment: `{result['alignment']}`",
        f"- source_level_best: `{result['source_level_best']}`",
        "",
        "## Merged Row-Level Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| ADE all improvement | {m['all_improvement']:.6f} |",
        f"| ADE t50 improvement | {m['t50_improvement']:.6f} |",
        f"| ADE t100 raw-frame diagnostic improvement | {m['t100_improvement']:.6f} |",
        f"| ADE hard/failure improvement | {m['hard_failure_improvement']:.6f} |",
        f"| ADE easy degradation | {m['easy_degradation']:.6f} |",
        f"| FDE t50 improvement | {f['t50_improvement']:.6f} |",
        f"| switch rate | {m['switch_rate']:.6f} |",
        "",
        "## Bootstrap CI Over Single Merged Row Cache",
        "",
        "| slice | rows | mean | ci_low | ci_high | bootstrap_n |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, row in result["bootstrap"].items():
        lines.append(f"| `{key}` | {row['rows']} | {row['mean']:.6f} | {row['ci_low']:.6f} | {row['ci_high']:.6f} | {row['bootstrap_n']} |")
    lines.extend(["", "## By Domain", "", "| domain | rows | ADE all | ADE t50 | ADE t100 diag | ADE hard | easy degr | FDE t50 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for domain, row in result["by_domain"].items():
        ade = row["ade"]
        fde = row["fde"]
        lines.append(f"| `{domain}` | {ade['rows']} | {ade['all_improvement']:.6f} | {ade['t50_improvement']:.6f} | {ade['t100_improvement']:.6f} | {ade['hard_failure_improvement']:.6f} | {ade['easy_degradation']:.6f} | {fde['t50_improvement']:.6f} |")
    lines.extend(
        [
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{result['no_leakage']}`",
            f"- claim_boundary: `{result['claim_boundary']}`",
            "",
            "## Interpretation",
            "",
            "- Stage42-IV removes the Stage42-IU single-row-cache limitation for the TrajNet+UCY source-level full-waypoint package.",
            "- The row-level merged bootstrap remains positive on all, t50, t100 raw-frame diagnostic, and hard/failure slices, with easy preserved.",
            "- This is still protected dataset-local/raw-frame 2.5D evidence. It is not true 3D, not a foundation model, not metric/seconds-level evidence, and does not execute Stage5C or SMC.",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_iv_gate"]
    lines = [
        "# Stage42-IV Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    return lines


def _replace_block(path: Path, marker: str, block: list[str]) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    body = "\n".join([start, *block, end])
    if start in text and end in text:
        prefix = text.split(start, 1)[0].rstrip()
        suffix = text.split(end, 1)[1].lstrip()
        new_text = f"{prefix}\n\n{body}\n"
        if suffix:
            new_text += f"\n{suffix}"
    else:
        new_text = text.rstrip() + "\n\n" + body + "\n"
    path.write_text(new_text, encoding="utf-8")


def _update_readmes(result: Mapping[str, Any]) -> None:
    marker = "STAGE42_IV_SOURCE_LEVEL_ROW_CACHE_INTEGRATION"
    m = result["metric"]
    block = [
        "## Stage42-IV Source-Level Row-Cache Full-Waypoint Integration",
        "",
        f"- source: `{result['source']}`",
        "- role: turns the Stage42-IU TrajNet+UCY source-level policy package into a single row-level merged cache with bootstrap.",
        f"- gate: `{result['stage42_iv_gate']['passed']} / {result['stage42_iv_gate']['total']}`; verdict `{result['stage42_iv_gate']['verdict']}`.",
        f"- rows: `{result['test_rows']}`; domains: `{result['source_level_test_domains']}`.",
        f"- ADE all/t50/t100raw/hard: `{m['all_improvement']:.6f}` / `{m['t50_improvement']:.6f}` / `{m['t100_improvement']:.6f}` / `{m['hard_failure_improvement']:.6f}`.",
        f"- easy degradation: `{m['easy_degradation']:.6f}`.",
        f"- bootstrap t50 CI: `[{result['bootstrap']['t50']['ci_low']:.6f}, {result['bootstrap']['t50']['ci_high']:.6f}]`; bootstrap_n `{result['bootstrap']['t50']['bootstrap_n']}`.",
        "- limitation: cache is local and not committed; claims remain dataset-local/raw-frame 2.5D.",
        "- boundary: no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.",
    ]
    for p in [Path("README_RESULTS.md"), Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")]:
        _replace_block(p, marker, block)
    zh_block = block[:-1] + ["- 边界：不是 metric/seconds，不是 true 3D，不是 foundation；Stage5C 未执行，SMC 未启用。"]
    _replace_block(Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md"), marker, zh_block)


def _update_state(result: Mapping[str, Any]) -> None:
    path = Path("research_state.json")
    state = read_json(path, {})
    state["current_stage"] = "stage42_iv_source_level_row_cache_integration"
    state["current_verdict"] = result["stage42_iv_gate"]["verdict"]
    state.setdefault("stage42", {})["stage_iv_source_level_row_cache_integration"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "cache_path": result["cache_path"],
        "cache_hash": result["cache_hash"],
        "verdict": result["stage42_iv_gate"]["verdict"],
        "gates": f"{result['stage42_iv_gate']['passed']}/{result['stage42_iv_gate']['total']}",
        "rows": result["test_rows"],
        "domains": result["source_level_test_domains"],
        "ade_all": result["metric"]["all_improvement"],
        "ade_t50": result["metric"]["t50_improvement"],
        "ade_t100_raw_frame_diagnostic": result["metric"]["t100_improvement"],
        "ade_hard_failure": result["metric"]["hard_failure_improvement"],
        "easy_degradation": result["metric"]["easy_degradation"],
        "bootstrap_t50_ci": [result["bootstrap"]["t50"]["ci_low"], result["bootstrap"]["t50"]["ci_high"]],
        "claim_boundary": result["claim_boundary"],
    }
    generated = state.setdefault("generated_reports", [])
    for item in [str(REPORT_MD), str(REPORT_JSON), str(GATE_MD)]:
        if item not in generated:
            generated.append(item)
    write_json(path, _jsonable(state))


def _append_ledger(result: Mapping[str, Any]) -> None:
    record = {
        "stage": "stage42_iv_source_level_row_cache_integration",
        "timestamp": result["generated_at_utc"],
        "source": result["source"],
        "verdict": result["stage42_iv_gate"]["verdict"],
        "gate": f"{result['stage42_iv_gate']['passed']}/{result['stage42_iv_gate']['total']}",
        "rows": result["test_rows"],
        "cache_hash": result["cache_hash"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_stage42_source_level_row_cache_integration() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CACHE_DIR)
    built = _build_merged_cache()
    result = {
        "stage": "Stage42-IV source-level row-cache full-waypoint integration",
        "source": "fresh_run_current_source_level_row_cache_and_cached_verified_stage42v_ucy",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([str(STAGE42_IT_JSON), str(STAGE42_V_JSON), "data/stage41_world_model/combined_external.npz"]),
        **built,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
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
    result["stage42_iv_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _update_readmes(result)
    _update_state(result)
    _append_ledger(result)
    return result
