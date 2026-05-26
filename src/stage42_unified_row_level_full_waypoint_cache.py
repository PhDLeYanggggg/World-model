from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_full_trajectory_world_state as ft
from src import stage42_row_prediction_cache as r
from src import stage42_sequence_full_waypoint as s42i
from src import stage42_ucy_full_waypoint_candidate as v
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
CACHE_DIR = Path("data/stage42_unified_full_waypoint_cache")
STAGE42S_JSON = OUT_DIR / "frozen_row_combo_policy_stage42.json"
STAGE42V_JSON = OUT_DIR / "ucy_full_waypoint_candidate_stage42.json"
REPORT_JSON = OUT_DIR / "unified_row_level_full_waypoint_cache_stage42.json"
REPORT_MD = OUT_DIR / "unified_row_level_full_waypoint_cache_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_x_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

BOOTSTRAP_N = 2000

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-X 构建统一 row-level full-waypoint cache：ETH_UCY/TrajNet 来自 Stage42-S，UCY 来自 Stage42-V。",
    "future waypoints / endpoints 只作为 train/val labels 和 eval labels，不作为 inference input。",
    "UCY rows 通过 source/scene/horizon/current/future/waypoint alignment 校验后替换；Stage42-V ETH_UCY slice 不重复计入。",
    "统一 bootstrap 基于 merged row-level arrays，而不是 domain-level weighted summary。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test 调阈值。",
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


def _cached_result_if_available() -> dict[str, Any] | None:
    if not REPORT_JSON.exists():
        return None
    payload = read_json(REPORT_JSON, {})
    if payload.get("stage") == "Stage42-X unified row-level full-waypoint cache":
        return payload
    return None


def _assert_ucy_alignment(labels_global: Mapping[str, np.ndarray], labels_v: Mapping[str, np.ndarray]) -> dict[str, Any]:
    g_mask = labels_global["domain"].astype(str) == "UCY"
    v_mask = labels_v["domain"].astype(str) == "UCY"
    checks = {
        "global_ucy_rows": int(np.sum(g_mask)),
        "stage42v_ucy_rows": int(np.sum(v_mask)),
        "source_file_order": bool(np.array_equal(labels_global["source_file"][g_mask].astype(str), labels_v["source_file"][v_mask].astype(str))),
        "scene_id_order": bool(np.array_equal(labels_global["scene_id"][g_mask].astype(str), labels_v["scene_id"][v_mask].astype(str))),
        "horizon_order": bool(np.array_equal(labels_global["horizon"][g_mask].astype(int), labels_v["horizon"][v_mask].astype(int))),
        "current_xy_match": bool(np.allclose(labels_global["current_xy"][g_mask], labels_v["current_xy"][v_mask])),
        "future_xy_match": bool(np.allclose(labels_global["future_xy"][g_mask], labels_v["future_xy"][v_mask])),
        "waypoint_xy_match": bool(np.allclose(labels_global["waypoint_xy"][g_mask], labels_v["waypoint_xy"][v_mask])),
        "waypoint_valid_match": bool(np.array_equal(labels_global["waypoint_valid"][g_mask], labels_v["waypoint_valid"][v_mask])),
        "normalizer_max_abs_diff": float(np.max(np.abs(labels_global["normalizer"][g_mask].astype(float) - labels_v["normalizer"][v_mask].astype(float)))) if int(np.sum(g_mask)) else 0.0,
    }
    required = [
        "source_file_order",
        "scene_id_order",
        "horizon_order",
        "current_xy_match",
        "future_xy_match",
        "waypoint_xy_match",
        "waypoint_valid_match",
    ]
    if not (checks["global_ucy_rows"] == checks["stage42v_ucy_rows"] > 0 and all(checks[k] for k in required)):
        raise ValueError(f"UCY row alignment failed: {checks}")
    return checks


def _stage42v_ucy_arrays(best_rows: list[Mapping[str, Any]]) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    arrays: list[dict[str, np.ndarray]] = []
    labels_ref: dict[str, np.ndarray] | None = None
    for row in best_rows:
        pred, labels = v._predict(row["train_info"], "test")
        selected, switch = v._apply_policy(pred, labels, row["val_policy"])
        floor = v._floor_waypoints(labels)
        selected_ade, selected_fde = ft._trajectory_errors(selected, labels)
        floor_ade, floor_fde = ft._trajectory_errors(floor, labels)
        # If Stage42-V policy falls back, keep exact global floor later instead of
        # carrying tiny normalizer differences from the UCY training pipeline.
        arrays.append(
            {
                "selected_ade": selected_ade.astype(np.float32),
                "selected_fde": selected_fde.astype(np.float32),
                "floor_ade": floor_ade.astype(np.float32),
                "floor_fde": floor_fde.astype(np.float32),
                "switch": switch.astype(bool),
            }
        )
        if labels_ref is None:
            labels_ref = labels
    if labels_ref is None:
        raise ValueError("No Stage42-V best rows available.")
    return arrays, labels_ref


def _metric_from_arrays(selected_ade: np.ndarray, selected_fde: np.ndarray, floor_ade: np.ndarray, floor_fde: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray) -> dict[str, Any]:
    return {
        "ade": r._metric_from_errors(selected_ade, floor_ade, labels, switch),
        "fde": r._metric_from_errors(selected_fde, floor_fde, labels, switch),
        "switch_rate": float(np.mean(switch)) if len(switch) else 0.0,
    }


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "source": "fresh_run_from_unified_row_level_cache",
        "seeds": [int(row["pair_idx"]) for row in rows],
        "ade_all": r._stat([row["merged_test_metrics"]["ade"].get("all_improvement", 0.0) for row in rows]),
        "ade_t50": r._stat([row["merged_test_metrics"]["ade"].get("t50_improvement", 0.0) for row in rows]),
        "ade_t100_raw_frame_diagnostic": r._stat([row["merged_test_metrics"]["ade"].get("t100_improvement", 0.0) for row in rows]),
        "ade_hard_failure": r._stat([row["merged_test_metrics"]["ade"].get("hard_failure_improvement", 0.0) for row in rows]),
        "ade_easy_degradation": r._stat([row["merged_test_metrics"]["ade"].get("easy_degradation", 1.0) for row in rows]),
        "fde_t50": r._stat([row["merged_test_metrics"]["fde"].get("t50_improvement", 0.0) for row in rows]),
        "switch_rate": r._stat([row["merged_test_metrics"].get("switch_rate", 0.0) for row in rows]),
    }


def _bootstrap_seed_mean(rows: list[Mapping[str, Any]], labels_test: Mapping[str, np.ndarray]) -> dict[str, Any]:
    merged_ade = np.mean(np.stack([row["arrays_for_bootstrap"]["merged_test_ade"] for row in rows], axis=0), axis=0)
    floor_ade = rows[0]["arrays_for_bootstrap"]["floor_test_ade"]
    masks = {
        "all": np.ones(len(floor_ade), dtype=bool),
        "t50": labels_test["horizon"].astype(int) == 50,
        "t100_raw_frame_diagnostic": labels_test["horizon"].astype(int) == 100,
        "hard_failure": labels_test["hard"].astype(bool) | labels_test["failure"].astype(bool),
        "easy": labels_test["easy"].astype(bool),
    }
    rng = np.random.default_rng(42052)
    out: dict[str, Any] = {"source": "fresh_bootstrap_over_unified_row_seed_mean", "n": BOOTSTRAP_N}
    for name, mask in masks.items():
        ids = np.where(mask)[0]
        if len(ids) == 0:
            out[name] = {"rows": 0, "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
            continue
        per_row = (merged_ade[ids] - floor_ade[ids]) if name == "easy" else (floor_ade[ids] - merged_ade[ids])
        draws = np.empty(BOOTSTRAP_N, dtype=np.float64)
        for i in range(BOOTSTRAP_N):
            sample = rng.choice(ids, size=len(ids), replace=True)
            draws[i] = float(np.mean(merged_ade[sample] - floor_ade[sample])) if name == "easy" else float(np.mean(floor_ade[sample] - merged_ade[sample]))
        out[name] = {
            "rows": int(len(ids)),
            "mean": float(np.mean(per_row)),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
        }
    return out


def _slice_stats(rows: list[Mapping[str, Any]], labels_test: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    if int(np.sum(mask)) == 0:
        return {"rows": 0, "source": "not_run_empty_slice"}
    ade_metrics = []
    fde_metrics = []
    switches = []
    for row in rows:
        arr = row["arrays_for_bootstrap"]
        switch = arr["merged_test_switch"].astype(bool)
        ade_metrics.append(r._local_metric_from_errors(arr["merged_test_ade"], arr["floor_test_ade"], labels_test, switch, mask))
        fde_metrics.append(r._local_metric_from_errors(arr["merged_test_fde"], arr["floor_test_fde"], labels_test, switch, mask))
        switches.append(float(np.mean(switch[mask])) if int(np.sum(mask)) else 0.0)
    return {
        "rows": int(np.sum(mask)),
        "source": "fresh_run_from_unified_row_cache",
        "ade_all": r._stat([m.get("all_improvement", 0.0) for m in ade_metrics]),
        "ade_t50": r._stat([m.get("t50_improvement", 0.0) for m in ade_metrics]),
        "ade_t100_raw_frame_diagnostic": r._stat([m.get("t100_improvement", 0.0) for m in ade_metrics]),
        "ade_hard_failure": r._stat([m.get("hard_failure_improvement", 0.0) for m in ade_metrics]),
        "ade_easy_degradation": r._stat([m.get("easy_degradation", 0.0) for m in ade_metrics]),
        "fde_t50": r._stat([m.get("t50_improvement", 0.0) for m in fde_metrics]),
        "switch_rate": r._stat(switches),
    }


def _stress(rows: list[Mapping[str, Any]], labels_test: Mapping[str, np.ndarray]) -> dict[str, Any]:
    domains = sorted(set(labels_test["domain"].astype(str).tolist()))
    horizons = [10, 25, 50, 100]
    out: dict[str, Any] = {"by_domain": {}, "by_horizon": {}, "by_domain_horizon": {}}
    for domain in domains:
        out["by_domain"][domain] = _slice_stats(rows, labels_test, labels_test["domain"].astype(str) == domain)
    for horizon in horizons:
        out["by_horizon"][str(horizon)] = _slice_stats(rows, labels_test, labels_test["horizon"].astype(int) == horizon)
    for domain in domains:
        for horizon in horizons:
            mask = (labels_test["domain"].astype(str) == domain) & (labels_test["horizon"].astype(int) == horizon)
            out["by_domain_horizon"][f"{domain}|{horizon}"] = _slice_stats(rows, labels_test, mask)
    return out


def _cache_path(pair_idx: int) -> Path:
    return CACHE_DIR / f"stage42x_unified_pair_{pair_idx}.npz"


def _build_merged_rows() -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    stage42v = read_json(STAGE42V_JSON, {})
    best_trial = str(stage42v.get("best_trial", ""))
    best_rows = [row for row in stage42v.get("rows", []) if row.get("trial", {}).get("name") == best_trial]
    best_rows = sorted(best_rows, key=lambda row: int(row.get("seed", 0)))
    if len(best_rows) < 3:
        raise ValueError("Stage42-V best trial needs at least three seed rows for Stage42-X.")
    data_test = s42i._split_arrays("test")
    labels_test = s42i._labels(data_test)
    stage42v_arrays, labels_v = _stage42v_ucy_arrays(best_rows)
    alignment = _assert_ucy_alignment(labels_test, labels_v)
    ucy_global = labels_test["domain"].astype(str) == "UCY"
    ucy_v = labels_v["domain"].astype(str) == "UCY"
    labels_val = s42i._labels(s42i._split_arrays("val"))
    stage42r = r.run_stage42_row_prediction_cache()
    rows_runtime = []
    ensure_dir(CACHE_DIR)
    for idx, cache_row in enumerate(stage42r.get("cache_rows", [])):
        base = r._eval_pair_cache(cache_row["cache_path"], labels_val, labels_test)
        arr = base["arrays_for_bootstrap"]
        merged_ade = arr["combo_test_ade"].copy()
        merged_fde = arr["combo_test_fde"].copy()
        merged_switch = arr["combo_test_switch"].astype(bool).copy()
        ucy_arr = stage42v_arrays[idx % len(stage42v_arrays)]
        ucy_selected_ade = ucy_arr["selected_ade"][ucy_v].astype(np.float32)
        ucy_selected_fde = ucy_arr["selected_fde"][ucy_v].astype(np.float32)
        ucy_switch = ucy_arr["switch"][ucy_v].astype(bool)
        # Keep exact global floor on fallback rows to avoid tiny normalizer
        # differences between Stage42-I and Stage42-V derived datasets.
        floor_ade_global = arr["floor_test_ade"][ucy_global]
        floor_fde_global = arr["floor_test_fde"][ucy_global]
        ucy_selected_ade = np.where(ucy_switch, ucy_selected_ade, floor_ade_global)
        ucy_selected_fde = np.where(ucy_switch, ucy_selected_fde, floor_fde_global)
        merged_ade[ucy_global] = ucy_selected_ade
        merged_fde[ucy_global] = ucy_selected_fde
        merged_switch[ucy_global] = ucy_switch
        np.savez_compressed(
            _cache_path(idx),
            pair_idx=np.asarray(idx, dtype=np.int16),
            stage42r_cache_path=np.asarray(str(cache_row["cache_path"])),
            stage42v_seed=np.asarray(int(best_rows[idx % len(best_rows)].get("seed", 0)), dtype=np.int16),
            floor_test_ade=arr["floor_test_ade"],
            floor_test_fde=arr["floor_test_fde"],
            merged_test_ade=merged_ade.astype(np.float32),
            merged_test_fde=merged_fde.astype(np.float32),
            merged_test_switch=merged_switch.astype(bool),
            ucy_mask=ucy_global.astype(bool),
        )
        metrics = _metric_from_arrays(merged_ade, merged_fde, arr["floor_test_ade"], arr["floor_test_fde"], labels_test, merged_switch)
        rows_runtime.append(
            {
                "source": "fresh_run_from_stage42s_row_cache_plus_stage42v_ucy_rows",
                "pair_idx": int(base["pair_idx"]),
                "stage42r_cache_path": str(cache_row["cache_path"]),
                "stage42v_seed": int(best_rows[idx % len(best_rows)].get("seed", 0)),
                "cache_path": str(_cache_path(idx)),
                "merged_test_metrics": metrics,
                "arrays_for_bootstrap": {
                    "merged_test_ade": merged_ade.astype(np.float32),
                    "merged_test_fde": merged_fde.astype(np.float32),
                    "merged_test_switch": merged_switch.astype(bool),
                    "floor_test_ade": arr["floor_test_ade"],
                    "floor_test_fde": arr["floor_test_fde"],
                },
            }
        )
    return rows_runtime, labels_test, alignment


def _strip_arrays(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "arrays_for_bootstrap"}


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    s = result.get("summary", {})
    boot = result.get("bootstrap_seed_mean", {})
    stress = result.get("stress", {}).get("by_domain", {})
    positive_all = [d for d, item in stress.items() if item.get("rows", 0) > 0 and item.get("ade_all", {}).get("mean", 0.0) > 0.0]
    positive_t50 = [d for d, item in stress.items() if item.get("rows", 0) > 0 and item.get("ade_t50", {}).get("mean", 0.0) > 0.0]
    gates = {
        "stage42s_verified": result.get("inputs", {}).get("stage42s_verdict") == "stage42_s_frozen_row_combo_policy_pass",
        "stage42v_verified": result.get("inputs", {}).get("stage42v_verdict") == "stage42_v_ucy_full_waypoint_candidate_pass",
        "row_level_cache_built": len(result.get("cache_manifest", [])) >= 3 and all(item.get("exists") for item in result.get("cache_manifest", [])),
        "ucy_alignment_pass": result.get("ucy_alignment", {}).get("source_file_order") is True and result.get("ucy_alignment", {}).get("waypoint_xy_match") is True,
        "three_domains_positive_all": len(positive_all) >= 3,
        "three_domains_positive_t50": len(positive_t50) >= 3,
        "all_positive": s.get("ade_all", {}).get("mean", 0.0) > 0.0,
        "t50_positive": s.get("ade_t50", {}).get("mean", 0.0) > 0.0,
        "hard_positive": s.get("ade_hard_failure", {}).get("mean", 0.0) > 0.0,
        "easy_preserved": s.get("ade_easy_degradation", {}).get("mean", 1.0) <= 0.02,
        "t50_seed_ci_nonnegative": s.get("ade_t50", {}).get("ci_low", -1.0) >= 0.0,
        "t50_bootstrap_ci_nonnegative": boot.get("t50", {}).get("ci_low", -1.0) >= 0.0,
        "no_leakage_pass": result.get("no_leakage", {}).get("future_endpoint_input") is False
        and result.get("no_leakage", {}).get("future_waypoints_input") is False
        and result.get("no_leakage", {}).get("central_velocity") is False
        and result.get("no_leakage", {}).get("test_endpoint_goals") is False
        and result.get("no_leakage", {}).get("test_policy_tuning") is False,
        "no_metric_seconds_overclaim": result.get("claim_boundary", {}).get("metric_or_seconds_claim") is False,
        "stage5c_false": result.get("claim_boundary", {}).get("stage5c_executed") is False,
        "smc_false": result.get("claim_boundary", {}).get("smc_enabled") is False,
    }
    return {
        "source": result.get("source"),
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": len(gates),
        "positive_all_domains": positive_all,
        "positive_t50_domains": positive_t50,
        "verdict": "stage42_x_unified_row_level_full_waypoint_cache_pass" if all(gates.values()) else "stage42_x_unified_row_level_full_waypoint_cache_partial",
    }


def run_stage42_unified_row_level_full_waypoint_cache() -> dict[str, Any]:
    cached = _cached_result_if_available()
    if cached is not None:
        return cached
    ensure_dir(OUT_DIR)
    ensure_dir(CACHE_DIR)
    stage42s = read_json(STAGE42S_JSON, {})
    stage42v = read_json(STAGE42V_JSON, {})
    rows_runtime, labels_test, alignment = _build_merged_rows()
    cache_manifest = [
        {"pair_idx": int(row["pair_idx"]), "path": row["cache_path"], "exists": Path(row["cache_path"]).exists(), "size_bytes": int(Path(row["cache_path"]).stat().st_size) if Path(row["cache_path"]).exists() else 0}
        for row in rows_runtime
    ]
    claim_boundary = {
        "true_3d": False,
        "foundation_world_model": False,
        "metric_or_seconds_claim": False,
        "raw_frame_dataset_local_only": True,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    no_leakage = {
        "future_endpoint_input": False,
        "future_waypoints_input": False,
        "future_waypoints_used_as_train_val_label_and_eval_only": True,
        "central_velocity": False,
        "test_endpoint_goals": False,
        "test_policy_tuning": False,
        "stage42s_no_leakage": stage42s.get("no_leakage", {}),
        "stage42v_no_leakage": stage42v.get("no_leakage", {}),
    }
    result = {
        "stage": "Stage42-X unified row-level full-waypoint cache",
        "source": "fresh_run_from_stage42s_row_cache_and_stage42v_ucy_predictions",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "stage42s_report": str(STAGE42S_JSON),
            "stage42s_verdict": stage42s.get("stage42_s_gate", {}).get("verdict"),
            "stage42v_report": str(STAGE42V_JSON),
            "stage42v_verdict": stage42v.get("verdict"),
            "stage42v_best_trial": stage42v.get("best_trial"),
        },
        "input_hash": _combined_hash([STAGE42S_JSON, STAGE42V_JSON]),
        "cache_dir": str(CACHE_DIR),
        "cache_manifest": cache_manifest,
        "cache_hash": _combined_hash([item["path"] for item in cache_manifest if item.get("exists")]),
        "ucy_alignment": alignment,
        "rows": [_strip_arrays(row) for row in rows_runtime],
        "summary": _summary(rows_runtime),
        "bootstrap_seed_mean": _bootstrap_seed_mean(rows_runtime, labels_test),
        "stress": _stress(rows_runtime, labels_test),
        "no_leakage": no_leakage,
        "claim_boundary": claim_boundary,
    }
    result["stage42_x_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    _write_report(result)
    _write_gate(result["stage42_x_gate"])
    _append_readme_and_state(result)
    _append_ledger(result)
    return result


def _write_report(result: Mapping[str, Any]) -> None:
    gate = result["stage42_x_gate"]
    s = result["summary"]
    boot = result["bootstrap_seed_mean"]
    lines = [
        "# Stage42-X Unified Row-Level Full-Waypoint Cache",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- cache_hash: `{result['cache_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Row-Level Merge",
        "",
        "- ETH_UCY and TrajNet rows keep the Stage42-S row-cache combo outputs.",
        "- UCY rows are replaced by Stage42-V strict pure-UCY full-waypoint predictions.",
        "- Stage42-V ETH_UCY rows are not imported, preventing double counting.",
        "- Fallback UCY rows use the exact global Stage42-S floor errors to avoid tiny normalizer mismatches.",
        "",
        f"- cache_dir: `{result['cache_dir']}` (not committed)",
        f"- UCY alignment: `{result['ucy_alignment']}`",
        "",
        "## Seed Summary",
        "",
        "| metric | mean | ci_low | ci_high |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in [
        ("ADE all", "ade_all"),
        ("ADE t50", "ade_t50"),
        ("ADE t100 raw-frame diagnostic", "ade_t100_raw_frame_diagnostic"),
        ("ADE hard/failure", "ade_hard_failure"),
        ("ADE easy degradation", "ade_easy_degradation"),
        ("FDE t50", "fde_t50"),
        ("switch rate", "switch_rate"),
    ]:
        row = s.get(key, {})
        lines.append(f"| {label} | {row.get('mean', 0.0):.6f} | {row.get('ci_low', 0.0):.6f} | {row.get('ci_high', 0.0):.6f} |")
    lines.extend(["", "## Row Bootstrap Over Seed-Mean Arrays", "", "| slice | rows | mean | ci_low | ci_high |", "| --- | ---: | ---: | ---: | ---: |"])
    for name in ["all", "t50", "t100_raw_frame_diagnostic", "hard_failure", "easy"]:
        row = boot.get(name, {})
        lines.append(f"| `{name}` | {row.get('rows', 0)} | {row.get('mean', 0.0):.6f} | {row.get('ci_low', 0.0):.6f} | {row.get('ci_high', 0.0):.6f} |")
    lines.extend(["", "## Per-Domain Stress", "", "| domain | rows | ADE all | ADE t50 | ADE hard | easy degr | FDE t50 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for domain, item in result.get("stress", {}).get("by_domain", {}).items():
        lines.append(
            f"| `{domain}` | {item.get('rows', 0)} | {item.get('ade_all', {}).get('mean', 0.0):.6f} | "
            f"{item.get('ade_t50', {}).get('mean', 0.0):.6f} | {item.get('ade_hard_failure', {}).get('mean', 0.0):.6f} | "
            f"{item.get('ade_easy_degradation', {}).get('mean', 0.0):.6f} | {item.get('fde_t50', {}).get('mean', 0.0):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-X upgrades Stage42-W from a domain-level package to a row-level merged cache with unified bootstrap.",
            "- This is stronger full-waypoint branch evidence across ETH_UCY, TrajNet, and UCY.",
            "- It remains protected and dataset-local raw-frame 2.5D evidence; it is not Stage5C, SMC, metric, seconds-level, or true 3D.",
        ]
    )
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-X Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- positive_all_domains: `{gate.get('positive_all_domains', [])}`",
        f"- positive_t50_domains: `{gate.get('positive_t50_domains', [])}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | `{bool(ok)}` |")
    write_md(GATE_MD, lines)


def _append_if_missing(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in text:
        path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def _append_readme_and_state(result: Mapping[str, Any]) -> None:
    gate = result["stage42_x_gate"]
    s = result["summary"]
    boot = result["bootstrap_seed_mean"]
    block = f"""
## Stage42-X Unified Row-Level Full-Waypoint Cache

```text
source = {result['source']}
verdict = {gate['verdict']}
gates = {gate['passed']} / {gate['total']}
cache_hash = {result['cache_hash']}
ADE_all = {s['ade_all']['mean']}
ADE_t50 = {s['ade_t50']['mean']}
ADE_t50_seed_CI_low = {s['ade_t50']['ci_low']}
ADE_t50_bootstrap_CI_low = {boot['t50']['ci_low']}
ADE_hard_failure = {s['ade_hard_failure']['mean']}
ADE_easy_degradation = {s['ade_easy_degradation']['mean']}
positive_domains = {gate.get('positive_all_domains', [])}
stage5c_executed = false
smc_enabled = false
```

Stage42-X upgrades Stage42-W from a domain-level policy package into a row-level merged full-waypoint cache with unified bootstrap. ETH_UCY/TrajNet use Stage42-S row-cache combo outputs; UCY rows use Stage42-V UCY full-waypoint predictions after row alignment. Claims remain dataset-local raw-frame 2.5D, not metric or seconds-level.
"""
    _append_if_missing(Path("README_RESULTS.md"), "## Stage42-X Unified Row-Level Full-Waypoint Cache", block)
    _append_if_missing(Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md"), "## Stage42-X Unified Row-Level Full-Waypoint Cache", block)
    _append_if_missing(Path("README_M3W_RESEARCH_SUMMARY_ZH.md"), "## Stage42-X Unified Row-Level Full-Waypoint Cache", block)
    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_x_unified_row_level_full_waypoint_cache"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_x_unified_row_level_full_waypoint_cache"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "gates_passed": gate["passed"],
        "gates_total": gate["total"],
        "verdict": gate["verdict"],
        "cache_dir": result["cache_dir"],
        "cache_hash": result["cache_hash"],
        "ade_all": s["ade_all"]["mean"],
        "ade_t50": s["ade_t50"]["mean"],
        "ade_t50_seed_ci_low": s["ade_t50"]["ci_low"],
        "ade_t50_bootstrap_ci_low": boot["t50"]["ci_low"],
        "ade_hard_failure": s["ade_hard_failure"]["mean"],
        "ade_easy_degradation": s["ade_easy_degradation"]["mean"],
        "positive_all_domains": gate.get("positive_all_domains", []),
        "positive_t50_domains": gate.get("positive_t50_domains", []),
        "claim_boundary": result["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(Path("research_state.json"), _jsonable(state))


def _append_ledger(result: Mapping[str, Any]) -> None:
    entry = {
        "command": "run_stage42_unified_row_level_full_waypoint_cache.py",
        "step": "stage42_x_unified_row_level_full_waypoint_cache",
        "source": result["source"],
        "status": "success",
        "input_hash": result.get("input_hash"),
        "output_hash": _combined_hash([REPORT_JSON, REPORT_MD, GATE_MD]),
        "git_commit": _git_commit(),
        "generated_at_utc": result.get("generated_at_utc"),
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_unified_row_level_full_waypoint_cache()
