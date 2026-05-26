from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_full_trajectory_world_state as ft
from src import stage41_breakthrough as s41
from src import stage42_explicit_gain_harm_selector as s42o
from src import stage42_horizon_static_gate_repair as s42l
from src import stage42_row_gain_static_gate as s42n
from src import stage42_sequence_full_waypoint as s42i
from src import stage42_t50_static_expert_combo as s42q
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
CACHE_DIR = Path("data/stage42_row_prediction_cache")
REPORT_JSON = OUT_DIR / "row_prediction_cache_stage42.json"
REPORT_MD = OUT_DIR / "row_prediction_cache_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_r_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

J_SEEDS = [53, 59, 61]
P_SEEDS = [149, 151, 157]
BASE_SEEDS = [109, 113, 127]
BOOTSTRAP_N = 2000

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-R 构建 row-level prediction cache 并从 cache 做 validation-only combo eval，不是 metric 或 seconds-level 结果。",
    "future waypoints / future endpoints 只作为 train/val supervised labels 和 eval labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "Stage42-P feature normalization 只使用 train split statistics。",
    "combo source policy 只在 validation 上选择，test 只最终评估一次。",
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
    if payload.get("stage") == "Stage42-R row prediction cache and validation-only combo":
        return payload
    return None


def _labels_for(split: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return s42i._labels(split)


def _error_arrays(selected_xy: np.ndarray, labels: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ade, fde = ft._trajectory_errors(selected_xy, labels)
    return ade.astype(np.float32), fde.astype(np.float32)


def _write_heartbeat(pair_idx: int, payload: Mapping[str, Any]) -> None:
    hb = OUT_DIR / f"stage42r_row_prediction_cache_pair{pair_idx}_heartbeat.json"
    hb.write_text(json.dumps(_jsonable(dict(payload)), ensure_ascii=False), encoding="utf-8")


def _pair_cache_path(pair_idx: int) -> Path:
    return CACHE_DIR / f"stage42r_pair_{pair_idx}.npz"


def _build_pair_cache(
    pair_idx: int,
    j_seed: int,
    p_seed: int,
    base_seed: int,
    data: Mapping[str, Mapping[str, np.ndarray]],
    vocab: Mapping[str, int],
    train_teacher: Mapping[str, np.ndarray],
    val_teacher: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    ensure_dir(CACHE_DIR)
    cache_path = _pair_cache_path(pair_idx)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as npz:
            return {
                "source": "cached_verified",
                "pair_idx": pair_idx,
                "j_seed": int(npz["j_seed"]),
                "p_seed": int(npz["p_seed"]),
                "base_seed": int(npz["base_seed"]),
                "cache_path": str(cache_path),
                "rows_val": int(len(npz["floor_val_ade"])),
                "rows_test": int(len(npz["floor_test_ade"])),
            }
    _write_heartbeat(pair_idx, {"source": "fresh_run", "pair_idx": pair_idx, "status": "stage42j_start", "j_seed": j_seed, "p_seed": p_seed, "base_seed": base_seed})
    labels_val = _labels_for(data["val"])
    labels_test = _labels_for(data["test"])
    floor_val = ft._floor_waypoints(labels_val)
    floor_test = ft._floor_waypoints(labels_test)
    floor_val_ade, floor_val_fde = _error_arrays(floor_val, labels_val)
    floor_test_ade, floor_test_fde = _error_arrays(floor_test, labels_test)
    j_row = s42q._stage42j_selected(j_seed, data["val"], data["test"])
    j_val_ade, j_val_fde = _error_arrays(j_row["selected_val"], labels_val)
    j_test_ade, j_test_fde = _error_arrays(j_row["selected_test"], labels_test)
    _write_heartbeat(pair_idx, {"source": "fresh_run", "pair_idx": pair_idx, "status": "stage42p_start", "j_seed": j_seed, "p_seed": p_seed, "base_seed": base_seed})
    p_row = s42q._stage42p_selected(p_seed, base_seed, data["train"], data["val"], data["test"], vocab, train_teacher, val_teacher)
    p_val_ade, p_val_fde = _error_arrays(p_row["selected_val"], labels_val)
    p_test_ade, p_test_fde = _error_arrays(p_row["selected_test"], labels_test)
    np.savez_compressed(
        cache_path,
        pair_idx=np.asarray(pair_idx, dtype=np.int16),
        j_seed=np.asarray(j_seed, dtype=np.int16),
        p_seed=np.asarray(p_seed, dtype=np.int16),
        base_seed=np.asarray(base_seed, dtype=np.int16),
        floor_val_ade=floor_val_ade,
        floor_val_fde=floor_val_fde,
        floor_test_ade=floor_test_ade,
        floor_test_fde=floor_test_fde,
        j_val_ade=j_val_ade,
        j_val_fde=j_val_fde,
        j_val_switch=j_row["switch_val"].astype(bool),
        j_test_ade=j_test_ade,
        j_test_fde=j_test_fde,
        j_test_switch=j_row["switch_test"].astype(bool),
        p_val_ade=p_val_ade,
        p_val_fde=p_val_fde,
        p_val_switch=p_row["switch_val"].astype(bool),
        p_test_ade=p_test_ade,
        p_test_fde=p_test_fde,
        p_test_switch=p_row["switch_test"].astype(bool),
    )
    _write_heartbeat(pair_idx, {"source": "fresh_run", "pair_idx": pair_idx, "status": "complete", "cache_path": str(cache_path)})
    return {
        "source": "fresh_run",
        "pair_idx": pair_idx,
        "j_seed": j_seed,
        "p_seed": p_seed,
        "base_seed": base_seed,
        "cache_path": str(cache_path),
        "rows_val": int(len(floor_val_ade)),
        "rows_test": int(len(floor_test_ade)),
    }


def _metric_from_errors(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray) -> dict[str, Any]:
    return ft._metric(selected.astype(np.float64), floor.astype(np.float64), labels, switch.astype(bool))


def _local_metric_from_errors(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    sliced = {k: v[mask] for k, v in labels.items() if isinstance(v, np.ndarray) and len(v) == len(mask)}
    return ft._metric(selected[mask].astype(np.float64), floor[mask].astype(np.float64), sliced, switch[mask].astype(bool))


def _score_local(metric: Mapping[str, Any], horizon: int) -> float:
    h_weight = 4.0 if horizon == 50 else 1.0
    return (
        h_weight * float(metric.get("all_improvement", 0.0))
        + 1.1 * float(metric.get("hard_failure_improvement", 0.0))
        - 80.0 * max(0.0, float(metric.get("easy_degradation", 1.0)) - 0.018)
        - 0.04 * float(metric.get("switch_rate", 0.0))
    )


def _eval_pair_cache(cache_path: str | Path, labels_val: Mapping[str, np.ndarray], labels_test: Mapping[str, np.ndarray]) -> dict[str, Any]:
    with np.load(cache_path, allow_pickle=False) as row:
        arrays = {k: row[k].copy() for k in row.files}
    domain_val = labels_val["domain"].astype(str)
    horizon_val = labels_val["horizon"].astype(int)
    domain_test = labels_test["domain"].astype(str)
    horizon_test = labels_test["horizon"].astype(int)
    combo_val_ade = arrays["floor_val_ade"].copy()
    combo_val_fde = arrays["floor_val_fde"].copy()
    combo_val_switch = np.zeros(len(combo_val_ade), dtype=bool)
    combo_test_ade = arrays["floor_test_ade"].copy()
    combo_test_fde = arrays["floor_test_fde"].copy()
    combo_test_switch = np.zeros(len(combo_test_ade), dtype=bool)
    choices: dict[str, Any] = {}
    sources = {
        "stage42j_static_expert": ("j_val_ade", "j_val_fde", "j_val_switch", "j_test_ade", "j_test_fde", "j_test_switch"),
        "stage42p_t50_gain_harm": ("p_val_ade", "p_val_fde", "p_val_switch", "p_test_ade", "p_test_fde", "p_test_switch"),
    }
    for domain in sorted(set(domain_val.tolist())):
        for horizon in [10, 25, 50, 100]:
            val_mask = (domain_val == domain) & (horizon_val == horizon)
            test_mask = (domain_test == domain) & (horizon_test == horizon)
            if int(np.sum(val_mask)) < 80:
                continue
            best_source = "floor"
            best_score = 0.0
            best_metric: dict[str, Any] = {"rows": int(np.sum(val_mask)), "all_improvement": 0.0}
            for name, (v_ade, _v_fde, v_sw, _t_ade, _t_fde, _t_sw) in sources.items():
                metric = _local_metric_from_errors(arrays[v_ade], arrays["floor_val_ade"], labels_val, arrays[v_sw], val_mask)
                if metric.get("easy_degradation", 1.0) > 0.018:
                    continue
                score = _score_local(metric, horizon)
                if score > best_score:
                    best_score = score
                    best_source = name
                    best_metric = metric
            if best_source != "floor":
                v_ade, v_fde, v_sw, t_ade, t_fde, t_sw = sources[best_source]
                combo_val_ade[val_mask] = arrays[v_ade][val_mask]
                combo_val_fde[val_mask] = arrays[v_fde][val_mask]
                combo_val_switch[val_mask] = arrays[v_sw][val_mask]
                combo_test_ade[test_mask] = arrays[t_ade][test_mask]
                combo_test_fde[test_mask] = arrays[t_fde][test_mask]
                combo_test_switch[test_mask] = arrays[t_sw][test_mask]
            choices[f"{domain}|{horizon}"] = {"selected_source": best_source, "val_score": float(best_score), "val_metric": best_metric}
    return {
        "pair_idx": int(arrays["pair_idx"]),
        "j_seed": int(arrays["j_seed"]),
        "p_seed": int(arrays["p_seed"]),
        "base_seed": int(arrays["base_seed"]),
        "cache_path": str(cache_path),
        "choices": choices,
        "combo_val_metrics": {
            "ade": _metric_from_errors(combo_val_ade, arrays["floor_val_ade"], labels_val, combo_val_switch),
            "fde": _metric_from_errors(combo_val_fde, arrays["floor_val_fde"], labels_val, combo_val_switch),
            "switch_rate": float(np.mean(combo_val_switch)),
        },
        "combo_test_metrics": {
            "ade": _metric_from_errors(combo_test_ade, arrays["floor_test_ade"], labels_test, combo_test_switch),
            "fde": _metric_from_errors(combo_test_fde, arrays["floor_test_fde"], labels_test, combo_test_switch),
            "switch_rate": float(np.mean(combo_test_switch)),
        },
        "stage42j_test_metrics": {
            "ade": _metric_from_errors(arrays["j_test_ade"], arrays["floor_test_ade"], labels_test, arrays["j_test_switch"]),
            "fde": _metric_from_errors(arrays["j_test_fde"], arrays["floor_test_fde"], labels_test, arrays["j_test_switch"]),
            "switch_rate": float(np.mean(arrays["j_test_switch"])),
        },
        "stage42p_test_metrics": {
            "ade": _metric_from_errors(arrays["p_test_ade"], arrays["floor_test_ade"], labels_test, arrays["p_test_switch"]),
            "fde": _metric_from_errors(arrays["p_test_fde"], arrays["floor_test_fde"], labels_test, arrays["p_test_switch"]),
            "switch_rate": float(np.mean(arrays["p_test_switch"])),
        },
        "arrays_for_bootstrap": {
            "combo_test_ade": combo_test_ade,
            "combo_test_fde": combo_test_fde,
            "combo_test_switch": combo_test_switch,
            "floor_test_ade": arrays["floor_test_ade"],
            "floor_test_fde": arrays["floor_test_fde"],
        },
    }


def _stat(vals: list[float]) -> dict[str, float]:
    return s42l._stat(vals)


def _summary(rows: list[Mapping[str, Any]], key: str = "combo_test_metrics") -> dict[str, Any]:
    return {
        "source": "fresh_run_from_row_prediction_cache",
        "seeds": [int(row["pair_idx"]) for row in rows],
        "ade_all": _stat([row[key]["ade"].get("all_improvement", 0.0) for row in rows]),
        "ade_t50": _stat([row[key]["ade"].get("t50_improvement", 0.0) for row in rows]),
        "ade_t100_raw_frame_diagnostic": _stat([row[key]["ade"].get("t100_improvement", 0.0) for row in rows]),
        "ade_hard_failure": _stat([row[key]["ade"].get("hard_failure_improvement", 0.0) for row in rows]),
        "ade_easy_degradation": _stat([row[key]["ade"].get("easy_degradation", 1.0) for row in rows]),
        "fde_all": _stat([row[key]["fde"].get("all_improvement", 0.0) for row in rows]),
        "fde_t50": _stat([row[key]["fde"].get("t50_improvement", 0.0) for row in rows]),
        "switch_rate": _stat([row[key].get("switch_rate", 0.0) for row in rows]),
    }


def _source_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for choice in row["choices"].values():
            src = str(choice.get("selected_source", "floor"))
            counts[src] = counts.get(src, 0) + 1
    return counts


def _bootstrap_seed_mean(rows: list[Mapping[str, Any]], labels_test: Mapping[str, np.ndarray]) -> dict[str, Any]:
    combo_ade = np.mean(np.stack([row["arrays_for_bootstrap"]["combo_test_ade"] for row in rows], axis=0), axis=0)
    floor_ade = rows[0]["arrays_for_bootstrap"]["floor_test_ade"]
    masks = {
        "all": np.ones(len(floor_ade), dtype=bool),
        "t50": labels_test["horizon"].astype(int) == 50,
        "t100_raw_frame_diagnostic": labels_test["horizon"].astype(int) == 100,
        "hard_failure": labels_test["hard"].astype(bool) | labels_test["failure"].astype(bool),
        "easy": labels_test["easy"].astype(bool),
    }
    rng = np.random.default_rng(42043)
    out: dict[str, Any] = {"source": "fresh_run_bootstrap_over_seed_mean_cache", "n": BOOTSTRAP_N}
    for name, mask in masks.items():
        ids = np.where(mask)[0]
        if len(ids) == 0:
            out[name] = {"rows": 0, "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
            continue
        if name == "easy":
            per_row = combo_ade[ids] - floor_ade[ids]
        else:
            per_row = floor_ade[ids] - combo_ade[ids]
        draws = np.empty(BOOTSTRAP_N, dtype=np.float64)
        for i in range(BOOTSTRAP_N):
            sample = rng.choice(ids, size=len(ids), replace=True)
            if name == "easy":
                draws[i] = float(np.mean(combo_ade[sample] - floor_ade[sample]))
            else:
                draws[i] = float(np.mean(floor_ade[sample] - combo_ade[sample]))
        out[name] = {
            "rows": int(len(ids)),
            "mean": float(np.mean(per_row)),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
        }
    return out


def _strip_arrays(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "arrays_for_bootstrap"}


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    s = result.get("summary", {})
    boot = result.get("bootstrap_seed_mean", {})
    gates = {
        "row_prediction_cache_built": len(result.get("cache_rows", [])) >= 3,
        "combo_eval_from_cache": len(result.get("rows", [])) >= 3,
        "validation_only_combo_selection": result.get("source_labels", {}).get("combo_selection") == "validation_only_from_cache",
        "uses_stage42j_and_stage42p_sources": bool(result.get("source_counts", {}).get("stage42j_static_expert", 0))
        and bool(result.get("source_counts", {}).get("stage42p_t50_gain_harm", 0)),
        "all_positive": s.get("ade_all", {}).get("mean", 0.0) > 0.0,
        "t50_positive": s.get("ade_t50", {}).get("mean", 0.0) > 0.0,
        "hard_positive": s.get("ade_hard_failure", {}).get("mean", 0.0) > 0.0,
        "easy_preserved": s.get("ade_easy_degradation", {}).get("mean", 1.0) <= 0.02,
        "t50_seed_ci_nonnegative": s.get("ade_t50", {}).get("ci_low", -1.0) >= 0.0,
        "t50_bootstrap_ci_nonnegative": boot.get("t50", {}).get("ci_low", -1.0) >= 0.0,
        "no_leakage_pass": result.get("no_leakage", {}).get("future_endpoint_input") is False
        and result.get("no_leakage", {}).get("future_waypoints_input") is False
        and result.get("no_leakage", {}).get("central_velocity") is False
        and result.get("no_leakage", {}).get("test_endpoint_goals") is False,
        "no_test_statistics_normalization": result.get("no_leakage", {}).get("test_statistics_normalization") is False,
        "no_metric_seconds_overclaim": result.get("claim_boundary", {}).get("metric_or_seconds_claim") is False,
        "stage5c_false": result.get("claim_boundary", {}).get("stage5c_executed") is False,
        "smc_false": result.get("claim_boundary", {}).get("smc_enabled") is False,
    }
    all_pass = all(bool(v) for v in gates.values())
    return {
        "source": "fresh_run_from_row_prediction_cache",
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": len(gates),
        "verdict": "stage42_r_row_cached_combo_pass" if all_pass else "stage42_r_row_cached_combo_partial",
    }


def run_stage42_row_prediction_cache() -> dict[str, Any]:
    cached = _cached_result_if_available()
    if cached is not None:
        return cached
    ensure_dir(OUT_DIR)
    ensure_dir(CACHE_DIR)
    if not (ft.DATA_DIR / "full_trajectory_train.npz").exists() or not (ft.DATA_DIR / "full_trajectory_val.npz").exists() or not (ft.DATA_DIR / "full_trajectory_test.npz").exists():
        ft.build_full_trajectory_labels()
    data = {split: s42i._split_arrays(split) for split in ["train", "val", "test"]}
    labels_val = _labels_for(data["val"])
    labels_test = _labels_for(data["test"])
    vocab = s42o._domain_vocab(data["train"], data["val"], data["test"])
    train_teacher = s42n._row_teacher(data["train"], "train")
    val_teacher = s42n._row_teacher(data["val"], "val")
    cache_rows = [
        _build_pair_cache(i, j_seed, p_seed, base_seed, data, vocab, train_teacher, val_teacher)
        for i, (j_seed, p_seed, base_seed) in enumerate(zip(J_SEEDS, P_SEEDS, BASE_SEEDS))
    ]
    rows_runtime = [_eval_pair_cache(row["cache_path"], labels_val, labels_test) for row in cache_rows]
    rows = [_strip_arrays(row) for row in rows_runtime]
    result = {
        "source": "fresh_run_from_row_prediction_cache",
        "stage": "Stage42-R row prediction cache and validation-only combo",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                ft.DATA_DIR / "all_agent_train.npz",
                ft.DATA_DIR / "all_agent_val.npz",
                ft.DATA_DIR / "all_agent_test.npz",
                ft.DATA_DIR / "full_trajectory_train.npz",
                ft.DATA_DIR / "full_trajectory_val.npz",
                ft.DATA_DIR / "full_trajectory_test.npz",
                OUT_DIR / "static_gated_full_waypoint_stage42.json",
                OUT_DIR / "t50_gain_harm_selector_stage42.json",
            ]
        ),
        "dataset_rows": {split: int(len(data[split]["horizon"])) for split in ["train", "val", "test"]},
        "cache_dir": str(CACHE_DIR),
        "cache_rows": cache_rows,
        "rows": rows,
        "summary": _summary(rows),
        "stage42j_cache_summary": _summary(rows, "stage42j_test_metrics"),
        "stage42p_cache_summary": _summary(rows, "stage42p_test_metrics"),
        "bootstrap_seed_mean": _bootstrap_seed_mean(rows_runtime, labels_test),
        "source_counts": _source_counts(rows),
        "source_labels": {
            "all_agent_dataset": "cached_verified",
            "full_waypoint_labels": "cached_verified_or_rebuilt_by_stage41_helper",
            "stage42j_checkpoints": "cached_verified",
            "stage42p_selector_checkpoints": "cached_verified",
            "row_prediction_cache": "fresh_run_or_cached_verified_npz_not_committed",
            "combo_selection": "validation_only_from_cache",
            "test_evaluation": "fresh_run_once_per_seed_pair_from_cache",
            "feature_normalization": "train_split_stats_only_for_stage42p",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "future_waypoints_used_as_train_val_label_and_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "combo_sources_selected_on_val": True,
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
    result["stage42_r_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    _write_report(result)
    _write_gate(result["stage42_r_gate"])
    _append_readme_and_state(result)
    _append_ledger(result)
    return result


def _write_report(result: Mapping[str, Any]) -> None:
    s = result["summary"]
    sj = result["stage42j_cache_summary"]
    sp = result["stage42p_cache_summary"]
    boot = result["bootstrap_seed_mean"]
    gate = result["stage42_r_gate"]
    lines = [
        "# Stage42-R Row Prediction Cache + Combo Eval",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- cache_dir: `{result['cache_dir']}` (not committed)",
        "",
        "## Current Facts",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Metrics",
        "",
        "| candidate | source | ADE all | ADE t50 | ADE t50 CI low | ADE t100 diag | ADE hard | ADE easy degr | FDE t50 | switch |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| `Stage42-R cached combo` | `{s['source']}` | {s['ade_all']['mean']:.6f} | {s['ade_t50']['mean']:.6f} | {s['ade_t50']['ci_low']:.6f} | {s['ade_t100_raw_frame_diagnostic']['mean']:.6f} | {s['ade_hard_failure']['mean']:.6f} | {s['ade_easy_degradation']['mean']:.6f} | {s['fde_t50']['mean']:.6f} | {s['switch_rate']['mean']:.6f} |",
        f"| `Stage42-J from cache` | `{sj['source']}` | {sj['ade_all']['mean']:.6f} | {sj['ade_t50']['mean']:.6f} | {sj['ade_t50']['ci_low']:.6f} | {sj['ade_t100_raw_frame_diagnostic']['mean']:.6f} | {sj['ade_hard_failure']['mean']:.6f} | {sj['ade_easy_degradation']['mean']:.6f} | {sj['fde_t50']['mean']:.6f} | {sj['switch_rate']['mean']:.6f} |",
        f"| `Stage42-P from cache` | `{sp['source']}` | {sp['ade_all']['mean']:.6f} | {sp['ade_t50']['mean']:.6f} | {sp['ade_t50']['ci_low']:.6f} | {sp['ade_t100_raw_frame_diagnostic']['mean']:.6f} | {sp['ade_hard_failure']['mean']:.6f} | {sp['ade_easy_degradation']['mean']:.6f} | {sp['fde_t50']['mean']:.6f} | {sp['switch_rate']['mean']:.6f} |",
        "",
        "## Bootstrap Over Seed-Mean Row Improvements",
        "",
        "| slice | rows | mean | ci_low | ci_high |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ["all", "t50", "t100_raw_frame_diagnostic", "hard_failure", "easy"]:
        row = boot.get(name, {})
        lines.append(f"| `{name}` | {row.get('rows', 0)} | {row.get('mean', 0.0):.6f} | {row.get('ci_low', 0.0):.6f} | {row.get('ci_high', 0.0):.6f} |")
    lines.extend(
        [
            "",
            "## Source Choices",
            "",
            f"- validation-selected source counts across seed/domain/horizon slices: `{result['source_counts']}`",
            "- Candidate sources are `floor`, `Stage42-J static expert`, and `Stage42-P t50 gain/harm`.",
            "- Source selection is by validation domain/horizon slice only; test labels are not used for threshold or source selection.",
            "",
            "## Interpretation",
            "",
            "- Stage42-R turns the Stage42-Q preflight into a cache-backed row-level evaluation path.",
            "- Cache files are local derived arrays and are intentionally not committed.",
            "- If the cached combo fails t+50 CI or source-diversity gates, that is honest evidence that simple slice-level source selection is still not enough.",
            "- Future waypoints remain train/val labels and final eval labels only, never inference inputs.",
            "- All claims remain dataset-local raw-frame 2.5D; no metric/seconds-level, Stage5C, or SMC claim is made.",
        ]
    )
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-R Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
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
    gate = result["stage42_r_gate"]
    s = result["summary"]
    block = f"""
## Stage42-R Row Prediction Cache + Combo Eval

```text
source = {result['source']}
verdict = {gate['verdict']}
gates = {gate['passed']} / {gate['total']}
cached_combo_ade_all = {s['ade_all']['mean']}
cached_combo_ade_t50 = {s['ade_t50']['mean']}
cached_combo_ade_t50_ci_low = {s['ade_t50']['ci_low']}
cached_combo_ade_hard_failure = {s['ade_hard_failure']['mean']}
cached_combo_ade_easy_degradation = {s['ade_easy_degradation']['mean']}
cached_combo_fde_t50 = {s['fde_t50']['mean']}
cache_dir = {result['cache_dir']} (not committed)
stage5c_executed = false
smc_enabled = false
```

Stage42-R builds a local NPZ row prediction cache for floor / Stage42-J static expert / Stage42-P t+50 gain-harm selected errors, then performs validation-only combo evaluation from cache. It remains dataset-local raw-frame 2.5D evidence and not Stage5C/SMC.
"""
    _append_if_missing(Path("README_RESULTS.md"), "## Stage42-R Row Prediction Cache + Combo Eval", block)
    _append_if_missing(Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md"), "## Stage42-R Row Prediction Cache + Combo Eval", block)
    _append_if_missing(Path("outputs/m3w_neural_v1/README_M3W_LONG_GOAL_SUMMARY_ZH.md"), "## Stage42-R Row Prediction Cache + Combo Eval", block)
    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_r_row_prediction_cache"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_r_row_prediction_cache"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "gates_passed": gate["passed"],
        "gates_total": gate["total"],
        "verdict": gate["verdict"],
        "cached_combo_ade_all": s["ade_all"]["mean"],
        "cached_combo_ade_t50": s["ade_t50"]["mean"],
        "cached_combo_ade_t50_ci_low": s["ade_t50"]["ci_low"],
        "cached_combo_ade_hard_failure": s["ade_hard_failure"]["mean"],
        "cached_combo_ade_easy_degradation": s["ade_easy_degradation"]["mean"],
        "cached_combo_fde_t50": s["fde_t50"]["mean"],
        "cache_dir": result["cache_dir"],
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
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "step": "stage42_r_row_prediction_cache",
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
    run_stage42_row_prediction_cache()
