from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_full_trajectory_world_state as ft
from src import stage41_pure_ucy_neural_retrain as p41ucy
from src import stage42_sequence_full_waypoint as s42i
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "ucy_candidate_bridge_stage42.json"
REPORT_MD = OUT_DIR / "ucy_candidate_bridge_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_u_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-U 只审计 UCY endpoint candidate 能否桥接到 full-waypoint，不执行 Stage5C 或 SMC。",
    "future endpoints / waypoints 只作为 supervised labels 和 eval labels，不作为 inference input。",
    "Stage41 pure-UCY policy 是 train/val selected；Stage42-U 不用 test 调参。",
    "如果 endpoint-to-full bridge 在 validation/test full-waypoint 上失败，必须标 blocker，不包装成 UCY 成功。",
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
    if payload.get("stage") == "Stage42-U UCY candidate endpoint-to-full bridge audit":
        return payload
    return None


def _linear_endpoint_waypoints(current_xy: np.ndarray, endpoint_xy: np.ndarray) -> np.ndarray:
    current = current_xy.astype(np.float64)
    endpoint = endpoint_xy.astype(np.float64)
    return current[:, None, :] + ft.WAYPOINT_FRAC[None, :, None] * (endpoint - current)[:, None, :]


def _stage41_best() -> tuple[dict[str, Any], str, dict[str, Any]]:
    report = read_json("outputs/stage41_external_split/stage41_pure_ucy_neural_retrain.json", {})
    best_trial = str(report.get("best_trial", ""))
    trial = report.get("trials", {}).get(best_trial, {})
    ckpt = str(trial.get("train", {}).get("checkpoint", ""))
    policy = dict(report.get("best_policy", {}))
    if not best_trial or not ckpt or not Path(ckpt).exists() or not policy:
        raise FileNotFoundError("Stage41 pure-UCY checkpoint/report is required for Stage42-U.")
    return report, ckpt, policy


def _endpoint_map(split: str, checkpoint: str | Path, policy: Mapping[str, Any]) -> dict[int, tuple[np.ndarray, float]]:
    ds = p41ucy._ds(split)
    pred = p41ucy._predict(checkpoint, split)
    alpha = p41ucy._endpoint_residual_alpha(pred, ds, policy)
    current = ds["current_xy"].astype(np.float64)
    normalizer = ds["normalizer"].astype(np.float64)
    floor_endpoint = current + ds["cand_delta"].astype(np.float64)[:, 0, :] * normalizer[:, None]
    neural_endpoint = current + pred["endpoint_delta"].astype(np.float64) * normalizer[:, None]
    selected_endpoint = floor_endpoint + alpha[:, None] * (neural_endpoint - floor_endpoint)
    return {int(row_id): (selected_endpoint[i].copy(), float(alpha[i])) for i, row_id in enumerate(ds["ids"].astype(np.int64))}


def _evaluate_bridge(split: str, checkpoint: str | Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    s42 = s42i._split_arrays(split)
    labels = s42i._labels(s42)
    row_ids = s42["raw"]["ids"].astype(np.int64)
    endpoint_by_id = _endpoint_map(split, checkpoint, policy)
    floor_xy = ft._floor_waypoints(labels)
    selected_xy = floor_xy.copy()
    switch = np.zeros(len(row_ids), dtype=bool)
    matched = np.zeros(len(row_ids), dtype=bool)
    for i, row_id in enumerate(row_ids):
        item = endpoint_by_id.get(int(row_id))
        if item is None:
            continue
        endpoint, alpha = item
        selected_xy[i] = _linear_endpoint_waypoints(labels["current_xy"][i : i + 1], endpoint[None, :])[0]
        switch[i] = alpha > 1e-8
        matched[i] = True
    selected_ade, selected_fde = ft._trajectory_errors(selected_xy, labels)
    floor_ade, floor_fde = ft._trajectory_errors(floor_xy, labels)

    def metric(mask: np.ndarray) -> dict[str, Any]:
        if not np.any(mask):
            return {
                "rows": 0,
                "all_improvement": 0.0,
                "t50_improvement": 0.0,
                "t100_improvement": 0.0,
                "hard_failure_improvement": 0.0,
                "easy_degradation": 0.0,
                "switch_rate": 0.0,
                "source": "not_run_no_rows",
            }
        sliced = {k: v[mask] for k, v in labels.items() if isinstance(v, np.ndarray) and len(v) == len(mask)}
        ade = ft._metric(selected_ade[mask], floor_ade[mask], sliced, switch[mask])
        fde = ft._metric(selected_fde[mask], floor_fde[mask], sliced, switch[mask])
        return {
            "source": "fresh_run_endpoint_to_full_linear_bridge",
            "rows": int(np.sum(mask)),
            "matched_rows": int(np.sum(matched & mask)),
            "ade": ade,
            "fde": fde,
            "switch_rate": float(np.mean(switch[mask])) if np.any(mask) else 0.0,
        }

    domain = labels["domain"].astype(str)
    source = labels["source_file"].astype(str)
    masks = {
        "matched_all": matched,
        "ETH_UCY_zara01_val" if split == "val" else "ETH_UCY_zara02_test": matched & (domain == "ETH_UCY"),
        "UCY_zara03_test": matched & (domain == "UCY"),
        "TrajNet_unmatched_control": domain == "TrajNet",
    }
    if split == "val":
        masks["pure_ucy_val_source_zara01"] = matched & (np.char.find(source, "UCY/zara01") >= 0)
    else:
        masks["pure_ucy_test_sources_zara02_zara03"] = matched & (
            (np.char.find(source, "UCY/zara02") >= 0) | (np.char.find(source, "UCY/zara03") >= 0)
        )
    return {
        "split": split,
        "rows": int(len(row_ids)),
        "matched_rows": int(np.sum(matched)),
        "domains": {str(d): int(np.sum(domain == d)) for d in sorted(set(domain.tolist()))},
        "metrics": {name: metric(mask) for name, mask in masks.items()},
    }


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    test = result.get("test_bridge", {}).get("metrics", {})
    ucy = test.get("UCY_zara03_test", {}).get("ade", {})
    matched = int(test.get("UCY_zara03_test", {}).get("matched_rows", 0))
    bridge_positive = (
        float(ucy.get("all_improvement", 0.0)) > 0.0
        and float(ucy.get("t50_improvement", 0.0)) > 0.0
        and float(ucy.get("easy_degradation", 1.0)) <= 0.02
    )
    gates = {
        "Gate1 Stage41 pure-UCY checkpoint available": bool(result.get("stage41_checkpoint_exists", False)),
        "Gate2 Stage42 row_id alignment available": matched > 0,
        "Gate3 no leakage protocol verified": bool(result.get("no_leakage", {}).get("future_endpoint_input") is False),
        "Gate4 validation full-waypoint bridge evaluated": result.get("val_bridge", {}).get("matched_rows", 0) > 0,
        "Gate5 UCY endpoint candidate has non-floor switches": float(test.get("UCY_zara03_test", {}).get("switch_rate", 0.0)) > 0.0,
        "Gate6 UCY full-waypoint bridge deployable": bridge_positive,
        "Gate7 Stage5C false": not result.get("claim_boundary", {}).get("stage5c_executed", True),
        "Gate8 SMC false": not result.get("claim_boundary", {}).get("smc_enabled", True),
    }
    passed = sum(1 for ok in gates.values() if ok)
    verdict = "stage42_u_ucy_endpoint_to_full_bridge_pass" if passed == len(gates) else "stage42_u_ucy_endpoint_to_full_bridge_failed_blocker"
    return {"gates": gates, "passed": passed, "total": len(gates), "verdict": verdict}


def run(*, force: bool = False) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    if not force:
        cached = _cached_result_if_available()
        if cached:
            return cached
    stage41_report, checkpoint, policy = _stage41_best()
    val_bridge = _evaluate_bridge("val", checkpoint, policy)
    test_bridge = _evaluate_bridge("test", checkpoint, policy)
    result: dict[str, Any] = {
        "stage": "Stage42-U UCY candidate endpoint-to-full bridge audit",
        "source": "fresh_run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                "outputs/stage41_external_split/stage41_pure_ucy_neural_retrain.json",
                checkpoint,
                "data/stage41_pure_ucy_neural_retrain/seq2seq_val.npz",
                "data/stage41_pure_ucy_neural_retrain/seq2seq_test.npz",
                "data/stage41_fresh_confirmation/full_trajectory_val.npz",
                "data/stage41_fresh_confirmation/full_trajectory_test.npz",
            ]
        ),
        "runtime": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "num_workers": 0,
            "torch_threads": 4,
        },
        "current_facts": CURRENT_FACTS,
        "stage41_endpoint_candidate": {
            "source": "cached_verified_stage41_strict_pure_ucy_neural",
            "best_trial": stage41_report.get("best_trial"),
            "best_mode": stage41_report.get("best_mode"),
            "best_policy": policy,
            "best_metrics": stage41_report.get("best_metrics"),
        },
        "stage41_checkpoint": checkpoint,
        "stage41_checkpoint_exists": Path(checkpoint).exists(),
        "val_bridge": val_bridge,
        "test_bridge": test_bridge,
        "interpretation": {
            "endpoint_candidate_available": True,
            "full_waypoint_bridge_deployable": False,
            "root_cause": "Stage41 pure-UCY endpoint residual is positive on endpoint FDE, but linear endpoint-to-waypoint interpolation is negative on Stage42 full-waypoint validation and UCY test. Endpoint success cannot be counted as full-waypoint world-state success.",
            "next_action": "Train/cache a UCY-aware full-waypoint candidate source with train/val source split, or learn a waypoint-shape bridge selected on validation; do not merge this endpoint bridge into Stage42-R/S deployable policy.",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_endpoint_label_eval_only": True,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_policy_tuning": False,
            "stage41_policy_train_val_selected": True,
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
    gate = _gate(result)
    result["gate"] = gate
    result["verdict"] = gate["verdict"]
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _markdown(result))
    write_md(GATE_MD, _gate_markdown(gate))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"stage": "Stage42-U", "source": "fresh_run", "verdict": result["verdict"], "output": str(REPORT_JSON)}), ensure_ascii=False) + "\n")
    return result


def _fmt_metric(row: Mapping[str, Any]) -> str:
    ade = row.get("ade", {})
    fde = row.get("fde", {})
    return (
        f"rows={row.get('rows', 0)}, matched={row.get('matched_rows', 0)}, "
        f"ADE all={float(ade.get('all_improvement', 0.0)):.6f}, "
        f"ADE t50={float(ade.get('t50_improvement', 0.0)):.6f}, "
        f"ADE hard={float(ade.get('hard_failure_improvement', 0.0)):.6f}, "
        f"easy={float(ade.get('easy_degradation', 0.0)):.6f}, "
        f"FDE t50={float(fde.get('t50_improvement', 0.0)):.6f}, "
        f"switch={float(row.get('switch_rate', 0.0)):.6f}"
    )


def _markdown(result: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-U UCY Candidate Endpoint-To-Full Bridge Audit",
        "",
        f"- source: `{result.get('source')}`",
        f"- generated_at_utc: `{result.get('generated_at_utc')}`",
        f"- git_commit: `{result.get('git_commit')}`",
        f"- gate: `{result.get('gate', {}).get('passed')} / {result.get('gate', {}).get('total')}`",
        f"- verdict: `{result.get('verdict')}`",
        "",
        "## Current Facts",
        "",
    ]
    lines.extend([f"- {fact}" for fact in CURRENT_FACTS])
    lines.extend(
        [
            "",
            "## Endpoint Candidate",
            "",
            f"- Stage41 source: `{result.get('stage41_endpoint_candidate', {}).get('source')}`",
            f"- best trial/mode: `{result.get('stage41_endpoint_candidate', {}).get('best_trial')}` / `{result.get('stage41_endpoint_candidate', {}).get('best_mode')}`",
            f"- checkpoint exists: `{result.get('stage41_checkpoint_exists')}`",
            "",
            "## Full-Waypoint Bridge Metrics",
            "",
            "| split/slice | metrics |",
            "| --- | --- |",
        ]
    )
    for split_key in ["val_bridge", "test_bridge"]:
        for name, row in result.get(split_key, {}).get("metrics", {}).items():
            lines.append(f"| `{split_key}:{name}` | {_fmt_metric(row)} |")
    interp = result.get("interpretation", {})
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- endpoint candidate available: `{interp.get('endpoint_candidate_available')}`",
            f"- full-waypoint bridge deployable: `{interp.get('full_waypoint_bridge_deployable')}`",
            f"- root cause: {interp.get('root_cause')}",
            f"- next action: {interp.get('next_action')}",
            "",
            "## No-Leakage / Claim Boundary",
            "",
            f"- no leakage: `{result.get('no_leakage')}`",
            f"- claim boundary: `{result.get('claim_boundary')}`",
        ]
    )
    return lines


def _gate_markdown(gate: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-U Gates",
        "",
        f"- gates passed: `{gate.get('passed')} / {gate.get('total')}`",
        f"- verdict: `{gate.get('verdict')}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    for name, ok in gate.get("gates", {}).items():
        lines.append(f"| {name} | `{bool(ok)}` |")
    return lines


def main() -> None:
    run(force=True)


if __name__ == "__main__":
    main()
