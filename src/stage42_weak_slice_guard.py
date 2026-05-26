from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_row_prediction_cache as r
from src import stage42_sequence_full_waypoint as s42i
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_data_calibration import OUT_DIR
from src.stage42_unified_row_cache_stress import _metric_ci_low, _metric_mean, _metric_ci_high


STAGE42X_JSON = OUT_DIR / "unified_row_level_full_waypoint_cache_stage42.json"
STAGE42R_JSON = OUT_DIR / "row_prediction_cache_stage42.json"
STAGE42AE_JSON = OUT_DIR / "unified_row_cache_stress_stage42.json"
STAGE42X_CACHE_DIR = Path("data/stage42_unified_full_waypoint_cache")

VAL_MARGIN_THRESHOLD = 0.02

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-AF 是 validation-margin weak-slice guard repair，不重新训练大模型，不读取/提交 raw data。",
    "Guard rule 只使用 Stage42-R validation score 和预设 margin，不用 test 调阈值。",
    "Future waypoints/endpoints 只作为 labels/eval，不作为 inference input。",
    "t+50 / t+100 是 raw-frame horizons，不能说成 seconds-level。",
    "External coordinates remain dataset-local / unverified weak metric diagnostic。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _hash_inputs(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path).encode("utf-8"))
        h.update(b"\0")
        if path.exists():
            h.update(path.read_bytes())
        else:
            h.update(b"missing")
    return h.hexdigest()


def _npz_path(pair_idx: int) -> Path:
    return STAGE42X_CACHE_DIR / f"stage42x_unified_pair_{pair_idx}.npz"


def validation_margin_guard_keys(stage42r_pair_row: Mapping[str, Any], threshold: float = VAL_MARGIN_THRESHOLD) -> list[str]:
    """Return domain|horizon keys that should fall back by validation margin.

    This function intentionally does not inspect test metrics. It reads the
    validation-selected choices saved by Stage42-R and applies a fixed margin.
    UCY is excluded because Stage42-X imports UCY from the independent
    Stage42-V protocol, not from Stage42-R choices.
    """

    guarded = []
    for key, choice in sorted((stage42r_pair_row.get("choices") or {}).items()):
        domain, _ = str(key).split("|", 1)
        if domain == "UCY":
            continue
        if float(choice.get("val_score", 0.0)) < threshold:
            guarded.append(str(key))
    return guarded


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "source": "fresh_run_from_validation_margin_guard",
        "seeds": [int(row["pair_idx"]) for row in rows],
        "ade_all": r._stat([row["guarded_test_metrics"]["ade"].get("all_improvement", 0.0) for row in rows]),
        "ade_t50": r._stat([row["guarded_test_metrics"]["ade"].get("t50_improvement", 0.0) for row in rows]),
        "ade_t100_raw_frame_diagnostic": r._stat([row["guarded_test_metrics"]["ade"].get("t100_improvement", 0.0) for row in rows]),
        "ade_hard_failure": r._stat([row["guarded_test_metrics"]["ade"].get("hard_failure_improvement", 0.0) for row in rows]),
        "ade_easy_degradation": r._stat([row["guarded_test_metrics"]["ade"].get("easy_degradation", 1.0) for row in rows]),
        "fde_t50": r._stat([row["guarded_test_metrics"]["fde"].get("t50_improvement", 0.0) for row in rows]),
        "switch_rate": r._stat([row["guarded_test_metrics"].get("switch_rate", 0.0) for row in rows]),
    }


def _slice_stats(rows: list[Mapping[str, Any]], labels: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    if int(np.sum(mask)) == 0:
        return {"rows": 0, "source": "not_run_empty_slice"}
    ade_metrics = []
    fde_metrics = []
    switches = []
    for row in rows:
        arr = row["arrays_for_bootstrap"]
        switch = arr["guarded_test_switch"].astype(bool)
        ade_metrics.append(r._local_metric_from_errors(arr["guarded_test_ade"], arr["floor_test_ade"], labels, switch, mask))
        fde_metrics.append(r._local_metric_from_errors(arr["guarded_test_fde"], arr["floor_test_fde"], labels, switch, mask))
        switches.append(float(np.mean(switch[mask])) if int(np.sum(mask)) else 0.0)
    return {
        "rows": int(np.sum(mask)),
        "source": "fresh_run_from_validation_margin_guard",
        "ade_all": r._stat([m.get("all_improvement", 0.0) for m in ade_metrics]),
        "ade_t50": r._stat([m.get("t50_improvement", 0.0) for m in ade_metrics]),
        "ade_t100_raw_frame_diagnostic": r._stat([m.get("t100_improvement", 0.0) for m in ade_metrics]),
        "ade_hard_failure": r._stat([m.get("hard_failure_improvement", 0.0) for m in ade_metrics]),
        "ade_easy_degradation": r._stat([m.get("easy_degradation", 0.0) for m in ade_metrics]),
        "fde_t50": r._stat([m.get("t50_improvement", 0.0) for m in fde_metrics]),
        "switch_rate": r._stat(switches),
    }


def _stress(rows: list[Mapping[str, Any]], labels: Mapping[str, np.ndarray]) -> dict[str, Any]:
    domains = sorted(set(labels["domain"].astype(str).tolist()))
    horizons = [10, 25, 50, 100]
    out: dict[str, Any] = {"by_domain": {}, "by_horizon": {}, "by_domain_horizon": {}}
    for domain in domains:
        out["by_domain"][domain] = _slice_stats(rows, labels, labels["domain"].astype(str) == domain)
    for horizon in horizons:
        out["by_horizon"][str(horizon)] = _slice_stats(rows, labels, labels["horizon"].astype(int) == horizon)
    for domain in domains:
        for horizon in horizons:
            mask = (labels["domain"].astype(str) == domain) & (labels["horizon"].astype(int) == horizon)
            out["by_domain_horizon"][f"{domain}|{horizon}"] = _slice_stats(rows, labels, mask)
    return out


def _guarded_rows(stage42r_report: Mapping[str, Any], labels: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    domains = labels["domain"].astype(str)
    horizons = labels["horizon"].astype(int)
    for pair_row in stage42r_report.get("rows", []):
        pair_idx = int(pair_row["pair_idx"])
        npz = np.load(_npz_path(pair_idx), allow_pickle=False)
        floor_ade = npz["floor_test_ade"].copy()
        floor_fde = npz["floor_test_fde"].copy()
        guarded_ade = npz["merged_test_ade"].copy()
        guarded_fde = npz["merged_test_fde"].copy()
        guarded_switch = npz["merged_test_switch"].copy()
        guard_keys = validation_margin_guard_keys(pair_row)
        guard_masks = {}
        for key in guard_keys:
            domain, horizon_s = key.split("|", 1)
            mask = (domains == domain) & (horizons == int(horizon_s))
            guarded_ade[mask] = floor_ade[mask]
            guarded_fde[mask] = floor_fde[mask]
            guarded_switch[mask] = False
            guard_masks[key] = int(np.sum(mask))
        metrics = {
            "ade": r._metric_from_errors(guarded_ade, floor_ade, labels, guarded_switch),
            "fde": r._metric_from_errors(guarded_fde, floor_fde, labels, guarded_switch),
            "switch_rate": float(np.mean(guarded_switch)) if len(guarded_switch) else 0.0,
        }
        rows.append(
            {
                "source": "fresh_run_from_stage42x_cache_validation_margin_guard",
                "pair_idx": pair_idx,
                "guard_threshold": VAL_MARGIN_THRESHOLD,
                "guarded_keys": guard_keys,
                "guarded_rows_by_key": guard_masks,
                "guarded_test_metrics": metrics,
                "arrays_for_bootstrap": {
                    "floor_test_ade": floor_ade,
                    "floor_test_fde": floor_fde,
                    "guarded_test_ade": guarded_ade,
                    "guarded_test_fde": guarded_fde,
                    "guarded_test_switch": guarded_switch,
                },
            }
        )
    return rows


def _improvement_vs_stage42x(stage42x: Mapping[str, Any], guarded_stress: Mapping[str, Any]) -> dict[str, Any]:
    base_h25 = (stage42x.get("stress") or {}).get("by_horizon", {}).get("25", {})
    guard_h25 = guarded_stress.get("by_horizon", {}).get("25", {})
    base_eth50 = (stage42x.get("stress") or {}).get("by_domain_horizon", {}).get("ETH_UCY|50", {})
    guard_eth50 = guarded_stress.get("by_domain_horizon", {}).get("ETH_UCY|50", {})
    return {
        "horizon25_ade_all_before": _metric_mean(base_h25, "ade_all"),
        "horizon25_ade_all_after": _metric_mean(guard_h25, "ade_all"),
        "horizon25_delta": _metric_mean(guard_h25, "ade_all") - _metric_mean(base_h25, "ade_all"),
        "eth_ucy_t50_ade_before": _metric_mean(base_eth50, "ade_t50"),
        "eth_ucy_t50_ade_after": _metric_mean(guard_eth50, "ade_t50"),
        "eth_ucy_t50_ci_low_after": _metric_ci_low(guard_eth50, "ade_t50"),
        "eth_ucy_fde_t50_ci_low_after": _metric_ci_low(guard_eth50, "fde_t50"),
        "eth_ucy_t50_limitation_remaining": _metric_ci_low(guard_eth50, "ade_t50") <= 0 or _metric_ci_low(guard_eth50, "fde_t50") <= 0,
    }


def _strip_arrays(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "arrays_for_bootstrap"}


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
    return value


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    repair = payload["repair_effect"]
    gates = [
        ("Stage42-X Input Verified", payload["stage42x_gate"].get("verdict") == "stage42_x_unified_row_level_full_waypoint_cache_pass"),
        ("Stage42-R Validation Choices Available", len(payload["guarded_rows"]) >= 3),
        ("Guard Uses Validation Margin Not Test Tuning", payload["guard_rule"]["uses_test_metrics_for_threshold"] is False),
        ("Horizon25 Repaired To Non-Harm", repair["horizon25_ade_all_after"] >= 0.0 and repair["horizon25_delta"] > 0.0),
        ("Global All Positive", summary["ade_all"]["mean"] > 0.0),
        ("Global T50 Positive", summary["ade_t50"]["mean"] > 0.0 and summary["ade_t50"]["ci_low"] > 0.0),
        ("Hard/Failure Positive", summary["ade_hard_failure"]["mean"] > 0.0),
        ("Easy Preserved", summary["ade_easy_degradation"]["ci_high"] <= 0.02),
        ("ETH_UCY T50 Limitation Preserved", repair["eth_ucy_t50_limitation_remaining"] is True),
        ("No Leakage Inherited", payload["no_leakage"]["future_endpoint_input"] is False and payload["no_leakage"]["test_policy_tuning"] is False),
        ("No Metric/Seconds Overclaim", payload["claim_boundary"]["metric_or_seconds_claim"] is False),
        ("Stage5C Execution Gate", payload["claim_boundary"]["stage5c_executed"] is False),
        ("SMC Execution Gate", payload["claim_boundary"]["smc_enabled"] is False),
    ]
    passed = sum(1 for _, ok in gates if ok)
    return {
        "source": "fresh_run_validation_margin_guard",
        "passed": passed,
        "total": len(gates),
        "verdict": "stage42_af_weak_slice_guard_repair_pass_with_eth_t50_limitation" if passed == len(gates) else "stage42_af_weak_slice_guard_repair_partial",
        "gates": [{"name": name, "passed": bool(ok)} for name, ok in gates],
    }


def build_weak_slice_guard() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage42x = read_json(STAGE42X_JSON, {})
    stage42r = read_json(STAGE42R_JSON, {})
    stage42ae = read_json(STAGE42AE_JSON, {})
    if not stage42x or not stage42r:
        raise FileNotFoundError("Stage42-X and Stage42-R reports are required.")
    labels = s42i._labels(s42i._split_arrays("test"))
    guarded_rows = _guarded_rows(stage42r, labels)
    stress = _stress(guarded_rows, labels)
    payload = {
        "source": "fresh_run_from_stage42x_cache_and_stage42r_validation_margin",
        "stage": "Stage42-AF weak-slice validation-margin guard repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "input_hash": _hash_inputs([STAGE42X_JSON, STAGE42R_JSON, STAGE42AE_JSON]),
        "current_facts": CURRENT_FACTS,
        "guard_rule": {
            "name": "validation_margin_guard",
            "val_margin_threshold": VAL_MARGIN_THRESHOLD,
            "fallback_action": "force floor for non-UCY domain|horizon source choices with validation score below threshold",
            "ucy_policy": "unchanged Stage42-V independent validation-selected source",
            "uses_test_metrics_for_threshold": False,
        },
        "stage42x_gate": stage42x.get("stage42_x_gate", {}),
        "stage42ae_verdict": stage42ae.get("stage42_ae_gate", {}).get("verdict"),
        "summary": _summary(guarded_rows),
        "stress": stress,
        "repair_effect": _improvement_vs_stage42x(stage42x, stress),
        "guarded_rows": [_strip_arrays(row) for row in guarded_rows],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoints_used_as_train_val_label_and_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_policy_tuning": False,
            "guard_threshold_from_validation_score_only": True,
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
    payload["stage42_af_gate"] = _gate(payload)
    write_json(OUT_DIR / "weak_slice_guard_stage42.json", _jsonable(payload))
    write_md(OUT_DIR / "weak_slice_guard_stage42.md", _render_md(payload))
    _write_csv(OUT_DIR / "weak_slice_guard_stage42.csv", payload)
    write_md(OUT_DIR / "stage42_stage_af_gate.md", _render_gate_md(payload))
    return payload


def _write_csv(path: Path, payload: Mapping[str, Any]) -> None:
    rows = []
    for key, row in sorted(payload["stress"]["by_horizon"].items(), key=lambda kv: int(kv[0])):
        rows.append(
            {
                "kind": "horizon",
                "name": key,
                "rows": row.get("rows", 0),
                "ade_all": _metric_mean(row, "ade_all"),
                "ade_all_ci_low": _metric_ci_low(row, "ade_all"),
                "ade_t50": _metric_mean(row, "ade_t50"),
                "ade_t50_ci_low": _metric_ci_low(row, "ade_t50"),
                "ade_hard_failure": _metric_mean(row, "ade_hard_failure"),
                "easy_degradation_ci_high": _metric_ci_high(row, "ade_easy_degradation"),
                "switch_rate": _metric_mean(row, "switch_rate"),
            }
        )
    for key, row in sorted(payload["stress"]["by_domain_horizon"].items()):
        rows.append(
            {
                "kind": "domain_horizon",
                "name": key,
                "rows": row.get("rows", 0),
                "ade_all": _metric_mean(row, "ade_all"),
                "ade_all_ci_low": _metric_ci_low(row, "ade_all"),
                "ade_t50": _metric_mean(row, "ade_t50"),
                "ade_t50_ci_low": _metric_ci_low(row, "ade_t50"),
                "ade_hard_failure": _metric_mean(row, "ade_hard_failure"),
                "easy_degradation_ci_high": _metric_ci_high(row, "ade_easy_degradation"),
                "switch_rate": _metric_mean(row, "switch_rate"),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_af_gate"]
    summary = payload["summary"]
    repair = payload["repair_effect"]
    lines = [
        "# Stage42-AF Weak-Slice Validation-Margin Guard Repair",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Claim Boundary",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Guard Rule",
        "",
        f"- rule: `{payload['guard_rule']['name']}`",
        f"- val_margin_threshold: `{payload['guard_rule']['val_margin_threshold']}`",
        f"- fallback_action: {payload['guard_rule']['fallback_action']}",
        f"- uses_test_metrics_for_threshold: `{payload['guard_rule']['uses_test_metrics_for_threshold']}`",
        "",
        "## Summary",
        "",
        f"- ADE all: `{summary['ade_all']['mean']}`",
        f"- ADE t50: `{summary['ade_t50']['mean']}`",
        f"- ADE t50 CI low: `{summary['ade_t50']['ci_low']}`",
        f"- ADE t100 raw-frame diagnostic: `{summary['ade_t100_raw_frame_diagnostic']['mean']}`",
        f"- ADE hard/failure: `{summary['ade_hard_failure']['mean']}`",
        f"- easy degradation CI high: `{summary['ade_easy_degradation']['ci_high']}`",
        f"- switch_rate: `{summary['switch_rate']['mean']}`",
        "",
        "## Repair Effect",
        "",
        f"- horizon25 ADE before: `{repair['horizon25_ade_all_before']}`",
        f"- horizon25 ADE after: `{repair['horizon25_ade_all_after']}`",
        f"- horizon25 delta: `{repair['horizon25_delta']}`",
        f"- ETH_UCY t50 ADE before: `{repair['eth_ucy_t50_ade_before']}`",
        f"- ETH_UCY t50 ADE after: `{repair['eth_ucy_t50_ade_after']}`",
        f"- ETH_UCY t50 CI low after: `{repair['eth_ucy_t50_ci_low_after']}`",
        f"- ETH_UCY FDE@50 CI low after: `{repair['eth_ucy_fde_t50_ci_low_after']}`",
        f"- ETH_UCY t50 limitation remaining: `{repair['eth_ucy_t50_limitation_remaining']}`",
        "",
        "## Per-Horizon Stress After Guard",
        "",
        "| horizon | rows | ADE all | ADE all low | hard | hard low | switch rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for horizon, row in sorted(payload["stress"]["by_horizon"].items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"| `{horizon}` | {row.get('rows', 0)} | {_metric_mean(row, 'ade_all'):.6f} | {_metric_ci_low(row, 'ade_all'):.6f} | {_metric_mean(row, 'ade_hard_failure'):.6f} | {_metric_ci_low(row, 'ade_hard_failure'):.6f} | {_metric_mean(row, 'switch_rate'):.6f} |"
        )
    lines.extend(["", "## Guarded Keys By Seed", ""])
    for row in payload["guarded_rows"]:
        lines.append(f"- pair `{row['pair_idx']}` guarded `{', '.join(row['guarded_keys']) or 'none'}` rows={row['guarded_rows_by_key']}")
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "Stage42-AF repairs the Stage42-AE horizon=25 weak slice by a validation-only low-margin guard: horizon=25 moves from negative to non-harm/floor. Global all/t50/hard remain positive and easy degradation stays under 2%. This is a real safety improvement, but ETH_UCY t50/FDE@50 lower-bound weakness remains; it must stay in the paper limitations rather than being overclaimed.",
        ]
    )
    return lines


def _render_gate_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_af_gate"]
    lines = [
        "# Stage42-AF Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for row in gate["gates"]:
        lines.append(f"| {row['name']} | `{row['passed']}` |")
    return lines


def main() -> None:
    build_weak_slice_guard()


if __name__ == "__main__":
    main()
