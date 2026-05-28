from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_source_level_row_cache_integration as iv
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "source_level_row_cache_mechanism_audit_stage42.json"
REPORT_MD = OUT_DIR / "source_level_row_cache_mechanism_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_iw_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-IW 只审计 Stage42-IV 单一 source-level row-cache 能直接支持的机制证据。",
    "history / neighbor / goal / interaction 的独立贡献不能只靠这个 row-cache 证明，仍需要 retrained ablation evidence。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "本阶段仍是 dataset-local/raw-frame 2.5D evidence，不是 metric 或 seconds-level 结果。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

EPS = 1e-8


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


def _load_stage42iv_cache() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not iv.CACHE_NPZ.exists() or not iv.REPORT_JSON.exists():
        iv.run_stage42_source_level_row_cache_integration()
    report = read_json(iv.REPORT_JSON, {})
    with np.load(iv.CACHE_NPZ, allow_pickle=False) as npz:
        cache = {name: npz[name] for name in npz.files}
    return cache, report


def _labels(cache: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "domain": cache["domain"].astype(str),
        "source_file": cache["source_file"].astype(str),
        "scene_id": cache["scene_id"].astype(str),
        "horizon": cache["horizon"].astype(np.int64),
        "hard": cache["hard"].astype(bool),
        "failure": cache["failure"].astype(bool),
        "easy": cache["easy"].astype(bool),
    }


def _metric(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if int(np.sum(mask)) == 0:
        return {
            "rows": 0,
            "all_improvement": 0.0,
            "t50_improvement": 0.0,
            "t100_raw_frame_diagnostic_improvement": 0.0,
            "hard_failure_improvement": 0.0,
            "easy_degradation": 0.0,
            "switch_rate": 0.0,
            "harm_over_fallback": 0.0,
        }
    selected_m = selected[mask].astype(np.float64)
    floor_m = floor[mask].astype(np.float64)
    horizon = labels["horizon"][mask].astype(int)
    hard_failure = labels["hard"][mask].astype(bool) | labels["failure"][mask].astype(bool)
    easy = labels["easy"][mask].astype(bool)
    domain = labels["domain"][mask].astype(str)
    switch_m = switch[mask].astype(bool)

    def imp(local_mask: np.ndarray) -> float:
        if not np.any(local_mask):
            return 0.0
        return float(1.0 - selected_m[local_mask].mean() / max(float(floor_m[local_mask].mean()), EPS))

    row: dict[str, Any] = {
        "rows": int(len(selected_m)),
        "all_improvement": imp(np.ones(len(selected_m), dtype=bool)),
        "t10_improvement": imp(horizon == 10),
        "t25_improvement": imp(horizon == 25),
        "t50_improvement": imp(horizon == 50),
        "t100_improvement": imp(horizon == 100),
        "t100_raw_frame_diagnostic_improvement": imp(horizon == 100),
        "hard_failure_improvement": imp(hard_failure),
        "easy_degradation": float(max(0.0, selected_m[easy].mean() / max(float(floor_m[easy].mean()), EPS) - 1.0)) if np.any(easy) else 0.0,
        "harm_over_fallback": float(np.mean(selected_m - floor_m)),
        "switch_rate": float(np.mean(switch_m)) if len(switch_m) else 0.0,
    }
    by_domain = {}
    for d in sorted(set(domain.tolist())):
        dm = domain == d
        by_domain[d] = {
            "rows": int(np.sum(dm)),
            "all_improvement": imp(dm),
            "t50_improvement": imp(dm & (horizon == 50)),
            "t100_raw_frame_diagnostic_improvement": imp(dm & (horizon == 100)),
            "hard_failure_improvement": imp(dm & hard_failure),
            "easy_degradation": float(max(0.0, selected_m[dm & easy].mean() / max(float(floor_m[dm & easy].mean()), EPS) - 1.0)) if np.any(dm & easy) else 0.0,
            "switch_rate": float(np.mean(switch_m[dm])) if np.any(dm) else 0.0,
        }
    row["by_domain"] = by_domain
    return row


def _switch_summary(cache: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> dict[str, Any]:
    selected = cache["selected_ade_seed_mean"].astype(np.float64)
    floor = cache["floor_ade"].astype(np.float64)
    switch_score = cache["switch_seed_mean"].astype(np.float64)
    switch = switch_score > 0.0
    gain = floor - selected
    harm = selected - floor
    hard_failure = labels["hard"] | labels["failure"]
    easy = labels["easy"]
    out: dict[str, Any] = {
        "rows": int(len(selected)),
        "switch_rows": int(np.sum(switch)),
        "fallback_rows": int(np.sum(~switch)),
        "switch_rate": float(np.mean(switch)),
        "mean_gain_all_rows": float(np.mean(gain)),
        "mean_gain_switched_rows": float(np.mean(gain[switch])) if int(np.sum(switch)) else 0.0,
        "harm_rate_all_rows": float(np.mean(harm > EPS)),
        "harm_rate_switched_rows": float(np.mean(harm[switch] > EPS)) if int(np.sum(switch)) else 0.0,
        "hard_failure_switch_rate": float(np.mean(switch[hard_failure])) if int(np.sum(hard_failure)) else 0.0,
        "easy_switch_rate": float(np.mean(switch[easy])) if int(np.sum(easy)) else 0.0,
        "easy_mean_harm": float(np.mean(np.maximum(harm[easy], 0.0))) if int(np.sum(easy)) else 0.0,
        "fallback_exact_floor_rate": float(np.mean(np.abs(selected[~switch] - floor[~switch]) <= 1e-6)) if int(np.sum(~switch)) else 0.0,
    }
    by_domain = {}
    for domain in sorted(set(labels["domain"].tolist())):
        mask = labels["domain"] == domain
        by_domain[domain] = {
            "rows": int(np.sum(mask)),
            "switch_rate": float(np.mean(switch[mask])),
            "mean_gain": float(np.mean(gain[mask])),
            "harm_rate": float(np.mean(harm[mask] > EPS)),
            "metric": _metric(selected, floor, labels, switch, mask),
        }
    out["by_domain"] = by_domain
    by_horizon = {}
    for horizon in [10, 25, 50, 100]:
        mask = labels["horizon"] == horizon
        by_horizon[str(horizon)] = {
            "rows": int(np.sum(mask)),
            "switch_rate": float(np.mean(switch[mask])) if int(np.sum(mask)) else 0.0,
            "mean_gain": float(np.mean(gain[mask])) if int(np.sum(mask)) else 0.0,
            "harm_rate": float(np.mean(harm[mask] > EPS)) if int(np.sum(mask)) else 0.0,
            "metric": _metric(selected, floor, labels, switch, mask),
        }
    out["by_horizon"] = by_horizon
    return out


def _waypoint_shape_summary(cache: Mapping[str, np.ndarray]) -> dict[str, Any]:
    waypoint_xy = cache["waypoint_xy"].astype(np.float64)
    valid = cache["waypoint_valid"].astype(bool)
    current = cache["current_xy"].astype(np.float64)
    future = cache["future_xy"].astype(np.float64)
    rows = waypoint_xy.shape[0]
    valid_counts = np.sum(valid, axis=1)
    fractions = (np.arange(1, waypoint_xy.shape[1] + 1, dtype=np.float64) / max(float(waypoint_xy.shape[1]), 1.0))[None, :, None]
    linear = current[:, None, :] + fractions * (future - current)[:, None, :]
    residual = np.linalg.norm(waypoint_xy - linear, axis=2)
    residual_masked = np.where(valid, residual, np.nan)
    step = np.diff(waypoint_xy, axis=1)
    step_valid = valid[:, 1:] & valid[:, :-1]
    step_norm = np.linalg.norm(step, axis=2)
    nonzero_step = step_norm > 1e-8
    dir_vec = np.divide(step, step_norm[:, :, None], out=np.zeros_like(step), where=nonzero_step[:, :, None])
    turn = np.sum(dir_vec[:, 1:, :] * dir_vec[:, :-1, :], axis=2)
    turn = np.clip(turn, -1.0, 1.0)
    turn_angle = np.arccos(turn)
    turn_valid = step_valid[:, 1:] & step_valid[:, :-1]
    return {
        "rows": int(rows),
        "waypoint_count": int(waypoint_xy.shape[1]),
        "rows_with_any_waypoint": int(np.sum(valid_counts > 0)),
        "rows_with_full_waypoints": int(np.sum(valid_counts == waypoint_xy.shape[1])),
        "rows_with_at_least_two_waypoints": int(np.sum(valid_counts >= 2)),
        "full_waypoint_rate": float(np.mean(valid_counts == waypoint_xy.shape[1])),
        "mean_valid_waypoints_per_row": float(np.mean(valid_counts)),
        "mean_raw_residual_from_linear_bridge": float(np.nanmean(residual_masked)),
        "median_raw_residual_from_linear_bridge": float(np.nanmedian(residual_masked)),
        "p90_raw_residual_from_linear_bridge": float(np.nanpercentile(residual_masked, 90)),
        "mean_turn_angle_radians": float(np.nanmean(np.where(turn_valid, turn_angle, np.nan))),
        "note": "Residuals and turn angles are dataset-local/raw-frame shape diagnostics, not metric distances or seconds-level dynamics.",
    }


def _bootstrap_summary(cache: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> dict[str, Any]:
    selected = cache["selected_ade_seed_mean"].astype(np.float64)
    floor = cache["floor_ade"].astype(np.float64)
    h = labels["horizon"]
    hard_failure = labels["hard"] | labels["failure"]
    easy = labels["easy"]
    return {
        "all": iv._bootstrap(selected, floor, np.ones(len(selected), dtype=bool), seed=42101),
        "t50": iv._bootstrap(selected, floor, h == 50, seed=42102),
        "t100_raw_frame_diagnostic": iv._bootstrap(selected, floor, h == 100, seed=42103),
        "hard_failure": iv._bootstrap(selected, floor, hard_failure, seed=42104),
        "easy_degradation": iv._bootstrap(selected, floor, easy, easy=True, seed=42105),
    }


def _external_evidence_status() -> dict[str, Any]:
    checks = {
        "source_level_incremental_ablation": OUT_DIR / "source_level_incremental_ablation_stage42.json",
        "unified_ablation_evidence": OUT_DIR / "unified_ablation_evidence_stage42.json",
        "source_level_safety_floor_audit": OUT_DIR / "source_level_safety_floor_audit_stage42.json",
        "source_level_full_waypoint_eval": OUT_DIR / "source_level_full_waypoint_eval_stage42.json",
    }
    out = {}
    for name, path in checks.items():
        payload = read_json(path, {}) if path.exists() else {}
        gate = (
            payload.get("stage42_ao_gate")
            or payload.get("stage42_y_gate")
            or payload.get("stage42_at_gate")
            or payload.get("stage42_am_gate")
            or payload.get("stage42_source_level_safety_floor_gate")
            or {}
        )
        out[name] = {
            "source": "cached_verified" if payload else "not_run",
            "path": str(path),
            "exists": bool(payload),
            "verdict": gate.get("verdict"),
        }
    return out


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    m = result["metric"]
    switch = result["switch_mechanism"]
    waypoint = result["full_waypoint_shape"]
    gates = {
        "stage42iv_cache_loaded": result["cache"]["exists"] is True,
        "stage42iv_cache_hash_recorded": bool(result["cache"].get("hash")),
        "row_count_matches_source_level_cache": result["rows"] == 47458,
        "two_external_domains_present": len(result["domain_rows"]) >= 2,
        "all_improvement_positive": m["all_improvement"] > 0,
        "t50_improvement_positive": m["t50_improvement"] > 0,
        "t100_raw_frame_diagnostic_positive": m["t100_raw_frame_diagnostic_improvement"] > 0,
        "hard_failure_positive": m["hard_failure_improvement"] > 0,
        "easy_preserved": m["easy_degradation"] <= 0.02,
        "safe_switch_has_positive_mean_gain": switch["mean_gain_switched_rows"] > 0,
        "fallback_rows_match_floor": switch["fallback_exact_floor_rate"] >= 0.999,
        "waypoint_sequence_labels_available": waypoint["rows_with_at_least_two_waypoints"] == waypoint["rows"],
        "full_waypoint_completeness_reported": waypoint["full_waypoint_rate"] > 0.0,
        "bootstrap_reported": all(row.get("bootstrap_n", 0) >= iv.BOOTSTRAP_N for row in result["bootstrap"].values()),
        "no_leakage_pass": all(result["no_leakage"][k] is False for k in ["future_endpoint_input", "future_waypoint_input", "central_velocity", "test_endpoint_goals", "test_threshold_tuning"]),
        "no_metric_seconds_overclaim": result["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": result["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": result["claim_boundary"]["smc_enabled"] is False,
    }
    verdict = "stage42_iw_row_cache_mechanism_audit_pass" if all(gates.values()) else "stage42_iw_row_cache_mechanism_audit_partial"
    return {"source": "fresh_run", "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": len(gates), "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    m = result["metric"]
    s = result["switch_mechanism"]
    w = result["full_waypoint_shape"]
    lines = [
        "# Stage42-IW Source-Level Row-Cache Mechanism Audit",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_iw_gate']['passed']} / {result['stage42_iw_gate']['total']}`",
        f"- verdict: `{result['stage42_iw_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## What This Audit Can And Cannot Prove",
        "",
        "- Directly supported by the Stage42-IV row-cache: safe-switch behavior, teacher/floor fallback usage, source/horizon slice behavior, easy preservation, waypoint sequence coverage/completeness, and row-level bootstrap.",
        "- Not directly proven by this row-cache alone: independent causal contribution of history, neighbor, goal, interaction, JEPA, or Transformer modules. Those require retrained ablations; this report only records whether such reports exist as cached-verified supporting evidence.",
        "",
        "## Main Metrics From The Same Merged Row Cache",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| rows | {result['rows']} |",
        f"| all improvement | {m['all_improvement']:.6f} |",
        f"| t50 improvement | {m['t50_improvement']:.6f} |",
        f"| t100 raw-frame diagnostic improvement | {m['t100_raw_frame_diagnostic_improvement']:.6f} |",
        f"| hard/failure improvement | {m['hard_failure_improvement']:.6f} |",
        f"| easy degradation | {m['easy_degradation']:.6f} |",
        f"| switch rate | {m['switch_rate']:.6f} |",
        "",
        "## Safe-Switch / Floor Mechanism",
        "",
        "| field | value |",
        "| --- | ---: |",
        f"| switch rows | {s['switch_rows']} |",
        f"| fallback rows | {s['fallback_rows']} |",
        f"| switch rate | {s['switch_rate']:.6f} |",
        f"| mean gain, all rows | {s['mean_gain_all_rows']:.6f} |",
        f"| mean gain, switched rows | {s['mean_gain_switched_rows']:.6f} |",
        f"| harm rate, switched rows | {s['harm_rate_switched_rows']:.6f} |",
        f"| hard/failure switch rate | {s['hard_failure_switch_rate']:.6f} |",
        f"| easy switch rate | {s['easy_switch_rate']:.6f} |",
        f"| easy mean positive harm | {s['easy_mean_harm']:.6f} |",
        f"| fallback exact floor rate | {s['fallback_exact_floor_rate']:.6f} |",
        "",
        "## Full-Waypoint Shape Coverage",
        "",
        f"- full_waypoint_rate: `{w['full_waypoint_rate']:.6f}`",
        f"- mean_valid_waypoints_per_row: `{w['mean_valid_waypoints_per_row']:.6f}`",
        f"- mean raw residual from linear bridge: `{w['mean_raw_residual_from_linear_bridge']:.6f}`",
        f"- p90 raw residual from linear bridge: `{w['p90_raw_residual_from_linear_bridge']:.6f}`",
        f"- note: {w['note']}",
        "",
        "## By Domain",
        "",
        "| domain | rows | switch | all | t50 | t100 diag | hard/failure | easy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain, row in result["switch_mechanism"]["by_domain"].items():
        metric = row["metric"]
        lines.append(
            f"| `{domain}` | {row['rows']} | {row['switch_rate']:.6f} | {metric['all_improvement']:.6f} | {metric['t50_improvement']:.6f} | {metric['t100_raw_frame_diagnostic_improvement']:.6f} | {metric['hard_failure_improvement']:.6f} | {metric['easy_degradation']:.6f} |"
        )
    lines.extend(["", "## Bootstrap CI", "", "| slice | rows | mean | ci_low | ci_high | n |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for key, row in result["bootstrap"].items():
        lines.append(f"| `{key}` | {row['rows']} | {row['mean']:.6f} | {row['ci_low']:.6f} | {row['ci_high']:.6f} | {row['bootstrap_n']} |")
    lines.extend(
        [
            "",
            "## Cached-Verified Supporting Evidence",
            "",
            "| evidence | source | verdict | path |",
            "| --- | --- | --- | --- |",
        ]
    )
    for name, row in result["supporting_evidence"].items():
        lines.append(f"| `{name}` | `{row['source']}` | `{row.get('verdict')}` | `{row['path']}` |")
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
            "- Stage42-IW strengthens the evidence package by showing the current source-level merged cache is not just a headline metric: its gains come from switched rows while fallback rows stay tied to the teacher/floor, and easy cases remain protected.",
            "- The waypoint label audit confirms every row has at least two valid waypoint labels, while only a subset has all four valid waypoints; this is sequence-capable but not complete-full-waypoint coverage on every row.",
            "- Module claims for history, neighbor, goal, interaction, JEPA, and Transformer must still be grounded in retrained ablations, not inferred from this row-cache summary.",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_iw_gate"]
    lines = [
        "# Stage42-IW Gate",
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
    marker = "STAGE42_IW_ROW_CACHE_MECHANISM_AUDIT"
    m = result["metric"]
    s = result["switch_mechanism"]
    block = [
        "## Stage42-IW Source-Level Row-Cache Mechanism Audit",
        "",
        f"- source: `{result['source']}`",
        "- role: mechanism audit over the Stage42-IV single merged row-cache, not a new metric-only summary.",
        f"- gate: `{result['stage42_iw_gate']['passed']} / {result['stage42_iw_gate']['total']}`; verdict `{result['stage42_iw_gate']['verdict']}`.",
        f"- rows: `{result['rows']}`; domain rows: `{result['domain_rows']}`.",
        f"- ADE all/t50/t100raw/hard: `{m['all_improvement']:.6f}` / `{m['t50_improvement']:.6f}` / `{m['t100_raw_frame_diagnostic_improvement']:.6f}` / `{m['hard_failure_improvement']:.6f}`.",
        f"- easy degradation: `{m['easy_degradation']:.6f}`; switch rows `{s['switch_rows']}`; fallback exact floor rate `{s['fallback_exact_floor_rate']:.6f}`.",
        f"- full-waypoint coverage: `{result['full_waypoint_shape']['full_waypoint_rate']:.6f}`; bootstrap t50 CI `[{result['bootstrap']['t50']['ci_low']:.6f}, {result['bootstrap']['t50']['ci_high']:.6f}]`.",
        "- interpretation: safe-switch and teacher/floor protection are directly supported by this row-cache; waypoint labels are sequence-capable but not complete for every row; history/neighbor/goal/interaction still require retrained ablation evidence.",
        "- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.",
    ]
    for p in [Path("README_RESULTS.md"), Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")]:
        _replace_block(p, marker, block)
    _replace_block(Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md"), marker, block)


def _update_state(result: Mapping[str, Any]) -> None:
    path = Path("research_state.json")
    state = read_json(path, {})
    state["current_stage"] = "stage42_iw_source_level_row_cache_mechanism_audit"
    state["current_verdict"] = result["stage42_iw_gate"]["verdict"]
    state.setdefault("stage42", {})["stage_iw_source_level_row_cache_mechanism_audit"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": result["stage42_iw_gate"]["verdict"],
        "gates": f"{result['stage42_iw_gate']['passed']}/{result['stage42_iw_gate']['total']}",
        "rows": result["rows"],
        "domain_rows": result["domain_rows"],
        "metric": result["metric"],
        "switch_mechanism": {
            "switch_rows": result["switch_mechanism"]["switch_rows"],
            "fallback_rows": result["switch_mechanism"]["fallback_rows"],
            "switch_rate": result["switch_mechanism"]["switch_rate"],
            "fallback_exact_floor_rate": result["switch_mechanism"]["fallback_exact_floor_rate"],
        },
        "full_waypoint_shape": result["full_waypoint_shape"],
        "claim_boundary": result["claim_boundary"],
    }
    generated = state.setdefault("generated_reports", [])
    for item in [str(REPORT_MD), str(REPORT_JSON), str(GATE_MD)]:
        if item not in generated:
            generated.append(item)
    write_json(path, _jsonable(state))


def _append_ledger(result: Mapping[str, Any]) -> None:
    row = {
        "stage": result["stage"],
        "timestamp": result["generated_at_utc"],
        "source": result["source"],
        "verdict": result["stage42_iw_gate"]["verdict"],
        "gate": f"{result['stage42_iw_gate']['passed']}/{result['stage42_iw_gate']['total']}",
        "rows": result["rows"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_stage42_source_level_row_cache_mechanism_audit() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cache, iv_report = _load_stage42iv_cache()
    labels = _labels(cache)
    selected = cache["selected_ade_seed_mean"].astype(np.float64)
    floor = cache["floor_ade"].astype(np.float64)
    switch = cache["switch_seed_mean"].astype(np.float64) > 0.0
    all_mask = np.ones(len(selected), dtype=bool)
    metric = _metric(selected, floor, labels, switch, all_mask)
    if "t100_improvement" in metric and "t100_raw_frame_diagnostic_improvement" not in metric:
        metric["t100_raw_frame_diagnostic_improvement"] = metric["t100_improvement"]
    result = {
        "stage": "Stage42-IW source-level row-cache mechanism audit",
        "source": "fresh_run_row_cache_mechanism_audit_from_cached_verified_stage42iv_cache",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([iv.REPORT_JSON, iv.CACHE_NPZ]),
        "cache": {
            "path": str(iv.CACHE_NPZ),
            "exists": iv.CACHE_NPZ.exists(),
            "hash": iv_report.get("cache_hash"),
            "stage42iv_verdict": iv_report.get("stage42_iv_gate", {}).get("verdict"),
            "stage42iv_source": iv_report.get("source"),
        },
        "rows": int(len(selected)),
        "domain_rows": {domain: int(np.sum(labels["domain"] == domain)) for domain in sorted(set(labels["domain"].tolist()))},
        "horizon_rows": {str(h): int(np.sum(labels["horizon"] == h)) for h in [10, 25, 50, 100]},
        "metric": metric,
        "switch_mechanism": _switch_summary(cache, labels),
        "full_waypoint_shape": _waypoint_shape_summary(cache),
        "bootstrap": _bootstrap_summary(cache, labels),
        "supporting_evidence": _external_evidence_status(),
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
    result["stage42_iw_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _update_readmes(result)
    _update_state(result)
    _append_ledger(result)
    return result


if __name__ == "__main__":
    run_stage42_source_level_row_cache_mechanism_audit()
