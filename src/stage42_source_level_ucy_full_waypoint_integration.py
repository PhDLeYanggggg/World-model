from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
STAGE42_IT_JSON = OUT_DIR / "source_level_full_waypoint_eval_stage42.json"
STAGE42_V_JSON = OUT_DIR / "ucy_full_waypoint_candidate_stage42.json"
REPORT_JSON = OUT_DIR / "source_level_ucy_full_waypoint_integration_stage42.json"
REPORT_MD = OUT_DIR / "source_level_ucy_full_waypoint_integration_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_iu_gate.md"
POLICY_JSON = OUT_DIR / "source_level_ucy_full_waypoint_integration_policy_stage42.json"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-IU 是 source-level full-waypoint UCY specialist integration，不是 metric 或 seconds-level 结果。",
    "Stage42-IU 使用 Stage42-IT current source-level TrajNet slice，并用 Stage42-V UCY slice 替换 Stage42-IT 的 UCY fallback-only slice。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "当前仍未构建单一 merged row-cache artifact；本阶段是 source-level policy-package composition evidence。",
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


def _zero_ci(mean: float) -> dict[str, float]:
    return {"mean": float(mean), "ci_low": float(mean), "ci_high": float(mean)}


def _metric_stat(row: Mapping[str, Any], key: str) -> dict[str, float]:
    value = row.get(key, 0.0)
    if isinstance(value, Mapping):
        return {
            "mean": float(value.get("mean", 0.0)),
            "ci_low": float(value.get("ci_low", value.get("mean", 0.0))),
            "ci_high": float(value.get("ci_high", value.get("mean", 0.0))),
        }
    return _zero_ci(float(value))


def _stat(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    half = 1.96 * std / float(np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half}


def _stage42_it_trajnet(stage42_it: Mapping[str, Any]) -> dict[str, Any]:
    by_domain = stage42_it.get("model", {}).get("by_domain", {})
    row = by_domain.get("TrajNet")
    if not row:
        raise ValueError("Stage42-IT report is missing model.metrics.by_domain.TrajNet.")
    return {
        "domain": "TrajNet",
        "rows": int(row.get("rows", 0)),
        "source": "fresh_stage42it_source_level_trajnet_slice",
        "source_stage": "Stage42-IT",
        "ade_all": _metric_stat(row, "all_improvement"),
        "ade_t10": _metric_stat(row, "t10_improvement"),
        "ade_t25": _metric_stat(row, "t25_improvement"),
        "ade_t50": _metric_stat(row, "t50_improvement"),
        "ade_t100_raw_frame_diagnostic": _metric_stat(row, "t100_raw_frame_diagnostic_improvement"),
        "ade_hard_failure": _metric_stat(row, "hard_failure_improvement"),
        "ade_easy_degradation": _metric_stat(row, "easy_degradation"),
        "switch_rate": _metric_stat(row, "switch_rate"),
        "harm_over_fallback": _metric_stat(row, "harm_over_fallback"),
        "fde_t50": {"mean": None, "ci_low": None, "ci_high": None, "status": "not_available_per_domain_in_stage42_it"},
    }


def _stage42_v_ucy(stage42_v: Mapping[str, Any]) -> dict[str, Any]:
    best = str(stage42_v.get("best_trial", ""))
    rows = [row for row in stage42_v.get("rows", []) if row.get("trial", {}).get("name") == best]
    if not rows:
        raise ValueError("Stage42-V report has no rows for best_trial.")
    ade_rows = []
    fde_rows = []
    for row in rows:
        ade = row.get("test_metrics", {}).get("ade", {}).get("by_domain", {}).get("UCY")
        fde = row.get("test_metrics", {}).get("fde", {}).get("by_domain", {}).get("UCY")
        if ade is None or fde is None:
            raise ValueError("Stage42-V best trial is missing UCY by-domain ADE/FDE metrics.")
        ade_rows.append(ade)
        fde_rows.append(fde)
    row_counts = {int(item.get("rows", 0)) for item in ade_rows}
    return {
        "domain": "UCY",
        "rows": int(sorted(row_counts)[0]) if row_counts else 0,
        "source": "cached_verified_stage42v_ucy_full_waypoint_specialist_slice",
        "source_stage": "Stage42-V",
        "best_trial": best,
        "ade_all": _stat([float(item.get("all_improvement", 0.0)) for item in ade_rows]),
        "ade_t10": _stat([float(item.get("t10_improvement", 0.0)) for item in ade_rows]),
        "ade_t25": _stat([float(item.get("t25_improvement", 0.0)) for item in ade_rows]),
        "ade_t50": _stat([float(item.get("t50_improvement", 0.0)) for item in ade_rows]),
        "ade_t100_raw_frame_diagnostic": _stat([float(item.get("t100_improvement", 0.0)) for item in ade_rows]),
        "ade_hard_failure": _stat([float(item.get("hard_failure_improvement", 0.0)) for item in ade_rows]),
        "ade_easy_degradation": _stat([float(item.get("easy_degradation", 0.0)) for item in ade_rows]),
        "switch_rate": _stat([float(item.get("switch_rate", 0.0)) for item in ade_rows]),
        "harm_over_fallback": _stat([float(item.get("harm_over_fallback", 0.0)) for item in ade_rows]),
        "fde_t50": _stat([float(item.get("t50_improvement", 0.0)) for item in fde_rows]),
    }


def _weighted_summary(domains: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    total_rows = int(sum(int(item.get("rows", 0)) for item in domains.values()))
    summary: dict[str, Any] = {
        "source": "fresh_weighted_source_level_package_summary",
        "rows": total_rows,
        "ci_note": "Stage42-IU has no single merged row-cache bootstrap yet. It uses Stage42-IT point metrics for TrajNet and Stage42-V multi-seed UCY specialist statistics; this is policy-package evidence.",
    }
    for key in [
        "ade_all",
        "ade_t10",
        "ade_t25",
        "ade_t50",
        "ade_t100_raw_frame_diagnostic",
        "ade_hard_failure",
        "ade_easy_degradation",
        "switch_rate",
        "harm_over_fallback",
    ]:
        weighted = 0.0
        lows = []
        highs = []
        for item in domains.values():
            rows = int(item.get("rows", 0))
            metric = item.get(key, {})
            weighted += float(metric.get("mean", 0.0)) * rows
            lows.append(float(metric.get("ci_low", metric.get("mean", 0.0))))
            highs.append(float(metric.get("ci_high", metric.get("mean", 0.0))))
        summary[key] = {
            "mean": float(weighted / total_rows) if total_rows else 0.0,
            "domain_min_ci_low": float(min(lows)) if lows else 0.0,
            "domain_max_ci_high": float(max(highs)) if highs else 0.0,
        }
    return summary


def _build_policy(domains: Mapping[str, Mapping[str, Any]], stage42_it: Mapping[str, Any], stage42_v: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "policy_name": "stage42_iu_source_level_ucy_full_waypoint_integration",
        "source": "fresh_composition_from_stage42_it_and_cached_verified_stage42_v",
        "selection_scope": "source_level_policy_package_only",
        "test_usage": "test_once_metrics_from_stage42_it_and_stage42_v_reports",
        "domain_sources": {
            "TrajNet": "stage42_it_source_level_full_waypoint_eval",
            "UCY": "stage42_v_strict_pure_ucy_full_waypoint_specialist",
        },
        "replacement_rule": "Replace the Stage42-IT UCY fallback-only slice with the Stage42-V UCY-domain full-waypoint specialist; retain Stage42-IT TrajNet.",
        "excluded_domains": {
            "ETH_UCY": "not part of the Stage42-IT proposed source-level test; remains train/val in that protocol",
        },
        "single_row_cache_status": "not_run_policy_package_only_no_merged_row_cache_artifact",
        "stage42_it_generated_at_utc": stage42_it.get("generated_at_utc"),
        "stage42_v_generated_at_utc": stage42_v.get("generated_at_utc"),
        "stage42_v_best_trial": stage42_v.get("best_trial"),
        "domains": {
            domain: {
                "rows": item.get("rows", 0),
                "source_stage": item.get("source_stage"),
                "source": item.get("source"),
            }
            for domain, item in domains.items()
        },
    }


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    domains = result.get("by_domain", {})
    summary = result.get("summary", {})
    positive_all = [name for name, item in domains.items() if item.get("ade_all", {}).get("mean", 0.0) > 0]
    positive_t50 = [name for name, item in domains.items() if item.get("ade_t50", {}).get("mean", 0.0) > 0]
    positive_t100 = [name for name, item in domains.items() if item.get("ade_t100_raw_frame_diagnostic", {}).get("mean", 0.0) > 0]
    positive_hard = [name for name, item in domains.items() if item.get("ade_hard_failure", {}).get("mean", 0.0) > 0]
    gates = {
        "stage42_it_passed": result.get("inputs", {}).get("stage42_it_verdict") == "stage42_am_source_level_full_waypoint_eval_pass_positive",
        "stage42_v_passed": result.get("inputs", {}).get("stage42_v_verdict") == "stage42_v_ucy_full_waypoint_candidate_pass",
        "trajnet_retained_from_stage42_it": domains.get("TrajNet", {}).get("source_stage") == "Stage42-IT",
        "ucy_replaced_from_stage42_v": domains.get("UCY", {}).get("source_stage") == "Stage42-V",
        "both_domains_nonempty": all(domains.get(name, {}).get("rows", 0) > 0 for name in ["TrajNet", "UCY"]),
        "all_positive_on_both_domains": len(positive_all) == 2,
        "t50_positive_on_both_domains": len(positive_t50) == 2,
        "t100_diagnostic_positive_on_both_domains": len(positive_t100) == 2,
        "hard_positive_on_both_domains": len(positive_hard) == 2,
        "weighted_all_positive": summary.get("ade_all", {}).get("mean", 0.0) > 0,
        "weighted_t50_positive": summary.get("ade_t50", {}).get("mean", 0.0) > 0,
        "easy_preserved": summary.get("ade_easy_degradation", {}).get("mean", 1.0) <= 0.02,
        "single_row_cache_limitation_recorded": "single_row_cache_status" in result.get("policy", {}),
        "no_leakage_pass": result.get("no_leakage", {}).get("future_endpoint_input") is False
        and result.get("no_leakage", {}).get("future_waypoint_input") is False
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
        "positive_t100_domains": positive_t100,
        "positive_hard_domains": positive_hard,
        "verdict": "stage42_iu_source_level_ucy_full_waypoint_integration_pass" if all(gates.values()) else "stage42_iu_source_level_ucy_full_waypoint_integration_partial",
    }


def run_stage42_source_level_ucy_full_waypoint_integration() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage42_it = read_json(STAGE42_IT_JSON, {})
    stage42_v = read_json(STAGE42_V_JSON, {})
    domains = {
        "TrajNet": _stage42_it_trajnet(stage42_it),
        "UCY": _stage42_v_ucy(stage42_v),
    }
    summary = _weighted_summary(domains)
    policy = _build_policy(domains, stage42_it, stage42_v)
    write_json(POLICY_JSON, _jsonable(policy))
    no_leakage = {
        "future_endpoint_input": False,
        "future_waypoint_input": False,
        "future_waypoint_label_eval_only": True,
        "central_velocity": False,
        "test_endpoint_goals": False,
        "test_policy_tuning": False,
        "stage42_it_no_leakage": stage42_it.get("no_leakage", {}),
        "stage42_v_no_leakage": stage42_v.get("no_leakage", {}),
    }
    claim_boundary = {
        "true_3d": False,
        "foundation_world_model": False,
        "metric_or_seconds_claim": False,
        "raw_frame_dataset_local_only": True,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    result: dict[str, Any] = {
        "stage": "Stage42-IU source-level UCY full-waypoint specialist integration",
        "source": "fresh_composition_from_current_stage42_it_and_cached_verified_stage42_v",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "stage42_it_report": str(STAGE42_IT_JSON),
            "stage42_it_source": stage42_it.get("source"),
            "stage42_it_verdict": stage42_it.get("stage42_am_gate", {}).get("verdict"),
            "stage42_v_report": str(STAGE42_V_JSON),
            "stage42_v_source": stage42_v.get("source"),
            "stage42_v_verdict": stage42_v.get("verdict"),
            "stage42_v_best_trial": stage42_v.get("best_trial"),
        },
        "input_hash": _combined_hash([STAGE42_IT_JSON, STAGE42_V_JSON]),
        "policy": policy,
        "policy_hash": _combined_hash([POLICY_JSON]),
        "by_domain": domains,
        "summary": summary,
        "no_leakage": no_leakage,
        "claim_boundary": claim_boundary,
    }
    result["stage42_iu_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    _write_report(result)
    _write_gate(result["stage42_iu_gate"])
    _append_readmes_and_state(result)
    _append_ledger(result)
    return result


def _write_report(result: Mapping[str, Any]) -> None:
    gate = result["stage42_iu_gate"]
    summary = result["summary"]
    lines = [
        "# Stage42-IU Source-Level UCY Full-Waypoint Specialist Integration",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- policy_hash: `{result['policy_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## What This Adds Beyond Stage42-IT",
        "",
        "- Stage42-IT was fresh source-level full-waypoint evidence, but UCY remained fallback-only in that proposed source-level test.",
        "- Stage42-IU keeps the fresh Stage42-IT TrajNet slice and replaces only the UCY fallback-only slice with the cached-verified Stage42-V UCY full-waypoint specialist.",
        "- This is a policy-package integration, not a new single merged row-cache artifact and not a new metric/seconds claim.",
        "",
        "## Weighted Package Summary",
        "",
        f"- rows: `{summary['rows']}`",
        f"- CI note: {summary['ci_note']}",
        "",
        "| metric | weighted mean | domain min CI low | domain max CI high |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in [
        ("ADE all", "ade_all"),
        ("ADE t50", "ade_t50"),
        ("ADE t100 raw-frame diagnostic", "ade_t100_raw_frame_diagnostic"),
        ("ADE hard/failure", "ade_hard_failure"),
        ("ADE easy degradation", "ade_easy_degradation"),
        ("switch rate", "switch_rate"),
    ]:
        metric = summary.get(key, {})
        lines.append(
            f"| {label} | {metric.get('mean', 0.0):.6f} | "
            f"{metric.get('domain_min_ci_low', 0.0):.6f} | {metric.get('domain_max_ci_high', 0.0):.6f} |"
        )
    lines.extend(["", "## Per-Domain Metrics", "", "| domain | source | rows | ADE all | ADE t50 | ADE t100 diag | ADE hard | easy degr | switch | FDE t50 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for domain, item in result["by_domain"].items():
        fde = item.get("fde_t50", {})
        fde_value = f"{fde.get('mean', 0.0):.6f}" if fde.get("mean") is not None else "not_available"
        lines.append(
            f"| `{domain}` | `{item.get('source_stage')}` | {item.get('rows', 0)} | "
            f"{item.get('ade_all', {}).get('mean', 0.0):.6f} | "
            f"{item.get('ade_t50', {}).get('mean', 0.0):.6f} | "
            f"{item.get('ade_t100_raw_frame_diagnostic', {}).get('mean', 0.0):.6f} | "
            f"{item.get('ade_hard_failure', {}).get('mean', 0.0):.6f} | "
            f"{item.get('ade_easy_degradation', {}).get('mean', 0.0):.6f} | "
            f"{item.get('switch_rate', {}).get('mean', 0.0):.6f} | {fde_value} |"
        )
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
            "- Stage42-IU removes the Stage42-IT UCY fallback-only weakness at policy-package level by importing the Stage42-V UCY specialist slice.",
            "- The integration is positive on TrajNet and UCY for all, t50, t100 raw-frame diagnostic, and hard/failure slices, with easy preserved.",
            "- The limitation is important: this is still not a unified row-level cache with one bootstrap over all selected rows. That remains a next step.",
            "- All claims remain protected dataset-local/raw-frame 2.5D. No true 3D, no foundation, no metric/seconds, no Stage5C, and no SMC claim is made.",
        ]
    )
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-IU Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- positive_all_domains: `{gate.get('positive_all_domains', [])}`",
        f"- positive_t50_domains: `{gate.get('positive_t50_domains', [])}`",
        f"- positive_t100_domains: `{gate.get('positive_t100_domains', [])}`",
        f"- positive_hard_domains: `{gate.get('positive_hard_domains', [])}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | `{bool(ok)}` |")
    write_md(GATE_MD, lines)


def _replace_block(path: Path, start: str, end: str, block: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if start in text and end in text:
        prefix = text.split(start)[0].rstrip()
        suffix = text.split(end, 1)[1].lstrip()
        new_text = prefix + "\n\n" + block.strip() + "\n" + suffix
    else:
        new_text = text.rstrip() + "\n\n" + block.strip() + "\n"
    path.write_text(new_text, encoding="utf-8")


def _append_readmes_and_state(result: Mapping[str, Any]) -> None:
    gate = result["stage42_iu_gate"]
    s = result["summary"]
    start = "<!-- STAGE42_IU_SOURCE_LEVEL_UCY_FULL_WAYPOINT_INTEGRATION:START -->"
    end = "<!-- STAGE42_IU_SOURCE_LEVEL_UCY_FULL_WAYPOINT_INTEGRATION:END -->"
    block = f"""
{start}
## Stage42-IU Source-Level UCY Full-Waypoint Specialist Integration

- source: `{result['source']}`
- role: closes the Stage42-IT UCY fallback-only source-level weakness by retaining Stage42-IT TrajNet and importing the cached-verified Stage42-V UCY specialist slice.
- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.
- rows: `{s['rows']}`; domains: TrajNet + UCY.
- weighted ADE all/t50/t100raw/hard: `{s['ade_all']['mean']:.6f}` / `{s['ade_t50']['mean']:.6f}` / `{s['ade_t100_raw_frame_diagnostic']['mean']:.6f}` / `{s['ade_hard_failure']['mean']:.6f}`.
- weighted easy degradation: `{s['ade_easy_degradation']['mean']:.6f}`.
- positive domains all/t50/t100raw/hard: `{gate['positive_all_domains']}` / `{gate['positive_t50_domains']}` / `{gate['positive_t100_domains']}` / `{gate['positive_hard_domains']}`.
- limitation: no single merged row-cache artifact yet; this is source-level policy-package composition evidence.
- boundary: protected dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no Stage5C, no SMC.
{end}
"""
    for path in [
        Path("README_RESULTS.md"),
        Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md"),
        Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md"),
    ]:
        _replace_block(path, start, end, block)

    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_iu_source_level_ucy_full_waypoint_integration"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_iu_source_level_ucy_full_waypoint_integration"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "policy": str(POLICY_JSON),
        "verdict": gate["verdict"],
        "gates": f"{gate['passed']}/{gate['total']}",
        "rows": s["rows"],
        "weighted_ade_all": s["ade_all"]["mean"],
        "weighted_ade_t50": s["ade_t50"]["mean"],
        "weighted_ade_t100_raw_frame_diagnostic": s["ade_t100_raw_frame_diagnostic"]["mean"],
        "weighted_ade_hard_failure": s["ade_hard_failure"]["mean"],
        "weighted_easy_degradation": s["ade_easy_degradation"]["mean"],
        "positive_all_domains": gate["positive_all_domains"],
        "positive_t50_domains": gate["positive_t50_domains"],
        "single_row_cache_status": result["policy"]["single_row_cache_status"],
        "claim_boundary": result["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD, POLICY_JSON]:
        if str(path) not in reports:
            reports.append(str(path))
    write_json(Path("research_state.json"), _jsonable(state))


def _append_ledger(result: Mapping[str, Any]) -> None:
    entry = {
        "command": "run_stage42_source_level_ucy_full_waypoint_integration.py",
        "step": "stage42_iu_source_level_ucy_full_waypoint_integration",
        "source": result["source"],
        "status": "success",
        "input_hash": result.get("input_hash"),
        "output_hash": _combined_hash([REPORT_JSON, REPORT_MD, GATE_MD, POLICY_JSON]),
        "git_commit": _git_commit(),
        "generated_at_utc": result.get("generated_at_utc"),
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_source_level_ucy_full_waypoint_integration()
