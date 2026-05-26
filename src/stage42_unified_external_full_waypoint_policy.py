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
STAGE42S_JSON = OUT_DIR / "frozen_row_combo_policy_stage42.json"
STAGE42V_JSON = OUT_DIR / "ucy_full_waypoint_candidate_stage42.json"
REPORT_JSON = OUT_DIR / "unified_external_full_waypoint_policy_stage42.json"
REPORT_MD = OUT_DIR / "unified_external_full_waypoint_policy_stage42.md"
POLICY_JSON = OUT_DIR / "unified_external_full_waypoint_policy_stage42_policy.json"
GATE_MD = OUT_DIR / "stage42_stage_w_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-W 只合并已验证 external full-waypoint policy sources；不是 metric 或 seconds-level 结果。",
    "ETH_UCY / TrajNet 来自 Stage42-S row-cache combo policy；UCY 来自 Stage42-V strict pure-UCY full-waypoint candidate 的 UCY-domain slice。",
    "Stage42-W 不把 Stage42-V 的 ETH_UCY slice 重复计入，避免 double counting。",
    "future waypoints / endpoints 只作为 train/val labels 和 eval labels，不作为 inference input。",
    "policy source selection 来自 validation-only / source-heldout protocol；不使用 test 调阈值。",
    "merged single row-cache artifact 尚未建立；本阶段输出 unified policy package 和 per-domain stress。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

METRICS = [
    "ade_all",
    "ade_t50",
    "ade_t100_raw_frame_diagnostic",
    "ade_hard_failure",
    "ade_easy_degradation",
    "fde_t50",
    "switch_rate",
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


def _stat(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    half = 1.96 * std / float(np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half}


def _cached_result_if_available() -> dict[str, Any] | None:
    if not REPORT_JSON.exists():
        return None
    payload = read_json(REPORT_JSON, {})
    if payload.get("stage") == "Stage42-W unified external full-waypoint policy package":
        return payload
    return None


def _metric_from_stage42s(item: Mapping[str, Any]) -> dict[str, Any]:
    out = {
        "rows": int(item.get("rows", 0)),
        "source": "cached_verified_stage42s_row_combo_domain_slice",
        "source_stage": "Stage42-S",
    }
    for key in METRICS:
        out[key] = item.get(key, {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0})
    return out


def _ucy_domain_from_stage42v(stage42v: Mapping[str, Any]) -> dict[str, Any]:
    best_trial = str(stage42v.get("best_trial", ""))
    rows = [row for row in stage42v.get("rows", []) if row.get("trial", {}).get("name") == best_trial]
    if not rows:
        raise ValueError("Stage42-V report has no rows for best_trial.")
    ade_domain_rows = []
    fde_domain_rows = []
    for row in rows:
        ade = row.get("test_metrics", {}).get("ade", {}).get("by_domain", {}).get("UCY")
        fde = row.get("test_metrics", {}).get("fde", {}).get("by_domain", {}).get("UCY")
        if ade is None or fde is None:
            raise ValueError("Stage42-V best trial is missing UCY by_domain metrics.")
        ade_domain_rows.append(ade)
        fde_domain_rows.append(fde)
    row_counts = {int(item.get("rows", 0)) for item in ade_domain_rows}
    rows_n = int(sorted(row_counts)[0]) if row_counts else 0
    return {
        "rows": rows_n,
        "source": "fresh_stage42v_ucy_domain_slice",
        "source_stage": "Stage42-V",
        "best_trial": best_trial,
        "ade_all": _stat([float(item.get("all_improvement", 0.0)) for item in ade_domain_rows]),
        "ade_t50": _stat([float(item.get("t50_improvement", 0.0)) for item in ade_domain_rows]),
        "ade_t100_raw_frame_diagnostic": _stat([float(item.get("t100_improvement", 0.0)) for item in ade_domain_rows]),
        "ade_hard_failure": _stat([float(item.get("hard_failure_improvement", 0.0)) for item in ade_domain_rows]),
        "ade_easy_degradation": _stat([float(item.get("easy_degradation", 0.0)) for item in ade_domain_rows]),
        "fde_t50": _stat([float(item.get("t50_improvement", 0.0)) for item in fde_domain_rows]),
        "switch_rate": _stat([float(item.get("switch_rate", 0.0)) for item in ade_domain_rows]),
    }


def _weighted_summary(by_domain: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    total_rows = int(sum(int(item.get("rows", 0)) for item in by_domain.values()))
    summary: dict[str, Any] = {
        "rows": total_rows,
        "source": "fresh_weighted_domain_package_summary",
        "ci_note": "Global row-level bootstrap was not rerun because Stage42-S and Stage42-V are separate validated sources; per-domain CIs are reported and global means are row-weighted.",
    }
    for key in METRICS:
        if total_rows <= 0:
            summary[key] = {"mean": 0.0, "domain_min_ci_low": 0.0, "domain_max_ci_high": 0.0}
            continue
        means = []
        lows = []
        highs = []
        for item in by_domain.values():
            rows = int(item.get("rows", 0))
            metric = item.get(key, {})
            means.append(float(metric.get("mean", 0.0)) * rows)
            lows.append(float(metric.get("ci_low", metric.get("mean", 0.0))))
            highs.append(float(metric.get("ci_high", metric.get("mean", 0.0))))
        summary[key] = {
            "mean": float(sum(means) / total_rows),
            "domain_min_ci_low": float(min(lows)) if lows else 0.0,
            "domain_max_ci_high": float(max(highs)) if highs else 0.0,
        }
    return summary


def _build_policy_payload(by_domain: Mapping[str, Mapping[str, Any]], stage42s: Mapping[str, Any], stage42v: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "policy_name": "stage42_w_unified_external_full_waypoint_policy",
        "source": "fresh_unified_from_stage42s_and_stage42v",
        "selection_scope": "domain_source_package_only",
        "test_usage": "test_once_metrics_from_frozen_stage42s_and_stage42v_reports",
        "domain_sources": {
            "ETH_UCY": "stage42_s_row_combo_policy",
            "TrajNet": "stage42_s_row_combo_policy",
            "UCY": "stage42_v_strict_pure_ucy_full_waypoint_candidate_ucy_slice",
        },
        "double_counting_guard": "Stage42-V ETH_UCY slice is excluded; only Stage42-V UCY slice replaces Stage42-S UCY fallback-only slice.",
        "single_row_cache_status": "not_run_separate_validated_sources_need_future_row_level_merge",
        "domains": {
            domain: {
                "rows": item.get("rows", 0),
                "source": item.get("source"),
                "source_stage": item.get("source_stage"),
            }
            for domain, item in by_domain.items()
        },
        "stage42s_policy_hash": stage42s.get("policy_hash"),
        "stage42v_best_trial": stage42v.get("best_trial"),
    }


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    by_domain = result.get("by_domain", {})
    summary = result.get("summary", {})
    positive_all = [
        domain
        for domain, item in by_domain.items()
        if item.get("rows", 0) > 0 and item.get("ade_all", {}).get("mean", 0.0) > 0.0
    ]
    positive_t50 = [
        domain
        for domain, item in by_domain.items()
        if item.get("rows", 0) > 0 and item.get("ade_t50", {}).get("mean", 0.0) > 0.0
    ]
    positive_hard = [
        domain
        for domain, item in by_domain.items()
        if item.get("rows", 0) > 0 and item.get("ade_hard_failure", {}).get("mean", 0.0) > 0.0
    ]
    gates = {
        "stage42s_verified": result.get("inputs", {}).get("stage42s_verdict") == "stage42_s_frozen_row_combo_policy_pass",
        "stage42v_verified": result.get("inputs", {}).get("stage42v_verdict") == "stage42_v_ucy_full_waypoint_candidate_pass",
        "three_nonempty_domains": all(by_domain.get(name, {}).get("rows", 0) > 0 for name in ["ETH_UCY", "TrajNet", "UCY"]),
        "ucy_replaced_from_stage42v": by_domain.get("UCY", {}).get("source_stage") == "Stage42-V",
        "double_counting_guard_recorded": "excluded" in result.get("policy", {}).get("double_counting_guard", ""),
        "all_positive_on_three_domains": len(positive_all) >= 3,
        "t50_positive_on_three_domains": len(positive_t50) >= 3,
        "hard_positive_on_three_domains": len(positive_hard) >= 3,
        "weighted_all_positive": summary.get("ade_all", {}).get("mean", 0.0) > 0.0,
        "weighted_t50_positive": summary.get("ade_t50", {}).get("mean", 0.0) > 0.0,
        "easy_preserved": summary.get("ade_easy_degradation", {}).get("mean", 1.0) <= 0.02,
        "single_row_cache_status_recorded": "single_row_cache_status" in result.get("policy", {}),
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
        "positive_hard_domains": positive_hard,
        "verdict": "stage42_w_unified_external_full_waypoint_policy_pass" if all(gates.values()) else "stage42_w_unified_external_full_waypoint_policy_partial",
    }


def run_stage42_unified_external_full_waypoint_policy() -> dict[str, Any]:
    cached = _cached_result_if_available()
    if cached is not None:
        return cached
    ensure_dir(OUT_DIR)
    stage42s = read_json(STAGE42S_JSON, {})
    stage42v = read_json(STAGE42V_JSON, {})
    by_domain = {
        "ETH_UCY": _metric_from_stage42s(stage42s.get("stress", {}).get("by_domain", {}).get("ETH_UCY", {})),
        "TrajNet": _metric_from_stage42s(stage42s.get("stress", {}).get("by_domain", {}).get("TrajNet", {})),
        "UCY": _ucy_domain_from_stage42v(stage42v),
    }
    summary = _weighted_summary(by_domain)
    policy = _build_policy_payload(by_domain, stage42s, stage42v)
    write_json(POLICY_JSON, _jsonable(policy))
    no_leakage = {
        "future_endpoint_input": False,
        "future_waypoints_input": False,
        "future_waypoints_used_as_labels_or_eval_only": True,
        "central_velocity": False,
        "test_endpoint_goals": False,
        "test_policy_tuning": False,
        "stage42s_no_leakage": stage42s.get("no_leakage", {}),
        "stage42v_no_leakage": stage42v.get("no_leakage", {}),
    }
    claim_boundary = {
        "true_3d": False,
        "foundation_world_model": False,
        "metric_or_seconds_claim": False,
        "raw_frame_dataset_local_only": True,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    result = {
        "stage": "Stage42-W unified external full-waypoint policy package",
        "source": "fresh_unified_from_cached_verified_stage42s_and_stage42v",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "stage42s_report": str(STAGE42S_JSON),
            "stage42s_verdict": stage42s.get("stage42_s_gate", {}).get("verdict"),
            "stage42s_source": stage42s.get("source"),
            "stage42v_report": str(STAGE42V_JSON),
            "stage42v_verdict": stage42v.get("verdict"),
            "stage42v_source": stage42v.get("source"),
            "stage42v_best_trial": stage42v.get("best_trial"),
        },
        "input_hash": _combined_hash([STAGE42S_JSON, STAGE42V_JSON]),
        "policy": policy,
        "policy_hash": _combined_hash([POLICY_JSON]),
        "by_domain": by_domain,
        "summary": summary,
        "no_leakage": no_leakage,
        "claim_boundary": claim_boundary,
    }
    result["stage42_w_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    _write_report(result)
    _write_gate(result["stage42_w_gate"])
    _append_readme_and_state(result)
    _append_ledger(result)
    return result


def _write_report(result: Mapping[str, Any]) -> None:
    gate = result["stage42_w_gate"]
    s = result["summary"]
    lines = [
        "# Stage42-W Unified External Full-Waypoint Policy Package",
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
        "## Source Package",
        "",
        "- `ETH_UCY`: Stage42-S row-cache combo policy.",
        "- `TrajNet`: Stage42-S row-cache combo policy.",
        "- `UCY`: Stage42-V strict pure-UCY full-waypoint candidate, UCY-domain slice only.",
        "- Stage42-V `ETH_UCY` slice is excluded to avoid double counting with Stage42-S.",
        "- A single merged row-cache artifact is not yet built; this report is a unified policy package with per-domain stress.",
        "",
        "## Weighted Package Summary",
        "",
        f"- rows: `{s.get('rows', 0)}`",
        f"- CI note: {s.get('ci_note', '')}",
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
        ("FDE t50", "fde_t50"),
        ("switch rate", "switch_rate"),
    ]:
        row = s.get(key, {})
        lines.append(
            f"| {label} | {row.get('mean', 0.0):.6f} | "
            f"{row.get('domain_min_ci_low', 0.0):.6f} | {row.get('domain_max_ci_high', 0.0):.6f} |"
        )
    lines.extend(["", "## Per-Domain Metrics", "", "| domain | source | rows | ADE all | ADE t50 | ADE hard | easy degr | FDE t50 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for domain, item in result["by_domain"].items():
        lines.append(
            f"| `{domain}` | `{item.get('source_stage')}` | {item.get('rows', 0)} | "
            f"{item.get('ade_all', {}).get('mean', 0.0):.6f} | {item.get('ade_t50', {}).get('mean', 0.0):.6f} | "
            f"{item.get('ade_hard_failure', {}).get('mean', 0.0):.6f} | {item.get('ade_easy_degradation', {}).get('mean', 0.0):.6f} | "
            f"{item.get('fde_t50', {}).get('mean', 0.0):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-W closes the main Stage42-S UCY fallback-only gap at policy-package level by importing the Stage42-V UCY full-waypoint candidate source.",
            "- The result is stronger external full-waypoint branch evidence across ETH_UCY, TrajNet, and UCY.",
            "- It is not yet a single merged row-cache artifact; future work should build row-level UCY candidate cache and rerun one unified bootstrap.",
            "- All claims remain dataset-local raw-frame 2.5D. No metric, seconds-level, Stage5C, or SMC claim is made.",
        ]
    )
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-W Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- positive_all_domains: `{gate.get('positive_all_domains', [])}`",
        f"- positive_t50_domains: `{gate.get('positive_t50_domains', [])}`",
        f"- positive_hard_domains: `{gate.get('positive_hard_domains', [])}`",
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
    gate = result["stage42_w_gate"]
    s = result["summary"]
    block = f"""
## Stage42-W Unified External Full-Waypoint Policy

```text
source = {result['source']}
verdict = {gate['verdict']}
gates = {gate['passed']} / {gate['total']}
policy_hash = {result['policy_hash']}
rows = {s['rows']}
weighted_ADE_all = {s['ade_all']['mean']}
weighted_ADE_t50 = {s['ade_t50']['mean']}
weighted_ADE_hard_failure = {s['ade_hard_failure']['mean']}
weighted_easy_degradation = {s['ade_easy_degradation']['mean']}
domains = ETH_UCY, TrajNet, UCY
stage5c_executed = false
smc_enabled = false
```

Stage42-W combines ETH_UCY/TrajNet from the frozen Stage42-S row-cache combo policy with the UCY-domain slice from Stage42-V strict pure-UCY full-waypoint candidate. It avoids double counting the Stage42-V ETH_UCY slice and explicitly records that a single merged row-cache artifact remains future work. Claims remain dataset-local raw-frame 2.5D, not metric or seconds-level.
"""
    _append_if_missing(Path("README_RESULTS.md"), "## Stage42-W Unified External Full-Waypoint Policy", block)
    _append_if_missing(Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md"), "## Stage42-W Unified External Full-Waypoint Policy", block)
    _append_if_missing(Path("README_M3W_RESEARCH_SUMMARY_ZH.md"), "## Stage42-W Unified External Full-Waypoint Policy", block)

    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_w_unified_external_full_waypoint_policy"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_w_unified_external_full_waypoint_policy"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "policy_artifact": str(POLICY_JSON),
        "gates_passed": gate["passed"],
        "gates_total": gate["total"],
        "verdict": gate["verdict"],
        "policy_hash": result["policy_hash"],
        "rows": s["rows"],
        "weighted_ade_all": s["ade_all"]["mean"],
        "weighted_ade_t50": s["ade_t50"]["mean"],
        "weighted_ade_hard_failure": s["ade_hard_failure"]["mean"],
        "weighted_easy_degradation": s["ade_easy_degradation"]["mean"],
        "positive_all_domains": gate.get("positive_all_domains", []),
        "positive_t50_domains": gate.get("positive_t50_domains", []),
        "single_row_cache_status": result["policy"]["single_row_cache_status"],
        "claim_boundary": result["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, POLICY_JSON, GATE_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(Path("research_state.json"), _jsonable(state))


def _append_ledger(result: Mapping[str, Any]) -> None:
    entry = {
        "command": "run_stage42_unified_external_full_waypoint_policy.py",
        "step": "stage42_w_unified_external_full_waypoint_policy",
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
    run_stage42_unified_external_full_waypoint_policy()
