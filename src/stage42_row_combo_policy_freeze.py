from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_row_prediction_cache as r
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "frozen_row_combo_policy_stage42.json"
REPORT_MD = OUT_DIR / "frozen_row_combo_policy_stage42.md"
POLICY_JSON = OUT_DIR / "frozen_row_combo_policy_stage42_policy.json"
GATE_MD = OUT_DIR / "stage42_stage_s_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-S 冻结 Stage42-R row-cache combo policy，并做 stress audit；不是 metric 或 seconds-level 结果。",
    "future waypoints / endpoints 只作为 train/val supervised labels 和 eval labels，不作为 inference input。",
    "combo source 只由 validation domain/horizon slice 选择，test 只最终评估一次。",
    "row prediction cache 是本地 derived cache，不提交 GitHub。",
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
    if payload.get("stage") == "Stage42-S frozen row combo policy and stress audit":
        return payload
    return None


def _cache_file_manifest(stage42r: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for row in stage42r.get("cache_rows", []):
        path = Path(str(row.get("cache_path", "")))
        manifest.append(
            {
                "pair_idx": int(row.get("pair_idx", -1)),
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
            }
        )
    return manifest


def _policy_payload(stage42r: Mapping[str, Any]) -> dict[str, Any]:
    per_pair = []
    majority: dict[str, dict[str, int]] = {}
    for row in stage42r.get("rows", []):
        choices = {}
        for key, choice in row.get("choices", {}).items():
            source = str(choice.get("selected_source", "floor"))
            choices[key] = {
                "selected_source": source,
                "val_score": float(choice.get("val_score", 0.0)),
                "val_rows": int(choice.get("val_metric", {}).get("rows", 0)),
                "easy_degradation": float(choice.get("val_metric", {}).get("easy_degradation", 0.0)),
            }
            majority.setdefault(key, {})[source] = majority.setdefault(key, {}).get(source, 0) + 1
        per_pair.append(
            {
                "pair_idx": int(row.get("pair_idx", -1)),
                "j_seed": int(row.get("j_seed", -1)),
                "p_seed": int(row.get("p_seed", -1)),
                "base_seed": int(row.get("base_seed", -1)),
                "choices": choices,
            }
        )
    majority_policy = {
        key: {
            "selected_source": sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0],
            "votes": counts,
        }
        for key, counts in sorted(majority.items())
        if counts
    }
    return {
        "policy_name": "stage42_s_frozen_row_combo_policy",
        "source": "fresh_run_from_stage42r_row_cache",
        "base_stage": "Stage42-R",
        "selection_scope": "validation_domain_horizon_slice_only",
        "test_usage": "test_once_after_policy_freeze",
        "candidate_sources": ["floor", "stage42j_static_expert", "stage42p_t50_gain_harm"],
        "per_pair_policy": per_pair,
        "majority_policy_diagnostic": majority_policy,
        "no_leakage": stage42r.get("no_leakage", {}),
        "claim_boundary": stage42r.get("claim_boundary", {}),
    }


def _metric_summary(vals: list[float]) -> dict[str, float]:
    return r._stat(vals)


def _stress_from_runtime(rows_runtime: list[Mapping[str, Any]], labels_test: Mapping[str, np.ndarray]) -> dict[str, Any]:
    domains = sorted(set(labels_test["domain"].astype(str).tolist()))
    horizons = [10, 25, 50, 100]
    stress: dict[str, Any] = {"by_domain": {}, "by_horizon": {}, "by_domain_horizon": {}}
    for domain in domains:
        mask = labels_test["domain"].astype(str) == domain
        stress["by_domain"][domain] = _slice_stats(rows_runtime, labels_test, mask)
    for horizon in horizons:
        mask = labels_test["horizon"].astype(int) == horizon
        stress["by_horizon"][str(horizon)] = _slice_stats(rows_runtime, labels_test, mask)
    for domain in domains:
        for horizon in horizons:
            mask = (labels_test["domain"].astype(str) == domain) & (labels_test["horizon"].astype(int) == horizon)
            stress["by_domain_horizon"][f"{domain}|{horizon}"] = _slice_stats(rows_runtime, labels_test, mask)
    return stress


def _slice_stats(rows_runtime: list[Mapping[str, Any]], labels_test: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    rows = int(np.sum(mask))
    if rows == 0:
        return {"rows": 0, "source": "not_run_empty_slice"}
    ade_metrics = []
    fde_metrics = []
    for row in rows_runtime:
        arr = row["arrays_for_bootstrap"]
        switch = arr["combo_test_switch"].astype(bool)
        ade_metrics.append(r._local_metric_from_errors(arr["combo_test_ade"], arr["floor_test_ade"], labels_test, switch, mask))
        fde_metrics.append(r._local_metric_from_errors(arr["combo_test_fde"], arr["floor_test_fde"], labels_test, switch, mask))
    return {
        "rows": rows,
        "source": "fresh_run_from_row_cache",
        "ade_all": _metric_summary([m.get("all_improvement", 0.0) for m in ade_metrics]),
        "ade_t50": _metric_summary([m.get("t50_improvement", 0.0) for m in ade_metrics]),
        "ade_t100_raw_frame_diagnostic": _metric_summary([m.get("t100_improvement", 0.0) for m in ade_metrics]),
        "ade_hard_failure": _metric_summary([m.get("hard_failure_improvement", 0.0) for m in ade_metrics]),
        "ade_easy_degradation": _metric_summary([m.get("easy_degradation", 0.0) for m in ade_metrics]),
        "fde_all": _metric_summary([m.get("all_improvement", 0.0) for m in fde_metrics]),
        "fde_t50": _metric_summary([m.get("t50_improvement", 0.0) for m in fde_metrics]),
        "switch_rate": _metric_summary([m.get("switch_rate", 0.0) for m in ade_metrics]),
    }


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    stress = result.get("stress", {})
    by_domain = stress.get("by_domain", {})
    positive_domains = [
        name
        for name, item in by_domain.items()
        if item.get("rows", 0) > 0 and item.get("ade_all", {}).get("mean", 0.0) > 0.0
    ]
    t50_positive_domains = [
        name
        for name, item in by_domain.items()
        if item.get("rows", 0) > 0 and item.get("ade_t50", {}).get("mean", 0.0) > 0.0
    ]
    gates = {
        "stage42r_report_verified": result.get("stage42r_verdict") == "stage42_r_row_cached_combo_pass",
        "row_cache_available": all(item.get("exists") for item in result.get("cache_manifest", [])),
        "policy_artifact_written": POLICY_JSON.exists(),
        "policy_hash_recorded": bool(result.get("policy_hash")),
        "validation_only_policy": result.get("policy", {}).get("selection_scope") == "validation_domain_horizon_slice_only",
        "stress_complete": bool(stress.get("by_domain")) and bool(stress.get("by_horizon")) and bool(stress.get("by_domain_horizon")),
        "positive_on_at_least_two_domains": len(positive_domains) >= 2,
        "t50_positive_on_at_least_two_domains": len(t50_positive_domains) >= 2,
        "easy_preserved": result.get("summary", {}).get("ade_easy_degradation", {}).get("mean", 1.0) <= 0.02,
        "no_leakage_pass": result.get("no_leakage", {}).get("future_endpoint_input") is False
        and result.get("no_leakage", {}).get("future_waypoints_input") is False
        and result.get("no_leakage", {}).get("central_velocity") is False
        and result.get("no_leakage", {}).get("test_endpoint_goals") is False
        and result.get("no_leakage", {}).get("test_threshold_tuning") is False,
        "no_metric_seconds_overclaim": result.get("claim_boundary", {}).get("metric_or_seconds_claim") is False,
        "stage5c_false": result.get("claim_boundary", {}).get("stage5c_executed") is False,
        "smc_false": result.get("claim_boundary", {}).get("smc_enabled") is False,
    }
    return {
        "source": "fresh_run_from_stage42r_row_cache",
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": len(gates),
        "positive_domains": positive_domains,
        "t50_positive_domains": t50_positive_domains,
        "verdict": "stage42_s_frozen_row_combo_policy_pass" if all(gates.values()) else "stage42_s_frozen_row_combo_policy_partial",
    }


def run_stage42_row_combo_policy_freeze() -> dict[str, Any]:
    cached = _cached_result_if_available()
    if cached is not None:
        return cached
    ensure_dir(OUT_DIR)
    stage42r = r.run_stage42_row_prediction_cache()
    data_val = r.s42i._split_arrays("val")
    data_test = r.s42i._split_arrays("test")
    labels_val = r._labels_for(data_val)
    labels_test = r._labels_for(data_test)
    rows_runtime = []
    for row in stage42r.get("cache_rows", []):
        path = Path(str(row.get("cache_path", "")))
        if path.exists():
            rows_runtime.append(r._eval_pair_cache(path, labels_val, labels_test))
    policy = _policy_payload(stage42r)
    write_json(POLICY_JSON, _jsonable(policy))
    cache_manifest = _cache_file_manifest(stage42r)
    summary = stage42r.get("summary", {})
    result = {
        "source": "fresh_run_from_stage42r_row_cache",
        "stage": "Stage42-S frozen row combo policy and stress audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "stage42r_report": str(r.REPORT_MD),
        "stage42r_verdict": stage42r.get("stage42_r_gate", {}).get("verdict"),
        "stage42r_input_hash": stage42r.get("input_hash"),
        "policy": policy,
        "policy_hash": _combined_hash([POLICY_JSON]),
        "cache_manifest": cache_manifest,
        "cache_hash": _combined_hash([item["path"] for item in cache_manifest if item.get("exists")]),
        "feature_schema_hash": _combined_hash([r.REPORT_JSON, POLICY_JSON]),
        "summary": summary,
        "stress": _stress_from_runtime(rows_runtime, labels_test) if rows_runtime else {"source": "not_run_missing_row_cache"},
        "no_leakage": stage42r.get("no_leakage", {}),
        "claim_boundary": stage42r.get("claim_boundary", {}),
    }
    result["stage42_s_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    _write_report(result)
    _write_gate(result["stage42_s_gate"])
    _append_readme_and_state(result)
    _append_ledger(result)
    return result


def _write_report(result: Mapping[str, Any]) -> None:
    gate = result["stage42_s_gate"]
    s = result.get("summary", {})
    lines = [
        "# Stage42-S Frozen Row Combo Policy + Stress Audit",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- stage42r_verdict: `{result['stage42r_verdict']}`",
        f"- policy_hash: `{result['policy_hash']}`",
        f"- cache_hash: `{result['cache_hash']}`",
        f"- feature_schema_hash: `{result['feature_schema_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Frozen Policy Summary",
        "",
        f"- policy artifact: `{POLICY_JSON}`",
        f"- candidate sources: `{result['policy']['candidate_sources']}`",
        f"- positive domains: `{gate.get('positive_domains', [])}`",
        f"- t50 positive domains: `{gate.get('t50_positive_domains', [])}`",
        "",
        "## Core Metrics",
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
            "- Stage42-S freezes the Stage42-R row-cache combo policy as a lightweight policy artifact.",
            "- The policy is still protected and validation-selected; test labels are not used for source selection.",
            "- UCY remains fallback-only in the Stage42-R combo stress table, so this is stronger branch evidence but not a foundation-scale generalization claim.",
            "- All results remain dataset-local raw-frame 2.5D; no metric/seconds-level, Stage5C, or SMC claim is made.",
        ]
    )
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-S Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- positive_domains: `{gate.get('positive_domains', [])}`",
        f"- t50_positive_domains: `{gate.get('t50_positive_domains', [])}`",
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
    gate = result["stage42_s_gate"]
    s = result["summary"]
    block = f"""
## Stage42-S Frozen Row Combo Policy

```text
source = {result['source']}
verdict = {gate['verdict']}
gates = {gate['passed']} / {gate['total']}
policy_hash = {result['policy_hash']}
cache_hash = {result['cache_hash']}
ade_all = {s['ade_all']['mean']}
ade_t50 = {s['ade_t50']['mean']}
ade_t50_ci_low = {s['ade_t50']['ci_low']}
ade_hard_failure = {s['ade_hard_failure']['mean']}
ade_easy_degradation = {s['ade_easy_degradation']['mean']}
stage5c_executed = false
smc_enabled = false
```

Stage42-S freezes the Stage42-R row-cache combo into a lightweight policy artifact and reports per-domain/per-horizon stress. It remains dataset-local raw-frame 2.5D evidence and not a metric, seconds-level, Stage5C, or SMC result.
"""
    _append_if_missing(Path("README_RESULTS.md"), "## Stage42-S Frozen Row Combo Policy", block)
    _append_if_missing(Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md"), "## Stage42-S Frozen Row Combo Policy", block)
    _append_if_missing(Path("outputs/m3w_neural_v1/README_M3W_LONG_GOAL_SUMMARY_ZH.md"), "## Stage42-S Frozen Row Combo Policy", block)
    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_s_frozen_row_combo_policy"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_s_frozen_row_combo_policy"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "policy_artifact": str(POLICY_JSON),
        "gates_passed": gate["passed"],
        "gates_total": gate["total"],
        "verdict": gate["verdict"],
        "policy_hash": result["policy_hash"],
        "cache_hash": result["cache_hash"],
        "ade_all": s["ade_all"]["mean"],
        "ade_t50": s["ade_t50"]["mean"],
        "ade_t50_ci_low": s["ade_t50"]["ci_low"],
        "ade_hard_failure": s["ade_hard_failure"]["mean"],
        "ade_easy_degradation": s["ade_easy_degradation"]["mean"],
        "positive_domains": gate.get("positive_domains", []),
        "t50_positive_domains": gate.get("t50_positive_domains", []),
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
        "command": "run_stage42_freeze_row_combo_policy.py",
        "step": "stage42_s_frozen_row_combo_policy",
        "source": result["source"],
        "status": "success",
        "input_hash": _combined_hash([r.REPORT_JSON, POLICY_JSON]),
        "output_hash": _combined_hash([REPORT_JSON, REPORT_MD, GATE_MD, POLICY_JSON]),
        "git_commit": _git_commit(),
        "generated_at_utc": result.get("generated_at_utc"),
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_row_combo_policy_freeze()
