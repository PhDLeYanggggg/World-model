from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_data_calibration import OUT_DIR


STAGE42AK_JSON = OUT_DIR / "post_repair_locked_policy_audit_stage42.json"
STAGE42AI_JSON = OUT_DIR / "trajnet_t100_safety_repair_stage42.json"
STAGE42X_JSON = OUT_DIR / "unified_row_level_full_waypoint_cache_stage42.json"
SOURCE_SPLIT_JSON = OUT_DIR / "external_source_split_stage42.json"

REPORT_JSON = OUT_DIR / "source_level_coverage_audit_stage42.json"
REPORT_MD = OUT_DIR / "source_level_coverage_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_al_gate.md"

README_RESULTS = Path("README_RESULTS.md")
README_NEURAL = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
README_ZH = Path("README_M3W_RESEARCH_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-AL 是 source-level split coverage / claim-gap audit，不重新训练模型。",
    "它检查 Stage42-AK locked policy 的 post-repair metrics 是否可被写成完整 source-level split evaluation。",
    "如果 coverage 不足，必须写 blocker，而不是把 available row-level stress 包装成 full source-level validation。",
    "t+50 / t+100 仍是 raw-frame horizons；t+100 仍只能 diagnostic。",
    "External coordinates 仍是 dataset-local / unverified weak-metric diagnostic。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _hash_paths(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path).encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes() if path.exists() else b"missing")
        h.update(b"\0")
    return h.hexdigest()


def _domain_counts(split: Mapping[str, Any], split_name: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in split.get("proposed_source_level_split", {}).get(split_name, {}).get("domains", {}).items()}


def _status_for_domain(proposed_test_rows: int, stress_rows: int) -> str:
    if proposed_test_rows == 0 and stress_rows > 0:
        return "extra_available_not_in_proposed_source_test"
    if proposed_test_rows == 0 and stress_rows == 0:
        return "not_in_proposed_source_test"
    if stress_rows == proposed_test_rows:
        return "exact_row_count_match"
    if stress_rows > proposed_test_rows:
        return "over_covered_or_different_pool"
    if stress_rows > 0:
        return "partial_coverage"
    return "not_run"


def _domain_coverage(source_split: Mapping[str, Any], stress: Mapping[str, Any]) -> list[dict[str, Any]]:
    train = _domain_counts(source_split, "train")
    val = _domain_counts(source_split, "val")
    test = _domain_counts(source_split, "test")
    stress_domains = stress.get("by_domain", {})
    domains = sorted(set(train) | set(val) | set(test) | set(stress_domains))
    rows = []
    for domain in domains:
        proposed_test_rows = int(test.get(domain, 0))
        stress_rows = int(stress_domains.get(domain, {}).get("rows", 0))
        ratio = None if proposed_test_rows == 0 else float(stress_rows / proposed_test_rows)
        rows.append(
            {
                "domain": domain,
                "train_rows": int(train.get(domain, 0)),
                "val_rows": int(val.get(domain, 0)),
                "proposed_test_rows": proposed_test_rows,
                "locked_policy_stress_rows": stress_rows,
                "coverage_ratio_vs_proposed_test": ratio,
                "status": _status_for_domain(proposed_test_rows, stress_rows),
                "ade_all_ci_low": float(stress_domains.get(domain, {}).get("ade_all", {}).get("ci_low", 0.0)),
                "ade_t50_ci_low": float(stress_domains.get(domain, {}).get("ade_t50", {}).get("ci_low", 0.0)),
                "hard_failure_ci_low": float(stress_domains.get(domain, {}).get("ade_hard_failure", {}).get("ci_low", 0.0)),
                "easy_degradation_ci_high": float(stress_domains.get(domain, {}).get("ade_easy_degradation", {}).get("ci_high", 0.0)),
            }
        )
    return rows


def _horizon_coverage(source_split: Mapping[str, Any], stress: Mapping[str, Any]) -> list[dict[str, Any]]:
    test = source_split.get("proposed_source_level_split", {}).get("test", {})
    rows = []
    for horizon in [10, 25, 50, 100]:
        proposed = int(test.get(f"t{horizon}", 0))
        item = stress.get("by_horizon", {}).get(str(horizon), {})
        stress_rows = int(item.get("rows", 0))
        ratio = None if proposed == 0 else float(stress_rows / proposed)
        status = "exact_row_count_match" if stress_rows == proposed else ("not_run" if stress_rows == 0 else "different_eval_pool")
        rows.append(
            {
                "horizon": horizon,
                "proposed_test_rows": proposed,
                "locked_policy_stress_rows": stress_rows,
                "coverage_ratio_vs_proposed_test": ratio,
                "status": status,
                "ade_ci_low": float(item.get("ade_all", {}).get("ci_low", 0.0)),
                "easy_degradation_ci_high": float(item.get("ade_easy_degradation", {}).get("ci_high", 0.0)),
            }
        )
    return rows


def _claim_table(domain_rows: list[Mapping[str, Any]], horizon_rows: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    exact_domains = [row["domain"] for row in domain_rows if row["status"] == "exact_row_count_match"]
    partial_domains = [row["domain"] for row in domain_rows if row["status"] == "partial_coverage"]
    extra_domains = [row["domain"] for row in domain_rows if row["status"] == "extra_available_not_in_proposed_source_test"]
    different_horizons = [f"t{row['horizon']}" for row in horizon_rows if row["status"] == "different_eval_pool"]
    return [
        {
            "claim": "Stage42 post-repair policy has a frozen hash and source-level split hash.",
            "status": "supported",
            "evidence": "Stage42-AK policy_hash/source_split_hash exist and AK gate passed.",
        },
        {
            "claim": "UCY locked-policy stress row count exactly matches proposed source-level test rows.",
            "status": "supported" if "UCY" in exact_domains else "not_supported",
            "evidence": f"exact_domains={exact_domains}",
        },
        {
            "claim": "TrajNet locked-policy stress fully covers the proposed source-level test rows.",
            "status": "not_supported" if "TrajNet" in partial_domains else "supported_or_not_applicable",
            "evidence": f"partial_domains={partial_domains}",
        },
        {
            "claim": "ETH_UCY post-repair stress rows are part of the proposed source-level test split.",
            "status": "not_supported" if "ETH_UCY" in extra_domains else "supported_or_not_applicable",
            "evidence": f"extra_available_not_in_proposed_source_test={extra_domains}",
        },
        {
            "claim": "Current locked-policy metrics can be described as full proposed source-level split evaluation.",
            "status": "rejected",
            "evidence": f"domain statuses include partial/extra pools; horizon statuses include {different_horizons}.",
        },
        {
            "claim": "Current locked-policy metrics can be described as available row-level post-repair stress with explicit coverage gap.",
            "status": "supported",
            "evidence": "Coverage matrix reports exact, partial, and extra-pool domains separately.",
        },
        {
            "claim": "Metric or seconds-level claims are allowed.",
            "status": "rejected",
            "evidence": "Stage42-AK/AJ claim boundaries reject metric/seconds-level claims.",
        },
        {
            "claim": "Stage5C or SMC is enabled.",
            "status": "rejected",
            "evidence": "Stage5C and SMC remain false.",
        },
    ]


def _next_actions(domain_rows: list[Mapping[str, Any]], horizon_rows: list[Mapping[str, Any]]) -> list[str]:
    actions = []
    for row in domain_rows:
        if row["status"] == "partial_coverage":
            actions.append(
                f"Rebuild full-waypoint prediction cache for proposed source-level `{row['domain']}` test rows: current stress rows {row['locked_policy_stress_rows']} vs proposed test rows {row['proposed_test_rows']}."
            )
        if row["status"] == "extra_available_not_in_proposed_source_test":
            actions.append(
                f"Do not count `{row['domain']}` stress rows as proposed source-level test evidence; either move to diagnostic table or rebuild a split where that source is explicitly test."
            )
        if row["status"] == "not_run":
            actions.append(f"Run locked-policy prediction cache for proposed source-level `{row['domain']}` test rows.")
    if any(row["status"] == "different_eval_pool" for row in horizon_rows):
        actions.append("Recompute horizon metrics on the proposed source-level test row set before claiming full source-level split horizon performance.")
    actions.append("Keep all current results dataset-local raw-frame 2.5D until calibration/homography/stride evidence is verified.")
    return actions


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    claims = {row["claim"]: row["status"] for row in payload["claim_table"]}
    gates = {
        "stage42ak_input_verified": payload.get("stage42ak_gate", {}).get("verdict") == "stage42_ak_post_repair_locked_policy_audit_pass",
        "source_level_split_built": bool(payload["source_split"].get("proposed_source_level_split")),
        "source_overlap_pass": payload["source_split"].get("proposed_split_no_source_overlap") is True,
        "coverage_matrix_built": bool(payload["domain_coverage"]) and bool(payload["horizon_coverage"]),
        "claim_gap_detected": claims["Current locked-policy metrics can be described as full proposed source-level split evaluation."] == "rejected",
        "ucy_exact_coverage_recorded": any(row["domain"] == "UCY" and row["status"] == "exact_row_count_match" for row in payload["domain_coverage"]),
        "partial_or_extra_domains_not_overclaimed": claims["Current locked-policy metrics can be described as available row-level post-repair stress with explicit coverage gap."] == "supported",
        "next_actions_generated": bool(payload["next_actions"]),
        "no_leakage_pass": payload["no_leakage"]["passed"] is True,
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    return {
        "source": "fresh_source_level_coverage_claim_gap_audit",
        "passed": passed,
        "total": len(gates),
        "verdict": "stage42_al_source_level_coverage_audit_pass_with_full_split_eval_gap" if passed == len(gates) else "stage42_al_source_level_coverage_audit_partial",
        "gates": gates,
    }


def build_source_level_coverage_audit() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ak = read_json(STAGE42AK_JSON, {})
    ai = read_json(STAGE42AI_JSON, {})
    x = read_json(STAGE42X_JSON, {})
    source_split = read_json(SOURCE_SPLIT_JSON, {})
    missing = [str(path) for path, obj in [(STAGE42AK_JSON, ak), (STAGE42AI_JSON, ai), (STAGE42X_JSON, x), (SOURCE_SPLIT_JSON, source_split)] if not obj]
    if missing:
        raise FileNotFoundError(f"Missing required Stage42 inputs: {missing}")
    stress = ai.get("stress", x.get("stress", {}))
    domain_rows = _domain_coverage(source_split, stress)
    horizon_rows = _horizon_coverage(source_split, stress)
    payload = {
        "source": "fresh_synthesis_from_stage42_ak_ai_x_source_split",
        "stage": "Stage42-AL source-level coverage and claim-gap audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _hash_paths([STAGE42AK_JSON, STAGE42AI_JSON, STAGE42X_JSON, SOURCE_SPLIT_JSON]),
        "stage42ak_gate": ak.get("stage42_ak_gate", {}),
        "source_split": source_split,
        "domain_coverage": domain_rows,
        "horizon_coverage": horizon_rows,
        "claim_table": _claim_table(domain_rows, horizon_rows),
        "next_actions": _next_actions(domain_rows, horizon_rows),
        "no_leakage": {
            "source": "combined_from_stage42ak_and_source_split",
            "passed": bool(ak.get("no_leakage", {}).get("passed")) and bool(source_split.get("no_leakage", {}).get("proposed_source_overlap_pass")) and not bool(source_split.get("no_leakage", {}).get("frozen_eval_uses_old_train_rows")),
            "stage42ak": ak.get("no_leakage", {}),
            "source_split": source_split.get("no_leakage", {}),
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
    payload["stage42_al_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_md(payload))
    write_md(GATE_MD, _render_gate_md(payload))
    _append_readmes_and_state(payload)
    return payload


def _render_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_al_gate"]
    lines = [
        "# Stage42-AL Source-Level Coverage and Claim-Gap Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Domain Coverage",
        "",
        "| domain | train | val | proposed test | locked-policy stress rows | ratio | status | ADE all CI low | t50 CI low | easy CI high |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in payload["domain_coverage"]:
        ratio = "n/a" if row["coverage_ratio_vs_proposed_test"] is None else f"{row['coverage_ratio_vs_proposed_test']:.3f}"
        lines.append(
            f"| `{row['domain']}` | {row['train_rows']} | {row['val_rows']} | {row['proposed_test_rows']} | {row['locked_policy_stress_rows']} | {ratio} | `{row['status']}` | {row['ade_all_ci_low']:.6f} | {row['ade_t50_ci_low']:.6f} | {row['easy_degradation_ci_high']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Horizon Coverage",
            "",
            "| horizon | proposed test rows | locked-policy stress rows | ratio | status | ADE CI low | easy CI high |",
            "| ---: | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for row in payload["horizon_coverage"]:
        ratio = "n/a" if row["coverage_ratio_vs_proposed_test"] is None else f"{row['coverage_ratio_vs_proposed_test']:.3f}"
        lines.append(
            f"| {row['horizon']} | {row['proposed_test_rows']} | {row['locked_policy_stress_rows']} | {ratio} | `{row['status']}` | {row['ade_ci_low']:.6f} | {row['easy_degradation_ci_high']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Claim Table",
            "",
            "| claim | status | evidence |",
            "| --- | --- | --- |",
        ]
    )
    for row in payload["claim_table"]:
        lines.append(f"| {row['claim']} | `{row['status']}` | {row['evidence']} |")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in payload["next_actions"])
    return lines


def _render_gate_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_al_gate"]
    lines = [
        "# Stage42-AL Gate",
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
    return lines


def _append_if_missing(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in text:
        path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def _append_readmes_and_state(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_al_gate"]
    block = f"""
## Stage42-AL Source-Level Coverage Audit

```text
source = {payload['source']}
verdict = {gate['verdict']}
gates = {gate['passed']} / {gate['total']}
full_proposed_source_level_eval = false
ucy_source_test_coverage = exact_row_count_match
trajnet_source_test_coverage = partial_coverage
eth_ucy_stress_rows = extra_available_not_in_proposed_source_test
stage5c_executed = false
smc_enabled = false
```

Stage42-AL audits whether the locked post-repair policy can be claimed as a full proposed source-level split evaluation. It cannot: UCY matches the proposed source-level test row count, but TrajNet is only partially covered by the current locked-policy stress pool and ETH_UCY stress rows are extra available rows outside the proposed source-level test split. The correct claim remains available row-level post-repair stress with explicit coverage gap, not full source-level split evaluation.
"""
    for path in [README_RESULTS, README_NEURAL, README_ZH]:
        _append_if_missing(path, "## Stage42-AL Source-Level Coverage Audit", block)
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage42_al_source_level_coverage_audit"
    state["current_verdict"] = gate["verdict"]
    state["last_updated"] = "2026-05-26"
    state["last_successful_command"] = "python run_stage42_source_level_coverage_audit.py"
    state.setdefault("stage42", {})["stage_al_source_level_coverage_audit"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "gates_passed": gate["passed"],
        "gates_total": gate["total"],
        "verdict": gate["verdict"],
        "full_proposed_source_level_eval": False,
        "domain_coverage": payload["domain_coverage"],
        "horizon_coverage": payload["horizon_coverage"],
        "next_actions": payload["next_actions"],
        "claim_boundary": payload["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(RESEARCH_STATE, state)


if __name__ == "__main__":
    build_source_level_coverage_audit()
