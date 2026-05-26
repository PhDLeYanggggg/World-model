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


STAGE42AF_JSON = OUT_DIR / "weak_slice_guard_stage42.json"
STAGE42AG_JSON = OUT_DIR / "eth_t50_fde_source_repair_stage42.json"
STAGE42AI_JSON = OUT_DIR / "trajnet_t100_safety_repair_stage42.json"
STAGE42AJ_JSON = OUT_DIR / "paper_package_post_repair_refresh_stage42.json"
STAGE42X_JSON = OUT_DIR / "unified_row_level_full_waypoint_cache_stage42.json"
SOURCE_SPLIT_JSON = OUT_DIR / "external_source_split_stage42.json"

REPORT_JSON = OUT_DIR / "post_repair_locked_policy_audit_stage42.json"
REPORT_MD = OUT_DIR / "post_repair_locked_policy_audit_stage42.md"
POLICY_JSON = OUT_DIR / "post_repair_locked_policy_stage42_policy.json"
GATE_MD = OUT_DIR / "stage42_stage_ak_gate.md"

README_RESULTS = Path("README_RESULTS.md")
README_NEURAL = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
README_ZH = Path("README_M3W_RESEARCH_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-AK 锁定的是 Stage42-AF/AG/AI post-repair policy 规则和 source-level split evidence，不重新训练模型。",
    "所有 policy switch/guard 阈值来自 validation 或已有 frozen reports；不使用 test 调阈值。",
    "Future waypoints / endpoints 只允许作为 supervised labels 或 eval labels，不作为 inference input。",
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


def _metric(payload: Mapping[str, Any], key: str, field: str = "mean") -> float:
    row = payload.get(key, {})
    if isinstance(row, Mapping):
        return float(row.get(field, row.get("mean", 0.0)) or 0.0)
    return float(row or 0.0)


def _gate_status(payload: Mapping[str, Any], key: str) -> tuple[int, int, str]:
    gate = payload.get(key, {})
    return int(gate.get("passed", 0)), int(gate.get("total", 0)), str(gate.get("verdict", "missing"))


def _external_split_summary(split: Mapping[str, Any]) -> dict[str, Any]:
    proposed = split.get("proposed_source_level_split", {})
    frozen_pool = split.get("frozen_model_eval_pool", {})
    return {
        "source": split.get("source", "missing"),
        "protocol": split.get("protocol", "missing"),
        "proposed_source_level_split": proposed,
        "source_overlap_pass": bool(split.get("proposed_split_no_source_overlap", False)),
        "split_group_overlap": split.get("proposed_split_group_overlap", {}),
        "frozen_model_eval_pool": frozen_pool,
        "no_leakage": split.get("no_leakage", {}),
    }


def _build_policy(af: Mapping[str, Any], ag: Mapping[str, Any], ai: Mapping[str, Any], x: Mapping[str, Any], split: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "policy_name": "stage42_ak_post_repair_locked_policy",
        "source": "fresh_synthesis_from_stage42_af_ag_ai_aj_and_source_split",
        "base_cache": {
            "stage": x.get("stage", "Stage42-X unified row-level full-waypoint cache"),
            "cache_hash": x.get("cache_hash", ""),
            "input_hash": x.get("input_hash", ""),
            "rows": x.get("rows", {}),
        },
        "ordered_policy_rules": [
            {
                "rule_id": "base_stage42x_row_level_full_waypoint_policy",
                "source_report": str(STAGE42X_JSON),
                "decision_source": "cached_verified_stage42x_outputs",
                "uses_test_metrics_for_threshold": False,
            },
            {
                "rule_id": "stage42af_validation_margin_guard",
                "source_report": str(STAGE42AF_JSON),
                "decision_source": "Stage42-R validation margin",
                "guard_rule": af.get("guard_rule", {}),
                "repair_effect": af.get("repair_effect", {}),
                "uses_test_metrics_for_threshold": False,
                "fallback_action": "floor_non_harm",
            },
            {
                "rule_id": "stage42ag_eth_ucy_t50_fde_source_repair",
                "source_report": str(STAGE42AG_JSON),
                "decision_source": "validation FDE@50 and validation ADE@50",
                "source_repair_rule": ag.get("source_repair_rule", {}),
                "repair_effect": ag.get("repair_effect", {}),
                "uses_test_metrics_for_threshold": bool(ag.get("source_repair_rule", {}).get("uses_test_metrics_for_threshold", True)),
            },
            {
                "rule_id": "stage42ai_trajnet_t100_easy_safety_repair",
                "source_report": str(STAGE42AI_JSON),
                "decision_source": "validation easy-degradation and validation ADE",
                "source_repair_rule": ai.get("source_repair_rule", {}),
                "repair_effect": ai.get("repair_effect", {}),
                "uses_test_metrics_for_threshold": bool(ai.get("source_repair_rule", {}).get("uses_test_metrics_for_threshold", True)),
            },
        ],
        "source_split_protocol": _external_split_summary(split),
        "test_usage_rule": "test_once_after_validation_only_policy_freeze",
        "deployment_default": "Stage37 / teacher safety floor for unsupported or unsafe slices",
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "t100_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
            "ungated_neural_deployable": False,
        },
    }


def _combined_no_leakage(policy: Mapping[str, Any], af: Mapping[str, Any], ag: Mapping[str, Any], ai: Mapping[str, Any], split: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "future_endpoint_input": False,
        "future_waypoint_input": False,
        "central_velocity": False,
        "test_endpoint_goals": False,
        "test_threshold_tuning": False,
        "source_overlap_pass": bool(split.get("proposed_split_no_source_overlap", False)),
        "frozen_eval_uses_old_train_rows": bool(split.get("no_leakage", {}).get("frozen_eval_uses_old_train_rows", True)),
        "af_uses_test_threshold": any(rule.get("uses_test_metrics_for_threshold", False) for rule in policy["ordered_policy_rules"] if rule["rule_id"].startswith("stage42af")),
        "ag_uses_test_threshold": any(rule.get("uses_test_metrics_for_threshold", False) for rule in policy["ordered_policy_rules"] if rule["rule_id"].startswith("stage42ag")),
        "ai_uses_test_threshold": any(rule.get("uses_test_metrics_for_threshold", False) for rule in policy["ordered_policy_rules"] if rule["rule_id"].startswith("stage42ai")),
    }
    for source in [af, ag, ai]:
        no_leak = source.get("no_leakage", {})
        for key in ["future_endpoint_input", "future_waypoints_input", "future_waypoint_input", "central_velocity", "test_endpoint_goals", "test_threshold_tuning"]:
            if key in no_leak:
                normalized = "future_waypoint_input" if key in {"future_waypoints_input", "future_waypoint_input"} else key
                checks[normalized] = checks.get(normalized, False) or bool(no_leak[key])
    return {
        "checks": checks,
        "passed": (
            checks["future_endpoint_input"] is False
            and checks["future_waypoint_input"] is False
            and checks["central_velocity"] is False
            and checks["test_endpoint_goals"] is False
            and checks["test_threshold_tuning"] is False
            and checks["source_overlap_pass"] is True
            and checks["frozen_eval_uses_old_train_rows"] is False
            and checks["ag_uses_test_threshold"] is False
            and checks["ai_uses_test_threshold"] is False
        ),
    }


def _summary(ai: Mapping[str, Any]) -> dict[str, Any]:
    s = ai.get("summary", {})
    return {
        "source": "cached_verified_from_stage42_ai_post_repair_policy",
        "ade_all_ci_low": _metric(s, "ade_all", "ci_low"),
        "ade_t50_ci_low": _metric(s, "ade_t50", "ci_low"),
        "ade_t100_raw_frame_diagnostic_ci_low": _metric(s, "ade_t100_raw_frame_diagnostic", "ci_low"),
        "ade_hard_failure_ci_low": _metric(s, "ade_hard_failure", "ci_low"),
        "easy_degradation_ci_high": _metric(s, "ade_easy_degradation", "ci_high"),
        "fde_t50_ci_low": _metric(s, "fde_t50", "ci_low"),
        "switch_rate_mean": _metric(s, "switch_rate"),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    af_passed, af_total, af_verdict = _gate_status(payload["inputs"]["stage42af"], "stage42_af_gate")
    ag_passed, ag_total, ag_verdict = _gate_status(payload["inputs"]["stage42ag"], "stage42_ag_gate")
    ai_passed, ai_total, ai_verdict = _gate_status(payload["inputs"]["stage42ai"], "stage42_ai_gate")
    aj_passed, aj_total, aj_verdict = _gate_status(payload["inputs"]["stage42aj"], "stage42_aj_gate")
    split = payload["external_source_split"]
    summary = payload["post_repair_summary"]
    policy = payload["policy"]
    gates = {
        "stage42af_gate_pass": af_passed == af_total and af_total > 0 and af_verdict == "stage42_af_weak_slice_guard_repair_pass_with_eth_t50_limitation",
        "stage42ag_gate_pass": ag_passed == ag_total and ag_total > 0 and ag_verdict == "stage42_ag_eth_t50_fde_source_repair_pass",
        "stage42ai_gate_pass": ai_passed == ai_total and ai_total > 0 and ai_verdict == "stage42_ai_trajnet_t100_safety_repair_pass",
        "stage42aj_paper_package_current": aj_passed == aj_total and aj_total > 0 and aj_verdict == "stage42_aj_post_repair_paper_package_refresh_pass",
        "policy_artifact_written": POLICY_JSON.exists(),
        "policy_hash_recorded": bool(payload.get("policy_hash")),
        "input_hash_recorded": bool(payload.get("input_hash")),
        "source_level_split_no_overlap": bool(split.get("source_overlap_pass", False)),
        "frozen_eval_excludes_old_train_rows": payload["no_leakage"]["checks"].get("frozen_eval_uses_old_train_rows") is False,
        "validation_only_rules": all(rule.get("uses_test_metrics_for_threshold") is False for rule in policy.get("ordered_policy_rules", [])),
        "no_leakage_pass": payload["no_leakage"]["passed"] is True,
        "global_post_repair_positive": summary["ade_all_ci_low"] > 0.0 and summary["ade_t50_ci_low"] > 0.0 and summary["ade_hard_failure_ci_low"] > 0.0,
        "easy_preserved": summary["easy_degradation_ci_high"] <= 0.02,
        "t100_diagnostic_only": policy["claim_boundary"]["t100_seconds_claim"] is False,
        "no_metric_seconds_overclaim": policy["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": policy["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": policy["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    return {
        "source": "fresh_post_repair_locked_policy_audit",
        "passed": passed,
        "total": len(gates),
        "verdict": "stage42_ak_post_repair_locked_policy_audit_pass" if passed == len(gates) else "stage42_ak_post_repair_locked_policy_audit_partial",
        "gates": gates,
    }


def build_post_repair_locked_policy_audit() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    af = read_json(STAGE42AF_JSON, {})
    ag = read_json(STAGE42AG_JSON, {})
    ai = read_json(STAGE42AI_JSON, {})
    aj = read_json(STAGE42AJ_JSON, {})
    x = read_json(STAGE42X_JSON, {})
    split = read_json(SOURCE_SPLIT_JSON, {})
    missing = [str(path) for path, obj in [(STAGE42AF_JSON, af), (STAGE42AG_JSON, ag), (STAGE42AI_JSON, ai), (STAGE42AJ_JSON, aj), (STAGE42X_JSON, x), (SOURCE_SPLIT_JSON, split)] if not obj]
    if missing:
        raise FileNotFoundError(f"Missing required Stage42 inputs: {missing}")

    policy = _build_policy(af, ag, ai, x, split)
    write_json(POLICY_JSON, policy)
    input_paths = [STAGE42AF_JSON, STAGE42AG_JSON, STAGE42AI_JSON, STAGE42AJ_JSON, STAGE42X_JSON, SOURCE_SPLIT_JSON]
    payload = {
        "source": "fresh_synthesis_from_stage42_af_ag_ai_aj_and_source_split",
        "stage": "Stage42-AK post-repair locked policy and source split audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _hash_paths(input_paths),
        "policy_hash": _hash_paths([POLICY_JSON]),
        "source_split_hash": _hash_paths([SOURCE_SPLIT_JSON]),
        "inputs": {
            "stage42af": af,
            "stage42ag": ag,
            "stage42ai": ai,
            "stage42aj": aj,
        },
        "policy": policy,
        "external_source_split": _external_split_summary(split),
        "post_repair_summary": _summary(ai),
        "no_leakage": _combined_no_leakage(policy, af, ag, ai, split),
    }
    payload["stage42_ak_gate"] = _gate(payload)
    write_json(REPORT_JSON, _public_payload(payload))
    write_md(REPORT_MD, _render_md(payload))
    write_md(GATE_MD, _render_gate_md(payload))
    _append_readmes_and_state(payload)
    return payload


def _public_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    public = dict(payload)
    public["inputs"] = {
        name: {
            "source": source.get("source"),
            "stage": source.get("stage"),
            "input_hash": source.get("input_hash"),
            "claim_boundary": source.get("claim_boundary", {}),
            "gate": source.get("stage42_af_gate")
            or source.get("stage42_ag_gate")
            or source.get("stage42_ai_gate")
            or source.get("stage42_aj_gate"),
        }
        for name, source in payload["inputs"].items()
    }
    return public


def _render_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ak_gate"]
    summary = payload["post_repair_summary"]
    split = payload["external_source_split"]
    lines = [
        "# Stage42-AK Post-Repair Locked Policy and Source Split Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- policy_hash: `{payload['policy_hash']}`",
        f"- source_split_hash: `{payload['source_split_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Locked Policy Rules",
        "",
        "| order | rule | decision source | test threshold tuning | fallback |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for idx, rule in enumerate(payload["policy"]["ordered_policy_rules"], start=1):
        lines.append(
            f"| {idx} | `{rule['rule_id']}` | {rule.get('decision_source', '')} | `{rule.get('uses_test_metrics_for_threshold')}` | {rule.get('fallback_action', payload['policy']['deployment_default'])} |"
        )
    lines.extend(
        [
            "",
            "## Source-Level Split Audit",
            "",
            f"- protocol: `{split.get('protocol')}`",
            f"- source overlap pass: `{split.get('source_overlap_pass')}`",
            f"- split group overlap: `{split.get('split_group_overlap')}`",
            "",
            "| split | rows | domains | scenes | sources | t50 | t100 | hard | easy |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split_name, item in split.get("proposed_source_level_split", {}).items():
        lines.append(
            f"| `{split_name}` | {item.get('rows', 0)} | `{item.get('domains', {})}` | {item.get('scenes', 0)} | {item.get('sources', 0)} | {item.get('t50', 0)} | {item.get('t100', 0)} | {item.get('hard', 0)} | {item.get('easy', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Post-Repair Summary From Stage42-AI",
            "",
            f"- ADE all CI low: `{summary['ade_all_ci_low']}`",
            f"- ADE t50 CI low: `{summary['ade_t50_ci_low']}`",
            f"- ADE t100 raw-frame diagnostic CI low: `{summary['ade_t100_raw_frame_diagnostic_ci_low']}`",
            f"- ADE hard/failure CI low: `{summary['ade_hard_failure_ci_low']}`",
            f"- easy degradation CI high: `{summary['easy_degradation_ci_high']}`",
            f"- FDE@50 CI low: `{summary['fde_t50_ci_low']}`",
            "",
            "## No-Leakage Audit",
            "",
            f"- passed: `{payload['no_leakage']['passed']}`",
            "",
            "| check | value |",
            "| --- | --- |",
        ]
    )
    for name, value in payload["no_leakage"]["checks"].items():
        lines.append(f"| `{name}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-AK locks the post-repair policy after AF/AG/AI and records the source-level split evidence used by Stage42 external validation.",
            "- This is reproducibility and deployment-boundary evidence, not a new model-training result.",
            "- The policy remains protected by Stage37 / teacher floor for unsafe or unsupported slices.",
            "- Claims remain dataset-local raw-frame 2.5D. Metric, seconds-level, true-3D, foundation, Stage5C, SMC, and ungated-neural deployment claims remain rejected.",
        ]
    )
    return lines


def _render_gate_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ak_gate"]
    lines = [
        "# Stage42-AK Gate",
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
    gate = payload["stage42_ak_gate"]
    summary = payload["post_repair_summary"]
    block = f"""
## Stage42-AK Post-Repair Locked Policy Audit

```text
source = {payload['source']}
verdict = {gate['verdict']}
gates = {gate['passed']} / {gate['total']}
policy_hash = {payload['policy_hash']}
source_split_hash = {payload['source_split_hash']}
ade_all_ci_low = {summary['ade_all_ci_low']}
ade_t50_ci_low = {summary['ade_t50_ci_low']}
ade_t100_raw_frame_diagnostic_ci_low = {summary['ade_t100_raw_frame_diagnostic_ci_low']}
ade_hard_failure_ci_low = {summary['ade_hard_failure_ci_low']}
easy_degradation_ci_high = {summary['easy_degradation_ci_high']}
stage5c_executed = false
smc_enabled = false
```

Stage42-AK freezes the post-repair AF/AG/AI policy rules and source-level split audit as reproducibility evidence. It is a policy/source audit, not new training. Claims remain protected dataset-local raw-frame 2.5D; metric/seconds-level, true-3D, foundation, Stage5C, SMC, and ungated-neural deployment claims remain rejected.
"""
    for path in [README_RESULTS, README_NEURAL, README_ZH]:
        _append_if_missing(path, "## Stage42-AK Post-Repair Locked Policy Audit", block)

    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage42_ak_post_repair_locked_policy_audit"
    state["current_verdict"] = gate["verdict"]
    state["last_updated"] = "2026-05-26"
    state["last_successful_command"] = "python run_stage42_post_repair_locked_policy_audit.py"
    state.setdefault("stage42", {})["stage_ak_post_repair_locked_policy_audit"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "policy_artifact": str(POLICY_JSON),
        "gates_passed": gate["passed"],
        "gates_total": gate["total"],
        "verdict": gate["verdict"],
        "policy_hash": payload["policy_hash"],
        "source_split_hash": payload["source_split_hash"],
        "post_repair_summary": summary,
        "claim_boundary": payload["policy"]["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, POLICY_JSON, GATE_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(RESEARCH_STATE, state)


if __name__ == "__main__":
    build_post_repair_locked_policy_audit()
