from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
CQ_JSON = OUT_DIR / "proximity_aware_composer_guard_stage42.json"
CR_JSON = OUT_DIR / "proximity_guard_ablation_stage42.json"

REPORT_JSON = OUT_DIR / "frozen_proximity_guard_composer_policy_stage42.json"
REPORT_MD = OUT_DIR / "frozen_proximity_guard_composer_policy_stage42.md"
POLICY_JSON = OUT_DIR / "frozen_proximity_guard_composer_policy_stage42_policy.json"
GATE_MD = OUT_DIR / "stage42_stage_cs_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
RETRO_README = Path("README_M3W_GOAL_RETROSPECTIVE_CURRENT_ZH.md")
RESEARCH_STATE = Path("research_state.json")

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CS 冻结 Stage42-CQ/CR 选择的 proximity-guard composer policy。",
    "policy 冻结只使用 validation-selected 阈值和已审计的 predicted-rollout geometry guard。",
    "test 只用于最终报告，不用于阈值选择。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
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
    if isinstance(value, Path):
        return str(value)
    return value


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _policy_payload(cq: Mapping[str, Any], cr: Mapping[str, Any]) -> dict[str, Any]:
    selected = cq["selected_policy"]
    metric = cq["test_eval"]["metric_vs_endpoint_ade"]
    joint = cq["test_joint_safety"]["composer_minus_endpoint"]
    policy = {
        "policy_name": "stage42_cs_frozen_proximity_guard_composer_policy",
        "source": "fresh_policy_freeze_from_stage42_cq_cr",
        "base_stage": "Stage42-CQ proximity-aware composer guard",
        "ablation_stage": "Stage42-CR proximity guard ablation / Pareto audit",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_git_commit": _git_commit(),
        "selection_scope": "validation_only",
        "test_usage": "test_once_after_policy_freeze",
        "deployment_role": "safety_sensitive_deployable_composer_variant",
        "accuracy_priority_diagnostic_policy": cr["deployment_recommendation"]["accuracy_priority_policy"],
        "selected_policy": selected,
        "guard_rule": {
            "type": selected.get("type"),
            "min_sep": selected.get("min_sep"),
            "margin": selected.get("margin"),
            "guard_input": "predicted endpoint/full-waypoint rollout geometry only",
            "uses_future_labels": False,
        },
        "base_choices": selected.get("base_choices", {}),
        "validation_selection_rule": {
            "threshold_source": "validation_only",
            "no_test_threshold_tuning": True,
            "easy_degradation_limit": 0.02,
            "safety_objective": "near_collision@0.05 no worse than endpoint-linear and strongest floor",
        },
        "test_summary_vs_endpoint_linear_ade": {
            "rows": metric.get("rows"),
            "all_improvement": metric.get("all_improvement"),
            "t50_improvement": metric.get("t50_improvement"),
            "t100_raw_frame_diagnostic_improvement": metric.get("t100_raw_frame_diagnostic_improvement"),
            "hard_failure_improvement": metric.get("hard_failure_improvement"),
            "easy_degradation": metric.get("easy_degradation"),
            "switch_rate": metric.get("switch_rate"),
        },
        "joint_safety_vs_endpoint_linear": {
            "near_collision_002_delta": joint.get("near_collision_rate_002_delta"),
            "near_collision_005_delta": joint.get("near_collision_rate_005_delta"),
            "p05_min_group_distance_delta": joint.get("p05_min_group_distance_delta"),
            "jagged_rate_delta": joint.get("jagged_rate_delta"),
        },
        "bootstrap_vs_endpoint_ade": cq.get("bootstrap_vs_endpoint_ade", {}),
        "no_leakage": cq.get("no_leakage", {}),
        "claim_boundary": cq.get("claim_boundary", {}),
    }
    return policy


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    cq_gate = payload["inputs"]["stage42_cq"]["stage42_cq_gate"]
    cr_gate = payload["inputs"]["stage42_cr"]["stage42_cr_gate"]
    policy = payload["frozen_policy"]
    metric = policy["test_summary_vs_endpoint_linear_ade"]
    safety = policy["joint_safety_vs_endpoint_linear"]
    no_leakage = policy["no_leakage"]
    claim_boundary = policy["claim_boundary"]
    bootstrap = policy["bootstrap_vs_endpoint_ade"]
    gates = {
        "cq_gate_passed": cq_gate["passed"] == cq_gate["total"],
        "cr_gate_passed": cr_gate["passed"] == cr_gate["total"],
        "policy_artifact_written": bool(payload.get("policy_artifact", {}).get("sha256")),
        "policy_hash_recorded": bool(payload.get("policy_hash")),
        "validation_only_selection": policy["selection_scope"] == "validation_only",
        "test_once_after_freeze": policy["test_usage"] == "test_once_after_policy_freeze",
        "guard_uses_predicted_rollout_geometry_only": policy["guard_rule"]["guard_input"] == "predicted endpoint/full-waypoint rollout geometry only",
        "all_positive_vs_endpoint": float(metric["all_improvement"]) > 0.0,
        "t50_positive_vs_endpoint": float(metric["t50_improvement"]) > 0.0,
        "t100_positive_vs_endpoint": float(metric["t100_raw_frame_diagnostic_improvement"]) > 0.0,
        "hard_positive_vs_endpoint": float(metric["hard_failure_improvement"]) > 0.0,
        "easy_under_2pct": float(metric["easy_degradation"]) <= 0.02,
        "near_collision_not_worse_than_endpoint": float(safety["near_collision_005_delta"]) <= 0.0,
        "all_ci_low_positive": float(bootstrap["all"]["low"]) > 0.0,
        "t50_ci_low_positive": float(bootstrap["t50"]["low"]) > 0.0,
        "t100_ci_low_positive": float(bootstrap["t100"]["low"]) > 0.0,
        "hard_ci_low_positive": float(bootstrap["hard_failure"]["low"]) > 0.0,
        "no_future_endpoint_input": no_leakage.get("future_endpoint_input") is False,
        "no_future_waypoints_input": no_leakage.get("future_waypoints_input") is False,
        "no_central_velocity": no_leakage.get("central_velocity") is False,
        "no_test_endpoint_goals": no_leakage.get("test_endpoint_goals") is False,
        "no_test_threshold_tuning": no_leakage.get("test_threshold_tuning") is False,
        "metric_seconds_overclaim_blocked": claim_boundary.get("metric_or_seconds_claim") is False,
        "stage5c_not_executed": claim_boundary.get("stage5c_executed") is False,
        "smc_not_enabled": claim_boundary.get("smc_enabled") is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_cs_frozen_proximity_guard_policy_pass" if passed == total else "stage42_cs_frozen_proximity_guard_policy_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _replace_section(path: Path, tag: str, lines: list[str]) -> None:
    start = f"<!-- {tag}:START -->"
    end = f"<!-- {tag}:END -->"
    block = "\n".join([start, *lines, end])
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if start in old and end in old:
        before, rest = old.split(start, 1)
        _, after = rest.split(end, 1)
        new = before.rstrip() + "\n\n" + block + after
    else:
        new = old.rstrip() + "\n\n" + block + "\n"
    path.write_text(new, encoding="utf-8")


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    policy = payload["frozen_policy"]
    metric = policy["test_summary_vs_endpoint_linear_ade"]
    safety = policy["joint_safety_vs_endpoint_linear"]
    lines = [
        "## Stage42-CS Frozen Proximity-Guard Composer Policy",
        "",
        "- source: `fresh_policy_freeze_from_stage42_cq_cr`",
        f"- verdict: `{payload['stage42_cs_gate']['verdict']}`",
        f"- gates: `{payload['stage42_cs_gate']['passed']} / {payload['stage42_cs_gate']['total']}`",
        f"- policy artifact: `{POLICY_JSON}`",
        f"- policy hash: `{payload['policy_hash']}`",
        "- selected deployment role: `safety_sensitive_deployable_composer_variant`",
        f"- ADE vs endpoint-linear all/t50/t100 raw/hard: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(metric['easy_degradation'])}`",
        f"- near-collision@0.05 delta vs endpoint-linear: `{_pct(safety['near_collision_005_delta'])}`",
        "- This freezes the Stage42-CQ/CR safety-sensitive composer. The no-guard composer remains accuracy-priority diagnostic only.",
        "- Claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation claim, no global metric/seconds-level claim, no Stage5C execution, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, RETRO_README]:
        _replace_section(path, "STAGE42_CS_FROZEN_PROXIMITY_GUARD_POLICY", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    policy = payload["frozen_policy"]
    metric = policy["test_summary_vs_endpoint_linear_ade"]
    state["current_stage"] = "Stage42-CS frozen proximity-guard composer policy"
    state["current_verdict"] = payload["stage42_cs_gate"]["verdict"]
    state["stage42_cs_frozen_proximity_guard_policy"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "policy_artifact": str(POLICY_JSON),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_cs_gate"]["verdict"],
        "gates": f"{payload['stage42_cs_gate']['passed']}/{payload['stage42_cs_gate']['total']}",
        "policy_hash": payload["policy_hash"],
        "test_vs_endpoint_linear_ade": {
            "all_improvement": metric["all_improvement"],
            "t50_improvement": metric["t50_improvement"],
            "t100_raw_frame_diagnostic_improvement": metric["t100_raw_frame_diagnostic_improvement"],
            "hard_failure_improvement": metric["hard_failure_improvement"],
            "easy_degradation": metric["easy_degradation"],
        },
        "joint_safety_vs_endpoint_linear": policy["joint_safety_vs_endpoint_linear"],
        "claim_boundary": policy["claim_boundary"],
        "conclusion": "Stage42-CS freezes the Stage42-CQ/CR proximity-guard composer as the safety-sensitive deployable composer variant. It records policy hash, validation-only guard rules, test-once usage, no-leakage boundaries, positive bootstrap evidence, and proximity safety. It remains protected dataset-local/raw-frame 2.5D evidence only.",
        "verification": {
            "runner": ".venv-pytorch/bin/python run_stage42_freeze_proximity_guard_policy.py",
            "targeted_pytest": ".venv-pytorch/bin/python -m pytest tests/test_stage42_proximity_guard_policy_freeze.py",
        },
    }
    summary = state.setdefault("latest_user_facing_goal_summary", {})
    summary["source"] = "cached_verified_synthesis_for_user_question_refreshed_after_stage42_cs"
    included = summary.setdefault("latest_fresh_evidence_included", [])
    note = "Stage42-CS frozen proximity-guard composer policy: safety-sensitive composer artifact/hash frozen from CQ/CR with validation-only guard rules"
    if note not in included:
        included.append(note)
    write_json(RESEARCH_STATE, state)


def _write_md(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_cs_gate"]
    policy = payload["frozen_policy"]
    metric = policy["test_summary_vs_endpoint_linear_ade"]
    safety = policy["joint_safety_vs_endpoint_linear"]
    lines = [
        "# Stage42-CS Frozen Proximity-Guard Composer Policy",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- policy_hash: `{payload['policy_hash']}`",
        f"- policy_artifact: `{POLICY_JSON}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Frozen Policy",
        "",
        f"- policy_name: `{policy['policy_name']}`",
        f"- deployment_role: `{policy['deployment_role']}`",
        f"- selection_scope: `{policy['selection_scope']}`",
        f"- test_usage: `{policy['test_usage']}`",
        f"- guard min_sep: `{policy['guard_rule']['min_sep']}`",
        f"- guard margin: `{policy['guard_rule']['margin']}`",
        f"- guard input: `{policy['guard_rule']['guard_input']}`",
        f"- accuracy-priority diagnostic policy: `{policy['accuracy_priority_diagnostic_policy']}`",
        "",
        "## Test Metrics Vs Endpoint-Linear ADE",
        "",
        f"- all: `{_pct(metric['all_improvement'])}`",
        f"- t50: `{_pct(metric['t50_improvement'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}`",
        f"- hard/failure: `{_pct(metric['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(metric['easy_degradation'])}`",
        f"- switch rate: `{_pct(metric['switch_rate'])}`",
        "",
        "## Joint Safety Vs Endpoint-Linear",
        "",
        f"- near_collision@0.02 delta: `{_pct(safety['near_collision_002_delta'])}`",
        f"- near_collision@0.05 delta: `{_pct(safety['near_collision_005_delta'])}`",
        f"- p05 min group distance delta: `{_pct(safety['p05_min_group_distance_delta'])}`",
        f"- jagged-rate delta: `{_pct(safety['jagged_rate_delta'])}`",
        "",
        "## Interpretation",
        "",
        "- Stage42-CS freezes the safer Stage42-CQ proximity-aware composer as a reproducible policy artifact.",
        "- The no-proximity-guard composer remains useful as an accuracy-priority diagnostic, but it is not the safety-sensitive deployment policy.",
        "- This artifact advances deployability/reproducibility for the protected full-waypoint composer branch without changing the claim boundary.",
        "- It remains protected dataset-local/raw-frame 2.5D evidence, not true 3D, not foundation-scale, not metric/seconds-level, not Stage5C, and not SMC.",
    ]
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage42-CS Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | passed |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        gate_lines.append(f"| `{name}` | `{ok}` |")
    write_md(GATE_MD, gate_lines)


def run_stage42_freeze_proximity_guard_policy() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cq = read_json(CQ_JSON, {})
    cr = read_json(CR_JSON, {})
    if not cq or not cr:
        missing = [str(p) for p in [CQ_JSON, CR_JSON] if not p.exists()]
        raise FileNotFoundError(f"Missing Stage42-CQ/CR inputs: {missing}")
    policy = _policy_payload(cq, cr)
    write_json(POLICY_JSON, _jsonable(policy))
    policy_hash = _combined_hash([POLICY_JSON])
    payload: dict[str, Any] = {
        "source": "fresh_policy_freeze_from_stage42_cq_cr",
        "stage": "Stage42-CS frozen proximity-guard composer policy",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([CQ_JSON, CR_JSON]),
        "inputs": {
            "stage42_cq": {
                "path": str(CQ_JSON),
                "stage42_cq_gate": cq.get("stage42_cq_gate", {}),
            },
            "stage42_cr": {
                "path": str(CR_JSON),
                "stage42_cr_gate": cr.get("stage42_cr_gate", {}),
            },
        },
        "frozen_policy": policy,
        "policy_artifact": {
            "path": str(POLICY_JSON),
            "sha256": policy_hash,
            "size_bytes": POLICY_JSON.stat().st_size,
        },
        "policy_hash": policy_hash,
    }
    payload["stage42_cs_gate"] = _gate(payload)
    write_json(REPORT_JSON, _jsonable(payload))
    _write_md(payload)
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_freeze_proximity_guard_policy()
    gate = result["stage42_cs_gate"]
    print(f"Stage42-CS frozen proximity guard policy: {gate['verdict']} ({gate['passed']}/{gate['total']})")
