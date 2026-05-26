from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
DI_JSON = OUT_DIR / "group_consistency_full_waypoint_repair_stage42.json"

REPORT_JSON = OUT_DIR / "frozen_group_consistency_full_waypoint_policy_stage42.json"
REPORT_MD = OUT_DIR / "frozen_group_consistency_full_waypoint_policy_stage42.md"
POLICY_JSON = OUT_DIR / "frozen_group_consistency_full_waypoint_policy_stage42_policy.json"
GATE_MD = OUT_DIR / "stage42_stage_dj_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_README = Path("README_M3W_GOAL_SUMMARY_ZH.md")
CURRENT_RETROSPECTIVE = Path("README_M3W_CURRENT_FULL_RETROSPECTIVE_ZH.md")
RESEARCH_STATE = Path("research_state.json")

PAPER_FILES = [
    OUT_DIR / "method_draft_stage42.md",
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "reproducibility_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DJ 冻结 Stage42-DI group-consistency full-waypoint repair policy，形成可复现 policy artifact。",
    "policy 冻结只记录 validation-selected repair；test 指标来自 Stage42-DI test-once evidence。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
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


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _policy_payload(di: Mapping[str, Any]) -> dict[str, Any]:
    selected = di["repair"]["selected"]
    metric = di["repair"]["test"]["metric_vs_floor"]
    diag = di["repair"]["test"]["diagnostics"]
    comparison = di["comparison_to_prior"]
    return {
        "policy_name": "stage42_dj_frozen_group_consistency_full_waypoint_policy",
        "source": "fresh_policy_freeze_from_stage42_di",
        "base_stage": "Stage42-DI group-consistency full-waypoint repair",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_git_commit": _git_commit(),
        "deployment_role": "promoted_protected_source_level_full_waypoint_group_consistency_policy",
        "selection_scope": "validation_only",
        "test_usage": "test_once_from_stage42_di_after_validation_selection",
        "repair_rule": {
            "type": selected["candidate"].get("mode"),
            "min_sep": selected["candidate"].get("min_sep"),
            "margin": selected["candidate"].get("margin"),
            "strength": selected["candidate"].get("strength"),
            "alpha": selected["candidate"].get("alpha"),
            "safe_min_sep": selected["candidate"].get("safe_min_sep"),
            "input": "predicted full-waypoint rollout geometry + source/frame/horizon group key + agent id",
            "uses_future_labels": False,
        },
        "group_schema": di.get("group_schema", {}),
        "validation_selection": {
            "val_score": selected.get("val_score"),
            "val_metric": selected.get("val_metric"),
            "val_diagnostics": selected.get("val_diagnostics"),
            "no_test_threshold_tuning": True,
        },
        "test_summary_vs_train_horizon_causal_floor": {
            "rows": metric.get("rows"),
            "all_improvement": metric.get("all_improvement"),
            "t50_improvement": metric.get("t50_improvement"),
            "t100_raw_frame_diagnostic_improvement": metric.get("t100_raw_frame_diagnostic_improvement"),
            "hard_failure_improvement": metric.get("hard_failure_improvement"),
            "easy_degradation": metric.get("easy_degradation"),
            "switch_rate": metric.get("switch_rate"),
        },
        "test_group_safety": {
            "base_near_005": diag.get("base_near_005"),
            "final_near_005": diag.get("final_near_005"),
            "floor_near_005": diag.get("floor_near_005"),
            "base_p05_min_distance": diag.get("base_p05_min_distance"),
            "final_p05_min_distance": diag.get("final_p05_min_distance"),
            "floor_p05_min_distance": diag.get("floor_p05_min_distance"),
            "unsafe_rows": diag.get("unsafe_rows"),
            "unsafe_rate": diag.get("unsafe_rate"),
        },
        "delta_vs_stage42_am": comparison.get("delta_vs_stage42_am", {}),
        "delta_vs_stage42_cq": comparison.get("delta_vs_stage42_cq", {}),
        "delta_vs_stage42_dh": comparison.get("delta_vs_stage42_dh", {}),
        "bootstrap": di["repair"]["test"].get("bootstrap", {}),
        "no_leakage": di.get("no_leakage", {}),
        "claim_boundary": di.get("claim_boundary", {}),
    }


def _refresh_paper_files(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    policy = payload["frozen_policy"]
    metric = policy["test_summary_vs_train_horizon_causal_floor"]
    safety = policy["test_group_safety"]
    delta = policy["delta_vs_stage42_am"]
    lines = [
        "## Stage42-DJ Frozen Group-Consistency Full-Waypoint Policy",
        "",
        "- source: `fresh_policy_freeze_from_stage42_di`",
        "- role: freeze the Stage42-DI promoted group-consistency full-waypoint repair as a reproducible policy artifact.",
        "- repair uses predicted rollout geometry and source/frame/horizon group keys only; future waypoints remain labels/eval only.",
        f"- policy artifact: `{POLICY_JSON}`",
        f"- policy hash: `{payload['policy_hash']}`",
        f"- test vs train-horizon causal floor ADE: all `{_pct(metric['all_improvement'])}`, t50 `{_pct(metric['t50_improvement'])}`, t100 raw `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}`, hard `{_pct(metric['hard_failure_improvement'])}`, easy `{_pct(metric['easy_degradation'])}`.",
        f"- delta vs Stage42-AM all/t50/hard: `{_pct(delta.get('all_improvement'))}` / `{_pct(delta.get('t50_improvement'))}` / `{_pct(delta.get('hard_failure_improvement'))}`.",
        f"- near@0.05 base/final/floor: `{_pct(safety['base_near_005'])}` / `{_pct(safety['final_near_005'])}` / `{_pct(safety['floor_near_005'])}`.",
        "- claim boundary: still protected dataset-local/raw-frame 2.5D; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.",
    ]
    status = []
    for path in PAPER_FILES:
        if path.exists():
            _replace_section(path, "STAGE42_DJ_FROZEN_GROUP_CONSISTENCY_POLICY", lines)
            text = path.read_text(encoding="utf-8")
            status.append(
                {
                    "path": str(path),
                    "exists": True,
                    "contains_stage42_dj": "Stage42-DJ Frozen Group-Consistency Full-Waypoint Policy" in text,
                    "contains_claim_boundary": "no true 3D" in text and "no Stage5C" in text,
                }
            )
        else:
            status.append({"path": str(path), "exists": False, "contains_stage42_dj": False, "contains_claim_boundary": False})
    return status


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    di_gate = payload["inputs"]["stage42_di"]["stage42_di_gate"]
    policy = payload["frozen_policy"]
    metric = policy["test_summary_vs_train_horizon_causal_floor"]
    safety = policy["test_group_safety"]
    delta = policy["delta_vs_stage42_am"]
    no_leak = policy["no_leakage"]
    claim = policy["claim_boundary"]
    gates = {
        "stage42_di_gate_passed": di_gate.get("passed") == di_gate.get("total"),
        "policy_artifact_written": bool(payload.get("policy_artifact", {}).get("sha256")),
        "policy_hash_recorded": bool(payload.get("policy_hash")),
        "validation_only_selection": policy["selection_scope"] == "validation_only",
        "test_once_usage": policy["test_usage"] == "test_once_from_stage42_di_after_validation_selection",
        "repair_uses_rollout_group_geometry_only": policy["repair_rule"]["input"] == "predicted full-waypoint rollout geometry + source/frame/horizon group key + agent id",
        "test_all_positive": float(metric["all_improvement"]) > 0.0,
        "test_t50_positive": float(metric["t50_improvement"]) > 0.0,
        "test_hard_positive": float(metric["hard_failure_improvement"]) > 0.0,
        "easy_under_2pct": float(metric["easy_degradation"]) <= 0.02,
        "beats_stage42_am_all": float(delta["all_improvement"]) > 0.0,
        "beats_stage42_am_hard": float(delta["hard_failure_improvement"]) > 0.0,
        "near_collision_reduced_vs_base": float(safety["final_near_005"]) <= float(safety["base_near_005"]),
        "paper_files_refreshed": all(row["contains_stage42_dj"] for row in payload["paper_file_status"] if row["exists"]),
        "no_future_endpoint_input": no_leak.get("future_endpoint_input") is False,
        "no_future_waypoint_input": no_leak.get("future_waypoint_input") is False,
        "no_central_velocity": no_leak.get("central_velocity") is False,
        "no_test_endpoint_goals": no_leak.get("test_endpoint_goals") is False,
        "no_test_threshold_tuning": no_leak.get("test_threshold_tuning") is False,
        "metric_seconds_overclaim_blocked": claim.get("metric_or_seconds_claim") is False,
        "stage5c_not_executed": claim.get("stage5c_executed") is False,
        "smc_not_enabled": claim.get("smc_enabled") is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_dj_frozen_group_consistency_policy_pass" if passed == total else "stage42_dj_frozen_group_consistency_policy_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    policy = payload["frozen_policy"]
    metric = policy["test_summary_vs_train_horizon_causal_floor"]
    safety = policy["test_group_safety"]
    delta = policy["delta_vs_stage42_am"]
    lines = [
        "## Stage42-DJ Frozen Group-Consistency Full-Waypoint Policy",
        "",
        "- source: `fresh_policy_freeze_from_stage42_di`",
        "- role: freeze the Stage42-DI promoted group-consistency full-waypoint repair as a reproducible deployment/paper artifact.",
        f"- policy artifact: `{POLICY_JSON}`",
        f"- policy hash: `{payload['policy_hash']}`",
        f"- gate: `{payload['stage42_dj_gate']['passed']} / {payload['stage42_dj_gate']['total']}`; verdict `{payload['stage42_dj_gate']['verdict']}`.",
        f"- test vs train-horizon causal floor: all `{_pct(metric['all_improvement'])}`, t50 `{_pct(metric['t50_improvement'])}`, t100 raw `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}`, hard `{_pct(metric['hard_failure_improvement'])}`, easy `{_pct(metric['easy_degradation'])}`.",
        f"- delta vs Stage42-AM all/t50/hard: `{_pct(delta['all_improvement'])}` / `{_pct(delta['t50_improvement'])}` / `{_pct(delta['hard_failure_improvement'])}`.",
        f"- near@0.05 base/final/floor: `{_pct(safety['base_near_005'])}` / `{_pct(safety['final_near_005'])}` / `{_pct(safety['floor_near_005'])}`.",
        "- Stage5C remains false; SMC remains false; no metric/seconds claim.",
    ]
    for path in [README_RESULTS, M3W_README, GOAL_README, CURRENT_RETROSPECTIVE]:
        _replace_section(path, "STAGE42_DJ_FROZEN_GROUP_CONSISTENCY_POLICY", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    policy = payload["frozen_policy"]
    state["current_stage"] = "Stage42-DJ frozen group-consistency full-waypoint policy"
    state["current_verdict"] = payload["stage42_dj_gate"]["verdict"]
    state["stage42_dj_frozen_group_consistency_policy"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "policy_artifact": str(POLICY_JSON),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_dj_gate"]["verdict"],
        "gates": f"{payload['stage42_dj_gate']['passed']}/{payload['stage42_dj_gate']['total']}",
        "policy_hash": payload["policy_hash"],
        "test_metric_vs_train_horizon_causal_floor": policy["test_summary_vs_train_horizon_causal_floor"],
        "test_group_safety": policy["test_group_safety"],
        "delta_vs_stage42_am": policy["delta_vs_stage42_am"],
        "claim_boundary": policy["claim_boundary"],
        "conclusion": "Stage42-DJ freezes Stage42-DI's group-consistency full-waypoint repair as a reproducible protected policy artifact. It remains dataset-local/raw-frame 2.5D evidence only.",
        "verification": {
            "runner": ".venv-pytorch/bin/python run_stage42_freeze_group_consistency_policy.py",
            "targeted_pytest": ".venv-pytorch/bin/python -m pytest tests/test_stage42_group_consistency_policy_freeze.py",
        },
    }
    write_json(RESEARCH_STATE, state)


def _write_md(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_dj_gate"]
    policy = payload["frozen_policy"]
    metric = policy["test_summary_vs_train_horizon_causal_floor"]
    safety = policy["test_group_safety"]
    lines = [
        "# Stage42-DJ Frozen Group-Consistency Full-Waypoint Policy",
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
        f"- repair_rule: `{policy['repair_rule']}`",
        "",
        "## Test Metrics Vs Train-Horizon Causal Floor",
        "",
        f"- all: `{_pct(metric['all_improvement'])}`",
        f"- t50: `{_pct(metric['t50_improvement'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}`",
        f"- hard/failure: `{_pct(metric['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(metric['easy_degradation'])}`",
        f"- switch rate: `{_pct(metric['switch_rate'])}`",
        "",
        "## Group Safety",
        "",
        f"- base near@0.05: `{_pct(safety['base_near_005'])}`",
        f"- final near@0.05: `{_pct(safety['final_near_005'])}`",
        f"- floor near@0.05: `{_pct(safety['floor_near_005'])}`",
        f"- base p05 min distance: `{safety['base_p05_min_distance']}`",
        f"- final p05 min distance: `{safety['final_p05_min_distance']}`",
        "",
        "## Interpretation",
        "",
        "- Stage42-DJ freezes the Stage42-DI group-consistency full-waypoint repair as a reproducible policy artifact.",
        "- This advances deployability and paper reproducibility for the protected full-waypoint branch.",
        "- It remains protected dataset-local/raw-frame 2.5D evidence, not true 3D, not foundation-scale, not metric/seconds-level, not Stage5C, and not SMC.",
    ]
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage42-DJ Gate",
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


def run_stage42_freeze_group_consistency_policy() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    di = read_json(DI_JSON, {})
    if not di:
        raise FileNotFoundError(f"Missing Stage42-DI input: {DI_JSON}")
    policy = _policy_payload(di)
    write_json(POLICY_JSON, _jsonable(policy))
    policy_hash = _combined_hash([POLICY_JSON])
    payload: dict[str, Any] = {
        "source": "fresh_policy_freeze_from_stage42_di",
        "stage": "Stage42-DJ frozen group-consistency full-waypoint policy",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([DI_JSON]),
        "inputs": {
            "stage42_di": {
                "path": str(DI_JSON),
                "stage42_di_gate": di.get("stage42_di_gate", {}),
            }
        },
        "frozen_policy": policy,
        "policy_artifact": {
            "path": str(POLICY_JSON),
            "sha256": policy_hash,
            "size_bytes": POLICY_JSON.stat().st_size,
        },
        "policy_hash": policy_hash,
    }
    payload["paper_file_status"] = _refresh_paper_files(payload)
    payload["stage42_dj_gate"] = _gate(payload)
    write_json(REPORT_JSON, _jsonable(payload))
    _write_md(payload)
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_freeze_group_consistency_policy()
    gate = result["stage42_dj_gate"]
    print(f"Stage42-DJ frozen group-consistency policy: {gate['verdict']} ({gate['passed']}/{gate['total']})")
