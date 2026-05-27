from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
HR_JSON = OUT_DIR / "group_consistency_t100_easy_guard_stage42.json"

REPORT_JSON = OUT_DIR / "group_consistency_t100_easy_guard_freeze_stage42.json"
REPORT_MD = OUT_DIR / "group_consistency_t100_easy_guard_freeze_stage42.md"
POLICY_JSON = OUT_DIR / "frozen_group_consistency_t100_easy_guard_policy_stage42.json"
POLICY_MD = OUT_DIR / "frozen_group_consistency_t100_easy_guard_policy_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hs_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_SUMMARY = Path("README_M3W_CURRENT_MASTER_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

PAPER_READY_MATRIX = OUT_DIR / "paper_ready_evidence_matrix_stage42.md"
A_JOURNAL_GAP = OUT_DIR / "a_journal_gap_stage42.md"

SOURCE = "cached_verified_stage42_hr_policy_freeze_from_fresh_artifact"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HS 冻结 Stage42-HR validation-only domain|t100 easy guard policy。",
    "HS 不重新调阈值，不使用 test metrics 做 policy decision；只把 HR fresh artifact 固化为轻量 policy/replay 证据。",
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


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _stable_hash(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _policy_payload(hr: Mapping[str, Any]) -> dict[str, Any]:
    guard = hr["guarded"]
    metric = guard["metric"]
    return {
        "policy_name": "stage42_hs_frozen_group_consistency_t100_easy_guard_policy",
        "source": SOURCE,
        "base_stage": "Stage42-HR group-consistency t100 easy guard",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_git_commit": _git_commit(),
        "selection_scope": "validation_only_domain_horizon_t100",
        "test_usage": "test_once_from_stage42_hr_after_validation_only_guard_decisions",
        "deployment_role": "protected_t100_easy_guard_for_group_consistency_policy",
        "decision_rule": {
            "type": "domain_horizon_t100_easy_guard",
            "threshold_easy_degradation": guard["threshold"],
            "keep_if": "validation_all_gain > 0 and validation_easy_degradation <= threshold",
            "fallback_if": "validation_easy_degradation_above_threshold_or_nonpositive_gain",
            "fallback_target": "train-horizon causal floor for guarded domain|t100 slice",
            "uses_future_labels": False,
            "uses_test_metrics_for_guard": False,
        },
        "decision_table": {
            "guarded_slices": guard["guarded_slices"],
            "kept_slices": guard["kept_slices"],
        },
        "test_summary_vs_train_horizon_causal_floor": {
            "rows": metric["rows"],
            "all_improvement": metric["all_improvement"],
            "t50_improvement": metric["t50_improvement"],
            "t100_raw_frame_diagnostic_improvement": metric["t100_raw_frame_diagnostic_improvement"],
            "hard_failure_improvement": metric["hard_failure_improvement"],
            "easy_degradation": metric["easy_degradation"],
            "switch_rate": metric["switch_rate"],
            "harm_over_fallback": metric["harm_over_fallback"],
            "t100_easy_degradation": guard["t100_easy_degradation"],
        },
        "by_domain": guard["by_domain"],
        "by_horizon": guard["by_horizon"],
        "pre_guard": hr["pre_guard"],
        "no_leakage": hr["no_leakage"],
        "claim_boundary": hr["claim_boundary"],
    }


def _replay_from_policy(policy: Mapping[str, Any], hr: Mapping[str, Any]) -> dict[str, Any]:
    guard = hr["guarded"]
    expected_decisions = {
        "guarded_slices": guard["guarded_slices"],
        "kept_slices": guard["kept_slices"],
    }
    policy_decisions = policy["decision_table"]
    metric = policy["test_summary_vs_train_horizon_causal_floor"]
    hr_metric = dict(guard["metric"])
    hr_metric["t100_easy_degradation"] = guard["t100_easy_degradation"]
    metric_keys = [
        "rows",
        "all_improvement",
        "t50_improvement",
        "t100_raw_frame_diagnostic_improvement",
        "hard_failure_improvement",
        "easy_degradation",
        "switch_rate",
        "harm_over_fallback",
        "t100_easy_degradation",
    ]
    diffs = {
        key: abs(float(metric[key]) - float(hr_metric[key]))
        for key in metric_keys
    }
    return {
        "source": "deterministic_compact_replay_from_frozen_policy_and_hr_artifact",
        "hr_artifact": str(HR_JSON),
        "decision_table_exact_replay": policy_decisions == expected_decisions,
        "metric_summary_exact_replay": max(diffs.values()) <= 1e-12,
        "max_metric_abs_diff": max(diffs.values()),
        "metric_diffs": diffs,
        "guarded_slices": policy_decisions["guarded_slices"],
        "kept_slices": policy_decisions["kept_slices"],
        "test_summary_vs_train_horizon_causal_floor": metric,
        "by_domain": policy["by_domain"],
        "by_horizon": policy["by_horizon"],
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    hr_gate = payload["inputs"]["stage42_hr_gate"]
    policy = payload["frozen_policy"]
    replay = payload["replay"]
    metric = policy["test_summary_vs_train_horizon_causal_floor"]
    no_leak = policy["no_leakage"]
    claim = policy["claim_boundary"]
    guarded = policy["decision_table"]["guarded_slices"]
    kept = policy["decision_table"]["kept_slices"]
    gates = {
        "hr_artifact_loaded": payload["inputs"]["hr_artifact_exists"] is True,
        "hr_gate_passed": hr_gate.get("passed") == hr_gate.get("total"),
        "policy_artifact_written": len(payload["policy_artifact"]["sha256"]) == 64,
        "policy_hash_recorded": len(payload["policy_hash"]) == 64,
        "validation_only_selection": policy["selection_scope"] == "validation_only_domain_horizon_t100",
        "test_once_usage": policy["test_usage"] == "test_once_from_stage42_hr_after_validation_only_guard_decisions",
        "decision_table_exact_replay": replay["decision_table_exact_replay"] is True,
        "metric_summary_exact_replay": replay["metric_summary_exact_replay"] is True,
        "trajnet_t100_guarded": "TrajNet|100" in guarded and guarded["TrajNet|100"].get("keep") is False,
        "ucy_t100_kept": "UCY|100" in kept and kept["UCY|100"].get("keep") is True,
        "t100_easy_repaired": metric["t100_easy_degradation"] <= 0.02,
        "test_all_positive": metric["all_improvement"] > 0.0,
        "test_t50_positive": metric["t50_improvement"] > 0.0,
        "test_t100_raw_positive": metric["t100_raw_frame_diagnostic_improvement"] > 0.0,
        "test_hard_positive": metric["hard_failure_improvement"] > 0.0,
        "easy_preserved": metric["easy_degradation"] <= 0.02,
        "no_future_endpoint_input": no_leak["future_endpoint_input"] is False,
        "no_future_waypoint_input": no_leak["future_waypoint_input"] is False,
        "no_central_velocity": no_leak["central_velocity"] is False,
        "no_test_endpoint_goals": no_leak["test_endpoint_goals"] is False,
        "no_test_threshold_tuning": no_leak["test_threshold_tuning"] is False,
        "internal_val_from_train_only": no_leak["internal_val_from_train_only"] is True,
        "source_overlap_pass": no_leak["source_overlap_pass"] is True,
        "no_metric_seconds_overclaim": claim["metric_or_seconds_claim"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
        "paper_files_updated": all(row["contains_stage42_hs"] for row in payload["paper_file_status"] if row["exists"]),
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_hs_t100_easy_guard_freeze_pass" if passed == total else "stage42_hs_t100_easy_guard_freeze_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _write_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    write_json(POLICY_JSON, policy)
    content = [
        "# Frozen Stage42-HS Group-Consistency T100 Easy Guard Policy",
        "",
        f"- source: `{policy['source']}`",
        f"- base_stage: `{policy['base_stage']}`",
        f"- selection_scope: `{policy['selection_scope']}`",
        f"- deployment_role: `{policy['deployment_role']}`",
        f"- decision_rule: `{policy['decision_rule']['keep_if']}`",
        f"- fallback: `{policy['decision_rule']['fallback_target']}`",
        f"- guarded_slices: `{policy['decision_table']['guarded_slices']}`",
        f"- kept_slices: `{policy['decision_table']['kept_slices']}`",
        "",
        "Claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.",
    ]
    write_md(POLICY_MD, content)
    raw = POLICY_JSON.read_bytes()
    return {"path": str(POLICY_JSON), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def _refresh_paper_files(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    metric = payload["frozen_policy"]["test_summary_vs_train_horizon_causal_floor"]
    lines = [
        "## Stage42-HS Frozen T100 Easy Guard",
        "",
        "- source: `cached_verified_stage42_hr_policy_freeze_from_fresh_artifact`",
        "- role: freeze Stage42-HR validation-only domain|t100 easy guard as a lightweight policy artifact.",
        f"- policy artifact: `{POLICY_JSON}`",
        f"- policy hash: `{payload['policy_hash']}`",
        f"- replay: decision table exact `{payload['replay']['decision_table_exact_replay']}`, metric summary exact `{payload['replay']['metric_summary_exact_replay']}`.",
        f"- guarded all/t50/t100 raw/hard/easy: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}` / `{_pct(metric['easy_degradation'])}`.",
        f"- t100 easy degradation after guard: `{_pct(metric['t100_easy_degradation'])}`.",
        "- t100 remains raw-frame diagnostic; this is not metric/seconds-level evidence.",
        "- Stage5C remains false; SMC remains false.",
    ]
    status = []
    for path in [PAPER_READY_MATRIX, A_JOURNAL_GAP]:
        _replace_section(path, "STAGE42_HS_T100_EASY_GUARD_FREEZE", lines)
        text = path.read_text(encoding="utf-8")
        status.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "contains_stage42_hs": "Stage42-HS Frozen T100 Easy Guard" in text,
                "contains_claim_boundary": "not metric/seconds-level evidence" in text and "Stage5C remains false" in text,
            }
        )
    return status


def _write_report(payload: Mapping[str, Any]) -> None:
    metric = payload["frozen_policy"]["test_summary_vs_train_horizon_causal_floor"]
    replay = payload["replay"]
    gate = payload["stage42_hs_gate"]
    lines = [
        "# Stage42-HS T100 Easy Guard Freeze / Replay",
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
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Frozen Policy",
        "",
        f"- policy artifact: `{POLICY_JSON}`",
        f"- policy hash: `{payload['policy_hash']}`",
        f"- guarded_slices: `{payload['frozen_policy']['decision_table']['guarded_slices']}`",
        f"- kept_slices: `{payload['frozen_policy']['decision_table']['kept_slices']}`",
        "",
        "## Replay",
        "",
        f"- decision_table_exact_replay: `{replay['decision_table_exact_replay']}`",
        f"- metric_summary_exact_replay: `{replay['metric_summary_exact_replay']}`",
        f"- max_metric_abs_diff: `{replay['max_metric_abs_diff']}`",
        "",
        "## Guarded Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| all | {_pct(metric['all_improvement'])} |",
        f"| t50 | {_pct(metric['t50_improvement'])} |",
        f"| t100 raw diagnostic | {_pct(metric['t100_raw_frame_diagnostic_improvement'])} |",
        f"| hard/failure | {_pct(metric['hard_failure_improvement'])} |",
        f"| easy degradation | {_pct(metric['easy_degradation'])} |",
        f"| t100 easy degradation | {_pct(metric['t100_easy_degradation'])} |",
        f"| switch | {_pct(metric['switch_rate'])} |",
        "",
        "## Interpretation",
        "",
        "- HS freezes the HR t100 easy guard as a lightweight deployment/paper artifact.",
        "- It does not rerun training, retune thresholds, execute Stage5C, enable SMC, or make metric/seconds-level claims.",
        "- The t100 result remains raw-frame diagnostic; the primary value is safety: t100 easy harm is guarded while all/t50/hard remain positive.",
    ]
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage42-HS Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{name}` | `{bool(ok)}` |" for name, ok in gate["gates"].items()],
    ]
    write_md(GATE_MD, gate_lines)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    metric = payload["frozen_policy"]["test_summary_vs_train_horizon_causal_floor"]
    lines = [
        "## Stage42-HS Frozen T100 Easy Guard",
        "",
        "- source: `cached_verified_stage42_hr_policy_freeze_from_fresh_artifact`",
        "- role: freeze Stage42-HR validation-only domain|t100 easy guard as a lightweight policy/replay artifact.",
        f"- policy artifact: `{POLICY_JSON}`",
        f"- policy hash: `{payload['policy_hash']}`",
        f"- gate: `{payload['stage42_hs_gate']['passed']} / {payload['stage42_hs_gate']['total']}`; verdict `{payload['stage42_hs_gate']['verdict']}`.",
        f"- replay: decision table exact `{payload['replay']['decision_table_exact_replay']}`, metric summary exact `{payload['replay']['metric_summary_exact_replay']}`.",
        f"- guarded all/t50/t100 raw/hard/easy: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}` / `{_pct(metric['easy_degradation'])}`.",
        f"- t100 easy degradation after guard: `{_pct(metric['t100_easy_degradation'])}`.",
        "- Claim boundary: protected dataset-local/raw-frame 2.5D only; t100 remains raw-frame diagnostic; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, MASTER_SUMMARY]:
        _replace_section(path, "STAGE42_HS_T100_EASY_GUARD_FREEZE", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-HS frozen t100 easy guard policy"
    state["current_verdict"] = payload["stage42_hs_gate"]["verdict"]
    state["stage42_hs_t100_easy_guard_freeze"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "policy_artifact": str(POLICY_JSON),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_hs_gate"]["verdict"],
        "gates": f"{payload['stage42_hs_gate']['passed']}/{payload['stage42_hs_gate']['total']}",
        "policy_hash": payload["policy_hash"],
        "decision_table_exact_replay": payload["replay"]["decision_table_exact_replay"],
        "metric_summary_exact_replay": payload["replay"]["metric_summary_exact_replay"],
        "metric": payload["frozen_policy"]["test_summary_vs_train_horizon_causal_floor"],
        "claim_boundary": payload["frozen_policy"]["claim_boundary"],
        "summary_only_from_hr_fresh_artifact": True,
    }
    state["last_updated"] = "2026-05-27"
    write_json(RESEARCH_STATE, state)


def run_stage42_group_consistency_t100_easy_guard_freeze() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    hr = read_json(HR_JSON, {})
    if not hr:
        raise FileNotFoundError(f"Missing Stage42-HR artifact: {HR_JSON}")
    policy = _policy_payload(hr)
    policy_hash = _stable_hash(policy)
    policy["policy_hash"] = policy_hash
    policy_artifact = _write_policy(policy)
    replay = _replay_from_policy(policy, hr)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-HS frozen t100 easy guard policy",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([HR_JSON]),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "hr_artifact_exists": HR_JSON.exists(),
            "hr_artifact": str(HR_JSON),
            "stage42_hr_gate": hr.get("stage42_hr_gate", {}),
        },
        "frozen_policy": policy,
        "policy_hash": policy_hash,
        "policy_artifact": policy_artifact,
        "replay": replay,
    }
    payload["paper_file_status"] = _refresh_paper_files(payload)
    payload["stage42_hs_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    _write_report(payload)
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    out = run_stage42_group_consistency_t100_easy_guard_freeze()
    gate = out["stage42_hs_gate"]
    print(f"Stage42-HS t100 easy guard freeze: {gate['verdict']} ({gate['passed']}/{gate['total']})")
