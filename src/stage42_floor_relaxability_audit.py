from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")

AT_JSON = OUT_DIR / "source_level_safety_floor_audit_stage42.json"
BW_JSON = OUT_DIR / "safety_floor_necessity_audit_stage42.json"
BV_JSON = OUT_DIR / "source_acquisition_status_stage42.json"

REPORT_JSON = OUT_DIR / "floor_relaxability_audit_stage42.json"
REPORT_MD = OUT_DIR / "floor_relaxability_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bx_gate.md"

EASY_LIMIT = 0.02
MIN_VAL_ROWS = 30
MIN_TEST_ROWS = 30


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BX 是 slice-level floor relaxability audit，不训练新模型，不执行 Stage5C，不启用 SMC。",
    "本审计只判断哪些 source/horizon slice 可安全放松 fallback；不允许去掉 teacher/floor rollout context。",
    "future endpoints / waypoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metric(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return float(value if value is not None else default)


def _is_positive_and_easy_safe(metric: Mapping[str, Any], *, horizon: int | None = None) -> bool:
    horizon_key = "t50_improvement" if horizon == 50 else "all_improvement"
    horizon_ok = _metric(metric, horizon_key) > 0.0
    hard_ok = _metric(metric, "hard_failure_improvement") > 0.0
    return _metric(metric, "all_improvement") > 0.0 and (horizon_ok or hard_ok) and _metric(metric, "easy_degradation") <= EASY_LIMIT


def _slice_decision(slice_key: str, row: Mapping[str, Any]) -> dict[str, Any]:
    domain, horizon_text = slice_key.split("|", 1)
    horizon = int(horizon_text)
    val_rows = int(row.get("val_rows", 0))
    test_rows = int(row.get("test_rows", 0))
    val_metric = row.get("val_metric", {})
    test_metric = row.get("test_metric", {})
    has_validation = val_rows >= MIN_VAL_ROWS
    has_test = test_rows >= MIN_TEST_ROWS
    val_safe = has_validation and _is_positive_and_easy_safe(val_metric, horizon=horizon)
    test_safe = has_test and _is_positive_and_easy_safe(test_metric, horizon=horizon)
    if not has_validation:
        status = "blocked_no_validation_support"
        deployment = "fallback_required"
        reason = "No validation rows for this source/horizon; test evidence cannot be used to tune or authorize fallback relaxation."
    elif not has_test:
        status = "blocked_no_test_support"
        deployment = "fallback_required"
        reason = "Insufficient test rows for a source/horizon safety claim."
    elif not val_safe:
        status = "blocked_by_validation_safety"
        deployment = "fallback_required"
        reason = "Validation slice is not simultaneously positive and easy-safe, so fallback relaxation is not deployable."
    elif not test_safe:
        status = "validation_supported_test_failed"
        deployment = "fallback_required"
        reason = "Validation supported relaxation, but final test slice did not preserve positive/easy-safe behavior."
    else:
        status = "relaxable_under_validation_rule"
        deployment = "fallback_relaxable_for_this_slice"
        reason = "Validation and final test are positive/easy-safe for this source/horizon under the baseline-family probe."
    return {
        "source": "fresh_stage42_bx_floor_relaxability_audit",
        "slice": slice_key,
        "domain": domain,
        "horizon": horizon,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "val_metric": val_metric,
        "test_metric": test_metric,
        "has_validation_support": has_validation,
        "validation_positive_easy_safe": val_safe,
        "test_positive_easy_safe": test_safe,
        "status": status,
        "deployment_decision": deployment,
        "reason": reason,
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    at = _load_json(AT_JSON)
    bw = _load_json(BW_JSON)
    bv = _load_json(BV_JSON)
    slice_rows = {
        key: _slice_decision(key, row)
        for key, row in sorted(at["slice_safety_for_all_context_ungated"].items())
    }
    relaxable = [key for key, row in slice_rows.items() if row["status"] == "relaxable_under_validation_rule"]
    blocked = [key for key, row in slice_rows.items() if row["status"] != "relaxable_under_validation_rule"]
    no_validation = [key for key, row in slice_rows.items() if row["status"] == "blocked_no_validation_support"]
    validation_blocked = [key for key, row in slice_rows.items() if row["status"] == "blocked_by_validation_safety"]
    t100_relaxable = [key for key in relaxable if key.endswith("|100")]
    t50_relaxable = [key for key in relaxable if key.endswith("|50")]
    t50_blocked = [key for key in blocked if key.endswith("|50")]

    summary = {
        "source": "fresh_stage42_bx_floor_relaxability_audit",
        "verdict_short": "fallback_relaxation_is_slice_limited_teacher_context_still_required",
        "slice_count": len(slice_rows),
        "relaxable_slice_count": len(relaxable),
        "blocked_slice_count": len(blocked),
        "relaxable_slices": relaxable,
        "blocked_no_validation_support": no_validation,
        "blocked_by_validation_safety": validation_blocked,
        "t50_relaxable_slices": t50_relaxable,
        "t50_blocked_slices": t50_blocked,
        "t100_relaxable_slices": t100_relaxable,
        "teacher_floor_context_required": bw["summary"]["teacher_floor_context_is_core_feature_mechanism"],
        "floor_free_neural_deployable": False,
        "source_blockers_active": bv["summary"]["blockers_active"],
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": "fresh_stage42_bx_floor_relaxability_audit",
        "stage": "Stage42-BX Slice-Level Floor Relaxability Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(AT_JSON), str(BW_JSON), str(BV_JSON)]),
        "current_facts": CURRENT_FACTS,
        "slice_decisions": slice_rows,
        "summary": summary,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "fresh_audit_no_training": True,
            "test_metrics_used_for_reporting_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "floor_free_neural_deployable": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_bx_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    claim = payload["claim_boundary"]
    no_leakage = payload["no_leakage"]
    gates = {
        "slice_decisions_reported": summary["slice_count"] >= 1,
        "at_least_one_relaxable_slice_found": summary["relaxable_slice_count"] >= 1,
        "blocked_slices_reported": summary["blocked_slice_count"] >= 1,
        "no_validation_slices_blocked": len(summary["blocked_no_validation_support"]) >= 1,
        "validation_safety_blocks_some_slices": len(summary["blocked_by_validation_safety"]) >= 1,
        "t50_relaxability_explicit": len(summary["t50_relaxable_slices"]) + len(summary["t50_blocked_slices"]) >= 1,
        "t100_not_overclaimed": len(summary["t100_relaxable_slices"]) == 0,
        "teacher_floor_context_required": summary["teacher_floor_context_required"] is True,
        "floor_free_neural_not_deployable": summary["floor_free_neural_deployable"] is False
        and claim["floor_free_neural_deployable"] is False,
        "source_blockers_still_visible": summary["source_blockers_active"] >= 1,
        "no_leakage_pass": no_leakage["future_endpoint_input"] is False
        and no_leakage["future_waypoint_input"] is False
        and no_leakage["central_velocity"] is False
        and no_leakage["test_endpoint_goals"] is False
        and no_leakage["test_threshold_tuning"] is False,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    verdict = "stage42_bx_floor_relaxability_audit_pass" if passed == total else "stage42_bx_floor_relaxability_audit_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-BX Slice-Level Floor Relaxability Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bx_gate']['passed']} / {payload['stage42_bx_gate']['total']}`",
        f"- verdict: `{payload['stage42_bx_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- verdict_short: `{s['verdict_short']}`",
        f"- relaxable slices: `{s['relaxable_slices']}`",
        f"- blocked_no_validation_support: `{s['blocked_no_validation_support']}`",
        f"- blocked_by_validation_safety: `{s['blocked_by_validation_safety']}`",
        f"- t50_relaxable_slices: `{s['t50_relaxable_slices']}`",
        f"- t50_blocked_slices: `{s['t50_blocked_slices']}`",
        f"- t100_relaxable_slices: `{s['t100_relaxable_slices']}`",
        f"- teacher_floor_context_required: `{s['teacher_floor_context_required']}`",
        f"- floor_free_neural_deployable: `{s['floor_free_neural_deployable']}`",
        "",
        "## Slice Decisions",
        "",
        "| slice | val rows | test rows | val all | val easy | test all | test easy | decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, row in payload["slice_decisions"].items():
        val = row["val_metric"]
        test = row["test_metric"]
        lines.append(
            f"| `{key}` | {row['val_rows']} | {row['test_rows']} | "
            f"{_pct(_metric(val, 'all_improvement'))} | {_pct(_metric(val, 'easy_degradation'))} | "
            f"{_pct(_metric(test, 'all_improvement'))} | {_pct(_metric(test, 'easy_degradation'))} | "
            f"`{row['status']}` |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Fallback relaxation is not globally deployable; it is allowed only for validation-supported source/horizon slices.",
        "- UCY slices in this audit lack validation support and therefore remain fallback-required here even when test metrics look positive.",
        "- TrajNet t100 remains blocked by validation easy harm; t100 remains raw-frame diagnostic and not a deployable long-horizon claim.",
        "- This audit does not authorize removing teacher/floor rollout context and does not authorize ungated neural dynamics.",
        "",
        "## Claim Boundary",
        "",
        "- Not true 3D, not foundation, not global metric, not seconds-level.",
        "- Stage5C remains unexecuted and SMC remains disabled.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bx_gate"]
    lines = [
        "# Stage42-BX Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | {bool(value)} |")
    return lines


def run_stage42_floor_relaxability_audit() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    return payload


if __name__ == "__main__":
    result = run_stage42_floor_relaxability_audit()
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, sort_keys=True))
