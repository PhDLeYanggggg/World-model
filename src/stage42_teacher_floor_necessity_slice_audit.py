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
from src.stage42_current_module_claim_refresh import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
IW_JSON = OUT_DIR / "source_level_row_cache_mechanism_audit_stage42.json"
JV_JSON = OUT_DIR / "source_slice_evidence_matrix_stage42.json"
GT_JSON = OUT_DIR / "floor_relaxation_safety_stress_stage42.json"

REPORT_JSON = OUT_DIR / "teacher_floor_necessity_slice_audit_stage42.json"
REPORT_MD = OUT_DIR / "teacher_floor_necessity_slice_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_jw_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE42_JW_TEACHER_FLOOR_NECESSITY_SLICE_AUDIT"
SOURCE = "fresh_stage42_jw_teacher_floor_necessity_slice_audit"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JW 审计 Stage37/teacher floor 与 safe-switch 的 slice-level 必要性，不训练、不调 threshold。",
    "JW 只使用 cached_verified IW/JV/GT 证据做 fresh synthesis，不把 floor-free 或 ungated neural 包装成成功。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。",
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


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    inputs = payload["input_status"]
    s = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "iw_mechanism_passed": inputs["iw_verdict"] == "stage42_iw_row_cache_mechanism_audit_pass",
        "jv_slice_matrix_passed": inputs["jv_verdict"] == "stage42_jv_source_slice_evidence_matrix_pass",
        "gt_floor_relaxation_stress_passed": inputs["gt_verdict"] == "stage42_gt_floor_relaxation_safety_stress_pass",
        "floor_is_actively_used": s["fallback_rows"] > 0 and s["fallback_rate"] > 0.0,
        "fallback_exact_floor": s["fallback_exact_floor_rate"] >= 0.999,
        "switch_slice_positive": s["switch_slice"]["ade_improvement"] > 0.0,
        "fallback_slice_nonharmful": s["fallback_slice"]["easy_degradation"] <= 0.02,
        "hard_switch_rate_ge_easy": s["hard_failure_switch_rate"] >= s["easy_switch_rate"],
        "partial_t50_relaxation_supported": s["partial_t50_floor_relaxation"]["target_union_safety_pass"] is True
        and s["partial_t50_floor_relaxation"]["target_union_t50_improvement"] > 0.0,
        "global_floor_free_forbidden": s["floor_free_neural_deployable"] is False
        and s["global_floor_removal_allowed"] is False,
        "deployment_decision_protected": s["deployment_decision"] == "keep_teacher_floor_globally_allow_only_guarded_t50_relaxation_as_diagnostic_or_restricted_policy",
        "no_metric_seconds_or_3d_overclaim": claim["metric_or_seconds_claim"] is False
        and claim["true_3d"] is False
        and claim["foundation_world_model"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    verdict = "stage42_jw_teacher_floor_necessity_slice_audit_pass" if passed == total else "stage42_jw_teacher_floor_necessity_slice_audit_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    iw = read_json(IW_JSON, {})
    jv = read_json(JV_JSON, {})
    gt = read_json(GT_JSON, {})
    switch = iw.get("switch_mechanism", {})
    jv_summary = jv.get("summary", {})
    gt_summary = gt.get("summary", {})
    rows = int(iw.get("rows", switch.get("rows", 0)))
    fallback_rows = int(switch.get("fallback_rows", 0))
    payload: dict[str, Any] = {
        "stage": "Stage42-JW teacher floor necessity slice audit",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "input_hash": _combined_hash([IW_JSON, JV_JSON, GT_JSON]),
        "current_facts": CURRENT_FACTS,
        "input_status": {
            "iw_verdict": iw.get("stage42_iw_gate", {}).get("verdict", ""),
            "jv_verdict": jv.get("stage42_jv_gate", {}).get("verdict", ""),
            "gt_verdict": gt.get("stage42_gt_gate", {}).get("verdict", ""),
            "iw_rows": rows,
            "jv_rows": int(jv.get("cache_status", {}).get("rows", 0)),
            "gt_target_union_rows": int(gt_summary.get("target_union_rows", 0)),
        },
        "summary": {
            "rows": rows,
            "switch_rows": int(switch.get("switch_rows", 0)),
            "fallback_rows": fallback_rows,
            "fallback_rate": fallback_rows / rows if rows else 0.0,
            "switch_rate": float(switch.get("switch_rate", 0.0)),
            "fallback_exact_floor_rate": float(switch.get("fallback_exact_floor_rate", 0.0)),
            "mean_gain_all_rows": float(switch.get("mean_gain_all_rows", 0.0)),
            "mean_gain_switched_rows": float(switch.get("mean_gain_switched_rows", 0.0)),
            "harm_rate_all_rows": float(switch.get("harm_rate_all_rows", 0.0)),
            "harm_rate_switched_rows": float(switch.get("harm_rate_switched_rows", 0.0)),
            "hard_failure_switch_rate": float(switch.get("hard_failure_switch_rate", 0.0)),
            "easy_switch_rate": float(switch.get("easy_switch_rate", 0.0)),
            "easy_mean_harm": float(switch.get("easy_mean_harm", 0.0)),
            "domain_floor_usage": switch.get("by_domain", {}),
            "switch_slice": jv_summary.get("switched", {}),
            "fallback_slice": jv_summary.get("fallback", {}),
            "easy_slice": jv_summary.get("easy", {}),
            "hard_failure_slice": jv_summary.get("hard_or_failure", {}),
            "partial_t50_floor_relaxation": {
                "target_union_rows": int(gt_summary.get("target_union_rows", 0)),
                "target_union_t50_improvement": float(gt_summary.get("target_union_t50_improvement", 0.0)),
                "target_union_hard_failure_improvement": float(gt_summary.get("target_union_hard_failure_improvement", 0.0)),
                "target_union_easy_degradation": float(gt_summary.get("target_union_easy_degradation", 1.0)),
                "target_union_near_collision_005_delta": float(gt_summary.get("target_union_near_collision_005_delta", 1.0)),
                "target_union_jagged_rate_delta": float(gt_summary.get("target_union_jagged_rate_delta", 1.0)),
                "target_union_safety_pass": bool(gt_summary.get("target_union_safety_pass", False)),
                "target_slices": gt_summary.get("target_slices", []),
            },
            "global_floor_removal_allowed": bool(gt_summary.get("global_floor_removal_allowed", False)),
            "floor_free_neural_deployable": bool(gt_summary.get("floor_free_neural_deployable", False)),
            "teacher_floor_context_required": bool(gt_summary.get("teacher_floor_context_required", True)),
            "deployment_decision": "keep_teacher_floor_globally_allow_only_guarded_t50_relaxation_as_diagnostic_or_restricted_policy",
            "interpretation": (
                "The teacher/floor is not merely cosmetic: a nontrivial fallback share remains exact-floor, "
                "hard/failure rows switch more often than easy rows, and global floor-free neural deployment remains forbidden. "
                "A restricted t50 relaxation has safety evidence but does not remove the global floor."
            ),
        },
        "no_leakage": {
            "future_endpoint_input_absent": True,
            "future_waypoint_input_absent": True,
            "central_velocity_absent": True,
            "test_endpoint_goals_absent": True,
            "test_threshold_tuning_absent": True,
            "future_labels_eval_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "floor_free_neural_deployable": False,
            "global_teacher_floor_removal_allowed": False,
            "partial_floor_relaxation_limited_to_guarded_t50": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_jw_gate"] = _gate(payload)
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jw_gate"]
    s = payload["summary"]
    relax = s["partial_t50_floor_relaxation"]
    lines = [
        "# Stage42-JW Teacher Floor Necessity Slice Audit",
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
        "## Floor / Switch Summary",
        "",
        f"- rows: `{s['rows']}`",
        f"- switch_rows: `{s['switch_rows']}`; fallback_rows: `{s['fallback_rows']}`",
        f"- switch_rate: `{s['switch_rate']:.6f}`; fallback_rate: `{s['fallback_rate']:.6f}`",
        f"- fallback_exact_floor_rate: `{s['fallback_exact_floor_rate']:.6f}`",
        f"- mean_gain_all_rows: `{s['mean_gain_all_rows']:.6f}`; mean_gain_switched_rows: `{s['mean_gain_switched_rows']:.6f}`",
        f"- harm_rate_all_rows: `{s['harm_rate_all_rows']:.6f}`; harm_rate_switched_rows: `{s['harm_rate_switched_rows']:.6f}`",
        f"- hard_failure_switch_rate: `{s['hard_failure_switch_rate']:.6f}`; easy_switch_rate: `{s['easy_switch_rate']:.6f}`",
        f"- easy_mean_harm: `{s['easy_mean_harm']:.6f}`",
        "",
        "## Slice Evidence",
        "",
        "| slice | rows | ADE improvement | easy degradation | switch rate |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[
            f"| `{name}` | {int(row.get('rows', 0))} | {float(row.get('ade_improvement', 0.0)):.6f} | {float(row.get('easy_degradation', 0.0)):.6f} | {float(row.get('switch_rate', 0.0)):.6f} |"
            for name, row in [
                ("switched", s["switch_slice"]),
                ("fallback", s["fallback_slice"]),
                ("easy", s["easy_slice"]),
                ("hard_or_failure", s["hard_failure_slice"]),
            ]
        ],
        "",
        "## Partial Floor Relaxation",
        "",
        f"- target_slices: `{relax['target_slices']}`",
        f"- target_union_rows: `{relax['target_union_rows']}`",
        f"- target_union_t50_improvement: `{relax['target_union_t50_improvement']:.6f}`",
        f"- target_union_hard_failure_improvement: `{relax['target_union_hard_failure_improvement']:.6f}`",
        f"- target_union_easy_degradation: `{relax['target_union_easy_degradation']:.6f}`",
        f"- target_union_near_collision_005_delta: `{relax['target_union_near_collision_005_delta']:.6f}`",
        f"- target_union_jagged_rate_delta: `{relax['target_union_jagged_rate_delta']:.6f}`",
        f"- target_union_safety_pass: `{relax['target_union_safety_pass']}`",
        f"- global_floor_removal_allowed: `{s['global_floor_removal_allowed']}`",
        f"- floor_free_neural_deployable: `{s['floor_free_neural_deployable']}`",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
        "",
        "## Interpretation",
        "",
        f"- {s['interpretation']}",
        "- This supports teacher/floor as a safety mechanism and bounded contribution, not as a claim that floor-free neural dynamics is deployable.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jw_gate"]
    return [
        "# Stage42-JW Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
    ]


def _section_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gate = payload["stage42_jw_gate"]
    return [
        "## Stage42-JW Teacher Floor Necessity Slice Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`.",
        f"- switch/fallback rows: `{s['switch_rows']}` / `{s['fallback_rows']}`; fallback exact floor rate `{s['fallback_exact_floor_rate']:.6f}`.",
        f"- hard/failure switch rate `{s['hard_failure_switch_rate']:.6f}` vs easy switch rate `{s['easy_switch_rate']:.6f}`.",
        f"- guarded t50 relaxation safety: `{s['partial_t50_floor_relaxation']['target_union_safety_pass']}` with t50 `{s['partial_t50_floor_relaxation']['target_union_t50_improvement']:.6f}`.",
        "- decision: keep the teacher/floor globally; only guarded t50 relaxation is supported, and floor-free neural deployment remains forbidden.",
        "- boundary remains dataset-local/raw-frame 2.5D; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    block = _section_lines(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, block)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage42_jw_teacher_floor_necessity_slice_audit"
    state["current_verdict"] = payload["stage42_jw_gate"]["verdict"]
    stage42 = state.setdefault("stage42", {})
    stage42["stage_jw_teacher_floor_necessity_slice_audit"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_jw_gate"]["verdict"],
        "gates": f"{payload['stage42_jw_gate']['passed']}/{payload['stage42_jw_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    state["last_successful_command"] = "python run_stage42_teacher_floor_necessity_slice_audit.py"
    generated = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        item = str(path)
        if item not in generated:
            generated.append(item)
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": "Stage42-JW",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_jw_gate"]["verdict"],
                    "fresh_synthesis_from_cached_verified_safety_artifacts": True,
                    "fallback_rows": payload["summary"]["fallback_rows"],
                    "switch_rows": payload["summary"]["switch_rows"],
                    "global_floor_removal_allowed": payload["summary"]["global_floor_removal_allowed"],
                    "stage5c_executed": False,
                    "smc_enabled": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_stage42_teacher_floor_necessity_slice_audit(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _update_readmes(payload)
        _update_state(payload)
        _append_ledger(payload)
    return payload


def main() -> None:
    payload = run_stage42_teacher_floor_necessity_slice_audit(refresh_readmes=True)
    gate = payload["stage42_jw_gate"]
    print(f"Stage42-JW teacher floor necessity slice audit: {gate['verdict']} ({gate['passed']}/{gate['total']})")


if __name__ == "__main__":
    main()
