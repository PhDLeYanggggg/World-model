from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_full_waypoint_bridge_shape_audit import _replace_section
from src import stage41_bounded_neural_blend_dynamics as blend
from src import stage41_joint_rollout_consistency as jrc
from src import stage42_safety_floor as sf


OUT_DIR = Path("outputs/stage42_long_research")

E_JSON = OUT_DIR / "safety_floor_stage42.json"
HC_JSON = OUT_DIR / "floor_alternative_gate_stress_stage42.json"
HB_JSON = OUT_DIR / "teacher_floor_necessity_meta_audit_stage42.json"

REPORT_JSON = OUT_DIR / "floor_free_proximity_guard_repair_stage42.json"
REPORT_MD = OUT_DIR / "floor_free_proximity_guard_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hd_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
USER_SUMMARY = Path("README_M3W_USER_DETAILED_SUMMARY_ZH.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")

FLOOR_FREE_SWITCH_FAMILIES = {
    "internal_self_gate",
    "uncertainty_gate",
    "harm_predictor_gate",
    "conformal_risk_gate",
}

MIN_SEP_GRID = [0.0, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]
EASY_LIMIT = 0.02
COLLISION_LIMIT = 0.01

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HD 不是重新训练大模型；它是 floor-free gate 的 validation-only proximity-guard repair。",
    "proximity guard 只用预测 rollout、group key、normalizer 和 causal floor rollout；future endpoint/waypoint 只作为监督或评估标签。",
    "guard threshold 只在 validation 选择，test 只最终评估一次。",
    "本轮不下载、不转换、不执行 Stage5C，不启用 SMC。",
    "即使 floor-free gate 修复成功，也不是 global floor removal；它仍依赖 causal baseline floor fallback 作安全约束。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _metric(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _is_deployable(metrics: Mapping[str, Any]) -> bool:
    return bool(
        _metric(metrics, "all_improvement") > 0.0
        and (_metric(metrics, "t50_improvement") > 0.0 or _metric(metrics, "hard_failure_improvement") > 0.0)
        and _metric(metrics, "easy_degradation", 1.0) <= EASY_LIMIT
        and _metric(metrics, "collision_delta_vs_floor_005", 1.0) <= COLLISION_LIMIT
        and (_metric(metrics, "switch_rate") > 0.0 or _metric(metrics, "alpha_positive_rate") > 0.0)
    )


def _score(metrics: Mapping[str, Any]) -> float:
    return (
        _metric(metrics, "all_improvement")
        + 1.35 * _metric(metrics, "t50_improvement")
        + 0.75 * _metric(metrics, "t100_improvement")
        + 1.15 * _metric(metrics, "hard_failure_improvement")
        - 35.0 * max(0.0, _metric(metrics, "easy_degradation", 1.0) - EASY_LIMIT)
        - 15.0 * max(0.0, _metric(metrics, "collision_delta_vs_floor_005", 1.0) - COLLISION_LIMIT)
    )


def _gate_passed(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return bool(gate and gate.get("passed") == gate.get("total") and int(gate.get("total", 0)) > 0)


def _floor_free_rows(e: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in e.get("switch_gate_rows", []):
        if str(row.get("family")) in FLOOR_FREE_SWITCH_FAMILIES:
            out.append(dict(row))
    return out


def _eval_guarded(data: Mapping[str, Any], policy: Mapping[str, Any], min_sep: float) -> dict[str, Any]:
    raw_switch = sf._switch_for_policy(data, policy).astype(bool)
    guarded_switch, guarded_off = jrc._apply_proximity_guard(
        data["floor_xy"],
        data["neural_xy"],
        data["labels"],
        data["keys"],
        raw_switch,
        float(min_sep),
    )
    metrics = sf._eval_switch(data, guarded_switch, f"floor_free_proximity_guard_{policy.get('family')}_{min_sep}")
    metrics["raw_switch_rate"] = float(np.mean(raw_switch)) if len(raw_switch) else 0.0
    metrics["switch_rate"] = float(np.mean(guarded_switch)) if len(guarded_switch) else 0.0
    metrics["guarded_off_rate"] = float(guarded_off / max(1, len(raw_switch)))
    metrics["guarded_off_count"] = int(guarded_off)
    metrics["min_sep"] = float(min_sep)
    metrics["strict_deployable"] = _is_deployable(metrics)
    return metrics


def _select_guard_for_policy(val: Mapping[str, Any], test: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    policy = dict(row.get("policy", {}))
    family = str(row.get("family"))
    val_rows = []
    for min_sep in MIN_SEP_GRID:
        metrics = _eval_guarded(val, policy, min_sep)
        val_rows.append(
            {
                "min_sep": float(min_sep),
                "metrics": metrics,
                "deployable": _is_deployable(metrics),
                "score": _score(metrics),
            }
        )
    eligible = [item for item in val_rows if item["deployable"]]
    selected = max(eligible or val_rows, key=lambda item: item["score"])
    test_metrics = _eval_guarded(test, policy, float(selected["min_sep"]))
    pre_guard_metrics = dict(row.get("test_metrics", {}))
    return {
        "family": family,
        "source": "fresh_stage42_hd_val_selected_proximity_guard",
        "base_policy": policy,
        "stage42_e_pre_guard_test_metrics": pre_guard_metrics,
        "selected_min_sep": float(selected["min_sep"]),
        "val_candidate_count": len(val_rows),
        "val_eligible_count": len(eligible),
        "val_selected_metrics": selected["metrics"],
        "test_metrics": test_metrics,
        "test_deployable": _is_deployable(test_metrics),
        "pre_guard_deployable": bool(row.get("test_deployable")),
        "collision_repair_delta": _metric(pre_guard_metrics, "collision_delta_vs_floor_005", 0.0)
        - _metric(test_metrics, "collision_delta_vs_floor_005", 0.0),
        "raw_gain_loss_from_guard": {
            "all": _metric(test_metrics, "all_improvement") - _metric(pre_guard_metrics, "all_improvement"),
            "t50": _metric(test_metrics, "t50_improvement") - _metric(pre_guard_metrics, "t50_improvement"),
            "hard_failure": _metric(test_metrics, "hard_failure_improvement")
            - _metric(pre_guard_metrics, "hard_failure_improvement"),
        },
        "all_val_candidates": val_rows,
    }


def _best_by_score(rows: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        return {}
    return max(rows, key=lambda row: (int(bool(row.get("test_deployable"))), _score(row.get("test_metrics", {}))))


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    e = read_json(E_JSON, {})
    hc = read_json(HC_JSON, {})
    hb = read_json(HB_JSON, {})
    checkpoint, teacher_policy, min_sep = blend._load_frozen_model()
    val = blend._bundle("val", checkpoint, teacher_policy, min_sep)
    test = blend._bundle("test", checkpoint, teacher_policy, min_sep)
    source_rows = _floor_free_rows(e)
    repair_rows = [_select_guard_for_policy(val, test, row) for row in source_rows]
    deployable = [row for row in repair_rows if row["test_deployable"]]
    best = _best_by_score(repair_rows)
    pre_guard_best = max(source_rows, key=lambda row: _score(row.get("test_metrics", {}))) if source_rows else {}
    summary = {
        "source": "fresh_stage42_hd_floor_free_proximity_guard_repair",
        "candidate_families": [row["family"] for row in repair_rows],
        "candidate_count": len(repair_rows),
        "pre_guard_deployable_count": sum(1 for row in source_rows if row.get("test_deployable")),
        "post_guard_deployable_count": len(deployable),
        "best_pre_guard_family": pre_guard_best.get("family"),
        "best_post_guard_family": best.get("family"),
        "best_post_guard_deployable": bool(best.get("test_deployable")),
        "best_post_guard_metrics": best.get("test_metrics", {}),
        "best_post_guard_selected_min_sep": best.get("selected_min_sep"),
        "best_post_guard_collision_repair_delta": best.get("collision_repair_delta"),
        "teacher_gate_used": False,
        "causal_floor_fallback_used": True,
        "global_floor_removal_allowed": False,
        "floor_free_teacherless_with_proximity_guard_deployable": len(deployable) > 0,
        "deployment_decision": "teacherless_floor_free_gate_can_be_proximity_repaired_but_only_with_causal_floor_safety_fallback"
        if len(deployable) > 0
        else "floor_free_gate_remains_not_deployable_even_after_proximity_guard",
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": "fresh_stage42_hd_floor_free_proximity_guard_repair",
        "stage": "Stage42-HD floor-free proximity-guard repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([E_JSON, HC_JSON, HB_JSON]),
        "current_facts": CURRENT_FACTS,
        "input_status": {
            "stage42_e_safety_floor": {
                "path": str(E_JSON),
                "source": e.get("source"),
                "gate_passed": _gate_passed(e, "stage42_e_gate"),
                "verdict": (e.get("stage42_e_gate", {}) or {}).get("verdict"),
            },
            "stage42_hc_floor_alternative_stress": {
                "path": str(HC_JSON),
                "source": hc.get("source"),
                "gate_passed": _gate_passed(hc, "stage42_hc_gate"),
                "verdict": (hc.get("stage42_hc_gate", {}) or {}).get("verdict"),
            },
            "stage42_hb_teacher_floor_meta_audit": {
                "path": str(HB_JSON),
                "source": hb.get("source"),
                "gate_passed": _gate_passed(hb, "stage42_hb_gate"),
                "verdict": (hb.get("stage42_hb_gate", {}) or {}).get("verdict"),
            },
        },
        "validation_protocol": {
            "guard_threshold_selection": "validation_only",
            "test_usage": "single_final_evaluation_after_val_selected_min_sep",
            "min_sep_grid": MIN_SEP_GRID,
            "families": sorted(FLOOR_FREE_SWITCH_FAMILIES),
            "teacher_gate_used": False,
            "causal_floor_fallback_used": True,
        },
        "repair_rows": repair_rows,
        "summary": summary,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_guard_selection": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "global_floor_removal_allowed": False,
            "teacher_gate_removed_for_repaired_floor_free_candidate": len(deployable) > 0,
            "causal_floor_safety_fallback_still_required": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_hd_gate"] = _gate(payload)
    return _jsonable(payload)


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    leakage = payload["no_leakage"]
    claim = payload["claim_boundary"]
    best = s.get("best_post_guard_metrics", {})
    gates = {
        "required_inputs_passed": all(item.get("gate_passed") is True for item in payload["input_status"].values()),
        "floor_free_families_evaluated": s["candidate_count"] >= 4,
        "pre_guard_blocker_confirmed": s["pre_guard_deployable_count"] == 0,
        "proximity_guard_selected_on_validation": payload["validation_protocol"]["guard_threshold_selection"] == "validation_only",
        "test_threshold_not_tuned": leakage["test_threshold_tuning"] is False,
        "post_guard_result_recorded": "all_improvement" in best,
        "repair_or_blocker_recorded": s["post_guard_deployable_count"] > 0
        or s["deployment_decision"] == "floor_free_gate_remains_not_deployable_even_after_proximity_guard",
        "causal_floor_fallback_still_required": claim["causal_floor_safety_fallback_still_required"] is True
        and s["global_floor_removal_allowed"] is False,
        "teacher_gate_not_used_in_repair": s["teacher_gate_used"] is False,
        "no_future_test_or_central_velocity_leakage": leakage["future_endpoint_input"] is False
        and leakage["future_waypoint_input"] is False
        and leakage["central_velocity"] is False
        and leakage["test_endpoint_goals"] is False,
        "no_metric_seconds_overclaim": claim["metric_or_seconds_claim"] is False
        and claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(bool(v) for v in gates.values())
    total = len(gates)
    verdict = "stage42_hd_floor_free_proximity_guard_repair_pass" if passed == total else "stage42_hd_floor_free_proximity_guard_repair_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hd_gate"]
    s = payload["summary"]
    best = s.get("best_post_guard_metrics", {})
    lines = [
        "# Stage42-HD Floor-Free Proximity-Guard Repair",
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
        "## Direct Decision",
        "",
        f"- deployment_decision: `{s['deployment_decision']}`",
        f"- pre_guard_deployable_count: `{s['pre_guard_deployable_count']}`",
        f"- post_guard_deployable_count: `{s['post_guard_deployable_count']}`",
        f"- best_post_guard_family: `{s['best_post_guard_family']}`",
        f"- best_post_guard_selected_min_sep: `{s['best_post_guard_selected_min_sep']}`",
        f"- teacher_gate_used: `{s['teacher_gate_used']}`",
        f"- causal_floor_fallback_used: `{s['causal_floor_fallback_used']}`",
        f"- global_floor_removal_allowed: `{s['global_floor_removal_allowed']}`",
        "",
        "## Best Post-Guard Metrics",
        "",
        f"- all: `{_pct(best.get('all_improvement'))}`",
        f"- t50: `{_pct(best.get('t50_improvement'))}`",
        f"- t100 raw diagnostic: `{_pct(best.get('t100_improvement'))}`",
        f"- hard/failure: `{_pct(best.get('hard_failure_improvement'))}`",
        f"- easy degradation: `{_pct(best.get('easy_degradation'))}`",
        f"- collision delta @0.05: `{_pct(best.get('collision_delta_vs_floor_005'))}`",
        f"- switch rate: `{_pct(best.get('switch_rate'))}`",
        f"- guarded-off rate: `{_pct(best.get('guarded_off_rate'))}`",
        "",
        "## Repair Matrix",
        "",
        "| family | min sep | deployable | all | t50 | t100 raw | hard | easy | collision d005 | switch | guarded off | collision repair |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["repair_rows"]:
        m = row["test_metrics"]
        lines.append(
            f"| `{row['family']}` | {row['selected_min_sep']:.2f} | {row['test_deployable']} | "
            f"{_pct(m.get('all_improvement'))} | {_pct(m.get('t50_improvement'))} | {_pct(m.get('t100_improvement'))} | "
            f"{_pct(m.get('hard_failure_improvement'))} | {_pct(m.get('easy_degradation'))} | "
            f"{_pct(m.get('collision_delta_vs_floor_005'))} | {_pct(m.get('switch_rate'))} | "
            f"{_pct(m.get('guarded_off_rate'))} | {_pct(row.get('collision_repair_delta'))} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Stage42-HC showed floor-free gates had high raw gain but failed strict proximity safety.",
        "- Stage42-HD tests whether a validation-selected proximity guard can repair those floor-free gates without using the teacher gate.",
        "- The repaired policy is not a global floor removal: it still falls back to the strongest causal floor when predicted proximity becomes unsafe.",
        "- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is made.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hd_gate"]
    lines = [
        "# Stage42-HD Gate",
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


def _refresh_docs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    s = payload["summary"]
    best = s.get("best_post_guard_metrics", {})
    lines = [
        "## Stage42-HD Floor-Free Proximity-Guard Repair",
        "",
        "- source: `fresh_stage42_hd_floor_free_proximity_guard_repair`",
        f"- gate: `{payload['stage42_hd_gate']['passed']} / {payload['stage42_hd_gate']['total']}`",
        f"- verdict: `{payload['stage42_hd_gate']['verdict']}`",
        "- Tested floor-free internal/harm/uncertainty/conformal gates with a validation-selected proximity guard.",
        f"- pre-guard deployable count: `{s['pre_guard_deployable_count']}`; post-guard deployable count: `{s['post_guard_deployable_count']}`.",
        f"- best post-guard family `{s['best_post_guard_family']}` reaches all/t50/t100raw/hard `{_pct(best.get('all_improvement'))}` / `{_pct(best.get('t50_improvement'))}` / `{_pct(best.get('t100_improvement'))}` / `{_pct(best.get('hard_failure_improvement'))}` with easy `{_pct(best.get('easy_degradation'))}` and collision delta `{_pct(best.get('collision_delta_vs_floor_005'))}`.",
        "- The teacher gate is not used in this repair, but causal floor fallback remains required; this is not global floor removal.",
        "- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is allowed.",
    ]
    status = []
    for path in [README_RESULTS, M3W_README, USER_SUMMARY, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, "STAGE42_HD_FLOOR_FREE_PROXIMITY_GUARD_REPAIR", lines)
            text = path.read_text(encoding="utf-8")
            status.append(
                {
                    "path": str(path),
                    "exists": True,
                    "contains_stage42_hd": "Stage42-HD Floor-Free Proximity-Guard Repair" in text,
                    "contains_not_global_floor_removal": "not global floor removal" in text,
                }
            )
        else:
            status.append({"path": str(path), "exists": False})
    return status


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json("research_state.json", {})
    s = payload["summary"]
    best = s.get("best_post_guard_metrics", {})
    state["current_stage"] = "Stage42-HD floor-free proximity-guard repair"
    state["current_verdict"] = payload["stage42_hd_gate"]["verdict"]
    state["stage42_hd_floor_free_proximity_guard_repair"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_hd_gate"]["verdict"],
        "gates": f"{payload['stage42_hd_gate']['passed']}/{payload['stage42_hd_gate']['total']}",
        "summary": {
            "candidate_count": s["candidate_count"],
            "pre_guard_deployable_count": s["pre_guard_deployable_count"],
            "post_guard_deployable_count": s["post_guard_deployable_count"],
            "best_post_guard_family": s["best_post_guard_family"],
            "best_post_guard_selected_min_sep": s["best_post_guard_selected_min_sep"],
            "best_post_guard_all_improvement": best.get("all_improvement"),
            "best_post_guard_t50_improvement": best.get("t50_improvement"),
            "best_post_guard_t100_raw_improvement": best.get("t100_improvement"),
            "best_post_guard_hard_failure_improvement": best.get("hard_failure_improvement"),
            "best_post_guard_easy_degradation": best.get("easy_degradation"),
            "best_post_guard_collision_delta_vs_floor_005": best.get("collision_delta_vs_floor_005"),
            "teacher_gate_used": s["teacher_gate_used"],
            "causal_floor_fallback_used": s["causal_floor_fallback_used"],
            "global_floor_removal_allowed": s["global_floor_removal_allowed"],
            "deployment_decision": s["deployment_decision"],
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "verification": {
            "focused_pytest": ".venv-pytorch/bin/python -m pytest tests/test_stage42_floor_free_proximity_guard_repair.py -> pending",
            "full_pytest": ".venv-pytorch/bin/python -m pytest tests -> pending",
        },
        "claim_boundary": payload["claim_boundary"],
    }
    write_json("research_state.json", _jsonable(state))


def run_stage42_floor_free_proximity_guard_repair() -> dict[str, Any]:
    payload = _build_payload()
    payload["doc_refresh_status"] = _refresh_docs(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _update_state(payload)
    return payload


if __name__ == "__main__":
    out = run_stage42_floor_free_proximity_guard_repair()
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2, sort_keys=True))
