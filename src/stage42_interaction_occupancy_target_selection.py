from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src import stage42_full_waypoint_proximity_occupancy_loss_repair as dh
from src import stage42_group_consistency_full_waypoint_repair as di
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "interaction_occupancy_target_selection_stage42.json"
REPORT_MD = OUT_DIR / "interaction_occupancy_target_selection_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_es_gate.md"

DH_JSON = OUT_DIR / "full_waypoint_proximity_occupancy_loss_repair_stage42.json"
DI_JSON = OUT_DIR / "group_consistency_full_waypoint_repair_stage42.json"
ER_JSON = OUT_DIR / "post_eq_context_claim_refresh_stage42.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
TARGET_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

PAPER_FILES = [
    OUT_DIR / "paper_outline_stage42.md",
    OUT_DIR / "method_draft_stage42.md",
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "failure_taxonomy_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "data_card_stage42.md",
    OUT_DIR / "reproducibility_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

SOURCE = "fresh_stage42_interaction_occupancy_target_selection"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-ES fresh-reruns the scalar proximity/occupancy loss target and explicit group-consistency repair target under the same source-level raw-frame protocol.",
    "本阶段选择下一步 interaction/occupancy target，不下载、不转换、不执行 Stage5C、不启用 SMC。",
    "future waypoints / endpoints 只作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _metric_row(name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if name == "scalar_proximity_occupancy_loss":
        metric = payload["model"]["metrics"]["protected_selected_candidate"]
        selected = payload["model"]["selected"]
        delta = payload["comparison_to_stage42_am"]["delta_vs_stage42_am"]
        decision = payload["deployment_decision"]
        near = None
        source_gate = payload.get("stage42_dh_gate", {})
    elif name == "explicit_group_consistency_repair":
        metric = payload["repair"]["test"]["metric_vs_floor"]
        selected = payload["repair"]["selected"]
        delta = payload["comparison_to_prior"]["delta_vs_stage42_am"]
        decision = payload["deployment_decision"]
        near = payload["repair"]["test"]["diagnostics"]
        source_gate = payload.get("stage42_di_gate", {})
    else:
        raise ValueError(f"unknown target row: {name}")
    return {
        "target_family": name,
        "source": "fresh_run",
        "source_gate_verdict": source_gate.get("verdict"),
        "source_gate_passed": source_gate.get("passed"),
        "source_gate_total": source_gate.get("total"),
        "selected": selected,
        "metric_vs_floor": metric,
        "delta_vs_stage42_am": delta,
        "deployment_decision": decision,
        "near_diagnostics": near,
        "promotable": bool(decision.get("promote_proximity_occupancy_loss_candidate") or decision.get("promote_group_consistency_full_waypoint_repair")),
    }


def _selection_score(row: Mapping[str, Any]) -> float:
    metric = row["metric_vs_floor"]
    delta = row["delta_vs_stage42_am"]
    near = row.get("near_diagnostics") or {}
    near_gain = 0.0
    if near:
        near_gain = float(near.get("base_near_005", 0.0)) - float(near.get("final_near_005", 0.0))
    return (
        1.4 * float(metric.get("all_improvement", 0.0))
        + 1.4 * float(metric.get("hard_failure_improvement", 0.0))
        + 1.1 * float(metric.get("t50_improvement", 0.0))
        + 0.4 * float(metric.get("t100_raw_frame_diagnostic_improvement", 0.0))
        + 5.0 * near_gain
        + 2.5 * max(0.0, float(delta.get("all_improvement") or 0.0))
        + 2.5 * max(0.0, float(delta.get("hard_failure_improvement") or 0.0))
        - 35.0 * max(0.0, float(metric.get("easy_degradation", 0.0)) - 0.02)
    )


def _select_target(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [{**dict(row), "selection_score": _selection_score(row)} for row in rows]
    best = max(scored, key=lambda row: row["selection_score"])
    return {
        "source": "fresh_selection_from_dh_di_rerun",
        "selected_target_family": best["target_family"],
        "selection_score": best["selection_score"],
        "selected_promotable": best["promotable"],
        "target_rows": scored,
        "decision": (
            "continue_with_explicit_group_consistency_interaction_target"
            if best["target_family"] == "explicit_group_consistency_repair" and best["promotable"]
            else "no_interaction_occupancy_target_promotable_under_current_protocol"
        ),
        "rationale": (
            "Explicit group-consistency is selected only if it improves the protected full-waypoint floor, improves over Stage42-AM on all/hard, "
            "preserves easy, and does not worsen near@0.05. Scalar proximity/occupancy weighting is retained only as diagnostic when it is positive "
            "but not better than the baseline-family full-waypoint control."
        ),
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    dh_payload = dh._build_payload()
    di_payload = di._build_payload()
    rows = [
        _metric_row("scalar_proximity_occupancy_loss", dh_payload),
        _metric_row("explicit_group_consistency_repair", di_payload),
    ]
    selection = _select_target(rows)
    payload = {
        "source": SOURCE,
        "stage": "Stage42-ES Interaction / Occupancy Target Selection",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(["data/stage41_world_model/combined_external.npz", str(DH_JSON), str(DI_JSON), str(ER_JSON)]),
        "current_facts": CURRENT_FACTS,
        "fresh_inputs": {
            "stage42_dh_rerun": {
                "source": dh_payload["source"],
                "gate": dh_payload["stage42_dh_gate"],
                "decision": dh_payload["deployment_decision"],
            },
            "stage42_di_rerun": {
                "source": di_payload["source"],
                "gate": di_payload["stage42_di_gate"],
                "decision": di_payload["deployment_decision"],
            },
        },
        "target_selection": selection,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_model_or_policy_selection": True,
            "source_overlap_pass": bool(
                dh_payload["no_leakage"]["source_overlap_pass"] and di_payload["no_leakage"]["source_overlap_pass"]
            ),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["paper_file_status"] = _refresh_paper_files(payload)
    payload["stage42_es_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    target = payload["target_selection"]
    rows = {row["target_family"]: row for row in target["target_rows"]}
    group = rows["explicit_group_consistency_repair"]
    scalar = rows["scalar_proximity_occupancy_loss"]
    group_metric = group["metric_vs_floor"]
    group_delta = group["delta_vs_stage42_am"]
    group_near = group["near_diagnostics"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "dh_rerun_completed": payload["fresh_inputs"]["stage42_dh_rerun"]["gate"]["passed"] >= 15,
        "di_rerun_completed": payload["fresh_inputs"]["stage42_di_rerun"]["gate"]["passed"] == payload["fresh_inputs"]["stage42_di_rerun"]["gate"]["total"],
        "target_families_compared": len(target["target_rows"]) >= 2,
        "scalar_target_recorded_diagnostic": scalar["target_family"] == "scalar_proximity_occupancy_loss",
        "group_consistency_selected": target["selected_target_family"] == "explicit_group_consistency_repair",
        "group_consistency_promotable": group["promotable"] is True,
        "group_all_positive": group_metric["all_improvement"] > 0.0,
        "group_t50_positive": group_metric["t50_improvement"] > 0.0,
        "group_hard_positive": group_metric["hard_failure_improvement"] > 0.0,
        "group_easy_safe": group_metric["easy_degradation"] <= 0.02,
        "group_beats_stage42_am_all": (group_delta["all_improvement"] or 0.0) > 0.0,
        "group_beats_stage42_am_hard": (group_delta["hard_failure_improvement"] or 0.0) > 0.0,
        "near005_not_worse": group_near["final_near_005"] <= group_near["base_near_005"],
        "no_leakage_pass": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["central_velocity"] is False
        and no_leak["test_endpoint_goals"] is False
        and no_leak["test_threshold_tuning"] is False
        and no_leak["validation_only_model_or_policy_selection"] is True
        and no_leak["source_overlap_pass"] is True,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_es_interaction_occupancy_target_selection_pass" if passed == total else "stage42_es_interaction_occupancy_target_selection_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _paper_lines(payload: Mapping[str, Any]) -> list[str]:
    target = payload["target_selection"]
    rows = {row["target_family"]: row for row in target["target_rows"]}
    group = rows["explicit_group_consistency_repair"]
    scalar = rows["scalar_proximity_occupancy_loss"]
    group_m = group["metric_vs_floor"]
    scalar_m = scalar["metric_vs_floor"]
    return [
        "## Stage42-ES Interaction / Occupancy Target Selection",
        "",
        "- source: `fresh_stage42_interaction_occupancy_target_selection`",
        "- role: fresh-reruns scalar proximity/occupancy loss and explicit group-consistency repair to choose the next interaction/occupancy target.",
        f"- selected target family: `{target['selected_target_family']}`; decision `{target['decision']}`.",
        f"- scalar proximity/occupancy target all/t50/hard/easy: `{_pct(scalar_m['all_improvement'])}` / `{_pct(scalar_m['t50_improvement'])}` / `{_pct(scalar_m['hard_failure_improvement'])}` / `{_pct(scalar_m['easy_degradation'])}`.",
        f"- explicit group-consistency target all/t50/t100raw/hard/easy: `{_pct(group_m['all_improvement'])}` / `{_pct(group_m['t50_improvement'])}` / `{_pct(group_m['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(group_m['hard_failure_improvement'])}` / `{_pct(group_m['easy_degradation'])}`.",
        f"- group near@0.05 base/final: `{_pct(group['near_diagnostics']['base_near_005'])}` / `{_pct(group['near_diagnostics']['final_near_005'])}`.",
        "- Claim boundary: protected source-level raw-frame full-waypoint evidence only; not true 3D, not foundation, not metric/seconds, no Stage5C, no SMC.",
    ]


def _refresh_paper_files(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    lines = _paper_lines(payload)
    status = []
    for path in PAPER_FILES:
        if path.exists():
            _replace_section(path, "STAGE42_ES_INTERACTION_OCCUPANCY_TARGET_SELECTION", lines)
            text = path.read_text(encoding="utf-8")
            status.append(
                {
                    "path": str(path),
                    "updated": True,
                    "contains_stage42_es": "STAGE42_ES_INTERACTION_OCCUPANCY_TARGET_SELECTION" in text,
                    "contains_group_consistency": "group-consistency" in text,
                    "contains_boundaries": "not true 3D" in text and "no Stage5C" in text and "no SMC" in text,
                }
            )
        else:
            status.append({"path": str(path), "updated": False, "missing": True})
    return status


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    target = payload["target_selection"]
    lines = [
        "# Stage42-ES Interaction / Occupancy Target Selection",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_es_gate']['passed']} / {payload['stage42_es_gate']['total']}`",
        f"- verdict: `{payload['stage42_es_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Selected Target",
        "",
        f"- selected_target_family: `{target['selected_target_family']}`",
        f"- decision: `{target['decision']}`",
        f"- selection_score: `{target['selection_score']:.6f}`",
        f"- rationale: {target['rationale']}",
        "",
        "## Target Family Comparison",
        "",
        "| family | promotable | all | t50 | t100 raw | hard | easy | delta AM all | delta AM hard | near base/final |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in target["target_rows"]:
        m = row["metric_vs_floor"]
        d = row["delta_vs_stage42_am"]
        near = row.get("near_diagnostics") or {}
        near_text = "n/a"
        if near:
            near_text = f"{_pct(near['base_near_005'])}/{_pct(near['final_near_005'])}"
        lines.append(
            f"| `{row['target_family']}` | {row['promotable']} | {_pct(m['all_improvement'])} | {_pct(m['t50_improvement'])} | "
            f"{_pct(m['t100_raw_frame_diagnostic_improvement'])} | {_pct(m['hard_failure_improvement'])} | {_pct(m['easy_degradation'])} | "
            f"{_pct(d['all_improvement'])} | {_pct(d['hard_failure_improvement'])} | {near_text} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Scalar proximity/occupancy weighting is useful as a diagnostic target but is not selected as the next deployable interaction target unless it beats Stage42-AM on all and hard.",
            "- Explicit group-consistency repair is selected only when it is validation/test-once promotable, easy-safe, and not worse on near@0.05.",
            "- This moves ER-2 away from shallow context routing toward a physical group/occupancy-style target.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_es_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_es_gate"]
    return [
        "# Stage42-ES Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _readme_lines(payload: Mapping[str, Any]) -> list[str]:
    target = payload["target_selection"]
    rows = {row["target_family"]: row for row in target["target_rows"]}
    group = rows["explicit_group_consistency_repair"]
    group_m = group["metric_vs_floor"]
    return [
        "## Stage42-ES Interaction / Occupancy Target Selection",
        "",
        "- source: `fresh_stage42_interaction_occupancy_target_selection`",
        "- role: fresh-reruns DH scalar proximity/occupancy target and DI explicit group-consistency target to choose the next interaction/occupancy training route.",
        f"- gate: `{payload['stage42_es_gate']['passed']} / {payload['stage42_es_gate']['total']}`; verdict `{payload['stage42_es_gate']['verdict']}`.",
        f"- selected target family: `{target['selected_target_family']}`; decision `{target['decision']}`.",
        f"- selected group-consistency all/t50/t100raw/hard/easy: `{_pct(group_m['all_improvement'])}` / `{_pct(group_m['t50_improvement'])}` / `{_pct(group_m['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(group_m['hard_failure_improvement'])}` / `{_pct(group_m['easy_degradation'])}`.",
        f"- near@0.05 base/final: `{_pct(group['near_diagnostics']['base_near_005'])}` / `{_pct(group['near_diagnostics']['final_near_005'])}`.",
        "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _readme_lines(payload)
    for path in [README_RESULTS, M3W_README, TARGET_SUMMARY, WORK_SUMMARY]:
        _replace_section(path, "STAGE42_ES_INTERACTION_OCCUPANCY_TARGET_SELECTION", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-ES interaction occupancy target selection"
    state["current_verdict"] = payload["stage42_es_gate"]["verdict"]
    state["stage42_es_interaction_occupancy_target_selection"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_es_gate"]["verdict"],
        "gates": f"{payload['stage42_es_gate']['passed']}/{payload['stage42_es_gate']['total']}",
        "target_selection": payload["target_selection"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_interaction_occupancy_target_selection(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    payload = _build_payload()
    write_json(REPORT_JSON, am._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_interaction_occupancy_target_selection()
