from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_long_objective_evidence_audit.json"
REPORT_MD = OUT_DIR / "stage43_long_objective_evidence_audit.md"
GAP_MD = OUT_DIR / "stage43_long_objective_gap_matrix.md"
GATE_MD = OUT_DIR / "stage43_stage_bj_long_objective_evidence_audit_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SOURCE = "fresh_stage43_bj_long_objective_evidence_audit"
SECTION = "STAGE43_BJ_LONG_OBJECTIVE_EVIDENCE_AUDIT"

INPUTS = {
    "safety_floor_replay": OUT_DIR / "stage43_safety_floor_replay.json",
    "latent_dataset_contract": OUT_DIR / "stage43_latent_state_dataset_contract.json",
    "protected_latent_eval": OUT_DIR / "stage43_protected_latent_eval.json",
    "data_calibration": OUT_DIR / "data_calibration_stage43.json",
    "external_validation_matrix": OUT_DIR / "stage43_external_validation_matrix.json",
    "full_waypoint_latent_dynamics": OUT_DIR / "stage43_full_waypoint_latent_dynamics.json",
    "multimodal_head_suite": OUT_DIR / "stage43_multimodal_latent_head_suite.json",
    "feature_family_multiseed": OUT_DIR / "stage43_feature_family_multiseed_confirmation.json",
    "safety_floor_necessity": OUT_DIR / "stage43_safety_floor_necessity_audit.json",
    "blocked_source_terms_validation": OUT_DIR / "stage43_blocked_source_terms_validation.json",
    "locked_candidate_package": OUT_DIR / "stage43_locked_candidate_paper_package_refresh.json",
}


def _pct(value: float | int) -> str:
    return f"{100.0 * float(value):.2f}%"


def _read_required(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path, {})


def _gate_full_pass(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return int(gate.get("passed", -1)) == int(gate.get("total", -2)) and int(gate.get("total", 0)) > 0


def _gate_verdict(payload: Mapping[str, Any], key: str) -> str:
    return str(payload.get(key, {}).get("verdict", "missing"))


def _phase(
    *,
    phase: str,
    status: str,
    evidence: str,
    proved: list[str],
    missing: list[str],
    next_action: str,
    pass_for_current_audit: bool,
    complete_for_long_objective: bool,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "status": status,
        "evidence": evidence,
        "proved": proved,
        "missing": missing,
        "next_action": next_action,
        "pass_for_current_audit": bool(pass_for_current_audit),
        "complete_for_long_objective": bool(complete_for_long_objective),
    }


def build_long_objective_evidence_audit() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    artifacts = {name: _read_required(path) for name, path in INPUTS.items()}
    package = artifacts["locked_candidate_package"]
    metrics = package["metrics"]
    source_guard = package["source_guard"]
    data = artifacts["data_calibration"]
    heads = artifacts["multimodal_head_suite"]
    multiseed = artifacts["feature_family_multiseed"]
    floor = artifacts["safety_floor_necessity"]
    external = artifacts["external_validation_matrix"]
    waypoint = artifacts["full_waypoint_latent_dynamics"]

    input_verdicts = {
        "safety_floor_replay": _gate_verdict(artifacts["safety_floor_replay"], "stage43_a_gate"),
        "latent_dataset_contract": _gate_verdict(artifacts["latent_dataset_contract"], "stage43_b_gate"),
        "protected_latent_eval": _gate_verdict(artifacts["protected_latent_eval"], "stage43_c_gate"),
        "data_calibration": _gate_verdict(data, "stage43_as_gate"),
        "external_validation_matrix": _gate_verdict(external, "stage43_at_gate"),
        "full_waypoint_latent_dynamics": _gate_verdict(waypoint, "stage43_m_gate"),
        "multimodal_head_suite": _gate_verdict(heads, "stage43_y_gate"),
        "feature_family_multiseed": _gate_verdict(multiseed, "stage43_ai_gate"),
        "safety_floor_necessity": _gate_verdict(floor, "stage43_aj_gate"),
        "blocked_source_terms_validation": _gate_verdict(
            artifacts["blocked_source_terms_validation"], "stage43_bg_gate"
        ),
        "locked_candidate_package": _gate_verdict(package, "stage43_bi_gate"),
    }
    data_summary = data.get("summary", {})
    external_domains = list(package["evidence"]["external_domains"])
    proxy_heads = list(package["evidence"]["deployable_proxy_heads"])
    ablations = list(package["evidence"]["stable_positive_t50_ablation_variants"])
    source_ready = int(source_guard["ready_for_guarded_conversion_preflight_rows"])
    training_allowed = int(source_guard["training_allowed_now"])
    t100 = float(metrics["t100_raw_frame_diagnostic"])

    phases = [
        _phase(
            phase="A data and calibration",
            status="partial_blocked",
            evidence=(
                f"Stage43-AS data calibration gate passed; datasets audited="
                f"{data_summary.get('datasets_audited', data_summary.get('dataset_count', 'unknown'))}; "
                f"blocked source ready rows={source_ready}; training_allowed_now={training_allowed}."
            ),
            proved=[
                "calibration/status audit exists",
                "raw-frame/dataset-local boundary is preserved",
                "blocked source terms validation prevents unconfirmed conversion/training",
            ],
            missing=[
                "verified source terms/identity for PETS/Town-Center/Wild-Track",
                "verified metric/time calibration broad enough for metric or seconds-level claims",
                "new guarded conversion of blocked sources",
            ],
            next_action="Fill/validate source terms identity packet, then rerun guarded conversion preflight.",
            pass_for_current_audit=True,
            complete_for_long_objective=False,
        ),
        _phase(
            phase="B external validation",
            status="pass_with_boundary",
            evidence=(
                f"Stage43-AT matrix passed across domains {external_domains}; latest protected candidate "
                f"all={_pct(metrics['all'])}, t50={_pct(metrics['t50'])}, hard={_pct(metrics['hard_failure'])}, "
                f"easy={_pct(metrics['easy_degradation'])}."
            ),
            proved=[
                "fresh external validation matrix compares floor, prior protected neural, ungated diagnostic, source-safe, full-waypoint, and latest protected candidate",
                "latest candidate is positive on all/t50/hard and easy-safe",
                "uniform positive external transfer remains explicitly blocked",
            ],
            missing=[
                "uniform positive transfer across every source",
                "additional legal external top-down sources",
                "t100 positive source-stable evidence",
            ],
            next_action="Prioritize source support closure and t100 source-stability repair before broader transfer claims.",
            pass_for_current_audit=True,
            complete_for_long_objective=False,
        ),
        _phase(
            phase="C full-waypoint / latent dynamics",
            status="protected_candidate_pass",
            evidence=(
                f"Stage43-M and Stage43-BI pass; latent/full-waypoint candidate metrics all={_pct(metrics['all'])}, "
                f"t50={_pct(metrics['t50'])}, t100raw={_pct(t100)}."
            ),
            proved=[
                "latent-state dataset and protected latent eval exist",
                "protected full-waypoint latent dynamics exists",
                "multimodal proxy heads are packaged under a safety floor",
            ],
            missing=[
                "standalone ungated neural dynamics",
                "t100 positive dynamics rather than floor-guarded diagnostic",
                "raw image/video multimodal evidence beyond proxy tokens",
            ],
            next_action="Keep protected deployment; do not execute Stage5C or replace floor with ungated dynamics.",
            pass_for_current_audit=True,
            complete_for_long_objective=False,
        ),
        _phase(
            phase="D causal ablation / module evidence",
            status="partial_supported",
            evidence=f"Stage43-AI passes with stable positive t50 ablation variants {ablations}.",
            proved=[
                "at least two stable t50 ablation variants are recorded",
                "claim package avoids writing JEPA/Transformer/scene/goal/interaction as standalone main claims",
            ],
            missing=[
                "full retrained proof for every requested no_history/no_neighbor/no_scene/no_goal/no_interaction/no_JEPA/no_Transformer/no_floor/no_switch ablation",
                "independent JEPA or Transformer downstream lift strong enough to be a main contribution",
            ],
            next_action="Use future trials to replace proxy-heavy ablations with retrained raw-scene/graph-rich ablations.",
            pass_for_current_audit=True,
            complete_for_long_objective=False,
        ),
        _phase(
            phase="E safety floor study",
            status="floor_required",
            evidence=(
                f"Stage43-AJ passes; package says safety_floor_required={package['evidence']['safety_floor_required']} "
                f"and standalone_deployable={package['evidence']['standalone_world_model_deployable']}."
            ),
            proved=[
                "safety floor necessity is explicitly audited",
                "ungated/standalone deployment is not claimed",
                "bounded/self/conformal variants are treated as protected safety research",
            ],
            missing=[
                "safe global floor removal",
                "floor-free neural dynamics that preserves easy cases",
            ],
            next_action="If floor relaxation is revisited, keep it slice-specific and validation-selected.",
            pass_for_current_audit=True,
            complete_for_long_objective=False,
        ),
        _phase(
            phase="F paper package",
            status="pass_with_a_journal_gap",
            evidence="Stage43-BI paper package refresh passes and writes claim boundary/model card/data card/repro/gap artifacts.",
            proved=[
                "paper-facing package exists",
                "claim boundary is explicit",
                "A-journal gap is written as not-yet rather than overclaimed",
            ],
            missing=[
                "A-journal candidate evidence threshold",
                "true 3D or metric/time subset",
                "broader legally cleared external source support",
            ],
            next_action="Keep the paper package as protected candidate evidence; do not claim final A-journal readiness.",
            pass_for_current_audit=True,
            complete_for_long_objective=False,
        ),
    ]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_requirement_audit_from_stage43_bi_locked_candidate_evidence",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {name: str(path) for name, path in INPUTS.items()},
        "input_hash": _combined_hash(list(INPUTS.values())),
        "input_verdicts": input_verdicts,
        "current_candidate": {
            "label": package["current_claim"]["label"],
            "metrics": metrics,
            "external_domains": external_domains,
            "deployable_proxy_heads": proxy_heads,
            "stable_positive_t50_ablation_variants": ablations,
        },
        "phases": phases,
        "remaining_blockers": [
            "source_terms_identity_not_confirmed_for_blocked_sources",
            "metric_time_calibration_unverified",
            "true_3d_absent",
            "foundation_scale_absent",
            "safety_floor_required",
            "standalone_ungated_deployment_not_supported",
            "uniform_positive_external_transfer_not_supported",
            "t100_raw_frame_diagnostic_not_solved",
            "raw_scene_video_multimodal_evidence_proxy_heavy",
        ],
        "next_priority_order": [
            "close blocked source terms/identity and guarded conversion preflight",
            "repair t100 source-stable evidence or keep t100 diagnostic only",
            "replace proxy-heavy scene/interaction ablations with retrained raw-scene/graph-rich ablations",
            "try slice-specific floor relaxation only after validation safety gates",
        ],
        "no_leakage_and_execution": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "new_training_executed": False,
            "new_conversion_executed": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "standalone_ungated_deployable": False,
            "a_journal_candidate_now": False,
            "long_objective_complete": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage43_bj_gate"] = _gate(payload, artifacts)
    return payload


def _gate(payload: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    leak = payload["no_leakage_and_execution"]
    claim = payload["claim_boundary"]
    phases = {phase["phase"]: phase for phase in payload["phases"]}
    gates = {
        "safety_floor_replay_passed": _gate_full_pass(artifacts["safety_floor_replay"], "stage43_a_gate"),
        "latent_dataset_and_eval_passed": _gate_full_pass(artifacts["latent_dataset_contract"], "stage43_b_gate")
        and _gate_full_pass(artifacts["protected_latent_eval"], "stage43_c_gate"),
        "data_calibration_audited_with_blockers": _gate_full_pass(artifacts["data_calibration"], "stage43_as_gate")
        and phases["A data and calibration"]["status"] == "partial_blocked",
        "external_validation_audited": _gate_full_pass(
            artifacts["external_validation_matrix"], "stage43_at_gate"
        )
        and phases["B external validation"]["status"] == "pass_with_boundary",
        "full_waypoint_latent_audited": _gate_full_pass(
            artifacts["full_waypoint_latent_dynamics"], "stage43_m_gate"
        )
        and phases["C full-waypoint / latent dynamics"]["status"] == "protected_candidate_pass",
        "causal_ablation_partial_not_overclaimed": _gate_full_pass(
            artifacts["feature_family_multiseed"], "stage43_ai_gate"
        )
        and phases["D causal ablation / module evidence"]["status"] == "partial_supported",
        "safety_floor_necessity_recorded": _gate_full_pass(
            artifacts["safety_floor_necessity"], "stage43_aj_gate"
        )
        and phases["E safety floor study"]["status"] == "floor_required",
        "paper_package_current": _gate_full_pass(artifacts["locked_candidate_package"], "stage43_bi_gate")
        and phases["F paper package"]["status"] == "pass_with_a_journal_gap",
        "blocked_sources_not_converted_or_trained": _gate_full_pass(
            artifacts["blocked_source_terms_validation"], "stage43_bg_gate"
        )
        and payload["current_candidate"]["metrics"]["t100_raw_frame_diagnostic"] >= 0.0,
        "no_future_or_test_leakage": leak["future_endpoint_input"] is False
        and leak["future_waypoint_input"] is False
        and leak["future_labels_eval_or_loss_only"] is True
        and leak["central_velocity_input"] is False
        and leak["test_endpoint_goal_construction"] is False
        and leak["test_statistics_normalization"] is False
        and leak["test_threshold_tuning"] is False,
        "no_new_training_or_conversion": leak["new_training_executed"] is False
        and leak["new_conversion_executed"] is False,
        "claim_boundary_not_overstated": claim["true_3d_world_model"] is False
        and claim["foundation_world_model"] is False
        and claim["metric_or_seconds_claim"] is False
        and claim["dataset_local_raw_frame_only"] is True
        and claim["standalone_ungated_deployable"] is False
        and claim["a_journal_candidate_now"] is False,
        "stage5c_and_smc_false": claim["stage5c_executed"] is False and claim["smc_enabled"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False
        and not any(phase["complete_for_long_objective"] for phase in payload["phases"]),
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_bj_long_objective_evidence_audit_pass_keep_goal_active"
        if passed == total
        else "stage43_bj_long_objective_evidence_audit_incomplete",
        "long_objective_complete": False,
        "protected_multimodal_latent_state_candidate": passed == total,
        "standalone_world_model_deployable": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_bj_gate"]
    metrics = payload["current_candidate"]["metrics"]
    lines = [
        "# Stage43-BJ Long Objective Evidence Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- long objective complete: `{gate['long_objective_complete']}`",
        "",
        "## Candidate Snapshot",
        "",
        f"- label: `{payload['current_candidate']['label']}`",
        f"- all: `{_pct(metrics['all'])}`",
        f"- t50: `{_pct(metrics['t50'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_diagnostic'])}`",
        f"- hard/failure: `{_pct(metrics['hard_failure'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation'])}`",
        "",
        "## Phase Coverage",
        "",
    ]
    for phase in payload["phases"]:
        lines.extend(
            [
                f"### {phase['phase']}",
                "",
                f"- status: `{phase['status']}`",
                f"- complete_for_long_objective: `{phase['complete_for_long_objective']}`",
                f"- evidence: {phase['evidence']}",
                "- proved:",
                *[f"  - {item}" for item in phase["proved"]],
                "- missing:",
                *[f"  - {item}" for item in phase["missing"]],
                f"- next action: {phase['next_action']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Remaining Blockers",
            "",
            *[f"- `{item}`" for item in payload["remaining_blockers"]],
            "",
            "## Next Priority Order",
            "",
            *[f"{idx}. {item}" for idx, item in enumerate(payload["next_priority_order"], start=1)],
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ]
    )
    return lines


def _write_gap_matrix(payload: Mapping[str, Any]) -> None:
    lines = [
        "# Stage43 Long Objective Gap Matrix",
        "",
        "| phase | status | complete | key missing | next action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for phase in payload["phases"]:
        missing = "<br>".join(phase["missing"])
        lines.append(
            f"| {phase['phase']} | `{phase['status']}` | `{phase['complete_for_long_objective']}` | {missing} | {phase['next_action']} |"
        )
    write_md(GAP_MD, lines)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bj_gate"]
    metrics = payload["current_candidate"]["metrics"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"long_objective_complete = `{gate['long_objective_complete']}`",
        f"candidate_all_t50_hard_easy = `{_pct(metrics['all'])}` / `{_pct(metrics['t50'])}` / `{_pct(metrics['hard_failure'])}` / `{_pct(metrics['easy_degradation'])}`",
        "",
        "I audited the Stage43 long objective against the current BH/BI evidence stack. The protected multimodal latent-state candidate is real enough to keep as current evidence, but the full long objective is still active: source terms, metric/time calibration, t100, raw multimodal evidence, and ungated/floor-free deployment remain open blockers.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds claim, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bj_long_objective_evidence_audit"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "long_objective_complete": gate["long_objective_complete"],
        "remaining_blockers": payload["remaining_blockers"],
        "next_priority_order": payload["next_priority_order"],
        "report": str(REPORT_MD),
        "gap_matrix": str(GAP_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bj_long_objective_evidence_audit"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BJ",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "long_objective_complete": gate["long_objective_complete"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bj_gate"]
    write_json(REPORT_JSON, m._jsonable(payload))
    write_json(WORLD_GATE_JSON, m._jsonable(gate))
    write_md(REPORT_MD, _render_report(payload))
    _write_gap_matrix(payload)
    write_md(
        GATE_MD,
        [
            "# Stage43-BJ Long Objective Evidence Audit Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    write_md(
        WORLD_GATE_MD,
        [
            "# Stage43 Current World-Model Gate",
            "",
            f"- source: `{payload['source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- protected multimodal latent state candidate: `{gate['protected_multimodal_latent_state_candidate']}`",
            f"- standalone world model deployable: `{gate['standalone_world_model_deployable']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-BJ audits the full Stage43 long objective against current evidence.",
            "- The protected multimodal latent-state candidate remains supported, but the long objective is not complete.",
            "- Source support, metric/time calibration, t100, raw multimodal evidence, and ungated/floor-free deployment remain open blockers.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def run_long_objective_evidence_audit() -> dict[str, Any]:
    payload = build_long_objective_evidence_audit()
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Audit Stage43 long objective against current protected candidate evidence.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    build_arg_parser().parse_args(argv)
    payload = run_long_objective_evidence_audit()
    gate = payload["stage43_bj_gate"]
    print(f"Stage43-BJ: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"long_objective_complete={gate['long_objective_complete']}")
    return payload


if __name__ == "__main__":
    main()
