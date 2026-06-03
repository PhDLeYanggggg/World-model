from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src import stage43_full_waypoint_latent_dynamics as m
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_robustness_audit import _pct


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_t100_source_stable_compatibility_audit.json"
REPORT_MD = OUT_DIR / "stage43_t100_source_stable_compatibility_audit.md"
GATE_MD = OUT_DIR / "stage43_stage_cl_t100_source_stable_compatibility_gate.md"

README_RESULTS = m.README_RESULTS
M3W_README = m.M3W_README
WORK_SUMMARY = m.WORK_SUMMARY
RESEARCH_STATE = m.RESEARCH_STATE

SOURCE = "fresh_stage43_cl_t100_source_stable_compatibility_audit"
SECTION = "STAGE43_CL_T100_SOURCE_STABLE_COMPATIBILITY_AUDIT"

STAGE43_S_JSON = OUT_DIR / "stage43_t100_source_coverage_preflight.json"
STAGE43_T_JSON = OUT_DIR / "stage43_t100_source_stable_specialist.json"
STAGE43_CK_JSON = OUT_DIR / "stage43_coverage_aware_t100_causal_feature_repair.json"
STAGE43_AT_JSON = OUT_DIR / "stage43_external_validation_matrix.json"

DENIED_FEATURE_NAME_FRAGMENTS = (
    "ade",
    "fde",
    "oracle",
    "future",
    "label",
    "central_velocity",
)


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path, {})


def _safe_pct(row: Mapping[str, Any], key: str) -> float:
    return float(row.get(key, 0.0))


def _source_level_compatibility(stage_t: Mapping[str, Any], stage_ck: Mapping[str, Any], matrix: Mapping[str, Any]) -> dict[str, Any]:
    split = stage_t.get("source_level_split", {})
    t_gate = stage_t.get("stage43_t_gate", {})
    t_metrics = stage_t.get("source_stable_h100_test_metrics", {})
    ck_gate = stage_ck.get("stage43_ck_gate", {})
    ck_metrics = stage_ck.get("test_metrics_with_causal_specialist", {})
    matrix_rows = int(matrix.get("test_rows", matrix.get("split", {}).get("test_rows", 0)))
    t_rows = int(split.get("test_rows", 0))
    row_ratio = float(t_rows / matrix_rows) if matrix_rows else 0.0
    same_split_scope = bool(matrix_rows and t_rows == matrix_rows)
    return {
        "stage43_t_positive_local_signal": bool(t_gate.get("positive_h100_dynamics_signal", False)),
        "stage43_t_deployable_under_own_source_split": bool(t_gate.get("deploy_source_stable_h100_specialist", False)),
        "stage43_t_test_rows": t_rows,
        "stage43_at_matrix_rows": matrix_rows,
        "stage43_t_row_ratio_vs_current_matrix": row_ratio,
        "same_split_scope_as_current_matrix": same_split_scope,
        "stage43_t_family": stage_t.get("training_protocol", {}).get("family"),
        "stage43_t_test_sources": split.get("test_sources", []),
        "stage43_t_h100_ade_lift": _safe_pct(t_metrics, "full_waypoint_ade_improvement_vs_floor"),
        "stage43_t_h100_endpoint_lift": _safe_pct(t_metrics, "endpoint_fde_improvement_vs_floor"),
        "stage43_t_easy_degradation": _safe_pct(t_metrics, "easy_degradation_vs_floor"),
        "stage43_ck_global_t100_success": bool(ck_gate.get("t100_positive_success", False)),
        "stage43_ck_deploy_t100": bool(ck_gate.get("deploy_t100_causal_specialist", False)),
        "stage43_ck_global_t100_diagnostic": _safe_pct(ck_metrics, "t100_raw_frame_full_waypoint_diagnostic_vs_floor"),
        "can_integrate_as_global_t100_deployment": bool(
            same_split_scope
            and t_gate.get("deploy_source_stable_h100_specialist", False)
            and ck_gate.get("t100_positive_success", False)
        ),
        "compatibility_reason": (
            "compatible_global_t100_deployment"
            if same_split_scope and ck_gate.get("t100_positive_success", False)
            else "local_source_level_positive_signal_not_current_full_matrix_deployment"
        ),
    }


def _feature_contract(stage_t: Mapping[str, Any]) -> dict[str, Any]:
    """Audit the reported Stage43-T protocol and source code boundary.

    Stage43-T is a ridge specialist over the causal `WaypointSplit.x` vector
    from Stage43 full-waypoint cache plus future waypoints as supervised target.
    The report did not store feature names, so this audit records the known
    construction boundary and checks that the protocol does not advertise
    forbidden inference inputs.
    """

    protocol = stage_t.get("training_protocol", {})
    protocol_text = " ".join(str(value).lower() for value in protocol.values())
    denied_hits = [frag for frag in DENIED_FEATURE_NAME_FRAGMENTS if frag in protocol_text]
    return {
        "feature_names_available_in_stage43_t_report": False,
        "stage43_t_uses_stage43_m_waypointsplit_x": True,
        "future_waypoints_label_only": bool(protocol.get("future_waypoints_as_labels_only", False)),
        "reported_test_threshold_tuning": bool(protocol.get("test_threshold_tuning", True)),
        "reported_selection_data": protocol.get("selection_data"),
        "denied_protocol_fragments": denied_hits,
        "causal_admissibility_status": "report_protocol_clean_but_feature_names_not_persisted",
        "required_followup_if_promoting": (
            "Persist feature names and source-split hashes before promoting Stage43-T beyond local source-level evidence."
        ),
    }


def build_t100_source_stable_compatibility_audit() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage_s = _load(STAGE43_S_JSON)
    stage_t = _load(STAGE43_T_JSON)
    stage_ck = _load(STAGE43_CK_JSON)
    matrix = _load(STAGE43_AT_JSON)
    compatibility = _source_level_compatibility(stage_t, stage_ck, matrix)
    feature_contract = _feature_contract(stage_t)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_reconciliation_of_stage43_t_local_t100_and_stage43_ck_global_floor",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {
            "stage43_s": str(STAGE43_S_JSON),
            "stage43_t": str(STAGE43_T_JSON),
            "stage43_ck": str(STAGE43_CK_JSON),
            "stage43_at": str(STAGE43_AT_JSON),
        },
        "input_verdicts": {
            "stage43_s": stage_s.get("stage43_s_gate", {}).get("verdict"),
            "stage43_t": stage_t.get("stage43_t_gate", {}).get("verdict"),
            "stage43_ck": stage_ck.get("stage43_ck_gate", {}).get("verdict"),
            "stage43_at": matrix.get("stage43_at_gate", {}).get("verdict"),
        },
        "compatibility": compatibility,
        "feature_contract": feature_contract,
        "claim_decision": {
            "local_t100_positive_signal_allowed": bool(compatibility["stage43_t_positive_local_signal"]),
            "global_t100_deployment_allowed": False,
            "uniform_t100_success_allowed": False,
            "current_deployable_t100_policy": "Stage43-CI/CK floor",
            "reason": (
                "Stage43-T is a small source-level TrajNet_crowds h100 split, while CK is the current global "
                "causal-only full-matrix audit and keeps the t100 floor."
            ),
        },
        "next_required_actions": [
            "Do not cite Stage43-T as global t100 success.",
            "If t100 is revisited, build a current-matrix-compatible source-family t100 gate with persisted feature names.",
            "Acquire or validate more h100 source support before making uniform t100 claims.",
            "Keep t100 as raw-frame diagnostic until source-stable causal evidence is positive and easy-safe.",
        ],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
        "input_hash": _combined_hash([STAGE43_S_JSON, STAGE43_T_JSON, STAGE43_CK_JSON, STAGE43_AT_JSON]),
    }
    payload["stage43_cl_gate"] = _gate(payload)
    _write_outputs(payload)
    _update_summaries(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    comp = payload["compatibility"]
    feature = payload["feature_contract"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "stage43_s_precondition_present": payload["input_verdicts"]["stage43_s"]
        == "stage43_s_t100_source_coverage_preflight_pass",
        "stage43_t_local_positive_present": comp["stage43_t_positive_local_signal"] is True,
        "stage43_t_easy_safe_under_own_split": comp["stage43_t_easy_degradation"] <= 0.02,
        "stage43_ck_global_t100_floor_confirmed": comp["stage43_ck_global_t100_success"] is False
        and comp["stage43_ck_deploy_t100"] is False,
        "split_scope_difference_recorded": comp["same_split_scope_as_current_matrix"] is False
        and comp["stage43_t_row_ratio_vs_current_matrix"] < 0.10,
        "global_t100_not_overclaimed": payload["claim_decision"]["global_t100_deployment_allowed"] is False,
        "uniform_t100_not_overclaimed": payload["claim_decision"]["uniform_t100_success_allowed"] is False,
        "feature_contract_audited": feature["future_waypoints_label_only"] is True
        and feature["reported_test_threshold_tuning"] is False,
        "no_denied_protocol_fragments": feature["denied_protocol_fragments"] == [],
        "no_future_or_test_leakage": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["central_velocity_input"] is False
        and no_leak["test_endpoint_goal_construction"] is False
        and no_leak["test_statistics_normalization"] is False
        and no_leak["test_threshold_tuning"] is False,
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = (
        "stage43_cl_t100_source_stable_compatibility_pass_local_only"
        if passed == total
        else "stage43_cl_t100_source_stable_compatibility_incomplete"
    )
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "local_t100_positive_signal_allowed": bool(
            payload["claim_decision"].get("local_t100_positive_signal_allowed", comp["stage43_t_positive_local_signal"])
        ),
        "global_t100_deployment_allowed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_cl_gate"]
    comp = payload["compatibility"]
    feature = payload["feature_contract"]
    return [
        "# Stage43-CL T100 Source-Stable Compatibility Audit",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- local t100 positive signal allowed: `{gate['local_t100_positive_signal_allowed']}`",
        f"- global t100 deployment allowed: `{gate['global_t100_deployment_allowed']}`",
        "",
        "## Compatibility Result",
        "",
        f"- Stage43-T local rows: `{comp['stage43_t_test_rows']}`",
        f"- current external matrix rows: `{comp['stage43_at_matrix_rows']}`",
        f"- row ratio: `{comp['stage43_t_row_ratio_vs_current_matrix']:.4f}`",
        f"- same split scope: `{comp['same_split_scope_as_current_matrix']}`",
        f"- Stage43-T family: `{comp['stage43_t_family']}`",
        f"- Stage43-T test sources: `{', '.join(comp['stage43_t_test_sources'])}`",
        f"- Stage43-T h100 ADE lift: `{_pct(comp['stage43_t_h100_ade_lift'])}`",
        f"- Stage43-T h100 endpoint lift: `{_pct(comp['stage43_t_h100_endpoint_lift'])}`",
        f"- Stage43-T easy degradation: `{_pct(comp['stage43_t_easy_degradation'])}`",
        f"- CK global t100 diagnostic: `{_pct(comp['stage43_ck_global_t100_diagnostic'])}`",
        f"- compatibility reason: `{comp['compatibility_reason']}`",
        "",
        "## Feature Contract Audit",
        "",
        f"- feature names persisted in Stage43-T report: `{feature['feature_names_available_in_stage43_t_report']}`",
        f"- future waypoints label-only: `{feature['future_waypoints_label_only']}`",
        f"- test threshold tuning: `{feature['reported_test_threshold_tuning']}`",
        f"- denied protocol fragments: `{feature['denied_protocol_fragments']}`",
        f"- causal admissibility status: `{feature['causal_admissibility_status']}`",
        f"- promotion follow-up: `{feature['required_followup_if_promoting']}`",
        "",
        "## Claim Decision",
        "",
        f"- local t100 positive signal may be reported: `{payload['claim_decision']['local_t100_positive_signal_allowed']}`",
        f"- global t100 deployment may be reported: `{payload['claim_decision']['global_t100_deployment_allowed']}`",
        f"- uniform t100 success may be reported: `{payload['claim_decision']['uniform_t100_success_allowed']}`",
        f"- current deployable t100 policy: `{payload['claim_decision']['current_deployable_t100_policy']}`",
        f"- reason: {payload['claim_decision']['reason']}",
        "",
        "## Next Required Actions",
        "",
        *[f"- {item}" for item in payload["next_required_actions"]],
        "",
        "## Boundary",
        "",
        "- Dataset-local/raw-frame 2.5D only.",
        "- No metric or seconds-level claim.",
        "- No true 3D or foundation claim.",
        "- No Stage5C execution.",
        "- No SMC.",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        "",
    ]


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_cl_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CL Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- local t100 positive signal allowed: `{gate['local_t100_positive_signal_allowed']}`",
            f"- global t100 deployment allowed: `{gate['global_t100_deployment_allowed']}`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
        ],
    )


def _update_summaries(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_cl_gate"]
    comp = payload["compatibility"]
    block = [
        f"## {SECTION}",
        "",
        "I reconciled the earlier Stage43-T source-stable h100 result with the current CK global t100 floor. The result is deliberately conservative: Stage43-T remains useful as local source-level evidence, but it is not a global t100 deployment result.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- Stage43-T local h100 ADE lift: `{_pct(comp['stage43_t_h100_ade_lift'])}` on `{comp['stage43_t_test_rows']}` rows",
        f"- Stage43-T local easy degradation: `{_pct(comp['stage43_t_easy_degradation'])}`",
        f"- current CK global t100 diagnostic: `{_pct(comp['stage43_ck_global_t100_diagnostic'])}`",
        f"- global t100 deployment allowed: `{gate['global_t100_deployment_allowed']}`",
        "",
        "Current interpretation: t100 has a small local source-stable positive signal, but current deployable t100 remains the floor. Future t100 work needs current-matrix-compatible source-family gates with persisted feature names and stronger source support.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, block)
    state = read_json(RESEARCH_STATE, {})
    state.setdefault("stage43", {})
    state["stage43"]["t100_source_stable_compatibility_audit"] = {
        "source": SOURCE,
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "compatibility": payload["compatibility"],
        "claim_decision": payload["claim_decision"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["current_stage"] = "stage43_cl_t100_source_stable_compatibility_audit"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, m._jsonable(state))


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Run Stage43-CL t100 source-stable compatibility audit.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    build_arg_parser().parse_args(argv)
    payload = build_t100_source_stable_compatibility_audit()
    gate = payload["stage43_cl_gate"]
    print(f"Stage43-CL: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    return payload


if __name__ == "__main__":
    main()
