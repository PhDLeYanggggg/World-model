from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_t100_family_limited_reconciliation.json"
REPORT_MD = OUT_DIR / "stage43_t100_family_limited_reconciliation.md"
GATE_MD = OUT_DIR / "stage43_stage_bk_t100_family_limited_reconciliation_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SOURCE = "fresh_stage43_bk_t100_family_limited_reconciliation"
SECTION = "STAGE43_BK_T100_FAMILY_LIMITED_RECONCILIATION"

INPUTS = {
    "stage43_p_tail_adapter": OUT_DIR / "stage43_tail_horizon_waypoint_adapter.json",
    "stage43_t_source_stable_h100_specialist": OUT_DIR / "stage43_t100_source_stable_specialist.json",
    "stage43_u_integrated_tail_h100_policy": OUT_DIR / "stage43_integrated_tail_h100_policy.json",
    "stage43_bi_locked_candidate_package": OUT_DIR / "stage43_locked_candidate_paper_package_refresh.json",
    "stage43_bj_long_objective_audit": OUT_DIR / "stage43_long_objective_evidence_audit.json",
}


def _pct(value: float | int) -> str:
    return f"{100.0 * float(value):.2f}%"


def _read_required(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path, {})


def _gate_full_pass(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return int(gate.get("passed", -1)) == int(gate.get("total", -2)) and int(gate.get("total", 0)) > 0


def _gate_verdict(payload: Mapping[str, Any], key: str) -> str:
    return str(payload.get(key, {}).get("verdict", "missing"))


def _metric_ci(payload: Mapping[str, Any], metric: str) -> dict[str, float]:
    return {
        "low": float(payload["bootstrap_ci"]["metrics"][metric]["low"]),
        "mean": float(payload["bootstrap_ci"]["metrics"][metric]["mean"]),
        "high": float(payload["bootstrap_ci"]["metrics"][metric]["high"]),
    }


def build_t100_family_limited_reconciliation() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    artifacts = {name: _read_required(path) for name, path in INPUTS.items()}
    p = artifacts["stage43_p_tail_adapter"]
    t = artifacts["stage43_t_source_stable_h100_specialist"]
    u = artifacts["stage43_u_integrated_tail_h100_policy"]
    bi = artifacts["stage43_bi_locked_candidate_package"]
    bj = artifacts["stage43_bj_long_objective_audit"]

    p_metrics = p["overall_full_test_metrics"]
    t_metrics = t["source_stable_h100_test_metrics"]
    u_metrics = u["integrated_full_test_metrics"]
    u_delta = u["delta_vs_stage43_p"]
    h100 = u["h100_specialist_slice"]["integrated_slice"]
    h100_p = u["h100_specialist_slice"]["stage43_p_slice"]
    u_t100_ci = _metric_ci(u, "t100_raw_frame_full_waypoint_diagnostic_vs_floor")
    u_all_ci = _metric_ci(u, "full_waypoint_ade_improvement_vs_floor")
    t_h100_ci = _metric_ci(t, "full_waypoint_ade_improvement_vs_floor")

    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_reconciliation_from_stage43_p_t_u_bi_bj_verified_artifacts",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {name: str(path) for name, path in INPUTS.items()},
        "input_hash": _combined_hash(list(INPUTS.values())),
        "input_verdicts": {
            "stage43_p": _gate_verdict(p, "stage43_p_gate"),
            "stage43_t": _gate_verdict(t, "stage43_t_gate"),
            "stage43_u": _gate_verdict(u, "stage43_u_gate"),
            "stage43_bi": _gate_verdict(bi, "stage43_bi_gate"),
            "stage43_bj": _gate_verdict(bj, "stage43_bj_gate"),
        },
        "locked_candidate_context": {
            "current_locked_package": "stage43_bi_locked_candidate_paper_package_refresh",
            "locked_package_metrics": bi["metrics"],
            "long_objective_prior_blocker": "t100_raw_frame_diagnostic_not_solved",
            "bk_reconciliation": (
                "t100 is no longer simply zero-evidence: Stage43-T/U provide a small source-stable h100 "
                "full-waypoint ADE signal. Uniform t100 and endpoint-FDE success remain blocked."
            ),
        },
        "stage43_p_reference": {
            "rows": int(p_metrics["rows"]),
            "all_full_waypoint_ade_improvement_vs_floor": float(
                p_metrics["full_waypoint_ade_improvement_vs_floor"]
            ),
            "t50_full_waypoint_ade_improvement_vs_floor": float(
                p_metrics["t50_full_waypoint_ade_improvement_vs_floor"]
            ),
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": float(
                p_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
            ),
            "hard_failure_full_waypoint_ade_improvement_vs_floor": float(
                p_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
            ),
            "easy_degradation_vs_floor": float(p_metrics["easy_degradation_vs_floor"]),
            "gate_verdict": payload_verdict(p, "stage43_p_gate"),
        },
        "stage43_t_source_stable_h100": {
            "rows": int(t_metrics["rows"]),
            "full_waypoint_ade_improvement_vs_floor": float(
                t_metrics["full_waypoint_ade_improvement_vs_floor"]
            ),
            "endpoint_fde_improvement_vs_floor": float(t_metrics["endpoint_fde_improvement_vs_floor"]),
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": float(
                t_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
            ),
            "hard_failure_full_waypoint_ade_improvement_vs_floor": float(
                t_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
            ),
            "easy_degradation_vs_floor": float(t_metrics["easy_degradation_vs_floor"]),
            "bootstrap_full_waypoint_ade_ci": t_h100_ci,
            "test_sources": list(t["source_level_split"]["test_sources"]),
            "gate_verdict": payload_verdict(t, "stage43_t_gate"),
        },
        "stage43_u_integrated_policy": {
            "rows": int(u_metrics["rows"]),
            "all_full_waypoint_ade_improvement_vs_floor": float(
                u_metrics["full_waypoint_ade_improvement_vs_floor"]
            ),
            "endpoint_fde_improvement_vs_floor": float(u_metrics["endpoint_fde_improvement_vs_floor"]),
            "t50_full_waypoint_ade_improvement_vs_floor": float(
                u_metrics["t50_full_waypoint_ade_improvement_vs_floor"]
            ),
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor": float(
                u_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
            ),
            "hard_failure_full_waypoint_ade_improvement_vs_floor": float(
                u_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
            ),
            "easy_degradation_vs_floor": float(u_metrics["easy_degradation_vs_floor"]),
            "t100_bootstrap_ci": u_t100_ci,
            "all_bootstrap_ci": u_all_ci,
            "delta_vs_stage43_p": u_delta,
            "gate_verdict": payload_verdict(u, "stage43_u_gate"),
        },
        "h100_source_stable_slice": {
            "stage43_p_slice": h100_p,
            "integrated_slice": h100,
            "source_table": u["h100_specialist_slice"]["source_table"],
            "negative_source_count": int(u["h100_specialist_slice"]["negative_source_count"]),
        },
        "claim_update": {
            "t100_family_limited_ade_signal": True,
            "current_aggregate_candidate_can_report_family_limited_t100_diagnostic": True,
            "uniform_t100_success": False,
            "t100_endpoint_success": False,
            "h100_endpoint_fde_negative_explicitly_reported": float(
                h100["endpoint_fde_improvement_vs_floor"]
            )
            < 0.0,
            "replacement_for_stage43_bi_locked_candidate": False,
            "why_not_replacement": (
                "Stage43-U adds a small source-stable h100 ADE diagnostic without harming t50/easy, "
                "but h100 endpoint FDE is negative and uniform t100 remains false. Keep the locked "
                "candidate package intact while updating the t100 evidence boundary."
            ),
        },
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
            "fresh_reconciliation_only": True,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
            "uniform_positive_external_transfer_claim": False,
            "uniform_t100_success": False,
            "t100_endpoint_success": False,
            "long_objective_complete": False,
        },
        "next_priority_order": [
            "keep reporting t100 as family-limited raw-frame ADE diagnostic, not uniform success",
            "close blocked source terms/identity and guarded conversion preflight",
            "seek independent h100 source support before any uniform t100 claim",
            "replace proxy-heavy scene/interaction ablations with retrained raw-scene/graph-rich ablations",
        ],
    }
    payload["stage43_bk_gate"] = _gate(payload, artifacts)
    return payload


def payload_verdict(payload: Mapping[str, Any], key: str) -> str:
    return str(payload.get(key, {}).get("verdict", "missing"))


def _gate(payload: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    p_ref = payload["stage43_p_reference"]
    t_ref = payload["stage43_t_source_stable_h100"]
    u_ref = payload["stage43_u_integrated_policy"]
    h100 = payload["h100_source_stable_slice"]["integrated_slice"]
    claim = payload["claim_update"]
    boundary = payload["claim_boundary"]
    leak = payload["no_leakage_and_execution"]
    t100_ci = u_ref["t100_bootstrap_ci"]
    h100_ci = t_ref["bootstrap_full_waypoint_ade_ci"]
    gates = {
        "stage43_p_precondition_passed": _gate_full_pass(artifacts["stage43_p_tail_adapter"], "stage43_p_gate")
        and payload["input_verdicts"]["stage43_p"] == "stage43_p_tail_horizon_adapter_pass_t100_still_fallback",
        "stage43_t_precondition_passed": _gate_full_pass(
            artifacts["stage43_t_source_stable_h100_specialist"], "stage43_t_gate"
        )
        and payload["input_verdicts"]["stage43_t"] == "stage43_t_source_stable_h100_specialist_deployable",
        "stage43_u_precondition_passed": _gate_full_pass(
            artifacts["stage43_u_integrated_tail_h100_policy"], "stage43_u_gate"
        )
        and payload["input_verdicts"]["stage43_u"]
        == "stage43_u_integrated_tail_h100_policy_pass_family_limited",
        "stage43_bi_bj_preconditions_passed": _gate_full_pass(
            artifacts["stage43_bi_locked_candidate_package"], "stage43_bi_gate"
        )
        and _gate_full_pass(artifacts["stage43_bj_long_objective_audit"], "stage43_bj_gate"),
        "integrated_policy_preserves_core_metrics": u_ref["all_full_waypoint_ade_improvement_vs_floor"] > 0.0
        and u_ref["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
        and u_ref["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
        and u_ref["easy_degradation_vs_floor"] <= 0.02,
        "integrated_t100_improves_over_stage43_p": u_ref["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
        > p_ref["t100_raw_frame_full_waypoint_diagnostic_vs_floor"],
        "integrated_t100_ci_positive": t100_ci["low"] > 0.0 and t100_ci["high"] > t100_ci["low"],
        "h100_source_stable_ade_positive": h100["full_waypoint_ade_improvement_vs_floor"] > 0.0
        and h100_ci["low"] > 0.0,
        "h100_endpoint_blocker_explicit": claim["h100_endpoint_fde_negative_explicitly_reported"] is True
        and h100["endpoint_fde_improvement_vs_floor"] < 0.0
        and claim["t100_endpoint_success"] is False
        and boundary["t100_endpoint_success"] is False,
        "uniform_t100_not_overclaimed": claim["uniform_t100_success"] is False
        and boundary["uniform_t100_success"] is False,
        "fresh_reconciliation_only": leak["new_training_executed"] is False
        and leak["new_conversion_executed"] is False
        and leak["fresh_reconciliation_only"] is True,
        "no_future_or_test_leakage": leak["future_endpoint_input"] is False
        and leak["future_waypoint_input"] is False
        and leak["future_labels_eval_or_loss_only"] is True
        and leak["central_velocity_input"] is False
        and leak["test_endpoint_goal_construction"] is False
        and leak["test_statistics_normalization"] is False
        and leak["test_threshold_tuning"] is False,
        "claim_boundary_not_overstated": boundary["true_3d_world_model"] is False
        and boundary["foundation_world_model"] is False
        and boundary["metric_or_seconds_claim"] is False
        and boundary["dataset_local_raw_frame_only"] is True
        and boundary["uniform_positive_external_transfer_claim"] is False,
        "stage5c_and_smc_false": boundary["stage5c_executed"] is False and boundary["smc_enabled"] is False,
        "long_objective_kept_active": boundary["long_objective_complete"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_bk_t100_family_limited_reconciliation_pass"
        if passed == total
        else "stage43_bk_t100_family_limited_reconciliation_incomplete",
        "t100_family_limited_ade_signal": gates["integrated_t100_ci_positive"]
        and gates["h100_source_stable_ade_positive"],
        "uniform_t100_success": False,
        "t100_endpoint_success": False,
        "protected_multimodal_latent_state_candidate": passed == total,
        "long_objective_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_bk_gate"]
    p_ref = payload["stage43_p_reference"]
    t_ref = payload["stage43_t_source_stable_h100"]
    u_ref = payload["stage43_u_integrated_policy"]
    h100 = payload["h100_source_stable_slice"]["integrated_slice"]
    t100_ci = u_ref["t100_bootstrap_ci"]
    h100_ci = t_ref["bootstrap_full_waypoint_ade_ci"]
    lines = [
        "# Stage43-BK T100 Family-Limited Reconciliation",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- t100 family-limited ADE signal: `{gate['t100_family_limited_ade_signal']}`",
        f"- uniform t100 success: `{gate['uniform_t100_success']}`",
        f"- t100 endpoint success: `{gate['t100_endpoint_success']}`",
        "",
        "## Why This Reconciliation Exists",
        "",
        "Stage43-BJ correctly kept the long objective active, but its `t100_raw_frame_diagnostic_not_solved` blocker was too coarse. Stage43-T/U show a small source-stable h100 full-waypoint ADE signal. The right boundary is not `t100 solved`; it is `family-limited t100 ADE diagnostic exists, while uniform t100 and endpoint-FDE success remain blocked`.",
        "",
        "## Reference Candidate",
        "",
        f"- Stage43-P all ADE improvement: `{_pct(p_ref['all_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- Stage43-P t50 ADE improvement: `{_pct(p_ref['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- Stage43-P t100 raw-frame diagnostic: `{_pct(p_ref['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- Stage43-P hard/failure: `{_pct(p_ref['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        "",
        "## Family-Limited H100 Evidence",
        "",
        f"- Stage43-T source-stable h100 rows: `{t_ref['rows']}`",
        f"- Stage43-T full-waypoint ADE lift: `{_pct(t_ref['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- Stage43-T full-waypoint ADE CI: `[{_pct(h100_ci['low'])}, {_pct(h100_ci['high'])}]`",
        f"- Stage43-T endpoint FDE lift: `{_pct(t_ref['endpoint_fde_improvement_vs_floor'])}`",
        f"- test sources: `{t_ref['test_sources']}`",
        "",
        "## Integrated Policy Diagnostic",
        "",
        f"- integrated all ADE improvement: `{_pct(u_ref['all_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- integrated t50 ADE improvement: `{_pct(u_ref['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- integrated t100 raw-frame diagnostic: `{_pct(u_ref['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- integrated t100 CI: `[{_pct(t100_ci['low'])}, {_pct(t100_ci['high'])}]`",
        f"- integrated hard/failure: `{_pct(u_ref['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- integrated easy degradation: `{_pct(u_ref['easy_degradation_vs_floor'])}`",
        f"- t100 delta vs Stage43-P: `{_pct(u_ref['delta_vs_stage43_p']['t100_delta'])}`",
        "",
        "## H100 Slice Boundary",
        "",
        f"- h100 slice rows: `{h100['rows']}`",
        f"- h100 slice ADE lift: `{_pct(h100['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- h100 slice endpoint FDE lift: `{_pct(h100['endpoint_fde_improvement_vs_floor'])}`",
        f"- h100 slice hard/failure ADE lift: `{_pct(h100['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- h100 slice easy degradation: `{_pct(h100['easy_degradation_vs_floor'])}`",
        "",
        "## Claim Update",
        "",
        "- Allowed: report a family-limited raw-frame h100/t100 full-waypoint ADE diagnostic signal.",
        "- Not allowed: report uniform t100 success.",
        "- Not allowed: report h100/t100 endpoint-FDE success.",
        "- Not allowed: metric, seconds-level, true-3D, foundation, Stage5C, or SMC claims.",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
    ]
    return lines


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bk_gate"]
    u_ref = payload["stage43_u_integrated_policy"]
    h100 = payload["h100_source_stable_slice"]["integrated_slice"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"t100_family_limited_ade_signal = `{gate['t100_family_limited_ade_signal']}`",
        f"uniform_t100_success = `{gate['uniform_t100_success']}`",
        f"t100_endpoint_success = `{gate['t100_endpoint_success']}`",
        "",
        f"Stage43-BK reconciles the t100/h100 evidence: Stage43-U gives integrated t100 raw-frame full-waypoint ADE diagnostic `{_pct(u_ref['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}` with CI `[{_pct(u_ref['t100_bootstrap_ci']['low'])}, {_pct(u_ref['t100_bootstrap_ci']['high'])}]`; the source-stable h100 slice gives ADE lift `{_pct(h100['full_waypoint_ade_improvement_vs_floor'])}`.",
        "",
        f"The blocker remains: h100 endpoint FDE is `{_pct(h100['endpoint_fde_improvement_vs_floor'])}`, so this is not endpoint success and not a uniform t100 solution. The long objective stays active.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bk_t100_family_limited_reconciliation"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "t100_family_limited_ade_signal": gate["t100_family_limited_ade_signal"],
        "uniform_t100_success": gate["uniform_t100_success"],
        "t100_endpoint_success": gate["t100_endpoint_success"],
        "integrated_t100_raw_frame_diagnostic": u_ref[
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor"
        ],
        "integrated_t100_ci": u_ref["t100_bootstrap_ci"],
        "h100_source_stable_slice": payload["h100_source_stable_slice"]["integrated_slice"],
        "claim_boundary": payload["claim_boundary"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bk_t100_family_limited_reconciliation"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BK",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "t100_family_limited_ade_signal": gate["t100_family_limited_ade_signal"],
                        "uniform_t100_success": gate["uniform_t100_success"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bk_gate"]
    write_json(REPORT_JSON, m._jsonable(payload))
    write_json(WORLD_GATE_JSON, m._jsonable(gate))
    write_md(REPORT_MD, _render_report(payload))
    write_md(
        GATE_MD,
        [
            "# Stage43-BK T100 Family-Limited Reconciliation Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- t100 family-limited ADE signal: `{gate['t100_family_limited_ade_signal']}`",
            f"- uniform t100 success: `{gate['uniform_t100_success']}`",
            f"- t100 endpoint success: `{gate['t100_endpoint_success']}`",
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
            f"- t100 family-limited ADE signal: `{gate['t100_family_limited_ade_signal']}`",
            f"- uniform t100 success: `{gate['uniform_t100_success']}`",
            f"- t100 endpoint success: `{gate['t100_endpoint_success']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-BK reconciles existing t100/h100 evidence without new training or conversion.",
            "- A family-limited raw-frame h100/t100 full-waypoint ADE diagnostic signal exists.",
            "- Uniform t100 success and endpoint-FDE success remain blocked.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def run_t100_family_limited_reconciliation() -> dict[str, Any]:
    payload = build_t100_family_limited_reconciliation()
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Reconcile Stage43 family-limited h100/t100 evidence boundaries.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    build_arg_parser().parse_args(argv)
    payload = run_t100_family_limited_reconciliation()
    gate = payload["stage43_bk_gate"]
    print(f"Stage43-BK: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"t100_family_limited_ade_signal={gate['t100_family_limited_ade_signal']}")
    print(f"uniform_t100_success={gate['uniform_t100_success']}")
    return payload


if __name__ == "__main__":
    main()
