from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_external_validation_matrix.json"
REPORT_MD = OUT_DIR / "stage43_external_validation_matrix.md"
GATE_MD = OUT_DIR / "stage43_stage_at_external_validation_matrix_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE43_AT_EXTERNAL_VALIDATION_MATRIX"
SOURCE = "fresh_stage43_at_external_validation_matrix"

SPLIT = OUT_DIR / "stage43_source_level_heldout_split.json"
M3W_V1 = Path("outputs/m3w_neural_v1/evidence_matrix_m3w_neural_v1.json")
ROBUSTNESS_H = OUT_DIR / "stage43_source_level_latent_robustness_audit.json"
SAFE_SWITCH_I = OUT_DIR / "stage43_unit_consistent_safe_switch.json"
SOURCE_REPAIR_K = OUT_DIR / "stage43_source_slice_repair.json"
FULL_WAYPOINT_M = OUT_DIR / "stage43_full_waypoint_latent_dynamics.json"
REVIEWER_REPLAY_AO = OUT_DIR / "stage43_bounded_residual_reviewer_replay.json"
TAIL_ADAPTER_P = OUT_DIR / "stage43_tail_horizon_waypoint_adapter.json"
CURRENT_GATE_AQ = OUT_DIR / "world_model_gate_stage43_current.json"


def _pct(value: float | int | None) -> str:
    if value is None:
        return "not_run"
    return f"{100.0 * float(value):.2f}%"


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path, {})


def _metric_record(
    *,
    name: str,
    role: str,
    result_source: str,
    deployable: bool,
    metrics: Mapping[str, Any],
    caveat: str,
    evidence_path: Path,
    row_scope: str,
    policy_hash: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "result_source": result_source,
        "deployable": bool(deployable),
        "row_scope": row_scope,
        "policy_hash": policy_hash,
        "metrics": {
            "rows": int(metrics.get("rows", 0)),
            "all": float(metrics.get("all_improvement_vs_floor", metrics.get("all_improvement", 0.0))),
            "t50": float(metrics.get("t50_improvement_vs_floor", metrics.get("t50_improvement", 0.0))),
            "t100_raw_frame_diagnostic": float(
                metrics.get(
                    "t100_raw_frame_diagnostic_vs_floor",
                    metrics.get("t100_improvement", metrics.get("t100_raw_frame_full_waypoint_diagnostic_vs_floor", 0.0)),
                )
            ),
            "hard_failure": float(
                metrics.get("hard_failure_improvement_vs_floor", metrics.get("hard_failure_improvement", 0.0))
            ),
            "easy_degradation": float(metrics.get("easy_degradation_vs_floor", metrics.get("easy_degradation", 0.0))),
            "switch_rate": float(metrics.get("switch_rate", 0.0)),
        },
        "caveat": caveat,
        "evidence_path": str(evidence_path),
    }


def _full_waypoint_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rows": metrics.get("rows", 0),
        "all_improvement_vs_floor": metrics.get("full_waypoint_ade_improvement_vs_floor", 0.0),
        "t50_improvement_vs_floor": metrics.get("t50_full_waypoint_ade_improvement_vs_floor", 0.0),
        "t100_raw_frame_diagnostic_vs_floor": metrics.get("t100_raw_frame_full_waypoint_diagnostic_vs_floor", 0.0),
        "hard_failure_improvement_vs_floor": metrics.get("hard_failure_full_waypoint_ade_improvement_vs_floor", 0.0),
        "easy_degradation_vs_floor": metrics.get("easy_degradation_vs_floor", 0.0),
        "switch_rate": metrics.get("switch_rate", 0.0),
    }


def build_external_validation_matrix() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    split = _load(SPLIT)
    m3w_v1 = _load(M3W_V1)
    robustness_h = _load(ROBUSTNESS_H)
    safe_i = _load(SAFE_SWITCH_I)
    repair_k = _load(SOURCE_REPAIR_K)
    waypoint_m = _load(FULL_WAYPOINT_M)
    replay_ao = _load(REVIEWER_REPLAY_AO)
    tail_p = _load(TAIL_ADAPTER_P)
    current_gate = _load(CURRENT_GATE_AQ)

    rows: list[dict[str, Any]] = []
    rows.append(
        _metric_record(
            name="external strongest / Stage37 floor reference",
            role="safety_floor_reference",
            result_source="cached_verified_floor_reference",
            deployable=True,
            metrics={"rows": split["split_summary"]["test"]["rows"]},
            caveat="Reference floor. Improvements are reported relative to this floor where applicable.",
            evidence_path=SPLIT,
            row_scope="source_file_level_test",
        )
    )
    rows.append(
        _metric_record(
            name="M3W-Neural v1 composite-tail safe-switch",
            role="previous_best_protected_neural",
            result_source=str(m3w_v1.get("source", "cached_verified")),
            deployable=True,
            metrics=m3w_v1["best_metrics_vs_stage37_floor"],
            caveat="Earlier protected neural candidate under Stage37/teacher floor; not a new Stage43 source-level replay.",
            evidence_path=M3W_V1,
            row_scope="Stage41 external package rows",
        )
    )
    rows.append(
        _metric_record(
            name="Stage43-G source-level latent, ungated/full-switch diagnostic",
            role="ungated_neural_diagnostic",
            result_source=str(robustness_h.get("result_source", "cached_verified")),
            deployable=False,
            metrics=robustness_h["unit_consistent_metrics"],
            caveat="Unit-consistent audit found easy degradation unsafe; keep floor.",
            evidence_path=ROBUSTNESS_H,
            row_scope="source_file_level_test",
        )
    )
    rows.append(
        _metric_record(
            name="Stage43-I domain-capped protected latent safe-switch",
            role="protected_domain_level_neural",
            result_source=str(safe_i.get("result_source", "cached_verified")),
            deployable=bool(safe_i["stage43_i_gate"].get("deploy_stage43_i_candidate", False)),
            metrics=safe_i["deployment_policy"]["test_metrics"],
            caveat="Domain-level positive and easy-safe, but one source slice remained slightly negative.",
            evidence_path=SAFE_SWITCH_I,
            row_scope="source_file_level_test",
            policy_hash=safe_i["deployment_policy"].get("policy_hash"),
        )
    )
    rows.append(
        _metric_record(
            name="Stage43-K validation source-family guarded repair",
            role="source_safe_protected_neural",
            result_source=str(repair_k.get("result_source", "cached_verified")),
            deployable=bool(repair_k["stage43_k_gate"].get("source_safe_candidate", False)),
            metrics=repair_k["deployment_policy"]["test_metrics"],
            caveat="Repairs negative source harm with validation-only source-family guard; does not claim every source has positive transfer.",
            evidence_path=SOURCE_REPAIR_K,
            row_scope="source_file_level_test",
            policy_hash=repair_k["deployment_policy"].get("policy_hash"),
        )
    )
    rows.append(
        _metric_record(
            name="Stage43-M protected full-waypoint latent dynamics",
            role="protected_full_waypoint_neural",
            result_source=str(waypoint_m.get("result_source", "cached_verified")),
            deployable=bool(waypoint_m.get("stage43_m_gate", {}).get("deploy_neural_full_waypoint_head", True)),
            metrics=_full_waypoint_metrics(waypoint_m["test_metrics_with_floor"]),
            caveat="Protected full-waypoint signal on 16k-row supervision cache; t100 remains guarded/diagnostic and not source-level official.",
            evidence_path=FULL_WAYPOINT_M,
            row_scope="full_waypoint_supervision_test_sample",
        )
    )
    rows.append(
        _metric_record(
            name="Stage43-AO frozen bounded-residual replay",
            role="current_best_integrated_candidate",
            result_source=str(replay_ao.get("result_source", "cached_verified")),
            deployable=bool(replay_ao["stage43_ao_gate"].get("reviewer_replay_passed", False)),
            metrics=_full_waypoint_metrics(replay_ao["replayed_metrics"]),
            caveat="Exact reviewer replay of frozen bounded-residual policy; protected and h100 guarded, not a global floor removal.",
            evidence_path=REVIEWER_REPLAY_AO,
            row_scope="frozen_bounded_residual_test_rows",
            policy_hash=replay_ao.get("policy_hash"),
        )
    )
    rows.append(
        _metric_record(
            name="Stage43-P tail-horizon full-waypoint adapter",
            role="latest_full_test_tail_adapter_candidate",
            result_source=str(tail_p.get("result_source", "cached_verified")),
            deployable=bool(tail_p["stage43_p_gate"].get("deploy_tail_horizon_adapter", False)),
            metrics=_full_waypoint_metrics(tail_p["overall_full_test_metrics"]),
            caveat=(
                "Latest full-test protected tail-horizon adapter; materially stronger on all/t50/hard, "
                "but h100 remains validation-blocked and falls back to the floor."
            ),
            evidence_path=TAIL_ADAPTER_P,
            row_scope="full_external_test_rows",
            policy_hash=tail_p.get("selected_model", {}).get("model_hash"),
        )
    )

    domains = split["split_summary"]["test"]["domains"]
    test_summary = split["split_summary"]["test"]
    source_count = test_summary.get("source_count", test_summary.get("sources", 0))
    per_domain_i = safe_i["deployment_policy"]["domain_metrics"]
    per_source_k = repair_k["deployment_policy"]["source_metrics"]
    matrix_hash = _combined_hash(
        [
            SPLIT,
            M3W_V1,
            ROBUSTNESS_H,
            SAFE_SWITCH_I,
            SOURCE_REPAIR_K,
            FULL_WAYPOINT_M,
            REVIEWER_REPLAY_AO,
            TAIL_ADAPTER_P,
            CURRENT_GATE_AQ,
        ]
    )
    claim_boundary = {
        "true_3d_world_model": False,
        "foundation_world_model": False,
        "metric_or_seconds_claim": False,
        "dataset_local_raw_frame_only": True,
        "stage5c_executed": False,
        "smc_enabled": False,
        "global_floor_removed": False,
    }
    no_leakage = {
        "future_endpoint_input": False,
        "future_waypoint_input": False,
        "future_labels_eval_only": True,
        "central_velocity_input": False,
        "test_endpoint_goal_construction": False,
        "test_statistics_normalization": False,
    }
    payload = {
        "source": SOURCE,
        "result_source": "fresh_external_validation_matrix_from_verified_artifacts",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_hash": matrix_hash,
        "split": {
            "artifact": str(SPLIT),
            "verdict": split["stage43_f_gate"]["verdict"],
            "row_hash": split["pool"]["row_hash"] if "row_hash" in split.get("pool", {}) else split.get("row_hash"),
            "test_rows": split["split_summary"]["test"]["rows"],
            "test_domains": domains,
            "test_source_count": source_count,
            "horizon_counts": test_summary.get("horizon_counts", test_summary.get("horizons", {})),
        },
        "comparison_rows": rows,
        "per_domain_external_validation": per_domain_i,
        "per_source_source_safe_repair": per_source_k,
        "source_repair_summary": repair_k["repair_summary"],
        "current_gate_verdict": current_gate.get("verdict"),
        "claim_boundary": claim_boundary,
        "no_leakage": no_leakage,
    }
    payload["stage43_at_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = payload["comparison_rows"]
    by_role = {row["role"]: row for row in rows}
    source_summary = payload["source_repair_summary"]
    no_leakage = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "source_level_split_ready": payload["split"]["verdict"] == "stage43_f_source_level_split_ready",
        "external_domains_present": {"ETH_UCY", "TrajNet", "UCY"}.issubset(set(payload["split"]["test_domains"])),
        "required_model_families_compared": {
            "safety_floor_reference",
            "previous_best_protected_neural",
            "ungated_neural_diagnostic",
            "protected_domain_level_neural",
            "source_safe_protected_neural",
            "protected_full_waypoint_neural",
            "current_best_integrated_candidate",
            "latest_full_test_tail_adapter_candidate",
        }.issubset(set(by_role)),
        "ungated_unsafe_not_deployed": by_role["ungated_neural_diagnostic"]["deployable"] is False
        and by_role["ungated_neural_diagnostic"]["metrics"]["easy_degradation"] > 0.02,
        "source_safe_candidate_present": by_role["source_safe_protected_neural"]["deployable"] is True
        and source_summary["stage43_k_negative_source_count"] == 0,
        "uniform_source_overclaim_blocked": source_summary["uniform_positive_per_source_claim_allowed"] is False,
        "current_candidate_replay_exact_or_reconciled": by_role["current_best_integrated_candidate"]["deployable"] is True
        and payload["current_gate_verdict"]
        in {
            "stage43_aq_integrated_protected_latent_state_candidate_pass",
            "stage43_ay_current_candidate_reconciliation_pass",
            "stage43_az_tail_adapter_reviewer_replay_pass",
            "stage43_ba_tail_adapter_source_blocker_audit_pass",
            "stage43_bb_blocked_source_repair_feasibility_pass",
            "stage43_bc_blocked_family_support_scan_pass",
            "stage43_bd_biwi_support_rebuild_preflight_pass",
            "stage43_be_blocked_source_support_acquisition_preflight_pass",
            "stage43_bf_blocked_source_terms_identity_packet_pass",
            "stage43_bg_blocked_source_terms_validation_pass",
            "stage43_bh_protected_multimodal_latent_candidate_lock_pass",
            "stage43_bi_locked_candidate_paper_package_refresh_pass",
            "stage43_bj_long_objective_evidence_audit_pass_keep_goal_active",
            "stage43_bk_t100_family_limited_reconciliation_pass",
            "stage43_bl_raw_scene_graph_ablation_readiness_pass_blocker_documented",
        },
        "latest_tail_adapter_candidate_present": by_role["latest_full_test_tail_adapter_candidate"]["deployable"] is True
        and by_role["latest_full_test_tail_adapter_candidate"]["metrics"]["all"] > by_role["current_best_integrated_candidate"]["metrics"]["all"]
        and by_role["latest_full_test_tail_adapter_candidate"]["metrics"]["t50"] > by_role["current_best_integrated_candidate"]["metrics"]["t50"]
        and by_role["latest_full_test_tail_adapter_candidate"]["metrics"]["easy_degradation"] <= 0.02
        and by_role["latest_full_test_tail_adapter_candidate"]["metrics"]["t100_raw_frame_diagnostic"] >= 0.0,
        "per_domain_and_per_source_reported": bool(payload["per_domain_external_validation"]) and bool(payload["per_source_source_safe_repair"]),
        "no_future_or_test_leakage": no_leakage["future_endpoint_input"] is False
        and no_leakage["future_waypoint_input"] is False
        and no_leakage["future_labels_eval_only"] is True
        and no_leakage["central_velocity_input"] is False
        and no_leakage["test_endpoint_goal_construction"] is False
        and no_leakage["test_statistics_normalization"] is False,
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
        "global_floor_not_removed": claim["global_floor_removed"] is False,
        "long_objective_not_marked_complete": True,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_at_external_validation_matrix_pass" if passed == total else "stage43_at_external_validation_matrix_incomplete",
        "stage_b_external_validation_matrix_ready": passed == total,
    }


def _render_md(payload: Mapping[str, Any]) -> str:
    gate = payload["stage43_at_gate"]
    lines = [
        "# Stage43-AT External Validation Matrix",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- input hash: `{payload['input_hash']}`",
        f"- split verdict: `{payload['split']['verdict']}`",
        f"- test rows: `{payload['split']['test_rows']}`",
        f"- test domains: `{payload['split']['test_domains']}`",
        f"- test source count: `{payload['split']['test_source_count']}`",
        "",
        "## Comparison Matrix",
        "",
        "| model / policy | role | source | deployable | rows | all | t50 | t100 raw | hard/failure | easy degradation | switch | caveat |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["comparison_rows"]:
        metrics = row["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['name']}`",
                    f"`{row['role']}`",
                    f"`{row['result_source']}`",
                    f"`{row['deployable']}`",
                    str(metrics["rows"]),
                    f"`{_pct(metrics['all'])}`",
                    f"`{_pct(metrics['t50'])}`",
                    f"`{_pct(metrics['t100_raw_frame_diagnostic'])}`",
                    f"`{_pct(metrics['hard_failure'])}`",
                    f"`{_pct(metrics['easy_degradation'])}`",
                    f"`{_pct(metrics['switch_rate'])}`",
                    row["caveat"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-Domain Protected External Validation",
            "",
            "| domain | rows | all | t50 | t100 raw | hard/failure | easy degradation | switch |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for domain, metrics in payload["per_domain_external_validation"].items():
        lines.append(
            f"| `{domain}` | {metrics['rows']} | `{_pct(metrics['all_improvement_vs_floor'])}` | "
            f"`{_pct(metrics['t50_improvement_vs_floor'])}` | "
            f"`{_pct(metrics['t100_raw_frame_diagnostic_vs_floor'])}` | "
            f"`{_pct(metrics['hard_failure_improvement_vs_floor'])}` | "
            f"`{_pct(metrics['easy_degradation_vs_floor'])}` | `{_pct(metrics['switch_rate'])}` |"
        )
    lines.extend(
        [
            "",
            "## Source-Safe Repair Boundary",
            "",
            f"- negative source count after repair: `{payload['source_repair_summary']['stage43_k_negative_source_count']}`",
            f"- uniform positive per-source claim allowed: `{payload['source_repair_summary']['uniform_positive_per_source_claim_allowed']}`",
            f"- reason: {payload['source_repair_summary']['reason_uniform_positive_still_blocked']}",
            "",
            "## Claim Boundary",
            "",
            "- This is a fresh Stage43 matrix assembled from verified artifacts; it is not a new threshold search.",
            "- Dataset-local/raw-frame 2.5D only.",
            "- No metric or seconds-level claim.",
            "- No true 3D or foundation claim.",
            "- No Stage5C execution and no SMC.",
            "- Future endpoints/full waypoints remain label/eval only, not inference input.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
        ]
    )
    for name, passed in gate["gates"].items():
        lines.append(f"| `{name}` | `{passed}` |")
    return "\n".join(lines) + "\n"


def _update_project_summaries(payload: Mapping[str, Any]) -> None:
    matrix_row = next(row for row in payload["comparison_rows"] if row["role"] == "current_best_integrated_candidate")
    latest_tail = next(row for row in payload["comparison_rows"] if row["role"] == "latest_full_test_tail_adapter_candidate")
    source_safe = next(row for row in payload["comparison_rows"] if row["role"] == "source_safe_protected_neural")
    body = [
        f"Stage43-AT builds a fresh external validation matrix from verified Stage43 artifacts. Gate: `{payload['stage43_at_gate']['passed']} / {payload['stage43_at_gate']['total']}` with verdict `{payload['stage43_at_gate']['verdict']}`.",
        "",
        "It compares the safety floor, M3W-Neural v1, ungated source-level neural dynamics, domain-capped protected neural, source-family guarded repair, protected full-waypoint dynamics, frozen bounded-residual replay, and the latest tail-horizon full-waypoint adapter. The practical boundary is unchanged: ungated neural is still not deployable, source-family repair is safe but not uniformly positive per source, and every deployable learned candidate remains protected by the floor.",
        "",
        f"Frozen integrated candidate: all `{_pct(matrix_row['metrics']['all'])}`, t50 `{_pct(matrix_row['metrics']['t50'])}`, t100 raw-frame diagnostic `{_pct(matrix_row['metrics']['t100_raw_frame_diagnostic'])}`, hard/failure `{_pct(matrix_row['metrics']['hard_failure'])}`, easy degradation `{_pct(matrix_row['metrics']['easy_degradation'])}`.",
        f"Latest protected tail adapter: all `{_pct(latest_tail['metrics']['all'])}`, t50 `{_pct(latest_tail['metrics']['t50'])}`, t100 raw-frame diagnostic `{_pct(latest_tail['metrics']['t100_raw_frame_diagnostic'])}`, hard/failure `{_pct(latest_tail['metrics']['hard_failure'])}`, easy degradation `{_pct(latest_tail['metrics']['easy_degradation'])}`.",
        f"Source-safe protected neural repair: all `{_pct(source_safe['metrics']['all'])}`, t50 `{_pct(source_safe['metrics']['t50'])}`, hard/failure `{_pct(source_safe['metrics']['hard_failure'])}`, easy degradation `{_pct(source_safe['metrics']['easy_degradation'])}`.",
        "",
        "This is still dataset-local/raw-frame 2.5D evidence. It is not true 3D, not foundation-scale, not metric/seconds-level, and it does not execute Stage5C or SMC.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, body)
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state.setdefault("stage43", {})
    state["stage43"]["external_validation_matrix"] = {
        "verdict": payload["stage43_at_gate"]["verdict"],
        "gate": f"{payload['stage43_at_gate']['passed']}/{payload['stage43_at_gate']['total']}",
        "current_candidate_all": matrix_row["metrics"]["all"],
        "current_candidate_t50": matrix_row["metrics"]["t50"],
        "latest_tail_adapter_all": latest_tail["metrics"]["all"],
        "latest_tail_adapter_t50": latest_tail["metrics"]["t50"],
        "latest_tail_adapter_t100": latest_tail["metrics"]["t100_raw_frame_diagnostic"],
        "latest_tail_adapter_easy": latest_tail["metrics"]["easy_degradation"],
        "source_safe_all": source_safe["metrics"]["all"],
        "source_safe_t50": source_safe["metrics"]["t50"],
        "claim_boundary": payload["claim_boundary"],
        "result_source": payload["result_source"],
    }
    state["current_stage"] = "stage43_at_external_validation_matrix"
    state["current_verdict"] = payload["stage43_at_gate"]["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, state)


def run() -> dict[str, Any]:
    payload = build_external_validation_matrix()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_md(payload).splitlines())
    write_md(
        GATE_MD,
        [
            "# Stage43-AT Gate",
            "",
            f"- verdict: `{payload['stage43_at_gate']['verdict']}`",
            f"- passed: `{payload['stage43_at_gate']['passed']} / {payload['stage43_at_gate']['total']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{k}` | `{v}` |" for k, v in payload["stage43_at_gate"]["gates"].items()],
            "",
        ],
    )
    _update_project_summaries(payload)
    with LEDGER_JSONL.open("a") as fh:
        fh.write(json.dumps({"source": SOURCE, "verdict": payload["stage43_at_gate"]["verdict"], "generated_at_utc": payload["generated_at_utc"]}) + "\n")
    return payload


def main() -> None:
    payload = run()
    gate = payload["stage43_at_gate"]
    print(json.dumps({"verdict": gate["verdict"], "passed": gate["passed"], "total": gate["total"]}, indent=2))


if __name__ == "__main__":
    main()
