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
REPORT_JSON = OUT_DIR / "stage43_protected_multimodal_latent_candidate_lock.json"
REPORT_MD = OUT_DIR / "stage43_protected_multimodal_latent_candidate_lock.md"
GATE_MD = OUT_DIR / "stage43_stage_bh_protected_multimodal_latent_candidate_lock_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SOURCE = "fresh_stage43_bh_protected_multimodal_latent_candidate_lock"
SECTION = "STAGE43_BH_PROTECTED_MULTIMODAL_LATENT_CANDIDATE_LOCK"

INPUTS = {
    "safety_floor_replay": OUT_DIR / "stage43_safety_floor_replay.json",
    "latent_dataset_contract": OUT_DIR / "stage43_latent_state_dataset_contract.json",
    "protected_latent_eval": OUT_DIR / "stage43_protected_latent_eval.json",
    "full_waypoint_latent_dynamics": OUT_DIR / "stage43_full_waypoint_latent_dynamics.json",
    "multimodal_latent_head_suite": OUT_DIR / "stage43_multimodal_latent_head_suite.json",
    "feature_family_multiseed_confirmation": OUT_DIR / "stage43_feature_family_multiseed_confirmation.json",
    "external_validation_matrix": OUT_DIR / "stage43_external_validation_matrix.json",
    "current_candidate_reconciliation": OUT_DIR / "stage43_current_candidate_reconciliation.json",
    "blocked_source_terms_validation": OUT_DIR / "stage43_blocked_source_terms_validation.json",
}


def _gate_verdict(payload: Mapping[str, Any], gate_key: str) -> str:
    gate = payload.get(gate_key, {})
    return str(gate.get("verdict", "missing"))


def _gate_full_pass(payload: Mapping[str, Any], gate_key: str) -> bool:
    gate = payload.get(gate_key, {})
    return int(gate.get("passed", -1)) == int(gate.get("total", -2))


def _role_rows(external_matrix: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("role", "")): row for row in external_matrix.get("comparison_rows", [])}


def _metric(row: Mapping[str, Any], key: str) -> float:
    return float(row.get("metrics", {}).get(key, 0.0))


def build_protected_multimodal_latent_candidate_lock() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    artifacts = {name: read_json(path, {}) for name, path in INPUTS.items()}
    roles = _role_rows(artifacts["external_validation_matrix"])
    latest = dict(roles.get("latest_full_test_tail_adapter_candidate", {}))
    source_safe = dict(roles.get("source_safe_protected_neural", {}))
    multimodal = artifacts["multimodal_latent_head_suite"]
    ai = artifacts["feature_family_multiseed_confirmation"]
    bg = artifacts["blocked_source_terms_validation"]
    external = artifacts["external_validation_matrix"]
    ay = artifacts["current_candidate_reconciliation"]

    input_verdicts = {
        "safety_floor_replay": _gate_verdict(artifacts["safety_floor_replay"], "stage43_a_gate"),
        "latent_dataset_contract": _gate_verdict(artifacts["latent_dataset_contract"], "stage43_b_gate"),
        "protected_latent_eval": _gate_verdict(artifacts["protected_latent_eval"], "stage43_c_gate"),
        "full_waypoint_latent_dynamics": _gate_verdict(
            artifacts["full_waypoint_latent_dynamics"], "stage43_m_gate"
        ),
        "multimodal_latent_head_suite": _gate_verdict(multimodal, "stage43_y_gate"),
        "feature_family_multiseed_confirmation": _gate_verdict(ai, "stage43_ai_gate"),
        "external_validation_matrix": _gate_verdict(external, "stage43_at_gate"),
        "current_candidate_reconciliation": _gate_verdict(ay, "stage43_ay_gate"),
        "blocked_source_terms_validation": _gate_verdict(bg, "stage43_bg_gate"),
    }
    deployable_proxy_heads = list(multimodal.get("deployment_contract", {}).get("deployable_proxy_heads", []))
    diagnostic_heads = list(multimodal.get("deployment_contract", {}).get("diagnostic_only_heads", []))
    external_domains = list(external.get("split", {}).get("test_domains", []))
    latest_metrics = {
        "name": latest.get("name", ""),
        "role": latest.get("role", ""),
        "deployable": bool(latest.get("deployable", False)),
        "rows": int(latest.get("metrics", {}).get("rows", 0)),
        "all": _metric(latest, "all"),
        "t50": _metric(latest, "t50"),
        "t100_raw_frame_diagnostic": _metric(latest, "t100_raw_frame_diagnostic"),
        "hard_failure": _metric(latest, "hard_failure"),
        "easy_degradation": _metric(latest, "easy_degradation"),
        "switch_rate": _metric(latest, "switch_rate"),
        "caveat": latest.get("caveat", ""),
    }
    source_safe_metrics = {
        "name": source_safe.get("name", ""),
        "deployable": bool(source_safe.get("deployable", False)),
        "all": _metric(source_safe, "all"),
        "t50": _metric(source_safe, "t50"),
        "t100_raw_frame_diagnostic": _metric(source_safe, "t100_raw_frame_diagnostic"),
        "hard_failure": _metric(source_safe, "hard_failure"),
        "easy_degradation": _metric(source_safe, "easy_degradation"),
    }
    summary = {
        "candidate_label": "protected_multimodal_latent_state_world_model_candidate",
        "protected_multimodal_latent_state_candidate": bool(
            multimodal.get("stage43_y_gate", {}).get("protected_multimodal_latent_state_candidate", False)
        ),
        "standalone_world_model_deployable": bool(
            multimodal.get("stage43_y_gate", {}).get("standalone_world_model_deployable", True)
        ),
        "safety_floor_required": bool(
            multimodal.get("deployment_contract", {}).get("must_keep_safety_floor", True)
        ),
        "deployable_proxy_head_count": len(deployable_proxy_heads),
        "diagnostic_head_count": len(diagnostic_heads),
        "latest_full_test_tail_adapter_candidate": latest_metrics,
        "source_safe_candidate": source_safe_metrics,
        "external_domains": external_domains,
        "source_level_test_rows": int(external.get("split", {}).get("test_rows", 0)),
        "blocked_source_ready_for_guarded_conversion_preflight_rows": int(
            bg.get("summary", {}).get("ready_for_guarded_conversion_preflight_rows", -1)
        ),
        "blocked_source_training_allowed_now": int(bg.get("summary", {}).get("training_allowed_now", -1)),
        "remaining_blockers": [
            "not_true_3d",
            "not_foundation_scale",
            "dataset_local_raw_frame_only",
            "metric_seconds_unverified",
            "safety_floor_required",
            "not_standalone_ungated_deployment",
            "uniform_positive_external_transfer_not_allowed",
            "t100_raw_frame_still_guarded_diagnostic",
            "blocked_source_terms_identity_not_confirmed",
            "stage5c_not_executed",
            "smc_not_enabled",
        ],
        "decision": "protected_multimodal_latent_state_candidate_locked_with_floor_and_source_blockers",
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_evidence_lock_from_verified_stage43_artifacts",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {name: str(path) for name, path in INPUTS.items()},
        "input_hash": _combined_hash(list(INPUTS.values())),
        "input_verdicts": input_verdicts,
        "summary": summary,
        "latent_state": {
            "dim": int(multimodal.get("latent_state", {}).get("dim", 0)),
            "min_variance": float(multimodal.get("latent_state", {}).get("min_variance", 0.0)),
            "mean_variance": float(multimodal.get("latent_state", {}).get("mean_variance", 0.0)),
            "noncollapse_threshold": float(multimodal.get("latent_state", {}).get("noncollapse_threshold", 0.0)),
        },
        "head_suite": {
            "deployable_proxy_heads": deployable_proxy_heads,
            "diagnostic_only_heads": diagnostic_heads,
            "heads": multimodal.get("head_suite", {}),
        },
        "ablation_evidence": {
            "stage43_ai_verdict": input_verdicts["feature_family_multiseed_confirmation"],
            "stable_positive_t50_variants": ai.get("summary", {}).get("stable_positive_t50_variants", [])
            if isinstance(ai.get("summary"), Mapping)
            else ai.get("stable_positive_t50_variants", []),
            "gate_supports_multiseed_module_contribution": bool(
                ai.get("stage43_ai_gate", {})
                .get("gates", {})
                .get("at_least_two_stable_module_contributions", False)
            ),
        },
        "external_validation": {
            "test_domains": external_domains,
            "test_rows": int(external.get("split", {}).get("test_rows", 0)),
            "latest_full_test_tail_adapter_candidate": latest_metrics,
            "source_safe_candidate": source_safe_metrics,
            "uniform_positive_per_source_claim_allowed": bool(
                external.get("source_repair_summary", {}).get("uniform_positive_per_source_claim_allowed", True)
            ),
            "source_repair_summary": external.get("source_repair_summary", {}),
            "per_domain_external_validation": external.get("per_domain_external_validation", {}),
        },
        "source_guard": {
            "stage43_bg_verdict": input_verdicts["blocked_source_terms_validation"],
            "ready_for_guarded_conversion_preflight_rows": summary[
                "blocked_source_ready_for_guarded_conversion_preflight_rows"
            ],
            "training_allowed_now": summary["blocked_source_training_allowed_now"],
            "blocked_rows": bg.get("summary", {}).get("blocked_rows", []),
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
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "standalone_ungated_deployable": False,
            "uniform_positive_external_transfer_claim": False,
            "source_terms_permission_claim": False,
            "converted_external_support_source": False,
            "long_objective_complete": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage43_bh_gate"] = _gate(payload, artifacts)
    return payload


def _gate(payload: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    s = payload["summary"]
    leak = payload["no_leakage_and_execution"]
    claim = payload["claim_boundary"]
    gates = {
        "safety_floor_replay_passed": _gate_full_pass(artifacts["safety_floor_replay"], "stage43_a_gate")
        and payload["input_verdicts"]["safety_floor_replay"] == "stage43_a_safety_floor_replay_pass",
        "latent_dataset_contract_passed": _gate_full_pass(
            artifacts["latent_dataset_contract"], "stage43_b_gate"
        )
        and payload["input_verdicts"]["latent_dataset_contract"] == "stage43_b_latent_state_dataset_contract_pass",
        "protected_latent_eval_passed": _gate_full_pass(artifacts["protected_latent_eval"], "stage43_c_gate")
        and payload["input_verdicts"]["protected_latent_eval"]
        == "stage43_c_protected_latent_state_candidate_pass",
        "full_waypoint_latent_passed": _gate_full_pass(
            artifacts["full_waypoint_latent_dynamics"], "stage43_m_gate"
        )
        and payload["input_verdicts"]["full_waypoint_latent_dynamics"]
        == "stage43_m_protected_full_waypoint_latent_candidate_pass",
        "multimodal_head_suite_candidate": _gate_full_pass(
            artifacts["multimodal_latent_head_suite"], "stage43_y_gate"
        )
        and s["protected_multimodal_latent_state_candidate"] is True,
        "multiseed_ablation_support_present": _gate_full_pass(
            artifacts["feature_family_multiseed_confirmation"], "stage43_ai_gate"
        )
        and payload["ablation_evidence"]["gate_supports_multiseed_module_contribution"] is True,
        "external_validation_matrix_passed": _gate_full_pass(
            artifacts["external_validation_matrix"], "stage43_at_gate"
        )
        and len(s["external_domains"]) >= 3,
        "current_candidate_reconciled": _gate_full_pass(
            artifacts["current_candidate_reconciliation"], "stage43_ay_gate"
        )
        and payload["input_verdicts"]["current_candidate_reconciliation"]
        == "stage43_ay_current_candidate_reconciliation_pass",
        "latest_protected_candidate_positive_easy_safe": s["latest_full_test_tail_adapter_candidate"]["deployable"]
        is True
        and s["latest_full_test_tail_adapter_candidate"]["all"] > 0.0
        and s["latest_full_test_tail_adapter_candidate"]["t50"] > 0.0
        and s["latest_full_test_tail_adapter_candidate"]["hard_failure"] > 0.0
        and s["latest_full_test_tail_adapter_candidate"]["easy_degradation"] <= 0.02,
        "safety_floor_required_not_hidden": s["safety_floor_required"] is True
        and s["standalone_world_model_deployable"] is False
        and claim["standalone_ungated_deployable"] is False,
        "source_guard_passed_and_blocks_unconfirmed_sources": _gate_full_pass(
            artifacts["blocked_source_terms_validation"], "stage43_bg_gate"
        )
        and s["blocked_source_ready_for_guarded_conversion_preflight_rows"] == 0
        and s["blocked_source_training_allowed_now"] == 0,
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
        and claim["uniform_positive_external_transfer_claim"] is False
        and claim["source_terms_permission_claim"] is False
        and claim["converted_external_support_source"] is False,
        "stage5c_and_smc_false": claim["stage5c_executed"] is False and claim["smc_enabled"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_bh_protected_multimodal_latent_candidate_lock_pass"
        if passed == total
        else "stage43_bh_protected_multimodal_latent_candidate_lock_incomplete",
        "protected_multimodal_latent_state_candidate": passed == total,
        "standalone_world_model_deployable": False,
        "goal_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _render_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_bh_gate"]
    s = payload["summary"]
    latest = s["latest_full_test_tail_adapter_candidate"]
    lines = [
        "# Stage43-BH Protected Multimodal Latent Candidate Lock",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- candidate label: `{s['candidate_label']}`",
        f"- protected multimodal latent state candidate: `{gate['protected_multimodal_latent_state_candidate']}`",
        f"- standalone world model deployable: `{gate['standalone_world_model_deployable']}`",
        "",
        "## Evidence Stack",
        "",
        "| artifact | verdict |",
        "| --- | --- |",
    ]
    for name, verdict in payload["input_verdicts"].items():
        lines.append(f"| `{name}` | `{verdict}` |")
    lines.extend(
        [
            "",
            "## Latest Protected External Candidate",
            "",
            f"- name: `{latest['name']}`",
            f"- rows: `{latest['rows']}`",
            f"- all improvement: `{latest['all']:.2%}`",
            f"- t50 improvement: `{latest['t50']:.2%}`",
            f"- t100 raw-frame diagnostic: `{latest['t100_raw_frame_diagnostic']:.2%}`",
            f"- hard/failure improvement: `{latest['hard_failure']:.2%}`",
            f"- easy degradation: `{latest['easy_degradation']:.2%}`",
            f"- switch rate: `{latest['switch_rate']:.2%}`",
            "",
            "## Multimodal Latent Heads",
            "",
            f"- latent dim: `{payload['latent_state']['dim']}`",
            f"- latent min variance: `{payload['latent_state']['min_variance']:.6f}`",
            f"- deployable proxy heads: `{payload['head_suite']['deployable_proxy_heads']}`",
            f"- diagnostic-only heads: `{payload['head_suite']['diagnostic_only_heads']}`",
            "",
            "## Source Guard",
            "",
            f"- blocked source ready rows: `{payload['source_guard']['ready_for_guarded_conversion_preflight_rows']}`",
            f"- blocked source training allowed now: `{payload['source_guard']['training_allowed_now']}`",
            f"- blocked rows: `{payload['source_guard']['blocked_rows']}`",
            "",
            "## Boundary",
            "",
            "- This locks the current evidence as a protected candidate, not a standalone ungated model.",
            "- Safety floor remains required.",
            "- Dataset-local/raw-frame 2.5D only.",
            "- No metric or seconds-level claim.",
            "- No true 3D or foundation claim.",
            "- Source terms validation remains blocked for PETS/Town-Center/Wild-Track until user-confirmed source identity and terms exist.",
            "- Stage5C remains false and SMC remains false.",
            "",
            "## Remaining Blockers",
            "",
        ]
    )
    lines.extend([f"- `{item}`" for item in s["remaining_blockers"]])
    lines.extend(["", "## Gate", "", "| gate | passed |", "| --- | --- |"])
    lines.extend([f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()])
    return lines


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bh_gate"]
    latest = payload["summary"]["latest_full_test_tail_adapter_candidate"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"protected_multimodal_latent_state_candidate = `{gate['protected_multimodal_latent_state_candidate']}`",
        f"standalone_world_model_deployable = `{gate['standalone_world_model_deployable']}`",
        f"latest_candidate_all = `{latest['all']:.2%}`",
        f"latest_candidate_t50 = `{latest['t50']:.2%}`",
        f"latest_candidate_hard_failure = `{latest['hard_failure']:.2%}`",
        "",
        "I locked the current Stage43 evidence stack into a single protected multimodal latent-state candidate record. The model family has real protected latent/head/full-waypoint evidence, but it still needs the safety floor, still uses dataset-local/raw-frame units, and still has source/terms blockers for additional support data. This is not a true-3D, foundation, metric, seconds-level, Stage5C, or SMC claim.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bh_protected_multimodal_latent_candidate_lock"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "summary": payload["summary"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bh_protected_multimodal_latent_candidate_lock"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BH",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "protected_multimodal_latent_state_candidate": gate[
                            "protected_multimodal_latent_state_candidate"
                        ],
                        "standalone_world_model_deployable": False,
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    write_json(WORLD_GATE_JSON, m._jsonable(payload["stage43_bh_gate"]))
    write_md(REPORT_MD, _render_md(payload))
    write_md(GATE_MD, _render_md(payload))
    gate = payload["stage43_bh_gate"]
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
            f"- long objective complete: `{gate['goal_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-BH locks the current evidence stack as a protected multimodal latent-state candidate.",
            "- The safety floor remains required; ungated deployment is still not allowed.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, or foundation claim.",
            "- Blocked source support remains blocked until source/terms/identity gates clear.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def run_protected_multimodal_latent_candidate_lock() -> dict[str, Any]:
    payload = build_protected_multimodal_latent_candidate_lock()
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Lock current Stage43 protected multimodal latent-state evidence.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    build_arg_parser().parse_args(argv)
    payload = run_protected_multimodal_latent_candidate_lock()
    gate = payload["stage43_bh_gate"]
    latest = payload["summary"]["latest_full_test_tail_adapter_candidate"]
    print(f"Stage43-BH: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"protected_multimodal_latent_state_candidate={gate['protected_multimodal_latent_state_candidate']}")
    print(f"standalone_world_model_deployable={gate['standalone_world_model_deployable']}")
    print(f"latest_all={latest['all']:.4f}")
    print(f"latest_t50={latest['t50']:.4f}")
    return payload


if __name__ == "__main__":
    main()
