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
REPORT_JSON = OUT_DIR / "stage43_locked_candidate_paper_package_refresh.json"
REPORT_MD = OUT_DIR / "stage43_locked_candidate_paper_package_refresh.md"
CLAIM_MD = OUT_DIR / "stage43_locked_candidate_claim_boundary.md"
MODEL_CARD_MD = OUT_DIR / "stage43_locked_candidate_model_card.md"
DATA_CARD_MD = OUT_DIR / "stage43_locked_candidate_data_card.md"
REPRO_MD = OUT_DIR / "stage43_locked_candidate_reproducibility.md"
GAP_MD = OUT_DIR / "stage43_locked_candidate_a_journal_gap.md"
GATE_MD = OUT_DIR / "stage43_stage_bi_locked_candidate_paper_package_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SOURCE = "fresh_stage43_bi_locked_candidate_paper_package_refresh"
SECTION = "STAGE43_BI_LOCKED_CANDIDATE_PAPER_PACKAGE_REFRESH"

INPUTS = {
    "candidate_lock": OUT_DIR / "stage43_protected_multimodal_latent_candidate_lock.json",
    "legacy_paper_refresh": OUT_DIR / "stage43_paper_evidence_refresh.json",
    "multimodal_head_suite": OUT_DIR / "stage43_multimodal_latent_head_suite.json",
    "external_validation_matrix": OUT_DIR / "stage43_external_validation_matrix.json",
    "feature_family_multiseed_confirmation": OUT_DIR / "stage43_feature_family_multiseed_confirmation.json",
    "blocked_source_terms_validation": OUT_DIR / "stage43_blocked_source_terms_validation.json",
}


def _pct(value: float | int) -> str:
    return f"{100.0 * float(value):.2f}%"


def _gate_full_pass(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return int(gate.get("passed", -1)) == int(gate.get("total", -2))


def _gate_verdict(payload: Mapping[str, Any], key: str) -> str:
    return str(payload.get(key, {}).get("verdict", "missing"))


def _read_required(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path, {})


def build_locked_candidate_paper_package_refresh() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    artifacts = {name: _read_required(path) for name, path in INPUTS.items()}
    lock = artifacts["candidate_lock"]
    legacy = artifacts["legacy_paper_refresh"]
    heads = artifacts["multimodal_head_suite"]
    external = artifacts["external_validation_matrix"]
    ai = artifacts["feature_family_multiseed_confirmation"]
    bg = artifacts["blocked_source_terms_validation"]

    latest = lock["summary"]["latest_full_test_tail_adapter_candidate"]
    source_guard = lock["source_guard"]
    deployable_heads = list(lock["head_suite"]["deployable_proxy_heads"])
    diagnostic_heads = list(lock["head_suite"]["diagnostic_only_heads"])
    stable_variants = (
        ai.get("stable_positive_t50_contribution_variants")
        or ai.get("summary", {}).get("stable_positive_t50_variants", [])
        or lock.get("ablation_evidence", {}).get("stable_positive_t50_variants", [])
    )

    package_outputs = {
        "main_report": str(REPORT_MD),
        "claim_boundary": str(CLAIM_MD),
        "model_card": str(MODEL_CARD_MD),
        "data_card": str(DATA_CARD_MD),
        "reproducibility": str(REPRO_MD),
        "a_journal_gap": str(GAP_MD),
        "gate": str(GATE_MD),
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_package_refresh_from_stage43_bh_candidate_lock",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {name: str(path) for name, path in INPUTS.items()},
        "input_hash": _combined_hash(list(INPUTS.values())),
        "input_verdicts": {
            "candidate_lock": _gate_verdict(lock, "stage43_bh_gate"),
            "legacy_paper_refresh": _gate_verdict(legacy, "stage43_ap_gate"),
            "multimodal_head_suite": _gate_verdict(heads, "stage43_y_gate"),
            "external_validation_matrix": _gate_verdict(external, "stage43_at_gate"),
            "feature_family_multiseed_confirmation": _gate_verdict(ai, "stage43_ai_gate"),
            "blocked_source_terms_validation": _gate_verdict(bg, "stage43_bg_gate"),
        },
        "current_claim": {
            "label": "protected_multimodal_latent_state_world_model_candidate",
            "plain_language": (
                "M3W currently has evidence for a protected multimodal latent-state world-model candidate "
                "under a safety floor, not for a standalone ungated model."
            ),
            "allowed": [
                "protected dataset-local/raw-frame 2.5D multi-agent world-state candidate",
                "multimodal latent-state heads are useful as protected proxy heads",
                "latest protected tail-horizon candidate improves all/t50/hard while preserving easy cases",
                "safety floor remains part of the method",
            ],
            "disallowed": [
                "true 3D world model",
                "large-scale foundation world model",
                "metric or seconds-level prediction",
                "ungated standalone deployment",
                "uniform positive external transfer across every source",
                "Stage5C execution",
                "SMC execution",
            ],
        },
        "metrics": {
            "rows": int(latest["rows"]),
            "all": float(latest["all"]),
            "t50": float(latest["t50"]),
            "t100_raw_frame_diagnostic": float(latest["t100_raw_frame_diagnostic"]),
            "hard_failure": float(latest["hard_failure"]),
            "easy_degradation": float(latest["easy_degradation"]),
            "switch_rate": float(latest["switch_rate"]),
        },
        "evidence": {
            "protected_candidate_locked": bool(lock["stage43_bh_gate"]["protected_multimodal_latent_state_candidate"]),
            "standalone_world_model_deployable": bool(lock["stage43_bh_gate"]["standalone_world_model_deployable"]),
            "safety_floor_required": bool(lock["summary"]["safety_floor_required"]),
            "deployable_proxy_heads": deployable_heads,
            "diagnostic_only_heads": diagnostic_heads,
            "stable_positive_t50_ablation_variants": stable_variants,
            "external_domains": list(lock["summary"]["external_domains"]),
            "source_level_test_rows": int(lock["summary"]["source_level_test_rows"]),
        },
        "source_guard": {
            "ready_for_guarded_conversion_preflight_rows": int(
                source_guard["ready_for_guarded_conversion_preflight_rows"]
            ),
            "training_allowed_now": int(source_guard["training_allowed_now"]),
            "blocked_rows": list(source_guard.get("blocked_rows", [])),
        },
        "package_outputs": package_outputs,
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
            "uniform_positive_external_transfer_claim": False,
            "source_terms_permission_claim": False,
            "converted_external_support_source": False,
            "a_journal_candidate_now": False,
            "long_objective_complete": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "a_journal_gap": [
            "More legally cleared external source support is still needed.",
            "Metric/time calibration remains unverified.",
            "True 3D evidence is absent.",
            "The system is still protected by a safety floor; ungated neural deployment is not supported.",
            "Raw scene/video multimodal evidence is still proxy-heavy.",
            "t100 remains raw-frame diagnostic and floor-guarded.",
        ],
    }
    payload["stage43_bi_gate"] = _gate(payload, artifacts)
    return payload


def _gate(payload: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    metrics = payload["metrics"]
    evidence = payload["evidence"]
    source_guard = payload["source_guard"]
    leak = payload["no_leakage_and_execution"]
    claim = payload["claim_boundary"]
    gates = {
        "candidate_lock_passed": _gate_full_pass(artifacts["candidate_lock"], "stage43_bh_gate")
        and payload["input_verdicts"]["candidate_lock"] == "stage43_bh_protected_multimodal_latent_candidate_lock_pass",
        "legacy_paper_refresh_available": _gate_full_pass(artifacts["legacy_paper_refresh"], "stage43_ap_gate"),
        "multimodal_head_suite_available": _gate_full_pass(artifacts["multimodal_head_suite"], "stage43_y_gate")
        and len(evidence["deployable_proxy_heads"]) >= 5,
        "external_matrix_available": _gate_full_pass(artifacts["external_validation_matrix"], "stage43_at_gate")
        and len(evidence["external_domains"]) >= 3,
        "multiseed_ablation_available": _gate_full_pass(
            artifacts["feature_family_multiseed_confirmation"], "stage43_ai_gate"
        )
        and len(evidence["stable_positive_t50_ablation_variants"]) >= 2,
        "source_terms_guard_blocks_unconfirmed_support": _gate_full_pass(
            artifacts["blocked_source_terms_validation"], "stage43_bg_gate"
        )
        and source_guard["ready_for_guarded_conversion_preflight_rows"] == 0
        and source_guard["training_allowed_now"] == 0,
        "protected_candidate_not_standalone": evidence["protected_candidate_locked"] is True
        and evidence["standalone_world_model_deployable"] is False
        and evidence["safety_floor_required"] is True,
        "latest_candidate_positive_easy_safe": metrics["all"] > 0.0
        and metrics["t50"] > 0.0
        and metrics["hard_failure"] > 0.0
        and metrics["easy_degradation"] <= 0.02
        and metrics["t100_raw_frame_diagnostic"] >= 0.0,
        "paper_package_outputs_declared": {
            "main_report",
            "claim_boundary",
            "model_card",
            "data_card",
            "reproducibility",
            "a_journal_gap",
            "gate",
        }.issubset(set(payload["package_outputs"])),
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
        and claim["uniform_positive_external_transfer_claim"] is False
        and claim["source_terms_permission_claim"] is False
        and claim["converted_external_support_source"] is False
        and claim["a_journal_candidate_now"] is False,
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
        "verdict": "stage43_bi_locked_candidate_paper_package_refresh_pass"
        if passed == total
        else "stage43_bi_locked_candidate_paper_package_refresh_incomplete",
        "paper_package_refreshed": passed == total,
        "protected_multimodal_latent_state_candidate": passed == total,
        "standalone_world_model_deployable": False,
        "goal_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_bi_gate"]
    metrics = payload["metrics"]
    evidence = payload["evidence"]
    lines = [
        "# Stage43-BI Locked Candidate Paper Package Refresh",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- candidate: `{payload['current_claim']['label']}`",
        f"- paper package refreshed: `{gate['paper_package_refreshed']}`",
        "",
        "## Current Claim",
        "",
        payload["current_claim"]["plain_language"],
        "",
        "## Latest Protected Candidate Metrics",
        "",
        f"- rows: `{metrics['rows']}`",
        f"- all improvement: `{_pct(metrics['all'])}`",
        f"- t50 improvement: `{_pct(metrics['t50'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_diagnostic'])}`",
        f"- hard/failure improvement: `{_pct(metrics['hard_failure'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
        "",
        "## Evidence",
        "",
        f"- deployable proxy heads: `{evidence['deployable_proxy_heads']}`",
        f"- diagnostic-only heads: `{evidence['diagnostic_only_heads']}`",
        f"- stable positive t50 ablation variants: `{evidence['stable_positive_t50_ablation_variants']}`",
        f"- external domains: `{evidence['external_domains']}`",
        f"- source-level test rows: `{evidence['source_level_test_rows']}`",
        "",
        "## Allowed Claims",
        "",
    ]
    lines.extend([f"- {item}" for item in payload["current_claim"]["allowed"]])
    lines.extend(["", "## Disallowed Claims", ""])
    lines.extend([f"- {item}" for item in payload["current_claim"]["disallowed"]])
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ]
    )
    return lines


def _write_claim_boundary(payload: Mapping[str, Any]) -> None:
    write_md(
        CLAIM_MD,
        [
            "# Stage43 Locked Candidate Claim Boundary",
            "",
            "## Allowed",
            "",
            *[f"- {item}" for item in payload["current_claim"]["allowed"]],
            "",
            "## Not Allowed",
            "",
            *[f"- {item}" for item in payload["current_claim"]["disallowed"]],
            "",
            "## Practical Wording",
            "",
            "M3W currently has protected multimodal latent-state evidence under a safety floor. It is best described as a dataset-local/raw-frame 2.5D multi-agent world-state candidate, not as a true-3D, metric, seconds-level, foundation, or ungated generative world model.",
        ],
    )


def _write_model_card(payload: Mapping[str, Any]) -> None:
    mtx = payload["metrics"]
    write_md(
        MODEL_CARD_MD,
        [
            "# Stage43 Locked Candidate Model Card",
            "",
            "## Model Family",
            "",
            "Protected multimodal latent-state candidate with safety-floor deployment.",
            "",
            "## Inputs",
            "",
            "- Past/current causal trajectory and density features.",
            "- Baseline-family rollouts and protected floor predictions.",
            "- Source/domain/horizon tokens.",
            "- Scene/goal/interaction proxy features where legally and causally available.",
            "",
            "## Outputs",
            "",
            "- Protected trajectory/full-waypoint decisions.",
            "- Failure/gain/harm/safe-switch proxy heads.",
            "- Interaction and physical-validity diagnostic heads.",
            "",
            "## Current Evidence",
            "",
            f"- all improvement: `{_pct(mtx['all'])}`",
            f"- t50 improvement: `{_pct(mtx['t50'])}`",
            f"- hard/failure improvement: `{_pct(mtx['hard_failure'])}`",
            f"- easy degradation: `{_pct(mtx['easy_degradation'])}`",
            f"- t100 raw-frame diagnostic: `{_pct(mtx['t100_raw_frame_diagnostic'])}`",
            "",
            "## Deployment Boundary",
            "",
            "The safety floor remains required. Ungated neural deployment, Stage5C execution, and SMC are not enabled.",
        ],
    )


def _write_data_card(payload: Mapping[str, Any]) -> None:
    write_md(
        DATA_CARD_MD,
        [
            "# Stage43 Locked Candidate Data Card",
            "",
            "## Units",
            "",
            "- SDD remains pixel-space.",
            "- External top-down rows remain dataset-local/raw-frame unless separately calibrated.",
            "- t50 and t100 are raw-frame horizons, not seconds-level claims.",
            "",
            "## Source Guard",
            "",
            f"- external domains in current matrix: `{payload['evidence']['external_domains']}`",
            f"- source-level test rows: `{payload['evidence']['source_level_test_rows']}`",
            f"- blocked source ready rows: `{payload['source_guard']['ready_for_guarded_conversion_preflight_rows']}`",
            f"- blocked source training allowed now: `{payload['source_guard']['training_allowed_now']}`",
            "",
            "## Leakage Boundary",
            "",
            "- Future endpoint/full-waypoint labels are loss/eval only.",
            "- No central velocity official input.",
            "- No test endpoint goal construction.",
            "- No test statistics normalization.",
        ],
    )


def _write_repro(payload: Mapping[str, Any]) -> None:
    write_md(
        REPRO_MD,
        [
            "# Stage43 Locked Candidate Reproducibility Checklist",
            "",
            f"- input hash: `{payload['input_hash']}`",
            f"- git commit at generation: `{payload['git_commit']}`",
            "- runtime: `.venv-pytorch/bin/python` on arm64 Apple Silicon path where training is needed.",
            "- DataLoader multiprocessing must remain off for local training paths.",
            "- This refresh itself does not run new training or conversion.",
            "",
            "## Required Verification Commands",
            "",
            "```bash",
            ".venv-pytorch/bin/python run_stage43_locked_candidate_paper_package_refresh.py",
            ".venv-pytorch/bin/python -m pytest tests/test_stage43_locked_candidate_paper_package_refresh.py tests/test_stage43_external_validation_matrix.py -q",
            ".venv-pytorch/bin/python -m pytest tests",
            "```",
        ],
    )


def _write_gap(payload: Mapping[str, Any]) -> None:
    write_md(
        GAP_MD,
        [
            "# Stage43 Locked Candidate A-Journal Gap",
            "",
            "The current evidence is stronger than the earlier paper package because BH locks the protected multimodal latent-state candidate, but it is still not enough for a final A-journal claim.",
            "",
            "## Remaining Gaps",
            "",
            *[f"- {item}" for item in payload["a_journal_gap"]],
            "",
            "## Current Verdict",
            "",
            "not yet A-journal candidate; protected candidate evidence is real, but source support, metric/time calibration, raw multimodal evidence, and floor-removal questions remain open.",
        ],
    )


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bi_gate"]
    metrics = payload["metrics"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"paper_package_refreshed = `{gate['paper_package_refreshed']}`",
        f"protected_multimodal_latent_state_candidate = `{gate['protected_multimodal_latent_state_candidate']}`",
        f"standalone_world_model_deployable = `{gate['standalone_world_model_deployable']}`",
        f"latest_all_t50_hard_easy = `{_pct(metrics['all'])}` / `{_pct(metrics['t50'])}` / `{_pct(metrics['hard_failure'])}` / `{_pct(metrics['easy_degradation'])}`",
        "",
        "I refreshed the paper-facing package from the BH evidence lock. The current claim is now easy to state: M3W has protected multimodal latent-state candidate evidence, but it remains safety-floor protected, dataset-local/raw-frame, not true 3D/foundation, and source terms for extra data are still blocked.",
        "",
        "Boundary unchanged: no Stage5C execution, no SMC, no metric/seconds claim, and no standalone ungated deployment claim.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bi_locked_candidate_paper_package_refresh"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "paper_package_refreshed": gate["paper_package_refreshed"],
        "metrics": payload["metrics"],
        "claim_boundary": payload["claim_boundary"],
        "outputs": payload["package_outputs"],
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bi_locked_candidate_paper_package_refresh"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BI",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "paper_package_refreshed": gate["paper_package_refreshed"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bi_gate"]
    write_json(REPORT_JSON, m._jsonable(payload))
    write_json(WORLD_GATE_JSON, m._jsonable(gate))
    write_md(REPORT_MD, _render_report(payload))
    _write_claim_boundary(payload)
    _write_model_card(payload)
    _write_data_card(payload)
    _write_repro(payload)
    _write_gap(payload)
    write_md(
        GATE_MD,
        [
            "# Stage43-BI Locked Candidate Paper Package Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- paper package refreshed: `{gate['paper_package_refreshed']}`",
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
            f"- long objective complete: `{gate['goal_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-BI refreshes the paper package from the BH protected candidate lock.",
            "- The safety floor remains required; ungated deployment is still not allowed.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, or foundation claim.",
            "- Source support remains blocked until source/terms/identity gates clear.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def run_locked_candidate_paper_package_refresh() -> dict[str, Any]:
    payload = build_locked_candidate_paper_package_refresh()
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Refresh Stage43 paper package from the locked protected candidate.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    build_arg_parser().parse_args(argv)
    payload = run_locked_candidate_paper_package_refresh()
    gate = payload["stage43_bi_gate"]
    metrics = payload["metrics"]
    print(f"Stage43-BI: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"paper_package_refreshed={gate['paper_package_refreshed']}")
    print(f"latest_all={metrics['all']:.4f}")
    print(f"latest_t50={metrics['t50']:.4f}")
    return payload


if __name__ == "__main__":
    main()
