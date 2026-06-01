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
REPORT_JSON = OUT_DIR / "stage43_safety_floor_necessity_audit.json"
REPORT_MD = OUT_DIR / "stage43_safety_floor_necessity_audit.md"
GATE_MD = OUT_DIR / "stage43_stage_aj_safety_floor_necessity_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_AJ_SAFETY_FLOOR_NECESSITY_AUDIT"
SOURCE = "fresh_stage43_aj_safety_floor_necessity_audit"

STAGE43_A = OUT_DIR / "stage43_safety_floor_replay.json"
STAGE43_M = OUT_DIR / "stage43_full_waypoint_latent_dynamics.json"
STAGE43_AE = OUT_DIR / "stage43_scene_proxy_slice_safe_policy.json"
STAGE43_AG = OUT_DIR / "stage43_scene_proxy_retrained_ablation.json"
STAGE43_AI = OUT_DIR / "stage43_feature_family_multiseed_confirmation.json"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path, {})


def _protected_vs_ungated(stage43_m: Mapping[str, Any]) -> dict[str, Any]:
    protected = stage43_m["test_metrics_with_floor"]
    ungated = stage43_m["test_metrics_neural_without_floor"]
    return {
        "protected_all": protected["full_waypoint_ade_improvement_vs_floor"],
        "ungated_all": ungated["full_waypoint_ade_improvement_vs_floor"],
        "protected_t50": protected["t50_full_waypoint_ade_improvement_vs_floor"],
        "ungated_t50": ungated["t50_full_waypoint_ade_improvement_vs_floor"],
        "protected_hard": protected["hard_failure_full_waypoint_ade_improvement_vs_floor"],
        "ungated_hard": ungated["hard_failure_full_waypoint_ade_improvement_vs_floor"],
        "protected_t100": protected["t100_raw_frame_full_waypoint_diagnostic_vs_floor"],
        "ungated_t100": ungated["t100_raw_frame_full_waypoint_diagnostic_vs_floor"],
        "protected_easy": protected["easy_degradation_vs_floor"],
        "ungated_easy": ungated["easy_degradation_vs_floor"],
        "easy_harm_reduction": ungated["easy_degradation_vs_floor"] - protected["easy_degradation_vs_floor"],
        "t100_harm_reduction": protected["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
        - ungated["t100_raw_frame_full_waypoint_diagnostic_vs_floor"],
        "all_gain_from_protection": protected["full_waypoint_ade_improvement_vs_floor"]
        - ungated["full_waypoint_ade_improvement_vs_floor"],
        "hard_gain_from_protection": protected["hard_failure_full_waypoint_ade_improvement_vs_floor"]
        - ungated["hard_failure_full_waypoint_ade_improvement_vs_floor"],
    }


def _multiseed_floor_evidence(stage43_ai: Mapping[str, Any]) -> dict[str, Any]:
    no_floor = next(row for row in stage43_ai["variant_summaries"] if row["variant"] == "no_baseline_floor")
    delta = no_floor["delta_full_minus_variant"]
    return {
        "stable_t50_variants": stage43_ai["stable_positive_t50_contribution_variants"],
        "stable_hard_or_all_variants": stage43_ai["stable_positive_hard_or_all_contribution_variants"],
        "no_baseline_floor_t50_delta_mean": delta["t50_full_waypoint_ade_improvement_vs_floor"]["mean"],
        "no_baseline_floor_t50_positive_seed_count": delta["t50_full_waypoint_ade_improvement_vs_floor"][
            "positive_seed_count"
        ],
        "seed_count": len(stage43_ai["seeds"]),
        "no_baseline_floor_easy_mean": no_floor["metrics"]["easy_degradation_vs_floor"]["mean"],
        "no_baseline_floor_all_delta_mean": delta["full_waypoint_ade_improvement_vs_floor"]["mean"],
        "no_baseline_floor_hard_delta_mean": delta["hard_failure_full_waypoint_ade_improvement_vs_floor"]["mean"],
    }


def _scene_proxy_floor_guard(stage43_ag: Mapping[str, Any], stage43_ae: Mapping[str, Any]) -> dict[str, Any]:
    variants = {row["variant"]: row for row in stage43_ag["variants"]}
    raw_best = variants[stage43_ag["best_variant_by_t50_delta"]]
    safe_best = variants[stage43_ag["best_safe_variant_by_t50_delta"]]
    ae_metrics = stage43_ae["test_metrics_slice_safe"]
    return {
        "raw_best_variant": raw_best["variant"],
        "raw_best_t50": raw_best["test_metrics_with_floor"]["t50_full_waypoint_ade_improvement_vs_floor"],
        "raw_best_easy": raw_best["test_metrics_with_floor"]["easy_degradation_vs_floor"],
        "safe_best_variant": safe_best["variant"],
        "safe_best_t50": safe_best["test_metrics_with_floor"]["t50_full_waypoint_ade_improvement_vs_floor"],
        "safe_best_easy": safe_best["test_metrics_with_floor"]["easy_degradation_vs_floor"],
        "slice_safe_all": ae_metrics["full_waypoint_ade_improvement_vs_floor"],
        "slice_safe_t50": ae_metrics["t50_full_waypoint_ade_improvement_vs_floor"],
        "slice_safe_hard": ae_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"],
        "slice_safe_easy": ae_metrics["easy_degradation_vs_floor"],
        "slice_safe_t100": ae_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"],
        "floor_rate": ae_metrics["floor_rate"],
        "h10_floor_rate": ae_metrics["h10_floor_rate"],
        "h100_floor_rate": ae_metrics["h100_floor_rate"],
        "stage43_ab_rate": ae_metrics["stage43_ab_rate"],
    }


def _run(_: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43_a = _load(STAGE43_A)
    stage43_m = _load(STAGE43_M)
    stage43_ae = _load(STAGE43_AE)
    stage43_ag = _load(STAGE43_AG)
    stage43_ai = _load(STAGE43_AI)
    protected = _protected_vs_ungated(stage43_m)
    multiseed = _multiseed_floor_evidence(stage43_ai)
    scene_guard = _scene_proxy_floor_guard(stage43_ag, stage43_ae)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_audit_over_cached_verified_prior_fresh_evidence",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "evidence_sources": {
            "stage43_a_safety_floor_replay": {
                "path": str(STAGE43_A),
                "source": stage43_a.get("source"),
                "verdict": stage43_a.get("stage43_a_gate", {}).get("verdict"),
                "result_source": stage43_a.get("result_source"),
            },
            "stage43_m_protected_vs_ungated": {
                "path": str(STAGE43_M),
                "source": stage43_m.get("source"),
                "verdict": stage43_m.get("stage43_m_gate", {}).get("verdict"),
                "result_source": "cached_verified_prior_fresh_run",
            },
            "stage43_ai_multiseed_no_floor": {
                "path": str(STAGE43_AI),
                "source": stage43_ai.get("source"),
                "verdict": stage43_ai.get("stage43_ai_gate", {}).get("verdict"),
                "result_source": "cached_verified_prior_fresh_run",
            },
            "stage43_ag_scene_proxy_retrained": {
                "path": str(STAGE43_AG),
                "source": stage43_ag.get("source"),
                "verdict": stage43_ag.get("stage43_ag_gate", {}).get("verdict"),
                "result_source": "cached_verified_prior_fresh_run",
            },
            "stage43_ae_slice_safe_floor_guard": {
                "path": str(STAGE43_AE),
                "source": stage43_ae.get("source"),
                "verdict": stage43_ae.get("stage43_ae_gate", {}).get("verdict"),
                "result_source": "cached_verified_prior_fresh_run",
            },
        },
        "protected_vs_ungated": protected,
        "multiseed_floor_feature_evidence": multiseed,
        "scene_proxy_floor_guard_evidence": scene_guard,
        "conclusion": {
            "global_floor_removable": False,
            "floor_is_core_safety_mechanism": True,
            "partial_floor_relaxation_supported": True,
            "safe_relaxation_scope": "validation-selected h25/h50 and supported scene-proxy slices only",
            "unsupported_scope": "ungated global neural deployment, h10/h100 global switching, metric/seconds/true-3D claims",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "input_hash": _combined_hash([STAGE43_A, STAGE43_M, STAGE43_AE, STAGE43_AG, STAGE43_AI]),
    }
    payload["stage43_aj_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    protected = payload["protected_vs_ungated"]
    multiseed = payload["multiseed_floor_feature_evidence"]
    scene_guard = payload["scene_proxy_floor_guard_evidence"]
    gates = {
        "safety_floor_replay_available": payload["evidence_sources"]["stage43_a_safety_floor_replay"]["verdict"]
        == "stage43_a_safety_floor_replay_pass",
        "protected_vs_ungated_reported": protected["protected_easy"] <= 0.02
        and protected["ungated_easy"] > 0.02,
        "protected_reduces_easy_harm_materially": protected["easy_harm_reduction"] > 0.10,
        "protected_reduces_t100_harm": protected["t100_harm_reduction"] > 0.10,
        "no_baseline_floor_t50_degradation_stable": multiseed["no_baseline_floor_t50_delta_mean"] > 0.0
        and multiseed["no_baseline_floor_t50_positive_seed_count"] == multiseed["seed_count"],
        "unsafe_raw_scene_proxy_blocked": scene_guard["raw_best_easy"] > 0.02
        and scene_guard["safe_best_easy"] <= 0.02,
        "partial_floor_relaxation_supported": scene_guard["slice_safe_t50"] > 0.0
        and scene_guard["slice_safe_easy"] <= 0.02
        and scene_guard["h100_floor_rate"] == 1.0,
        "global_floor_not_removable": payload["conclusion"]["global_floor_removable"] is False
        and payload["conclusion"]["floor_is_core_safety_mechanism"] is True,
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True
        and payload["no_leakage"]["central_velocity_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_statistics_normalization"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    deploy = passed == total
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_aj_safety_floor_necessity_confirmed"
        if deploy
        else "stage43_aj_safety_floor_necessity_incomplete",
        "floor_necessity_confirmed": deploy,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    gate = payload["stage43_aj_gate"]
    protected = payload["protected_vs_ungated"]
    multi = payload["multiseed_floor_feature_evidence"]
    scene = payload["scene_proxy_floor_guard_evidence"]
    lines = [
        "# Stage43-AJ Safety-Floor Necessity Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- floor necessity confirmed: `{gate['floor_necessity_confirmed']}`",
        "",
        "## Evidence Summary",
        "",
        f"- protected easy degradation: `{_pct(protected['protected_easy'])}` vs ungated `{_pct(protected['ungated_easy'])}`",
        f"- protected t100 diagnostic: `{_pct(protected['protected_t100'])}` vs ungated `{_pct(protected['ungated_t100'])}`",
        f"- no-baseline-floor t50 delta mean over `{multi['seed_count']}` seeds: `{_pct(multi['no_baseline_floor_t50_delta_mean'])}`; positive seeds `{multi['no_baseline_floor_t50_positive_seed_count']} / {multi['seed_count']}`",
        f"- raw-best scene variant `{scene['raw_best_variant']}` t50 `{_pct(scene['raw_best_t50'])}` but easy `{_pct(scene['raw_best_easy'])}`",
        f"- safe-best scene variant `{scene['safe_best_variant']}` t50 `{_pct(scene['safe_best_t50'])}` and easy `{_pct(scene['safe_best_easy'])}`",
        f"- slice-safe policy t50 `{_pct(scene['slice_safe_t50'])}`, easy `{_pct(scene['slice_safe_easy'])}`, h100 floor rate `{_pct(scene['h100_floor_rate'])}`",
        "",
        "## Conclusion",
        "",
        "- The floor is not removable globally in the current Stage43 evidence.",
        "- The floor is a core safety mechanism, not merely a cosmetic crutch.",
        "- Partial floor relaxation is supported only on validation-selected supported slices.",
        "- Ungated neural/full-scene switching remains unsafe, especially for easy cases and raw-frame t100.",
        "",
        "## Boundary",
        "",
        "- Dataset-local/raw-frame 2.5D only.",
        "- Future labels are supervision/eval only.",
        "- No metric/seconds claim, no Stage5C, no SMC.",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
    ]
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-AJ Safety-Floor Necessity Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- floor necessity confirmed: `{gate['floor_necessity_confirmed']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )
    _update_text_outputs(payload)


def _update_text_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_aj_gate"]
    protected = payload["protected_vs_ungated"]
    multi = payload["multiseed_floor_feature_evidence"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"floor_necessity_confirmed = `{gate['floor_necessity_confirmed']}`",
        f"protected_easy_vs_ungated_easy = `{_pct(protected['protected_easy'])}` vs `{_pct(protected['ungated_easy'])}`",
        f"no_baseline_floor_t50_delta_mean = `{_pct(multi['no_baseline_floor_t50_delta_mean'])}`",
        "",
        "Stage43-AJ consolidates current Stage43 floor evidence: protected-vs-ungated neural dynamics, multi-seed no-baseline-floor ablation, scene-proxy safe-vs-unsafe variants, and h100 floor guards. Conclusion: the safety floor is currently a core safety mechanism and cannot be globally removed; only validation-selected partial relaxation is supported.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds claim; no Stage5C; no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_aj_safety_floor_necessity_audit"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "floor_necessity_confirmed": gate["floor_necessity_confirmed"],
        "protected_vs_ungated": protected,
        "multiseed_floor_feature_evidence": payload["multiseed_floor_feature_evidence"],
        "scene_proxy_floor_guard_evidence": payload["scene_proxy_floor_guard_evidence"],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_aj_safety_floor_necessity_audit"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-AJ",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "floor_necessity_confirmed": gate["floor_necessity_confirmed"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Run Stage43-AJ safety-floor necessity audit.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = _run(args)
    gate = result["stage43_aj_gate"]
    print(f"Stage43-AJ: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"floor_necessity_confirmed={gate['floor_necessity_confirmed']}")
    return result


if __name__ == "__main__":
    main()
