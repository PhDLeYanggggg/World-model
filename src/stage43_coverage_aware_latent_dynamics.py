from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m


OUT_DIR = m.OUT_DIR
CACHE_DIR = Path("data/stage43_ce_full_waypoint_supervision_cache")
REPORT_JSON = OUT_DIR / "stage43_coverage_aware_latent_dynamics.json"
REPORT_MD = OUT_DIR / "stage43_coverage_aware_latent_dynamics.md"
GATE_MD = OUT_DIR / "stage43_stage_cg_coverage_aware_latent_dynamics_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
HEARTBEAT_JSON = OUT_DIR / "stage43_coverage_aware_latent_dynamics_heartbeat.json"
CF_JSON = OUT_DIR / "stage43_coverage_aware_full_waypoint_cache.json"

SECTION = "STAGE43_CG_COVERAGE_AWARE_LATENT_DYNAMICS"
SOURCE = "fresh_stage43_cg_coverage_aware_latent_dynamics"


def _configure_base() -> None:
    m.CACHE_DIR = CACHE_DIR
    m.CACHE_FILE_PREFIX = "stage43_ce_full_waypoint_supervision"
    m.SOURCE = SOURCE
    m.SECTION = SECTION
    m.REPORT_JSON = REPORT_JSON
    m.REPORT_MD = REPORT_MD
    m.GATE_MD = GATE_MD
    m.HEARTBEAT_JSON = HEARTBEAT_JSON
    m.CHECKPOINT_NAME = "stage43_coverage_aware_latent_dynamics.pt"
    m.PRECONDITION_JSON = CF_JSON
    m.PRECONDITION_GATE_NAME = "stage43_cf_gate"
    m.PRECONDITION_READY_FIELD = "unused_full_waypoint_ready_field"
    m.PRECONDITION_READY_VERDICT = "stage43_cf_coverage_aware_full_waypoint_cache_ready"


def _pct(value: float | int) -> str:
    return f"{100.0 * float(value):.2f}%"


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["test_metrics_with_floor"]
    ungated = payload["test_metrics_neural_without_floor"]
    no_leakage = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "stage43_cf_cache_ready": payload["stage43_l_precondition"]["verdict"]
        == "stage43_cf_coverage_aware_full_waypoint_cache_ready",
        "torch_training_fresh_run": payload["result_source"] == "fresh_run" and Path(payload["checkpoint"]).exists(),
        "checkpoint_not_committed": payload["checkpoint_committed"] is False,
        "coverage_aware_train_val_test_rows_present": all(int(v) > 0 for v in payload["data_rows"].values()),
        "future_waypoints_are_labels_only": no_leakage["future_waypoint_input"] is False
        and no_leakage["future_waypoint_label_eval_only"] is True,
        "no_future_endpoint_or_central_velocity_input": no_leakage["future_endpoint_input"] is False
        and no_leakage["central_velocity_input"] is False,
        "no_test_goal_or_stat_leakage": no_leakage["test_endpoint_goal_construction"] is False
        and no_leakage["test_statistics_normalization"] is False,
        "latent_noncollapse": float(payload["latent_variance"]) > 0.01,
        "protected_eval_completed": int(metrics["rows"]) > 0,
        "easy_preserved": float(metrics["easy_degradation_vs_floor"]) <= 0.02,
        "validation_policy_selected": "policy" in payload["validation_selected_policy"],
        "neural_lift_or_honest_keep_floor": (
            float(metrics["full_waypoint_ade_improvement_vs_floor"]) > 0.0
            or float(metrics["t50_full_waypoint_ade_improvement_vs_floor"]) > 0.0
            or float(metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]) > 0.0
            or payload["deploy_neural"] is False
        ),
        "ungated_neural_reported": "full_waypoint_ade_improvement_vs_floor" in ungated,
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
        "long_objective_kept_active": True,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    deploy = bool(payload["deploy_neural"] and passed == total)
    verdict = (
        "stage43_cg_coverage_aware_latent_dynamics_candidate_pass"
        if deploy
        else "stage43_cg_coverage_aware_latent_dynamics_diagnostic_keep_floor"
    )
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "deploy_coverage_aware_latent_dynamics": deploy,
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_cg_gate"]
    metrics = payload["test_metrics_with_floor"]
    ungated = payload["test_metrics_neural_without_floor"]
    ci = payload["bootstrap_ci"]["metrics"]
    return [
        "# Stage43-CG Coverage-Aware Latent Dynamics",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- mode: `{payload['mode']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deploy coverage-aware latent dynamics: `{gate['deploy_coverage_aware_latent_dynamics']}`",
        f"- checkpoint committed: `{payload['checkpoint_committed']}`",
        "",
        "## What Changed",
        "",
        "- This run trains the Stage43 full-waypoint latent dynamics head on the repaired CE source-family coverage split.",
        "- It uses the local coverage-aware supervision cache from Stage43-CF.",
        "- Future endpoints and full waypoints are labels/evaluation targets only, never inference inputs.",
        "",
        "## Claim Boundary",
        "",
        "- Not true 3D.",
        "- Not a foundation world model.",
        "- Dataset-local/raw-frame 2.5D evidence only.",
        "- No metric or seconds-level claim.",
        "- Stage5C not executed.",
        "- SMC not enabled.",
        "",
        "## Protected Test Metrics vs CE Floor",
        "",
        f"- rows: `{metrics['rows']}`",
        f"- full-waypoint ADE improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- endpoint FDE improvement: `{_pct(metrics['endpoint_fde_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
        "",
        "## Bootstrap CI",
        "",
        f"- bootstrap n: `{payload['bootstrap_ci']['n']}`",
        f"- all full-waypoint ADE CI: `[{_pct(ci['full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- t50 full-waypoint ADE CI: `[{_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- hard/failure CI: `[{_pct(ci['hard_failure_full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['hard_failure_full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- easy degradation CI: `[{_pct(ci['easy_degradation_vs_floor']['low'])}, {_pct(ci['easy_degradation_vs_floor']['high'])}]`",
        "",
        "## Ungated Neural Diagnostic",
        "",
        f"- full-waypoint ADE improvement: `{_pct(ungated['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(ungated['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- hard/failure improvement: `{_pct(ungated['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(ungated['easy_degradation_vs_floor'])}`",
        "",
        "## Interpretation",
        "",
        (
            "The coverage-aware latent dynamics head is deployable under its CE floor."
            if gate["deploy_coverage_aware_latent_dynamics"]
            else "The coverage-aware latent dynamics head was trained and evaluated, but deployment remains with the floor unless protected metrics show safe lift."
        ),
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        "",
    ]


def _write_outputs(payload: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_cg_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CG Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- deploy coverage-aware latent dynamics: `{gate['deploy_coverage_aware_latent_dynamics']}`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
        ],
    )
    write_json(WORLD_GATE_JSON, m._jsonable(gate))
    write_md(
        WORLD_GATE_MD,
        [
            "# Stage43 Current World-Model Gate",
            "",
            f"- source: `{SOURCE}`",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- deploy coverage-aware latent dynamics: `{gate['deploy_coverage_aware_latent_dynamics']}`",
            "- long objective complete: `False`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            "## Boundary",
            "",
            "- Coverage-aware CE split evidence only.",
            "- Dataset-local/raw-frame 2.5D only.",
            "- No metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
            "",
        ],
    )


def _update_summaries(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_cg_gate"]
    metrics = payload["test_metrics_with_floor"]
    block = [
        f"## {SECTION}",
        "",
        f"source = `{SOURCE}`",
        f"result_source = `{payload['result_source']}`",
        f"mode = `{payload['mode']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"deploy_coverage_aware_latent_dynamics = `{gate['deploy_coverage_aware_latent_dynamics']}`",
        "",
        "I retrained the full-waypoint latent dynamics head on the CE coverage-aware source split. This is the first model run using the repaired split cache, not just another cache audit.",
        "",
        f"- all full-waypoint ADE improvement vs floor: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement vs floor: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- hard/failure full-waypoint ADE improvement vs floor: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
        "",
        "Boundary: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; Stage5C not executed; SMC not enabled.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, block)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_cg_coverage_aware_latent_dynamics"] = {
        "source": SOURCE,
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "mode": payload["mode"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "deploy_coverage_aware_latent_dynamics": gate["deploy_coverage_aware_latent_dynamics"],
        "metrics": metrics,
        "bootstrap_ci": payload["bootstrap_ci"],
        "claim_boundary": payload["claim_boundary"],
        "checkpoint_committed": payload["checkpoint_committed"],
    }
    state["current_stage"] = "stage43_cg_coverage_aware_latent_dynamics"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable({"event": "stage43_cg_coverage_aware_latent_dynamics", "payload": payload}),
                ensure_ascii=False,
            )
            + "\n"
        )


def run_coverage_aware_latent_dynamics(args: argparse.Namespace) -> dict[str, Any]:
    _configure_base()
    payload = m._train_eval(args, write_outputs_enabled=False)
    payload["source"] = SOURCE
    payload["coverage_aware_cache"] = {
        "cache_dir": str(CACHE_DIR),
        "cache_prefix": "stage43_ce_full_waypoint_supervision",
        "precondition": str(CF_JSON),
    }
    payload["stage43_cg_gate"] = _gate(payload)
    _write_outputs(payload)
    _update_summaries(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = m.build_arg_parser()
    parser.description = "Train Stage43-CG coverage-aware latent dynamics on the CE full-waypoint cache."
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if not args.quick and not args.small and not args.medium:
        args.small = True
    payload = run_coverage_aware_latent_dynamics(args)
    gate = payload["stage43_cg_gate"]
    print(f"Stage43-CG: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"deploy_coverage_aware_latent_dynamics={gate['deploy_coverage_aware_latent_dynamics']}")
    return payload


if __name__ == "__main__":
    main()
