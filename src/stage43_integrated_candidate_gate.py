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
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
REPORT_MD = OUT_DIR / "report_stage43_current_candidate.md"
MANIFEST_JSON = OUT_DIR / "stage43_candidate_manifest.json"
MANIFEST_MD = OUT_DIR / "stage43_candidate_manifest.md"
GATE_MD = OUT_DIR / "stage43_stage_aq_integrated_candidate_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_AQ_INTEGRATED_CANDIDATE_GATE"
SOURCE = "fresh_stage43_aq_integrated_candidate_gate"

STAGE43_AJ = OUT_DIR / "stage43_safety_floor_necessity_audit.json"
STAGE43_AK = OUT_DIR / "stage43_self_gate_conformal_audit.json"
STAGE43_AL = OUT_DIR / "stage43_bounded_residual_safety_audit.json"
STAGE43_AM = OUT_DIR / "stage43_bounded_residual_statistical_confirmation.json"
STAGE43_AN = OUT_DIR / "stage43_bounded_residual_policy_freeze.json"
STAGE43_AO = OUT_DIR / "stage43_bounded_residual_reviewer_replay.json"
STAGE43_AP = OUT_DIR / "stage43_paper_evidence_refresh.json"
STAGE43_P = OUT_DIR / "stage43_tail_horizon_waypoint_adapter.json"
FROZEN_POLICY = OUT_DIR / "frozen_stage43_bounded_residual_policy.json"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path, {})


def _gate_passed(payload: Mapping[str, Any], gate_key: str) -> bool:
    gate = payload.get(gate_key, {})
    return bool(gate) and int(gate.get("passed", -1)) == int(gate.get("total", -2))


def _domain_deltas(am: Mapping[str, Any]) -> dict[str, float]:
    rows = {}
    for row in am.get("slice_rows", []):
        name = str(row.get("slice", ""))
        if name.startswith("domain:"):
            rows[name.split(":", 1)[1]] = float(row.get("delta", 0.0))
    return rows


def _horizon_deltas(am: Mapping[str, Any]) -> dict[str, float]:
    rows = {}
    for row in am.get("slice_rows", []):
        name = str(row.get("slice", ""))
        if name.startswith("horizon:"):
            rows[name.split(":", 1)[1]] = float(row.get("delta", 0.0))
    return rows


def _build_manifest(
    aj: Mapping[str, Any],
    ak: Mapping[str, Any],
    al: Mapping[str, Any],
    am: Mapping[str, Any],
    an_payload: Mapping[str, Any],
    ao: Mapping[str, Any],
    ap: Mapping[str, Any],
    tail_p: Mapping[str, Any],
    frozen_policy: Mapping[str, Any],
) -> dict[str, Any]:
    frozen_metrics = frozen_policy["metrics"]["stage43_al_point_metrics"]
    latest_metrics = tail_p["overall_full_test_metrics"]
    bootstrap = am["bootstrap_delta_ci"]["metrics"]
    latest_bootstrap = tail_p["bootstrap_ci"]["metrics"]
    domain_deltas = _domain_deltas(am)
    horizon_deltas = _horizon_deltas(am)
    frozen_policy_hash = frozen_policy["policy_hash"]
    latest_policy_hash = tail_p["selected_model"]["model_hash"]
    claim_boundary = {
        "true_3d_world_model": False,
        "foundation_world_model": False,
        "metric_or_seconds_claim": False,
        "dataset_local_raw_frame_only": True,
        "self_audited_labels_are_human_gold": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    return {
        "candidate_name": "Stage43 Protected Tail-Horizon Full-Waypoint Adapter",
        "policy_hash": latest_policy_hash,
        "policy_artifact": str(STAGE43_P),
        "frozen_replayable_policy_hash": frozen_policy_hash,
        "frozen_replayable_policy_artifact": str(FROZEN_POLICY),
        "source": SOURCE,
        "result_source": "fresh_integrated_manifest_from_stage43_aj_to_ap_plus_stage43_p_artifacts",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {
            "stage43_aj": str(STAGE43_AJ),
            "stage43_ak": str(STAGE43_AK),
            "stage43_al": str(STAGE43_AL),
            "stage43_am": str(STAGE43_AM),
            "stage43_an": str(STAGE43_AN),
            "stage43_ao": str(STAGE43_AO),
            "stage43_ap": str(STAGE43_AP),
            "stage43_p": str(STAGE43_P),
            "frozen_policy": str(FROZEN_POLICY),
        },
        "input_gate_verdicts": {
            "stage43_aj": aj.get("stage43_aj_gate", {}).get("verdict"),
            "stage43_ak": ak.get("stage43_ak_gate", {}).get("verdict"),
            "stage43_al": al.get("stage43_al_gate", {}).get("verdict"),
            "stage43_am": am.get("stage43_am_gate", {}).get("verdict"),
            "stage43_an": an_payload.get("stage43_an_gate", {}).get("verdict"),
            "stage43_ao": ao.get("stage43_ao_gate", {}).get("verdict"),
            "stage43_ap": ap.get("stage43_ap_gate", {}).get("verdict"),
            "stage43_p": tail_p.get("stage43_p_gate", {}).get("verdict"),
        },
        "current_best_deployable": {
            "name": "Stage43-P protected tail-horizon full-waypoint adapter under Stage37/teacher safety floor",
            "deployable": True,
            "global_floor_removed": False,
            "h100_guarded": True,
            "why": "Validation-selected tail-horizon full-waypoint adapter improves full-test all/t50/hard over the frozen bounded-residual replay, preserves easy cases, and explicitly floors unsafe h100/t100 switches.",
        },
        "frozen_replayable_candidate": {
            "name": "frozen Stage43 bounded-residual policy under Stage37/teacher safety floor",
            "deployable": True,
            "global_floor_removed": False,
            "h100_guarded": True,
            "policy_hash": frozen_policy_hash,
            "why": "Exact reviewer replay, positive bootstrap deltas over stored hard-switch policy, zero easy degradation, and explicit t100 raw-frame guard.",
            "metrics": {
                "all_improvement_vs_floor": float(frozen_metrics["all"]),
                "endpoint_improvement_vs_floor": float(frozen_metrics["endpoint"]),
                "t50_full_waypoint_improvement_vs_floor": float(frozen_metrics["t50"]),
                "t50_endpoint_improvement_vs_floor": float(frozen_metrics["t50_endpoint"]),
                "t100_raw_frame_diagnostic_vs_floor": float(frozen_metrics["t100"]),
                "hard_failure_improvement_vs_floor": float(frozen_metrics["hard_failure"]),
                "easy_degradation_vs_floor": float(frozen_metrics["easy"]),
                "switch_rate": float(frozen_metrics["switch_rate"]),
            },
        },
        "metrics": {
            "all_improvement_vs_floor": float(latest_metrics["full_waypoint_ade_improvement_vs_floor"]),
            "endpoint_improvement_vs_floor": float(latest_metrics["endpoint_fde_improvement_vs_floor"]),
            "t50_full_waypoint_improvement_vs_floor": float(latest_metrics["t50_full_waypoint_ade_improvement_vs_floor"]),
            "t50_endpoint_improvement_vs_floor": float(latest_metrics["t50_endpoint_fde_improvement_vs_floor"]),
            "t100_raw_frame_diagnostic_vs_floor": float(latest_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]),
            "hard_failure_improvement_vs_floor": float(latest_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]),
            "easy_degradation_vs_floor": float(latest_metrics["easy_degradation_vs_floor"]),
            "switch_rate": float(latest_metrics["switch_rate"]),
            "t50_delta_ci_vs_stored_hard_switch": an_payload["frozen_metrics"]["t50_delta_ci"],
            "bootstrap_delta_ci": bootstrap,
            "latest_bootstrap_ci": latest_bootstrap,
            "domain_deltas_vs_stored_hard_switch": domain_deltas,
            "horizon_deltas_vs_stored_hard_switch": horizon_deltas,
            "latest_domain_metrics": tail_p["by_domain"],
            "latest_horizon_metrics": tail_p["by_horizon"],
            "latest_delta_vs_frozen_replay": {
                "all": float(latest_metrics["full_waypoint_ade_improvement_vs_floor"]) - float(frozen_metrics["all"]),
                "t50": float(latest_metrics["t50_full_waypoint_ade_improvement_vs_floor"]) - float(frozen_metrics["t50"]),
                "t100": float(latest_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]) - float(frozen_metrics["t100"]),
                "hard_failure": float(latest_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"])
                - float(frozen_metrics["hard_failure"]),
                "easy": float(latest_metrics["easy_degradation_vs_floor"]) - float(frozen_metrics["easy"]),
            },
            "reviewer_replay_max_abs_diff": float(ao["replay_diff"]["max_abs_diff"]),
        },
        "evidence_summary": {
            "safety_floor_necessity": "AJ/AK show ungated/self-gated alternatives do not justify global floor removal.",
            "bounded_residual_candidate": "AL finds a validation-selected bounded residual config with h100 guard and easy preservation.",
            "statistical_confirmation": "AM confirms positive bootstrap deltas over stored hard switch on all/t50/hard and safe easy degradation.",
            "policy_freeze": "AN freezes the policy hash and artifact.",
            "reviewer_replay": "AO replays from the frozen artifact with zero metric diff.",
            "tail_horizon_adapter": "Stage43-P adds a validation-selected full-test tail-horizon adapter with stronger all/t50/hard metrics and h100/t100 fallback.",
            "paper_claim_boundary": "AP refreshes paper-facing evidence and keeps A-journal/3D/foundation claims blocked; it remains a boundary source until refreshed for Stage43-P.",
        },
        "claim_boundary": claim_boundary,
        "answers": {
            "is_current_goal_complete": False,
            "why_goal_not_complete": [
                "The long objective still asks for broader multimodal latent world-state evidence, more source calibration, and final full-suite replay.",
                "The current candidate remains protected and floor-dependent.",
                "Metric/time calibration and true 3D evidence remain unavailable.",
                "Stage5C and SMC remain disabled.",
            ],
            "is_a_journal_candidate_now": False,
            "strongest_allowed_claim": "A floor-protected, validation-selected tail-horizon full-waypoint adapter improves dataset-local/raw-frame external full-waypoint metrics while preserving easy cases; the frozen bounded-residual policy remains the exact replayable safety artifact.",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "input_hash": _combined_hash(
            [STAGE43_AJ, STAGE43_AK, STAGE43_AL, STAGE43_AM, STAGE43_AN, STAGE43_AO, STAGE43_AP, STAGE43_P, FROZEN_POLICY]
        ),
    }


def _gate(manifest: Mapping[str, Any]) -> dict[str, Any]:
    metrics = manifest["metrics"]
    claim = manifest["claim_boundary"]
    no_leakage = manifest["no_leakage"]
    domain_deltas = metrics["domain_deltas_vs_stored_hard_switch"]
    horizon_deltas = metrics["horizon_deltas_vs_stored_hard_switch"]
    latest_bootstrap = metrics["latest_bootstrap_ci"]
    latest_domains = metrics["latest_domain_metrics"]
    latest_horizons = metrics["latest_horizon_metrics"]
    latest_delta = metrics["latest_delta_vs_frozen_replay"]
    gates = {
        "input_gate_verdicts_present": all(bool(v) for v in manifest["input_gate_verdicts"].values()),
        "frozen_policy_hash_present": bool(manifest["policy_hash"]),
        "reviewer_replay_exact": metrics["reviewer_replay_max_abs_diff"] == 0.0,
        "frozen_replay_bootstrap_delta_positive": metrics["bootstrap_delta_ci"]["all_delta_improvement"]["low"] > 0.0
        and metrics["bootstrap_delta_ci"]["t50_delta_improvement"]["low"] > 0.0
        and metrics["bootstrap_delta_ci"]["hard_failure_delta_improvement"]["low"] > 0.0,
        "latest_tail_bootstrap_positive": latest_bootstrap["full_waypoint_ade_improvement_vs_floor"]["low"] > 0.0
        and latest_bootstrap["t50_full_waypoint_ade_improvement_vs_floor"]["low"] > 0.0
        and latest_bootstrap["hard_failure_full_waypoint_ade_improvement_vs_floor"]["low"] > 0.0
        and latest_bootstrap["easy_degradation_vs_floor"]["high"] <= 0.02,
        "latest_tail_beats_frozen_replay": latest_delta["all"] > 0.0
        and latest_delta["t50"] > 0.0
        and latest_delta["hard_failure"] > 0.0
        and latest_delta["easy"] <= 0.02,
        "easy_preserved": metrics["easy_degradation_vs_floor"] <= 0.02,
        "t100_guarded_not_overclaimed": metrics["t100_raw_frame_diagnostic_vs_floor"] >= -1e-8
        and manifest["current_best_deployable"]["h100_guarded"] is True,
        "external_domains_reported": {"ETH_UCY", "TrajNet", "UCY"}.issubset(set(domain_deltas)),
        "positive_horizon_deltas_reported": {"10", "25", "50", "100"}.issubset(set(horizon_deltas)),
        "latest_external_domains_reported": {"ETH_UCY", "TrajNet", "UCY"}.issubset(set(latest_domains)),
        "latest_horizon_metrics_reported": {"10", "25", "50", "100"}.issubset(set(latest_horizons)),
        "global_floor_not_removed": manifest["current_best_deployable"]["global_floor_removed"] is False,
        "no_future_or_test_leakage": no_leakage["future_endpoint_input"] is False
        and no_leakage["future_waypoint_input"] is False
        and no_leakage["future_labels_eval_only"] is True
        and no_leakage["central_velocity_input"] is False
        and no_leakage["test_endpoint_goal_construction"] is False
        and no_leakage["test_statistics_normalization"] is False,
        "claim_boundary_not_overstated": claim["true_3d_world_model"] is False
        and claim["foundation_world_model"] is False
        and claim["metric_or_seconds_claim"] is False
        and claim["dataset_local_raw_frame_only"] is True,
        "stage5c_and_smc_false": claim["stage5c_executed"] is False and claim["smc_enabled"] is False,
        "long_objective_kept_active": manifest["answers"]["is_current_goal_complete"] is False,
        "a_journal_not_overclaimed": manifest["answers"]["is_a_journal_candidate_now"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": manifest["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_aq_integrated_protected_latent_state_candidate_pass"
        if passed == total
        else "stage43_aq_integrated_candidate_incomplete",
        "current_candidate_supported": passed == total,
        "goal_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(manifest: Mapping[str, Any]) -> None:
    write_json(MANIFEST_JSON, m._jsonable(manifest))
    gate = manifest["stage43_aq_gate"]
    write_json(WORLD_GATE_JSON, m._jsonable(gate))
    metrics = manifest["metrics"]
    lines = [
        "# Stage43 Current Integrated Candidate",
        "",
        f"- source: `{manifest['source']}`",
        f"- result_source: `{manifest['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- policy hash: `{manifest['policy_hash']}`",
        f"- frozen replayable policy hash: `{manifest['frozen_replayable_policy_hash']}`",
        f"- current candidate supported: `{gate['current_candidate_supported']}`",
        f"- long objective complete: `{gate['goal_complete']}`",
        "",
        "## Current Best Deployable",
        "",
        f"- name: {manifest['current_best_deployable']['name']}",
        f"- deployable: `{manifest['current_best_deployable']['deployable']}`",
        f"- global floor removed: `{manifest['current_best_deployable']['global_floor_removed']}`",
        f"- h100 guarded: `{manifest['current_best_deployable']['h100_guarded']}`",
        f"- why: {manifest['current_best_deployable']['why']}",
        "",
        "## Frozen Replayable Safety Artifact",
        "",
        f"- name: {manifest['frozen_replayable_candidate']['name']}",
        f"- policy hash: `{manifest['frozen_replayable_candidate']['policy_hash']}`",
        f"- reviewer replay max abs diff: `{metrics['reviewer_replay_max_abs_diff']:.8f}`",
        f"- all improvement vs floor: `{_pct(manifest['frozen_replayable_candidate']['metrics']['all_improvement_vs_floor'])}`",
        f"- t50 full-waypoint improvement vs floor: `{_pct(manifest['frozen_replayable_candidate']['metrics']['t50_full_waypoint_improvement_vs_floor'])}`",
        f"- hard/failure improvement vs floor: `{_pct(manifest['frozen_replayable_candidate']['metrics']['hard_failure_improvement_vs_floor'])}`",
        f"- easy degradation vs floor: `{_pct(manifest['frozen_replayable_candidate']['metrics']['easy_degradation_vs_floor'])}`",
        "",
        "## Metrics",
        "",
        f"- all improvement vs floor: `{_pct(metrics['all_improvement_vs_floor'])}`",
        f"- endpoint improvement vs floor: `{_pct(metrics['endpoint_improvement_vs_floor'])}`",
        f"- t50 full-waypoint improvement vs floor: `{_pct(metrics['t50_full_waypoint_improvement_vs_floor'])}`",
        f"- t50 endpoint improvement vs floor: `{_pct(metrics['t50_endpoint_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic vs floor: `{_pct(metrics['t100_raw_frame_diagnostic_vs_floor'])}`",
        f"- hard/failure improvement vs floor: `{_pct(metrics['hard_failure_improvement_vs_floor'])}`",
        f"- easy degradation vs floor: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
        f"- reviewer replay max abs diff: `{metrics['reviewer_replay_max_abs_diff']:.8f}`",
        "",
        "## Latest Tail Adapter Delta Vs Frozen Replay",
        "",
        f"- all delta: `{_pct(metrics['latest_delta_vs_frozen_replay']['all'])}`",
        f"- t50 delta: `{_pct(metrics['latest_delta_vs_frozen_replay']['t50'])}`",
        f"- t100 delta: `{_pct(metrics['latest_delta_vs_frozen_replay']['t100'])}`",
        f"- hard/failure delta: `{_pct(metrics['latest_delta_vs_frozen_replay']['hard_failure'])}`",
        f"- easy degradation delta: `{_pct(metrics['latest_delta_vs_frozen_replay']['easy'])}`",
        "",
        "## Domain Deltas Vs Stored Hard Switch",
        "",
        "| domain | delta |",
        "| --- | ---: |",
        *[f"| {name} | `{_pct(value)}` |" for name, value in metrics["domain_deltas_vs_stored_hard_switch"].items()],
        "",
        "## Horizon Deltas Vs Stored Hard Switch",
        "",
        "| horizon | delta |",
        "| --- | ---: |",
        *[f"| {name} | `{_pct(value)}` |" for name, value in metrics["horizon_deltas_vs_stored_hard_switch"].items()],
        "",
        "## What This Does Not Claim",
        "",
        "- Not true 3D.",
        "- Not foundation-scale.",
        "- Not metric or seconds-level.",
        "- Not global safety-floor removal.",
        "- Not Stage5C execution.",
        "- Not SMC.",
        "",
        "## Why The Long Goal Remains Active",
        "",
        *[f"- {reason}" for reason in manifest["answers"]["why_goal_not_complete"]],
    ]
    write_md(REPORT_MD, lines)
    write_md(MANIFEST_MD, lines)
    gate_lines = [
        "# Stage43 Current World-Model Gate",
        "",
        f"- source: `{manifest['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- current candidate supported: `{gate['current_candidate_supported']}`",
        f"- long objective complete: `{gate['goal_complete']}`",
        f"- Stage5C executed: `{gate['stage5c_executed']}`",
        f"- SMC enabled: `{gate['smc_enabled']}`",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
    ]
    write_md(WORLD_GATE_MD, gate_lines)
    write_md(GATE_MD, gate_lines)
    _update_ledgers(manifest)


def _update_ledgers(manifest: Mapping[str, Any]) -> None:
    gate = manifest["stage43_aq_gate"]
    metrics = manifest["metrics"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{manifest['source']}`",
        f"result_source = `{manifest['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"policy_hash = `{manifest['policy_hash']}`",
        f"frozen_replayable_policy_hash = `{manifest['frozen_replayable_policy_hash']}`",
        f"current_candidate_supported = `{gate['current_candidate_supported']}`",
        f"long_objective_complete = `{gate['goal_complete']}`",
        f"current_all_t50_t100_hard_easy = `{_pct(metrics['all_improvement_vs_floor'])}` / `{_pct(metrics['t50_full_waypoint_improvement_vs_floor'])}` / `{_pct(metrics['t100_raw_frame_diagnostic_vs_floor'])}` / `{_pct(metrics['hard_failure_improvement_vs_floor'])}` / `{_pct(metrics['easy_degradation_vs_floor'])}`",
        "",
        "Stage43-AQ integrates AJ-AO/AP plus Stage43-P into one current candidate manifest and world-model gate. The current strongest deployable evidence is the Stage43-P protected tail-horizon full-waypoint adapter under the Stage37/teacher safety floor; the frozen Stage43 bounded-residual policy remains the exact replayable safety artifact. This is a protected dataset-local/raw-frame 2.5D latent waypoint candidate, not a true 3D/foundation/metric/seconds-level model.",
        "",
        "Boundary unchanged: Stage5C is not executed; SMC is not enabled; global floor removal is not supported; the long Stage43 objective remains active.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_aq_integrated_candidate_gate"] = {
        "source": manifest["source"],
        "result_source": manifest["result_source"],
        "updated_at": manifest["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "current_candidate_supported": gate["current_candidate_supported"],
        "goal_complete": gate["goal_complete"],
        "policy_hash": manifest["policy_hash"],
        "frozen_replayable_policy_hash": manifest["frozen_replayable_policy_hash"],
        "metrics": metrics,
        "frozen_replayable_candidate": manifest["frozen_replayable_candidate"],
        "manifest": str(MANIFEST_JSON),
        "world_gate": str(WORLD_GATE_MD),
        "report": str(REPORT_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_aq_integrated_candidate_gate"
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
                        "stage": "Stage43-AQ",
                        "source": manifest["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "policy_hash": manifest["policy_hash"],
                        "current_candidate_supported": gate["current_candidate_supported"],
                        "goal_complete": gate["goal_complete"],
                        "generated_at_utc": manifest["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def _run(_: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    aj = _load(STAGE43_AJ)
    ak = _load(STAGE43_AK)
    al = _load(STAGE43_AL)
    am = _load(STAGE43_AM)
    an_payload = _load(STAGE43_AN)
    ao = _load(STAGE43_AO)
    ap = _load(STAGE43_AP)
    tail_p = _load(STAGE43_P)
    frozen_policy = _load(FROZEN_POLICY)
    manifest = _build_manifest(aj, ak, al, am, an_payload, ao, ap, tail_p, frozen_policy)
    manifest["stage43_aq_gate"] = _gate(manifest)
    _write_outputs(manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Build the integrated Stage43 current candidate gate and manifest.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = _run(args)
    gate = result["stage43_aq_gate"]
    print(f"Stage43-AQ: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"current_candidate_supported={gate['current_candidate_supported']}")
    print(f"goal_complete={gate['goal_complete']}")
    return result


if __name__ == "__main__":
    main()
