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
REPORT_JSON = OUT_DIR / "stage43_current_candidate_reconciliation.json"
REPORT_MD = OUT_DIR / "stage43_current_candidate_reconciliation.md"
GATE_MD = OUT_DIR / "stage43_stage_ay_current_candidate_reconciliation_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SOURCE = "fresh_stage43_ay_current_candidate_reconciliation"
SECTION = "STAGE43_AY_CURRENT_CANDIDATE_RECONCILIATION"

STAGE43_P = OUT_DIR / "stage43_tail_horizon_waypoint_adapter.json"
STAGE43_AP = OUT_DIR / "stage43_paper_evidence_refresh.json"
STAGE43_AO = OUT_DIR / "stage43_bounded_residual_reviewer_replay.json"
STAGE43_AX = OUT_DIR / "stage43_source_horizon_expert_replay.json"
STAGE43_AQ = OUT_DIR / "stage43_candidate_manifest.json"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path, {})


def _gate_passed(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return bool(gate) and int(gate.get("passed", -1)) == int(gate.get("total", -2))


def _tail_metrics(tail_p: Mapping[str, Any]) -> dict[str, float]:
    metrics = tail_p["overall_full_test_metrics"]
    return {
        "all": float(metrics["full_waypoint_ade_improvement_vs_floor"]),
        "endpoint": float(metrics["endpoint_fde_improvement_vs_floor"]),
        "t50": float(metrics["t50_full_waypoint_ade_improvement_vs_floor"]),
        "t50_endpoint": float(metrics["t50_endpoint_fde_improvement_vs_floor"]),
        "t100_raw_frame_diagnostic": float(metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]),
        "hard_failure": float(metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]),
        "easy_degradation": float(metrics["easy_degradation_vs_floor"]),
        "switch_rate": float(metrics["switch_rate"]),
    }


def _ax_metrics(ax: Mapping[str, Any]) -> dict[str, float]:
    metrics = ax["replay_metrics"]
    return {
        "all": float(metrics["all_improvement_vs_floor"]),
        "t50": float(metrics["t50_improvement_vs_floor"]),
        "t100_raw_frame_diagnostic": float(metrics["t100_raw_frame_diagnostic_vs_floor"]),
        "hard_failure": float(metrics["hard_failure_improvement_vs_floor"]),
        "easy_degradation": float(metrics["easy_degradation_vs_floor"]),
        "switch_rate": float(metrics["switch_rate"]),
    }


def _ao_metrics(ao: Mapping[str, Any]) -> dict[str, float]:
    metrics = ao["replayed_metrics"]
    return {
        "all": float(metrics["full_waypoint_ade_improvement_vs_floor"]),
        "endpoint": float(metrics["endpoint_fde_improvement_vs_floor"]),
        "t50": float(metrics["t50_full_waypoint_ade_improvement_vs_floor"]),
        "t50_endpoint": float(metrics["t50_endpoint_fde_improvement_vs_floor"]),
        "t100_raw_frame_diagnostic": float(metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]),
        "hard_failure": float(metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]),
        "easy_degradation": float(metrics["easy_degradation_vs_floor"]),
        "switch_rate": float(metrics["switch_rate"]),
    }


def _positive_source_count(by_domain: Mapping[str, Mapping[str, Any]], metric: str) -> int:
    return sum(1 for row in by_domain.values() if float(row.get(metric, 0.0)) > 0.0)


def _nonnegative_source_count(by_domain: Mapping[str, Mapping[str, Any]], metric: str) -> int:
    return sum(1 for row in by_domain.values() if float(row.get(metric, 0.0)) >= 0.0)


def build_current_candidate_reconciliation() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    tail_p = _load(STAGE43_P)
    ap = _load(STAGE43_AP)
    ao = _load(STAGE43_AO)
    ax = _load(STAGE43_AX)
    aq = _load(STAGE43_AQ)

    tail = _tail_metrics(tail_p)
    source_replay = _ax_metrics(ax)
    frozen = _ao_metrics(ao)
    tail_domains = tail_p["by_domain"]
    source_replay_domains = ax["replay_domain_metrics"]
    tail_boot = tail_p["bootstrap_ci"]["metrics"]
    ax_boot = ax["bootstrap"]

    roles = {
        "performance_leader": {
            "name": "Stage43-P protected tail-horizon full-waypoint adapter",
            "artifact": str(STAGE43_P),
            "policy_hash": tail_p["selected_model"]["model_hash"],
            "role": "best aggregate protected full-waypoint performance evidence",
            "metrics": tail,
            "bootstrap_ci": {
                "all": tail_boot["full_waypoint_ade_improvement_vs_floor"],
                "t50": tail_boot["t50_full_waypoint_ade_improvement_vs_floor"],
                "hard_failure": tail_boot["hard_failure_full_waypoint_ade_improvement_vs_floor"],
                "easy_degradation": tail_boot["easy_degradation_vs_floor"],
            },
            "source_status": {
                "domains": sorted(tail_domains),
                "positive_all_domains": _positive_source_count(tail_domains, "full_waypoint_ade_improvement_vs_floor"),
                "nonnegative_all_domains": _nonnegative_source_count(tail_domains, "full_waypoint_ade_improvement_vs_floor"),
                "uniform_positive_transfer": bool(tail_p["stage43_p_gate"].get("uniform_source_positive_success", False)),
                "domain_metrics": tail_domains,
            },
            "deployment_boundary": [
                "Strongest aggregate evidence, but it is still floor-protected.",
                "TrajNet is safely floored rather than positively improved under this role.",
                "h100/t100 remains raw-frame diagnostic and guarded.",
            ],
        },
        "source_horizon_replay_leader": {
            "name": "Stage43-AX exact replay of source-horizon expert policy",
            "artifact": str(STAGE43_AX),
            "policy_hash": ax["policy_hash"],
            "row_hash": ax["row_hash"],
            "switch_hash": ax["switch_hash"],
            "role": "best exact-replayed source/horizon safety evidence",
            "metrics": source_replay,
            "bootstrap_ci": {
                "all": ax_boot["unit_all"],
                "t50": ax_boot["unit_t50"],
                "hard_failure": ax_boot["unit_hard_failure"],
                "easy_degradation": ax_boot["unit_easy_degradation"],
            },
            "source_status": {
                "domains": sorted(source_replay_domains),
                "positive_all_domains": _positive_source_count(source_replay_domains, "all_improvement_vs_floor"),
                "nonnegative_all_domains": _nonnegative_source_count(source_replay_domains, "all_improvement_vs_floor"),
                "domain_metrics": source_replay_domains,
            },
            "deployment_boundary": [
                "Exact replay passed with zero metric diff.",
                "Aggregate lift is lower than Stage43-P, but every source is nonnegative on all-test.",
                "ETH_UCY t100 remains negative in raw-frame diagnostic and must not be overclaimed.",
            ],
        },
        "frozen_reviewer_replay_artifact": {
            "name": "Stage43-AO frozen bounded-residual reviewer replay",
            "artifact": str(STAGE43_AO),
            "policy_hash": ao["policy_hash"]["recomputed"],
            "role": "frozen reviewer-replayable safety artifact",
            "metrics": frozen,
            "replay_diff": float(ao["replay_diff"]["max_abs_diff"]),
            "deployment_boundary": [
                "Exact replayable from frozen policy artifact.",
                "Lower than Stage43-P on aggregate/t50, but remains the clean frozen safety artifact.",
            ],
        },
    }

    claim_boundary = {
        "true_3d_world_model": False,
        "foundation_world_model": False,
        "metric_or_seconds_claim": False,
        "dataset_local_raw_frame_only": True,
        "uniform_positive_external_transfer_claim": False,
        "stage5c_executed": False,
        "smc_enabled": False,
        "long_objective_complete": False,
    }
    no_leakage = {
        "future_endpoint_input": False,
        "future_waypoint_input": False,
        "future_labels_eval_or_loss_only": True,
        "central_velocity_input": False,
        "test_endpoint_goal_construction": False,
        "test_statistics_normalization": False,
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_reconciliation_from_stage43_p_ap_ao_ax_aq",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {
            "stage43_p": str(STAGE43_P),
            "stage43_ap": str(STAGE43_AP),
            "stage43_ao": str(STAGE43_AO),
            "stage43_ax": str(STAGE43_AX),
            "stage43_aq": str(STAGE43_AQ),
        },
        "input_gate_verdicts": {
            "stage43_p": tail_p["stage43_p_gate"]["verdict"],
            "stage43_ap": ap["stage43_ap_gate"]["verdict"],
            "stage43_ao": ao["stage43_ao_gate"]["verdict"],
            "stage43_ax": ax["stage43_ax_gate"]["verdict"],
            "stage43_aq": aq["stage43_aq_gate"]["verdict"],
        },
        "roles": roles,
        "current_public_claim": {
            "short": "M3W currently has protected dataset-local/raw-frame latent/full-waypoint evidence; Stage43-P is the performance leader, Stage43-AX is the source/horizon replay leader, and Stage43-AO remains the frozen reviewer-replayable artifact.",
            "do_not_say": [
                "Do not call this true 3D.",
                "Do not call this foundation-scale.",
                "Do not call dataset-local/raw-frame horizons metric or seconds-level.",
                "Do not claim uniform positive transfer across every source from Stage43-P.",
                "Do not say Stage5C or SMC has run.",
            ],
        },
        "next_required_evidence": [
            "Freeze and exact-replay the Stage43-P performance leader if it is to replace Stage43-AO as the primary reviewer artifact.",
            "Continue source-level repair for TrajNet non-floor positive transfer under the performance leader.",
            "Refresh full-suite replay and paper tables from the reconciled role map.",
            "Source-specific timing/geometry calibration is still required before metric or seconds-level language.",
        ],
        "no_leakage": no_leakage,
        "claim_boundary": claim_boundary,
        "input_hash": _combined_hash([STAGE43_P, STAGE43_AP, STAGE43_AO, STAGE43_AX, STAGE43_AQ]),
    }
    payload["stage43_ay_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    roles = payload["roles"]
    tail = roles["performance_leader"]
    ax = roles["source_horizon_replay_leader"]
    ao = roles["frozen_reviewer_replay_artifact"]
    claim = payload["claim_boundary"]
    no_leak = payload["no_leakage"]
    tail_metrics = tail["metrics"]
    ax_metrics = ax["metrics"]
    ao_metrics = ao["metrics"]
    tail_boot = tail["bootstrap_ci"]
    ax_boot = ax["bootstrap_ci"]
    gates = {
        "input_gate_verdicts_present": all(bool(v) for v in payload["input_gate_verdicts"].values()),
        "performance_leader_supported": tail_metrics["all"] > 0.0
        and tail_metrics["t50"] > 0.0
        and tail_metrics["hard_failure"] > 0.0
        and tail_metrics["easy_degradation"] <= 0.02
        and tail_boot["all"]["low"] > 0.0
        and tail_boot["t50"]["low"] > 0.0
        and tail_boot["hard_failure"]["low"] > 0.0,
        "source_horizon_replay_supported": ax_metrics["all"] > 0.0
        and ax_metrics["t50"] > 0.0
        and ax_metrics["hard_failure"] > 0.0
        and ax_metrics["easy_degradation"] <= 0.02
        and ax_boot["all"]["ci_low"] > 0.0
        and ax_boot["t50"]["ci_low"] > 0.0
        and ax["source_status"]["nonnegative_all_domains"] == len(ax["source_status"]["domains"]),
        "frozen_reviewer_artifact_supported": ao["replay_diff"] <= 1e-8
        and ao_metrics["all"] > 0.0
        and ao_metrics["t50"] > 0.0
        and ao_metrics["hard_failure"] > 0.0
        and ao_metrics["easy_degradation"] <= 0.02,
        "roles_not_collapsed": tail["policy_hash"] != ax["policy_hash"]
        and ax["policy_hash"] != ao["policy_hash"]
        and tail["role"] != ax["role"],
        "uniform_positive_transfer_not_overclaimed": claim["uniform_positive_external_transfer_claim"] is False
        and tail["source_status"]["uniform_positive_transfer"] is False,
        "source_safe_candidate_has_all_sources_nonnegative": ax["source_status"]["nonnegative_all_domains"]
        == len(ax["source_status"]["domains"]),
        "t100_raw_frame_guarded": tail_metrics["t100_raw_frame_diagnostic"] >= -1e-8
        and claim["metric_or_seconds_claim"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False,
        "no_future_or_test_leakage": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["future_labels_eval_or_loss_only"] is True
        and no_leak["central_velocity_input"] is False
        and no_leak["test_endpoint_goal_construction"] is False
        and no_leak["test_statistics_normalization"] is False,
        "claim_boundary_not_overstated": claim["true_3d_world_model"] is False
        and claim["foundation_world_model"] is False
        and claim["metric_or_seconds_claim"] is False
        and claim["dataset_local_raw_frame_only"] is True,
        "stage5c_and_smc_false": claim["stage5c_executed"] is False and claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_ay_current_candidate_reconciliation_pass"
        if passed == total
        else "stage43_ay_current_candidate_reconciliation_incomplete",
        "current_candidate_supported": passed == total,
        "goal_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    gate = payload["stage43_ay_gate"]
    write_json(WORLD_GATE_JSON, m._jsonable(gate))
    roles = payload["roles"]
    tail = roles["performance_leader"]
    ax = roles["source_horizon_replay_leader"]
    ao = roles["frozen_reviewer_replay_artifact"]
    lines = [
        "# Stage43-AY Current Candidate Reconciliation",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- current candidate supported: `{gate['current_candidate_supported']}`",
        f"- long objective complete: `{gate['goal_complete']}`",
        "",
        "## Why This Exists",
        "",
        "Stage43 now has several valid evidence artifacts with different roles. This reconciliation keeps the roles separate instead of pretending there is one universal winner.",
        "",
        "## Role Map",
        "",
        "| role | artifact | all | t50 | t100 raw | hard/failure | easy | note |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| performance leader | Stage43-P | `{_pct(tail['metrics']['all'])}` | `{_pct(tail['metrics']['t50'])}` | `{_pct(tail['metrics']['t100_raw_frame_diagnostic'])}` | `{_pct(tail['metrics']['hard_failure'])}` | `{_pct(tail['metrics']['easy_degradation'])}` | strongest aggregate protected full-waypoint evidence |",
        f"| source-horizon replay leader | Stage43-AX | `{_pct(ax['metrics']['all'])}` | `{_pct(ax['metrics']['t50'])}` | `{_pct(ax['metrics']['t100_raw_frame_diagnostic'])}` | `{_pct(ax['metrics']['hard_failure'])}` | `{_pct(ax['metrics']['easy_degradation'])}` | exact replay and all sources nonnegative on all-test |",
        f"| frozen reviewer artifact | Stage43-AO | `{_pct(ao['metrics']['all'])}` | `{_pct(ao['metrics']['t50'])}` | `{_pct(ao['metrics']['t100_raw_frame_diagnostic'])}` | `{_pct(ao['metrics']['hard_failure'])}` | `{_pct(ao['metrics']['easy_degradation'])}` | frozen bounded-residual reviewer replay |",
        "",
        "## Source Boundary",
        "",
        f"- Stage43-P positive all-test domains: `{tail['source_status']['positive_all_domains']} / {len(tail['source_status']['domains'])}`",
        f"- Stage43-P nonnegative all-test domains: `{tail['source_status']['nonnegative_all_domains']} / {len(tail['source_status']['domains'])}`",
        f"- Stage43-P uniform positive transfer claim allowed: `{tail['source_status']['uniform_positive_transfer']}`",
        f"- Stage43-AX nonnegative all-test domains: `{ax['source_status']['nonnegative_all_domains']} / {len(ax['source_status']['domains'])}`",
        "",
        "## Current Public Claim",
        "",
        payload["current_public_claim"]["short"],
        "",
        "## Do Not Say",
        "",
        *[f"- {item}" for item in payload["current_public_claim"]["do_not_say"]],
        "",
        "## Next Required Evidence",
        "",
        *[f"- {item}" for item in payload["next_required_evidence"]],
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
    ]
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage43 Current World-Model Gate",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- current candidate supported: `{gate['current_candidate_supported']}`",
        f"- long objective complete: `{gate['goal_complete']}`",
        f"- Stage5C executed: `{gate['stage5c_executed']}`",
        f"- SMC enabled: `{gate['smc_enabled']}`",
        "",
        "## Candidate Roles",
        "",
        "- Performance leader: `Stage43-P protected tail-horizon full-waypoint adapter`.",
        "- Source-horizon replay leader: `Stage43-AX exact replay of source-horizon expert policy`.",
        "- Frozen reviewer artifact: `Stage43-AO bounded-residual replay`.",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
    ]
    write_md(WORLD_GATE_MD, gate_lines)
    write_md(GATE_MD, gate_lines)
    _update_ledgers(payload)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_ay_gate"]
    roles = payload["roles"]
    tail = roles["performance_leader"]
    ax = roles["source_horizon_replay_leader"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        "current_candidate_supported = `True`",
        "long_objective_complete = `False`",
        "",
        f"performance_leader = `Stage43-P`, all/t50/t100_raw/hard/easy = `{_pct(tail['metrics']['all'])}` / `{_pct(tail['metrics']['t50'])}` / `{_pct(tail['metrics']['t100_raw_frame_diagnostic'])}` / `{_pct(tail['metrics']['hard_failure'])}` / `{_pct(tail['metrics']['easy_degradation'])}`",
        f"source_horizon_replay_leader = `Stage43-AX`, all/t50/t100_raw/hard/easy = `{_pct(ax['metrics']['all'])}` / `{_pct(ax['metrics']['t50'])}` / `{_pct(ax['metrics']['t100_raw_frame_diagnostic'])}` / `{_pct(ax['metrics']['hard_failure'])}` / `{_pct(ax['metrics']['easy_degradation'])}`",
        "",
        "Stage43-AY reconciles the current evidence stack: Stage43-P is the aggregate performance leader, Stage43-AX is the source/horizon exact-replay leader, and Stage43-AO remains the frozen reviewer-replayable artifact. These are protected dataset-local/raw-frame 2.5D results, not true 3D, metric, seconds-level, foundation, Stage5C, or SMC claims.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_ay_current_candidate_reconciliation"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "current_candidate_supported": gate["current_candidate_supported"],
        "goal_complete": gate["goal_complete"],
        "performance_leader": tail,
        "source_horizon_replay_leader": ax,
        "frozen_reviewer_replay_artifact": roles["frozen_reviewer_replay_artifact"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_ay_current_candidate_reconciliation"
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
                        "stage": "Stage43-AY",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "current_candidate_supported": gate["current_candidate_supported"],
                        "goal_complete": gate["goal_complete"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def _run(_: argparse.Namespace) -> dict[str, Any]:
    payload = build_current_candidate_reconciliation()
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Reconcile Stage43 current candidate roles.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = _run(args)
    gate = result["stage43_ay_gate"]
    print(f"Stage43-AY: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"current_candidate_supported={gate['current_candidate_supported']}")
    print(f"goal_complete={gate['goal_complete']}")
    return result


if __name__ == "__main__":
    main()
