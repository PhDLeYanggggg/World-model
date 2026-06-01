from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_self_gate_conformal_audit as ak


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_bounded_residual_policy_freeze.json"
REPORT_MD = OUT_DIR / "stage43_bounded_residual_policy_freeze.md"
GATE_MD = OUT_DIR / "stage43_stage_an_bounded_residual_policy_freeze_gate.md"
POLICY_JSON = OUT_DIR / "frozen_stage43_bounded_residual_policy.json"
POLICY_MD = OUT_DIR / "frozen_stage43_bounded_residual_policy.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_AN_BOUNDED_RESIDUAL_POLICY_FREEZE"
SOURCE = "fresh_stage43_an_bounded_residual_policy_freeze"

STAGE43_AL = OUT_DIR / "stage43_bounded_residual_safety_audit.json"
STAGE43_AM = OUT_DIR / "stage43_bounded_residual_statistical_confirmation.json"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _stable_hash(obj: Mapping[str, Any]) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _tracked_by_git(path: Path) -> bool:
    try:
        out = subprocess.check_output(["git", "ls-files", str(path)], text=True).strip()
    except Exception:
        return False
    return bool(out)


def _load_required(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path, {})


def _build_policy(al: Mapping[str, Any], am: Mapping[str, Any]) -> dict[str, Any]:
    config = al["best_safe_bounded_residual"]["config"]
    policy = {
        "policy_name": "stage43_bounded_residual_policy_v1",
        "policy_type": "protected_bounded_residual_latent_waypoint",
        "version": "stage43-an",
        "result_source": "frozen_from_stage43_al_am_fresh_evidence",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "deployment_rule": {
            "base_prediction": "Stage43-M floor waypoint",
            "residual_source": "Stage43-M latent full-waypoint neural head",
            "formula": "selected = floor_waypoint + alpha * clip_norm(neural_waypoint - floor_waypoint)",
            "fallback": "floor waypoint",
            "global_floor_removal": False,
            "h100_guard": bool(config.get("force_h100_floor", False)),
            "test_threshold_tuning": False,
        },
        "bounded_residual_config": config,
        "safety_constraints": {
            "easy_degradation_max": 0.02,
            "t100_raw_frame_guard": bool(config.get("force_h100_floor", False)),
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "metrics": {
            "stage43_al_point_metrics": al["best_safe_bounded_residual"],
            "stage43_am_bootstrap_delta_ci": am["bootstrap_delta_ci"],
            "stage43_am_slice_rows": am["slice_rows"],
        },
        "evidence": {
            "stage43_al_verdict": al["stage43_al_gate"]["verdict"],
            "stage43_al_gate": f"{al['stage43_al_gate']['passed']} / {al['stage43_al_gate']['total']}",
            "stage43_am_verdict": am["stage43_am_gate"]["verdict"],
            "stage43_am_gate": f"{am['stage43_am_gate']['passed']} / {am['stage43_am_gate']['total']}",
            "stage43_m_replay_max_abs_diff": am["stored_policy_replay_diff"]["max_abs_diff"],
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    policy["policy_hash"] = _stable_hash({k: v for k, v in policy.items() if k != "policy_hash"})
    return policy


def _run(_: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    al = _load_required(STAGE43_AL)
    am = _load_required(STAGE43_AM)
    stage43_m = _load_required(ak.STAGE43_M)
    policy = _build_policy(al, am)
    checkpoint = ak.STAGE43_M_CKPT
    hashes = {
        "stage43_m_report_sha256": m._sha256(ak.STAGE43_M),
        "stage43_m_checkpoint_sha256": m._sha256(checkpoint),
        "stage43_al_report_sha256": m._sha256(STAGE43_AL),
        "stage43_am_report_sha256": m._sha256(STAGE43_AM),
        "cache_row_hashes": am["cache_row_hashes"],
        "policy_hash": policy["policy_hash"],
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_freeze_from_statistically_confirmed_stage43_am_candidate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "policy": policy,
        "policy_artifact": str(POLICY_JSON),
        "hashes": hashes,
        "evidence_sources": {
            "stage43_m": {
                "report": str(ak.STAGE43_M),
                "checkpoint": str(checkpoint),
                "verdict": stage43_m.get("stage43_m_gate", {}).get("verdict"),
                "checkpoint_tracked_by_git": _tracked_by_git(checkpoint),
            },
            "stage43_al": {
                "report": str(STAGE43_AL),
                "verdict": al.get("stage43_al_gate", {}).get("verdict"),
                "deploy_bounded_residual": al.get("stage43_al_gate", {}).get("deploy_bounded_residual"),
            },
            "stage43_am": {
                "report": str(STAGE43_AM),
                "verdict": am.get("stage43_am_gate", {}).get("verdict"),
                "statistically_confirmed": am.get("stage43_am_gate", {}).get(
                    "bounded_residual_statistically_confirmed"
                ),
            },
        },
        "frozen_metrics": {
            "all": al["best_safe_bounded_residual"]["all"],
            "t50": al["best_safe_bounded_residual"]["t50"],
            "t100": al["best_safe_bounded_residual"]["t100"],
            "hard_failure": al["best_safe_bounded_residual"]["hard_failure"],
            "easy": al["best_safe_bounded_residual"]["easy"],
            "switch_rate": al["best_safe_bounded_residual"]["switch_rate"],
            "all_delta_ci": am["bootstrap_delta_ci"]["metrics"]["all_delta_improvement"],
            "t50_delta_ci": am["bootstrap_delta_ci"]["metrics"]["t50_delta_improvement"],
            "hard_failure_delta_ci": am["bootstrap_delta_ci"]["metrics"]["hard_failure_delta_improvement"],
            "easy_degradation_ci": am["bootstrap_delta_ci"]["metrics"]["easy_degradation_bounded"],
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "thresholds_selected_on_test": False,
        },
        "claim_boundary": policy["claim_boundary"],
        "input_hash": _combined_hash([ak.STAGE43_M, checkpoint, STAGE43_AL, STAGE43_AM]),
    }
    payload["stage43_an_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["frozen_metrics"]
    gates = {
        "stage43_al_candidate_passed": payload["evidence_sources"]["stage43_al"]["deploy_bounded_residual"] is True,
        "stage43_am_statistically_confirmed": payload["evidence_sources"]["stage43_am"]["statistically_confirmed"]
        is True,
        "policy_hash_present": isinstance(payload["policy"]["policy_hash"], str)
        and len(payload["policy"]["policy_hash"]) == 64,
        "policy_artifact_written": Path(payload["policy_artifact"]).exists(),
        "checkpoint_not_tracked_by_git": payload["evidence_sources"]["stage43_m"]["checkpoint_tracked_by_git"]
        is False,
        "replay_diff_zero": payload["policy"]["evidence"]["stage43_m_replay_max_abs_diff"] == 0.0,
        "bootstrap_ci_supports_policy": metrics["all_delta_ci"]["low"] > 0.0
        and metrics["t50_delta_ci"]["low"] > 0.0
        and metrics["hard_failure_delta_ci"]["low"] > 0.0
        and metrics["easy_degradation_ci"]["high"] <= 0.02,
        "frozen_metrics_safe": metrics["easy"] <= 0.02 and metrics["t100"] >= -1e-8,
        "global_floor_not_removed": payload["policy"]["deployment_rule"]["global_floor_removal"] is False,
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True
        and payload["no_leakage"]["central_velocity_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_statistics_normalization"] is False
        and payload["no_leakage"]["thresholds_selected_on_test"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_an_bounded_residual_policy_frozen"
        if passed == total
        else "stage43_an_bounded_residual_policy_freeze_incomplete",
        "policy_frozen": passed == total,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_policy_md(policy: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    metrics = payload["frozen_metrics"]
    ci = metrics["t50_delta_ci"]
    write_md(
        POLICY_MD,
        [
            "# Frozen Stage43 Bounded Residual Policy",
            "",
            f"- policy name: `{policy['policy_name']}`",
            f"- policy hash: `{policy['policy_hash']}`",
            f"- policy type: `{policy['policy_type']}`",
            f"- formula: `{policy['deployment_rule']['formula']}`",
            f"- fallback: `{policy['deployment_rule']['fallback']}`",
            f"- h100 guard: `{policy['deployment_rule']['h100_guard']}`",
            "",
            "## Frozen Metrics",
            "",
            f"- all: `{_pct(metrics['all'])}`",
            f"- t50: `{_pct(metrics['t50'])}`",
            f"- t100 diagnostic: `{_pct(metrics['t100'])}`",
            f"- hard/failure: `{_pct(metrics['hard_failure'])}`",
            f"- easy degradation: `{_pct(metrics['easy'])}`",
            f"- t50 delta CI vs stored hard switch: `[{_pct(ci['low'])}, {_pct(ci['high'])}]`",
            "",
            "## Boundary",
            "",
            "- Dataset-local/raw-frame 2.5D only.",
            "- Future waypoints/endpoints are labels/eval only.",
            "- No metric/seconds claim, no Stage5C, no SMC.",
        ],
    )


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(POLICY_JSON, m._jsonable(payload["policy"]))
    _write_policy_md(payload["policy"], payload)
    payload["stage43_an_gate"] = _gate(payload)
    write_json(REPORT_JSON, m._jsonable(payload))
    gate = payload["stage43_an_gate"]
    metrics = payload["frozen_metrics"]
    lines = [
        "# Stage43-AN Bounded Residual Policy Freeze",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- policy frozen: `{gate['policy_frozen']}`",
        f"- policy hash: `{payload['policy']['policy_hash']}`",
        f"- checkpoint tracked by git: `{payload['evidence_sources']['stage43_m']['checkpoint_tracked_by_git']}`",
        "",
        "## Frozen Policy Metrics",
        "",
        f"- all: `{_pct(metrics['all'])}`",
        f"- t50: `{_pct(metrics['t50'])}`",
        f"- t100 diagnostic: `{_pct(metrics['t100'])}`",
        f"- hard/failure: `{_pct(metrics['hard_failure'])}`",
        f"- easy degradation: `{_pct(metrics['easy'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
        "",
        "## Hashes",
        "",
        f"- stage43_m_report_sha256: `{payload['hashes']['stage43_m_report_sha256']}`",
        f"- stage43_m_checkpoint_sha256: `{payload['hashes']['stage43_m_checkpoint_sha256']}`",
        f"- stage43_al_report_sha256: `{payload['hashes']['stage43_al_report_sha256']}`",
        f"- stage43_am_report_sha256: `{payload['hashes']['stage43_am_report_sha256']}`",
        "",
        "## Boundary",
        "",
        "- Global floor is not removed.",
        "- Dataset-local/raw-frame 2.5D only.",
        "- No metric/seconds claim, no Stage5C, no SMC.",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
    ]
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-AN Bounded Residual Policy Freeze Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- policy frozen: `{gate['policy_frozen']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_text_outputs(payload)


def _update_text_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_an_gate"]
    metrics = payload["frozen_metrics"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"policy_frozen = `{gate['policy_frozen']}`",
        f"policy_hash = `{payload['policy']['policy_hash']}`",
        f"frozen_all_t50_t100_hard_easy = `{_pct(metrics['all'])}` / `{_pct(metrics['t50'])}` / `{_pct(metrics['t100'])}` / `{_pct(metrics['hard_failure'])}` / `{_pct(metrics['easy'])}`",
        "",
        "Stage43-AN freezes the statistically confirmed Stage43 bounded-residual latent waypoint policy into a reproducible artifact with policy/config/checkpoint/report/row hashes. It remains floor-protected and h100-guarded; this is not global floor removal.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds claim; no Stage5C; no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_an_bounded_residual_policy_freeze"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "policy_frozen": gate["policy_frozen"],
        "policy_hash": payload["policy"]["policy_hash"],
        "policy_artifact": str(POLICY_JSON),
        "policy_readme": str(POLICY_MD),
        "metrics": metrics,
        "hashes": payload["hashes"],
        "global_floor_removable": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_an_bounded_residual_policy_freeze"
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
                        "stage": "Stage43-AN",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "policy_hash": payload["policy"]["policy_hash"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Freeze the Stage43 bounded residual policy artifact.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = _run(args)
    gate = result["stage43_an_gate"]
    print(f"Stage43-AN: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"policy_hash={result['policy']['policy_hash']}")
    return result


if __name__ == "__main__":
    main()
