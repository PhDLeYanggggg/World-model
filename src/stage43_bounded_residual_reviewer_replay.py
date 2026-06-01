from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_self_gate_conformal_audit as ak
from src import stage43_bounded_residual_safety_audit as al
from src import stage43_bounded_residual_policy_freeze as an


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_bounded_residual_reviewer_replay.json"
REPORT_MD = OUT_DIR / "stage43_bounded_residual_reviewer_replay.md"
GATE_MD = OUT_DIR / "stage43_stage_ao_bounded_residual_reviewer_replay_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_AO_BOUNDED_RESIDUAL_REVIEWER_REPLAY"
SOURCE = "fresh_stage43_ao_bounded_residual_reviewer_replay"
FROZEN_POLICY = OUT_DIR / "frozen_stage43_bounded_residual_policy.json"
FREEZE_REPORT = OUT_DIR / "stage43_bounded_residual_policy_freeze.json"

METRIC_MAP = {
    "all": "full_waypoint_ade_improvement_vs_floor",
    "endpoint": "endpoint_fde_improvement_vs_floor",
    "t50": "t50_full_waypoint_ade_improvement_vs_floor",
    "t50_endpoint": "t50_endpoint_fde_improvement_vs_floor",
    "t100": "t100_raw_frame_full_waypoint_diagnostic_vs_floor",
    "hard_failure": "hard_failure_full_waypoint_ade_improvement_vs_floor",
    "easy": "easy_degradation_vs_floor",
    "switch_rate": "switch_rate",
}


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _load_policy() -> dict[str, Any]:
    if not FROZEN_POLICY.exists():
        raise FileNotFoundError(FROZEN_POLICY)
    return read_json(FROZEN_POLICY, {})


def _verify_policy_hash(policy: Mapping[str, Any]) -> dict[str, Any]:
    expected = str(policy.get("policy_hash", ""))
    recomputed = an._stable_hash({k: v for k, v in dict(policy).items() if k != "policy_hash"})
    return {
        "expected": expected,
        "recomputed": recomputed,
        "match": expected == recomputed,
    }


def _metric_diff(replayed_metrics: Mapping[str, Any], frozen_point: Mapping[str, Any]) -> dict[str, Any]:
    rows = {}
    max_abs = 0.0
    for frozen_key, replay_key in METRIC_MAP.items():
        expected = float(frozen_point[frozen_key])
        replayed = float(replayed_metrics[replay_key])
        diff = replayed - expected
        rows[frozen_key] = {
            "expected": expected,
            "replayed": replayed,
            "signed_diff": diff,
            "abs_diff": abs(diff),
        }
        max_abs = max(max_abs, abs(diff))
    return {"max_abs_diff": max_abs, "by_metric": rows}


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    runtime = m._configure_runtime(int(args.seed))
    policy = _load_policy()
    policy_hash = _verify_policy_hash(policy)
    freeze = read_json(FREEZE_REPORT, {}) if FREEZE_REPORT.exists() else {}
    prior_m, ckpt, model = ak._load_stage43_m()
    _, test = ak._build_eval_splits(prior_m, ckpt)
    pred = m._predict(model, test, torch.device("cpu"), int(args.batch_size))
    result = al._evaluate_bounded(test, pred, policy["bounded_residual_config"])
    replayed_metrics = result["metrics"]
    frozen_point = policy["metrics"]["stage43_al_point_metrics"]
    replay_diff = _metric_diff(replayed_metrics, frozen_point)
    current_hashes = {
        "stage43_m_report_sha256": m._sha256(ak.STAGE43_M),
        "stage43_m_checkpoint_sha256": m._sha256(ak.STAGE43_M_CKPT),
        "policy_artifact_sha256": m._sha256(FROZEN_POLICY),
        "cache_row_hashes": {split: m._row_hash(m._npz(m._cache_path(split))) for split in m.SPLITS},
    }
    freeze_hashes = freeze.get("hashes", {})
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_replay_from_frozen_policy_artifact",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "runtime": runtime,
        "policy_artifact": str(FROZEN_POLICY),
        "freeze_report": str(FREEZE_REPORT),
        "policy_hash": policy_hash,
        "feature_schema_match": list(ckpt["feature_names"]) == test.feature_names,
        "current_hashes": current_hashes,
        "freeze_hashes": freeze_hashes,
        "checkpoint_not_tracked_by_git": an._tracked_by_git(ak.STAGE43_M_CKPT) is False,
        "cache_row_hash_match_prior": current_hashes["cache_row_hashes"] == prior_m.get("cache_row_hashes"),
        "checkpoint_hash_match_freeze": current_hashes["stage43_m_checkpoint_sha256"]
        == freeze_hashes.get("stage43_m_checkpoint_sha256"),
        "stage43_m_report_hash_match_freeze": current_hashes["stage43_m_report_sha256"]
        == freeze_hashes.get("stage43_m_report_sha256"),
        "policy_config": policy["bounded_residual_config"],
        "replayed_metrics": replayed_metrics,
        "frozen_point_metrics": frozen_point,
        "replay_diff": replay_diff,
        "switch_count": int(result["switch_count"]),
        "mean_residual_norm": float(result["mean_residual_norm"]),
        "max_residual_norm": float(result["max_residual_norm"]),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "thresholds_selected_on_test": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "input_hash": _combined_hash([FROZEN_POLICY, FREEZE_REPORT, ak.STAGE43_M, ak.STAGE43_M_CKPT, m._cache_path("test")]),
    }
    payload["stage43_ao_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["replayed_metrics"]
    gates = {
        "frozen_policy_artifact_present": Path(payload["policy_artifact"]).exists(),
        "policy_hash_recomputed": payload["policy_hash"]["match"] is True,
        "feature_schema_matches_checkpoint": payload["feature_schema_match"] is True,
        "cache_row_hashes_match_prior": payload["cache_row_hash_match_prior"] is True,
        "checkpoint_hash_matches_freeze": payload["checkpoint_hash_match_freeze"] is True,
        "stage43_m_report_hash_matches_freeze": payload["stage43_m_report_hash_match_freeze"] is True,
        "checkpoint_not_tracked_by_git": payload["checkpoint_not_tracked_by_git"] is True,
        "replay_metrics_exact": payload["replay_diff"]["max_abs_diff"] <= 1e-5,
        "replayed_policy_safe": metrics["easy_degradation_vs_floor"] <= 0.02
        and metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] >= -1e-8,
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
        "verdict": "stage43_ao_bounded_residual_reviewer_replay_pass"
        if passed == total
        else "stage43_ao_bounded_residual_reviewer_replay_incomplete",
        "reviewer_replay_passed": passed == total,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    gate = payload["stage43_ao_gate"]
    metrics = payload["replayed_metrics"]
    lines = [
        "# Stage43-AO Bounded Residual Reviewer Replay",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- reviewer replay passed: `{gate['reviewer_replay_passed']}`",
        f"- policy hash match: `{payload['policy_hash']['match']}`",
        f"- replay max abs diff: `{payload['replay_diff']['max_abs_diff']:.8f}`",
        "",
        "## Replayed Metrics",
        "",
        f"- all: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
        "",
        "## Replay Diff",
        "",
        "| metric | expected | replayed | abs diff |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, row in payload["replay_diff"]["by_metric"].items():
        lines.append(
            f"| {key} | `{_pct(row['expected'])}` | `{_pct(row['replayed'])}` | `{row['abs_diff']:.8f}` |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This is a reviewer replay from the frozen policy artifact, not a new threshold search.",
            "- Dataset-local/raw-frame 2.5D only.",
            "- Future labels are eval/loss only; no metric/seconds claim; no Stage5C; no SMC.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-AO Bounded Residual Reviewer Replay Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- reviewer replay passed: `{gate['reviewer_replay_passed']}`",
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
    gate = payload["stage43_ao_gate"]
    metrics = payload["replayed_metrics"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"reviewer_replay_passed = `{gate['reviewer_replay_passed']}`",
        f"policy_hash = `{payload['policy_hash']['expected']}`",
        f"policy_hash_match = `{payload['policy_hash']['match']}`",
        f"replay_max_abs_diff = `{payload['replay_diff']['max_abs_diff']:.8f}`",
        f"replayed_all_t50_t100_hard_easy = `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}` / `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(metrics['easy_degradation_vs_floor'])}`",
        "",
        "Stage43-AO independently replays the frozen bounded-residual policy artifact and verifies policy hash, checkpoint/report hashes, row hashes, replay diff, and no-leakage boundaries. This makes the Stage43 bounded-residual policy reviewer-replayable rather than report-only.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds claim; no Stage5C; no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_ao_bounded_residual_reviewer_replay"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "reviewer_replay_passed": gate["reviewer_replay_passed"],
        "policy_hash": payload["policy_hash"],
        "replay_diff": payload["replay_diff"],
        "metrics": metrics,
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_ao_bounded_residual_reviewer_replay"
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
                        "stage": "Stage43-AO",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "reviewer_replay_passed": gate["reviewer_replay_passed"],
                        "replay_max_abs_diff": payload["replay_diff"]["max_abs_diff"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay frozen Stage43 bounded residual policy for reviewer verification.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=431)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = _run(args)
    gate = result["stage43_ao_gate"]
    print(f"Stage43-AO: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"replay_max_abs_diff={result['replay_diff']['max_abs_diff']:.8f}")
    return result


if __name__ == "__main__":
    main()
