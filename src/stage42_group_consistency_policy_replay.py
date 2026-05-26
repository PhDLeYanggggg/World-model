from __future__ import annotations

import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section, _pct


OUT_DIR = Path("outputs/stage42_long_research")

DI_JSON = OUT_DIR / "group_consistency_full_waypoint_repair_stage42.json"
DJ_JSON = OUT_DIR / "frozen_group_consistency_full_waypoint_policy_stage42.json"
POLICY_JSON = OUT_DIR / "frozen_group_consistency_full_waypoint_policy_stage42_policy.json"

REPORT_JSON = OUT_DIR / "group_consistency_policy_replay_stage42.json"
REPORT_MD = OUT_DIR / "group_consistency_policy_replay_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_dk_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_README = Path("README_M3W_GOAL_SUMMARY_ZH.md")
CURRENT_RETROSPECTIVE = Path("README_M3W_CURRENT_FULL_RETROSPECTIVE_ZH.md")
RESEARCH_STATE = Path("research_state.json")

PAPER_FILES = [
    OUT_DIR / "method_draft_stage42.md",
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "reproducibility_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DK 是 Stage42-DJ frozen group-consistency full-waypoint policy 的 artifact replay / reproducibility verifier。",
    "DK 不重新训练、不重新选择阈值、不使用 test metrics 调参。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "group-consistency replay 只验证 frozen artifact 与 Stage42-DI/DJ 源证据一致。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


METRIC_KEYS = [
    "all_improvement",
    "t50_improvement",
    "t100_raw_frame_diagnostic_improvement",
    "hard_failure_improvement",
    "easy_degradation",
    "switch_rate",
]

SAFETY_KEYS = [
    "base_near_005",
    "final_near_005",
    "floor_near_005",
    "base_p05_min_distance",
    "final_p05_min_distance",
    "floor_p05_min_distance",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _close(a: Any, b: Any, tol: float = 1e-12) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)
    return a == b


def _dict_close(a: Mapping[str, Any], b: Mapping[str, Any], keys: list[str], tol: float = 1e-12) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "artifact": a.get(key),
            "source": b.get(key),
            "match": _close(a.get(key), b.get(key), tol=tol),
        }
        for key in keys
    }


def _policy_matches_di(policy: Mapping[str, Any], di: Mapping[str, Any]) -> dict[str, Any]:
    selected = di["repair"]["selected"]
    metric = di["repair"]["test"]["metric_vs_floor"]
    safety = di["repair"]["test"]["diagnostics"]
    return {
        "repair_rule_matches_selected_candidate": {
            "mode": policy.get("repair_rule", {}).get("type") == selected.get("candidate", {}).get("mode"),
            "min_sep": _close(policy.get("repair_rule", {}).get("min_sep"), selected.get("candidate", {}).get("min_sep")),
            "margin": _close(policy.get("repair_rule", {}).get("margin"), selected.get("candidate", {}).get("margin")),
            "strength": _close(policy.get("repair_rule", {}).get("strength"), selected.get("candidate", {}).get("strength")),
        },
        "validation_selection_replays_di": {
            "val_score": _close(policy.get("validation_selection", {}).get("val_score"), selected.get("val_score")),
            "val_metric": _dict_close(
                policy.get("validation_selection", {}).get("val_metric", {}),
                selected.get("val_metric", {}),
                METRIC_KEYS,
            ),
        },
        "metric_matches": _dict_close(
            policy.get("test_summary_vs_train_horizon_causal_floor", {}),
            metric,
            METRIC_KEYS,
        ),
        "safety_matches": _dict_close(
            policy.get("test_group_safety", {}),
            {
                "base_near_005": safety.get("base_near_005"),
                "final_near_005": safety.get("final_near_005"),
                "floor_near_005": safety.get("floor_near_005"),
                "base_p05_min_distance": safety.get("base_p05_min_distance"),
                "final_p05_min_distance": safety.get("final_p05_min_distance"),
                "floor_p05_min_distance": safety.get("floor_p05_min_distance"),
            },
            SAFETY_KEYS,
        ),
        "bootstrap_matches": policy.get("bootstrap") == di["repair"]["test"].get("bootstrap", {}),
        "no_leakage_matches": policy.get("no_leakage") == di.get("no_leakage", {}),
        "claim_boundary_matches": policy.get("claim_boundary") == di.get("claim_boundary", {}),
    }


def _all_match(rows: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(bool(row.get("match")) for row in rows.values())


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    policy = payload["policy_artifact_payload"]
    metric = policy["test_summary_vs_train_horizon_causal_floor"]
    safety = policy["test_group_safety"]
    no_leakage = policy["no_leakage"]
    claim = policy["claim_boundary"]
    replay = payload["replay_checks"]
    di_replay = replay["di_replay"]
    repair_matches = di_replay["repair_rule_matches_selected_candidate"]
    val_metric_matches = di_replay["validation_selection_replays_di"]["val_metric"]
    metric_matches = di_replay["metric_matches"]
    safety_matches = di_replay["safety_matches"]
    gates = {
        "policy_artifact_exists": payload["policy_artifact"]["exists"],
        "policy_hash_recomputed_matches_dj": replay["policy_hash_recomputed_matches_dj"],
        "policy_json_matches_dj_embedded_policy": replay["policy_json_matches_dj_embedded_policy"],
        "di_gate_passed": payload["inputs"]["stage42_di"]["stage42_di_gate"].get("passed")
        == payload["inputs"]["stage42_di"]["stage42_di_gate"].get("total"),
        "dj_gate_passed": payload["inputs"]["stage42_dj"]["stage42_dj_gate"].get("passed")
        == payload["inputs"]["stage42_dj"]["stage42_dj_gate"].get("total"),
        "repair_rule_replays_di_selected": all(repair_matches.values()),
        "validation_score_replays_di": di_replay["validation_selection_replays_di"]["val_score"],
        "validation_metric_replays_di": _all_match(val_metric_matches),
        "all_metric_replays_di": metric_matches["all_improvement"]["match"],
        "t50_metric_replays_di": metric_matches["t50_improvement"]["match"],
        "t100_metric_replays_di": metric_matches["t100_raw_frame_diagnostic_improvement"]["match"],
        "hard_metric_replays_di": metric_matches["hard_failure_improvement"]["match"],
        "easy_metric_replays_di": metric_matches["easy_degradation"]["match"],
        "base_near_replays_di": safety_matches["base_near_005"]["match"],
        "final_near_replays_di": safety_matches["final_near_005"]["match"],
        "floor_near_replays_di": safety_matches["floor_near_005"]["match"],
        "bootstrap_replays_di": di_replay["bootstrap_matches"],
        "no_leakage_replays_di": di_replay["no_leakage_matches"],
        "claim_boundary_replays_di": di_replay["claim_boundary_matches"],
        "all_positive": float(metric["all_improvement"]) > 0.0,
        "t50_positive": float(metric["t50_improvement"]) > 0.0,
        "t100_raw_positive": float(metric["t100_raw_frame_diagnostic_improvement"]) > 0.0,
        "hard_positive": float(metric["hard_failure_improvement"]) > 0.0,
        "easy_under_2pct": float(metric["easy_degradation"]) <= 0.02,
        "final_near_collision_reduced_vs_base": float(safety["final_near_005"]) <= float(safety["base_near_005"]),
        "no_future_endpoint_input": no_leakage.get("future_endpoint_input") is False,
        "no_future_waypoint_input": no_leakage.get("future_waypoint_input") is False,
        "no_central_velocity": no_leakage.get("central_velocity") is False,
        "no_test_endpoint_goals": no_leakage.get("test_endpoint_goals") is False,
        "no_test_threshold_tuning": no_leakage.get("test_threshold_tuning") is False,
        "metric_seconds_overclaim_blocked": claim.get("metric_or_seconds_claim") is False,
        "stage5c_not_executed": claim.get("stage5c_executed") is False,
        "smc_not_enabled": claim.get("smc_enabled") is False,
        "paper_files_refreshed": all(row["contains_stage42_dk"] for row in payload.get("paper_file_status", []) if row["exists"]),
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_dk_group_consistency_policy_replay_pass" if passed == total else "stage42_dk_group_consistency_policy_replay_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    policy = payload["policy_artifact_payload"]
    metric = policy["test_summary_vs_train_horizon_causal_floor"]
    safety = policy["test_group_safety"]
    lines = [
        "## Stage42-DK Group-Consistency Policy Replay",
        "",
        "- source: `fresh_replay_from_frozen_group_consistency_policy_artifact`",
        f"- verdict: `{payload['stage42_dk_gate']['verdict']}`",
        f"- gates: `{payload['stage42_dk_gate']['passed']} / {payload['stage42_dk_gate']['total']}`",
        f"- replayed policy artifact: `{POLICY_JSON}`",
        f"- policy hash: `{payload['policy_hash_recomputed']}`",
        "- replay check: policy artifact matches Stage42-DJ embedded policy and Stage42-DI selected repair / metrics / safety.",
        f"- ADE vs train-horizon causal floor all/t50/t100 raw/hard: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(metric['easy_degradation'])}`",
        f"- near@0.05 base/final/floor: `{_pct(safety['base_near_005'])}` / `{_pct(safety['final_near_005'])}` / `{_pct(safety['floor_near_005'])}`",
        "- Claim boundary unchanged: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, GOAL_README, CURRENT_RETROSPECTIVE]:
        _replace_section(path, "STAGE42_DK_GROUP_CONSISTENCY_POLICY_REPLAY", lines)


def _refresh_paper_files(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    policy = payload["policy_artifact_payload"]
    metric = policy["test_summary_vs_train_horizon_causal_floor"]
    safety = policy["test_group_safety"]
    lines = [
        "## Stage42-DK Group-Consistency Policy Replay",
        "",
        "- source: `fresh_replay_from_frozen_group_consistency_policy_artifact`",
        "- role: replay the Stage42-DJ frozen group-consistency full-waypoint policy artifact against Stage42-DI/DJ source evidence.",
        "- replay performs no retraining, no threshold reselection, and no test tuning.",
        f"- policy artifact: `{POLICY_JSON}`",
        f"- policy hash: `{payload['policy_hash_recomputed']}`",
        f"- gate: `{payload['stage42_dk_gate']['passed']} / {payload['stage42_dk_gate']['total']}`; verdict `{payload['stage42_dk_gate']['verdict']}`.",
        f"- replayed ADE vs train-horizon causal floor: all `{_pct(metric['all_improvement'])}`, t50 `{_pct(metric['t50_improvement'])}`, t100 raw `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}`, hard `{_pct(metric['hard_failure_improvement'])}`, easy `{_pct(metric['easy_degradation'])}`.",
        f"- replayed near@0.05 base/final/floor: `{_pct(safety['base_near_005'])}` / `{_pct(safety['final_near_005'])}` / `{_pct(safety['floor_near_005'])}`.",
        "- claim boundary: still protected dataset-local/raw-frame 2.5D; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.",
    ]
    status = []
    for path in PAPER_FILES:
        if path.exists():
            _replace_section(path, "STAGE42_DK_GROUP_CONSISTENCY_POLICY_REPLAY", lines)
            text = path.read_text(encoding="utf-8")
            status.append(
                {
                    "path": str(path),
                    "exists": True,
                    "contains_stage42_dk": "Stage42-DK Group-Consistency Policy Replay" in text,
                    "contains_claim_boundary": "no true 3D" in text and "no Stage5C" in text,
                }
            )
        else:
            status.append(
                {
                    "path": str(path),
                    "exists": False,
                    "contains_stage42_dk": False,
                    "contains_claim_boundary": False,
                }
            )
    return status


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    policy = payload["policy_artifact_payload"]
    metric = policy["test_summary_vs_train_horizon_causal_floor"]
    state["current_stage"] = "Stage42-DK group-consistency policy replay"
    state["current_verdict"] = payload["stage42_dk_gate"]["verdict"]
    state["stage42_dk_group_consistency_policy_replay"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "policy_artifact": str(POLICY_JSON),
        "policy_hash_recomputed": payload["policy_hash_recomputed"],
        "verdict": payload["stage42_dk_gate"]["verdict"],
        "gates": f"{payload['stage42_dk_gate']['passed']}/{payload['stage42_dk_gate']['total']}",
        "replay_checks": payload["replay_checks"],
        "test_metric_vs_train_horizon_causal_floor": {
            "all_improvement": metric["all_improvement"],
            "t50_improvement": metric["t50_improvement"],
            "t100_raw_frame_diagnostic_improvement": metric["t100_raw_frame_diagnostic_improvement"],
            "hard_failure_improvement": metric["hard_failure_improvement"],
            "easy_degradation": metric["easy_degradation"],
        },
        "claim_boundary": policy["claim_boundary"],
        "conclusion": "Stage42-DK replays the frozen Stage42-DJ group-consistency policy artifact against Stage42-DI/DJ source evidence. It is reproducibility/deployment evidence, not new model training.",
        "verification": {
            "runner": ".venv-pytorch/bin/python run_stage42_replay_group_consistency_policy.py",
            "targeted_pytest": ".venv-pytorch/bin/python -m pytest tests/test_stage42_group_consistency_policy_replay.py",
        },
    }
    write_json(RESEARCH_STATE, state)


def _write_md(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_dk_gate"]
    policy = payload["policy_artifact_payload"]
    metric = policy["test_summary_vs_train_horizon_causal_floor"]
    safety = policy["test_group_safety"]
    replay = payload["replay_checks"]
    lines = [
        "# Stage42-DK Group-Consistency Policy Replay",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- policy_hash_recomputed: `{payload['policy_hash_recomputed']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Replay Checks",
        "",
        f"- policy hash matches Stage42-DJ: `{replay['policy_hash_recomputed_matches_dj']}`",
        f"- policy JSON matches Stage42-DJ embedded policy: `{replay['policy_json_matches_dj_embedded_policy']}`",
        f"- repair rule matches Stage42-DI selected candidate: `{all(replay['di_replay']['repair_rule_matches_selected_candidate'].values())}`",
        f"- validation score matches Stage42-DI: `{replay['di_replay']['validation_selection_replays_di']['val_score']}`",
        f"- bootstrap matches Stage42-DI: `{replay['di_replay']['bootstrap_matches']}`",
        f"- no-leakage flags match Stage42-DI: `{replay['di_replay']['no_leakage_matches']}`",
        "",
        "## Replayed Metrics Vs Train-Horizon Causal Floor",
        "",
        f"- all: `{_pct(metric['all_improvement'])}`",
        f"- t50: `{_pct(metric['t50_improvement'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}`",
        f"- hard/failure: `{_pct(metric['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(metric['easy_degradation'])}`",
        "",
        "## Replayed Group Safety",
        "",
        f"- base near@0.05: `{_pct(safety['base_near_005'])}`",
        f"- final near@0.05: `{_pct(safety['final_near_005'])}`",
        f"- floor near@0.05: `{_pct(safety['floor_near_005'])}`",
        f"- base p05 min distance: `{safety['base_p05_min_distance']}`",
        f"- final p05 min distance: `{safety['final_p05_min_distance']}`",
        "",
        "## Interpretation",
        "",
        "- Stage42-DK proves the frozen Stage42-DJ group-consistency policy artifact replays the Stage42-DI source evidence exactly.",
        "- It does not retrain, does not retune, and does not add a fresh score; it hardens the paper/deployment reproducibility chain.",
        "- The policy remains protected dataset-local/raw-frame 2.5D evidence, not true 3D, not foundation-scale, not metric/seconds-level, not Stage5C, and not SMC.",
    ]
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage42-DK Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | passed |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        gate_lines.append(f"| `{name}` | `{ok}` |")
    write_md(GATE_MD, gate_lines)


def run_stage42_replay_group_consistency_policy() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    di = read_json(DI_JSON, {})
    dj = read_json(DJ_JSON, {})
    policy = read_json(POLICY_JSON, {})
    if not all([di, dj, policy]):
        missing = [str(path) for path in [DI_JSON, DJ_JSON, POLICY_JSON] if not path.exists()]
        raise FileNotFoundError(f"Missing Stage42-DK replay inputs: {missing}")
    policy_hash = _combined_hash([POLICY_JSON])
    replay_checks = {
        "policy_hash_recomputed_matches_dj": policy_hash == dj.get("policy_hash"),
        "policy_json_matches_dj_embedded_policy": policy == dj.get("frozen_policy"),
        "di_replay": _policy_matches_di(policy, di),
    }
    payload: dict[str, Any] = {
        "source": "fresh_replay_from_frozen_group_consistency_policy_artifact",
        "stage": "Stage42-DK group-consistency policy replay / reproducibility verifier",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([DI_JSON, DJ_JSON, POLICY_JSON]),
        "inputs": {
            "stage42_di": {"path": str(DI_JSON), "stage42_di_gate": di.get("stage42_di_gate", {})},
            "stage42_dj": {"path": str(DJ_JSON), "stage42_dj_gate": dj.get("stage42_dj_gate", {})},
        },
        "policy_artifact": {
            "path": str(POLICY_JSON),
            "exists": POLICY_JSON.exists(),
            "size_bytes": POLICY_JSON.stat().st_size if POLICY_JSON.exists() else 0,
        },
        "policy_artifact_payload": policy,
        "policy_hash_recomputed": policy_hash,
        "replay_checks": replay_checks,
    }
    payload["paper_file_status"] = []
    payload["stage42_dk_gate"] = _gate(payload)
    payload["paper_file_status"] = _refresh_paper_files(payload)
    payload["stage42_dk_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    _write_md(payload)
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_replay_group_consistency_policy()
    gate = result["stage42_dk_gate"]
    print(f"Stage42-DK group-consistency policy replay: {gate['verdict']} ({gate['passed']}/{gate['total']})")
