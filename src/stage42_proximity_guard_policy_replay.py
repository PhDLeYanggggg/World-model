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
CQ_JSON = OUT_DIR / "proximity_aware_composer_guard_stage42.json"
CR_JSON = OUT_DIR / "proximity_guard_ablation_stage42.json"
CS_JSON = OUT_DIR / "frozen_proximity_guard_composer_policy_stage42.json"
POLICY_JSON = OUT_DIR / "frozen_proximity_guard_composer_policy_stage42_policy.json"

REPORT_JSON = OUT_DIR / "frozen_proximity_guard_policy_replay_stage42.json"
REPORT_MD = OUT_DIR / "frozen_proximity_guard_policy_replay_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ct_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
RETRO_README = Path("README_M3W_GOAL_RETROSPECTIVE_CURRENT_ZH.md")
RESEARCH_STATE = Path("research_state.json")

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CT 是 Stage42-CS frozen policy artifact 的 replay / reproducibility verifier。",
    "CT 不重新选择阈值，不读取 test endpoint 构建 goals，不执行 Stage5C，不启用 SMC。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "replay 的目的不是新增模型分数，而是证明 frozen policy artifact 与 CQ/CR/CS 证据一致、可复核。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
]

FLOAT_KEYS = {
    "all_improvement",
    "t50_improvement",
    "t100_raw_frame_diagnostic_improvement",
    "hard_failure_improvement",
    "easy_degradation",
    "switch_rate",
    "near_collision_002_delta",
    "near_collision_005_delta",
    "p05_min_group_distance_delta",
    "jagged_rate_delta",
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _close(a: Any, b: Any, tol: float = 1e-12) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)
    return a == b


def _dict_close(a: Mapping[str, Any], b: Mapping[str, Any], keys: list[str], tol: float = 1e-12) -> dict[str, Any]:
    rows = {}
    for key in keys:
        rows[key] = {
            "artifact": a.get(key),
            "source": b.get(key),
            "match": _close(a.get(key), b.get(key), tol=tol),
        }
    return rows


def _policy_matches_cq(policy: Mapping[str, Any], cq: Mapping[str, Any]) -> dict[str, Any]:
    cq_metric = cq["test_eval"]["metric_vs_endpoint_ade"]
    policy_metric = policy["test_summary_vs_endpoint_linear_ade"]
    cq_safety = cq["test_joint_safety"]["composer_minus_endpoint"]
    policy_safety = policy["joint_safety_vs_endpoint_linear"]
    return {
        "selected_policy_match": policy.get("selected_policy") == cq.get("selected_policy"),
        "base_choices_match": policy.get("base_choices") == cq.get("selected_policy", {}).get("base_choices"),
        "metric_matches": _dict_close(
            policy_metric,
            cq_metric,
            [
                "all_improvement",
                "t50_improvement",
                "t100_raw_frame_diagnostic_improvement",
                "hard_failure_improvement",
                "easy_degradation",
                "switch_rate",
            ],
        ),
        "safety_matches": _dict_close(
            policy_safety,
            {
                "near_collision_002_delta": cq_safety.get("near_collision_rate_002_delta"),
                "near_collision_005_delta": cq_safety.get("near_collision_rate_005_delta"),
                "p05_min_group_distance_delta": cq_safety.get("p05_min_group_distance_delta"),
                "jagged_rate_delta": cq_safety.get("jagged_rate_delta"),
            },
            [
                "near_collision_002_delta",
                "near_collision_005_delta",
                "p05_min_group_distance_delta",
                "jagged_rate_delta",
            ],
        ),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    policy = payload["policy_artifact_payload"]
    metric = policy["test_summary_vs_endpoint_linear_ade"]
    safety = policy["joint_safety_vs_endpoint_linear"]
    no_leakage = policy["no_leakage"]
    claim = policy["claim_boundary"]
    replay = payload["replay_checks"]
    metric_matches = replay["cq_replay"]["metric_matches"]
    safety_matches = replay["cq_replay"]["safety_matches"]
    gates = {
        "policy_artifact_exists": payload["policy_artifact"]["exists"],
        "policy_hash_recomputed_matches_cs": replay["policy_hash_recomputed_matches_cs"],
        "policy_json_matches_cs_embedded_policy": replay["policy_json_matches_cs_embedded_policy"],
        "cq_gate_passed": payload["inputs"]["stage42_cq"]["stage42_cq_gate"]["passed"] == payload["inputs"]["stage42_cq"]["stage42_cq_gate"]["total"],
        "cr_gate_passed": payload["inputs"]["stage42_cr"]["stage42_cr_gate"]["passed"] == payload["inputs"]["stage42_cr"]["stage42_cr_gate"]["total"],
        "cs_gate_passed": payload["inputs"]["stage42_cs"]["stage42_cs_gate"]["passed"] == payload["inputs"]["stage42_cs"]["stage42_cs_gate"]["total"],
        "selected_policy_replays_cq": replay["cq_replay"]["selected_policy_match"],
        "base_choices_replay_cq": replay["cq_replay"]["base_choices_match"],
        "all_metric_replays_cq": metric_matches["all_improvement"]["match"],
        "t50_metric_replays_cq": metric_matches["t50_improvement"]["match"],
        "t100_metric_replays_cq": metric_matches["t100_raw_frame_diagnostic_improvement"]["match"],
        "hard_metric_replays_cq": metric_matches["hard_failure_improvement"]["match"],
        "easy_metric_replays_cq": metric_matches["easy_degradation"]["match"],
        "near_collision_replays_cq": safety_matches["near_collision_005_delta"]["match"],
        "jagged_rate_replays_cq": safety_matches["jagged_rate_delta"]["match"],
        "cr_recommendation_replayed": replay["cr_safety_policy_matches_artifact"],
        "all_positive": float(metric["all_improvement"]) > 0.0,
        "t50_positive": float(metric["t50_improvement"]) > 0.0,
        "t100_positive": float(metric["t100_raw_frame_diagnostic_improvement"]) > 0.0,
        "hard_positive": float(metric["hard_failure_improvement"]) > 0.0,
        "easy_under_2pct": float(metric["easy_degradation"]) <= 0.02,
        "near_collision_not_worse_than_endpoint": float(safety["near_collision_005_delta"]) <= 0.0,
        "no_future_endpoint_input": no_leakage.get("future_endpoint_input") is False,
        "no_future_waypoints_input": no_leakage.get("future_waypoints_input") is False,
        "no_central_velocity": no_leakage.get("central_velocity") is False,
        "no_test_endpoint_goals": no_leakage.get("test_endpoint_goals") is False,
        "no_test_threshold_tuning": no_leakage.get("test_threshold_tuning") is False,
        "metric_seconds_overclaim_blocked": claim.get("metric_or_seconds_claim") is False,
        "stage5c_not_executed": claim.get("stage5c_executed") is False,
        "smc_not_enabled": claim.get("smc_enabled") is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_ct_frozen_policy_replay_pass" if passed == total else "stage42_ct_frozen_policy_replay_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    policy = payload["policy_artifact_payload"]
    metric = policy["test_summary_vs_endpoint_linear_ade"]
    safety = policy["joint_safety_vs_endpoint_linear"]
    lines = [
        "## Stage42-CT Frozen Policy Replay / Reproducibility Verifier",
        "",
        "- source: `fresh_replay_from_frozen_policy_artifact`",
        f"- verdict: `{payload['stage42_ct_gate']['verdict']}`",
        f"- gates: `{payload['stage42_ct_gate']['passed']} / {payload['stage42_ct_gate']['total']}`",
        f"- replayed policy artifact: `{POLICY_JSON}`",
        f"- policy hash: `{payload['policy_hash_recomputed']}`",
        "- replay check: policy artifact matches Stage42-CS embedded policy and Stage42-CQ source metrics/safety.",
        f"- ADE vs endpoint-linear all/t50/t100 raw/hard: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(metric['easy_degradation'])}`",
        f"- near-collision@0.05 delta vs endpoint-linear: `{_pct(safety['near_collision_005_delta'])}`",
        "- Claim boundary unchanged: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, RETRO_README]:
        _replace_section(path, "STAGE42_CT_FROZEN_POLICY_REPLAY", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    policy = payload["policy_artifact_payload"]
    metric = policy["test_summary_vs_endpoint_linear_ade"]
    state["current_stage"] = "Stage42-CT frozen proximity-guard policy replay"
    state["current_verdict"] = payload["stage42_ct_gate"]["verdict"]
    state["stage42_ct_frozen_policy_replay"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_ct_gate"]["verdict"],
        "gates": f"{payload['stage42_ct_gate']['passed']}/{payload['stage42_ct_gate']['total']}",
        "policy_artifact": str(POLICY_JSON),
        "policy_hash_recomputed": payload["policy_hash_recomputed"],
        "replay_checks": payload["replay_checks"],
        "test_vs_endpoint_linear_ade": {
            "all_improvement": metric["all_improvement"],
            "t50_improvement": metric["t50_improvement"],
            "t100_raw_frame_diagnostic_improvement": metric["t100_raw_frame_diagnostic_improvement"],
            "hard_failure_improvement": metric["hard_failure_improvement"],
            "easy_degradation": metric["easy_degradation"],
        },
        "claim_boundary": policy["claim_boundary"],
        "conclusion": "Stage42-CT replays the frozen Stage42-CS proximity-guard composer policy from artifact and verifies hash, CQ metric/safety consistency, CR recommendation consistency, no-leakage flags, and claim boundaries. It is reproducibility/deployment evidence, not new model training.",
        "verification": {
            "runner": ".venv-pytorch/bin/python run_stage42_replay_proximity_guard_policy.py",
            "targeted_pytest": ".venv-pytorch/bin/python -m pytest tests/test_stage42_proximity_guard_policy_replay.py",
        },
    }
    summary = state.setdefault("latest_user_facing_goal_summary", {})
    summary["source"] = "cached_verified_synthesis_for_user_question_refreshed_after_stage42_ct"
    included = summary.setdefault("latest_fresh_evidence_included", [])
    note = "Stage42-CT frozen policy replay: policy artifact hash and metrics replay against CQ/CR/CS source evidence"
    if note not in included:
        included.append(note)
    write_json(RESEARCH_STATE, state)


def _write_md(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_ct_gate"]
    policy = payload["policy_artifact_payload"]
    metric = policy["test_summary_vs_endpoint_linear_ade"]
    safety = policy["joint_safety_vs_endpoint_linear"]
    replay = payload["replay_checks"]
    lines = [
        "# Stage42-CT Frozen Policy Replay / Reproducibility Verifier",
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
        f"- policy hash matches CS: `{replay['policy_hash_recomputed_matches_cs']}`",
        f"- policy JSON matches CS embedded policy: `{replay['policy_json_matches_cs_embedded_policy']}`",
        f"- selected policy matches CQ: `{replay['cq_replay']['selected_policy_match']}`",
        f"- base choices match CQ: `{replay['cq_replay']['base_choices_match']}`",
        f"- CR safety recommendation matches artifact: `{replay['cr_safety_policy_matches_artifact']}`",
        "",
        "## Replayed Metrics Vs Endpoint-Linear ADE",
        "",
        f"- all: `{_pct(metric['all_improvement'])}`",
        f"- t50: `{_pct(metric['t50_improvement'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}`",
        f"- hard/failure: `{_pct(metric['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(metric['easy_degradation'])}`",
        "",
        "## Replayed Joint Safety Vs Endpoint-Linear",
        "",
        f"- near_collision@0.02 delta: `{_pct(safety['near_collision_002_delta'])}`",
        f"- near_collision@0.05 delta: `{_pct(safety['near_collision_005_delta'])}`",
        f"- p05 min group distance delta: `{_pct(safety['p05_min_group_distance_delta'])}`",
        f"- jagged-rate delta: `{_pct(safety['jagged_rate_delta'])}`",
        "",
        "## Interpretation",
        "",
        "- Stage42-CT proves the frozen policy artifact is reproducible from the stored CQ/CR/CS evidence.",
        "- It does not add a new score and does not tune on test; it prevents the deployable policy from being a loose narrative claim.",
        "- The frozen artifact remains protected dataset-local/raw-frame 2.5D evidence, not metric/seconds-level, true 3D, Stage5C, or SMC evidence.",
    ]
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage42-CT Gate",
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


def run_stage42_replay_proximity_guard_policy() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cq = read_json(CQ_JSON, {})
    cr = read_json(CR_JSON, {})
    cs = read_json(CS_JSON, {})
    policy = read_json(POLICY_JSON, {})
    if not all([cq, cr, cs, policy]):
        missing = [str(p) for p in [CQ_JSON, CR_JSON, CS_JSON, POLICY_JSON] if not p.exists()]
        raise FileNotFoundError(f"Missing replay inputs: {missing}")
    policy_hash = _combined_hash([POLICY_JSON])
    replay_checks = {
        "policy_hash_recomputed_matches_cs": policy_hash == cs.get("policy_hash"),
        "policy_json_matches_cs_embedded_policy": policy == cs.get("frozen_policy"),
        "cq_replay": _policy_matches_cq(policy, cq),
        "cr_safety_policy_matches_artifact": (
            cr.get("deployment_recommendation", {}).get("safety_sensitive_policy") == "proximity_guard"
            and policy.get("deployment_role") == "safety_sensitive_deployable_composer_variant"
        ),
    }
    payload: dict[str, Any] = {
        "source": "fresh_replay_from_frozen_policy_artifact",
        "stage": "Stage42-CT frozen policy replay / reproducibility verifier",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([CQ_JSON, CR_JSON, CS_JSON, POLICY_JSON]),
        "inputs": {
            "stage42_cq": {"path": str(CQ_JSON), "stage42_cq_gate": cq.get("stage42_cq_gate", {})},
            "stage42_cr": {"path": str(CR_JSON), "stage42_cr_gate": cr.get("stage42_cr_gate", {})},
            "stage42_cs": {"path": str(CS_JSON), "stage42_cs_gate": cs.get("stage42_cs_gate", {})},
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
    payload["stage42_ct_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    _write_md(payload)
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_replay_proximity_guard_policy()
    gate = result["stage42_ct_gate"]
    print(f"Stage42-CT frozen policy replay: {gate['verdict']} ({gate['passed']}/{gate['total']})")
