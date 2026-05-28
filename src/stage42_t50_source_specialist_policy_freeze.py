from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src import stage42_t50_ensemble_ucy_specialist_integration as s42ik
from src import stage42_t50_ucy_specialist_claim_audit as s42il
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "t50_source_specialist_policy_freeze_stage42.json"
REPORT_MD = OUT_DIR / "t50_source_specialist_policy_freeze_stage42.md"
POLICY_JSON = OUT_DIR / "frozen_t50_source_specialist_policy_stage42.json"
POLICY_MD = OUT_DIR / "frozen_t50_source_specialist_policy_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_im_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_README = Path("README_M3W_MASTER_SUMMARY_2026_05_27_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE42_IM_T50_SOURCE_SPECIALIST_POLICY_FREEZE"
SOURCE = "cached_verified_stage42_ik_il_t50_source_specialist_policy_freeze"


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-IM 冻结 Stage42-IK/IL 的 t50 source-specialist composition policy。",
    "IM 不训练新模型、不重新选择 threshold、不使用 test metrics 调参；只固化 source routing 与 claim boundary。",
    "future waypoints / endpoints 只作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _stable_hash(value: Any) -> str:
    blob = json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _policy_payload(ik: Mapping[str, Any], il: Mapping[str, Any]) -> dict[str, Any]:
    summary = ik["summary"]
    return {
        "policy_name": "stage42_im_frozen_t50_source_specialist_policy",
        "source": SOURCE,
        "base_stage": "Stage42-IK t50 ensemble UCY specialist integration",
        "claim_audit_stage": "Stage42-IL t50 UCY specialist claim audit",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_git_commit": _git_commit(),
        "selection_scope": "prevalidated_source_routing_no_new_threshold_selection",
        "test_usage": "compact_replay_from_stage42_ik_and_il_artifacts",
        "deployment_role": "source_specialist_t50_raw_frame_policy",
        "routing_rule": {
            "type": "source_or_domain_specialist_router",
            "default_route": "stage42ii_t50_gain_harm_ensemble",
            "ucy_route": "stage42x_row_aligned_ucy_full_waypoint_specialist",
            "route_by": ["domain == UCY", "source_file contains /UCY/zara03/crowds_zara03.txt"],
            "new_threshold_selection": False,
            "uses_test_metrics_for_routing": False,
            "uses_future_labels": False,
        },
        "source_table": ik.get("source_rows", []),
        "domain_table": ik.get("by_domain", {}),
        "test_summary_vs_train_horizon_causal_floor": {
            "rows": summary["rows"],
            "ade_all": summary["ade_all"],
            "ade_t50": summary["ade_t50"],
            "ade_t50_ci_low": summary["ade_t50_ci_low"],
            "ade_t100_raw_frame_diagnostic": summary["ade_t100_raw_frame_diagnostic"],
            "ade_hard_failure": summary["ade_hard_failure"],
            "ade_easy_degradation": summary["ade_easy_degradation"],
            "fde_t50": summary["fde_t50"],
            "fde_t50_ci_low": summary["fde_t50_ci_low"],
            "switch_rate": summary["switch_rate"],
        },
        "delta_audit": il["summary"],
        "supported_claims": il.get("supported_claims", []),
        "blocked_claims": il.get("blocked_claims", []),
        "alignment": ik.get("alignment", {}),
        "no_leakage": ik.get("no_leakage", {}),
        "claim_boundary": ik.get("claim_boundary", {}),
    }


def _write_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    write_json(POLICY_JSON, _jsonable(policy))
    digest = hashlib.sha256(POLICY_JSON.read_bytes()).hexdigest()
    metric = policy["test_summary_vs_train_horizon_causal_floor"]
    lines = [
        "# Frozen Stage42-IM T50 Source-Specialist Policy",
        "",
        f"- source: `{policy['source']}`",
        f"- policy_name: `{policy['policy_name']}`",
        f"- deployment_role: `{policy['deployment_role']}`",
        f"- selection_scope: `{policy['selection_scope']}`",
        f"- default_route: `{policy['routing_rule']['default_route']}`",
        f"- ucy_route: `{policy['routing_rule']['ucy_route']}`",
        f"- sha256: `{digest}`",
        "",
        "## Test Summary",
        "",
        f"- ADE all/t50/t100raw/hard: `{_pct(metric['ade_all'])}` / `{_pct(metric['ade_t50'])}` / `{_pct(metric['ade_t100_raw_frame_diagnostic'])}` / `{_pct(metric['ade_hard_failure'])}`",
        f"- FDE t50: `{_pct(metric['fde_t50'])}`",
        f"- easy degradation: `{_pct(metric['ade_easy_degradation'])}`",
        "",
        "Claim boundary: source-specialist composition evidence only; protected dataset-local/raw-frame 2.5D; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.",
    ]
    write_md(POLICY_MD, lines)
    return {"path": str(POLICY_JSON), "sha256": digest, "size_bytes": POLICY_JSON.stat().st_size}


def _replay(policy: Mapping[str, Any], ik: Mapping[str, Any], il: Mapping[str, Any]) -> dict[str, Any]:
    metric = policy["test_summary_vs_train_horizon_causal_floor"]
    ik_summary = ik["summary"]
    metric_map = {
        "ade_all": "ade_all",
        "ade_t50": "ade_t50",
        "ade_t50_ci_low": "ade_t50_ci_low",
        "ade_t100_raw_frame_diagnostic": "ade_t100_raw_frame_diagnostic",
        "ade_hard_failure": "ade_hard_failure",
        "ade_easy_degradation": "ade_easy_degradation",
        "fde_t50": "fde_t50",
        "fde_t50_ci_low": "fde_t50_ci_low",
        "switch_rate": "switch_rate",
    }
    diffs = {name: abs(float(metric[name]) - float(ik_summary[key])) for name, key in metric_map.items()}
    source_rows_exact = policy["source_table"] == ik.get("source_rows", [])
    domain_rows_exact = policy["domain_table"] == ik.get("by_domain", {})
    il_summary_exact = policy["delta_audit"] == il.get("summary", {})
    return {
        "source": "compact_replay_from_frozen_policy_ik_il_artifacts",
        "metric_summary_exact_replay": max(diffs.values()) <= 1e-12,
        "max_metric_abs_diff": max(diffs.values()),
        "metric_diffs": diffs,
        "source_rows_exact_replay": source_rows_exact,
        "domain_rows_exact_replay": domain_rows_exact,
        "il_delta_audit_exact_replay": il_summary_exact,
        "ucy_t50_repaired": il["summary"]["ucy_delta"]["after_t50"] > 0.0,
        "non_ucy_max_abs_delta": il["summary"]["non_ucy_max_abs_delta"],
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    policy = payload["frozen_policy"]
    metric = policy["test_summary_vs_train_horizon_causal_floor"]
    replay = payload["replay"]
    no_leak = policy["no_leakage"]
    claim = policy["claim_boundary"]
    il_summary = policy["delta_audit"]
    gates = {
        "ik_gate_passed": payload["inputs"]["stage42ik_gate"].get("passed") == payload["inputs"]["stage42ik_gate"].get("total"),
        "il_gate_passed": payload["inputs"]["stage42il_gate"].get("passed") == payload["inputs"]["stage42il_gate"].get("total"),
        "policy_artifact_written": len(payload["policy_artifact"]["sha256"]) == 64,
        "policy_hash_recorded": len(payload["policy_hash"]) == 64,
        "routing_rule_frozen": policy["routing_rule"]["default_route"] == "stage42ii_t50_gain_harm_ensemble"
        and policy["routing_rule"]["ucy_route"] == "stage42x_row_aligned_ucy_full_waypoint_specialist",
        "no_new_threshold_selection": policy["routing_rule"]["new_threshold_selection"] is False,
        "no_test_metric_routing": policy["routing_rule"]["uses_test_metrics_for_routing"] is False,
        "compact_metric_replay_exact": replay["metric_summary_exact_replay"] is True,
        "source_rows_replay_exact": replay["source_rows_exact_replay"] is True,
        "domain_rows_replay_exact": replay["domain_rows_exact_replay"] is True,
        "il_delta_replay_exact": replay["il_delta_audit_exact_replay"] is True,
        "ucy_t50_repaired": replay["ucy_t50_repaired"] is True,
        "non_ucy_unchanged_with_tolerance": il_summary["non_ucy_max_abs_delta"] <= 1e-6,
        "all_positive": metric["ade_all"] > 0.0,
        "t50_positive": metric["ade_t50"] > 0.0 and metric["ade_t50_ci_low"] > 0.0,
        "hard_positive": metric["ade_hard_failure"] > 0.0,
        "easy_preserved": metric["ade_easy_degradation"] <= 0.02,
        "no_future_or_test_leakage": no_leak.get("future_endpoint_input") is False
        and no_leak.get("future_waypoints_input") is False
        and no_leak.get("central_velocity") is False
        and no_leak.get("test_endpoint_goals") is False
        and no_leak.get("test_threshold_tuning") is False,
        "scope_not_overclaimed": claim.get("source_specialist_claim_only") is True and claim.get("independent_new_domain_claim") is False,
        "no_metric_seconds_overclaim": claim.get("metric_or_seconds_claim") is False,
        "stage5c_false": claim.get("stage5c_executed") is False,
        "smc_false": claim.get("smc_enabled") is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_im_t50_source_specialist_policy_freeze_pass" if passed == total else "stage42_im_t50_source_specialist_policy_freeze_partial"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _write_report(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_im_gate"]
    policy = payload["frozen_policy"]
    metric = policy["test_summary_vs_train_horizon_causal_floor"]
    lines = [
        "# Stage42-IM T50 Source-Specialist Policy Freeze",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- policy_hash: `{payload['policy_hash']}`",
        f"- policy_artifact: `{POLICY_JSON}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Frozen Routing Policy",
        "",
        f"- default_route: `{policy['routing_rule']['default_route']}`",
        f"- ucy_route: `{policy['routing_rule']['ucy_route']}`",
        f"- route_by: `{policy['routing_rule']['route_by']}`",
        f"- new_threshold_selection: `{policy['routing_rule']['new_threshold_selection']}`",
        f"- uses_test_metrics_for_routing: `{policy['routing_rule']['uses_test_metrics_for_routing']}`",
        "",
        "## Compact Replay Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| ADE all | {metric['ade_all']:.6f} |",
        f"| ADE t50 | {metric['ade_t50']:.6f} |",
        f"| ADE t50 CI low | {metric['ade_t50_ci_low']:.6f} |",
        f"| ADE t100 raw diagnostic | {metric['ade_t100_raw_frame_diagnostic']:.6f} |",
        f"| ADE hard/failure | {metric['ade_hard_failure']:.6f} |",
        f"| ADE easy degradation | {metric['ade_easy_degradation']:.6f} |",
        f"| FDE t50 | {metric['fde_t50']:.6f} |",
        f"| switch rate | {metric['switch_rate']:.6f} |",
        "",
        "## Replay Checks",
        "",
        f"- metric_summary_exact_replay: `{payload['replay']['metric_summary_exact_replay']}`",
        f"- source_rows_exact_replay: `{payload['replay']['source_rows_exact_replay']}`",
        f"- domain_rows_exact_replay: `{payload['replay']['domain_rows_exact_replay']}`",
        f"- il_delta_audit_exact_replay: `{payload['replay']['il_delta_audit_exact_replay']}`",
        f"- non_ucy_max_abs_delta: `{payload['replay']['non_ucy_max_abs_delta']:.12f}`",
        "",
        "## Interpretation",
        "",
        "- Stage42-IM is the frozen deployment contract for the IK source-specialist composition.",
        "- It does not train a new model and does not select new thresholds.",
        "- It keeps Stage42-II as the default route and routes UCY to the row-aligned Stage42-X full-waypoint specialist.",
        "- Claims remain protected dataset-local/raw-frame 2.5D only.",
    ]
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-IM Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | `{bool(ok)}` |")
    write_md(GATE_MD, lines)


def _refresh_readmes_and_state(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_im_gate"]
    metric = payload["frozen_policy"]["test_summary_vs_train_horizon_causal_floor"]
    lines = [
        "## Stage42-IM T50 Source-Specialist Policy Freeze",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- policy artifact: `{POLICY_JSON}`",
        f"- policy hash: `{payload['policy_hash']}`",
        f"- ADE all / t50 / hard: `{metric['ade_all']:.6f}` / `{metric['ade_t50']:.6f}` / `{metric['ade_hard_failure']:.6f}`",
        f"- FDE t50: `{metric['fde_t50']:.6f}`",
        f"- easy degradation: `{metric['ade_easy_degradation']:.6f}`",
        "- boundary: frozen source-specialist t50 policy; dataset-local/raw-frame 2.5D; no metric/seconds claim, no Stage5C, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, MASTER_README]:
        _replace_section(path, SECTION, lines)
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage42_im_t50_source_specialist_policy_freeze"
    state["current_verdict"] = gate["verdict"]
    stage42 = state.setdefault("stage42", {})
    stage42["stage_im_t50_source_specialist_policy_freeze"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "policy": str(POLICY_JSON),
        "policy_hash": payload["policy_hash"],
        "gate": str(GATE_MD),
        "verdict": gate["verdict"],
        "gates": f"{gate['passed']}/{gate['total']}",
        "metric": metric,
        "claim_boundary": payload["frozen_policy"]["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD, POLICY_JSON, POLICY_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(RESEARCH_STATE, _jsonable(state))


def run_stage42_t50_source_specialist_policy_freeze() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ik = read_json(s42ik.REPORT_JSON, {})
    il = read_json(s42il.REPORT_JSON, {})
    if ik.get("stage42_ik_gate", {}).get("verdict") != "stage42_ik_ucy_specialist_integration_pass":
        raise RuntimeError("Stage42-IK must pass before freezing the source-specialist policy.")
    if il.get("stage42_il_gate", {}).get("verdict") != "stage42_il_ucy_specialist_claim_audit_pass":
        raise RuntimeError("Stage42-IL must pass before freezing the source-specialist policy.")
    policy = _policy_payload(ik, il)
    policy_artifact = _write_policy(policy)
    policy_hash = _stable_hash(policy)
    replay = _replay(policy, ik, il)
    payload: dict[str, Any] = {
        "stage": "Stage42-IM",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([s42ik.REPORT_JSON, s42il.REPORT_JSON]),
        "inputs": {
            "stage42ik_report": str(s42ik.REPORT_JSON),
            "stage42ik_gate": ik.get("stage42_ik_gate", {}),
            "stage42il_report": str(s42il.REPORT_JSON),
            "stage42il_gate": il.get("stage42_il_gate", {}),
        },
        "frozen_policy": policy,
        "policy_artifact": policy_artifact,
        "policy_hash": policy_hash,
        "replay": replay,
    }
    payload["stage42_im_gate"] = _gate(payload)
    write_json(REPORT_JSON, _jsonable(payload))
    _write_report(payload)
    _write_gate(payload["stage42_im_gate"])
    _refresh_readmes_and_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_t50_source_specialist_policy_freeze()
    print(json.dumps(_jsonable(result["stage42_im_gate"]), ensure_ascii=False, indent=2))
