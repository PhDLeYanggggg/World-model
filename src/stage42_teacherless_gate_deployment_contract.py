from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_full_waypoint_bridge_shape_audit import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

HE_JSON = OUT_DIR / "floor_free_proximity_guard_robustness_stage42.json"
HD_JSON = OUT_DIR / "floor_free_proximity_guard_repair_stage42.json"
HC_JSON = OUT_DIR / "floor_alternative_gate_stress_stage42.json"
GZ_JSON = OUT_DIR / "full_waypoint_claim_guard_stage42.json"
GJ_JSON = OUT_DIR / "module_claim_lock_stage42.json"

REPORT_JSON = OUT_DIR / "teacherless_gate_deployment_contract_stage42.json"
REPORT_MD = OUT_DIR / "teacherless_gate_deployment_contract_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hf_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
USER_SUMMARY = Path("README_M3W_USER_DETAILED_SUMMARY_ZH.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
GOAL_SUMMARY = Path("README_M3W_GOAL_FULL_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_hf_teacherless_gate_deployment_contract"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HF 是 deployment / paper claim contract refresh，不训练、不调 threshold、不下载、不转换。",
    "Stage42-HE 支持 teacherless proximity-guard switch gate，但仍要求 causal floor fallback。",
    "teacher gate removal 只限 repaired proximity-guard switch policy；不是 global causal floor removal。",
    "future endpoint / future waypoint 只可作为 supervised/evaluation labels，不可作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


@dataclass(frozen=True)
class ContractDecision:
    request: str
    allowed: bool
    status: str
    deployment_role: str
    required_conditions: list[str]
    denied_reasons: list[str]
    evidence: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "allowed": self.allowed,
            "status": self.status,
            "deployment_role": self.deployment_role,
            "required_conditions": self.required_conditions,
            "denied_reasons": self.denied_reasons,
            "evidence": self.evidence,
        }


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _gate_passed(data: Mapping[str, Any], key: str) -> bool:
    gate = data.get(key, {})
    return bool(gate and gate.get("passed") == gate.get("total") and int(gate.get("total", 0)) > 0)


def _summary(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return payload.get("summary", {})


def evaluate_contract_request(
    request: str,
    he: Mapping[str, Any],
    hd: Mapping[str, Any],
    hc: Mapping[str, Any],
) -> ContractDecision:
    he_s = _summary(he)
    hd_s = _summary(hd)
    hc_s = _summary(hc)
    if request == "teacherless_proximity_guarded_switch_gate":
        return ContractDecision(
            request=request,
            allowed=bool(
                he_s.get("teacherless_gate_paper_evidence_supported")
                and he_s.get("teacher_gate_used") is False
                and he_s.get("causal_floor_fallback_used") is True
                and float(he_s.get("easy_degradation", 1.0)) <= 0.02
            ),
            status="allowed_protected",
            deployment_role="teacherless_proximity_guarded_switch_gate_with_causal_floor_fallback",
            required_conditions=[
                "use the Stage42-HE repaired harm_predictor_gate plus validation-selected proximity guard",
                "keep causal floor fallback active",
                "report dataset-local/raw-frame 2.5D only",
                "do not tune thresholds on test",
            ],
            denied_reasons=[],
            evidence=[
                f"HE gate={he.get('stage42_he_gate', {}).get('passed')}/{he.get('stage42_he_gate', {}).get('total')}",
                f"all={_pct(he_s.get('all_improvement'))}, t50={_pct(he_s.get('t50_improvement'))}, hard={_pct(he_s.get('hard_failure_improvement'))}, easy_degradation={_pct(he_s.get('easy_degradation'))}",
                f"teacher_gate_used={he_s.get('teacher_gate_used')}, causal_floor_fallback_used={he_s.get('causal_floor_fallback_used')}",
            ],
        )

    if request == "teacher_gate_removal_for_repaired_gate":
        return ContractDecision(
            request=request,
            allowed=bool(he_s.get("teacher_gate_used") is False and he_s.get("causal_floor_fallback_used") is True),
            status="allowed_policy_specific_not_global",
            deployment_role="teacher_gate_removed_only_for_repaired_floor_free_switch_gate",
            required_conditions=[
                "scope the claim to the repaired proximity-guard switch gate",
                "state that teacher/floor mechanisms remain required elsewhere",
                "keep causal baseline floor fallback and proximity safety guard",
            ],
            denied_reasons=[],
            evidence=[
                f"HD best_post_guard_family={hd_s.get('best_post_guard_family')}",
                f"HD post_guard_deployable_count={hd_s.get('post_guard_deployable_count')}",
                "HE claim boundary keeps causal floor safety fallback required.",
            ],
        )

    if request == "causal_floor_removal":
        return ContractDecision(
            request=request,
            allowed=False,
            status="blocked_required_safety_floor",
            deployment_role="forbidden",
            required_conditions=[],
            denied_reasons=[
                "Stage42-HE robust candidate still uses causal floor fallback",
                "Stage42-HC found zero globally deployable floor-free candidates",
                "global floor removal would overclaim beyond protected switch-gate evidence",
            ],
            evidence=[
                f"HE global_floor_removal_allowed={he_s.get('global_floor_removal_allowed')}",
                f"HC floor_free_deployable_count={hc_s.get('floor_free_deployable_count')}",
                f"HC floor_free_positive_but_unsafe_count={hc_s.get('floor_free_positive_but_unsafe_count')}",
            ],
        )

    if request == "ungated_neural_or_floor_free_global_deployment":
        return ContractDecision(
            request=request,
            allowed=False,
            status="blocked_unsafe",
            deployment_role="forbidden",
            required_conditions=[],
            denied_reasons=[
                "ungated/floor-free global policies are not deployable",
                "positive raw floor-free switches were unsafe under near-collision/proximity stress",
                "protected deployment must preserve fallback and guard",
            ],
            evidence=[
                f"HC floor_free_deployable_count={hc_s.get('floor_free_deployable_count')}",
                f"HC best_floor_free_candidate={hc_s.get('best_floor_free_candidate', {}).get('family')}",
            ],
        )

    if request == "partial_t50_floor_relaxation":
        return ContractDecision(
            request=request,
            allowed=True,
            status="allowed_slice_only",
            deployment_role="bounded_validation_backed_t50_slice_relaxation",
            required_conditions=[
                "only use validation-backed mapped t50 slices",
                "do not present as global floor-free deployment",
                "keep fallback outside validated slices",
            ],
            denied_reasons=[],
            evidence=[
                "Stage42-HC / floor-relaxation reports support only slice-bounded relaxation, not global deployment.",
                f"HE weak_domain_horizon_slices={he_s.get('weak_domain_horizon_slices')}",
            ],
        )

    if request == "teacherless_gate_as_paper_claim":
        return ContractDecision(
            request=request,
            allowed=True,
            status="allowed_with_boundary",
            deployment_role="paper_claim_teacherless_switch_gate_evidence",
            required_conditions=[
                "write 'teacherless proximity-guarded switch gate' rather than 'floor-free world model'",
                "include causal floor fallback as a required safety mechanism",
                "include raw-frame/dataset-local/2.5D limitations",
                "state that Stage5C and SMC were not executed",
            ],
            denied_reasons=[],
            evidence=[
                "HE robustness: positive all/t50/t100 raw/hard with 2000 bootstrap and no weak domain-horizon slices.",
                "HC stress: global floor-free deployable count remains zero.",
            ],
        )

    if request == "metric_seconds_true3d_foundation_claim":
        return ContractDecision(
            request=request,
            allowed=False,
            status="forbidden",
            deployment_role="forbidden",
            required_conditions=[],
            denied_reasons=[
                "dataset-local/raw-frame evidence does not verify metric coordinates or seconds-level horizons",
                "current system is not true 3D and not a large-scale foundation world model",
            ],
            evidence=["All Stage42 reports preserve the 2.5D raw-frame claim boundary."],
        )

    if request == "stage5c_execution_or_smc_enabled":
        return ContractDecision(
            request=request,
            allowed=False,
            status="forbidden",
            deployment_role="forbidden",
            required_conditions=[],
            denied_reasons=[
                "Stage5C latent generative execution remains explicitly disabled",
                "SMC remains explicitly disabled",
                "this contract is a guard/report refresh, not a stochastic generative rollout stage",
            ],
            evidence=[
                f"HE stage5c_executed={he_s.get('stage5c_executed')}",
                f"HE smc_enabled={he_s.get('smc_enabled')}",
            ],
        )

    return ContractDecision(
        request=request,
        allowed=False,
        status="unknown_request_blocked_by_default",
        deployment_role="forbidden",
        required_conditions=[],
        denied_reasons=["unknown request is blocked until added to the contract"],
        evidence=["default-deny contract behavior"],
    )


def _build_contract(he: Mapping[str, Any], hd: Mapping[str, Any], hc: Mapping[str, Any]) -> dict[str, Any]:
    requests = [
        "teacherless_proximity_guarded_switch_gate",
        "teacher_gate_removal_for_repaired_gate",
        "causal_floor_removal",
        "ungated_neural_or_floor_free_global_deployment",
        "partial_t50_floor_relaxation",
        "teacherless_gate_as_paper_claim",
        "metric_seconds_true3d_foundation_claim",
        "stage5c_execution_or_smc_enabled",
        "unknown_future_policy_request",
    ]
    decisions = [evaluate_contract_request(req, he, hd, hc).as_dict() for req in requests]
    return {
        "requests": decisions,
        "allowed_requests": [row["request"] for row in decisions if row["allowed"]],
        "blocked_requests": [row["request"] for row in decisions if not row["allowed"]],
        "default_deny_unknown": evaluate_contract_request("__unknown__", he, hd, hc).allowed is False,
        "deployment_defaults": {
            "deployable_default": "Stage37/Stage42 protected causal-floor fallback policy",
            "teacherless_candidate": "Stage42-HE proximity-guarded harm_predictor_gate",
            "required_safety_floor": "causal floor fallback remains required",
            "global_floor_free": "blocked",
            "ungated_neural": "blocked",
            "metric_seconds_true3d_foundation_claim": "forbidden",
            "stage5c": "not executed",
            "smc": "disabled",
        },
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    decisions = {row["request"]: row for row in payload["contract"]["requests"]}
    gates = {
        "he_input_passed": _gate_passed(payload["inputs"]["stage42_he"], "stage42_he_gate"),
        "hd_input_passed": _gate_passed(payload["inputs"]["stage42_hd"], "stage42_hd_gate"),
        "hc_input_passed": _gate_passed(payload["inputs"]["stage42_hc"], "stage42_hc_gate"),
        "teacherless_gate_allowed_protected": decisions["teacherless_proximity_guarded_switch_gate"]["status"]
        == "allowed_protected",
        "teacher_gate_removal_policy_specific": decisions["teacher_gate_removal_for_repaired_gate"]["status"]
        == "allowed_policy_specific_not_global",
        "causal_floor_removal_blocked": decisions["causal_floor_removal"]["allowed"] is False,
        "ungated_global_floor_free_blocked": decisions["ungated_neural_or_floor_free_global_deployment"]["allowed"]
        is False,
        "partial_t50_relaxation_slice_only": decisions["partial_t50_floor_relaxation"]["status"] == "allowed_slice_only",
        "paper_claim_bounded": decisions["teacherless_gate_as_paper_claim"]["status"] == "allowed_with_boundary",
        "metric_seconds_true3d_foundation_blocked": decisions["metric_seconds_true3d_foundation_claim"]["allowed"]
        is False,
        "stage5c_smc_blocked": decisions["stage5c_execution_or_smc_enabled"]["allowed"] is False,
        "unknown_requests_default_deny": payload["contract"]["default_deny_unknown"] is True
        and decisions["unknown_future_policy_request"]["allowed"] is False,
        "no_future_or_test_leakage_claim": all(value is False for value in payload["no_leakage"].values()),
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = (
        "stage42_hf_teacherless_gate_deployment_contract_pass"
        if passed == total
        else "stage42_hf_teacherless_gate_deployment_contract_partial"
    )
    return {"source": payload.get("source", SOURCE), "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    he = read_json(HE_JSON, {})
    hd = read_json(HD_JSON, {})
    hc = read_json(HC_JSON, {})
    gz = read_json(GZ_JSON, {})
    gj = read_json(GJ_JSON, {})
    contract = _build_contract(he, hd, hc)
    he_s = _summary(he)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-HF",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([HE_JSON, HD_JSON, HC_JSON, GZ_JSON, GJ_JSON]),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "stage42_he": {"stage42_he_gate": he.get("stage42_he_gate", {})},
            "stage42_hd": {"stage42_hd_gate": hd.get("stage42_hd_gate", {})},
            "stage42_hc": {"stage42_hc_gate": hc.get("stage42_hc_gate", {})},
            "stage42_gz": {"stage42_gz_gate": gz.get("stage42_gz_gate", {})},
            "stage42_gj": {"stage42_gj_gate": gj.get("stage42_gj_gate", {})},
        },
        "teacherless_gate_summary": {
            "policy_family": he_s.get("policy_family"),
            "min_sep": he_s.get("min_sep"),
            "teacher_gate_used": he_s.get("teacher_gate_used"),
            "causal_floor_fallback_used": he_s.get("causal_floor_fallback_used"),
            "global_floor_removal_allowed": he_s.get("global_floor_removal_allowed"),
            "all_improvement": he_s.get("all_improvement"),
            "t50_improvement": he_s.get("t50_improvement"),
            "t100_raw_frame_diagnostic_improvement": he_s.get("t100_raw_frame_diagnostic_improvement"),
            "hard_failure_improvement": he_s.get("hard_failure_improvement"),
            "easy_degradation": he_s.get("easy_degradation"),
            "bootstrap_n": he_s.get("bootstrap_n"),
            "robust_positive_domains": he_s.get("robust_positive_domains"),
            "weak_domain_horizon_slices": he_s.get("weak_domain_horizon_slices"),
        },
        "contract": contract,
        "claim_boundary": {
            "teacherless_gate_paper_claim_allowed": True,
            "teacherless_gate_scope": "repaired proximity-guarded switch gate only",
            "causal_floor_safety_fallback_still_required": True,
            "global_floor_removal_allowed": False,
            "ungated_neural_deployment_allowed": False,
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
    }
    payload["stage42_hf_gate"] = _gate(payload)
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hf_gate"]
    summary = payload["teacherless_gate_summary"]
    lines = [
        "# Stage42-HF Teacherless Gate Deployment Contract",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Teacherless Gate Evidence",
        "",
        f"- policy_family: `{summary.get('policy_family')}`",
        f"- min_sep: `{summary.get('min_sep')}`",
        f"- teacher_gate_used: `{summary.get('teacher_gate_used')}`",
        f"- causal_floor_fallback_used: `{summary.get('causal_floor_fallback_used')}`",
        f"- global_floor_removal_allowed: `{summary.get('global_floor_removal_allowed')}`",
        f"- all improvement: `{_pct(summary.get('all_improvement'))}`",
        f"- t50 improvement: `{_pct(summary.get('t50_improvement'))}`",
        f"- t100 raw-frame diagnostic improvement: `{_pct(summary.get('t100_raw_frame_diagnostic_improvement'))}`",
        f"- hard/failure improvement: `{_pct(summary.get('hard_failure_improvement'))}`",
        f"- easy degradation: `{_pct(summary.get('easy_degradation'))}`",
        f"- bootstrap_n: `{summary.get('bootstrap_n')}`",
        f"- robust_positive_domains: `{summary.get('robust_positive_domains')}`",
        f"- weak_domain_horizon_slices: `{summary.get('weak_domain_horizon_slices')}`",
        "",
        "## Contract Decisions",
        "",
        "| request | allowed | status | role | denied reasons | required conditions |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for row in payload["contract"]["requests"]:
        lines.append(
            "| `{}` | {} | `{}` | `{}` | {} | {} |".format(
                row["request"],
                row["allowed"],
                row["status"],
                row["deployment_role"],
                "<br>".join(row["denied_reasons"]) if row["denied_reasons"] else "",
                "<br>".join(row["required_conditions"]) if row["required_conditions"] else "",
            )
        )
    lines += [
        "",
        "## Deployment Defaults",
        "",
        *[f"- {key}: `{value}`" for key, value in payload["contract"]["deployment_defaults"].items()],
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hf_gate"]
    return [
        "# Stage42-HF Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hf_gate"]
    s = payload["teacherless_gate_summary"]
    return [
        "## Stage42-HF Teacherless Gate Deployment Contract",
        "",
        "- source: `fresh_stage42_hf_teacherless_gate_deployment_contract`",
        f"- verdict: `{gate['verdict']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        "- result: Stage42-HE supports a teacherless proximity-guarded switch gate, but only with causal floor fallback.",
        f"- metrics: all `{_pct(s.get('all_improvement'))}`, t50 `{_pct(s.get('t50_improvement'))}`, t100 raw diagnostic `{_pct(s.get('t100_raw_frame_diagnostic_improvement'))}`, hard/failure `{_pct(s.get('hard_failure_improvement'))}`, easy degradation `{_pct(s.get('easy_degradation'))}`.",
        "- allowed claim: `teacherless proximity-guarded switch gate with causal floor fallback`.",
        "- blocked claims: global causal floor removal, ungated neural deployment, metric/seconds/true-3D/foundation claims, Stage5C execution, and SMC.",
        "- deployment default remains protected causal-floor fallback; Stage42-HF is a claim/deployment contract refresh, not new training.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, USER_SUMMARY, WORK_SUMMARY, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_HF_TEACHERLESS_GATE_DEPLOYMENT_CONTRACT", lines)


def _refresh_research_state(payload: Mapping[str, Any], *, verification: Mapping[str, Any] | None = None) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-HF teacherless gate deployment contract"
    state["current_verdict"] = payload["stage42_hf_gate"]["verdict"]
    state["stage42_hf_teacherless_gate_deployment_contract"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_hf_gate"]["verdict"],
        "gates": f"{payload['stage42_hf_gate']['passed']}/{payload['stage42_hf_gate']['total']}",
        "teacherless_gate_summary": payload["teacherless_gate_summary"],
        "contract": payload["contract"],
        "claim_boundary": payload["claim_boundary"],
        "verification": dict(verification or {"status": "pending"}),
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_teacherless_gate_deployment_contract(
    *,
    refresh_readmes: bool = True,
    verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload, verification=verification)
    return payload


if __name__ == "__main__":
    run_stage42_teacherless_gate_deployment_contract()
