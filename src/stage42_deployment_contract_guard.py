from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
DN_JSON = OUT_DIR / "deployment_variant_card_stage42.json"
EM_JSON = OUT_DIR / "official_source_link_audit_stage42.json"
EN_JSON = OUT_DIR / "floor_removability_decision_map_stage42.json"
EO_JSON = OUT_DIR / "paper_package_post_en_refresh_stage42.json"

REPORT_JSON = OUT_DIR / "deployment_contract_stage42.json"
REPORT_MD = OUT_DIR / "deployment_contract_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ep_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_deployment_contract_guard"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EP turns deployment/paper claim boundaries into a machine-readable contract guard.",
    "This stage does not train, download, convert, or tune thresholds.",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
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


def _gate_passed(data: Mapping[str, Any], key: str) -> bool:
    gate = data.get(key, {})
    return bool(gate) and gate.get("passed") == gate.get("total")


def _variant(dn: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for row in dn.get("deployment_variants", []):
        if row.get("variant") == name:
            return row
    return {}


def _component(en: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for row in en.get("component_decision_map", []):
        if row.get("component") == name:
            return row
    return {}


def evaluate_contract_request(request: str, dn: Mapping[str, Any], em: Mapping[str, Any], en: Mapping[str, Any]) -> ContractDecision:
    em_summary = em.get("summary", {})
    en_summary = en.get("summary", {})
    prox = _variant(dn, "proximity_guard")
    no_guard = _variant(dn, "no_proximity_guard")
    group = _variant(dn, "group_consistency_full_waypoint_runtime")
    ungated = _component(en, "ungated_neural_endpoint_or_full_waypoint")
    teacher = _component(en, "teacher_floor_rollout_context")
    fallback = _component(en, "deployment_fallback_floor")
    source = _component(en, "source_expansion_without_terms")

    if request == "safety_sensitive_bridge_shape_deployment":
        return ContractDecision(
            request=request,
            allowed=True,
            status="allowed_protected",
            deployment_role="safety_sensitive_deployable_bridge_shape_policy",
            required_conditions=[
                "use proximity_guard variant",
                "keep Stage37/teacher floor",
                "report dataset-local/raw-frame only",
                "do not use future endpoint/waypoint inputs",
            ],
            denied_reasons=[],
            evidence=[
                f"DN proximity_guard status={prox.get('deployment_status')}",
                "EN proximity_guard required for safety-sensitive reporting",
            ],
        )

    if request == "accuracy_priority_no_guard_reporting":
        return ContractDecision(
            request=request,
            allowed=True,
            status="allowed_diagnostic_only",
            deployment_role="accuracy_priority_diagnostic",
            required_conditions=[
                "label as diagnostic/accuracy-priority",
                "do not present as safety-sensitive deployable",
                "include near-collision caveat",
            ],
            denied_reasons=[],
            evidence=[
                f"DN no_guard status={no_guard.get('deployment_status')}",
                "CR/EN: no-guard has higher ADE but worsens near-collision",
            ],
        )

    if request == "source_level_full_waypoint_runtime":
        return ContractDecision(
            request=request,
            allowed=True,
            status="allowed_protocol_specific",
            deployment_role="source_level_full_waypoint_group_consistency_runtime_policy",
            required_conditions=[
                "state source-level protocol baseline",
                "keep protected floor context",
                "do not rank-mix with endpoint-linear composer without baseline caveat",
                "report raw-frame/dataset-local only",
            ],
            denied_reasons=[],
            evidence=[
                f"DN group runtime status={group.get('deployment_status')}",
                "EO supported claim: protected source-level group-consistency full-waypoint raw-frame 2.5D evidence",
            ],
        )

    if request == "global_floor_free_neural_deployment":
        return ContractDecision(
            request=request,
            allowed=False,
            status="blocked",
            deployment_role="forbidden",
            required_conditions=[],
            denied_reasons=[
                "global floor-free neural is not deployable",
                "ungated endpoint/full-waypoint easy degradation violates the 2% safety limit",
            ],
            evidence=[
                f"EN floor_free_neural_deployable={en_summary.get('floor_free_neural_deployable')}",
                f"EN ungated decision={ungated.get('decision')}",
            ],
        )

    if request == "teacher_floor_rollout_context_removal":
        return ContractDecision(
            request=request,
            allowed=False,
            status="blocked_required_mechanism",
            deployment_role="forbidden",
            required_conditions=[],
            denied_reasons=[
                "teacher/floor rollout context remains a core mechanism",
                "removing floor/safe baseline rollout context hurts protected t50",
            ],
            evidence=[
                f"EN teacher removal allowed={en_summary.get('teacher_floor_rollout_context_removal_allowed')}",
                f"EN teacher decision={teacher.get('decision')}",
            ],
        )

    if request == "validation_backed_t50_slice_relaxation":
        return ContractDecision(
            request=request,
            allowed=True,
            status="allowed_slice_only",
            deployment_role="partial_t50_floor_relaxation",
            required_conditions=[
                "only on mapped t50 slices",
                "use train/internal-validation policy",
                "do not generalize to global floor-free deployment",
                "keep teacher/floor rollout context",
            ],
            denied_reasons=[],
            evidence=[
                f"EN partial slices={en_summary.get('partial_relaxation_components')}",
                f"EN fallback decision={fallback.get('decision')}",
            ],
        )

    if request == "source_conversion_without_user_terms":
        return ContractDecision(
            request=request,
            allowed=False,
            status="blocked_manual_terms_required",
            deployment_role="forbidden",
            required_conditions=[],
            denied_reasons=[
                "official links are not license acceptance",
                "user must confirm terms, allowed use, local path, and source identity",
                "auto download/conversion/evaluation are not allowed now",
            ],
            evidence=[
                f"EM conversion_ready_now={em_summary.get('conversion_ready_now')}",
                f"EM auto_download_allowed_now={em_summary.get('auto_download_allowed_now')}",
                f"EN source decision={source.get('decision')}",
            ],
        )

    if request == "metric_seconds_or_foundation_claim":
        return ContractDecision(
            request=request,
            allowed=False,
            status="forbidden",
            deployment_role="forbidden",
            required_conditions=[],
            denied_reasons=[
                "raw-frame/dataset-local evidence does not support metric or seconds-level claims",
                "current model is not true 3D and not a foundation model",
                "Stage5C and SMC remain disabled",
            ],
            evidence=[
                "EO claim boundary: true 3D/foundation/metric/seconds/Stage5C/SMC forbidden",
            ],
        )

    return ContractDecision(
        request=request,
        allowed=False,
        status="unknown_request_blocked_by_default",
        deployment_role="forbidden",
        required_conditions=[],
        denied_reasons=["unknown request is blocked until explicitly added to the contract"],
        evidence=["default-deny contract behavior"],
    )


def _build_contract(dn: Mapping[str, Any], em: Mapping[str, Any], en: Mapping[str, Any]) -> dict[str, Any]:
    requests = [
        "safety_sensitive_bridge_shape_deployment",
        "accuracy_priority_no_guard_reporting",
        "source_level_full_waypoint_runtime",
        "global_floor_free_neural_deployment",
        "teacher_floor_rollout_context_removal",
        "validation_backed_t50_slice_relaxation",
        "source_conversion_without_user_terms",
        "metric_seconds_or_foundation_claim",
        "unknown_future_policy_request",
    ]
    decisions = [evaluate_contract_request(req, dn, em, en).as_dict() for req in requests]
    return {
        "requests": decisions,
        "allowed_requests": [row["request"] for row in decisions if row["allowed"]],
        "blocked_requests": [row["request"] for row in decisions if not row["allowed"]],
        "default_deny_unknown": evaluate_contract_request("__unknown__", dn, em, en).allowed is False,
        "deployment_defaults": {
            "safety_sensitive_default": "proximity_guard",
            "source_level_runtime_candidate": "group_consistency_full_waypoint_runtime",
            "accuracy_priority_diagnostic": "no_proximity_guard",
            "global_floor_free_neural": "blocked",
            "source_conversion_without_terms": "blocked",
            "metric_seconds_foundation_claim": "blocked",
        },
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    decisions = {row["request"]: row for row in payload["contract"]["requests"]}
    gates = {
        "dn_input_passed": _gate_passed(payload["inputs"]["stage42_dn"], "stage42_dn_gate"),
        "em_input_passed": _gate_passed(payload["inputs"]["stage42_em"], "stage42_em_gate"),
        "en_input_passed": _gate_passed(payload["inputs"]["stage42_en"], "stage42_en_gate"),
        "eo_input_passed": _gate_passed(payload["inputs"]["stage42_eo"], "stage42_eo_gate"),
        "safety_sensitive_default_allowed": decisions["safety_sensitive_bridge_shape_deployment"]["allowed"] is True,
        "no_guard_diagnostic_only": decisions["accuracy_priority_no_guard_reporting"]["status"] == "allowed_diagnostic_only",
        "source_level_runtime_protocol_specific": decisions["source_level_full_waypoint_runtime"]["status"]
        == "allowed_protocol_specific",
        "floor_free_neural_blocked": decisions["global_floor_free_neural_deployment"]["allowed"] is False,
        "teacher_context_removal_blocked": decisions["teacher_floor_rollout_context_removal"]["allowed"] is False,
        "partial_t50_slice_relaxation_allowed": decisions["validation_backed_t50_slice_relaxation"]["status"]
        == "allowed_slice_only",
        "source_conversion_without_terms_blocked": decisions["source_conversion_without_user_terms"]["allowed"] is False,
        "metric_seconds_foundation_blocked": decisions["metric_seconds_or_foundation_claim"]["allowed"] is False,
        "unknown_requests_default_deny": payload["contract"]["default_deny_unknown"] is True
        and decisions["unknown_future_policy_request"]["allowed"] is False,
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_ep_deployment_contract_guard_pass" if passed == total else "stage42_ep_deployment_contract_guard_partial"
    return {"source": payload.get("source", "unknown"), "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ep_gate"]
    lines = [
        "# Stage42-EP Deployment Contract Guard",
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
    gate = payload["stage42_ep_gate"]
    return [
        "# Stage42-EP Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ep_gate"]
    defaults = payload["contract"]["deployment_defaults"]
    return [
        "## Stage42-EP Deployment Contract Guard",
        "",
        "- source: `fresh_stage42_deployment_contract_guard`",
        f"- verdict: `{gate['verdict']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        "- role: machine-readable guard for deployment and paper-claim requests after Stage42-DN/EM/EN/EO.",
        f"- safety_sensitive_default: `{defaults['safety_sensitive_default']}`.",
        f"- source_level_runtime_candidate: `{defaults['source_level_runtime_candidate']}`.",
        "- allowed only as diagnostic: `no_proximity_guard` accuracy-priority reporting.",
        "- blocked: global floor-free neural deployment, teacher-floor rollout context removal, source conversion without user terms, metric/seconds/foundation claims, Stage5C execution, and SMC.",
        "- unknown future policy requests are denied by default until explicitly added to the contract.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, "STAGE42_EP_DEPLOYMENT_CONTRACT_GUARD", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-EP deployment contract guard"
    state["current_verdict"] = payload["stage42_ep_gate"]["verdict"]
    state["stage42_ep_deployment_contract_guard"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_ep_gate"]["verdict"],
        "gates": f"{payload['stage42_ep_gate']['passed']}/{payload['stage42_ep_gate']['total']}",
        "contract": payload["contract"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_deployment_contract_guard(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    dn = read_json(DN_JSON, {})
    em = read_json(EM_JSON, {})
    en = read_json(EN_JSON, {})
    eo = read_json(EO_JSON, {})
    contract = _build_contract(dn, em, en)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-EP",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([DN_JSON, EM_JSON, EN_JSON, EO_JSON]),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "stage42_dn": {"stage42_dn_gate": dn.get("stage42_dn_gate", {})},
            "stage42_em": {"stage42_em_gate": em.get("stage42_em_gate", {})},
            "stage42_en": {"stage42_en_gate": en.get("stage42_en_gate", {})},
            "stage42_eo": {"stage42_eo_gate": eo.get("stage42_eo_gate", {})},
        },
        "contract": contract,
        "claim_boundary": {
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
    payload["stage42_ep_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_deployment_contract_guard()
