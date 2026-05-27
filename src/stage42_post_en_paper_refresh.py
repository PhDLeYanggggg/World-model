from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
EG_JSON = OUT_DIR / "paper_package_post_ef_refresh_stage42.json"
EM_JSON = OUT_DIR / "official_source_link_audit_stage42.json"
EN_JSON = OUT_DIR / "floor_removability_decision_map_stage42.json"

REPORT_JSON = OUT_DIR / "paper_package_post_en_refresh_stage42.json"
REPORT_MD = OUT_DIR / "paper_package_post_en_refresh_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_eo_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

PAPER_FILES = [
    OUT_DIR / "paper_outline_stage42.md",
    OUT_DIR / "method_draft_stage42.md",
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "failure_taxonomy_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "data_card_stage42.md",
    OUT_DIR / "reproducibility_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

SOURCE = "fresh_paper_refresh_from_stage42_eg_em_en"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EO is a post-EM/EN paper-package refresh; it does not download, convert, train, or tune thresholds.",
    "本阶段把 official source link/manual terms blocker 和 floor-removability decision map 写入 paper package。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "source_conversion_claim_allowed": False,
    "floor_free_neural_deployable": False,
    "global_floor_removal_allowed": False,
    "teacher_floor_rollout_context_removal_allowed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _gate_passed(data: Mapping[str, Any], key: str) -> bool:
    gate = data.get(key, {})
    return bool(gate) and gate.get("passed") == gate.get("total")


def _find_decision(en: Mapping[str, Any], component: str) -> Mapping[str, Any]:
    for row in en.get("component_decision_map", []):
        if row.get("component") == component:
            return row
    return {}


def _paper_claim_matrix(eg: Mapping[str, Any], em: Mapping[str, Any], en: Mapping[str, Any]) -> list[dict[str, Any]]:
    eg_summary = eg["paper_refresh_summary"]
    em_summary = em["summary"]
    en_summary = en["summary"]
    fallback = _find_decision(en, "deployment_fallback_floor")
    proximity = _find_decision(en, "proximity_guard")
    source = _find_decision(en, "source_expansion_without_terms")
    teacher = _find_decision(en, "teacher_floor_rollout_context")
    ungated = _find_decision(en, "ungated_neural_endpoint_or_full_waypoint")
    return [
        {
            "claim": "protected_source_level_group_consistency_full_waypoint",
            "status": "supported_bounded",
            "main_claim_allowed": True,
            "evidence": "Stage42-EG preserves the protected source-level group-consistency full-waypoint claim.",
            "boundary": "protected, source-level, dataset-local/raw-frame only; not global ungated/foundation/metric",
            "key_numbers": eg_summary.get("supported_main_claims", []),
        },
        {
            "claim": "official_source_expansion_or_conversion",
            "status": "blocked_until_manual_terms_path_source_identity",
            "main_claim_allowed": False,
            "evidence": "Stage42-EM records official/toolkit source candidates but conversion_ready_now and auto_download_allowed_now are both zero.",
            "boundary": "official links are not license acceptance; no raw download/conversion/eval claim now",
            "key_numbers": {
                "targets": em_summary["targets"],
                "official_or_toolkit_source_candidates": em_summary["official_or_toolkit_source_candidates"],
                "manual_terms_required_targets": em_summary["manual_terms_required_targets"],
                "auto_download_allowed_now": em_summary["auto_download_allowed_now"],
                "conversion_ready_now": em_summary["conversion_ready_now"],
                "estimated_t50_after_terms": em_summary["estimated_t50_after_terms"],
                "estimated_t100_after_terms": em_summary["estimated_t100_after_terms"],
            },
        },
        {
            "claim": "floor_free_neural_deployment",
            "status": "blocked",
            "main_claim_allowed": False,
            "evidence": "Stage42-EN maps ungated endpoint/full-waypoint neural to blocked because easy degradation violates deployment safety.",
            "boundary": "do not deploy ungated neural; keep protected fallback unless a future gate proves otherwise",
            "key_numbers": ungated.get("key_metrics", {}),
        },
        {
            "claim": "teacher_floor_rollout_context_removal",
            "status": "blocked_required_mechanism",
            "main_claim_allowed": False,
            "evidence": "Stage42-EN shows removing teacher/floor rollout context hurts protected t50.",
            "boundary": "teacher/floor rollout context is a core mechanism, not a removable implementation detail",
            "key_numbers": teacher.get("key_metrics", {}),
        },
        {
            "claim": "validation_backed_t50_slice_floor_relaxation",
            "status": "partial_supported",
            "main_claim_allowed": True,
            "evidence": "Stage42-EN allows only narrow validation-backed t50 slice relaxation while global floor removal remains blocked.",
            "boundary": "slice-only under train/internal-validation policy; not global floor-free deployment",
            "key_numbers": fallback.get("key_metrics", {}),
        },
        {
            "claim": "proximity_guard_for_safety_sensitive_reporting",
            "status": "required",
            "main_claim_allowed": True,
            "evidence": "Stage42-EN records that no-guard improves ADE more but worsens near-collision; guard repairs proximity.",
            "boundary": "use guarded variant for safety-sensitive claims; no-guard remains accuracy-priority diagnostic",
            "key_numbers": proximity.get("key_metrics", {}),
        },
        {
            "claim": "global_metric_seconds_foundation_or_stage5c_smc",
            "status": "forbidden",
            "main_claim_allowed": False,
            "evidence": "Stage42-EM/EN do not change metric/time/foundation/Stage5C/SMC boundaries.",
            "boundary": "no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC",
            "key_numbers": {
                "global_metric_claim_allowed": False,
                "global_seconds_claim_allowed": False,
                "stage5c_executed": False,
                "smc_enabled": False,
            },
        },
    ]


def _summary(eg: Mapping[str, Any], em: Mapping[str, Any], en: Mapping[str, Any]) -> dict[str, Any]:
    matrix = _paper_claim_matrix(eg, em, en)
    em_summary = em["summary"]
    en_summary = en["summary"]
    return {
        "source": SOURCE,
        "paper_claim_matrix": matrix,
        "supported_main_claims": [row["claim"] for row in matrix if row["main_claim_allowed"]],
        "blocked_or_diagnostic_claims": [row["claim"] for row in matrix if not row["main_claim_allowed"]],
        "official_source_link_boundary": {
            "targets": em_summary["targets"],
            "official_or_toolkit_source_candidates": em_summary["official_or_toolkit_source_candidates"],
            "manual_terms_required_targets": em_summary["manual_terms_required_targets"],
            "auto_download_allowed_now": em_summary["auto_download_allowed_now"],
            "conversion_ready_now": em_summary["conversion_ready_now"],
            "converted_now": em_summary["converted_now"],
            "evaluated_now": em_summary["evaluated_now"],
            "estimated_t50_after_terms": em_summary["estimated_t50_after_terms"],
            "estimated_t100_after_terms": em_summary["estimated_t100_after_terms"],
            "next_validator_command": em_summary["next_validator_command"],
            "next_guarded_launcher_command": em_summary["next_guarded_launcher_command"],
        },
        "floor_removability_boundary": {
            "components_audited": en_summary["components_audited"],
            "floor_free_neural_deployable": en_summary["floor_free_neural_deployable"],
            "global_floor_removal_allowed": en_summary["global_floor_removal_allowed"],
            "teacher_floor_rollout_context_removal_allowed": en_summary["teacher_floor_rollout_context_removal_allowed"],
            "safe_partial_floor_relaxation_available": en_summary["safe_partial_floor_relaxation_available"],
            "partial_relaxation_components": en_summary["partial_relaxation_components"],
            "safety_required_components": en_summary["safety_required_components"],
            "proximity_guard_required_for_safety_claim": en_summary["proximity_guard_required_for_safety_claim"],
            "conversion_ready_now": en_summary["conversion_ready_now"],
        },
        "paper_verdict": {
            "paper_package_refreshed_after_em_en": True,
            "source_conversion_claim_allowed": False,
            "floor_free_neural_deployable": False,
            "global_floor_removal_allowed": False,
            "teacher_floor_rollout_context_removal_allowed": False,
            "partial_t50_relaxation_allowed": en_summary["safe_partial_floor_relaxation_available"],
            "proximity_guard_required_for_safety_claim": en_summary["proximity_guard_required_for_safety_claim"],
            "metric_seconds_claim_allowed": False,
            "foundation_claim_allowed": False,
            "stage5c_execution_allowed": False,
            "smc_allowed": False,
            "a_journal_candidate_status": "protected_2p5d_candidate_package_with_source_and_floor_boundaries_refreshed",
        },
        "eg_verdict": eg.get("stage42_eg_gate", {}).get("verdict", ""),
        "em_verdict": em.get("stage42_em_gate", {}).get("verdict", ""),
        "en_verdict": en.get("stage42_en_gate", {}).get("verdict", ""),
    }


def _refresh_lines(summary: Mapping[str, Any]) -> list[str]:
    source = summary["official_source_link_boundary"]
    floor = summary["floor_removability_boundary"]
    return [
        "## Stage42-EO Post-EM/EN Paper Package Refresh",
        "",
        "- source: `fresh_paper_refresh_from_stage42_eg_em_en`",
        "- role: propagate official-source/manual-terms blockers and floor-removability decisions into the paper package.",
        "- This is a paper-package refresh, not new training, conversion, download, or threshold tuning.",
        "",
        "### Source / Legal Boundary",
        "",
        f"- official/toolkit source candidates: `{source['official_or_toolkit_source_candidates']}` / `{source['targets']}`.",
        f"- manual terms required targets: `{source['manual_terms_required_targets']}`.",
        f"- auto_download_allowed_now: `{source['auto_download_allowed_now']}`; conversion_ready_now: `{source['conversion_ready_now']}`; converted/evaluated now: `{source['converted_now']}` / `{source['evaluated_now']}`.",
        f"- after-terms potential t50/t100 windows: `{source['estimated_t50_after_terms']}` / `{source['estimated_t100_after_terms']}`.",
        "- Official links are not license acceptance; user must confirm terms, allowed use, local path, and source identity before conversion.",
        "",
        "### Safety Floor Boundary",
        "",
        f"- floor_free_neural_deployable: `{floor['floor_free_neural_deployable']}`.",
        f"- global_floor_removal_allowed: `{floor['global_floor_removal_allowed']}`.",
        f"- teacher_floor_rollout_context_removal_allowed: `{floor['teacher_floor_rollout_context_removal_allowed']}`.",
        f"- safe_partial_floor_relaxation_available: `{floor['safe_partial_floor_relaxation_available']}` on `{floor['partial_relaxation_components']}`.",
        f"- proximity_guard_required_for_safety_claim: `{floor['proximity_guard_required_for_safety_claim']}`.",
        "",
        "### Updated Paper Claim Boundary",
        "",
        "- Supported: protected source-level group-consistency full-waypoint raw-frame 2.5D evidence.",
        "- Supported only as narrow slice evidence: validation-backed t50 floor relaxation on mapped slices.",
        "- Required: Stage37/teacher floor rollout context, deployment fallback floor, and proximity guard for safety-sensitive reporting.",
        "- Blocked: source conversion without user terms/path/source identity; global floor-free neural; teacher-floor rollout context removal.",
        "- Still forbidden: true 3D, foundation model, global metric/seconds-level claims, Stage5C execution, and SMC readiness.",
    ]


def _refresh_paper_files(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    lines = _refresh_lines(summary)
    status: list[dict[str, Any]] = []
    for path in PAPER_FILES:
        _replace_section(path, "STAGE42_EO_POST_EM_EN_PAPER_REFRESH", lines)
        text = path.read_text(encoding="utf-8")
        status.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "contains_stage42_eo": "Stage42-EO Post-EM/EN Paper Package Refresh" in text,
                "contains_source_blocker": "Official links are not license acceptance" in text,
                "contains_floor_blocker": "floor_free_neural_deployable" in text
                and "global_floor_removal_allowed" in text,
                "contains_partial_t50": "validation-backed t50 floor relaxation" in text,
                "contains_proximity_guard": "proximity guard for safety-sensitive reporting" in text,
                "contains_non_claims": "Still forbidden: true 3D" in text and "Stage5C execution" in text and "SMC readiness" in text,
            }
        )
    return status


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["paper_refresh_summary"]
    source = summary["official_source_link_boundary"]
    floor = summary["floor_removability_boundary"]
    paper = summary["paper_verdict"]
    gates = {
        "eg_input_passed": _gate_passed(payload["inputs"]["stage42_eg"], "stage42_eg_gate"),
        "em_input_passed": _gate_passed(payload["inputs"]["stage42_em"], "stage42_em_gate"),
        "en_input_passed": _gate_passed(payload["inputs"]["stage42_en"], "stage42_en_gate"),
        "paper_files_refreshed": all(row["contains_stage42_eo"] for row in payload["paper_file_status"]),
        "source_blocker_preserved": source["conversion_ready_now"] == 0
        and source["auto_download_allowed_now"] == 0
        and paper["source_conversion_claim_allowed"] is False,
        "floor_free_neural_blocked": floor["floor_free_neural_deployable"] is False
        and paper["floor_free_neural_deployable"] is False,
        "global_floor_removal_blocked": floor["global_floor_removal_allowed"] is False
        and paper["global_floor_removal_allowed"] is False,
        "teacher_context_required": floor["teacher_floor_rollout_context_removal_allowed"] is False,
        "partial_t50_relaxation_recorded": floor["safe_partial_floor_relaxation_available"] is True
        and len(floor["partial_relaxation_components"]) >= 1,
        "proximity_guard_recorded": floor["proximity_guard_required_for_safety_claim"] is True,
        "no_metric_seconds_overclaim": paper["metric_seconds_claim_allowed"] is False,
        "foundation_overclaim_blocked": paper["foundation_claim_allowed"] is False,
        "stage5c_false": paper["stage5c_execution_allowed"] is False,
        "smc_false": paper["smc_allowed"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_eo_post_em_en_paper_refresh_pass" if passed == total else "stage42_eo_post_em_en_paper_refresh_partial"
    return {"source": payload.get("source", "unknown"), "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["paper_refresh_summary"]
    gate = payload["stage42_eo_gate"]
    lines = [
        "# Stage42-EO Post-EM/EN Paper Package Refresh",
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
        "## Refresh Content",
        "",
        *_refresh_lines(summary),
        "",
        "## Claim Matrix",
        "",
        "| claim | status | main claim allowed | evidence | boundary |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in summary["paper_claim_matrix"]:
        lines.append(
            f"| `{row['claim']}` | `{row['status']}` | {row['main_claim_allowed']} | {row['evidence']} | {row['boundary']} |"
        )
    lines += [
        "",
        "## Paper File Status",
        "",
        "| file | refreshed | source blocker | floor blocker | partial t50 | proximity guard | non-claims |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["paper_file_status"]:
        lines.append(
            f"| `{row['path']}` | {row['contains_stage42_eo']} | {row['contains_source_blocker']} | {row['contains_floor_blocker']} | {row['contains_partial_t50']} | {row['contains_proximity_guard']} | {row['contains_non_claims']} |"
        )
    lines += [
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_eo_gate"]
    return [
        "# Stage42-EO Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload["paper_refresh_summary"])
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY, WORK_SUMMARY]:
        _replace_section(path, "STAGE42_EO_POST_EM_EN_PAPER_REFRESH", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-EO post-EM/EN paper package refresh"
    state["current_verdict"] = payload["stage42_eo_gate"]["verdict"]
    state["stage42_eo_post_em_en_paper_refresh"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_eo_gate"]["verdict"],
        "gates": f"{payload['stage42_eo_gate']['passed']}/{payload['stage42_eo_gate']['total']}",
        "summary": payload["paper_refresh_summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_post_en_paper_refresh(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    eg = read_json(EG_JSON, {})
    em = read_json(EM_JSON, {})
    en = read_json(EN_JSON, {})
    summary = _summary(eg, em, en)
    paper_status = _refresh_paper_files(summary)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-EO",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([EG_JSON, EM_JSON, EN_JSON]),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "stage42_eg": {"stage42_eg_gate": eg.get("stage42_eg_gate", {})},
            "stage42_em": {"stage42_em_gate": em.get("stage42_em_gate", {})},
            "stage42_en": {"stage42_en_gate": en.get("stage42_en_gate", {})},
        },
        "paper_refresh_summary": summary,
        "paper_file_status": paper_status,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_eo_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_post_en_paper_refresh()
