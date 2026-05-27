from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
EQ_JSON = OUT_DIR / "sequence_graph_context_router_stage42.json"
EL_JSON = OUT_DIR / "context_gain_router_stage42.json"
AR_JSON = OUT_DIR / "source_level_sequence_context_stage42.json"
AS_JSON = OUT_DIR / "source_level_graph_context_stage42.json"
DA_JSON = OUT_DIR / "next_action_queue_stage42.json"
EM_JSON = OUT_DIR / "official_source_link_audit_stage42.json"
EN_JSON = OUT_DIR / "floor_removability_decision_map_stage42.json"

REPORT_JSON = OUT_DIR / "post_eq_context_claim_refresh_stage42.json"
REPORT_MD = OUT_DIR / "post_eq_context_claim_refresh_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_er_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
TARGET_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
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

SOURCE = "fresh_post_eq_context_claim_refresh"
MEANINGFUL_CONTEXT_THRESHOLD = 0.01

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-ER 是 post-EQ paper/action refresh，不下载、不转换、不训练、不调 threshold。",
    "Stage42-EQ fresh result shows sequence+graph context router did not provide a meaningful independent increment over baseline-family protected control.",
    "future waypoints / endpoints 只作为 supervised/evaluation labels，不作为 inference input。",
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
    "stage5c_executed": False,
    "smc_enabled": False,
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


def _eq_metric(eq: Mapping[str, Any]) -> Mapping[str, Any]:
    return eq.get("summary", {}).get("best_router_test_metric_vs_baseline_family", {})


def _context_evidence_matrix(
    eq: Mapping[str, Any],
    el: Mapping[str, Any],
    ar: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> list[dict[str, Any]]:
    eq_metric = _eq_metric(eq)
    el_metric = el.get("summary", {}).get("best_router_test_metric_vs_baseline_family", {})
    ar_best = ar.get("summary", {}).get("best_variant", "")
    ar_rows = ar.get("sequence_deltas", {})
    as_rows = graph.get("graph_deltas", {})
    return [
        {
            "protocol": "context_gain_router",
            "source": "cached_verified",
            "report": str(EL_JSON),
            "positive_variants": el.get("positive_context_gain_routers", []),
            "best_variant": el.get("best_router", ""),
            "all_delta": el_metric.get("all_improvement"),
            "t50_delta": el_metric.get("t50_improvement"),
            "hard_delta": el_metric.get("hard_failure_improvement"),
            "verdict": el.get("summary", {}).get("context_increment_verdict", "unknown"),
        },
        {
            "protocol": "sequence_residual_context",
            "source": "cached_verified",
            "report": str(AR_JSON),
            "positive_variants": ar.get("positive_sequence_context_variants", []),
            "best_variant": ar_best,
            "all_delta": ar_rows.get(ar_best, {}).get("delta_vs_baseline_family_only", {}).get("all_improvement"),
            "t50_delta": ar_rows.get(ar_best, {}).get("delta_vs_baseline_family_only", {}).get("t50_improvement"),
            "hard_delta": ar_rows.get(ar_best, {}).get("delta_vs_baseline_family_only", {}).get("hard_failure_improvement"),
            "verdict": ar.get("summary", {}).get("sequence_context_verdict", "unknown"),
        },
        {
            "protocol": "graph_residual_context",
            "source": "cached_verified",
            "report": str(AS_JSON),
            "positive_variants": graph.get("positive_graph_context_variants", []),
            "best_variant": "graph_history_goal",
            "all_delta": as_rows.get("graph_history_goal", {}).get("delta_vs_baseline_family_only", {}).get("all_improvement"),
            "t50_delta": as_rows.get("graph_history_goal", {}).get("delta_vs_baseline_family_only", {}).get("t50_improvement"),
            "hard_delta": as_rows.get("graph_history_goal", {}).get("delta_vs_baseline_family_only", {}).get("hard_failure_improvement"),
            "verdict": graph.get("summary", {}).get("graph_context_verdict", "unknown"),
        },
        {
            "protocol": "sequence_graph_context_router",
            "source": "fresh_run",
            "report": str(EQ_JSON),
            "positive_variants": eq.get("positive_sequence_graph_context_routers", []),
            "best_variant": eq.get("best_router", ""),
            "all_delta": eq_metric.get("all_improvement"),
            "t50_delta": eq_metric.get("t50_improvement"),
            "hard_delta": eq_metric.get("hard_failure_improvement"),
            "verdict": eq.get("summary", {}).get("sequence_graph_increment_verdict", "unknown"),
        },
    ]


def _context_claim_decision(matrix: list[Mapping[str, Any]]) -> dict[str, Any]:
    positive = [
        row
        for row in matrix
        if row.get("positive_variants")
        or max(
            float(row.get("all_delta") or 0.0),
            float(row.get("t50_delta") or 0.0),
            float(row.get("hard_delta") or 0.0),
        )
        > MEANINGFUL_CONTEXT_THRESHOLD
    ]
    return {
        "source": "fresh_decision_from_stage42_eq_plus_cached_context_evidence",
        "meaningful_context_threshold": MEANINGFUL_CONTEXT_THRESHOLD,
        "independent_context_main_claim_allowed": bool(positive),
        "positive_protocols": [row["protocol"] for row in positive],
        "closed_protocols": [row["protocol"] for row in matrix if row not in positive],
        "decision": (
            "context_independent_main_claim_supported"
            if positive
            else "close_current_shallow_sequence_graph_context_protocol"
        ),
        "paper_wording": (
            "Scene/goal/neighbor/sequence/graph context can be reported as an auxiliary or diagnostic probe under the current protocols; "
            "it must not be written as an independent main contribution until a future protocol produces validation-selected, "
            "bootstrap-positive all/t50/hard lift with easy preservation."
        )
        if not positive
        else (
            "A context protocol has crossed the meaningful threshold; restrict the claim to that protocol and still keep raw-frame/"
            "dataset-local 2.5D boundaries."
        ),
    }


def _updated_actions(da: Mapping[str, Any], decision: Mapping[str, Any], em: Mapping[str, Any], en: Mapping[str, Any]) -> list[dict[str, Any]]:
    old_actions = da.get("next_actions", [])
    retained = []
    for row in old_actions:
        if row.get("id") == "DA-2":
            retained.append(
                {
                    **row,
                    "status": "closed_negative_fresh_run",
                    "closed_by": str(REPORT_JSON),
                    "closure_reason": decision["decision"],
                    "next_valid_variant": "Do not repeat shallow residual/router context. Replace target with joint occupancy, interaction-constraint, or new data/source support.",
                }
            )
        else:
            retained.append(row)
    source_summary = em.get("summary", {})
    floor_summary = en.get("summary", {})
    new_actions = [
        {
            "id": "ER-1",
            "title": "Use official terms confirmation to unlock independent external sources",
            "priority": 105,
            "status": "not_run_next_action",
            "why_now": "Context protocols are closed negative; the largest remaining publication blocker is legally converted independent source diversity.",
            "evidence": [str(EM_JSON), str(OUT_DIR / "source_terms_gap_audit_stage42.md")],
            "next_commands": [
                ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
                ".venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py",
            ],
            "requires_user_or_external_state": True,
            "blocked_claim_until_done": "broader external generalization and metric/time subset claims",
            "success_gate": "user-confirmed terms/local path/source identity followed by conversion/no-leakage/source-CV/final test-once.",
            "current_blocker_numbers": {
                "auto_download_allowed_now": source_summary.get("auto_download_allowed_now"),
                "conversion_ready_now": source_summary.get("conversion_ready_now"),
                "manual_terms_required_targets": source_summary.get("manual_terms_required_targets"),
            },
        },
        {
            "id": "ER-2",
            "title": "Replace shallow context probes with joint occupancy or interaction-constraint target",
            "priority": 96,
            "status": "not_run_next_action",
            "why_now": "EQ/EL/AR/AS close shallow context routing/residual protocols, but interaction may still help if trained against a physical group/occupancy target.",
            "evidence": [str(EQ_JSON), str(AR_JSON), str(AS_JSON), str(OUT_DIR / "group_consistency_contribution_audit_stage42.md")],
            "next_commands": [
                ".venv-pytorch/bin/python run_stage42_group_consistency_contribution_audit.py",
                ".venv-pytorch/bin/python run_stage42_full_waypoint_proximity_occupancy_loss_repair.py",
            ],
            "requires_user_or_external_state": False,
            "blocked_claim_until_done": "scene/goal/neighbor/interaction as an independent main contribution",
            "success_gate": "new target beats protected baseline-family/floor on all/t50/hard and improves proximity/occupancy without easy degradation.",
        },
        {
            "id": "ER-3",
            "title": "Map safe slice-level floor relaxation without global floor removal",
            "priority": 88,
            "status": "not_run_next_action",
            "why_now": "EN blocks global floor removal but permits validation-backed t50 slice relaxation.",
            "evidence": [str(EN_JSON), str(OUT_DIR / "t50_floor_relaxability_repair_stage42.md")],
            "next_commands": [
                ".venv-pytorch/bin/python run_stage42_t50_floor_relaxability_repair.py",
                ".venv-pytorch/bin/python run_stage42_weak_slice_guard.py",
            ],
            "requires_user_or_external_state": False,
            "blocked_claim_until_done": "floor-reduction evidence beyond narrow t50 slices",
            "success_gate": "slice-level relaxation remains validation-backed, source/horizon scoped, easy-safe, and never advertised as global floor-free neural.",
            "partial_relaxation_components": floor_summary.get("partial_relaxation_components", []),
        },
    ]
    return sorted(new_actions + retained, key=lambda row: int(row.get("priority", 0)), reverse=True)


def _summary(eq: Mapping[str, Any], el: Mapping[str, Any], ar: Mapping[str, Any], graph: Mapping[str, Any], da: Mapping[str, Any], em: Mapping[str, Any], en: Mapping[str, Any]) -> dict[str, Any]:
    matrix = _context_evidence_matrix(eq, el, ar, graph)
    decision = _context_claim_decision(matrix)
    actions = _updated_actions(da, decision, em, en)
    eq_metric = _eq_metric(eq)
    return {
        "source": SOURCE,
        "context_evidence_matrix": matrix,
        "context_claim_decision": decision,
        "best_eq_metric": {
            "best_router": eq.get("best_router"),
            "all_delta": eq_metric.get("all_improvement"),
            "t50_delta": eq_metric.get("t50_improvement"),
            "t100_raw_delta": eq_metric.get("t100_raw_frame_diagnostic_improvement"),
            "hard_delta": eq_metric.get("hard_failure_improvement"),
            "easy_delta": eq_metric.get("easy_degradation"),
        },
        "updated_next_actions": actions,
        "paper_verdict": {
            "post_eq_refresh_complete": True,
            "independent_context_main_claim_allowed": decision["independent_context_main_claim_allowed"],
            "source_legal_blocker_still_primary": em.get("summary", {}).get("conversion_ready_now", 0) == 0,
            "floor_free_neural_deployable": en.get("summary", {}).get("floor_free_neural_deployable", False),
            "global_floor_removal_allowed": en.get("summary", {}).get("global_floor_removal_allowed", False),
            "metric_seconds_claim_allowed": False,
            "stage5c_execution_allowed": False,
            "smc_allowed": False,
            "recommended_next_primary_action": actions[0]["id"] if actions else "",
        },
    }


def _build_payload() -> dict[str, Any]:
    eq = read_json(EQ_JSON, {})
    el = read_json(EL_JSON, {})
    ar = read_json(AR_JSON, {})
    graph = read_json(AS_JSON, {})
    da = read_json(DA_JSON, {})
    em = read_json(EM_JSON, {})
    en = read_json(EN_JSON, {})
    payload = {
        "source": SOURCE,
        "stage": "Stage42-ER Post-EQ Context Claim Refresh",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(EQ_JSON), str(EL_JSON), str(AR_JSON), str(AS_JSON), str(DA_JSON), str(EM_JSON), str(EN_JSON)]),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "stage42_eq": {"source": eq.get("source"), "verdict": eq.get("stage42_eq_gate", {}).get("verdict")},
            "stage42_el": {"source": el.get("source"), "verdict": el.get("stage42_el_gate", {}).get("verdict")},
            "stage42_ar": {"source": ar.get("source"), "verdict": ar.get("stage42_ar_gate", {}).get("verdict")},
            "stage42_as": {"source": graph.get("source"), "verdict": graph.get("stage42_as_gate", {}).get("verdict")},
            "stage42_da": {"source": da.get("source"), "verdict": da.get("stage42_da_gate", {}).get("verdict")},
            "stage42_em": {"source": em.get("source"), "verdict": em.get("stage42_em_gate", {}).get("verdict")},
            "stage42_en": {"source": en.get("source"), "verdict": en.get("stage42_en_gate", {}).get("verdict")},
        },
        "paper_refresh_summary": _summary(eq, el, ar, graph, da, em, en),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["paper_file_status"] = _refresh_paper_files(payload["paper_refresh_summary"])
    payload["stage42_er_gate"] = _gate(payload)
    return payload


def _paper_section_lines(summary: Mapping[str, Any]) -> list[str]:
    decision = summary["context_claim_decision"]
    best = summary["best_eq_metric"]
    return [
        "## Stage42-ER Post-EQ Context Claim Refresh",
        "",
        "- source: `fresh_post_eq_context_claim_refresh`",
        "- role: integrate the fresh Stage42-EQ sequence+graph router result into the paper/gap/action package.",
        "- This is not new training, conversion, download, or threshold tuning.",
        f"- Stage42-EQ best router: `{best['best_router']}` with all/t50/t100raw/hard deltas `{_pct(best['all_delta'])}` / `{_pct(best['t50_delta'])}` / `{_pct(best['t100_raw_delta'])}` / `{_pct(best['hard_delta'])}`.",
        f"- context decision: `{decision['decision']}`.",
        f"- independent context main claim allowed: `{decision['independent_context_main_claim_allowed']}`.",
        f"- closed protocols: `{decision['closed_protocols']}`.",
        "- Paper wording: sequence/graph/goal/neighbor context remains auxiliary or diagnostic under current protocols, not an independent main contribution.",
        "- Next primary route: source/legal/time conversion and stronger joint occupancy / interaction-constraint targets, not repeating shallow context routers.",
        "- Boundary: raw-frame/dataset-local 2.5D only; no true 3D, no foundation, no metric/seconds, no Stage5C, no SMC.",
    ]


def _refresh_paper_files(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    lines = _paper_section_lines(summary)
    status = []
    for path in PAPER_FILES:
        if path.exists():
            _replace_section(path, "STAGE42_ER_POST_EQ_CONTEXT_CLAIM_REFRESH", lines)
            text = path.read_text(encoding="utf-8")
            status.append(
                {
                    "path": str(path),
                    "updated": True,
                    "contains_stage42_er": "STAGE42_ER_POST_EQ_CONTEXT_CLAIM_REFRESH" in text,
                    "contains_context_decision": "independent context main claim allowed" in text,
                    "contains_boundaries": "no true 3D" in text and "no Stage5C" in text and "no SMC" in text,
                }
            )
        else:
            status.append({"path": str(path), "updated": False, "missing": True})
    return status


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["paper_refresh_summary"]
    decision = summary["context_claim_decision"]
    verdict = summary["paper_verdict"]
    files = payload["paper_file_status"]
    claim = payload["claim_boundary"]
    actions = summary["updated_next_actions"]
    gates = {
        "eq_input_present": payload["inputs"]["stage42_eq"]["verdict"] == "stage42_eq_sequence_graph_context_router_pass",
        "context_matrix_has_four_protocols": len(summary["context_evidence_matrix"]) >= 4,
        "fresh_eq_included": any(row["protocol"] == "sequence_graph_context_router" and row["source"] == "fresh_run" for row in summary["context_evidence_matrix"]),
        "context_decision_recorded": decision["decision"] in {"close_current_shallow_sequence_graph_context_protocol", "context_independent_main_claim_supported"},
        "negative_context_claim_bounded": decision["independent_context_main_claim_allowed"] is False,
        "da2_closed_negative": any(row.get("id") == "DA-2" and row.get("status") == "closed_negative_fresh_run" for row in actions),
        "new_primary_source_action_added": any(row.get("id") == "ER-1" and row.get("requires_user_or_external_state") for row in actions),
        "paper_files_refreshed": all(row.get("contains_stage42_er") for row in files if row.get("updated")),
        "paper_boundaries_refreshed": all(row.get("contains_boundaries") for row in files if row.get("updated")),
        "source_legal_blocker_preserved": verdict["source_legal_blocker_still_primary"] is True,
        "no_floor_free_overclaim": verdict["floor_free_neural_deployable"] is False and verdict["global_floor_removal_allowed"] is False,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": "fresh_run",
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage42_er_post_eq_context_claim_refresh_pass" if passed == total else "stage42_er_post_eq_context_claim_refresh_partial",
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["paper_refresh_summary"]
    decision = summary["context_claim_decision"]
    best = summary["best_eq_metric"]
    lines = [
        "# Stage42-ER Post-EQ Context Claim Refresh",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_er_gate']['passed']} / {payload['stage42_er_gate']['total']}`",
        f"- verdict: `{payload['stage42_er_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Context Claim Decision",
        "",
        f"- decision: `{decision['decision']}`",
        f"- independent_context_main_claim_allowed: `{decision['independent_context_main_claim_allowed']}`",
        f"- positive_protocols: `{decision['positive_protocols']}`",
        f"- closed_protocols: `{decision['closed_protocols']}`",
        f"- paper_wording: {decision['paper_wording']}",
        "",
        "## Fresh EQ Summary",
        "",
        f"- best_router: `{best['best_router']}`",
        f"- all/t50/t100raw/hard delta: `{_pct(best['all_delta'])}` / `{_pct(best['t50_delta'])}` / `{_pct(best['t100_raw_delta'])}` / `{_pct(best['hard_delta'])}`",
        "",
        "## Context Evidence Matrix",
        "",
        "| protocol | source | best/positive | all delta | t50 delta | hard delta | verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary["context_evidence_matrix"]:
        lines.append(
            f"| `{row['protocol']}` | `{row['source']}` | `{row.get('best_variant')}` / `{row.get('positive_variants')}` | "
            f"{_pct(row.get('all_delta'))} | {_pct(row.get('t50_delta'))} | {_pct(row.get('hard_delta'))} | `{row.get('verdict')}` |"
        )
    lines.extend(
        [
            "",
            "## Updated Next Actions",
            "",
            "| id | priority | status | title |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in summary["updated_next_actions"][:8]:
        lines.append(f"| `{row['id']}` | {row['priority']} | `{row['status']}` | {row['title']} |")
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_er_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_er_gate"]
    return [
        "# Stage42-ER Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _readme_lines(summary: Mapping[str, Any], gate: Mapping[str, Any]) -> list[str]:
    decision = summary["context_claim_decision"]
    best = summary["best_eq_metric"]
    return [
        "## Stage42-ER Post-EQ Context Claim Refresh",
        "",
        "- source: `fresh_post_eq_context_claim_refresh`",
        "- role: updates paper/action boundaries after the fresh Stage42-EQ sequence+graph router result.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- Stage42-EQ best all/t50/t100raw/hard delta: `{_pct(best['all_delta'])}` / `{_pct(best['t50_delta'])}` / `{_pct(best['t100_raw_delta'])}` / `{_pct(best['hard_delta'])}`.",
        f"- context decision: `{decision['decision']}`; independent context main claim allowed `{decision['independent_context_main_claim_allowed']}`.",
        "- DA-2 is closed negative under the current shallow sequence/graph residual/router protocols.",
        "- New priority: source/legal/time conversion plus stronger joint occupancy or interaction-constraint targets.",
        "- Boundary: raw-frame/dataset-local 2.5D only; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _readme_lines(payload["paper_refresh_summary"], payload["stage42_er_gate"])
    for path in [README_RESULTS, M3W_README, TARGET_SUMMARY, WORK_SUMMARY]:
        _replace_section(path, "STAGE42_ER_POST_EQ_CONTEXT_CLAIM_REFRESH", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-ER post-EQ context claim refresh"
    state["current_verdict"] = payload["stage42_er_gate"]["verdict"]
    state["stage42_er_post_eq_context_claim_refresh"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_er_gate"]["verdict"],
        "gates": f"{payload['stage42_er_gate']['passed']}/{payload['stage42_er_gate']['total']}",
        "summary": payload["paper_refresh_summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_post_eq_context_claim_refresh(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_post_eq_context_claim_refresh()
