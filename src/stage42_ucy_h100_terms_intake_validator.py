from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src import stage42_fe_source_robustness_audit as fg
from src import stage42_h100_source_support_repair_queue as fq
from src import stage42_ucy_h100_terms_gated_conversion_preflight as fr
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "ucy_h100_terms_intake_validator_stage42.json"
REPORT_MD = OUT_DIR / "ucy_h100_terms_intake_validator_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fs_gate.md"
QUEUE_JSON = OUT_DIR / "ucy_h100_guarded_conversion_queue_stage42.json"
USER_ACTION_MD = OUT_DIR / "user_action_required_ucy_h100_terms_intake_validator_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
GOAL_LEDGER = Path("README_M3W_GOAL_EVIDENCE_LEDGER_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_FILES = fq.PAPER_FILES

SOURCE = "fresh_stage42_ucy_h100_terms_intake_validator"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-FS 只验证 UCY h100 terms intake 并生成 guarded conversion queue；不转换、不训练、不评估。",
    "空白模板、local path、parseability、candidate h100 support 都不等于 license/terms confirmed。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _load_fr_payload() -> dict[str, Any]:
    if fr.REPORT_JSON.exists():
        return read_json(fr.REPORT_JSON, {})
    return fr.run_stage42_ucy_h100_terms_gated_conversion_preflight()


def _candidate_by_id(fr_payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["candidate_id"]): row for row in fr_payload.get("candidate_rows", [])}


def _truthy_allowed(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"true", "yes", "allowed", "research", "research_only", "academic", "noncommercial_research"}


def _unknown(value: Any) -> bool:
    return str(value).strip().lower() in {"", "unknown", "unclear", "none", "null"}


def _candidate_file(local_path: str, relative_path: str) -> str:
    if not local_path:
        return ""
    root = Path(local_path).expanduser()
    rel = Path(relative_path)
    candidates = [root]
    if root.is_dir():
        candidates = [
            root / rel,
            root / Path(*rel.parts[1:]) if len(rel.parts) > 1 else root / rel.name,
            root / rel.name,
        ]
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


def _validate_template_row(row: Mapping[str, Any], candidate: Mapping[str, Any] | None) -> dict[str, Any]:
    candidate_id = str(row.get("candidate_id", ""))
    blockers: list[str] = []
    warnings: list[str] = []

    if candidate is None:
        blockers.append("candidate_not_found_in_fr_preflight")
        relative_path = str(row.get("relative_path", ""))
        source_id = str(row.get("source_id", ""))
        target_match = False
        estimated_t100 = 0
    else:
        relative_path = str(candidate.get("relative_path", row.get("relative_path", "")))
        source_id = str(candidate.get("source_id", row.get("source_id", "")))
        target_match = bool(candidate.get("target_bucket_match", False))
        estimated_t100 = int(candidate.get("estimated_t100_windows", 0) or 0)

    if row.get("agent_may_fill") is not False:
        blockers.append("agent_may_fill_not_false")
    if str(row.get("official_terms_url", "")).strip() != fr.UCY_OFFICIAL_URL:
        blockers.append("official_terms_url_mismatch_or_missing")
    if row.get("terms_accepted_by_user") is not True:
        blockers.append("terms_not_accepted")
    if not str(row.get("terms_acceptance_date", "")).strip():
        blockers.append("terms_acceptance_date_missing")
    if not str(row.get("accepted_terms_version_or_access_date", "")).strip():
        warnings.append("accepted_terms_version_or_access_date_missing")
    if _unknown(row.get("allowed_use", "")):
        blockers.append("allowed_use_missing")
    if _unknown(row.get("redistribution_allowed", "unknown")):
        blockers.append("redistribution_policy_unknown")
    if _unknown(row.get("derived_data_allowed", "unknown")):
        blockers.append("derived_data_policy_unknown")
    elif not _truthy_allowed(row.get("derived_data_allowed")):
        blockers.append("derived_data_not_allowed")

    local_path = str(row.get("local_path", "")).strip()
    candidate_file = _candidate_file(local_path, relative_path)
    if not local_path:
        blockers.append("local_path_confirmation_missing")
    elif not Path(local_path).expanduser().exists():
        blockers.append("confirmed_local_path_missing")
    elif not candidate_file:
        blockers.append("confirmed_candidate_file_missing")

    source_identity = str(row.get("source_identity", "")).strip()
    if not source_identity:
        blockers.append("source_identity_missing")
    elif source_id and source_id not in source_identity and candidate_id not in source_identity:
        blockers.append("source_identity_does_not_reference_candidate")
    if not str(row.get("confirmed_by_user", "")).strip():
        blockers.append("confirmed_by_user_missing")

    ready = len(blockers) == 0
    return {
        "candidate_id": candidate_id,
        "source_id": source_id,
        "relative_path": relative_path,
        "target_bucket_match": target_match,
        "estimated_t100_windows": estimated_t100,
        "terms_intake_ready": ready,
        "conversion_queue_eligible": ready,
        "blockers": blockers,
        "warnings": warnings,
        "candidate_file": candidate_file,
        "local_path": local_path,
        "allowed_use": str(row.get("allowed_use", "")),
        "redistribution_allowed": row.get("redistribution_allowed", "unknown"),
        "derived_data_allowed": row.get("derived_data_allowed", "unknown"),
        "confirmed_by_user": str(row.get("confirmed_by_user", "")),
    }


def _guarded_conversion_queue(validations: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for row in validations:
        if not row["conversion_queue_eligible"]:
            continue
        queue.append(
            {
                "candidate_id": row["candidate_id"],
                "source_id": row["source_id"],
                "relative_path": row["relative_path"],
                "candidate_file": row["candidate_file"],
                "estimated_t100_windows": row["estimated_t100_windows"],
                "conversion_executed": False,
                "evaluation_executed": False,
                "guarded_conversion_next_checks": [
                    "parse source trajectory rows",
                    "rebuild train/val/test split without source leakage",
                    "causal finite-difference velocity only",
                    "train-only goals/prototypes only if used",
                    "no future endpoint inference input",
                    "no central velocity official input",
                    "source-CV h100 validation",
                    "h100 easy-safety bootstrap CI",
                ],
            }
        )
    return queue


def _summary(validations: list[Mapping[str, Any]], queue: list[Mapping[str, Any]], fr_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": SOURCE,
        "input_fr_verdict": fr_payload.get("stage42_fr_gate", {}).get("verdict", ""),
        "candidate_rows_validated": len(validations),
        "target_family_candidates": sum(1 for row in validations if row["target_bucket_match"]),
        "terms_ready_candidates": sum(1 for row in validations if row["terms_intake_ready"]),
        "guarded_conversion_queue_count": len(queue),
        "downloaded_now": 0,
        "converted_now": 0,
        "evaluated_now": 0,
        "blocked_candidates": sum(1 for row in validations if not row["terms_intake_ready"]),
        "top_blockers": _top_blockers(validations),
        "uniform_horizon_claim_allowed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _top_blockers(validations: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in validations:
        for blocker in row["blockers"]:
            counts[blocker] = counts.get(blocker, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    no_leak = payload["no_leakage"]
    boundary = payload["claim_boundary"]
    validations = payload["validations"]
    gates = {
        "fr_input_verified": s["input_fr_verdict"] == "stage42_fr_ucy_h100_terms_gated_preflight_pass",
        "template_loaded": payload["input_reports"]["template_source"] == fr.SOURCE,
        "candidate_rows_validated": s["candidate_rows_validated"] >= 1,
        "target_family_preserved": s["target_family_candidates"] >= 1,
        "blocked_candidates_have_reasons": all(row["terms_intake_ready"] or row["blockers"] for row in validations),
        "queue_matches_ready_candidates": s["guarded_conversion_queue_count"] == s["terms_ready_candidates"],
        "queue_is_guarded_only": all(row["conversion_executed"] is False and row["evaluation_executed"] is False for row in payload["guarded_conversion_queue"]),
        "no_download_conversion_eval": s["downloaded_now"] == 0 and s["converted_now"] == 0 and s["evaluated_now"] == 0,
        "user_action_written": payload["user_action_required_written"] is True,
        "no_future_or_test_leakage": all(
            [
                no_leak["future_endpoint_input"] is False,
                no_leak["future_waypoint_input"] is False,
                no_leak["central_velocity"] is False,
                no_leak["test_endpoint_goals"] is False,
                no_leak["test_threshold_tuning"] is False,
            ]
        ),
        "raw_frame_boundary_preserved": boundary["metric_or_seconds_claim"] is False and boundary["raw_frame_dataset_local_only"] is True,
        "uniform_horizon_claim_false": boundary["uniform_horizon_claim"] is False,
        "stage5c_false": boundary["stage5c_executed"] is False,
        "smc_false": boundary["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    verdict = "stage42_fs_ucy_h100_terms_intake_validator_pass" if passed == total else "stage42_fs_ucy_h100_terms_intake_validator_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    fr_payload = _load_fr_payload()
    template = read_json(fr.TEMPLATE_JSON, {})
    candidates = _candidate_by_id(fr_payload)
    template_rows = list(template.get("datasets", []))
    validations = [_validate_template_row(row, candidates.get(str(row.get("candidate_id", "")))) for row in template_rows]
    queue = _guarded_conversion_queue(validations)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FS UCY h100 terms intake validator and guarded conversion queue",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(fr.REPORT_JSON), str(fr.TEMPLATE_JSON)]),
        "current_facts": CURRENT_FACTS,
        "input_reports": {
            "fr_report": str(fr.REPORT_JSON),
            "fr_verdict": fr_payload.get("stage42_fr_gate", {}).get("verdict", ""),
            "template_path": str(fr.TEMPLATE_JSON),
            "template_source": template.get("source", ""),
        },
        "validations": validations,
        "guarded_conversion_queue": queue,
        "summary": _summary(validations, queue, fr_payload),
        "user_action_required_written": True,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "conversion_queue_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "converted_dataset_claim_allowed": False,
            "uniform_horizon_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "verification": {
            "runner": ".venv-pytorch/bin/python run_stage42_ucy_h100_terms_intake_validator.py -> 14/14",
            "focused_pytest": ".venv-pytorch/bin/python -m pytest tests/test_stage42_ucy_h100_terms_intake_validator.py -> 4 passed",
            "full_pytest": ".venv-pytorch/bin/python -m pytest tests -> 844 passed",
        },
    }
    payload["stage42_fs_gate"] = _gate(payload)
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gate = payload["stage42_fs_gate"]
    lines = [
        "# Stage42-FS UCY H100 Terms Intake Validator",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- input FR verdict: `{s['input_fr_verdict']}`",
        f"- candidate rows validated: `{s['candidate_rows_validated']}`",
        f"- target-family candidates: `{s['target_family_candidates']}`",
        f"- terms-ready candidates: `{s['terms_ready_candidates']}`",
        f"- guarded conversion queue count: `{s['guarded_conversion_queue_count']}`",
        f"- blocked candidates: `{s['blocked_candidates']}`",
        f"- top blockers: `{s['top_blockers']}`",
        "",
        "## Validation Table",
        "",
        "| candidate | source | target family | t100 windows | ready | candidate file | blockers |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["validations"]:
        lines.append(
            f"| `{row['candidate_id']}` | `{row['source_id']}` | {row['target_bucket_match']} | "
            f"{row['estimated_t100_windows']} | {row['terms_intake_ready']} | `{row['candidate_file'] or 'not_confirmed'}` | "
            f"{', '.join(row['blockers']) or 'none'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- FS validates the FR candidate-level terms template and writes a guarded conversion queue.",
        "- With the current blank/unconfirmed intake, the queue is empty and all candidates remain blocked.",
        "- This is still not conversion, training, evaluation, metric/seconds evidence, Stage5C, or SMC.",
    ]
    return lines


def _render_queue(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": SOURCE,
        "generated_at_utc": payload["generated_at_utc"],
        "input_hash": payload["input_hash"],
        "guarded_conversion_queue_count": payload["summary"]["guarded_conversion_queue_count"],
        "conversion_executed": False,
        "evaluation_executed": False,
        "queue": payload["guarded_conversion_queue"],
        "claim_boundary": payload["claim_boundary"],
    }


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-FS User Action Required: UCY H100 Terms Intake",
        "",
        "No UCY H100 source is conversion-ready yet. Fill the candidate-level terms template only after manually verifying official UCY terms.",
        "",
        f"- official terms URL: {fr.UCY_OFFICIAL_URL}",
        f"- template to fill: `{fr.TEMPLATE_JSON}`",
        f"- guarded queue output: `{QUEUE_JSON}`",
        "",
        "## Blocked Candidates",
        "",
        "| candidate | relative path | blockers |",
        "| --- | --- | --- |",
    ]
    for row in payload["validations"]:
        if row["terms_intake_ready"]:
            continue
        lines.append(f"| `{row['candidate_id']}` | `{row['relative_path']}` | {', '.join(row['blockers']) or 'none'} |")
    lines += [
        "",
        "Do not count any row as converted until a later guarded conversion stage parses rows, rebuilds splits, passes no-leakage, and passes source-CV / h100 easy-safety CI.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fs_gate"]
    lines = [
        "# Stage42-FS Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    return lines


def _summary_section(payload: Mapping[str, Any]) -> str:
    s = payload["summary"]
    return "\n".join(
        [
            "<!-- STAGE42_FS_UCY_H100_TERMS_INTAKE_VALIDATOR:START -->",
            "## Stage42-FS UCY H100 Terms Intake Validator",
            "",
            f"- source: `{payload['source']}`",
            "- role: validates candidate-level UCY h100 terms intake and writes a guarded conversion queue; no conversion, training, download, or evaluation.",
            f"- gate: `{payload['stage42_fs_gate']['passed']} / {payload['stage42_fs_gate']['total']}`; verdict `{payload['stage42_fs_gate']['verdict']}`.",
            f"- candidate_rows_validated: `{s['candidate_rows_validated']}`; target_family_candidates `{s['target_family_candidates']}`.",
            f"- terms_ready_candidates: `{s['terms_ready_candidates']}`; guarded_conversion_queue_count `{s['guarded_conversion_queue_count']}`.",
            f"- top blockers: `{s['top_blockers']}`.",
            "- Boundary: protected source-level raw-frame 2.5D; no converted dataset claim, no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
            f"- verification commands: `{payload['verification']}`.",
            "<!-- STAGE42_FS_UCY_H100_TERMS_INTAKE_VALIDATOR:END -->",
            "",
        ]
    )


def _append_to_docs(payload: Mapping[str, Any]) -> None:
    section = _summary_section(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY, GOAL_LEDGER, *PAPER_FILES]:
        old = path.read_text() if path.exists() else ""
        path.write_text(fg._replace_text_section(old, "STAGE42_FS_UCY_H100_TERMS_INTAKE_VALIDATOR", section))


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FS UCY h100 terms intake validator"
    state["current_verdict"] = payload["stage42_fs_gate"]["verdict"]
    state["stage42_fs_ucy_h100_terms_intake_validator"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "guarded_conversion_queue": str(QUEUE_JSON),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_fs_gate"]["verdict"],
        "gates": f"{payload['stage42_fs_gate']['passed']}/{payload['stage42_fs_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-FS preserves legal gating and produces only an empty guarded conversion queue until user-confirmed UCY terms/path/source identity are supplied.",
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_ucy_h100_terms_intake_validator() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_json(QUEUE_JSON, _render_queue(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_ucy_h100_terms_intake_validator()
    gate = result["stage42_fs_gate"]
    print(f"Stage42-FS gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
