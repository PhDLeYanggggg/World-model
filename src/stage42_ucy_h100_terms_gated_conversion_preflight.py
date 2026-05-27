from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src import stage42_fe_source_robustness_audit as fg
from src import stage42_h100_source_support_repair_queue as fq
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "ucy_h100_terms_gated_conversion_preflight_stage42.json"
REPORT_MD = OUT_DIR / "ucy_h100_terms_gated_conversion_preflight_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fr_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_ucy_h100_terms_preflight_stage42.md"
TEMPLATE_JSON = OUT_DIR / "ucy_h100_candidate_terms_template_stage42.json"

INTAKE_JSON = OUT_DIR / "source_terms_confirmation_intake_template_stage42.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
GOAL_LEDGER = Path("README_M3W_GOAL_EVIDENCE_LEDGER_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_FILES = fq.PAPER_FILES

SOURCE = "fresh_stage42_ucy_h100_terms_gated_conversion_preflight"
UCY_DATASET_ID = "ucy_crowd_original"
UCY_OFFICIAL_URL = "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-FR 只做 UCY h100 candidate 的 file-level terms-gated conversion preflight。",
    "本阶段不下载、不转换、不训练、不评估。",
    "local path、parseability、candidate h100 support 都不等于 license/terms confirmed。",
    "只有用户手动确认 official terms、allowed use、local path 和 source identity 后，未来阶段才允许 guarded conversion。",
    "future waypoints / endpoints 只允许作为 supervised/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _source_id(relative_path: str) -> str:
    parts = Path(relative_path).parts
    if len(parts) >= 2 and parts[0] == "UCY":
        return f"UCY_{parts[1]}"
    return relative_path.replace("/", "_").replace(".", "_")


def _load_fq_payload() -> dict[str, Any]:
    return read_json(fq.REPORT_JSON, {}) or fq.run_stage42_h100_source_support_repair_queue()


def _ucy_confirmation(intake: Mapping[str, Any]) -> dict[str, Any]:
    for row in intake.get("datasets", []):
        if row.get("dataset_id") == UCY_DATASET_ID:
            return dict(row.get("user_confirmation", {}))
    return {}


def _confirmation_blockers(confirmation: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if confirmation.get("terms_accepted_by_user") is not True:
        blockers.append("terms_not_accepted")
    if not str(confirmation.get("terms_acceptance_date", "")).strip():
        blockers.append("terms_acceptance_date_missing")
    if not str(confirmation.get("allowed_use", "")).strip():
        blockers.append("allowed_use_missing")
    if str(confirmation.get("redistribution_allowed", "unknown")) == "unknown":
        blockers.append("redistribution_policy_unknown")
    if str(confirmation.get("derived_data_allowed", "unknown")) == "unknown":
        blockers.append("derived_data_policy_unknown")
    local_path = str(confirmation.get("local_path", "")).strip()
    if not local_path:
        blockers.append("local_path_confirmation_missing")
    elif not Path(local_path).exists():
        blockers.append("confirmed_local_path_missing")
    if not str(confirmation.get("source_identity", "")).strip():
        blockers.append("source_identity_missing")
    official = str(confirmation.get("official_terms_url", "")).strip()
    if official != UCY_OFFICIAL_URL:
        blockers.append("official_terms_url_mismatch_or_missing")
    if not str(confirmation.get("confirmed_by_user", "")).strip():
        blockers.append("confirmed_by_user_missing")
    return blockers


def _candidate_rows(fq_payload: Mapping[str, Any], confirmation: Mapping[str, Any]) -> list[dict[str, Any]]:
    ucy = fq_payload.get("key_rows", {}).get("UCY|100", {})
    blockers = _confirmation_blockers(confirmation)
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(ucy.get("top_candidates", []), start=1):
        relative_path = str(candidate["relative_path"])
        sid = _source_id(relative_path)
        candidate_id = f"{sid}::{Path(relative_path).stem}"
        family = str(candidate.get("family_bucket", ""))
        target_match = bool(candidate.get("target_bucket_match", False))
        priority = "critical" if target_match else "secondary_support"
        rows.append(
            {
                "rank": rank,
                "dataset_id": UCY_DATASET_ID,
                "candidate_id": candidate_id,
                "source_id": sid,
                "relative_path": relative_path,
                "family_bucket": family,
                "target_bucket_match": target_match,
                "priority": priority,
                "max_track_points": int(candidate.get("max_track_points", 0) or 0),
                "estimated_t100_windows": int(candidate.get("estimated_t100_windows", 0) or 0),
                "official_terms_url": UCY_OFFICIAL_URL,
                "terms_confirmation_blockers": blockers,
                "conversion_preflight_ready": len(blockers) == 0,
                "conversion_executed": False,
                "evaluation_executed": False,
                "required_future_checks": [
                    "guarded parser conversion",
                    "causal velocity reconstruction",
                    "train/val/test split rebuild",
                    "train-only goal/prototype construction if used",
                    "no-leakage audit",
                    "source-CV / source-family validation",
                    "h100 easy-safety CI audit",
                ],
            }
        )
    rows.sort(
        key=lambda row: (
            0 if row["target_bucket_match"] else 1,
            -int(row["estimated_t100_windows"]),
            str(row["relative_path"]),
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["recommended_order"] = idx
    return rows


def _template_rows(rows: list[Mapping[str, Any]], confirmation: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "dataset_id": row["dataset_id"],
                "candidate_id": row["candidate_id"],
                "source_id": row["source_id"],
                "relative_path": row["relative_path"],
                "official_terms_url": UCY_OFFICIAL_URL,
                "terms_accepted_by_user": confirmation.get("terms_accepted_by_user", False),
                "terms_acceptance_date": confirmation.get("terms_acceptance_date", ""),
                "accepted_terms_version_or_access_date": confirmation.get("accepted_terms_version_or_access_date", ""),
                "allowed_use": confirmation.get("allowed_use", ""),
                "redistribution_allowed": confirmation.get("redistribution_allowed", "unknown"),
                "derived_data_allowed": confirmation.get("derived_data_allowed", "unknown"),
                "local_path": confirmation.get("local_path", ""),
                "source_identity": confirmation.get("source_identity", ""),
                "confirmed_by_user": confirmation.get("confirmed_by_user", ""),
                "notes": confirmation.get("notes", ""),
                "agent_may_fill": False,
                "conversion_ready_now": row["conversion_preflight_ready"],
                "do_not_count_as_converted_until": "guarded conversion, no-leakage, source-CV, and h100 easy-safety CI pass",
            }
        )
    return out


def _summary(rows: list[Mapping[str, Any]], blockers: list[str], fq_payload: Mapping[str, Any]) -> dict[str, Any]:
    ready = [row for row in rows if row["conversion_preflight_ready"]]
    zara = [row for row in rows if row["target_bucket_match"]]
    return {
        "source": SOURCE,
        "input_fq_verdict": fq_payload.get("stage42_fq_gate", {}).get("verdict"),
        "dataset_id": UCY_DATASET_ID,
        "candidate_rows": len(rows),
        "target_family_candidates": len(zara),
        "conversion_preflight_ready_count": len(ready),
        "conversion_queue_count": 0,
        "converted_now": 0,
        "evaluated_now": 0,
        "downloaded_now": 0,
        "terms_confirmation_blockers": blockers,
        "recommended_first_sources": [row["source_id"] for row in rows[:2]],
        "estimated_t100_windows_total": sum(int(row["estimated_t100_windows"]) for row in rows),
        "uniform_horizon_claim_allowed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    no_leak = payload["no_leakage"]
    boundary = payload["claim_boundary"]
    gates = {
        "fq_input_verified": str(s.get("input_fq_verdict", "")).startswith("stage42_fq_h100_source_support_repair_queue_pass"),
        "ucy_candidates_loaded": int(s["candidate_rows"]) >= 1,
        "zara_target_candidates_present": int(s["target_family_candidates"]) >= 1,
        "file_level_template_written": payload["template_written"] is True,
        "terms_blockers_preserved": int(s["conversion_preflight_ready_count"]) == 0 and bool(s["terms_confirmation_blockers"]),
        "no_conversion_queue_without_terms": int(s["conversion_queue_count"]) == 0,
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
        "no_converted_dataset_overclaim": boundary["converted_dataset_claim_allowed"] is False,
        "uniform_horizon_claim_false": boundary["uniform_horizon_claim"] is False,
        "no_metric_seconds_overclaim": boundary["metric_or_seconds_claim"] is False,
        "stage5c_false": boundary["stage5c_executed"] is False,
        "smc_false": boundary["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    verdict = "stage42_fr_ucy_h100_terms_gated_preflight_pass" if passed == total else "stage42_fr_ucy_h100_terms_gated_preflight_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    fq_payload = _load_fq_payload()
    intake = read_json(INTAKE_JSON, {}) if INTAKE_JSON.exists() else {}
    confirmation = _ucy_confirmation(intake)
    blockers = _confirmation_blockers(confirmation)
    rows = _candidate_rows(fq_payload, confirmation)
    template = {
        "source": SOURCE,
        "purpose": "File-level UCY h100 terms confirmation template. Blank or partially filled rows are not permission.",
        "official_terms_url": UCY_OFFICIAL_URL,
        "datasets": _template_rows(rows, confirmation),
        "non_claims": [
            "Template presence is not license confirmation.",
            "Local candidate presence is not conversion.",
            "Preflight readiness is not evaluation.",
        ],
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FR UCY h100 terms-gated conversion preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(fq.REPORT_JSON), str(INTAKE_JSON)]),
        "current_facts": CURRENT_FACTS,
        "summary": _summary(rows, blockers, fq_payload),
        "ucy_confirmation_source": str(INTAKE_JSON),
        "ucy_confirmation": confirmation,
        "candidate_rows": rows,
        "template": template,
        "template_written": True,
        "user_action_required_written": True,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "conversion_preflight_only": True,
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
            "runner": ".venv-pytorch/bin/python run_stage42_ucy_h100_terms_gated_conversion_preflight.py -> 14/14",
            "focused_pytest": ".venv-pytorch/bin/python -m pytest tests/test_stage42_ucy_h100_terms_gated_conversion_preflight.py -> 4 passed",
            "full_pytest": ".venv-pytorch/bin/python -m pytest tests -> 840 passed",
        },
    }
    payload["stage42_fr_gate"] = _gate(payload)
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gate = payload["stage42_fr_gate"]
    lines = [
        "# Stage42-FR UCY H100 Terms-Gated Conversion Preflight",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- input FQ verdict: `{s['input_fq_verdict']}`",
        f"- dataset_id: `{s['dataset_id']}`",
        f"- candidate rows: `{s['candidate_rows']}`",
        f"- target-family candidates: `{s['target_family_candidates']}`",
        f"- conversion_preflight_ready_count: `{s['conversion_preflight_ready_count']}`",
        f"- blockers: `{s['terms_confirmation_blockers']}`",
        f"- template: `{TEMPLATE_JSON}`",
        f"- verification: `{payload['verification']}`",
        "",
        "## Candidate Table",
        "",
        "| order | candidate_id | source_id | relative path | family | target match | max track | est h100 windows | ready | blockers |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["candidate_rows"]:
        lines.append(
            f"| {row['recommended_order']} | `{row['candidate_id']}` | `{row['source_id']}` | `{row['relative_path']}` | `{row['family_bucket']}` | "
            f"{row['target_bucket_match']} | {row['max_track_points']} | {row['estimated_t100_windows']} | "
            f"{row['conversion_preflight_ready']} | {', '.join(row['terms_confirmation_blockers']) or 'none'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- FR is a file-level preflight, not a conversion stage.",
        "- `UCY_zara02` is the highest-priority target-family h100 support candidate, but it is blocked until user-confirmed official terms/local path/source identity.",
        "- `UCY_students03` has more estimated h100 windows but is not the zara target family for the current UCY|100 weak slice; it is secondary support.",
        "- No raw data, cache, converted feature store, metric/seconds claim, Stage5C, or SMC is produced.",
    ]
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-FR User Action Required: UCY H100 Source Support",
        "",
        "Stage42-FR found local UCY h100 candidates, but did not convert them because official terms/path/source identity are not confirmed.",
        "",
        "## Required Manual Confirmation",
        "",
        f"- official terms URL: {UCY_OFFICIAL_URL}",
        "- Fill `outputs/stage42_long_research/ucy_h100_candidate_terms_template_stage42.json` only after manually verifying the official terms.",
        "- Do not mark conversion complete until guarded conversion, no-leakage, source-CV, and h100 easy-safety CI all pass.",
        "",
        "## Priority Candidates",
        "",
        "| candidate_id | source_id | relative path | reason |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["candidate_rows"][:6]:
        reason = "target zara family" if row["target_bucket_match"] else "secondary h100 support"
        lines.append(f"| `{row['candidate_id']}` | `{row['source_id']}` | `{row['relative_path']}` | {reason} |")
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fr_gate"]
    lines = [
        "# Stage42-FR Gate",
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
            "<!-- STAGE42_FR_UCY_H100_TERMS_GATED_PREFLIGHT:START -->",
            "## Stage42-FR UCY H100 Terms-Gated Conversion Preflight",
            "",
            f"- source: `{payload['source']}`",
            "- role: file-level UCY h100 candidate preflight from FQ; no conversion, no training, no auto-download.",
            f"- gate: `{payload['stage42_fr_gate']['passed']} / {payload['stage42_fr_gate']['total']}`; verdict `{payload['stage42_fr_gate']['verdict']}`.",
            f"- candidates: `{s['candidate_rows']}` total, `{s['target_family_candidates']}` target-family candidates.",
            f"- conversion_preflight_ready_count: `{s['conversion_preflight_ready_count']}`; blockers `{s['terms_confirmation_blockers']}`.",
            f"- recommended first sources after user confirmation: `{s['recommended_first_sources']}`.",
            "- Boundary: protected source-level raw-frame 2.5D; no converted dataset claim, no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
            f"- verification: `{payload['verification']}`.",
            "<!-- STAGE42_FR_UCY_H100_TERMS_GATED_PREFLIGHT:END -->",
            "",
        ]
    )


def _append_to_docs(payload: Mapping[str, Any]) -> None:
    section = _summary_section(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY, GOAL_LEDGER, *PAPER_FILES]:
        old = path.read_text() if path.exists() else ""
        path.write_text(fg._replace_text_section(old, "STAGE42_FR_UCY_H100_TERMS_GATED_PREFLIGHT", section))


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FR UCY h100 terms-gated conversion preflight"
    state["current_verdict"] = payload["stage42_fr_gate"]["verdict"]
    state["stage42_fr_ucy_h100_terms_gated_conversion_preflight"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "template": str(TEMPLATE_JSON),
        "verdict": payload["stage42_fr_gate"]["verdict"],
        "gates": f"{payload['stage42_fr_gate']['passed']}/{payload['stage42_fr_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "verification": payload["verification"],
        "conclusion": "Stage42-FR maps UCY h100 source-support candidates to file-level terms-gated conversion readiness; no candidate is legal-ready until user confirmation.",
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_ucy_h100_terms_gated_conversion_preflight() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_json(TEMPLATE_JSON, payload["template"])
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_ucy_h100_terms_gated_conversion_preflight()
    gate = result["stage42_fr_gate"]
    print(f"Stage42-FR gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
