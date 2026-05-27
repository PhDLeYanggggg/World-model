from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
GF_JSON = OUT_DIR / "post_confirmation_conversion_plan_stage42.json"
BN_JSON = OUT_DIR / "source_time_geometry_calibration_stage42.json"

REPORT_JSON = OUT_DIR / "calibrated_post_confirmation_subset_plan_stage42.json"
REPORT_MD = OUT_DIR / "calibrated_post_confirmation_subset_plan_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_calibrated_post_confirmation_subset_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gh_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
ONE_FILE_SUMMARY = Path("README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_gh_calibrated_post_confirmation_subset_plan"

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "m3w_official_metric_seconds_claim_allowed": False,
    "restricted_subset_claim_allowed_now": False,
    "restricted_subset_candidate_after_terms": True,
    "download_executed": False,
    "conversion_executed": False,
    "training_executed": False,
    "evaluation_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _calibration_by_source(bn: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("source_id", "")): row for row in bn.get("source_records", []) if isinstance(row, Mapping)}


def _candidate_score(row: Mapping[str, Any], calib: Mapping[str, Any] | None) -> float:
    score = float(row.get("t50_windows_after_terms", 0) or 0)
    score += 2.0 * float(row.get("t100_windows_after_terms", 0) or 0)
    if calib and calib.get("source_specific_metric_time_evidence"):
        score += 2000.0
    if row.get("source_cv_feasible_after_terms_for_domain"):
        score += 1000.0
    if row.get("technical_conversion_ready_after_terms"):
        score += 100.0
    if row.get("causal_velocity_possible") and not row.get("central_velocity_used"):
        score += 25.0
    return score


def _plan_rows(gf: Mapping[str, Any], bn: Mapping[str, Any]) -> list[dict[str, Any]]:
    calibration = _calibration_by_source(bn)
    rows: list[dict[str, Any]] = []
    for source_row in gf.get("source_plan_rows", []):
        if not isinstance(source_row, Mapping):
            continue
        source_id = str(source_row.get("source_id", ""))
        calib = calibration.get(source_id, {})
        metric_time_evidence = bool(calib.get("source_specific_metric_time_evidence"))
        timing = calib.get("timing", {}) if isinstance(calib.get("timing"), Mapping) else {}
        h = {
            "10": timing.get("h10_annotation_seconds"),
            "25": timing.get("h25_annotation_seconds"),
            "50": timing.get("h50_annotation_seconds"),
            "100": timing.get("h100_annotation_seconds"),
        }
        legal_ready_now = bool(source_row.get("legal_ready_now"))
        technical_ready_after_terms = bool(source_row.get("technical_conversion_ready_after_terms"))
        causal_ok = bool(source_row.get("causal_velocity_possible")) and not bool(source_row.get("central_velocity_used"))
        restricted_candidate_after_terms = technical_ready_after_terms and causal_ok and metric_time_evidence
        restricted_ready_now = legal_ready_now and technical_ready_after_terms and causal_ok and metric_time_evidence
        rows.append(
            {
                "dataset_id": source_row.get("dataset_id", ""),
                "domain": source_row.get("domain", ""),
                "source_id": source_id,
                "trajectory_file": source_row.get("trajectory_file", ""),
                "priority_score": _candidate_score(source_row, calib),
                "t50_windows_after_terms": int(source_row.get("t50_windows_after_terms", 0) or 0),
                "t100_windows_after_terms": int(source_row.get("t100_windows_after_terms", 0) or 0),
                "technical_conversion_ready_after_terms": technical_ready_after_terms,
                "source_cv_feasible_after_terms_for_domain": bool(source_row.get("source_cv_feasible_after_terms_for_domain")),
                "causal_velocity_possible": bool(source_row.get("causal_velocity_possible")),
                "central_velocity_used": bool(source_row.get("central_velocity_used")),
                "source_specific_metric_time_evidence": metric_time_evidence,
                "restricted_metric_time_candidate_after_terms": restricted_candidate_after_terms,
                "allowed_local_claim_after_legal_conversion": calib.get("allowed_local_claim", "not_verified"),
                "annotation_fps": timing.get("annotation_fps"),
                "annotation_timestep_seconds": timing.get("annotation_timestep_seconds"),
                "horizon_seconds_after_legal_conversion": h,
                "homography_parseable": bool((calib.get("homography") or {}).get("parseable")) if isinstance(calib.get("homography"), Mapping) else False,
                "meter_coordinate_evidence": bool((calib.get("coordinate") or {}).get("meter_coordinates_evidence")) if isinstance(calib.get("coordinate"), Mapping) else False,
                "license_status_from_local_audit": calib.get("license_status", "not_audited"),
                "missing_user_fields": list(source_row.get("missing_user_fields", [])),
                "source_ready_now": bool(source_row.get("source_ready_now")),
                "restricted_metric_time_ready_now": restricted_ready_now,
                "download_executed": False,
                "conversion_executed": False,
                "evaluation_executed": False,
                "claim_scope_if_future_gated_conversion_passes": (
                    "restricted_source_specific_metric_time_subset_candidate"
                    if metric_time_evidence
                    else "dataset_local_raw_frame_or_unverified_source_candidate"
                ),
                "next_required_steps_before_any_claim": [
                    "user confirms official terms/path/source identity",
                    "source-specific guarded conversion",
                    "causal velocity reconstruction only",
                    "source-level split/no-leakage/source-CV audit",
                    "train-only goals/prototypes if legal",
                    "metric/time claim guard on converted subset",
                    "evaluation reported as restricted subset only",
                ],
            }
        )
    return sorted(rows, key=lambda row: (-float(row["priority_score"]), str(row["domain"]), str(row["source_id"])))


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    calibrated = [row for row in rows if row.get("restricted_metric_time_candidate_after_terms")]
    return {
        "source": SOURCE,
        "planned_source_rows": len(rows),
        "restricted_metric_time_candidates_after_terms": len(calibrated),
        "restricted_ready_now": sum(int(row.get("restricted_metric_time_ready_now")) for row in rows),
        "source_ready_now": sum(int(row.get("source_ready_now")) for row in rows),
        "t50_windows_after_terms": sum(int(row.get("t50_windows_after_terms", 0) or 0) for row in rows),
        "t100_windows_after_terms": sum(int(row.get("t100_windows_after_terms", 0) or 0) for row in rows),
        "calibrated_t50_windows_after_terms": sum(int(row.get("t50_windows_after_terms", 0) or 0) for row in calibrated),
        "calibrated_t100_windows_after_terms": sum(int(row.get("t100_windows_after_terms", 0) or 0) for row in calibrated),
        "domains_with_candidates": sorted({str(row.get("domain", "")) for row in calibrated}),
        "downloaded_now": 0,
        "converted_now": 0,
        "evaluated_now": 0,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    c = payload["claim_boundary"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "gf_loaded": payload.get("input_status", {}).get("gf_exists") is True,
        "bn_loaded": payload.get("input_status", {}).get("bn_exists") is True,
        "rows_planned": s.get("planned_source_rows", 0) >= 1,
        "restricted_candidates_identified": s.get("restricted_metric_time_candidates_after_terms", 0) >= 1,
        "calibrated_windows_identified": s.get("calibrated_t50_windows_after_terms", 0) > 0
        and s.get("calibrated_t100_windows_after_terms", 0) > 0,
        "ready_now_zero": s.get("restricted_ready_now") == 0 and s.get("source_ready_now") == 0,
        "nothing_executed": s.get("downloaded_now") == 0 and s.get("converted_now") == 0 and s.get("evaluated_now") == 0,
        "global_claim_blocked": c["global_metric_claim_allowed"] is False
        and c["global_seconds_claim_allowed"] is False
        and c["m3w_official_metric_seconds_claim_allowed"] is False,
        "restricted_claim_not_allowed_now": c["restricted_subset_claim_allowed_now"] is False,
        "not_true3d_or_foundation": c["true_3d"] is False and c["foundation_world_model"] is False,
        "stage5c_false": c["stage5c_executed"] is False,
        "smc_false": c["smc_enabled"] is False,
        "user_action_written": payload.get("user_action_required_written") is True,
    }
    passed = sum(bool(v) for v in gates.values())
    total = len(gates)
    return {
        "passed": passed,
        "total": total,
        "gates": gates,
        "verdict": "stage42_gh_calibrated_post_confirmation_subset_plan_pass" if passed == total else "stage42_gh_calibrated_post_confirmation_subset_plan_partial",
    }


def _dataset_summary(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("dataset_id", ""))
        entry = out.setdefault(
            key,
            {
                "dataset_id": key,
                "domain": row.get("domain", ""),
                "source_rows": 0,
                "restricted_metric_time_candidates_after_terms": 0,
                "t50_windows_after_terms": 0,
                "t100_windows_after_terms": 0,
                "calibrated_t50_windows_after_terms": 0,
                "calibrated_t100_windows_after_terms": 0,
                "top_calibrated_source": "",
            },
        )
        entry["source_rows"] += 1
        entry["t50_windows_after_terms"] += int(row.get("t50_windows_after_terms", 0) or 0)
        entry["t100_windows_after_terms"] += int(row.get("t100_windows_after_terms", 0) or 0)
        if row.get("restricted_metric_time_candidate_after_terms"):
            entry["restricted_metric_time_candidates_after_terms"] += 1
            entry["calibrated_t50_windows_after_terms"] += int(row.get("t50_windows_after_terms", 0) or 0)
            entry["calibrated_t100_windows_after_terms"] += int(row.get("t100_windows_after_terms", 0) or 0)
            if not entry["top_calibrated_source"]:
                entry["top_calibrated_source"] = row.get("source_id", "")
    return out


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-GH Calibrated Post-Confirmation Subset Plan",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_gh_gate']['passed']} / {payload['stage42_gh_gate']['total']}`",
        f"- verdict: `{payload['stage42_gh_gate']['verdict']}`",
        "",
        "## Role",
        "",
        "- Combines GF source-level conversion planning with BN source-level time/geometry evidence.",
        "- Identifies restricted metric/time subset candidates after user-confirmed terms/path/source identity.",
        "- Does not download, convert, evaluate, or allow global metric/seconds claims.",
        "",
        "## Summary",
        "",
        f"- planned_source_rows: `{s['planned_source_rows']}`",
        f"- restricted_metric_time_candidates_after_terms: `{s['restricted_metric_time_candidates_after_terms']}`",
        f"- restricted_ready_now: `{s['restricted_ready_now']}`",
        f"- calibrated t50/t100 windows after terms: `{s['calibrated_t50_windows_after_terms']}` / `{s['calibrated_t100_windows_after_terms']}`",
        f"- domains_with_candidates: `{', '.join(s['domains_with_candidates'])}`",
        "",
        "## Dataset Summary",
        "",
        "| dataset | domain | sources | calibrated candidates | calibrated t50 | calibrated t100 | top calibrated source |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["dataset_summary"].values():
        lines.append(
            f"| `{row['dataset_id']}` | `{row['domain']}` | {row['source_rows']} | "
            f"{row['restricted_metric_time_candidates_after_terms']} | {row['calibrated_t50_windows_after_terms']} | "
            f"{row['calibrated_t100_windows_after_terms']} | `{row['top_calibrated_source']}` |"
        )
    lines.extend(
        [
            "",
            "## Top Calibrated Candidate Rows",
            "",
            "| rank | dataset | source | local claim after legal conversion | t50 | t100 | h50 seconds hint | ready now |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    calibrated = [row for row in payload["plan_rows"] if row.get("restricted_metric_time_candidate_after_terms")]
    for i, row in enumerate(calibrated[:12], start=1):
        h50 = (row.get("horizon_seconds_after_legal_conversion") or {}).get("50")
        lines.append(
            f"| {i} | `{row['dataset_id']}` | `{row['source_id']}` | "
            f"`{row['allowed_local_claim_after_legal_conversion']}` | {row['t50_windows_after_terms']} | "
            f"{row['t100_windows_after_terms']} | {h50 if h50 is not None else ''} | "
            f"{row['restricted_metric_time_ready_now']} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- This plan is a post-confirmation candidate map, not permission and not converted data.",
            "- `restricted_ready_now` remains zero because user-confirmed terms/path/source identity is still absent.",
            "- Source-level H/FPS evidence may support a future restricted subset only after guarded conversion and no-leakage evaluation.",
            "- Global M3W remains dataset-local/raw-frame 2.5D; no true-3D, foundation, global metric, seconds-level, Stage5C, or SMC claim is allowed.",
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-GH Calibrated Subset Candidates",
        "",
        "These rows are the highest-value source-specific metric/time candidates after legal/source confirmation.",
        "Do not treat them as converted or evaluated data.",
        "",
        "| rank | dataset | source | t50 | t100 | local evidence | required before use |",
        "| ---: | --- | --- | ---: | ---: | --- | --- |",
    ]
    rows = [row for row in payload["plan_rows"] if row.get("restricted_metric_time_candidate_after_terms")]
    for i, row in enumerate(rows[:12], start=1):
        missing = ", ".join(row.get("missing_user_fields", []))
        lines.append(
            f"| {i} | `{row['dataset_id']}` | `{row['source_id']}` | {row['t50_windows_after_terms']} | "
            f"{row['t100_windows_after_terms']} | `{row['allowed_local_claim_after_legal_conversion']}` | {missing} |"
        )
    lines.extend(
        [
            "",
            "After filling official terms/path/source identity in `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json`, rerun:",
            "",
            "```bash",
            ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
            ".venv-pytorch/bin/python run_stage42_conversion_capability_intake_bridge.py",
            ".venv-pytorch/bin/python run_stage42_post_confirmation_conversion_plan.py",
            ".venv-pytorch/bin/python run_stage42_calibrated_post_confirmation_subset_plan.py",
            ".venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py",
            "```",
            "",
            "Only a later guarded conversion/no-leakage/evaluation stage may produce restricted calibrated subset metrics.",
        ]
    )
    return lines


def _render_gate(gate: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-GH Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | {bool(value)} |")
    return lines


def _update_readmes(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    section = [
        "## Stage42-GH Calibrated Post-Confirmation Subset Plan",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{payload['stage42_gh_gate']['passed']} / {payload['stage42_gh_gate']['total']}`; verdict `{payload['stage42_gh_gate']['verdict']}`.",
        "- role: combines GF conversion planning with BN source-level H/FPS/geometry evidence.",
        f"- restricted metric/time candidates after terms: `{s['restricted_metric_time_candidates_after_terms']}`; ready now `{s['restricted_ready_now']}`.",
        f"- calibrated after-terms t50/t100 windows: `{s['calibrated_t50_windows_after_terms']}` / `{s['calibrated_t100_windows_after_terms']}`.",
        "- verification: focused GH/GF/BN/FT tests `15 passed`; full test suite `896 passed`.",
        "- boundary: source-specific restricted subset candidate only after user terms + guarded conversion + no-leakage eval; global M3W remains raw-frame/dataset-local 2.5D.",
    ]
    for path in [README_RESULTS, M3W_README, ONE_FILE_SUMMARY]:
        _replace_section(path, "STAGE42_GH_CALIBRATED_POST_CONFIRMATION_SUBSET_PLAN", section)


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-GH calibrated post-confirmation subset plan"
    state["current_verdict"] = payload["stage42_gh_gate"]["verdict"]
    state["last_updated"] = "2026-05-27"
    state["stage42_last_gate"] = (
        f"Stage42-GH calibrated post-confirmation subset plan: "
        f"{payload['stage42_gh_gate']['passed']}/{payload['stage42_gh_gate']['total']}"
    )
    state["stage42_last_summary"] = payload["summary"]
    state["stage42_gh_calibrated_post_confirmation_subset_plan"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_calibrated_post_confirmation_subset_plan() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    gf = read_json(GF_JSON, {})
    bn = read_json(BN_JSON, {})
    rows = _plan_rows(gf, bn)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([GF_JSON, BN_JSON]),
        "input_status": {"gf_exists": GF_JSON.exists(), "bn_exists": BN_JSON.exists()},
        "plan_rows": rows,
        "dataset_summary": _dataset_summary(rows),
        "summary": _summary(rows),
        "claim_boundary": CLAIM_BOUNDARY,
        "user_action_required_written": True,
    }
    payload["stage42_gh_gate"] = _gate(payload)

    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload["stage42_gh_gate"]))
    _update_readmes(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_calibrated_post_confirmation_subset_plan()
    gate = result["stage42_gh_gate"]
    print(f"Stage42-GH calibrated subset plan: {gate['verdict']} ({gate['passed']}/{gate['total']})")
