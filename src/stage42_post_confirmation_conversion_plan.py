from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_prefill_intake_bridge import _has_user_confirmation
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
INTAKE_JSON = OUT_DIR / "source_terms_confirmation_intake_template_stage42.json"
MANIFEST_JSON = OUT_DIR / "source_conversion_readiness_manifest_stage42.json"
GE_JSON = OUT_DIR / "conversion_capability_intake_bridge_stage42.json"

REPORT_JSON = OUT_DIR / "post_confirmation_conversion_plan_stage42.json"
REPORT_MD = OUT_DIR / "post_confirmation_conversion_plan_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_post_confirmation_conversion_plan_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gf_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
ONE_FILE_SUMMARY = Path("README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_gf_post_confirmation_conversion_plan"

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "download_executed": False,
    "conversion_executed": False,
    "training_executed": False,
    "evaluation_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
    "plan_is_permission": False,
    "plan_is_conversion": False,
}


def _intake_rows(intake: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(intake.get("datasets", []))


def _source_score(source: Mapping[str, Any], source_cv_feasible: bool) -> float:
    h = source.get("horizon_counts", {}) or {}
    t50 = int(h.get("50", 0) or 0)
    t100 = int(h.get("100", 0) or 0)
    score = float(t50 + 2 * t100)
    if source_cv_feasible:
        score += 1000.0
    if source.get("technical_conversion_ready_after_terms"):
        score += 100.0
    if source.get("causal_velocity_possible") and not source.get("central_velocity_used"):
        score += 25.0
    return score


def _missing_user_fields(row: Mapping[str, Any]) -> list[str]:
    user = row.get("user_confirmation", {}) or {}
    missing: list[str] = []
    if user.get("terms_accepted_by_user") is not True:
        missing.append("terms_accepted_by_user")
    for key in ["terms_acceptance_date", "allowed_use", "local_path", "source_identity", "confirmed_by_user"]:
        if not str(user.get(key, "")).strip():
            missing.append(key)
    if str(user.get("redistribution_allowed", "unknown")).strip().lower() == "unknown":
        missing.append("redistribution_allowed")
    if str(user.get("derived_data_allowed", "unknown")).strip().lower() == "unknown":
        missing.append("derived_data_allowed")
    return missing


def _source_plan_rows(intake: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in _intake_rows(intake):
        cap = dataset.get("conversion_capability_prefill", {}) or {}
        source_cv = bool(cap.get("source_cv_feasible_after_terms"))
        missing = _missing_user_fields(dataset)
        user_confirmed = _has_user_confirmation(dataset)
        for source in cap.get("source_rows", []):
            source_copy = deepcopy(dict(source))
            technical_ready = bool(source_copy.get("technical_conversion_ready_after_terms"))
            central_velocity_used = bool(source_copy.get("central_velocity_used"))
            causal_ok = bool(source_copy.get("causal_velocity_possible")) and not central_velocity_used
            legal_ready_now = bool(user_confirmed and not missing)
            source_ready_now = legal_ready_now and technical_ready and causal_ok
            h = source_copy.get("horizon_counts", {}) or {}
            rows.append(
                {
                    "dataset_id": dataset.get("dataset_id", ""),
                    "domain": cap.get("domain", dataset.get("domain", "")),
                    "source_id": source_copy.get("source_id", ""),
                    "trajectory_file": source_copy.get("trajectory_file", ""),
                    "priority_score": _source_score(source_copy, source_cv),
                    "technical_conversion_ready_after_terms": technical_ready,
                    "source_cv_feasible_after_terms_for_domain": source_cv,
                    "t50_windows_after_terms": int(h.get("50", 0) or 0),
                    "t100_windows_after_terms": int(h.get("100", 0) or 0),
                    "causal_velocity_possible": bool(source_copy.get("causal_velocity_possible")),
                    "central_velocity_used": central_velocity_used,
                    "legal_ready_now": legal_ready_now,
                    "source_ready_now": source_ready_now,
                    "queued_now": False,
                    "conversion_executed": False,
                    "evaluation_executed": False,
                    "missing_user_fields": missing,
                    "blocked_by": list(source_copy.get("blocked_by", [])) + ([] if technical_ready else ["technical_conversion_not_ready_after_terms"]) + ([] if causal_ok else ["causal_velocity_preflight_failed"]),
                    "allowed_future_steps_after_confirmation": [
                        "source-specific parser",
                        "causal velocity reconstruction only",
                        "source-level split/source-CV rebuild",
                        "train-only goals/prototypes if legal",
                        "no-leakage audit",
                        "strongest causal baseline recomputation",
                        "protected policy replay/evaluation",
                        "metric/time claim guard",
                    ],
                    "forbidden_now": [
                        "do not download",
                        "do not convert",
                        "do not evaluate",
                        "do not infer metric/seconds",
                        "do not use future/test endpoint goals",
                        "do not execute Stage5C or SMC",
                    ],
                }
            )
    return sorted(rows, key=lambda row: (-float(row["priority_score"]), str(row["domain"]), str(row["source_id"])))


def _dataset_summary(plan_rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in plan_rows:
        ds = str(row.get("dataset_id", ""))
        current = out.setdefault(
            ds,
            {
                "dataset_id": ds,
                "domain": row.get("domain", ""),
                "source_count": 0,
                "technical_ready_after_terms_sources": 0,
                "source_ready_now": 0,
                "t50_windows_after_terms": 0,
                "t100_windows_after_terms": 0,
                "source_cv_feasible_after_terms": bool(row.get("source_cv_feasible_after_terms_for_domain")),
                "top_source_id": "",
            },
        )
        current["source_count"] += 1
        current["technical_ready_after_terms_sources"] += int(bool(row.get("technical_conversion_ready_after_terms")))
        current["source_ready_now"] += int(bool(row.get("source_ready_now")))
        current["t50_windows_after_terms"] += int(row.get("t50_windows_after_terms", 0) or 0)
        current["t100_windows_after_terms"] += int(row.get("t100_windows_after_terms", 0) or 0)
        current["source_cv_feasible_after_terms"] = current["source_cv_feasible_after_terms"] or bool(row.get("source_cv_feasible_after_terms_for_domain"))
        if not current["top_source_id"]:
            current["top_source_id"] = row.get("source_id", "")
    return out


def _summary(intake: Mapping[str, Any], manifest: Mapping[str, Any], plan_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    dataset_summary = _dataset_summary(plan_rows)
    return {
        "source": SOURCE,
        "intake_rows": len(_intake_rows(intake)),
        "planned_source_rows": len(plan_rows),
        "datasets_with_source_plan": len(dataset_summary),
        "technical_ready_after_terms_sources": sum(int(row.get("technical_conversion_ready_after_terms")) for row in plan_rows),
        "source_cv_feasible_after_terms_datasets": sum(1 for row in dataset_summary.values() if row.get("source_cv_feasible_after_terms")),
        "t50_windows_after_terms": sum(int(row.get("t50_windows_after_terms", 0) or 0) for row in plan_rows),
        "t100_windows_after_terms": sum(int(row.get("t100_windows_after_terms", 0) or 0) for row in plan_rows),
        "source_ready_now": sum(int(row.get("source_ready_now")) for row in plan_rows),
        "conversion_ready_targets_in_manifest": len(manifest.get("conversion_ready_targets", [])),
        "queued_now": 0,
        "downloaded_now": 0,
        "converted_now": 0,
        "evaluated_now": 0,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "intake_loaded": payload.get("input_status", {}).get("intake_exists") is True,
        "ge_loaded": payload.get("input_status", {}).get("ge_exists") is True,
        "manifest_loaded": payload.get("input_status", {}).get("manifest_exists") is True,
        "source_rows_planned": s.get("planned_source_rows", 0) >= 1,
        "technical_ready_after_terms_present": s.get("technical_ready_after_terms_sources", 0) >= 1,
        "source_cv_dataset_present": s.get("source_cv_feasible_after_terms_datasets", 0) >= 1,
        "horizon_windows_present": s.get("t50_windows_after_terms", 0) > 0 and s.get("t100_windows_after_terms", 0) > 0,
        "no_sources_ready_now": s.get("source_ready_now") == 0,
        "manifest_ready_zero_preserved": s.get("conversion_ready_targets_in_manifest") == 0,
        "nothing_queued_or_executed": s.get("queued_now") == 0
        and s.get("downloaded_now") == 0
        and s.get("converted_now") == 0
        and s.get("evaluated_now") == 0,
        "user_action_written": payload.get("user_action_required_written") is True,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "no_true3d_foundation_overclaim": claim["true_3d"] is False and claim["foundation_world_model"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(bool(v) for v in gates.values())
    total = len(gates)
    return {
        "passed": passed,
        "total": total,
        "gates": gates,
        "verdict": "stage42_gf_post_confirmation_conversion_plan_pass" if passed == total else "stage42_gf_post_confirmation_conversion_plan_partial",
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-GF Post-Confirmation Conversion Plan",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_gf_gate']['passed']} / {payload['stage42_gf_gate']['total']}`",
        f"- verdict: `{payload['stage42_gf_gate']['verdict']}`",
        "",
        "## Role",
        "",
        "- Turns GE `conversion_capability_prefill` into a ranked source-level post-confirmation execution plan.",
        "- It is not legal permission, not a conversion queue, not converted data, and not evaluation.",
        "- It tells the next guarded conversion stage which source rows become worth converting after user-confirmed terms/path/source identity.",
        "",
        "## Summary",
        "",
        f"- planned_source_rows: `{s['planned_source_rows']}`",
        f"- technical_ready_after_terms_sources: `{s['technical_ready_after_terms_sources']}`",
        f"- source_cv_feasible_after_terms_datasets: `{s['source_cv_feasible_after_terms_datasets']}`",
        f"- t50/t100 windows after terms: `{s['t50_windows_after_terms']}` / `{s['t100_windows_after_terms']}`",
        f"- source_ready_now: `{s['source_ready_now']}`",
        f"- conversion_ready_targets_in_manifest: `{s['conversion_ready_targets_in_manifest']}`",
        "",
        "## Dataset Summary",
        "",
        "| dataset | domain | sources | tech-ready after terms | source-CV after terms | t50 | t100 | top source |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["dataset_summary"].values():
        lines.append(
            f"| `{row['dataset_id']}` | `{row['domain']}` | {row['source_count']} | "
            f"{row['technical_ready_after_terms_sources']} | {row['source_cv_feasible_after_terms']} | "
            f"{row['t50_windows_after_terms']} | {row['t100_windows_after_terms']} | `{row['top_source_id']}` |"
        )
    lines.extend(
        [
            "",
            "## Top Source Plan Rows",
            "",
            "| rank | dataset | domain | source | score | t50 | t100 | tech after terms | source ready now | missing user fields |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for i, row in enumerate(payload["source_plan_rows"][:12], start=1):
        lines.append(
            f"| {i} | `{row['dataset_id']}` | `{row['domain']}` | `{row['source_id']}` | "
            f"{row['priority_score']:.1f} | {row['t50_windows_after_terms']} | {row['t100_windows_after_terms']} | "
            f"{row['technical_conversion_ready_after_terms']} | {row['source_ready_now']} | {', '.join(row['missing_user_fields']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- This plan is intentionally non-executing.",
            "- `source_ready_now` remains zero because user confirmation is still absent.",
            "- It does not permit metric/seconds, true-3D, foundation, Stage5C, or SMC claims.",
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-GF Post-Confirmation Conversion Plan",
        "",
        "Open `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json` and inspect each row's `conversion_capability_prefill` before filling `user_confirmation`.",
        "",
        "Highest-value sources after terms confirmation:",
        "",
        "| rank | dataset | source | t50 | t100 | why |",
        "| ---: | --- | --- | ---: | ---: | --- |",
    ]
    for i, row in enumerate(payload["source_plan_rows"][:8], start=1):
        reason = "source-CV after terms" if row["source_cv_feasible_after_terms_for_domain"] else "technical source-specific candidate"
        lines.append(
            f"| {i} | `{row['dataset_id']}` | `{row['source_id']}` | {row['t50_windows_after_terms']} | {row['t100_windows_after_terms']} | {reason} |"
        )
    lines.extend(
        [
            "",
            "Required user-confirmed fields before any conversion:",
            "",
            "- `terms_accepted_by_user`",
            "- `terms_acceptance_date`",
            "- `allowed_use`",
            "- `redistribution_allowed`",
            "- `derived_data_allowed`",
            "- `local_path`",
            "- `source_identity`",
            "- `confirmed_by_user`",
            "",
            "Then rerun:",
            "",
            "```bash",
            ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
            ".venv-pytorch/bin/python run_stage42_post_confirmation_conversion_plan.py",
            ".venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py",
            "```",
        ]
    )
    return lines


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-GF Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    write_md(GATE_MD, lines)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "## Stage42-GF Post-Confirmation Conversion Plan",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{payload['stage42_gf_gate']['passed']} / {payload['stage42_gf_gate']['total']}`; verdict `{payload['stage42_gf_gate']['verdict']}`.",
        "- role: ranks GE source-specific conversion capability rows into a post-confirmation execution plan.",
        f"- planned source rows: `{s['planned_source_rows']}`; technical-ready-after-terms sources `{s['technical_ready_after_terms_sources']}`; source-CV-capable datasets `{s['source_cv_feasible_after_terms_datasets']}`.",
        f"- t50/t100 after-terms windows: `{s['t50_windows_after_terms']}` / `{s['t100_windows_after_terms']}`; source_ready_now `{s['source_ready_now']}`; manifest ready targets `{s['conversion_ready_targets_in_manifest']}`.",
        "- boundary: plan is not permission, not conversion, not evaluation; no metric/seconds/true-3D/foundation/Stage5C/SMC claim.",
    ]
    for path in [README_RESULTS, M3W_README, ONE_FILE_SUMMARY]:
        _replace_section(path, "STAGE42_GF_POST_CONFIRMATION_CONVERSION_PLAN", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-GF post-confirmation conversion plan"
    state["current_verdict"] = payload["stage42_gf_gate"]["verdict"]
    state["stage42_gf_post_confirmation_conversion_plan"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "updated_at": payload["generated_at_utc"],
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_post_confirmation_conversion_plan() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    intake = read_json(INTAKE_JSON, {})
    manifest = read_json(MANIFEST_JSON, {})
    ge = read_json(GE_JSON, {})
    plan_rows = _source_plan_rows(intake)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([INTAKE_JSON, MANIFEST_JSON, GE_JSON]),
        "input_status": {
            "intake_exists": INTAKE_JSON.exists(),
            "manifest_exists": MANIFEST_JSON.exists(),
            "ge_exists": GE_JSON.exists(),
            "ge_verdict": ge.get("stage42_ge_gate", {}).get("verdict", ""),
        },
        "dataset_summary": _dataset_summary(plan_rows),
        "source_plan_rows": plan_rows,
        "summary": _summary(intake, manifest, plan_rows),
        "claim_boundary": CLAIM_BOUNDARY,
        "user_action_required_written": True,
    }
    payload["stage42_gf_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    _write_gate(payload["stage42_gf_gate"])
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


__all__ = [
    "run_stage42_post_confirmation_conversion_plan",
    "_source_plan_rows",
    "_source_score",
    "_missing_user_fields",
    "_gate",
]
