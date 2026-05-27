from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section
from src.stage42_source_terms_confirmation_intake import REQUIRED_FIELDS


OUT_DIR = Path("outputs/stage42_long_research")
INTAKE_JSON = OUT_DIR / "source_terms_confirmation_intake_template_stage42.json"
MANIFEST_JSON = OUT_DIR / "source_conversion_readiness_manifest_stage42.json"
GH_JSON = OUT_DIR / "calibrated_post_confirmation_subset_plan_stage42.json"
EJ_JSON = OUT_DIR / "guarded_source_conversion_launcher_stage42.json"

REPORT_JSON = OUT_DIR / "source_conversion_contract_stage42.json"
REPORT_MD = OUT_DIR / "source_conversion_contract_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_conversion_contract_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gl_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
ONE_FILE_SUMMARY = Path("README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_gl_source_conversion_contract"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-GL 是 source/legal/calibration 到 guarded conversion 的合约检查，不下载、不转换、不训练、不评估。",
    "post-confirmation calibrated subset candidates 只有在用户确认 official terms、allowed use、local path、source identity 后才可能进入 guarded conversion。",
    "blank intake、prefill suggestion、parseability、technical dry-run 都不等于 permission、conversion 或 evaluation。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon；不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "restricted_subset_claim_allowed_now": False,
    "download_executed": False,
    "conversion_executed": False,
    "training_executed": False,
    "evaluation_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _field_value_present(value: Any) -> bool:
    if isinstance(value, bool):
        return value is True
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().lower() not in {"unknown", "todo", "tbd"}
    return bool(value)


def _confirmation_status(row: Mapping[str, Any]) -> dict[str, Any]:
    user = row.get("user_confirmation", {}) if isinstance(row.get("user_confirmation"), Mapping) else {}
    field_status = {field: _field_value_present(user.get(field)) for field in REQUIRED_FIELDS}
    official_url = str(user.get("official_terms_url") or row.get("official_url_from_prior_audit") or "")
    if not official_url.startswith("http"):
        field_status["official_terms_url"] = False
    missing = [field for field, present in field_status.items() if not present]
    local_path = str(user.get("local_path", "")).strip()
    local_path_found = False
    if local_path:
        p = Path(local_path).expanduser()
        if not p.is_absolute():
            p = Path.cwd() / p
        local_path_found = p.exists()
    return {
        "field_status": field_status,
        "missing_fields": missing,
        "all_required_fields_filled": not missing,
        "local_path": local_path,
        "local_path_found": local_path_found,
        "official_terms_url": official_url,
        "terms_accepted_by_user": user.get("terms_accepted_by_user") is True,
        "source_identity": str(user.get("source_identity", "")).strip(),
        "agent_may_fill": bool(row.get("agent_may_fill")),
    }


def _manifest_by_dataset(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for key in ["conversion_ready_targets", "blocked_targets"]:
        for row in manifest.get(key, []):
            if isinstance(row, Mapping):
                rows[str(row.get("dataset_id", ""))] = row
    return rows


def _gh_by_dataset(gh: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in gh.get("plan_rows", []):
        if not isinstance(row, Mapping):
            continue
        dataset_id = str(row.get("dataset_id", ""))
        entry = out.setdefault(
            dataset_id,
            {
                "dataset_id": dataset_id,
                "domain": row.get("domain", ""),
                "restricted_metric_time_candidates_after_terms": 0,
                "restricted_ready_now": 0,
                "calibrated_t50_windows_after_terms": 0,
                "calibrated_t100_windows_after_terms": 0,
                "candidate_source_ids_after_terms": [],
                "allowed_local_claim_after_legal_conversion": set(),
            },
        )
        if row.get("restricted_metric_time_candidate_after_terms"):
            entry["restricted_metric_time_candidates_after_terms"] += 1
            entry["calibrated_t50_windows_after_terms"] += int(row.get("t50_windows_after_terms", 0) or 0)
            entry["calibrated_t100_windows_after_terms"] += int(row.get("t100_windows_after_terms", 0) or 0)
            entry["candidate_source_ids_after_terms"].append(row.get("source_id", ""))
            entry["allowed_local_claim_after_legal_conversion"].add(
                row.get("allowed_local_claim_after_legal_conversion", "not_verified")
            )
        if row.get("restricted_metric_time_ready_now"):
            entry["restricted_ready_now"] += 1
    for entry in out.values():
        entry["allowed_local_claim_after_legal_conversion"] = sorted(entry["allowed_local_claim_after_legal_conversion"])
    return out


def _suggested_paths(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    prefill = row.get("prefill_suggestion", {}) if isinstance(row.get("prefill_suggestion"), Mapping) else {}
    paths = prefill.get("local_path_candidates", [])
    if not isinstance(paths, list):
        return []
    return [
        {
            "path": item.get("path", ""),
            "exists": bool(item.get("exists")),
            "is_raw_source_candidate": bool(item.get("is_raw_source_candidate")),
            "is_derived_or_cache": bool(item.get("is_derived_or_cache")),
            "size_mb": item.get("size_mb"),
        }
        for item in paths
        if isinstance(item, Mapping)
    ]


def _contract_rows(
    intake: Mapping[str, Any],
    manifest: Mapping[str, Any],
    gh: Mapping[str, Any],
) -> list[dict[str, Any]]:
    manifest_rows = _manifest_by_dataset(manifest)
    gh_rows = _gh_by_dataset(gh)
    out: list[dict[str, Any]] = []
    for row in intake.get("datasets", []):
        if not isinstance(row, Mapping):
            continue
        dataset_id = str(row.get("dataset_id", ""))
        confirmation = _confirmation_status(row)
        manifest_row = manifest_rows.get(dataset_id, {})
        gh_row = gh_rows.get(dataset_id, {})
        manifest_ready = bool(manifest_row.get("conversion_ready"))
        contract_ready = (
            manifest_ready
            and confirmation["all_required_fields_filled"]
            and confirmation["local_path_found"]
            and confirmation["agent_may_fill"] is False
        )
        if contract_ready:
            status = "queued_for_future_guarded_conversion"
        elif manifest_ready and not contract_ready:
            status = "validator_ready_but_contract_recheck_failed"
        else:
            status = "blocked_until_user_terms_path_source_confirmation"
        out.append(
            {
                "priority_rank": row.get("priority_rank"),
                "dataset_id": dataset_id,
                "domain": row.get("domain", ""),
                "official_url": confirmation["official_terms_url"],
                "confirmation": confirmation,
                "manifest_conversion_ready": manifest_ready,
                "manifest_blockers": {
                    "confirmation_blockers": list(manifest_row.get("confirmation_blockers", [])),
                    "cf_blockers": list(manifest_row.get("cf_blockers", [])),
                    "next_action": manifest_row.get(
                        "next_action",
                        "fill explicit official terms/path/source-identity confirmation before conversion",
                    ),
                },
                "after_terms_potential": row.get("after_terms_potential", {}),
                "calibrated_subset_after_terms": gh_row
                or {
                    "restricted_metric_time_candidates_after_terms": 0,
                    "restricted_ready_now": 0,
                    "calibrated_t50_windows_after_terms": 0,
                    "calibrated_t100_windows_after_terms": 0,
                    "candidate_source_ids_after_terms": [],
                    "allowed_local_claim_after_legal_conversion": [],
                },
                "suggested_local_paths_for_user_review": _suggested_paths(row),
                "contract_status": status,
                "contract_conversion_ready_now": contract_ready,
                "download_executed": False,
                "conversion_executed": False,
                "evaluation_executed": False,
                "next_safe_commands": [
                    ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
                    ".venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py",
                    "future guarded converter only after queued target appears",
                ],
            }
        )
    return out


def _summary(
    rows: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    gh: Mapping[str, Any],
    ej: Mapping[str, Any],
) -> dict[str, Any]:
    calibrated = [row for row in rows if row["calibrated_subset_after_terms"]["restricted_metric_time_candidates_after_terms"] > 0]
    return {
        "source": SOURCE,
        "intake_datasets": len(rows),
        "manifest_ready_targets": len(manifest.get("conversion_ready_targets", [])),
        "manifest_blocked_targets": len(manifest.get("blocked_targets", [])),
        "contract_ready_now": sum(int(row["contract_conversion_ready_now"]) for row in rows),
        "guarded_launcher_queue_count": len(ej.get("conversion_queue", [])),
        "post_confirmation_calibrated_candidate_datasets": len(calibrated),
        "post_confirmation_calibrated_source_rows": gh.get("summary", {}).get(
            "restricted_metric_time_candidates_after_terms", 0
        ),
        "calibrated_t50_windows_after_terms": sum(
            int(row["calibrated_subset_after_terms"]["calibrated_t50_windows_after_terms"]) for row in rows
        ),
        "calibrated_t100_windows_after_terms": sum(
            int(row["calibrated_subset_after_terms"]["calibrated_t100_windows_after_terms"]) for row in rows
        ),
        "download_executed": False,
        "conversion_executed": False,
        "training_executed": False,
        "evaluation_executed": False,
        "next_required_action": "user fills official terms/path/source identity, then reruns validator and guarded launcher",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    c = payload["claim_boundary"]
    rows = payload["contract_rows"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "intake_loaded": payload.get("input_status", {}).get("intake_exists") is True,
        "manifest_loaded": payload.get("input_status", {}).get("manifest_exists") is True,
        "gh_loaded": payload.get("input_status", {}).get("gh_exists") is True,
        "ej_loaded": payload.get("input_status", {}).get("ej_exists") is True,
        "required_fields_enforced": all("terms_accepted_by_user" in row["confirmation"]["field_status"] for row in rows),
        "blank_or_incomplete_rows_not_ready": all(
            row["contract_conversion_ready_now"] is False
            for row in rows
            if not row["confirmation"]["all_required_fields_filled"]
        ),
        "contract_ready_matches_launcher_queue": s["contract_ready_now"] == s["guarded_launcher_queue_count"],
        "post_confirmation_candidates_recorded": s["post_confirmation_calibrated_candidate_datasets"] >= 1,
        "calibrated_opportunity_recorded": s["calibrated_t50_windows_after_terms"] > 0
        and s["calibrated_t100_windows_after_terms"] > 0,
        "no_download_conversion_training_eval": not (
            s["download_executed"] or s["conversion_executed"] or s["training_executed"] or s["evaluation_executed"]
        ),
        "post_confirmation_candidates_not_claimed_as_data": s["contract_ready_now"] == 0
        and c["restricted_subset_claim_allowed_now"] is False,
        "user_action_written": payload.get("user_action_required_written") is True,
        "no_metric_seconds_overclaim": c["global_metric_claim_allowed"] is False
        and c["global_seconds_claim_allowed"] is False,
        "stage5c_false": c["stage5c_executed"] is False,
        "smc_false": c["smc_enabled"] is False,
    }
    passed = sum(bool(value) for value in gates.values())
    total = len(gates)
    verdict = "stage42_gl_source_conversion_contract_pass" if passed == total else "stage42_gl_source_conversion_contract_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-GL Source Conversion Contract",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_gl_gate']['passed']} / {payload['stage42_gl_gate']['total']}`",
        f"- verdict: `{payload['stage42_gl_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in s.items()],
        "",
        "## Contract Table",
        "",
        "| rank | dataset | domain | status | missing fields | local path found | after-terms calibrated sources | t50/t100 after terms |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in payload["contract_rows"]:
        missing = ", ".join(row["confirmation"]["missing_fields"]) or "none"
        calibrated = row["calibrated_subset_after_terms"]
        lines.append(
            f"| {row['priority_rank']} | `{row['dataset_id']}` | `{row['domain']}` | `{row['contract_status']}` | "
            f"{missing} | {row['confirmation']['local_path_found']} | "
            f"{calibrated['restricted_metric_time_candidates_after_terms']} | "
            f"{calibrated['calibrated_t50_windows_after_terms']} / {calibrated['calibrated_t100_windows_after_terms']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `post_confirmation` rows are only opportunity rows. They are not permission, not converted data, and not evaluated evidence.",
            "- `contract_ready_now = 0` means no future converter may run yet.",
            "- The next meaningful user action is to fill official terms/path/source identity in the intake template and rerun the validator.",
            "- Even after conversion, any metric/time wording must be limited to the restricted source-specific subset that passes the metric/time claim guard.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_gl_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-GL Source Conversion Contract",
        "",
        "No source is conversion-ready yet. Fill the intake template only after checking official terms and confirming local source identity.",
        "",
        "| dataset | official URL | missing fields | suggested raw paths to inspect |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["contract_rows"]:
        missing = ", ".join(row["confirmation"]["missing_fields"]) or "none"
        paths = [
            item["path"]
            for item in row["suggested_local_paths_for_user_review"]
            if item.get("exists") and item.get("is_raw_source_candidate")
        ]
        lines.append(
            f"| `{row['dataset_id']}` | {row['official_url']} | {missing} | {', '.join(paths) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "Allowed next commands after the user fills required fields:",
            "",
            "1. `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`",
            "2. `.venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py`",
            "3. A future guarded converter may run only for queued targets and must redo no-leakage/source-CV/metric-time checks.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_gl_gate"]
    return [
        "# Stage42-GL Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    return [
        "## Stage42-GL Source Conversion Contract",
        "",
        "- source: `fresh_stage42_gl_source_conversion_contract`",
        "- role: locks the path from user terms/path/source identity confirmation to future guarded conversion.",
        f"- gate: `{payload['stage42_gl_gate']['passed']} / {payload['stage42_gl_gate']['total']}`; verdict `{payload['stage42_gl_gate']['verdict']}`.",
        f"- intake datasets: `{s['intake_datasets']}`; manifest ready targets: `{s['manifest_ready_targets']}`; contract ready now: `{s['contract_ready_now']}`.",
        f"- post-confirmation calibrated source rows: `{s['post_confirmation_calibrated_source_rows']}`; calibrated t50/t100 opportunity after terms: `{s['calibrated_t50_windows_after_terms']}` / `{s['calibrated_t100_windows_after_terms']}`.",
        "- No download, conversion, training, or evaluation was executed.",
        "- Boundary: these candidates are not permission, converted data, metric/seconds claims, Stage5C, or SMC evidence.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, ONE_FILE_SUMMARY]:
        _replace_section(path, "STAGE42_GL_SOURCE_CONVERSION_CONTRACT", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-GL source conversion contract"
    state["current_verdict"] = payload["stage42_gl_gate"]["verdict"]
    state["stage42_gl_source_conversion_contract"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_gl_gate"]["verdict"],
        "gates": f"{payload['stage42_gl_gate']['passed']}/{payload['stage42_gl_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_source_conversion_contract(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    intake = read_json(INTAKE_JSON, {})
    manifest = read_json(MANIFEST_JSON, {})
    gh = read_json(GH_JSON, {})
    ej = read_json(EJ_JSON, {})
    rows = _contract_rows(intake, manifest, gh)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-GL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([INTAKE_JSON, MANIFEST_JSON, GH_JSON, EJ_JSON]),
        "input_status": {
            "intake_exists": INTAKE_JSON.exists(),
            "manifest_exists": MANIFEST_JSON.exists(),
            "gh_exists": GH_JSON.exists(),
            "ej_exists": EJ_JSON.exists(),
        },
        "current_facts": CURRENT_FACTS,
        "input_paths": {
            "intake": str(INTAKE_JSON),
            "manifest": str(MANIFEST_JSON),
            "calibrated_plan": str(GH_JSON),
            "guarded_launcher": str(EJ_JSON),
        },
        "contract_rows": rows,
        "summary": _summary(rows, manifest, gh, ej),
        "claim_boundary": CLAIM_BOUNDARY,
        "user_action_required_written": True,
    }
    payload["stage42_gl_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_source_conversion_contract()
