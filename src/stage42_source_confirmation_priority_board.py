from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
CONTRACT_JSON = OUT_DIR / "source_conversion_contract_stage42.json"
HARNESS_JSON = OUT_DIR / "guarded_conversion_harness_stage42.json"
CALIBRATED_PLAN_JSON = OUT_DIR / "calibrated_post_confirmation_subset_plan_stage42.json"
CALIBRATION_MANIFEST_JSON = OUT_DIR / "calibration_candidate_manifest_stage42.json"

REPORT_JSON = OUT_DIR / "source_confirmation_priority_board_stage42.json"
REPORT_MD = OUT_DIR / "source_confirmation_priority_board_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_confirmation_priority_board_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gn_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
ONE_FILE_SUMMARY = Path("README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_gn_source_confirmation_priority_board"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-GN 只排序 source/legal/calibration unblock 优先级；不下载、不转换、不训练、不评估。",
    "contract_ready_now=0 时，任何 post-confirmation opportunity 都不能写成 converted dataset 或 evaluation result。",
    "用户必须亲自确认 official terms、allowed use、local path、source identity；agent 不能代填 acceptance。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon；不能写成 seconds-level，除非未来 source-specific guard 通过。",
    "dataset-local/raw-frame 不能写成 global metric；restricted source-specific metric/time subset 也必须等 legal conversion 后再审计。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "restricted_metric_time_claim_allowed_now": False,
    "converted_dataset_claim_allowed": False,
    "download_executed": False,
    "conversion_executed": False,
    "feature_store_built": False,
    "training_executed": False,
    "evaluation_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}

REQUIRED_USER_FIELDS = [
    "terms_accepted_by_user",
    "terms_acceptance_date",
    "official_terms_url",
    "accepted_terms_version_or_access_date",
    "allowed_use",
    "redistribution_allowed",
    "derived_data_allowed",
    "local_path",
    "source_identity",
    "confirmed_by_user",
]


def _calibration_by_dataset(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in manifest.get("candidate_rows", []):
        if isinstance(row, Mapping):
            out[str(row.get("dataset_id", ""))] = row
    return out


def _plan_rows_by_dataset(plan: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    for row in plan.get("plan_rows", []):
        if isinstance(row, Mapping):
            out.setdefault(str(row.get("dataset_id", "")), []).append(row)
    return out


def _path_score(paths: list[Mapping[str, Any]]) -> int:
    raw_found = any(path.get("exists") and path.get("is_raw_source_candidate") for path in paths)
    derived_found = any(path.get("exists") and path.get("is_derived_or_cache") for path in paths)
    return (200 if raw_found else 0) + (25 if derived_found else 0)


def _priority_score(row: Mapping[str, Any], plan_rows: list[Mapping[str, Any]], calibration: Mapping[str, Any]) -> float:
    after_terms = row.get("after_terms_potential", {}) if isinstance(row.get("after_terms_potential"), Mapping) else {}
    calibrated = row.get("calibrated_subset_after_terms", {}) if isinstance(row.get("calibrated_subset_after_terms"), Mapping) else {}
    suggested_paths = row.get("suggested_local_paths_for_user_review", [])
    if not isinstance(suggested_paths, list):
        suggested_paths = []
    t50 = int(after_terms.get("estimated_t50_windows", 0) or 0)
    t100 = int(after_terms.get("estimated_t100_windows", 0) or 0)
    calibrated_t50 = int(calibrated.get("calibrated_t50_windows_after_terms", 0) or 0)
    calibrated_t100 = int(calibrated.get("calibrated_t100_windows_after_terms", 0) or 0)
    calibrated_sources = int(calibrated.get("restricted_metric_time_candidates_after_terms", 0) or 0)
    source_cv = bool(after_terms.get("source_cv_after_terms"))
    metric_hint = bool(calibration.get("metric_time_subset_hint")) or any(
        bool(item.get("restricted_metric_time_candidate_after_terms")) for item in plan_rows
    )
    official_url = str(row.get("official_url", ""))
    official_url_score = 100 if official_url.startswith("http") else -200
    return (
        0.03 * t50
        + 0.08 * t100
        + 0.04 * calibrated_t50
        + 0.10 * calibrated_t100
        + 350 * calibrated_sources
        + (250 if source_cv else 0)
        + (350 if metric_hint else 0)
        + _path_score(suggested_paths)
        + official_url_score
    )


def _dataset_value_class(score: float, row: Mapping[str, Any]) -> str:
    calibrated = row.get("calibrated_subset_after_terms", {}) if isinstance(row.get("calibrated_subset_after_terms"), Mapping) else {}
    if int(calibrated.get("calibrated_t100_windows_after_terms", 0) or 0) > 0:
        return "calibrated_t50_t100_unlock"
    if int(calibrated.get("calibrated_t50_windows_after_terms", 0) or 0) > 0:
        return "calibrated_t50_unlock"
    if score >= 500:
        return "trajectory_window_unlock"
    return "low_or_diagnostic_unlock"


def _next_fields(row: Mapping[str, Any]) -> list[str]:
    confirmation = row.get("confirmation", {}) if isinstance(row.get("confirmation"), Mapping) else {}
    missing = list(confirmation.get("missing_fields", [])) if isinstance(confirmation.get("missing_fields"), list) else []
    return [field for field in REQUIRED_USER_FIELDS if field in missing]


def _rank_rows(
    contract: Mapping[str, Any],
    calibrated_plan: Mapping[str, Any],
    calibration_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_dataset_plan = _plan_rows_by_dataset(calibrated_plan)
    by_dataset_calibration = _calibration_by_dataset(calibration_manifest)
    ranked: list[dict[str, Any]] = []
    for contract_row in contract.get("contract_rows", []):
        if not isinstance(contract_row, Mapping):
            continue
        dataset_id = str(contract_row.get("dataset_id", ""))
        plan_rows = by_dataset_plan.get(dataset_id, [])
        calibration = by_dataset_calibration.get(dataset_id, {})
        score = _priority_score(contract_row, plan_rows, calibration)
        after_terms = contract_row.get("after_terms_potential", {}) if isinstance(contract_row.get("after_terms_potential"), Mapping) else {}
        calibrated = (
            contract_row.get("calibrated_subset_after_terms", {})
            if isinstance(contract_row.get("calibrated_subset_after_terms"), Mapping)
            else {}
        )
        suggested_paths = contract_row.get("suggested_local_paths_for_user_review", [])
        if not isinstance(suggested_paths, list):
            suggested_paths = []
        raw_paths = [path for path in suggested_paths if path.get("exists") and path.get("is_raw_source_candidate")]
        ranked.append(
            {
                "dataset_id": dataset_id,
                "domain": contract_row.get("domain", ""),
                "official_url": contract_row.get("official_url", ""),
                "contract_status": contract_row.get("contract_status", ""),
                "contract_conversion_ready_now": bool(contract_row.get("contract_conversion_ready_now")),
                "priority_score": round(score, 3),
                "value_class": _dataset_value_class(score, contract_row),
                "estimated_t50_windows_after_terms": int(after_terms.get("estimated_t50_windows", 0) or 0),
                "estimated_t100_windows_after_terms": int(after_terms.get("estimated_t100_windows", 0) or 0),
                "source_cv_after_terms": bool(after_terms.get("source_cv_after_terms")),
                "technical_ready_source_ids": list(after_terms.get("technical_ready_source_ids", []))
                if isinstance(after_terms.get("technical_ready_source_ids", []), list)
                else [],
                "calibrated_source_rows_after_terms": int(
                    calibrated.get("restricted_metric_time_candidates_after_terms", 0) or 0
                ),
                "calibrated_t50_windows_after_terms": int(calibrated.get("calibrated_t50_windows_after_terms", 0) or 0),
                "calibrated_t100_windows_after_terms": int(calibrated.get("calibrated_t100_windows_after_terms", 0) or 0),
                "candidate_source_ids_after_terms": list(calibrated.get("candidate_source_ids_after_terms", []))
                if isinstance(calibrated.get("candidate_source_ids_after_terms", []), list)
                else [],
                "metric_time_subset_hint": bool(calibration.get("metric_time_subset_hint")),
                "h_matrix_hints": int(calibration.get("h_matrix_hints", 0) or 0),
                "time_hints": int(calibration.get("time_hints", 0) or 0),
                "frame_stride_hints": int(calibration.get("frame_stride_hints", 0) or 0),
                "existing_raw_path_candidates": raw_paths,
                "missing_user_fields": _next_fields(contract_row),
                "blocked_reason": "missing_user_terms_path_source_identity"
                if not contract_row.get("contract_conversion_ready_now")
                else "",
                "minimum_next_action": "fill user_confirmation fields in source_terms_confirmation_intake_template_stage42.json and rerun validator/contract/harness",
                "result_source": "fresh_priority_from_cached_verified_stage42_gl_gm_gh_dv",
                "download_executed": False,
                "conversion_executed": False,
                "evaluation_executed": False,
            }
        )
    ranked.sort(key=lambda item: (-float(item["priority_score"]), str(item["dataset_id"])))
    for idx, row in enumerate(ranked, start=1):
        row["priority_rank"] = idx
    return ranked


def _summary(rows: list[Mapping[str, Any]], harness: Mapping[str, Any]) -> dict[str, Any]:
    top = rows[0] if rows else {}
    return {
        "source": SOURCE,
        "targets_ranked": len(rows),
        "ready_now": sum(1 for row in rows if row.get("contract_conversion_ready_now")),
        "blocked_now": sum(1 for row in rows if not row.get("contract_conversion_ready_now")),
        "top_priority_dataset": top.get("dataset_id", ""),
        "top_priority_domain": top.get("domain", ""),
        "top_priority_value_class": top.get("value_class", ""),
        "total_t50_after_terms": sum(int(row.get("estimated_t50_windows_after_terms", 0) or 0) for row in rows),
        "total_t100_after_terms": sum(int(row.get("estimated_t100_windows_after_terms", 0) or 0) for row in rows),
        "calibrated_t50_after_terms": sum(int(row.get("calibrated_t50_windows_after_terms", 0) or 0) for row in rows),
        "calibrated_t100_after_terms": sum(int(row.get("calibrated_t100_windows_after_terms", 0) or 0) for row in rows),
        "contract_ready_now_from_gm": harness.get("summary", {}).get("contract_ready_now", None),
        "gm_conversion_executed": harness.get("summary", {}).get("conversion_executed", False),
        "download_executed": False,
        "conversion_executed": False,
        "training_executed": False,
        "evaluation_executed": False,
        "next_best_user_action": "confirm UCY crowd official terms/local path/source identity first, then ETH/BIWI; TrajNet remains useful but h100-limited unless longer official raw sources are provided",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    c = payload["claim_boundary"]
    rows = payload["priority_rows"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "contract_loaded": payload.get("input_status", {}).get("contract_exists") is True,
        "harness_loaded": payload.get("input_status", {}).get("harness_exists") is True,
        "calibrated_plan_loaded": payload.get("input_status", {}).get("calibrated_plan_exists") is True,
        "calibration_manifest_loaded": payload.get("input_status", {}).get("calibration_manifest_exists") is True,
        "all_contract_rows_ranked": len(rows) == payload.get("input_status", {}).get("contract_row_count", -1),
        "blocked_rows_not_marked_ready": s["ready_now"] == 0 and s["blocked_now"] >= 1,
        "opportunity_windows_preserved": s["total_t50_after_terms"] >= s["calibrated_t50_after_terms"]
        and s["total_t100_after_terms"] >= s["calibrated_t100_after_terms"],
        "top_priority_actionable_after_user_confirmation": bool(s["top_priority_dataset"]) and rows[0]["missing_user_fields"],
        "no_download_conversion_training_eval": not (
            s["download_executed"] or s["conversion_executed"] or s["training_executed"] or s["evaluation_executed"]
        ),
        "user_action_written": payload.get("user_action_required_written") is True,
        "no_converted_or_metric_overclaim": c["converted_dataset_claim_allowed"] is False
        and c["restricted_metric_time_claim_allowed_now"] is False
        and c["global_metric_claim_allowed"] is False
        and c["global_seconds_claim_allowed"] is False,
        "stage5c_false": c["stage5c_executed"] is False,
        "smc_false": c["smc_enabled"] is False,
    }
    passed = sum(bool(value) for value in gates.values())
    total = len(gates)
    verdict = (
        "stage42_gn_source_confirmation_priority_board_pass"
        if passed == total
        else "stage42_gn_source_confirmation_priority_board_partial"
    )
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-GN Source Confirmation Priority Board",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_gn_gate']['passed']} / {payload['stage42_gn_gate']['total']}`",
        f"- verdict: `{payload['stage42_gn_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in s.items()],
        "",
        "## Ranked Source Confirmation Queue",
        "",
        "| rank | dataset | domain | value class | score | t50/t100 after terms | calibrated t50/t100 | raw path candidates | missing user fields |",
        "| ---: | --- | --- | --- | ---: | --- | --- | ---: | --- |",
    ]
    for row in payload["priority_rows"]:
        lines.append(
            f"| {row['priority_rank']} | `{row['dataset_id']}` | `{row['domain']}` | `{row['value_class']}` | "
            f"{row['priority_score']:.3f} | {row['estimated_t50_windows_after_terms']} / {row['estimated_t100_windows_after_terms']} | "
            f"{row['calibrated_t50_windows_after_terms']} / {row['calibrated_t100_windows_after_terms']} | "
            f"{len(row['existing_raw_path_candidates'])} | {', '.join(row['missing_user_fields']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `UCY` is first because it has the largest post-confirmation t50/t100 and calibrated subset opportunity already visible locally.",
            "- `ETH_UCY` is second because it has source-specific calibration value but much smaller unlocked row count.",
            "- `TrajNet` remains important for diversity, but current local material is short-snippet / h100-limited and cannot repair the long-horizon blocker by itself.",
            "- `AerialMPT / other_topdown` and `OpenTraj toolkit` are lower priority until official terms/source identity and parseable trajectory scope are confirmed.",
            "- This board is an unblock queue only: no data was converted, no evaluation ran, and no metric/seconds claim is allowed.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_gn_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-GN Source Confirmation Priority Board",
        "",
        "Fill only after checking the official dataset terms yourself. The agent must not accept terms for you.",
        "",
        "Recommended order:",
        "",
    ]
    for row in payload["priority_rows"][:5]:
        paths = row["existing_raw_path_candidates"]
        path_text = ", ".join(f"`{p.get('path', '')}`" for p in paths[:3]) or "`<user-provided verified local path>`"
        lines.extend(
            [
                f"## {row['priority_rank']}. {row['dataset_id']} ({row['domain']})",
                "",
                f"- official_url: `{row['official_url']}`",
                f"- value_class: `{row['value_class']}`",
                f"- post-confirmation t50/t100 opportunity: `{row['estimated_t50_windows_after_terms']} / {row['estimated_t100_windows_after_terms']}`",
                f"- calibrated t50/t100 opportunity: `{row['calibrated_t50_windows_after_terms']} / {row['calibrated_t100_windows_after_terms']}`",
                f"- suggested path to confirm, if it is truly the official/source dataset: {path_text}",
                f"- missing fields: {', '.join(row['missing_user_fields']) or 'none'}",
                "- after filling, run:",
                "  `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`",
                "  `.venv-pytorch/bin/python run_stage42_source_conversion_contract.py`",
                "  `.venv-pytorch/bin/python run_stage42_guarded_conversion_harness.py`",
                "",
            ]
        )
    lines.extend(
        [
            "Do not count this priority board as permission, conversion, feature store, no-leakage audit, source-CV, or model evidence.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_gn_gate"]
    return [
        "# Stage42-GN Gate",
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
        "## Stage42-GN Source Confirmation Priority Board",
        "",
        "- source: `fresh_stage42_gn_source_confirmation_priority_board`",
        "- role: ranks user-confirmation actions needed before any guarded source conversion can legally run.",
        f"- gate: `{payload['stage42_gn_gate']['passed']} / {payload['stage42_gn_gate']['total']}`; verdict `{payload['stage42_gn_gate']['verdict']}`.",
        f"- targets_ranked: `{s['targets_ranked']}`; ready_now: `{s['ready_now']}`; blocked_now: `{s['blocked_now']}`.",
        f"- top priority: `{s['top_priority_dataset']}` / `{s['top_priority_domain']}`; value class `{s['top_priority_value_class']}`.",
        f"- after-terms opportunity: t50 `{s['total_t50_after_terms']}`, t100 `{s['total_t100_after_terms']}`; calibrated t50/t100 `{s['calibrated_t50_after_terms']}` / `{s['calibrated_t100_after_terms']}`.",
        "- No download, conversion, feature-store build, no-leakage audit, source-CV, training, or evaluation was executed.",
        "- Boundary: this is a source/legal/calibration unblock queue only; not converted data, not metric/seconds evidence, not Stage5C, and not SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, ONE_FILE_SUMMARY]:
        _replace_section(path, "STAGE42_GN_SOURCE_CONFIRMATION_PRIORITY_BOARD", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-GN source confirmation priority board"
    state["current_verdict"] = payload["stage42_gn_gate"]["verdict"]
    state["stage42_gn_source_confirmation_priority_board"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_gn_gate"]["verdict"],
        "gates": f"{payload['stage42_gn_gate']['passed']}/{payload['stage42_gn_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_source_confirmation_priority_board(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    contract = read_json(CONTRACT_JSON, {})
    harness = read_json(HARNESS_JSON, {})
    calibrated_plan = read_json(CALIBRATED_PLAN_JSON, {})
    calibration_manifest = read_json(CALIBRATION_MANIFEST_JSON, {})
    rows = _rank_rows(contract, calibrated_plan, calibration_manifest)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-GN",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([CONTRACT_JSON, HARNESS_JSON, CALIBRATED_PLAN_JSON, CALIBRATION_MANIFEST_JSON]),
        "input_status": {
            "contract_exists": CONTRACT_JSON.exists(),
            "harness_exists": HARNESS_JSON.exists(),
            "calibrated_plan_exists": CALIBRATED_PLAN_JSON.exists(),
            "calibration_manifest_exists": CALIBRATION_MANIFEST_JSON.exists(),
            "contract_row_count": len(contract.get("contract_rows", [])),
        },
        "current_facts": CURRENT_FACTS,
        "priority_rows": rows,
        "summary": _summary(rows, harness),
        "claim_boundary": CLAIM_BOUNDARY,
        "user_action_required_written": True,
    }
    payload["stage42_gn_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_source_confirmation_priority_board()
