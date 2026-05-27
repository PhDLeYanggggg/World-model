from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
HY_JSON = OUT_DIR / "source_local_path_prefill_stage42.json"

PACKET_JSON = OUT_DIR / "source_terms_confirmation_packet_stage42.json"
PACKET_MD = OUT_DIR / "source_terms_confirmation_packet_stage42.md"
TEMPLATE_JSON = OUT_DIR / "source_terms_confirmation_template_stage42.json"
READINESS_JSON = OUT_DIR / "source_terms_readiness_validation_stage42.json"
READINESS_MD = OUT_DIR / "source_terms_readiness_validation_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_terms_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hz_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_SUMMARY = Path("README_M3W_CURRENT_MASTER_SUMMARY_ZH.md")
ROUTES_SUMMARY = Path("README_M3W_RESEARCH_ROUTES_FAILURES_SUCCESSES_2026_05_27_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_MATRIX = OUT_DIR / "paper_ready_evidence_matrix_stage42.md"

SECTION = "STAGE42_HZ_SOURCE_TERMS_CONFIRMATION_PACKET"
SOURCE = "fresh_stage42_hz_source_terms_confirmation_packet_from_hy_prefill"

REQUIRED_CONFIRMATION_FIELDS = [
    "official_source_url_confirmed",
    "local_path_confirmed",
    "source_identity_confirmed",
    "terms_accepted_by_user",
    "terms_acceptance_date",
    "allowed_use",
    "derived_data_allowed",
    "redistribution_allowed",
    "citation_required",
    "confirmed_by_user",
]

POSITIVE_CONFIRMATION_FIELDS = {
    "official_source_url_confirmed",
    "local_path_confirmed",
    "source_identity_confirmed",
    "terms_accepted_by_user",
}

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HZ 只生成 source/terms confirmation packet 和 readiness validator，不下载、不转换、不训练、不评估。",
    "local path found 不等于 legal terms accepted，不等于 official source identity confirmed。",
    "future endpoints / waypoints 只作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _existing_best_summary(row: Mapping[str, Any]) -> Mapping[str, Any]:
    best = row.get("best_local_path_candidate", "")
    for summary in row.get("path_summaries", []):
        if summary.get("path") == best:
            return summary
    return {}


def _packet_row(row: Mapping[str, Any]) -> dict[str, Any]:
    best_summary = _existing_best_summary(row)
    return {
        "dataset_id": row.get("dataset_id", ""),
        "name": row.get("name", row.get("dataset_id", "")),
        "domain": row.get("domain", ""),
        "official_url": row.get("official_url", ""),
        "best_local_path_candidate": row.get("best_local_path_candidate", ""),
        "local_path_found": bool(row.get("local_path_found", False)),
        "source_identity_hint": row.get("source_identity_hint", ""),
        "technical_parseability": {
            "file_count": int(best_summary.get("file_count", 0) or 0),
            "total_size_bytes": int(best_summary.get("total_size_bytes", 0) or 0),
            "first_hash": best_summary.get("first_hash", ""),
            "has_homography_file": bool(best_summary.get("has_homography_file", False)),
            "has_obsmat": bool(best_summary.get("has_obsmat", False)),
            "has_video": bool(best_summary.get("has_video", False)),
            "has_reference_image": bool(best_summary.get("has_reference_image", False)),
            "has_ndjson": bool(best_summary.get("has_ndjson", False)),
            "has_zip": bool(best_summary.get("has_zip", False)),
            "homography_example": best_summary.get("homography_example", ""),
            "trajectory_example": best_summary.get("trajectory_example", ""),
            "video_example": best_summary.get("video_example", ""),
        },
        "potential_after_terms": {
            "estimated_t50_windows": int(row.get("estimated_t50_windows_after_terms", 0) or 0),
            "estimated_t100_windows": int(row.get("estimated_t100_windows_after_terms", 0) or 0),
            "source_cv_after_terms": bool(row.get("source_cv_after_terms", False)),
        },
        "confirmation_required": REQUIRED_CONFIRMATION_FIELDS,
        "conversion_ready_now": False,
        "conversion_allowed_now": False,
        "converted_now": False,
        "evaluated_now": False,
        "claim_boundary": "technical path only; legal/source confirmation required before guarded conversion",
    }


def _template_row(packet_row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": packet_row["dataset_id"],
        "official_url": packet_row["official_url"],
        "local_path": packet_row["best_local_path_candidate"],
        "source_identity": "",
        "official_source_url_confirmed": False,
        "local_path_confirmed": False,
        "source_identity_confirmed": False,
        "terms_accepted_by_user": False,
        "terms_acceptance_date": "",
        "allowed_use": "",
        "derived_data_allowed": None,
        "redistribution_allowed": None,
        "citation_required": None,
        "confirmed_by_user": "",
        "notes": "Fill this row only after reading and accepting the official dataset/source terms. Do not use OpenTraj mirror identity as official permission unless that is explicitly confirmed.",
    }


def _missing_fields(template_row: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for field in REQUIRED_CONFIRMATION_FIELDS:
        value = template_row.get(field)
        if field in POSITIVE_CONFIRMATION_FIELDS:
            if value is not True:
                missing.append(field)
        elif isinstance(value, bool):
            continue
        elif value is None or value == "":
            missing.append(field)
    return missing


def _validate_readiness(packet_rows: list[Mapping[str, Any]], template_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    template_by_id = {row.get("dataset_id"): row for row in template_rows}
    rows = []
    for packet_row in packet_rows:
        dataset_id = packet_row["dataset_id"]
        template_row = template_by_id.get(dataset_id, {})
        missing = _missing_fields(template_row)
        local_path_matches = template_row.get("local_path") == packet_row.get("best_local_path_candidate")
        ready = (
            bool(packet_row.get("local_path_found", False))
            and local_path_matches
            and not missing
            and bool(packet_row.get("official_url", ""))
        )
        rows.append(
            {
                "dataset_id": dataset_id,
                "domain": packet_row.get("domain", ""),
                "local_path_found": bool(packet_row.get("local_path_found", False)),
                "local_path_matches_packet": bool(local_path_matches),
                "missing_fields": missing,
                "conversion_ready": bool(ready),
                "conversion_allowed_now": bool(ready),
                "reason": "ready_for_guarded_conversion_dry_run" if ready else "source_terms_or_identity_confirmation_missing",
            }
        )
    return {
        "rows": rows,
        "ready_count": sum(1 for row in rows if row["conversion_ready"]),
        "blocked_count": sum(1 for row in rows if not row["conversion_ready"]),
    }


def _build_payload(hy: Mapping[str, Any]) -> dict[str, Any]:
    packet_rows = [_packet_row(row) for row in hy.get("prefill_rows", [])]
    template_rows = [_template_row(row) for row in packet_rows]
    readiness = _validate_readiness(packet_rows, template_rows)
    return {
        "stage": "Stage42-HZ",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([HY_JSON]),
        "current_facts": CURRENT_FACTS,
        "input_hy_verdict": hy.get("stage42_hy_gate", {}).get("verdict", ""),
        "packet_rows": packet_rows,
        "template_rows": template_rows,
        "readiness": readiness,
        "summary": {
            "targets": len(packet_rows),
            "local_paths_found": sum(1 for row in packet_rows if row["local_path_found"]),
            "ready_for_guarded_conversion_now": readiness["ready_count"],
            "blocked_by_user_terms_or_source_identity": readiness["blocked_count"],
            "potential_t50_after_terms": sum(int(row["potential_after_terms"]["estimated_t50_windows"]) for row in packet_rows),
            "potential_t100_after_terms": sum(int(row["potential_after_terms"]["estimated_t100_windows"]) for row in packet_rows),
        },
        "actions": {"downloaded": False, "converted": False, "trained": False, "evaluated": False},
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def _gate(payload: Mapping[str, Any], readme_updates: Mapping[str, Any]) -> dict[str, Any]:
    rows = payload["packet_rows"]
    template = payload["template_rows"]
    readiness = payload["readiness"]
    gates = {
        "hy_input_present": HY_JSON.exists(),
        "hy_input_has_rows": len(rows) >= 5,
        "packet_written": PACKET_JSON.exists() and PACKET_MD.exists(),
        "template_written": TEMPLATE_JSON.exists(),
        "readiness_written": READINESS_JSON.exists() and READINESS_MD.exists(),
        "all_prefill_rows_included": len(rows) == len(template) and len(rows) == len(readiness["rows"]),
        "local_paths_preserved": all("best_local_path_candidate" in row for row in rows),
        "official_urls_preserved": all(bool(row.get("official_url")) for row in rows),
        "confirmation_fields_blank": all(row.get("terms_accepted_by_user") is False for row in template),
        "conversion_ready_false_until_user_confirmation": readiness["ready_count"] == 0,
        "legal_block_preserved": readiness["blocked_count"] == len(rows),
        "potential_windows_carried": payload["summary"]["potential_t50_after_terms"] > 0,
        "no_download": payload["actions"]["downloaded"] is False,
        "no_conversion": payload["actions"]["converted"] is False,
        "no_training": payload["actions"]["trained"] is False,
        "no_evaluation": payload["actions"]["evaluated"] is False,
        "user_action_written": USER_ACTION_MD.exists(),
        "readmes_updated": bool(readme_updates.get("readmes_updated", False)),
        "paper_matrix_updated": bool(readme_updates.get("paper_matrix_updated", False)),
        "no_metric_seconds_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_not_enabled": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_hz_source_terms_confirmation_packet_pass" if passed == total else "stage42_hz_source_terms_confirmation_packet_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _packet_table(rows: list[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| dataset | domain | local path | parseable hints | t50 after terms | t100 after terms | conversion now |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        tech = row["technical_parseability"]
        hints = []
        for key, label in [
            ("has_obsmat", "obsmat"),
            ("has_ndjson", "ndjson"),
            ("has_homography_file", "homography"),
            ("has_video", "video"),
            ("has_reference_image", "reference_image"),
            ("has_zip", "zip"),
        ]:
            if tech.get(key):
                hints.append(label)
        lines.append(
            f"| `{row['dataset_id']}` | `{row['domain']}` | `{row['best_local_path_candidate'] or 'not_found'}` | "
            f"{', '.join(hints) or 'none'} | {row['potential_after_terms']['estimated_t50_windows']} | "
            f"{row['potential_after_terms']['estimated_t100_windows']} | `{row['conversion_ready_now']}` |"
        )
    return lines


def _write_reports(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    write_json(PACKET_JSON, payload)
    write_json(TEMPLATE_JSON, {"source": SOURCE, "instructions": "User-fillable confirmation template. Do not set true unless terms/source identity/local path are confirmed.", "confirmations": payload["template_rows"]})
    write_json(READINESS_JSON, {"source": SOURCE, "readiness": payload["readiness"], "claim_boundary": payload["claim_boundary"]})
    packet_lines = [
        "# Stage42-HZ Source Terms Confirmation Packet",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate.get('passed', 'pending')} / {gate.get('total', 'pending')}`",
        f"- verdict: `{gate.get('verdict', 'pending')}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Packet Rows",
        "",
        *_packet_table(payload["packet_rows"]),
        "",
        "## Interpretation",
        "",
        "- This packet converts the HY local-path prefill into a user-confirmable source/terms checklist.",
        "- It intentionally keeps every source blocked until the user confirms terms, local path, official/source identity, and allowed use.",
        "- No conversion, no training, no evaluation, no metric/time claim, no Stage5C, and no SMC occurred.",
    ]
    write_md(PACKET_MD, packet_lines)
    readiness_lines = [
        "# Stage42-HZ Source Terms Readiness Validation",
        "",
        f"- source: `{payload['source']}`",
        f"- ready_for_guarded_conversion_now: `{payload['readiness']['ready_count']}`",
        f"- blocked: `{payload['readiness']['blocked_count']}`",
        "",
        "| dataset | ready | reason | missing fields |",
        "| --- | ---: | --- | --- |",
        *[
            f"| `{row['dataset_id']}` | `{row['conversion_ready']}` | `{row['reason']}` | {', '.join(row['missing_fields']) or 'none'} |"
            for row in payload["readiness"]["rows"]
        ],
        "",
        "Conversion remains blocked because this generated template is intentionally blank and must be user-confirmed before any guarded conversion.",
    ]
    write_md(READINESS_MD, readiness_lines)
    user_lines = [
        "# User Action Required: Stage42-HZ Source Terms Confirmation",
        "",
        "请复制或编辑下面的 JSON 模板，在你确认官方页面、条款、source identity、本地路径和允许用途后填写。未确认前不允许 guarded conversion。",
        "",
        f"- template: `{TEMPLATE_JSON}`",
        f"- readiness report: `{READINESS_MD}`",
        "",
        "必须确认的字段：",
        "",
        *[f"- `{field}`" for field in REQUIRED_CONFIRMATION_FIELDS],
        "",
        "注意：OpenTraj 本地镜像不自动等于官方许可。只有你确认该路径和官方条款匹配后，才能进入后续 source-specific guarded conversion。",
    ]
    write_md(USER_ACTION_MD, user_lines)
    write_md(
        GATE_MD,
        [
            "# Stage42-HZ Gate",
            "",
            f"- verdict: `{gate.get('verdict', 'pending')}`",
            f"- passed: `{gate.get('passed', 'pending')} / {gate.get('total', 'pending')}`",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(ok)}` |" for name, ok in gate.get("gates", {}).items()],
        ],
    )


def _refresh_lines(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> list[str]:
    return [
        "## Stage42-HZ Source Terms Confirmation Packet",
        "",
        f"- source: `{payload['source']}`",
        "- role: turn HY local path prefill into a user-confirmable source/terms packet and readiness validator.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- sources in packet: `{payload['summary']['targets']}`; ready for guarded conversion now: `{payload['summary']['ready_for_guarded_conversion_now']}`.",
        f"- potential after-terms t50/t100 windows preserved: `{payload['summary']['potential_t50_after_terms']}` / `{payload['summary']['potential_t100_after_terms']}`.",
        "- Remaining blocker: user must fill and confirm terms/source/local-path fields before conversion.",
        "- Claim boundary: no download, no conversion, no training, no evaluation, no metric/seconds-level claim, no Stage5C, no SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, bool]:
    lines = _refresh_lines(payload, gate)
    readme_paths = [README_RESULTS, M3W_README, MASTER_SUMMARY, ROUTES_SUMMARY]
    for path in readme_paths:
        _replace_section(path, SECTION, lines)
    matrix_lines = [
        "## Stage42-HZ Source Terms Confirmation Packet",
        "",
        "- HZ adds a user-fillable source/terms confirmation template and readiness validator.",
        f"- gate: `{gate['passed']} / {gate['total']}`.",
        f"- ready for guarded conversion now: `{payload['summary']['ready_for_guarded_conversion_now']}`.",
        "- This reduces A-stage friction but does not close legal/source blockers or authorize metric/time conversion.",
    ]
    _replace_section(PAPER_MATRIX, SECTION, matrix_lines)
    return {
        "readmes_updated": all(SECTION in path.read_text(encoding="utf-8") for path in readme_paths),
        "paper_matrix_updated": SECTION in PAPER_MATRIX.read_text(encoding="utf-8"),
    }


def _refresh_state(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-HZ source terms confirmation packet"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_hz_source_terms_confirmation_packet"] = {
        "source": payload["source"],
        "packet": str(PACKET_MD),
        "packet_json": str(PACKET_JSON),
        "template": str(TEMPLATE_JSON),
        "readiness": str(READINESS_MD),
        "readiness_json": str(READINESS_JSON),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "gates": f"{gate['passed']}/{gate['total']}",
        "verdict": gate["verdict"],
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [PACKET_MD, PACKET_JSON, TEMPLATE_JSON, READINESS_MD, READINESS_JSON, GATE_MD, USER_ACTION_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(RESEARCH_STATE, state)


def run_stage42_source_terms_packet() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    hy = read_json(HY_JSON, {})
    payload = _build_payload(hy)
    pending_gate = {"passed": "pending", "total": "pending", "verdict": "pending", "gates": {}}
    _write_reports(payload, pending_gate)
    readme_updates = _refresh_readmes(payload, pending_gate | {"passed": "pending", "total": "pending", "verdict": "pending"})
    gate = _gate(payload, readme_updates)
    _write_reports(payload, gate)
    readme_updates = _refresh_readmes(payload, gate)
    gate = _gate(payload, readme_updates)
    payload["stage42_hz_gate"] = gate
    payload["readme_updates"] = readme_updates
    write_json(PACKET_JSON, payload)
    write_json(READINESS_JSON, {"source": SOURCE, "readiness": payload["readiness"], "stage42_hz_gate": gate, "claim_boundary": payload["claim_boundary"]})
    _refresh_state(payload, gate)
    return payload


if __name__ == "__main__":
    result = run_stage42_source_terms_packet()
    gate = result["stage42_hz_gate"]
    print(f"Stage42-HZ source terms packet: {gate['verdict']} ({gate['passed']}/{gate['total']})")
