from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_full_waypoint_bridge_shape_audit import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

HJ_JSON = OUT_DIR / "restricted_metric_time_source_cv_preflight_stage42.json"
HK_JSON = OUT_DIR / "restricted_metric_time_eth_ucy_source_support_stage42.json"
HL_JSON = OUT_DIR / "restricted_metric_time_post_hk_claim_guard_stage42.json"

REPORT_JSON = OUT_DIR / "restricted_metric_time_terms_intake_v2_stage42.json"
REPORT_MD = OUT_DIR / "restricted_metric_time_terms_intake_v2_stage42.md"
TEMPLATE_JSON = OUT_DIR / "restricted_metric_time_terms_intake_v2_template_stage42.json"
MANIFEST_JSON = OUT_DIR / "restricted_metric_time_terms_intake_v2_manifest_stage42.json"
GATE_MD = OUT_DIR / "stage42_stage_hm_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_restricted_metric_time_terms_intake_v2_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
CURRENT_SUMMARY = Path("README_M3W_CURRENT_DETAILED_SUMMARY_2026_05_27_ZH.md")
A_JOURNAL_GAP = OUT_DIR / "a_journal_gap_stage42.md"
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_hm_restricted_metric_time_terms_intake_v2"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HM 是 restricted metric/time source-level terms intake v2，不下载、不转换、不训练、不评估。",
    "本阶段把 Stage42-HJ/HK 的 source-level UCY/ETH/ETH-Person 候选转成用户可填写的 terms/source identity template。",
    "空白 template、local file present、parseability、technical dry-run 都不等于 legal conversion readiness。",
    "future endpoint / future waypoint 只可作为 supervised/evaluation labels，不可作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 建 goals，不使用 test metrics 调 threshold。",
    "restricted seconds/metric wording 仍需 user terms confirmation、guarded conversion、no-leakage、source-CV、final test。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

REQUIRED_CONFIRMATION_FIELDS = [
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

OFFICIAL_URLS = {
    "ucy_crowd_original": "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
    "eth_biwi_original": "https://vision.ee.ethz.ch/datasets/",
    "eth_person_local_candidates": "user_verified_official_eth_person_source_terms_url_required",
}

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "restricted_metric_time_claim_allowed_now": False,
    "download_executed": False,
    "conversion_executed": False,
    "evaluation_executed": False,
    "training_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _gate_passed(payload: Mapping[str, Any], gate_key: str) -> bool:
    gate = payload.get(gate_key, {})
    return bool(gate and gate.get("passed") == gate.get("total") and int(gate.get("total", 0)) > 0)


def _user_confirmation_blank(official_url: str) -> dict[str, Any]:
    return {
        "terms_accepted_by_user": False,
        "terms_acceptance_date": "",
        "official_terms_url": "" if "required" in official_url else official_url,
        "accepted_terms_version_or_access_date": "",
        "allowed_use": "",
        "redistribution_allowed": "unknown",
        "derived_data_allowed": "unknown",
        "local_path": "",
        "source_identity": "",
        "confirmed_by_user": "",
        "notes": "",
    }


def _candidate_from_hj(row: Mapping[str, Any]) -> dict[str, Any]:
    target = str(row.get("terms_target_id", ""))
    official_url = OFFICIAL_URLS.get(target, "user_verified_official_terms_url_required")
    source_id = str(row.get("source_id", ""))
    return {
        "candidate_id": f"hj::{source_id}",
        "source_row_origin": "fresh_run_stage42_hj_source_cv_preflight",
        "source_id": source_id,
        "domain": row.get("domain", ""),
        "dataset": row.get("dataset", ""),
        "terms_target_id": target,
        "official_terms_url_hint": official_url,
        "trajectory_file": row.get("trajectory_file", ""),
        "trajectory_file_found": bool(row.get("trajectory_file_found")),
        "homography_parseable": bool(row.get("homography_parseable")),
        "annotation_fps_if_restricted": row.get("annotation_fps"),
        "annotation_timestep_seconds_if_restricted": row.get("annotation_timestep_seconds"),
        "h50_seconds_if_restricted": row.get("h50_seconds_if_restricted"),
        "h100_seconds_if_restricted": row.get("h100_seconds_if_restricted"),
        "t50_windows_after_terms": int(row.get("t50_windows", 0) or 0),
        "t100_windows_after_terms": int(row.get("t100_windows", 0) or 0),
        "source_cv_usable_after_terms": bool(row.get("source_cv_usable_after_terms")),
        "legal_ready_now": False,
        "restricted_metric_time_ready_now": False,
        "user_confirmation": _user_confirmation_blank(official_url),
    }


def _candidate_from_hk(row: Mapping[str, Any]) -> dict[str, Any]:
    source_id = str(row.get("source_id", ""))
    is_eth_person = source_id.startswith("ETH-Person_")
    target = "eth_person_local_candidates" if is_eth_person else "eth_biwi_original"
    official_url = OFFICIAL_URLS[target]
    return {
        "candidate_id": f"hk::{source_id}",
        "source_row_origin": "cached_verified_stage42_hk_eth_ucy_source_support_preflight",
        "source_id": source_id,
        "domain": "ETH_UCY",
        "dataset": "ETH-Person" if is_eth_person else "ETH",
        "terms_target_id": target,
        "official_terms_url_hint": official_url,
        "trajectory_file": row.get("relative_path", ""),
        "trajectory_file_found": True,
        "homography_parseable": False,
        "annotation_fps_if_restricted": None,
        "annotation_timestep_seconds_if_restricted": None,
        "h50_seconds_if_restricted": None,
        "h100_seconds_if_restricted": None,
        "t50_windows_after_terms": int(row.get("t50_windows", 0) or 0),
        "t100_windows_after_terms": int(row.get("t100_windows", 0) or 0),
        "source_cv_usable_after_terms": bool(row.get("usable_after_terms")),
        "legal_ready_now": False,
        "restricted_metric_time_ready_now": False,
        "user_confirmation": _user_confirmation_blank(official_url),
    }


def _candidate_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row.get("source_id", "")), str(row.get("domain", "")), str(row.get("trajectory_file", ""))


def _candidate_rows(hj: Mapping[str, Any], hk: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in hj.get("source_rows", []):
        row = _candidate_from_hj(raw)
        key = _candidate_key(row)
        if key not in seen:
            seen.add(key)
            rows.append(row)
    for raw in hk.get("eth_ucy_augmented_sources", []):
        row = _candidate_from_hk(raw)
        key = _candidate_key(row)
        if key not in seen:
            seen.add(key)
            rows.append(row)
    rows.sort(
        key=lambda item: (
            -int(item.get("source_cv_usable_after_terms", False)),
            -int(item.get("t100_windows_after_terms", 0) or 0),
            -int(item.get("t50_windows_after_terms", 0) or 0),
            str(item.get("source_id", "")),
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["priority_rank"] = idx
    return rows


def _template_from_candidates(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "source": SOURCE,
        "purpose": "Manual source-level confirmation for restricted metric/time conversion candidates. The agent must not fill user confirmation fields.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "required_confirmation_fields": REQUIRED_CONFIRMATION_FIELDS,
        "datasets": rows,
        "non_claims": [
            "This template does not grant permission.",
            "This template does not convert or evaluate data.",
            "This template does not allow metric/seconds-level claims.",
            "This template does not execute Stage5C or SMC.",
        ],
    }


def _load_or_write_template(candidates: list[Mapping[str, Any]], *, force_rebuild_template: bool) -> dict[str, Any]:
    if TEMPLATE_JSON.exists() and not force_rebuild_template:
        return read_json(TEMPLATE_JSON, {})
    template = _template_from_candidates([dict(row) for row in candidates])
    write_json(TEMPLATE_JSON, template)
    return template


def _normalise_template_rows(template: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("candidate_id", "")): row for row in template.get("datasets", [])}


def validate_candidate(candidate: Mapping[str, Any], template_row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    row = template_row or candidate
    user = dict(row.get("user_confirmation", {}))
    blockers: list[str] = []
    official_hint = str(candidate.get("official_terms_url_hint", ""))
    official_url = str(user.get("official_terms_url", "")).strip()
    if "required" in official_hint:
        blockers.append("official_terms_url_requires_user_verified_official_source")
    if not official_url:
        blockers.append("official_terms_url_missing")
    elif official_url != official_hint and "required" not in official_hint:
        blockers.append("official_terms_url_mismatch")
    if user.get("terms_accepted_by_user") is not True:
        blockers.append("terms_not_accepted_by_user")
    if not str(user.get("terms_acceptance_date", "")).strip():
        blockers.append("terms_acceptance_date_missing")
    if not str(user.get("accepted_terms_version_or_access_date", "")).strip():
        blockers.append("accepted_terms_version_or_access_date_missing")
    if not str(user.get("allowed_use", "")).strip() or str(user.get("allowed_use", "")).strip().lower() == "unknown":
        blockers.append("allowed_use_missing_or_unknown")
    if str(user.get("redistribution_allowed", "unknown")).strip().lower() == "unknown":
        blockers.append("redistribution_allowed_unknown")
    if str(user.get("derived_data_allowed", "unknown")).strip().lower() == "unknown":
        blockers.append("derived_data_allowed_unknown")
    local_path = str(user.get("local_path", "")).strip()
    if not local_path:
        blockers.append("local_path_missing")
    elif not Path(local_path).exists():
        blockers.append("local_path_not_found")
    if not str(user.get("source_identity", "")).strip():
        blockers.append("source_identity_missing")
    if not str(user.get("confirmed_by_user", "")).strip():
        blockers.append("confirmed_by_user_missing")
    if not candidate.get("source_cv_usable_after_terms"):
        blockers.append("source_cv_not_usable_even_after_terms")
    conversion_ready = not blockers
    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "source_id": candidate.get("source_id", ""),
        "domain": candidate.get("domain", ""),
        "terms_target_id": candidate.get("terms_target_id", ""),
        "t50_windows_after_terms": int(candidate.get("t50_windows_after_terms", 0) or 0),
        "t100_windows_after_terms": int(candidate.get("t100_windows_after_terms", 0) or 0),
        "source_cv_usable_after_terms": bool(candidate.get("source_cv_usable_after_terms")),
        "terms_accepted_by_user": user.get("terms_accepted_by_user") is True,
        "conversion_ready": conversion_ready,
        "restricted_metric_time_ready_now": False,
        "converted_now": False,
        "evaluated_now": False,
        "blockers": blockers,
        "next_action": "ready for future guarded conversion" if conversion_ready else "complete user-confirmed terms/path/source fields before conversion",
    }


def _validate_template(candidates: list[Mapping[str, Any]], template: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_id = _normalise_template_rows(template)
    return [validate_candidate(candidate, by_id.get(str(candidate.get("candidate_id", "")))) for candidate in candidates]


def _source_cv_domains(validations: list[Mapping[str, Any]], *, ready_only: bool) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in validations:
        if ready_only and not row.get("conversion_ready"):
            continue
        if not row.get("source_cv_usable_after_terms"):
            continue
        domain = str(row.get("domain", ""))
        counts[domain] = counts.get(domain, 0) + 1
    return counts


def _build_payload(*, force_rebuild_template: bool = False) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    hj = read_json(HJ_JSON, {})
    hk = read_json(HK_JSON, {})
    hl = read_json(HL_JSON, {})
    candidates = _candidate_rows(hj, hk)
    template = _load_or_write_template(candidates, force_rebuild_template=force_rebuild_template)
    validations = _validate_template(candidates, template)
    ready = [row for row in validations if row["conversion_ready"]]
    blocked = [row for row in validations if not row["conversion_ready"]]
    after_terms_domains = _source_cv_domains(validations, ready_only=False)
    ready_domains = _source_cv_domains(validations, ready_only=True)
    summary = {
        "source": SOURCE,
        "hj_verdict": hj.get("stage42_hj_gate", {}).get("verdict"),
        "hk_verdict": hk.get("stage42_hk_gate", {}).get("verdict"),
        "hl_verdict": hl.get("stage42_hl_gate", {}).get("verdict"),
        "source_level_candidates": len(candidates),
        "source_cv_usable_after_terms_candidates": sum(1 for row in candidates if row.get("source_cv_usable_after_terms")),
        "conversion_ready_candidates_now": len(ready),
        "blocked_candidates_now": len(blocked),
        "after_terms_domains_with_source_cv_candidate_count": after_terms_domains,
        "ready_now_domains_with_source_cv_candidate_count": ready_domains,
        "after_terms_total_t50_windows": sum(int(row.get("t50_windows_after_terms", 0) or 0) for row in candidates),
        "after_terms_total_t100_windows": sum(int(row.get("t100_windows_after_terms", 0) or 0) for row in candidates),
        "ready_now_t50_windows": sum(int(row.get("t50_windows_after_terms", 0) or 0) for row in ready),
        "ready_now_t100_windows": sum(int(row.get("t100_windows_after_terms", 0) or 0) for row in ready),
        "template_path": str(TEMPLATE_JSON),
        "manifest_path": str(MANIFEST_JSON),
        "download_executed": False,
        "conversion_executed": False,
        "evaluation_executed": False,
        "training_executed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    manifest = {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ready_candidates": ready,
        "blocked_candidates": blocked,
        "conversion_executed": False,
        "evaluation_executed": False,
        "metric_seconds_claim_allowed_now": False,
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-HM Restricted Metric/Time Source-Level Terms Intake v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([HJ_JSON, HK_JSON, HL_JSON, TEMPLATE_JSON]),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "hj_gate_passed": _gate_passed(hj, "stage42_hj_gate"),
            "hk_gate_passed": _gate_passed(hk, "stage42_hk_gate"),
            "hl_gate_passed": _gate_passed(hl, "stage42_hl_gate"),
        },
        "candidate_rows": candidates,
        "template_source": template.get("source", ""),
        "validations": validations,
        "manifest": manifest,
        "summary": summary,
        "claim_boundary": CLAIM_BOUNDARY,
        "user_action_required": [
            "Open outputs/stage42_long_research/restricted_metric_time_terms_intake_v2_template_stage42.json.",
            "For each desired UCY/ETH/ETH-Person source, manually verify official terms, source identity, allowed use, local path, redistribution, and derived-data rules.",
            "Rerun `.venv-pytorch/bin/python run_stage42_restricted_metric_time_terms_intake_v2.py --validate-only`.",
            "Only if ready candidates appear, run a future guarded conversion/no-leakage/source-CV stage.",
        ],
    }
    payload["stage42_hm_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    c = payload["claim_boundary"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "hj_input_passed": payload["inputs"]["hj_gate_passed"] is True,
        "hk_input_passed": payload["inputs"]["hk_gate_passed"] is True,
        "hl_input_passed": payload["inputs"]["hl_gate_passed"] is True,
        "source_level_candidates_present": s["source_level_candidates"] >= 10,
        "ucy_and_eth_ucy_after_terms_present": set(s["after_terms_domains_with_source_cv_candidate_count"]).issuperset({"UCY", "ETH_UCY"}),
        "after_terms_t50_t100_support_present": s["after_terms_total_t50_windows"] > 0
        and s["after_terms_total_t100_windows"] > 0,
        "ready_now_zero_until_user_confirms": s["conversion_ready_candidates_now"] == 0
        and s["ready_now_t50_windows"] == 0
        and s["ready_now_t100_windows"] == 0,
        "template_written": Path(s["template_path"]).exists(),
        "manifest_built": "manifest" in payload and isinstance(payload.get("manifest"), Mapping),
        "blocked_candidates_preserved": s["blocked_candidates_now"] == s["source_level_candidates"],
        "no_download_conversion_training_eval": not (
            s["download_executed"] or s["conversion_executed"] or s["training_executed"] or s["evaluation_executed"]
        ),
        "no_metric_seconds_claim": c["global_metric_claim_allowed"] is False
        and c["global_seconds_claim_allowed"] is False
        and c["restricted_metric_time_claim_allowed_now"] is False,
        "stage5c_false": c["stage5c_executed"] is False,
        "smc_false": c["smc_enabled"] is False,
    }
    passed = sum(1 for value in gates.values() if value)
    total = len(gates)
    verdict = (
        "stage42_hm_restricted_metric_time_terms_intake_v2_pass_blocked_until_user_confirmation"
        if passed == total
        else "stage42_hm_restricted_metric_time_terms_intake_v2_partial"
    )
    return {"source": payload.get("source", SOURCE), "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hm_gate"]
    s = payload["summary"]
    lines = [
        "# Stage42-HM Restricted Metric/Time Source-Level Terms Intake v2",
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
        "## Summary",
        "",
        f"- source_level_candidates: `{s['source_level_candidates']}`",
        f"- source_cv_usable_after_terms_candidates: `{s['source_cv_usable_after_terms_candidates']}`",
        f"- conversion_ready_candidates_now: `{s['conversion_ready_candidates_now']}`",
        f"- blocked_candidates_now: `{s['blocked_candidates_now']}`",
        f"- after_terms_domains_with_source_cv_candidate_count: `{s['after_terms_domains_with_source_cv_candidate_count']}`",
        f"- ready_now_domains_with_source_cv_candidate_count: `{s['ready_now_domains_with_source_cv_candidate_count']}`",
        f"- after_terms_total_t50/t100_windows: `{s['after_terms_total_t50_windows']}` / `{s['after_terms_total_t100_windows']}`",
        f"- ready_now_t50/t100_windows: `{s['ready_now_t50_windows']}` / `{s['ready_now_t100_windows']}`",
        f"- template_path: `{s['template_path']}`",
        f"- manifest_path: `{s['manifest_path']}`",
        "",
        "## Candidate Table",
        "",
        "| rank | source | domain | target | t50 after terms | t100 after terms | source-CV usable after terms | ready now | blockers |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    validation_by_id = {row["candidate_id"]: row for row in payload["validations"]}
    for row in payload["candidate_rows"]:
        val = validation_by_id.get(row["candidate_id"], {})
        lines.append(
            f"| {row['priority_rank']} | `{row['source_id']}` | `{row['domain']}` | `{row['terms_target_id']}` | "
            f"{row['t50_windows_after_terms']} | {row['t100_windows_after_terms']} | "
            f"{row['source_cv_usable_after_terms']} | {val.get('conversion_ready', False)} | "
            f"{', '.join(val.get('blockers', [])) or 'none'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- UCY and ETH_UCY both have source-level candidates that could support restricted metric/time source-CV after user terms confirmation.",
        "- Current ready-now count is zero because the template is intentionally blank and ETH-Person official/source terms still require user verification.",
        "- This is a source-level intake and validator artifact, not a conversion/evaluation result and not a metric/seconds claim.",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hm_gate"]
    return [
        "# Stage42-HM Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _render_manifest(payload: Mapping[str, Any]) -> list[str]:
    m = payload["manifest"]
    return [
        "# Stage42-HM Source-Level Terms Intake v2 Manifest",
        "",
        f"- source: `{m['source']}`",
        f"- ready_candidates: `{len(m['ready_candidates'])}`",
        f"- blocked_candidates: `{len(m['blocked_candidates'])}`",
        f"- conversion_executed: `{m['conversion_executed']}`",
        f"- evaluation_executed: `{m['evaluation_executed']}`",
        f"- metric_seconds_claim_allowed_now: `{m['metric_seconds_claim_allowed_now']}`",
        "",
        "Current manifest blocks all conversion until user confirmation fields are filled and validated.",
    ]


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    return [
        "# User Action Required: Stage42-HM Restricted Metric/Time Terms Intake v2",
        "",
        "To unlock a future restricted metric/time conversion, manually fill:",
        "",
        f"- `{TEMPLATE_JSON}`",
        "",
        "Required fields per source:",
        "",
        *[f"- `{field}`" for field in REQUIRED_CONFIRMATION_FIELDS],
        "",
        "Important:",
        "",
        "- The agent cannot accept terms for you.",
        "- The current template is not permission, not conversion, not evaluation, and not metric/seconds evidence.",
        "- After filling, run `.venv-pytorch/bin/python run_stage42_restricted_metric_time_terms_intake_v2.py --validate-only`.",
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hm_gate"]
    s = payload["summary"]
    return [
        "## Stage42-HM Restricted Metric/Time Terms Intake v2",
        "",
        "- source: `fresh_stage42_hm_restricted_metric_time_terms_intake_v2`",
        f"- verdict: `{gate['verdict']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- source-level candidates / ready now: `{s['source_level_candidates']}` / `{s['conversion_ready_candidates_now']}`.",
        f"- after-terms domains: `{s['after_terms_domains_with_source_cv_candidate_count']}`.",
        f"- after-terms t50/t100 windows: `{s['after_terms_total_t50_windows']}` / `{s['after_terms_total_t100_windows']}`.",
        f"- template: `{s['template_path']}`.",
        "- conclusion: UCY/ETH_UCY restricted metric/time source-level candidates are now represented in a user-fillable intake v2, but all conversion/evaluation remains blocked until user-confirmed terms/source identity/path and a guarded rerun.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, CURRENT_SUMMARY, A_JOURNAL_GAP]:
        _replace_section(path, "STAGE42_HM_RESTRICTED_METRIC_TIME_TERMS_INTAKE_V2", lines)


def _refresh_research_state(payload: Mapping[str, Any], verification: Mapping[str, Any] | None = None) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-HM restricted metric/time terms intake v2"
    state["current_verdict"] = payload["stage42_hm_gate"]["verdict"]
    state["stage42_hm_restricted_metric_time_terms_intake_v2"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "template": str(TEMPLATE_JSON),
        "manifest": str(MANIFEST_JSON),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_hm_gate"]["verdict"],
        "gates": f"{payload['stage42_hm_gate']['passed']}/{payload['stage42_hm_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "verification": dict(verification or {"status": "pending"}),
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_restricted_metric_time_terms_intake_v2(
    *,
    refresh_readmes: bool = True,
    force_rebuild_template: bool = False,
    verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _build_payload(force_rebuild_template=force_rebuild_template)
    write_json(REPORT_JSON, payload)
    write_json(MANIFEST_JSON, payload["manifest"])
    write_md(REPORT_MD, _render_report(payload))
    write_md(MANIFEST_JSON.with_suffix(".md"), _render_manifest(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload, verification=verification)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rebuild-template", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    run_stage42_restricted_metric_time_terms_intake_v2(
        force_rebuild_template=args.force_rebuild_template and not args.validate_only
    )


if __name__ == "__main__":
    main()
