from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
JO_JSON = OUT_DIR / "local_calibrated_source_guarded_conversion_preflight_stage42.json"
REPORT_JSON = OUT_DIR / "local_calibrated_source_terms_prefill_stage42.json"
REPORT_MD = OUT_DIR / "local_calibrated_source_terms_prefill_stage42.md"
PREFILL_JSON = OUT_DIR / "local_calibrated_source_terms_prefill_template_stage42.json"
GATE_MD = OUT_DIR / "stage42_stage_jp_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_local_calibrated_terms_prefill_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE42_JP_LOCAL_CALIBRATED_SOURCE_TERMS_PREFILL"
SOURCE = "fresh_stage42_jp_local_calibrated_source_terms_prefill"

README_BY_DATASET = {
    "Town-Center": Path("external_data/OpenTraj/datasets/Town-Center/README.md"),
    "Wild-Track": Path("external_data/OpenTraj/datasets/Wild-Track/README.md"),
    "PETS-2009-S2L1": Path("external_data/OpenTraj/datasets/PETS-2009/README.md"),
}

OFFICIAL_HINTS = {
    "Town-Center": {
        "official_url_candidates": [
            "http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html"
        ],
        "official_source_status": "historical_official_url_reported_but_local_readme_has_no_license_and_current_distribution_not_verified",
        "terms_status_hint": "manual_terms_required_high_risk",
        "source_confidence": "low",
    },
    "Wild-Track": {
        "official_url_candidates": ["https://www.epfl.ch/labs/cvlab/data/data-wildtrack/"],
        "official_source_status": "official_epfl_cvlab_dataset_page_identified",
        "terms_status_hint": "manual_terms_or_download_page_review_required",
        "source_confidence": "high",
    },
    "PETS-2009-S2L1": {
        "official_url_candidates": [
            "http://www.cvg.reading.ac.uk/PETS2009/a.html",
            "https://centaur.reading.ac.uk/14669/",
        ],
        "official_source_status": "local_readme_points_to_university_of_reading_pets_page_and_reading_publication_record",
        "terms_status_hint": "manual_terms_review_required_before_conversion",
        "source_confidence": "medium",
    },
}

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JP turns JO's guarded preflight into a user-fillable source/terms prefill, not permission.",
    "No local calibrated candidate is converted, downloaded, trained, evaluated, or counted as benchmark evidence.",
    "Official/source hints must be checked by the user before acceptance fields can be filled.",
    "Dataset-local/raw-frame and source-specific calibration hints are not global metric/seconds claims.",
    "Stage5C and SMC remain disabled.",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _links_from_text(text: str) -> list[str]:
    if not text:
        return []
    links = re.findall(r"https?://[^\s)\]>]+", text)
    cleaned = []
    for link in links:
        link = link.rstrip(".,")
        if link not in cleaned:
            cleaned.append(link)
    return cleaned


def _license_excerpt(text: str) -> str:
    if not text:
        return "no_local_readme"
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if "license" in line.lower() or "legal note" in line.lower():
            return " ".join(lines[idx : idx + 4]).strip()[:600]
    return "no_license_section_found"


def _terms_row(candidate: Mapping[str, Any]) -> dict[str, Any]:
    name = str(candidate["dataset_name"])
    readme_path = README_BY_DATASET.get(name, Path(""))
    text = _read_text(readme_path)
    hints = OFFICIAL_HINTS.get(name, {})
    local_links = _links_from_text(text)
    official_candidates = list(hints.get("official_url_candidates", []))
    for link in local_links:
        if link not in official_candidates:
            official_candidates.append(link)
    preferred_official = official_candidates[0] if official_candidates else ""
    license_excerpt = _license_excerpt(text)
    license_found = license_excerpt not in {"no_local_readme", "no_license_section_found"} and "No information is available" not in license_excerpt
    return {
        "dataset_name": name,
        "result_source": "fresh_prefill_from_stage42_jo_and_local_readme_plus_manual_official_hint",
        "retrieval_date": datetime.now(timezone.utc).date().isoformat(),
        "local_path": candidate.get("local_path", ""),
        "readme_path": str(readme_path) if readme_path else "",
        "readme_exists": readme_path.exists() if readme_path else False,
        "official_url_candidates": official_candidates,
        "preferred_official_url": preferred_official,
        "local_readme_links": local_links,
        "official_source_status": hints.get("official_source_status", "official_source_not_resolved"),
        "source_confidence": hints.get("source_confidence", "unknown"),
        "license_excerpt": license_excerpt,
        "license_found_in_local_readme": license_found,
        "terms_status_hint": hints.get("terms_status_hint", "manual_terms_required"),
        "technical_ready_after_terms": candidate.get("technical_ready_for_guarded_conversion_after_terms", False),
        "t50_rows": candidate.get("t50_rows", 0),
        "t100_rows": candidate.get("t100_rows", 0),
        "agent_tracks": candidate.get("agent_tracks", 0),
        "metric_status": candidate.get("metric_status", ""),
        "coordinate_unit": candidate.get("coordinate_unit", ""),
        "must_be_filled_by_user": {
            "official_url_confirmed": False,
            "official_terms_url_confirmed": False,
            "terms_accepted_by_user": False,
            "accepted_by_user": "",
            "accepted_at_utc": "",
            "allowed_use": "",
            "source_identity_confirmed": False,
            "conversion_scope_confirmed": False,
        },
        "conversion_ready_now": False,
        "converted_now": False,
        "evaluated_now": False,
        "next_action": "User verifies the official source and terms, then copies confirmed values into local_calibrated_source_terms_template_stage42.json before rerunning JO/guarded conversion.",
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    jo_payload = read_json(JO_JSON, {})
    candidates = list(jo_payload.get("candidate_preflights", []))
    prefill_rows = [_terms_row(row) for row in candidates]
    payload: dict[str, Any] = {
        "stage": "Stage42-JP",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_report": str(JO_JSON),
        "input_hash": _combined_hash([str(JO_JSON), *[str(path) for path in README_BY_DATASET.values()]]),
        "current_facts": CURRENT_FACTS,
        "prefill_rows": prefill_rows,
        "prefill_template": {
            "source": SOURCE,
            "purpose": "User-fillable official source/terms prefill for local calibrated candidate sources. This is not permission.",
            "terms_confirmation_is_currently_absent": True,
            "datasets": prefill_rows,
        },
        "summary": {
            "datasets_prefilled": len(prefill_rows),
            "official_hint_rows": sum(1 for row in prefill_rows if row["official_url_candidates"]),
            "license_found_rows": sum(1 for row in prefill_rows if row["license_found_in_local_readme"]),
            "conversion_ready_now": 0,
            "converted_now": 0,
            "evaluated_now": 0,
            "high_confidence_official_source_rows": [row["dataset_name"] for row in prefill_rows if row["source_confidence"] == "high"],
            "manual_only_rows": [row["dataset_name"] for row in prefill_rows if row["terms_status_hint"].startswith("manual")],
            "decision": "terms_prefill_written_no_conversion_permission",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "download_executed": False,
            "conversion_executed": False,
            "evaluation_executed": False,
            "prefill_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "global_metric_or_seconds_claim": False,
            "converted_external_support_source": False,
            "prefill_is_permission": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_jp_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = payload["prefill_rows"]
    summary = payload["summary"]
    gates = {
        "stage42_jo_input_loaded": Path(payload["input_report"]).exists() and len(rows) >= 3,
        "prefill_rows_written": summary["datasets_prefilled"] >= 3,
        "official_hints_present": summary["official_hint_rows"] >= 2,
        "manual_terms_preserved": all(row["must_be_filled_by_user"]["terms_accepted_by_user"] is False for row in rows),
        "conversion_ready_zero": summary["conversion_ready_now"] == 0,
        "converted_zero": summary["converted_now"] == 0,
        "evaluated_zero": summary["evaluated_now"] == 0,
        "town_center_not_overclaimed": any(
            row["dataset_name"] == "Town-Center" and row["source_confidence"] == "low" and row["conversion_ready_now"] is False
            for row in rows
        ),
        "wildtrack_official_hint_present": any(
            row["dataset_name"] == "Wild-Track" and "epfl.ch" in " ".join(row["official_url_candidates"])
            for row in rows
        ),
        "pets_official_hint_present": any(
            row["dataset_name"] == "PETS-2009-S2L1" and "reading.ac.uk" in " ".join(row["official_url_candidates"])
            for row in rows
        ),
        "no_leakage_pass": all(
            payload["no_leakage"][key] is False
            for key in [
                "future_endpoint_input",
                "central_velocity",
                "test_endpoint_goals",
                "test_threshold_tuning",
                "download_executed",
                "conversion_executed",
                "evaluation_executed",
            ]
        )
        and payload["no_leakage"]["prefill_only"],
        "no_metric_seconds_overclaim": payload["claim_boundary"]["global_metric_or_seconds_claim"] is False,
        "prefill_not_permission": payload["claim_boundary"]["prefill_is_permission"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    verdict = "stage42_jp_local_calibrated_source_terms_prefill_pass" if passed == len(gates) else "stage42_jp_local_calibrated_source_terms_prefill_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": len(gates), "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jp_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage42-JP Local Calibrated Source Terms Prefill",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_report: `{payload['input_report']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- decision: `{summary['decision']}`",
        f"- datasets_prefilled: `{summary['datasets_prefilled']}`",
        f"- official_hint_rows: `{summary['official_hint_rows']}`",
        f"- license_found_rows: `{summary['license_found_rows']}`",
        f"- high_confidence_official_source_rows: `{summary['high_confidence_official_source_rows']}`",
        f"- manual_only_rows: `{summary['manual_only_rows']}`",
        f"- conversion_ready_now: `{summary['conversion_ready_now']}`",
        "",
        "## Prefill Rows",
        "",
        "| dataset | source confidence | preferred official/source URL | license local? | conversion ready now | status hint |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in payload["prefill_rows"]:
        lines.append(
            f"| `{row['dataset_name']}` | `{row['source_confidence']}` | `{row['preferred_official_url']}` | "
            f"`{row['license_found_in_local_readme']}` | `{row['conversion_ready_now']}` | `{row['terms_status_hint']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This stage makes the user action more concrete; it does not grant terms acceptance.",
            "- Wild-Track and PETS have stronger official/source hints than Town-Center, but all still require user confirmation before conversion.",
            "- Town-Center remains high-risk/manual-only because the local README says license information is unavailable and the historical official distribution is not verified here.",
            "- Metric/time claims remain disabled even where calibration files exist.",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{payload['no_leakage']}`",
            f"- claim_boundary: `{payload['claim_boundary']}`",
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Local Calibrated Source Terms Prefill",
        "",
        "Fill or correct the template below only after checking official dataset terms. This prefill is not permission and is not conversion readiness.",
        "",
        f"- template_json: `{PREFILL_JSON}`",
        "",
    ]
    for row in payload["prefill_rows"]:
        lines.extend(
            [
                f"## {row['dataset_name']}",
                "",
                f"- local_path: `{row['local_path']}`",
                f"- preferred_official_url: `{row['preferred_official_url']}`",
                f"- official_url_candidates: `{row['official_url_candidates']}`",
                f"- source_confidence: `{row['source_confidence']}`",
                f"- license_excerpt: `{row['license_excerpt']}`",
                f"- terms_status_hint: `{row['terms_status_hint']}`",
                "- user must confirm: official_url, official_terms_url, license_name, terms_accepted_by_user, accepted_by_user, accepted_at_utc, allowed_use, source_identity_confirmed, conversion_scope_confirmed.",
                "",
            ]
        )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jp_gate"]
    lines = [
        "# Stage42-JP Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    return lines


def _section_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jp_gate"]
    summary = payload["summary"]
    return [
        "## Stage42-JP Local Calibrated Source Terms Prefill",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`",
        f"- official_hint_rows: `{summary['official_hint_rows']}`; license_found_rows: `{summary['license_found_rows']}`; conversion_ready_now: `{summary['conversion_ready_now']}`.",
        f"- high_confidence_official_source_rows: `{summary['high_confidence_official_source_rows']}`; manual_only_rows: `{summary['manual_only_rows']}`.",
        "- boundary: terms prefill only; no permission, no conversion, no evaluation, no metric/seconds overclaim, no Stage5C, no SMC.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, _section_lines(payload))


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    stage42 = state.setdefault("stage42", {})
    stage42["local_calibrated_source_terms_prefill"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_jp_gate"]["verdict"],
        "gate": {"passed": payload["stage42_jp_gate"]["passed"], "total": payload["stage42_jp_gate"]["total"]},
        "decision": payload["summary"]["decision"],
        "datasets_prefilled": payload["summary"]["datasets_prefilled"],
        "official_hint_rows": payload["summary"]["official_hint_rows"],
        "conversion_ready_now": 0,
        "converted": False,
        "global_metric_or_seconds_claim": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    state["last_successful_command"] = "python run_stage42_local_calibrated_source_terms_prefill.py"
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(LEDGER.parent)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": "Stage42-JP",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_jp_gate"]["verdict"],
                    "result": payload["summary"]["decision"],
                    "fresh_run": True,
                    "downloaded": False,
                    "converted": False,
                    "trained": False,
                    "evaluated": False,
                    "stage5c_executed": False,
                    "smc_enabled": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_stage42_local_calibrated_source_terms_prefill(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    write_json(PREFILL_JSON, _jsonable(payload["prefill_template"]))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    if refresh_readmes:
        _update_readmes(payload)
        _update_state(payload)
        _append_ledger(payload)
    return payload


def main() -> None:
    payload = run_stage42_local_calibrated_source_terms_prefill(refresh_readmes=True)
    gate = payload["stage42_jp_gate"]
    print(f"Stage42-JP local calibrated source terms prefill: {gate['verdict']} ({gate['passed']}/{gate['total']})")


if __name__ == "__main__":
    main()
