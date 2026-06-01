from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_blocked_source_terms_identity_packet.json"
REPORT_MD = OUT_DIR / "stage43_blocked_source_terms_identity_packet.md"
TEMPLATE_JSON = OUT_DIR / "stage43_blocked_source_terms_identity_template.json"
USER_ACTION_MD = OUT_DIR / "user_action_required_stage43_blocked_source_terms_identity.md"
GATE_MD = OUT_DIR / "stage43_stage_bf_blocked_source_terms_identity_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SOURCE = "fresh_stage43_bf_blocked_source_terms_identity_packet"
SECTION = "STAGE43_BF_BLOCKED_SOURCE_TERMS_IDENTITY_PACKET"

STAGE43_BE = OUT_DIR / "stage43_blocked_source_support_acquisition_preflight.json"

README_CANDIDATES = {
    "Town-Center": [
        Path("external_data/OpenTraj/datasets/Town-Center/README.md"),
    ],
    "Wild-Track": [
        Path("external_data/OpenTraj/datasets/Wild-Track/README.md"),
    ],
    "PETS-2009-S2L1": [
        Path("external_data/OpenTraj/datasets/PETS-2009/README.md"),
        Path("external_data/OpenTraj/datasets/PETS-2009/data/README.md"),
    ],
}

OFFICIAL_HINTS = {
    "Town-Center": {
        "official_url_candidates": [
            "http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html"
        ],
        "source_confidence": "low",
        "source_identity_status": "historical_project_page_hint_only_local_distribution_not_verified",
        "terms_status": "manual_terms_required_high_risk",
    },
    "Wild-Track": {
        "official_url_candidates": ["https://www.epfl.ch/labs/cvlab/data/data-wildtrack/"],
        "source_confidence": "high",
        "source_identity_status": "official_epfl_cvlab_dataset_page_hint_present",
        "terms_status": "manual_terms_or_download_page_review_required",
    },
    "PETS-2009-S2L1": {
        "official_url_candidates": [
            "http://www.cvg.reading.ac.uk/PETS2009/a.html",
            "https://centaur.reading.ac.uk/14669/",
        ],
        "source_confidence": "medium",
        "source_identity_status": "university_of_reading_pets_page_and_publication_record_hint",
        "terms_status": "manual_terms_review_required_before_conversion",
    },
}


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _links_from_text(text: str) -> list[str]:
    if not text:
        return []
    links = re.findall(r"https?://[^\s)\]>]+", text)
    out: list[str] = []
    for link in links:
        cleaned = link.rstrip(".,")
        if cleaned not in out:
            out.append(cleaned)
    return out


def _license_excerpt(text: str) -> str:
    if not text:
        return "no_local_readme"
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if "license" in lowered or "legal note" in lowered or "terms" in lowered:
            return " ".join(lines[idx : idx + 4]).strip()[:700]
    return "no_license_or_terms_section_found"


def _terms_packet_row(candidate: Mapping[str, Any]) -> dict[str, Any]:
    name = str(candidate.get("dataset_name", "unknown"))
    readme_path = _first_existing(README_CANDIDATES.get(name, []))
    readme_text = _read_text(readme_path) if readme_path is not None else ""
    readme_links = _links_from_text(readme_text)
    hints = OFFICIAL_HINTS.get(name, {})
    official_url_candidates = list(hints.get("official_url_candidates", []))
    for link in readme_links:
        if link not in official_url_candidates:
            official_url_candidates.append(link)
    blockers = [
        "terms_not_confirmed_by_user",
        "source_identity_not_confirmed_by_user",
        "conversion_scope_not_confirmed_by_user",
        "not_converted_into_stage43_feature_store",
    ]
    if not candidate.get("technical_support_candidate", False):
        blockers.append("technical_support_not_established")
    if not official_url_candidates:
        blockers.append("official_url_hint_missing")
    if int(candidate.get("calibration_file_count", 0)) <= 0:
        blockers.append("calibration_file_missing")
    return {
        "dataset_name": name,
        "source": "fresh_terms_identity_prefill_from_stage43_be",
        "local_path": str(candidate.get("local_path", "")),
        "readme_path": str(readme_path) if readme_path is not None else "",
        "readme_exists": readme_path is not None,
        "readme_links": readme_links,
        "license_or_terms_excerpt": _license_excerpt(readme_text),
        "official_url_candidates": official_url_candidates,
        "preferred_official_url": official_url_candidates[0] if official_url_candidates else "",
        "source_confidence": str(hints.get("source_confidence", "unknown")),
        "source_identity_status": str(hints.get("source_identity_status", "manual_review_required")),
        "terms_status": str(hints.get("terms_status", "manual_terms_required")),
        "technical_support_candidate": bool(candidate.get("technical_support_candidate", False)),
        "support_family": str(candidate.get("support_family", "unknown")),
        "point_rows": int(candidate.get("point_rows", 0)),
        "agent_tracks": int(candidate.get("agent_tracks", 0)),
        "t50_candidate_rows": int(candidate.get("t50_candidate_rows", 0)),
        "t100_candidate_rows": int(candidate.get("t100_candidate_rows", 0)),
        "calibration_file_count": int(candidate.get("calibration_file_count", 0)),
        "coordinate_unit": str(candidate.get("coordinate_unit", "unknown")),
        "metric_status": str(candidate.get("metric_status", "unverified")),
        "manual_fields_required": {
            "official_url_confirmed": False,
            "official_terms_url": "",
            "license_name": "",
            "terms_accepted_by_user": False,
            "accepted_by_user": "",
            "accepted_at_utc": "",
            "allowed_use": "",
            "source_identity_confirmed": False,
            "calibration_projection_scope_confirmed": False,
            "conversion_scope_confirmed": False,
            "can_use_for_stage43_support": False,
        },
        "conversion_ready_now": False,
        "guarded_conversion_allowed_now": False,
        "training_allowed_now": False,
        "blockers": blockers,
        "next_action": "fill_manual_terms_identity_fields_then_run_guarded_conversion_preflight",
    }


def _biwi_packet(be_payload: Mapping[str, Any]) -> dict[str, Any]:
    biwi = be_payload.get("family_readiness", {}).get("TrajNet_biwi", {})
    return {
        "family": "TrajNet_biwi",
        "source": "fresh_terms_identity_prefill_from_stage43_be",
        "status": biwi.get("status", "blocked"),
        "technical_candidate_count": int(biwi.get("technical_candidate_count", 0)),
        "conversion_ready_count": int(biwi.get("conversion_ready_count", 0)),
        "repair_training_allowed_now": False,
        "blockers": [
            "independent_biwi_like_source_missing",
            "current_useful_biwi_support_entangled_with_heldout_test_source",
            "source_level_train_val_test_story_not_closed",
        ],
        "manual_fields_required": {
            "new_independent_source_path": "",
            "official_url_confirmed": False,
            "terms_accepted_by_user": False,
            "source_identity_confirmed": False,
            "heldout_source_disjoint_from_train_val": False,
        },
        "next_action": "locate_or_acquire_independent_biwi_like_source_before_any_repair_training",
    }


def build_blocked_source_terms_identity_packet() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    be_payload = read_json(STAGE43_BE, {})
    candidates = [
        _terms_packet_row(row)
        for row in be_payload.get("local_source_candidates", [])
        if row.get("technical_support_candidate", False)
    ]
    biwi_packet = _biwi_packet(be_payload)
    template = {
        "source": SOURCE,
        "purpose": "Manual source/terms/identity confirmation template for Stage43 blocked source support. This file is not permission and does not convert data.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": candidates,
        "biwi_independent_source": biwi_packet,
    }
    summary = {
        "dataset_terms_packets": int(len(candidates)),
        "technical_candidates": int(sum(bool(row["technical_support_candidate"]) for row in candidates)),
        "official_hint_rows": int(sum(bool(row["official_url_candidates"]) for row in candidates)),
        "manual_terms_required_rows": int(sum(row["manual_fields_required"]["terms_accepted_by_user"] is False for row in candidates)),
        "conversion_ready_now": 0,
        "guarded_conversion_allowed_now": 0,
        "training_allowed_now": 0,
        "biwi_independent_source_ready": False,
        "decision": "terms_identity_packet_written_conversion_still_blocked",
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_terms_identity_packet_from_stage43_be_local_candidates",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {"stage43_be": str(STAGE43_BE)},
        "input_verdicts": {"stage43_be": be_payload.get("stage43_be_gate", {}).get("verdict")},
        "summary": summary,
        "dataset_packets": candidates,
        "biwi_independent_source_packet": biwi_packet,
        "template": template,
        "next_required_actions": [
            "User or data owner confirms official source URL and terms for each technical candidate.",
            "Fill the generated template only after terms/source identity are confirmed.",
            "Run guarded conversion preflight after template confirmation; do not skip no-leakage and source-level split checks.",
            "Keep blocked source families floor-only until conversion, split, baseline, and replay gates pass.",
        ],
        "no_leakage_and_execution": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "template_is_permission": False,
            "converted_external_support_source": False,
            "blocked_source_repair_success_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
        "input_hash": _combined_hash([STAGE43_BE]),
    }
    payload["stage43_bf_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    rows = payload["dataset_packets"]
    no_leak = payload["no_leakage_and_execution"]
    claim = payload["claim_boundary"]
    gates = {
        "stage43_be_precondition_passed": payload["input_verdicts"]["stage43_be"]
        == "stage43_be_blocked_source_support_acquisition_preflight_pass",
        "terms_identity_packets_written": summary["dataset_terms_packets"] >= 3,
        "technical_candidates_preserved": summary["technical_candidates"] >= 3,
        "official_hints_recorded": summary["official_hint_rows"] >= 2,
        "manual_terms_required_preserved": summary["manual_terms_required_rows"] == summary["dataset_terms_packets"],
        "conversion_still_blocked": summary["conversion_ready_now"] == 0
        and summary["guarded_conversion_allowed_now"] == 0,
        "training_still_blocked": summary["training_allowed_now"] == 0,
        "biwi_independent_source_not_ready": summary["biwi_independent_source_ready"] is False
        and payload["biwi_independent_source_packet"]["repair_training_allowed_now"] is False,
        "all_rows_have_blockers": all(row["blockers"] for row in rows),
        "next_actions_recorded": len(payload["next_required_actions"]) >= 4,
        "no_future_or_test_leakage": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["future_labels_eval_or_loss_only"] is True
        and no_leak["central_velocity_input"] is False
        and no_leak["test_endpoint_goal_construction"] is False
        and no_leak["test_statistics_normalization"] is False
        and no_leak["test_threshold_tuning"] is False,
        "no_execution": no_leak["download_executed"] is False
        and no_leak["conversion_executed"] is False
        and no_leak["training_executed"] is False
        and no_leak["evaluation_executed"] is False,
        "claim_boundary_not_overstated": claim["true_3d_world_model"] is False
        and claim["foundation_world_model"] is False
        and claim["metric_or_seconds_claim"] is False
        and claim["template_is_permission"] is False
        and claim["converted_external_support_source"] is False
        and claim["blocked_source_repair_success_claim"] is False,
        "stage5c_and_smc_false": claim["stage5c_executed"] is False and claim["smc_enabled"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_bf_blocked_source_terms_identity_packet_pass"
        if passed == total
        else "stage43_bf_blocked_source_terms_identity_packet_incomplete",
        "stage5c_executed": False,
        "smc_enabled": False,
        "goal_complete": False,
    }


def _render_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_bf_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage43-BF Blocked Source Terms / Identity Packet",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- dataset packets: `{summary['dataset_terms_packets']}`",
        f"- conversion-ready now: `{summary['conversion_ready_now']}`",
        f"- training allowed now: `{summary['training_allowed_now']}`",
        "",
        "## Dataset Packets",
        "",
        "| dataset | source confidence | official hints | readme | t50 | t100 | conversion ready | blockers |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for row in payload["dataset_packets"]:
        lines.append(
            f"| `{row['dataset_name']}` | `{row['source_confidence']}` | {len(row['official_url_candidates'])} | "
            f"`{row['readme_path'] or 'missing'}` | {row['t50_candidate_rows']} | {row['t100_candidate_rows']} | "
            f"`{row['conversion_ready_now']}` | {', '.join(row['blockers'])} |"
        )
    biwi = payload["biwi_independent_source_packet"]
    lines.extend(
        [
            "",
            "## Biwi Independent Source Packet",
            "",
            f"- status: `{biwi['status']}`",
            f"- technical candidates already seen: `{biwi['technical_candidate_count']}`",
            f"- repair training allowed now: `{biwi['repair_training_allowed_now']}`",
            f"- blockers: `{biwi['blockers']}`",
            "",
            "## Interpretation",
            "",
            "This packet is deliberately boring in the right way: it turns local technical candidates into a user-fillable source/terms checklist, while keeping conversion and training blocked. PETS, Town-Center, and Wild-Track may help the MOT-like blocked family later, but only after source identity, terms, calibration scope, and guarded conversion pass.",
            "",
            "## Next Required Actions",
            "",
            *[f"- {item}" for item in payload["next_required_actions"]],
            "",
            "## Claim Boundary",
            "",
            "- The generated template is not permission.",
            "- No data conversion, training, threshold search, or evaluation is executed here.",
            "- Dataset-local/raw-frame 2.5D only.",
            "- No metric or seconds-level claim.",
            "- No Stage5C execution and no SMC.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
        ]
    )
    lines.extend([f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()])
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage43 Blocked Source Terms / Identity",
        "",
        "These rows are local technical candidates only. Fill the template after confirming official source identity and terms; do not treat this file as permission.",
        "",
    ]
    for row in payload["dataset_packets"]:
        lines.extend(
            [
                f"## {row['dataset_name']}",
                "",
                f"- local_path: `{row['local_path']}`",
                f"- preferred_official_url_hint: `{row['preferred_official_url']}`",
                f"- source_confidence: `{row['source_confidence']}`",
                f"- terms_status: `{row['terms_status']}`",
                f"- blockers: `{row['blockers']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## TrajNet_biwi",
            "",
            "- action: locate or acquire an independent biwi-like source before repair training.",
            "- reason: current useful support is entangled with the held-out source.",
            "",
        ]
    )
    return lines


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bf_gate"]
    summary = payload["summary"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"dataset_packets = `{summary['dataset_terms_packets']}`",
        f"conversion_ready_now = `{summary['conversion_ready_now']}`",
        f"training_allowed_now = `{summary['training_allowed_now']}`",
        "",
        "I turned the BE local support candidates into a concrete source/terms/identity packet. This is not permission and not a conversion: PETS, Town-Center, and Wild-Track still need official source and terms confirmation before guarded conversion; biwi still needs an independent held-out source. Blocked source families remain floor-only.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bf_blocked_source_terms_identity_packet"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "summary": summary,
        "report": str(REPORT_MD),
        "template": str(TEMPLATE_JSON),
        "user_action": str(USER_ACTION_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bf_blocked_source_terms_identity_packet"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BF",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "summary": summary,
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    write_json(TEMPLATE_JSON, m._jsonable(payload["template"]))
    write_json(WORLD_GATE_JSON, m._jsonable(payload["stage43_bf_gate"]))
    lines = _render_md(payload)
    write_md(REPORT_MD, lines)
    write_md(GATE_MD, lines)
    write_md(USER_ACTION_MD, _render_user_action(payload))
    gate = payload["stage43_bf_gate"]
    write_md(
        WORLD_GATE_MD,
        [
            "# Stage43 Current World-Model Gate",
            "",
            f"- source: `{payload['source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- long objective complete: `{gate['goal_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-P / AZ remains the performance leader and exact replay artifact.",
            "- Stage43-BF writes a source/terms/identity packet, not permission and not conversion.",
            "- Blocked biwi/mot source families stay floor-only until terms/source/split gates clear.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def run_blocked_source_terms_identity_packet() -> dict[str, Any]:
    payload = build_blocked_source_terms_identity_packet()
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Build Stage43 blocked source terms/source-identity packet without conversion or training."
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    build_arg_parser().parse_args(argv)
    payload = run_blocked_source_terms_identity_packet()
    gate = payload["stage43_bf_gate"]
    summary = payload["summary"]
    print(f"Stage43-BF: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"dataset_packets={summary['dataset_terms_packets']}")
    print(f"conversion_ready_now={summary['conversion_ready_now']}")
    print(f"training_allowed_now={summary['training_allowed_now']}")
    return payload


if __name__ == "__main__":
    main()
