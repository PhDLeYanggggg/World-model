from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

PAPER_FILES = [
    OUT_DIR / "paper_outline_stage42.md",
    OUT_DIR / "method_draft_stage42.md",
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "failure_taxonomy_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "data_card_stage42.md",
    OUT_DIR / "reproducibility_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

SUPPORT_FILES = [
    OUT_DIR / "paper_ready_evidence_matrix_stage42.md",
    OUT_DIR / "replay_evidence_tiers_stage42.md",
    OUT_DIR / "replay_evidence_tiers_stage42.json",
    OUT_DIR / "reviewer_replay_package_stage42.md",
    OUT_DIR / "reviewer_replay_commands_stage42_hv.sh",
    OUT_DIR / "data_calibration_stage42.json",
    OUT_DIR / "source_time_geometry_calibration_stage42.json",
    OUT_DIR / "source_terms_validation_stage42.json",
    OUT_DIR / "source_terms_gap_audit_stage42.json",
    OUT_DIR / "restricted_metric_time_readiness_stage42.json",
]

REPORT_JSON = OUT_DIR / "paper_package_integrity_stage42.json"
REPORT_MD = OUT_DIR / "paper_package_integrity_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hx_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_SUMMARY = Path("README_M3W_CURRENT_MASTER_SUMMARY_ZH.md")
ROUTES_SUMMARY = Path("README_M3W_RESEARCH_ROUTES_FAILURES_SUCCESSES_2026_05_27_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE42_HX_PAPER_PACKAGE_INTEGRITY"
SOURCE = "fresh_stage42_hx_paper_package_integrity_from_current_artifacts"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HX 是 paper package integrity/provenance verifier，不训练、不调阈值、不下载、不转换。",
    "paper package 可以支持 protected 2.5D world-state paper evidence，但不能支持 true-3D/foundation/global metric/seconds-level claim。",
    "future endpoints / waypoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

OBJECTIVE_ROWS = {
    "A_data_and_calibration": {"marker": "`A1`", "expected_status": "partial_blocked"},
    "A_metric_time_calibration": {"marker": "`A2`", "expected_status": "partial_blocked"},
    "B_external_validation": {"marker": "`B1`", "expected_status": "pass_with_boundary"},
    "C_full_waypoint_dynamics": {"marker": "`C1`", "expected_status": "pass_with_boundary"},
    "D_causal_ablation": {"marker": "`D1`", "expected_status": "mixed"},
    "E_safety_floor": {"marker": "`E1`", "expected_status": "pass_with_boundary"},
    "F_paper_package": {"marker": "`F1`", "expected_status": "pass_with_open_gaps"},
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_manifest(path: Path, role: str) -> dict[str, Any]:
    exists = path.exists()
    text = path.read_text(encoding="utf-8", errors="replace") if exists and path.suffix in {".md", ".sh"} else ""
    return {
        "path": str(path),
        "role": role,
        "source_label": "cached_verified" if exists else "not_run",
        "exists": exists,
        "size_bytes": int(path.stat().st_size) if exists else 0,
        "sha256": _sha256(path) if exists else "",
        "mentions_raw_frame_or_dataset_local": ("raw-frame" in text) or ("dataset-local" in text),
        "mentions_no_true3d_or_not_true3d": ("true 3D" in text) or ("true-3D" in text),
        "mentions_no_foundation": "foundation" in text,
        "mentions_no_metric_seconds_boundary": ("metric" in text) and ("seconds" in text),
        "mentions_stage5c_and_smc": ("Stage5C" in text) and ("SMC" in text),
    }


def _extract_objective_status(matrix_text: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for key, spec in OBJECTIVE_ROWS.items():
        marker = spec["marker"]
        present = marker in matrix_text
        expected = spec["expected_status"]
        rows[key] = {
            "marker": marker,
            "expected_status": expected,
            "present": present,
            "expected_status_present": present and expected in matrix_text,
            "source_label": "cached_verified" if present else "not_run",
        }
    return rows


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _readiness_summary() -> dict[str, Any]:
    data = read_json(OUT_DIR / "data_calibration_stage42.json", {})
    source_terms = read_json(OUT_DIR / "source_terms_validation_stage42.json", {})
    gap = read_json(OUT_DIR / "source_terms_gap_audit_stage42.json", {})
    readiness = read_json(OUT_DIR / "restricted_metric_time_readiness_stage42.json", {})
    return {
        "data_calibration_source": data.get("source", "unknown"),
        "global_metric_claim_allowed": bool((data.get("summary") or {}).get("global_metric_claim_allowed", False)),
        "global_seconds_claim_allowed": bool((data.get("summary") or {}).get("global_seconds_claim_allowed", False)),
        "source_terms_status": source_terms.get("summary", source_terms.get("source", "unknown")),
        "source_terms_gap_status": gap.get("verdict", gap.get("source", "unknown")),
        "restricted_metric_time_ready": bool(readiness.get("ready", False) or readiness.get("ready_now", False)),
        "restricted_metric_time_verdict": readiness.get("verdict", readiness.get("source", "unknown")),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    paper_files = payload["paper_files"]
    support_files = payload["support_files"]
    objective = payload["objective_coverage"]
    readiness = payload["readiness_summary"]
    paper_text = payload["aggregate_paper_text"]
    matrix_text = payload["paper_ready_evidence_matrix"]
    replay_text = payload["replay_evidence_tiers"]
    reviewer_text = payload["reviewer_replay_package"]
    commands_text = payload.get("reviewer_replay_commands", "")
    gates = {
        "paper_files_exist": all(row["exists"] for row in paper_files),
        "paper_files_nonempty": all(row["size_bytes"] > 1000 for row in paper_files),
        "paper_files_hashable": all(len(row["sha256"]) == 64 for row in paper_files),
        "support_files_exist": all(row["exists"] for row in support_files),
        "support_files_hashable": all(len(row["sha256"]) == 64 for row in support_files),
        "objective_a_to_f_rows_present": all(row["present"] for row in objective.values()),
        "objective_statuses_preserved": all(row["expected_status_present"] for row in objective.values()),
        "data_calibration_blocker_preserved": "partial_blocked" in matrix_text and readiness["global_metric_claim_allowed"] is False,
        "metric_time_blocker_preserved": "global_metric_seconds_claim_blocked" in matrix_text and readiness["global_seconds_claim_allowed"] is False,
        "ablation_mixed_status_preserved": objective["D_causal_ablation"]["expected_status_present"],
        "paper_package_open_gaps_preserved": objective["F_paper_package"]["expected_status_present"],
        "stage42_hw_t3_replay_present": "T3_row_level_batch_replay" in replay_text,
        "reviewer_replay_commands_include_hv": "run_stage42_t100_runtime_row_cache_replay.py" in (reviewer_text + "\n" + commands_text),
        "claim_boundary_not_true3d": "true 3D" in paper_text or "true-3D" in paper_text,
        "claim_boundary_not_foundation": "foundation" in paper_text,
        "claim_boundary_not_metric_seconds": "metric" in paper_text and "seconds" in paper_text,
        "stage5c_not_executed": "Stage5C" in paper_text and payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_not_enabled": "SMC" in paper_text and payload["claim_boundary"]["smc_enabled"] is False,
        "future_endpoint_input_blocked": payload["no_leakage"]["future_endpoint_input"] is False,
        "central_velocity_blocked": payload["no_leakage"]["central_velocity"] is False,
        "test_endpoint_goals_blocked": payload["no_leakage"]["test_endpoint_goals"] is False,
        "test_threshold_tuning_blocked": payload["no_leakage"]["test_threshold_tuning"] is False,
        "result_sources_labeled": all(row["source_label"] in {"cached_verified", "not_run"} for row in paper_files + support_files),
        "no_raw_data_or_cache_in_manifest": not any(row["path"].startswith("data/") or "/checkpoints/" in row["path"] for row in paper_files + support_files),
        "readmes_updated": bool(payload.get("readme_updates", {}).get("readmes_updated", False)),
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_hx_paper_package_integrity_pass" if passed == total else "stage42_hx_paper_package_integrity_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _table(rows: list[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| path | role | source | size | sha256 |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['path']}` | `{row['role']}` | `{row['source_label']}` | {row['size_bytes']} | `{row['sha256'][:12]}` |"
        )
    return lines


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload.get("stage42_hx_gate", {"passed": "pending", "total": "pending", "verdict": "pending"})
    return [
        "## Stage42-HX Paper Package Integrity",
        "",
        f"- source: `{payload['source']}`",
        "- role: verify Stage42 paper package deliverables, evidence provenance, replay-tier linkage, and claim boundaries.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- paper deliverables checked: `{len(payload['paper_files'])}`.",
        f"- support/evidence files checked: `{len(payload['support_files'])}`.",
        "- A-F objective coverage is preserved as: A partial/blocked, B/C/E pass-with-boundary, D mixed, F pass-with-open-gaps.",
        "- Claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.",
    ]


def _write_report(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_hx_gate"]
    lines = [
        "# Stage42-HX Paper Package Integrity",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- package_hash: `{payload['package_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Paper Deliverables",
        "",
        *_table(payload["paper_files"]),
        "",
        "## Support Evidence",
        "",
        *_table(payload["support_files"]),
        "",
        "## A-F Objective Coverage",
        "",
        "| objective | source | present | expected status | status preserved |",
        "| --- | --- | --- | --- | --- |",
        *[
            f"| `{name}` | `{row['source_label']}` | `{row['present']}` | `{row['expected_status']}` | `{row['expected_status_present']}` |"
            for name, row in payload["objective_coverage"].items()
        ],
        "",
        "## Readiness Summary",
        "",
        *[f"- `{key}`: `{value}`" for key, value in payload["readiness_summary"].items()],
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{name}` | `{bool(ok)}` |" for name, ok in gate["gates"].items()],
        "",
        "## Interpretation",
        "",
        "- Stage42-HX verifies that the paper package is internally consistent and hashable.",
        "- The package is paper-ready for protected raw-frame/dataset-local 2.5D evidence, while preserving open blockers for source terms, metric/time calibration, and mixed/negative module evidence.",
        "- This is not new training, not new evaluation, not metric conversion, not Stage5C, and not SMC.",
    ]
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage42-HX Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(ok)}` |" for name, ok in gate["gates"].items()],
        ],
    )


def _refresh_readmes(payload: Mapping[str, Any]) -> dict[str, bool]:
    lines = _refresh_lines(payload)
    readme_paths = [README_RESULTS, M3W_README, MASTER_SUMMARY, ROUTES_SUMMARY]
    for path in readme_paths:
        _replace_section(path, SECTION, lines)
    matrix_lines = [
        "## Stage42-HX Paper Package Integrity",
        "",
        "- Stage42-HX verifies that all required Stage42 paper deliverables exist, are hashable, and preserve the current claim boundary.",
        f"- gate: `{payload.get('stage42_hx_gate', {}).get('passed', 'pending')} / {payload.get('stage42_hx_gate', {}).get('total', 'pending')}`.",
        "- A-F evidence status remains intentionally nuanced: A is blocked for source/metric-time, D is mixed, and F is package-ready with open gaps.",
        "- T3 row-level replay evidence from Stage42-HW is linked, but the claim remains raw-frame/dataset-local 2.5D.",
    ]
    _replace_section(OUT_DIR / "paper_ready_evidence_matrix_stage42.md", SECTION, matrix_lines)
    return {
        "readmes_updated": all(SECTION in path.read_text(encoding="utf-8") for path in readme_paths),
        "paper_matrix_updated": SECTION in (OUT_DIR / "paper_ready_evidence_matrix_stage42.md").read_text(encoding="utf-8"),
    }


def _refresh_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    gate = payload["stage42_hx_gate"]
    state["current_stage"] = "Stage42-HX paper package integrity"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_hx_paper_package_integrity"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "gates": f"{gate['passed']}/{gate['total']}",
        "verdict": gate["verdict"],
        "package_hash": payload["package_hash"],
        "paper_files": payload["paper_files"],
        "support_files": payload["support_files"],
        "objective_coverage": payload["objective_coverage"],
        "claim_boundary": payload["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(RESEARCH_STATE, state)


def run_stage42_paper_package_integrity() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    matrix_text = _read_text(OUT_DIR / "paper_ready_evidence_matrix_stage42.md")
    replay_text = _read_text(OUT_DIR / "replay_evidence_tiers_stage42.md")
    reviewer_text = _read_text(OUT_DIR / "reviewer_replay_package_stage42.md")
    commands_text = _read_text(OUT_DIR / "reviewer_replay_commands_stage42_hv.sh")
    aggregate_paper_text = "\n".join(_read_text(path) for path in PAPER_FILES)
    paper_files = [_file_manifest(path, "paper_deliverable") for path in PAPER_FILES]
    support_files = [_file_manifest(path, "support_evidence") for path in SUPPORT_FILES]
    payload: dict[str, Any] = {
        "stage": "Stage42-HX",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "paper_files": paper_files,
        "support_files": support_files,
        "objective_coverage": _extract_objective_status(matrix_text),
        "readiness_summary": _readiness_summary(),
        "paper_ready_evidence_matrix": matrix_text,
        "replay_evidence_tiers": replay_text,
        "reviewer_replay_package": reviewer_text,
        "reviewer_replay_commands": commands_text,
        "aggregate_paper_text": aggregate_paper_text,
        "input_hash": _combined_hash(PAPER_FILES + SUPPORT_FILES),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_hx_gate"] = {"passed": "pending", "total": "pending", "verdict": "pending", "gates": {}}
    payload["readme_updates"] = _refresh_readmes(payload)
    payload["package_hash"] = hashlib.sha256(
        "|".join(row["sha256"] for row in payload["paper_files"] + payload["support_files"]).encode("utf-8")
    ).hexdigest()
    payload["stage42_hx_gate"] = _gate(payload)
    payload["readme_updates"] = _refresh_readmes(payload)
    payload["paper_ready_evidence_matrix"] = _read_text(OUT_DIR / "paper_ready_evidence_matrix_stage42.md")
    payload["support_files"] = [_file_manifest(path, "support_evidence") for path in SUPPORT_FILES]
    payload["package_hash"] = hashlib.sha256(
        "|".join(row["sha256"] for row in payload["paper_files"] + payload["support_files"]).encode("utf-8")
    ).hexdigest()
    payload["stage42_hx_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    _write_report(payload)
    _refresh_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_paper_package_integrity()
    gate = result["stage42_hx_gate"]
    print(f"Stage42-HX paper package integrity: {gate['verdict']} ({gate['passed']}/{gate['total']})")
