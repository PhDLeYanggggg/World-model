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
GAP_JSON = OUT_DIR / "source_terms_gap_audit_stage42.json"
VALIDATION_JSON = OUT_DIR / "source_terms_validation_stage42.json"

REPORT_JSON = OUT_DIR / "source_local_path_prefill_stage42.json"
REPORT_MD = OUT_DIR / "source_local_path_prefill_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_local_path_prefill_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hy_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_SUMMARY = Path("README_M3W_CURRENT_MASTER_SUMMARY_ZH.md")
ROUTES_SUMMARY = Path("README_M3W_RESEARCH_ROUTES_FAILURES_SUCCESSES_2026_05_27_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE42_HY_SOURCE_LOCAL_PATH_PREFILL"
SOURCE = "fresh_stage42_hy_source_local_path_prefill_from_local_files"

LOCAL_CANDIDATES = {
    "ucy_crowd_original": [
        Path("external_data/OpenTraj/datasets/UCY"),
        Path("/Users/yangyue/Downloads/UCY"),
        Path("/Users/yangyue/Downloads/World/external_data/UCY"),
    ],
    "eth_biwi_original": [
        Path("external_data/OpenTraj/datasets/ETH"),
        Path("/Users/yangyue/Downloads/ETH_UCY"),
        Path("/Users/yangyue/Downloads/World/external_data/ETH_UCY"),
    ],
    "trajnetplusplus_official": [
        Path("external_data/OpenTraj/datasets/TrajNet++"),
        Path("external_data/OpenTraj/datasets/TrajNet"),
        Path("/Users/yangyue/Downloads/trajnetplusplusdataset"),
    ],
    "opentraj_toolkit": [
        Path("external_data/OpenTraj"),
        Path("/Users/yangyue/Downloads/OpenTraj"),
    ],
    "aerialmpt_or_other_topdown": [
        Path("data/aerialmpt"),
        Path("/Users/yangyue/Downloads/AerialMPT"),
        Path("/Users/yangyue/Downloads/World/external_data/AerialMPT"),
    ],
}

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HY 只做本地 source path / parseability 预填，不下载、不转换、不训练、不评估。",
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


def _sha256_small(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
    return digest.hexdigest()


def _iter_files(path: Path, max_files: int = 5000) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    files: list[Path] = []
    for item in path.rglob("*"):
        if item.is_file():
            files.append(item)
            if len(files) >= max_files:
                break
    return files


def _count_suffixes(files: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in files:
        suffix = path.suffix.lower() or "<no_ext>"
        counts[suffix] = counts.get(suffix, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _find_first(files: list[Path], names: set[str] | None = None, suffixes: set[str] | None = None) -> str:
    for path in files:
        if names and path.name in names:
            return str(path)
        if suffixes and path.suffix.lower() in suffixes:
            return str(path)
    return ""


def _path_summary(path: Path) -> dict[str, Any]:
    files = _iter_files(path)
    sample_files = [str(p) for p in files[:12]]
    suffix_counts = _count_suffixes(files)
    total_size = sum(p.stat().st_size for p in files if p.exists())
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file() if path.exists() else False,
        "file_count": len(files),
        "total_size_bytes": int(total_size),
        "suffix_counts": suffix_counts,
        "sample_files": sample_files,
        "first_hash": _sha256_small(files[0]) if files else "",
        "has_homography_file": bool(_find_first(files, names={"H.txt", "H-cam.txt", "H-old.txt"})),
        "has_obsmat": bool(_find_first(files, names={"obsmat.txt", "obsmat_px.txt"})),
        "has_video": bool(_find_first(files, suffixes={".avi", ".mp4", ".mov"})),
        "has_reference_image": bool(_find_first(files, names={"reference.png", "bg.png", "map.png"})),
        "has_ndjson": bool(_find_first(files, suffixes={".ndjson"})),
        "has_zip": bool(_find_first(files, suffixes={".zip"})),
        "homography_example": _find_first(files, names={"H.txt", "H-cam.txt", "H-old.txt"}),
        "trajectory_example": _find_first(files, names={"obsmat.txt", "students001-trajnet.txt"}, suffixes={".ndjson", ".txt"}),
        "video_example": _find_first(files, suffixes={".avi", ".mp4", ".mov"}),
    }


def _best_candidate(paths: list[Path]) -> dict[str, Any]:
    summaries = [_path_summary(path) for path in paths]
    existing = [row for row in summaries if row["exists"]]
    if not existing:
        return {"best_path": "", "local_path_found": False, "summaries": summaries}
    ranked = sorted(
        existing,
        key=lambda row: (
            row["has_obsmat"] or row["has_ndjson"] or row["has_zip"],
            row["has_homography_file"],
            row["has_video"],
            row["file_count"],
        ),
        reverse=True,
    )
    return {"best_path": ranked[0]["path"], "local_path_found": True, "summaries": summaries}


def _source_identity_hint(dataset_id: str, best_path: str) -> str:
    if not best_path:
        return "not_prefilled"
    if "OpenTraj" in best_path:
        return f"local OpenTraj-hosted copy for {dataset_id}; user must confirm whether this path is acceptable for official-source conversion"
    if dataset_id == "aerialmpt_or_other_topdown" and best_path.endswith(".zip"):
        return "local AerialMPT-like zip candidate; user must confirm official source URL and license terms before extraction/conversion"
    return f"local path candidate for {dataset_id}; user must confirm official source identity and terms"


def _prefill_rows(gap: Mapping[str, Any], validation: Mapping[str, Any]) -> list[dict[str, Any]]:
    validation_by_id = {row.get("dataset_id"): row for row in validation.get("validations", [])}
    rows = []
    for gap_row in gap.get("gap_rows", []):
        dataset_id = gap_row["dataset_id"]
        candidate = _best_candidate(LOCAL_CANDIDATES.get(dataset_id, []))
        validation_row = validation_by_id.get(dataset_id, {})
        rows.append(
            {
                "dataset_id": dataset_id,
                "name": validation_row.get("name", dataset_id),
                "domain": gap_row.get("domain", validation_row.get("domain", "")),
                "official_url": gap_row.get("official_url", validation_row.get("official_url", "")),
                "source_label": "fresh_run_local_path_prefill",
                "best_local_path_candidate": candidate["best_path"],
                "local_path_found": candidate["local_path_found"],
                "source_identity_hint": _source_identity_hint(dataset_id, candidate["best_path"]),
                "terms_accepted_by_user": False,
                "conversion_ready_now": False,
                "conversion_allowed_now": False,
                "converted_now": False,
                "evaluated_now": False,
                "missing_confirmation_fields_after_prefill": [
                    field
                    for field in gap_row.get("missing_confirmation_fields", [])
                    if field not in {"local_path", "source_identity"} or not candidate["local_path_found"]
                ],
                "estimated_t50_windows_after_terms": gap_row.get("estimated_t50_windows_after_terms", 0),
                "estimated_t100_windows_after_terms": gap_row.get("estimated_t100_windows_after_terms", 0),
                "source_cv_after_terms": gap_row.get("source_cv_after_terms", False),
                "blocker_class": gap_row.get("blocker_class", ""),
                "path_summaries": candidate["summaries"],
                "next_user_action": "confirm terms, allowed use, acceptance date, local path, and source identity before any guarded conversion",
            }
        )
    return rows


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = payload["prefill_rows"]
    by_id = {row["dataset_id"]: row for row in rows}
    gates = {
        "gap_input_present": payload["inputs"]["gap_exists"],
        "validation_input_present": payload["inputs"]["validation_exists"],
        "all_gap_targets_audited": len(rows) >= 5,
        "ucy_path_prefilled": by_id.get("ucy_crowd_original", {}).get("local_path_found") is True,
        "eth_path_prefilled": by_id.get("eth_biwi_original", {}).get("local_path_found") is True,
        "trajnet_path_checked": "trajnetplusplus_official" in by_id,
        "aerialmpt_path_checked": "aerialmpt_or_other_topdown" in by_id,
        "parseability_hints_present": any(
            any(summary.get("has_obsmat") or summary.get("has_ndjson") or summary.get("has_zip") for summary in row["path_summaries"])
            for row in rows
        ),
        "technical_windows_preserved": sum(int(row.get("estimated_t50_windows_after_terms", 0)) for row in rows) > 0,
        "legal_block_preserved": all(row["terms_accepted_by_user"] is False and row["conversion_ready_now"] is False for row in rows),
        "no_download": payload["actions"]["downloaded"] is False,
        "no_conversion": payload["actions"]["converted"] is False,
        "no_training": payload["actions"]["trained"] is False,
        "no_evaluation": payload["actions"]["evaluated"] is False,
        "user_action_written": payload["user_action_required"]["exists"] is True,
        "no_metric_seconds_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_not_enabled": payload["claim_boundary"]["smc_enabled"] is False,
        "readmes_updated": bool(payload.get("readme_updates", {}).get("readmes_updated", False)),
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_hy_source_local_path_prefill_pass" if passed == total else "stage42_hy_source_local_path_prefill_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _rows_table(rows: list[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| dataset | domain | local path found | best path | t50 after terms | t100 after terms | remaining confirmation |",
        "| --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['dataset_id']}` | `{row.get('domain', '')}` | `{row['local_path_found']}` | `{row['best_local_path_candidate'] or 'not_found'}` | "
            f"{int(row.get('estimated_t50_windows_after_terms', 0))} | {int(row.get('estimated_t100_windows_after_terms', 0))} | "
            f"{', '.join(row['missing_confirmation_fields_after_prefill']) or 'terms/identity still user-confirmation-gated'} |"
        )
    return lines


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload.get("stage42_hy_gate", {"passed": "pending", "total": "pending", "verdict": "pending"})
    found = sum(1 for row in payload["prefill_rows"] if row["local_path_found"])
    return [
        "## Stage42-HY Source Local Path Prefill",
        "",
        f"- source: `{payload['source']}`",
        "- role: reduce source/legal blocker by pre-filling local path and parseability candidates without claiming legal conversion.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- targets audited: `{len(payload['prefill_rows'])}`; local path candidates found: `{found}`.",
        f"- estimated after-terms t50/t100 windows preserved: `{payload['summary']['estimated_t50_windows_after_terms']}` / `{payload['summary']['estimated_t100_windows_after_terms']}`.",
        "- Remaining blocker: user must confirm official terms, allowed use, acceptance date, local path, and source identity before guarded conversion.",
        "- Claim boundary: no download, no conversion, no training, no evaluation, no metric/seconds-level claim, no Stage5C, no SMC.",
    ]


def _write_reports(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_hy_gate"]
    lines = [
        "# Stage42-HY Source Local Path Prefill",
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
        "## Prefill Rows",
        "",
        *_rows_table(payload["prefill_rows"]),
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{name}` | `{bool(ok)}` |" for name, ok in gate["gates"].items()],
        "",
        "## Interpretation",
        "",
        "- HY reduces a concrete blocker by locating local candidate paths and parseability hints for UCY/ETH/TrajNet/OpenTraj/AerialMPT-like sources.",
        "- It does not accept terms, does not confirm official source identity, and does not run conversion/evaluation.",
        "- The next valid step is user confirmation followed by guarded conversion + no-leakage + source-CV evaluation.",
    ]
    write_md(REPORT_MD, lines)
    action_lines = [
        "# User Action Required: Stage42-HY Source Local Path Confirmation",
        "",
        "请只在你确认官方条款、允许用途、source identity、local path 都正确后，再允许后续 guarded conversion。",
        "",
        *_rows_table(payload["prefill_rows"]),
        "",
        "Required fields per source:",
        "",
        "- `terms_accepted_by_user`: true/false",
        "- `terms_acceptance_date`: YYYY-MM-DD",
        "- `allowed_use`: e.g. research_only / commercial_allowed / unknown",
        "- `local_path`: exact local path to use",
        "- `source_identity`: official source or mirror identity; OpenTraj mirror is not automatically official-source permission",
        "",
        "No conversion/evaluation/metric-time claim is allowed until these fields are confirmed and a guarded conversion passes.",
    ]
    write_md(USER_ACTION_MD, action_lines)
    write_md(
        GATE_MD,
        [
            "# Stage42-HY Gate",
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
        "## Stage42-HY Source Local Path Prefill",
        "",
        "- HY pre-fills local path candidates for source terms targets, reducing but not closing A-stage source/legal blockers.",
        f"- gate: `{payload.get('stage42_hy_gate', {}).get('passed', 'pending')} / {payload.get('stage42_hy_gate', {}).get('total', 'pending')}`.",
        f"- local path candidates found: `{sum(1 for row in payload['prefill_rows'] if row['local_path_found'])}` / `{len(payload['prefill_rows'])}`.",
        "- No terms were accepted, no conversion/evaluation happened, and metric/seconds claims remain blocked.",
    ]
    _replace_section(OUT_DIR / "paper_ready_evidence_matrix_stage42.md", SECTION, matrix_lines)
    return {
        "readmes_updated": all(SECTION in path.read_text(encoding="utf-8") for path in readme_paths),
        "paper_matrix_updated": SECTION in (OUT_DIR / "paper_ready_evidence_matrix_stage42.md").read_text(encoding="utf-8"),
    }


def _refresh_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    gate = payload["stage42_hy_gate"]
    state["current_stage"] = "Stage42-HY source local path prefill"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_hy_source_local_path_prefill"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "gates": f"{gate['passed']}/{gate['total']}",
        "verdict": gate["verdict"],
        "summary": payload["summary"],
        "prefill_rows": payload["prefill_rows"],
        "claim_boundary": payload["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD, USER_ACTION_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(RESEARCH_STATE, state)


def run_stage42_source_local_path_prefill() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    gap = read_json(GAP_JSON, {})
    validation = read_json(VALIDATION_JSON, {})
    rows = _prefill_rows(gap, validation)
    summary = {
        "targets": len(rows),
        "local_path_candidates_found": sum(1 for row in rows if row["local_path_found"]),
        "estimated_t50_windows_after_terms": sum(int(row.get("estimated_t50_windows_after_terms", 0)) for row in rows),
        "estimated_t100_windows_after_terms": sum(int(row.get("estimated_t100_windows_after_terms", 0)) for row in rows),
        "conversion_ready_now": 0,
        "converted_now": 0,
        "evaluated_now": 0,
    }
    payload: dict[str, Any] = {
        "stage": "Stage42-HY",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "inputs": {"gap_exists": GAP_JSON.exists(), "validation_exists": VALIDATION_JSON.exists()},
        "input_hash": _combined_hash([GAP_JSON, VALIDATION_JSON]),
        "prefill_rows": rows,
        "summary": summary,
        "actions": {"downloaded": False, "converted": False, "trained": False, "evaluated": False},
        "user_action_required": {"exists": False, "path": str(USER_ACTION_MD)},
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_hy_gate"] = {"passed": "pending", "total": "pending", "verdict": "pending", "gates": {}}
    _write_reports({**payload, "stage42_hy_gate": payload["stage42_hy_gate"]})
    payload["user_action_required"] = {"exists": USER_ACTION_MD.exists(), "path": str(USER_ACTION_MD)}
    payload["readme_updates"] = _refresh_readmes(payload)
    payload["stage42_hy_gate"] = _gate(payload)
    _write_reports(payload)
    payload["readme_updates"] = _refresh_readmes(payload)
    payload["stage42_hy_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    _refresh_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_source_local_path_prefill()
    gate = result["stage42_hy_gate"]
    print(f"Stage42-HY source local path prefill: {gate['verdict']} ({gate['passed']}/{gate['total']})")
