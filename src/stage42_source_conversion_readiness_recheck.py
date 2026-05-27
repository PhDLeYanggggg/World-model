from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
TERMS_JSON = OUT_DIR / "source_terms_validation_stage42.json"
LEGAL_TIME_JSON = OUT_DIR / "source_legal_time_action_package_stage42.json"
TIME_JSON = OUT_DIR / "source_time_geometry_calibration_stage42.json"

REPORT_JSON = OUT_DIR / "source_conversion_readiness_recheck_stage42.json"
REPORT_MD = OUT_DIR / "source_conversion_readiness_recheck_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_conversion_recheck_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ds_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "global_t100_deployable_claim_allowed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DS 只做本地 source conversion readiness recheck，不下载、不转换、不训练、不评估。",
    "local path found 不等于 legal conversion ready。",
    "derived cache found 不等于 raw official dataset verified。",
    "terms/source identity/allowed use 未确认时，conversion_ready 必须保持 false。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon；dataset-local/raw-frame 不能写成 global metric/seconds。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


TARGETS: list[dict[str, Any]] = [
    {
        "dataset_id": "ucy_crowd_original",
        "domain": "UCY",
        "official_url": "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
        "role": "external_topdown_pedestrian_source_candidate",
        "raw_candidates": [
            "/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY",
            "/Users/yangyue/Downloads/OpenTraj/datasets/UCY",
            "/Users/yangyue/Downloads/ETH_UCY/UCY",
        ],
        "derived_candidates": [
            "data/stage20_world_state/ucy_crowd",
            "data/stage32_domain_alignment/UCY",
            "outputs/stage37_t50_history",
        ],
    },
    {
        "dataset_id": "eth_biwi_original",
        "domain": "ETH_UCY",
        "official_url": "https://vision.ee.ethz.ch/datsets.html",
        "role": "external_topdown_pedestrian_source_candidate",
        "raw_candidates": [
            "/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/ETH",
            "/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/ETH-Person",
            "/Users/yangyue/Downloads/OpenTraj/datasets/ETH",
            "/Users/yangyue/Downloads/ETH_UCY/ETH",
        ],
        "derived_candidates": [
            "data/stage20_world_state/eth_ucy_full",
            "data/stage5b_world_state/eth_ucy",
            "outputs/stage41_domain_local",
        ],
    },
    {
        "dataset_id": "trajnetplusplus_official",
        "domain": "TrajNet",
        "official_url": "https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/",
        "role": "external_topdown_pedestrian_source_candidate",
        "raw_candidates": [
            "/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet++",
            "/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet",
            "/Users/yangyue/Downloads/World/data/stage5b_raw/trajnetplusplusdataset",
            "/Users/yangyue/Downloads/trajnetplusplusdataset",
        ],
        "derived_candidates": [
            "data/stage20_world_state/trajnet_full",
            "data/stage5b_world_state/trajnet",
            "outputs/stage41_domain_local",
        ],
    },
    {
        "dataset_id": "opentraj_toolkit",
        "domain": "OpenTraj",
        "official_url": "https://github.com/crowdbotp/OpenTraj",
        "role": "toolkit_or_mirror_requires_source_identity_confirmation",
        "raw_candidates": [
            "/Users/yangyue/Downloads/World/external_data/OpenTraj",
            "/Users/yangyue/Downloads/OpenTraj",
        ],
        "derived_candidates": [
            "data/stage20_world_state/opentraj",
            "outputs/stage31_m3w_external",
            "outputs/stage32_domain_alignment",
        ],
    },
    {
        "dataset_id": "aerialmpt_or_other_topdown",
        "domain": "other_topdown",
        "official_url": "user_or_web_verified_official_url_required",
        "role": "drone_or_topdown_candidate_diagnostic_until_official_source_confirmed",
        "raw_candidates": [
            "/Users/yangyue/Downloads/World/data/aerialmpt/DLR_AerialMPT_Dataset.zip",
            "/Users/yangyue/Downloads/World/data/aerialmpt",
        ],
        "derived_candidates": [
            "data/stage14_multimodal_episodes/aerialmpt",
            "data/stage12_annotations/aerialmpt",
        ],
    },
    {
        "dataset_id": "stanford_drone_dataset",
        "domain": "SDD",
        "official_url": "https://cvgl.stanford.edu/projects/uav_data/",
        "role": "already_converted_sdd_pixel_raw_frame_reference_not_external_repair",
        "raw_candidates": [
            "/Users/yangyue/Downloads/World/external_data/StanfordDroneDataset",
        ],
        "derived_candidates": [
            "data/stage21_sdd_world_state",
            "data/stage24_sdd_fast_cache",
            "data/stage24_sdd_medium_index",
        ],
    },
    {
        "dataset_id": "tgsim_diagnostic",
        "domain": "traffic_diagnostic",
        "official_url": "https://github.com/NextGen-Cities-Institute/TGSIM",
        "role": "traffic_metric_time_diagnostic_only_not_pedestrian_official",
        "raw_candidates": [
            "/Users/yangyue/Downloads/World/external_data/TGSIM",
            "/Users/yangyue/Downloads/TGSIM",
        ],
        "derived_candidates": [
            "data/stage20_world_state/tgsim",
            "outputs/reports/tgsim_units.md",
        ],
    },
]


def _terms_by_dataset(terms_payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("dataset_id")): row for row in terms_payload.get("validations", [])}


def _sample_path(path: Path, limit: int = 30, cap: int = 250) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "is_dir": False,
            "is_file": False,
            "file_count_capped": 0,
            "cap_reached": False,
            "suffix_counts": {},
            "sample_files": [],
        }
    suffix_counts: dict[str, int] = {}
    samples: list[str] = []
    file_count = 0
    cap_reached = False
    if path.is_file():
        file_count = 1
        suffix_counts[path.suffix.lower() or "<none>"] = 1
        samples = [str(path)]
    else:
        for child in path.rglob("*"):
            if not child.is_file():
                continue
            file_count += 1
            suffix_counts[child.suffix.lower() or "<none>"] = suffix_counts.get(child.suffix.lower() or "<none>", 0) + 1
            if len(samples) < limit:
                samples.append(str(child))
            if file_count >= cap:
                cap_reached = True
                break
    return {
        "path": str(path),
        "exists": True,
        "is_dir": path.is_dir(),
        "is_file": path.is_file(),
        "file_count_capped": file_count,
        "cap_reached": cap_reached,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "sample_files": samples,
    }


def _parseability_hint(path_summaries: list[Mapping[str, Any]]) -> str:
    suffixes: set[str] = set()
    for summary in path_summaries:
        suffixes.update(str(k) for k in summary.get("suffix_counts", {}).keys())
    if not suffixes:
        return "missing_or_empty"
    if suffixes & {".txt", ".csv", ".ndjson", ".json", ".xml"}:
        return "trajectory_like_files_present"
    if suffixes & {".zip", ".tgz", ".gz", ".tar"}:
        return "archive_present_requires_terms_and_extraction_policy"
    if suffixes & {".py", ".md"}:
        return "toolkit_or_docs_present_not_raw_trajectory_proof"
    return "files_present_manual_schema_check_required"


def _scan_target(target: Mapping[str, Any], terms_row: Mapping[str, Any] | None) -> dict[str, Any]:
    raw_summaries = [_sample_path(Path(p)) for p in target["raw_candidates"]]
    derived_summaries = [_sample_path(Path(p)) for p in target["derived_candidates"]]
    raw_path_found = any(row["exists"] and row["file_count_capped"] > 0 for row in raw_summaries)
    derived_cache_found = any(row["exists"] and row["file_count_capped"] > 0 for row in derived_summaries)
    terms_confirmed = bool(terms_row and terms_row.get("terms_accepted_by_user") and terms_row.get("conversion_ready"))
    local_path_confirmed = bool(terms_row and str(terms_row.get("confirmed_local_path", "")).strip())
    source_identity_confirmed = bool(terms_row and str(terms_row.get("source_identity", "")).strip())
    official_url = str((terms_row or {}).get("official_url") or target.get("official_url"))
    official_source_confirmed = official_url.startswith("http")
    conversion_ready = bool(
        raw_path_found
        and terms_confirmed
        and local_path_confirmed
        and source_identity_confirmed
        and official_source_confirmed
    )
    blockers: list[str] = []
    if not raw_path_found:
        blockers.append("raw_local_path_missing_or_empty")
    if derived_cache_found:
        blockers.append("derived_cache_found_but_not_counted_as_raw_verified_dataset")
    if not terms_confirmed:
        blockers.append("terms_allowed_use_or_acceptance_not_confirmed")
    if not local_path_confirmed:
        blockers.append("user_confirmed_local_path_missing")
    if not source_identity_confirmed:
        blockers.append("source_identity_missing")
    if not official_source_confirmed:
        blockers.append("official_source_url_missing_or_user_verification_required")
    if target["dataset_id"] == "stanford_drone_dataset":
        blockers.append("already_sdd_reference_not_new_external_source")
    if target["domain"] == "traffic_diagnostic":
        blockers.append("traffic_diagnostic_only_not_pedestrian_official")
    return {
        "dataset_id": target["dataset_id"],
        "domain": target["domain"],
        "role": target["role"],
        "official_url": official_url,
        "official_source_confirmed": official_source_confirmed,
        "terms_confirmed": terms_confirmed,
        "local_path_confirmed_by_user": local_path_confirmed,
        "source_identity_confirmed": source_identity_confirmed,
        "raw_path_found": raw_path_found,
        "derived_cache_found": derived_cache_found,
        "technical_preflight_possible": raw_path_found or derived_cache_found,
        "conversion_ready": conversion_ready,
        "parseability_hint": _parseability_hint(raw_summaries + derived_summaries),
        "raw_path_summaries": raw_summaries,
        "derived_path_summaries": derived_summaries,
        "blockers": blockers,
        "next_action": _next_action(target, raw_path_found, derived_cache_found, official_source_confirmed),
    }


def _next_action(
    target: Mapping[str, Any],
    raw_path_found: bool,
    derived_cache_found: bool,
    official_source_confirmed: bool,
) -> str:
    if target["dataset_id"] == "stanford_drone_dataset":
        return "keep_as_sdd_pixel_raw_frame_reference; do not count as new external source"
    if target["domain"] == "traffic_diagnostic":
        return "keep_as_diagnostic_only; do not use as pedestrian topdown official benchmark"
    if not official_source_confirmed:
        return "provide/verify official dataset URL before any conversion claim"
    if raw_path_found:
        return "user must confirm terms acceptance, allowed use, acceptance date, local path, and source identity before conversion"
    if derived_cache_found:
        return "locate raw official source path; derived cache alone is insufficient"
    return "provide legally obtained local raw path or official download/application confirmation"


def _time_candidates(time_payload: Mapping[str, Any]) -> dict[str, list[str]]:
    by_domain: dict[str, list[str]] = {}
    for row in time_payload.get("source_records", []):
        if row.get("source_specific_metric_time_evidence") or row.get("timing", {}).get("annotation_fps"):
            by_domain.setdefault(str(row.get("domain", "unknown")), []).append(str(row.get("source_id")))
    return by_domain


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "targets_checked": s["targets_checked"] >= 7,
        "raw_path_scan_completed": all("raw_path_summaries" in row for row in payload["target_rows"]),
        "derived_cache_not_counted_ready": all(
            not (row["derived_cache_found"] and not row["terms_confirmed"] and row["conversion_ready"])
            for row in payload["target_rows"]
        ),
        "legal_blockers_preserved": s["conversion_ready_targets"] == 0,
        "technical_preflight_separated": s["technical_preflight_possible_targets"] >= 1,
        "user_action_required_present": s["user_action_required_targets"] >= 5,
        "sdd_not_new_external": next(
            row for row in payload["target_rows"] if row["dataset_id"] == "stanford_drone_dataset"
        )["conversion_ready"]
        is False,
        "traffic_diagnostic_not_official": next(
            row for row in payload["target_rows"] if row["dataset_id"] == "tgsim_diagnostic"
        )["conversion_ready"]
        is False,
        "no_conversion_claim": s["converted_datasets_now"] == 0,
        "no_evaluation_claim": s["evaluated_datasets_now"] == 0,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_ds_source_conversion_readiness_recheck_pass" if passed == total else "stage42_ds_source_conversion_readiness_recheck_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-DS Source Conversion Readiness Recheck",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{payload['stage42_ds_gate']['passed']} / {payload['stage42_ds_gate']['total']}`",
        f"- verdict: `{payload['stage42_ds_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in payload["summary"].items()],
        "",
        "## Target Rows",
        "",
        "| dataset | domain | raw path | derived cache | preflight | conversion ready | parseability | next action |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["target_rows"]:
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | {} |".format(
                row["dataset_id"],
                row["domain"],
                row["raw_path_found"],
                row["derived_cache_found"],
                row["technical_preflight_possible"],
                row["conversion_ready"],
                row["parseability_hint"],
                row["next_action"],
            )
        )
    lines.extend(
        [
            "",
            "## Important Boundary",
            "",
            "- Raw-looking local paths were found for several sources, especially OpenTraj/UCY/ETH/TrajNet and SDD.",
            "- These paths are not treated as conversion-ready because user-confirmed terms, allowed use, acceptance date, local path, and source identity are still missing in the Stage42 terms validator.",
            "- Derived caches and previous converted outputs are useful technical hints but are not raw official dataset evidence.",
            "- SDD remains an already-converted pixel raw-frame reference, not a new external repair source.",
            "- TGSIM remains traffic diagnostic only, not a top-down pedestrian official benchmark.",
            "- Stage5C and SMC remain disabled.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{key}` | `{value}` |" for key, value in payload["stage42_ds_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-DS Source Conversion Readiness",
        "",
        "本步骤发现了若干本地路径和 derived cache，但没有把它们当作 conversion-ready。继续转换前需要用户确认 legal/source 信息。",
        "",
        "| dataset | local/path status | required action |",
        "| --- | --- | --- |",
    ]
    for row in payload["target_rows"]:
        if row["conversion_ready"]:
            continue
        status = []
        if row["raw_path_found"]:
            status.append("raw path found")
        if row["derived_cache_found"]:
            status.append("derived cache found")
        if not status:
            status.append("missing")
        lines.append(f"| `{row['dataset_id']}` | {', '.join(status)} | {row['next_action']} |")
    lines.extend(
        [
            "",
            "必须提供的信息：",
            "",
            "- official dataset/source URL",
            "- 是否已接受条款、接受日期、允许用途",
            "- 本地 raw path",
            "- source identity / dataset version",
            "- 是否允许 derived conversion / redistribution / publication use",
            "",
            "在这些信息缺失前，任何 local path 或 derived cache 都不能写成 legally converted dataset。",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ds_gate"]
    return [
        "# Stage42-DS Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{key}` | `{value}` |" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gate = payload["stage42_ds_gate"]
    return [
        "## Stage42-DS Source Conversion Readiness Recheck",
        "",
        "- source: `fresh_local_path_scan_after_stage42_do`",
        "- role: separates local raw-path/derived-cache hints from legal conversion readiness.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- targets checked: `{s['targets_checked']}`; raw-path found: `{s['raw_path_found_targets']}`; derived-cache found: `{s['derived_cache_found_targets']}`.",
        f"- technical preflight possible: `{s['technical_preflight_possible_targets']}`; conversion-ready targets: `{s['conversion_ready_targets']}`.",
        "- No dataset was converted or evaluated in this step; legal/source blockers remain preserved.",
        f"- report: `{REPORT_MD}`.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_DS_SOURCE_CONVERSION_READINESS_RECHECK", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-DS source conversion readiness recheck"
    state["current_verdict"] = payload["stage42_ds_gate"]["verdict"]
    state["stage42_ds_source_conversion_readiness_recheck"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "user_action_required": str(USER_ACTION_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_ds_gate"]["verdict"],
        "gates": f"{payload['stage42_ds_gate']['passed']}/{payload['stage42_ds_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_source_conversion_readiness_recheck() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    terms_payload = read_json(TERMS_JSON, {})
    legal_time_payload = read_json(LEGAL_TIME_JSON, {})
    time_payload = read_json(TIME_JSON, {})
    terms_map = _terms_by_dataset(terms_payload)
    rows = [_scan_target(target, terms_map.get(str(target["dataset_id"]))) for target in TARGETS]
    time_candidates = _time_candidates(time_payload)
    payload: dict[str, Any] = {
        "source": "fresh_local_path_scan_after_stage42_do",
        "stage": "Stage42-DS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "terms_json": str(TERMS_JSON),
            "legal_time_json": str(LEGAL_TIME_JSON),
            "time_geometry_json": str(TIME_JSON),
            "prior_legal_time_verdict": legal_time_payload.get("stage42_do_gate", {}).get("verdict"),
        },
        "target_rows": rows,
        "time_geometry_candidates_by_domain": time_candidates,
        "summary": {
            "targets_checked": len(rows),
            "raw_path_found_targets": sum(1 for row in rows if row["raw_path_found"]),
            "derived_cache_found_targets": sum(1 for row in rows if row["derived_cache_found"]),
            "technical_preflight_possible_targets": sum(1 for row in rows if row["technical_preflight_possible"]),
            "conversion_ready_targets": sum(1 for row in rows if row["conversion_ready"]),
            "conversion_ready_ids": [row["dataset_id"] for row in rows if row["conversion_ready"]],
            "user_action_required_targets": sum(1 for row in rows if not row["conversion_ready"]),
            "converted_datasets_now": 0,
            "evaluated_datasets_now": 0,
            "raw_path_found_ids": [row["dataset_id"] for row in rows if row["raw_path_found"]],
            "derived_cache_found_ids": [row["dataset_id"] for row in rows if row["derived_cache_found"]],
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_ds_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_source_conversion_readiness_recheck()

