from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src import stage42_data_calibration as calib
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
BB_JSON = OUT_DIR / "t100_data_gap_audit_stage42.json"
REPORT_JSON = OUT_DIR / "t100_source_acquisition_plan_stage42.json"
REPORT_MD = OUT_DIR / "t100_source_acquisition_plan_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_t100_sources_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bc_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BC 是 Stage42-BB 后的 t100 source acquisition planner，不训练模型、不执行 Stage5C、不启用 SMC。",
    "t100 positive gain 仍缺独立 train-only source-CV 支持。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "所有 pedestrian / top-down claims 继续保持 raw-frame / dataset-local，除非未来官方 FPS/stride/homography/scale 验证完成。",
]


OFFICIAL_SOURCE_CANDIDATES = [
    {
        "id": "trajnetpp_epfl_aicrowd",
        "dataset_name": "TrajNet++",
        "target_domains": ["TrajNet", "ETH_UCY", "UCY"],
        "official_url": "https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/",
        "secondary_url": "https://www.epfl.ch/labs/vita/datasets/",
        "official_source_found": True,
        "retrieval_date": "2026-05-26",
        "official_evidence_summary": "EPFL describes TrajNet++ as an open trajectory forecasting challenge with accompanying data and evaluation code, curated categories, and reproducible sampling; challenge/data access is linked through AIcrowd.",
        "license_access_summary": "Public benchmark page, but challenge/data access may require AIcrowd account or terms. Treat as manual terms until user confirms accepted access.",
        "requires_login_or_terms": True,
        "auto_download_allowed": False,
        "local_path_candidates": [
            "external_data/OpenTraj/datasets/TrajNet",
            "external_data/OpenTraj/datasets/TrajNet++",
            "/Users/yangyue/Downloads/trajnetplusplusdataset",
        ],
        "t100_repair_value": "high",
        "expected_t100_role": "adds/validates independent trajectory sources for TrajNet and possibly ETH/UCY-style splits",
        "metric_time_value": "low_without_source_specific_fps_stride",
    },
    {
        "id": "eth_ucy_original_sources",
        "dataset_name": "ETH/UCY original pedestrian sources",
        "target_domains": ["ETH_UCY", "UCY"],
        "official_url": "ETH/BIWI and UCY original dataset pages; source-specific terms must be manually verified",
        "secondary_url": "https://github.com/crowdbotp/OpenTraj",
        "official_source_found": False,
        "retrieval_date": "2026-05-26",
        "official_evidence_summary": "Current local evidence comes through OpenTraj/source files; Stage42 calibration found homography-like/FPS evidence, but official source terms and calibration direction still need manual verification.",
        "license_access_summary": "Do not treat OpenTraj or local mirrored files as license override. User/source-specific verification required.",
        "requires_login_or_terms": True,
        "auto_download_allowed": False,
        "local_path_candidates": [
            "external_data/OpenTraj/datasets/ETH",
            "external_data/OpenTraj/datasets/ETH-Person",
            "external_data/OpenTraj/datasets/UCY",
            "/Users/yangyue/Downloads/ETH_UCY",
        ],
        "t100_repair_value": "high",
        "expected_t100_role": "highest-priority source-level repair target for ETH_UCY and UCY t100 support",
        "metric_time_value": "medium_if_homography_fps_stride_verified",
    },
    {
        "id": "ucy_crowd_original",
        "dataset_name": "UCY Crowd",
        "target_domains": ["UCY", "ETH_UCY"],
        "official_url": "http://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
        "secondary_url": "https://github.com/crowdbotp/OpenTraj",
        "official_source_found": True,
        "retrieval_date": "2026-05-26",
        "official_evidence_summary": "UCY Crowd is a top-down/fixed-camera pedestrian crowd source used by ETH/UCY/TrajNet style benchmarks; current local source support is too small for stable t100 source-CV.",
        "license_access_summary": "Manual official terms and citation verification required before new downloads/redistribution.",
        "requires_login_or_terms": True,
        "auto_download_allowed": False,
        "local_path_candidates": [
            "external_data/OpenTraj/datasets/UCY",
            "/Users/yangyue/Downloads/UCY",
        ],
        "t100_repair_value": "high",
        "expected_t100_role": "adds the missing independent UCY t100-capable original-train source if legally available",
        "metric_time_value": "medium_if_homography_fps_stride_verified",
    },
    {
        "id": "opentraj_toolkit",
        "dataset_name": "OpenTraj toolkit",
        "target_domains": ["ETH_UCY", "TrajNet", "UCY"],
        "official_url": "https://github.com/crowdbotp/OpenTraj",
        "secondary_url": "",
        "official_source_found": True,
        "retrieval_date": "2026-05-26",
        "official_evidence_summary": "OpenTraj is a toolkit/source hub for multiple pedestrian trajectory datasets and is useful for reproducible loaders and source grouping.",
        "license_access_summary": "Toolkit availability does not override licenses/terms of underlying datasets.",
        "requires_login_or_terms": False,
        "auto_download_allowed": False,
        "local_path_candidates": [
            "external_data/OpenTraj",
            "/Users/yangyue/Downloads/World/external_data/OpenTraj",
            "/Users/yangyue/Downloads/OpenTraj",
        ],
        "t100_repair_value": "medium",
        "expected_t100_role": "source discovery/loader hub; may expose additional legal source files already local",
        "metric_time_value": "low_without_underlying_source_calibration",
    },
    {
        "id": "aerialmpt_dlr",
        "dataset_name": "AerialMPT",
        "target_domains": ["AerialMPT"],
        "official_url": "https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/aerialmpt-a-dataset-for-pedestrian-tracking-in-aerial-imagery",
        "secondary_url": "https://elib.dlr.de/136057/",
        "official_source_found": True,
        "retrieval_date": "2026-05-26",
        "official_evidence_summary": "DLR reports 14 aerial pedestrian tracking sequences, 307 frames, co-registered/georeferenced crops, 2 fps, 2,528 pedestrians, and 44,740 point annotations. Public download is about 75 MB.",
        "license_access_summary": "DLR page credits CC BY-NC-ND 3.0; do not redistribute raw data or derived data without checking terms. Commercial/derivative use is restricted.",
        "requires_login_or_terms": False,
        "auto_download_allowed": False,
        "local_path_candidates": [
            "data/aerialmpt/DLR_AerialMPT_Dataset.zip",
            "/Users/yangyue/Downloads/AerialMPT",
            "external_data/AerialMPT",
            "data/stage11_scene_packs/aerialmpt",
        ],
        "t100_repair_value": "low",
        "expected_t100_role": "useful aerial calibration/scene diagnostic; likely too short for robust t100 raw-frame support because only 307 frames across 14 sequences",
        "metric_time_value": "medium_for_aerial_diagnostic_if_terms_and_georeferencing_verified",
    },
    {
        "id": "sdd_stanford",
        "dataset_name": "Stanford Drone Dataset",
        "target_domains": ["SDD"],
        "official_url": "https://cvgl.stanford.edu/projects/uav_data/",
        "secondary_url": "",
        "official_source_found": True,
        "retrieval_date": "2026-05-26",
        "official_evidence_summary": "SDD is already available locally and supports the SDD pixel raw-frame benchmark, but it does not by itself repair external ETH/UCY/TrajNet/UCY t100 source-CV gaps.",
        "license_access_summary": "Use official terms; do not commit raw videos/images or derived large shards.",
        "requires_login_or_terms": True,
        "auto_download_allowed": False,
        "local_path_candidates": [
            "external_data/StanfordDroneDataset",
            "/Users/yangyue/Downloads/World/external_data/StanfordDroneDataset",
            "data/stage21_sdd_world_state",
        ],
        "t100_repair_value": "low_for_external_high_for_sdd",
        "expected_t100_role": "SDD-specific pixel raw-frame benchmark; not external source-CV repair unless a separate cross-domain protocol is designed",
        "metric_time_value": "low_until_homography_scale_verified",
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _candidate_local_status(candidate: Mapping[str, Any]) -> dict[str, Any]:
    stats = [calib._walk_stats(Path(path)) for path in candidate["local_path_candidates"]]
    found = [row for row in stats if row.get("exists")]
    return {
        "source": "fresh_local_path_scan",
        "local_path_found": bool(found),
        "found_paths": [row["path"] for row in found],
        "path_stats": stats,
        "has_video": any(row.get("has_video") for row in found),
        "has_image": any(row.get("has_image") for row in found),
        "homography_like_files_found": [p for row in found for p in row.get("homography_like_files", [])][:20],
        "scale_like_files_found": [p for row in found for p in row.get("scale_like_files", [])][:20],
    }


def _priority_score(candidate: Mapping[str, Any], local: Mapping[str, Any], bb_gap: Mapping[str, Any]) -> int:
    score = 0
    if candidate["official_source_found"]:
        score += 20
    if local["local_path_found"]:
        score += 15
    if candidate["t100_repair_value"] == "high":
        score += 30
    elif candidate["t100_repair_value"] == "medium":
        score += 15
    elif candidate["t100_repair_value"] == "low_for_external_high_for_sdd":
        score += 8
    else:
        score += 5
    if candidate["metric_time_value"].startswith("medium"):
        score += 10
    if candidate["requires_login_or_terms"]:
        score -= 8
    if "NC-ND" in candidate["license_access_summary"] or "restricted" in candidate["license_access_summary"]:
        score -= 8
    target_domains = set(candidate["target_domains"])
    unsupported = set(bb_gap.get("unsupported_t100_domains", []))
    if target_domains & unsupported:
        score += 20
    return int(max(0, min(100, score)))


def _download_policy(candidate: Mapping[str, Any]) -> dict[str, Any]:
    blocked_reasons: list[str] = []
    if candidate["requires_login_or_terms"]:
        blocked_reasons.append("requires user confirmation of official terms/login/challenge access")
    if "NC-ND" in candidate["license_access_summary"] or "restricted" in candidate["license_access_summary"]:
        blocked_reasons.append("license restricts derivative/commercial/redistribution use; keep manual review")
    if not candidate["official_source_found"]:
        blocked_reasons.append("official source not fully resolved in Stage42-BC")
    if candidate["auto_download_allowed"] is False:
        blocked_reasons.append("auto_download_allowed is false by policy")
    return {
        "auto_download_allowed": False,
        "download_status": "not_run",
        "blocked_reasons": blocked_reasons,
        "safe_next_step": "verify local path and official terms; do not auto-download raw/gated/restricted data",
    }


def run_stage42_t100_source_acquisition_plan() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bb = _load_json(BB_JSON)
    unsupported = bb["summary"]["unsupported_t100_domains"]
    candidates: list[dict[str, Any]] = []
    for raw in OFFICIAL_SOURCE_CANDIDATES:
        local = _candidate_local_status(raw)
        score = _priority_score(raw, local, bb["summary"])
        candidate = {
            "source": "fresh_synthesis_from_official_web_and_local_paths",
            **raw,
            "local_status": local,
            "priority_score": score,
            "priority_group": "A" if score >= 75 else "B" if score >= 55 else "C" if score >= 35 else "diagnostic",
            "download_policy": _download_policy(raw),
        }
        candidates.append(candidate)
    candidates.sort(key=lambda row: (-int(row["priority_score"]), row["id"]))
    high_priority = [row for row in candidates if row["priority_group"] in {"A", "B"}]
    payload = {
        "source": "fresh_synthesis_from_stage42_bb_plus_official_web_pages",
        "stage": "Stage42-BC T100 Source Acquisition Plan",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([str(BB_JSON), "outputs/stage42_long_research/t100_data_gap_audit_stage42.md"]),
        "web_sources_used": [
            {
                "name": "AerialMPT DLR official page",
                "url": OFFICIAL_SOURCE_CANDIDATES[4]["official_url"],
                "retrieval_date": "2026-05-26",
                "summary": OFFICIAL_SOURCE_CANDIDATES[4]["official_evidence_summary"],
            },
            {
                "name": "VITA EPFL datasets page",
                "url": OFFICIAL_SOURCE_CANDIDATES[0]["secondary_url"],
                "retrieval_date": "2026-05-26",
                "summary": "Lists TrajNet++ as an interaction-centric human trajectory forecasting benchmark.",
            },
            {
                "name": "TrajNet++ EPFL open-research page",
                "url": OFFICIAL_SOURCE_CANDIDATES[0]["official_url"],
                "retrieval_date": "2026-05-26",
                "summary": OFFICIAL_SOURCE_CANDIDATES[0]["official_evidence_summary"],
            },
        ],
        "bb_verdict": bb.get("stage42_bb_gate", {}).get("verdict"),
        "bb_unsupported_t100_domains": unsupported,
        "candidates": candidates,
        "summary": {
            "source": "fresh_synthesis_from_stage42_bb_plus_official_web_pages",
            "candidate_sources": len(candidates),
            "official_sources_found": sum(1 for row in candidates if row["official_source_found"]),
            "local_path_found_sources": sum(1 for row in candidates if row["local_status"]["local_path_found"]),
            "high_priority_sources": [row["id"] for row in high_priority],
            "auto_download_allowed_sources": [row["id"] for row in candidates if row["download_policy"]["auto_download_allowed"]],
            "auto_download_executed": False,
            "user_action_required_count": sum(1 for row in candidates if row["download_policy"]["blocked_reasons"]),
            "t100_repair_priority_order": [row["id"] for row in high_priority],
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "paper_claim": "T100 repair should prioritize legal independent TrajNet++ / ETH-UCY / UCY source support and source-specific safety repair. AerialMPT is useful diagnostic/calibration context but likely too short for robust t100.",
        },
        "user_action_required": _user_actions(candidates, bb["summary"]),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_metrics_for_threshold": False,
            "no_raw_download_executed": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "t100_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_bc_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_ledger(payload)
    return payload


def _user_actions(candidates: list[Mapping[str, Any]], bb_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for domain, needed in bb_summary.get("additional_t100_sources_needed_by_domain", {}).items():
        if int(needed) > 0:
            matching = [row["id"] for row in candidates if domain in row["target_domains"]]
            actions.append(
                {
                    "source": "fresh_synthesis_from_stage42_bb",
                    "priority": "high",
                    "target": domain,
                    "minimum_extra_t100_sources_needed": int(needed),
                    "candidate_source_ids": matching,
                    "action": "Provide or approve legal source-specific t100-capable train data, then rerun source-CV without using test metrics.",
                }
            )
    for row in candidates:
        if row["download_policy"]["blocked_reasons"]:
            actions.append(
                {
                    "source": row["source"],
                    "priority": "high" if row["priority_group"] == "A" else "medium",
                    "target": row["id"],
                    "dataset_name": row["dataset_name"],
                    "official_url": row["official_url"],
                    "local_paths_found": row["local_status"]["found_paths"],
                    "action": row["download_policy"]["safe_next_step"],
                    "blocked_reasons": row["download_policy"]["blocked_reasons"],
                }
            )
    return actions


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    gates = {
        "bb_input_verified": payload["bb_verdict"] == "stage42_bb_t100_data_gap_audit_pass_with_data_blocker",
        "official_source_candidates_listed": summary["candidate_sources"] >= 5 and summary["official_sources_found"] >= 4,
        "local_paths_checked": summary["local_path_found_sources"] >= 3,
        "t100_priority_sources_identified": len(summary["t100_repair_priority_order"]) >= 3,
        "user_actions_generated": summary["user_action_required_count"] > 0 and bool(payload["user_action_required"]),
        "no_auto_download_of_restricted_or_gated_data": not summary["auto_download_executed"] and not summary["auto_download_allowed_sources"],
        "web_sources_recorded": len(payload["web_sources_used"]) >= 3,
        "no_leakage_pass": all(
            payload["no_leakage"][k] is False
            for k in ["future_endpoint_input", "future_waypoint_input", "central_velocity", "test_endpoint_goals", "test_metrics_for_threshold"]
        )
        and payload["no_leakage"]["no_raw_download_executed"],
        "no_metric_seconds_overclaim": not payload["claim_boundary"]["metric_or_seconds_claim"] and not payload["claim_boundary"]["t100_seconds_claim"],
        "stage5c_false": not payload["claim_boundary"]["stage5c_executed"],
        "smc_false": not payload["claim_boundary"]["smc_enabled"],
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_bc_t100_source_acquisition_plan_pass" if passed == total else "stage42_bc_t100_source_acquisition_plan_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    lines = [
        "# Stage42-BC T100 Source Acquisition Plan",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bc_gate']['passed']} / {payload['stage42_bc_gate']['total']}`",
        f"- verdict: `{payload['stage42_bc_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Official Web Sources Used",
        "",
        "| source | url | retrieval_date | summary |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["web_sources_used"]:
        lines.append(f"| {row['name']} | `{row['url']}` | `{row['retrieval_date']}` | {row['summary']} |")
    lines.extend(
        [
            "",
            "## Candidate Source Ranking",
            "",
            "| id | dataset | priority | score | local path | t100 value | auto download | reason |",
            "| --- | --- | --- | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for row in payload["candidates"]:
        lines.append(
            f"| `{row['id']}` | {row['dataset_name']} | {row['priority_group']} | {row['priority_score']} | `{row['local_status']['local_path_found']}` | {row['t100_repair_value']} | `{row['download_policy']['auto_download_allowed']}` | {row['expected_t100_role']} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- bb_unsupported_t100_domains: `{payload['bb_unsupported_t100_domains']}`",
            f"- candidate_sources: `{summary['candidate_sources']}`",
            f"- official_sources_found: `{summary['official_sources_found']}`",
            f"- local_path_found_sources: `{summary['local_path_found_sources']}`",
            f"- high_priority_sources: `{summary['high_priority_sources']}`",
            f"- auto_download_allowed_sources: `{summary['auto_download_allowed_sources']}`",
            f"- auto_download_executed: `{summary['auto_download_executed']}`",
            f"- user_action_required_count: `{summary['user_action_required_count']}`",
            f"- global_metric_claim_allowed: `{summary['global_metric_claim_allowed']}`",
            f"- global_seconds_claim_allowed: `{summary['global_seconds_claim_allowed']}`",
            "",
            "## Interpretation",
            "",
            "- Stage42-BC does not download restricted/gated data and does not bypass license terms.",
            "- Highest-priority t100 repair path is legal independent TrajNet++ / ETH-UCY / UCY source support plus rerunning train-only source-CV.",
            "- AerialMPT is official and locally useful as aerial calibration/diagnostic context, but its 307 frames across 14 sequences make it unlikely to solve t100 support alone.",
            "- TGSIM/traffic-style metric diagnostics must not be converted into pedestrian/top-down world-model success claims.",
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-BC User Action Required For T100 Source Acquisition",
        "",
        f"- source: `{payload['source']}`",
        "- purpose: official-source and local-path actions needed to repair t100 source support without bypassing terms.",
        "",
    ]
    for action in payload["user_action_required"]:
        lines.extend(
            [
                f"## {action['target']}",
                "",
                f"- priority: `{action['priority']}`",
                f"- action: {action['action']}",
            ]
        )
        if "minimum_extra_t100_sources_needed" in action:
            lines.append(f"- minimum_extra_t100_sources_needed: `{action['minimum_extra_t100_sources_needed']}`")
        if "candidate_source_ids" in action:
            lines.append(f"- candidate_source_ids: `{action['candidate_source_ids']}`")
        if "dataset_name" in action:
            lines.append(f"- dataset_name: {action['dataset_name']}")
        if "official_url" in action:
            lines.append(f"- official_url: `{action['official_url']}`")
        if "local_paths_found" in action:
            lines.append(f"- local_paths_found: `{action['local_paths_found']}`")
        if "blocked_reasons" in action:
            lines.append("- blocked_reasons:")
            lines.extend([f"  - {reason}" for reason in action["blocked_reasons"]])
        lines.append("")
    lines.extend(
        [
            "## Non-Claims",
            "",
            "- Do not call AerialMPT/SDD/TGSIM evidence a repair for ETH_UCY/TrajNet/UCY t100 unless a separate source-level protocol proves it.",
            "- Do not claim metric or seconds-level pedestrian prediction from local homography/FPS hints alone.",
            "- Do not auto-download or redistribute gated/restricted/raw third-party data.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bc_gate"]
    lines = [
        "# Stage42-BC Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    return lines


def _append_ledger(payload: Mapping[str, Any]) -> None:
    row = {
        "stage": payload["stage"],
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_bc_gate"]["verdict"],
        "gate": f"{payload['stage42_bc_gate']['passed']}/{payload['stage42_bc_gate']['total']}",
        "git_commit": payload["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_t100_source_acquisition_plan()
