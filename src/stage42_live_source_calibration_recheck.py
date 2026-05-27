from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "live_source_calibration_recheck_stage42.json"
REPORT_MD = OUT_DIR / "live_source_calibration_recheck_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ga_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_live_source_calibration_stage42.md"

SOURCE_ACTION_JSON = OUT_DIR / "source_action_consolidator_stage42.json"
DATA_CALIBRATION_JSON = OUT_DIR / "data_calibration_stage42.json"
UNIFIED_QUEUE_JSON = OUT_DIR / "unified_guarded_conversion_queue_stage42.json"
SOURCE_TERMS_TEMPLATE_JSON = OUT_DIR / "source_terms_confirmation_template_stage42.json"
UCY_H100_TEMPLATE_JSON = OUT_DIR / "ucy_h100_candidate_terms_template_stage42.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
ONE_FILE_SUMMARY = Path("README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_live_source_calibration_recheck"

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "download_executed": False,
    "conversion_executed": False,
    "training_executed": False,
    "evaluation_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


TARGETS = [
    {
        "target_id": "sdd",
        "domain": "SDD",
        "official_url": "https://cvgl.stanford.edu/projects/uav_data/",
        "data_role": "official_eval / supervised_training",
        "paths": ["external_data/StanfordDroneDataset", "data/stage21_sdd_world_state", "data/stage24_sdd_fast_cache"],
        "known_status": "already_converted_cached_verified_pixel_raw_frame",
        "required_before_new_claim": ["homography/scale verification", "effective seconds audit"],
    },
    {
        "target_id": "opentraj_toolkit",
        "domain": "OpenTraj",
        "official_url": "https://github.com/crowdbotp/OpenTraj",
        "data_role": "toolkit/source index; not automatically official benchmark",
        "paths": ["external_data/OpenTraj", "/Users/yangyue/Downloads/OpenTraj"],
        "known_status": "local_toolkit_found_but_not_new_conversion_ready",
        "required_before_new_claim": ["source identity", "dataset-specific terms", "source-CV split"],
    },
    {
        "target_id": "eth_biwi_original",
        "domain": "ETH_UCY",
        "official_url": "https://vision.ee.ethz.ch/datsets.html",
        "data_role": "external top-down pedestrian candidate",
        "paths": ["external_data/OpenTraj/datasets/ETH", "external_data/OpenTraj/datasets/ETH-Person", "data/stage20_world_state/eth_ucy_full"],
        "known_status": "local_candidates_found_terms_unconfirmed",
        "required_before_new_claim": ["terms confirmation", "source identity", "guarded conversion", "no-leakage/source-CV"],
    },
    {
        "target_id": "trajnetplusplus_official",
        "domain": "TrajNet",
        "official_url": "https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/",
        "data_role": "external top-down pedestrian candidate",
        "paths": ["external_data/OpenTraj/datasets/TrajNet", "external_data/OpenTraj/datasets/TrajNet++", "data/stage5b_raw/trajnetplusplusdataset"],
        "known_status": "local_candidates_found_but_h100_long_source_blocked",
        "required_before_new_claim": ["official longer h100-capable source", "terms confirmation", "guarded conversion", "no-leakage/source-CV"],
    },
    {
        "target_id": "ucy_crowd_original",
        "domain": "UCY",
        "official_url": "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
        "data_role": "external top-down pedestrian candidate",
        "paths": ["external_data/OpenTraj/datasets/UCY", "data/stage20_world_state/ucy_crowd"],
        "known_status": "local_candidates_found_terms_unconfirmed",
        "required_before_new_claim": ["terms confirmation", "guarded conversion", "no-leakage/source-CV", "h100 candidate terms confirmation"],
    },
    {
        "target_id": "tgsim",
        "domain": "TGSIM",
        "official_url": "https://data.transportation.gov/",
        "data_role": "traffic diagnostic only",
        "paths": ["data/stage5_world_state/tgsim", "data/stage5b_world_state/tgsim", "data/stage5b_world_state/tgsim_i90"],
        "known_status": "diagnostic_only_not_pedestrian_world_model_success",
        "required_before_new_claim": ["keep as diagnostic; do not report pedestrian/top-down success"],
    },
    {
        "target_id": "aerialmpt_or_other_topdown",
        "domain": "AerialMPT",
        "official_url": "user_or_web_verified_official_url_required",
        "data_role": "external top-down/aerial candidate",
        "paths": ["data/aerialmpt/DLR_AerialMPT_Dataset.zip", "data/stage14_multimodal_episodes/aerialmpt"],
        "known_status": "local_artifact_or_cache_found_but_official_terms_url_unverified",
        "required_before_new_claim": ["official URL", "terms confirmation", "parser/schema", "guarded conversion", "no-leakage/source-CV"],
    },
]


def _inspect_path(path: str) -> dict[str, Any]:
    p = Path(path)
    exists = p.exists()
    row: dict[str, Any] = {
        "path": path,
        "exists": exists,
        "is_file": p.is_file() if exists else False,
        "is_dir": p.is_dir() if exists else False,
        "size_mb": None,
        "sample_extensions": {},
        "sample_count": 0,
    }
    if not exists:
        return row
    if p.is_file():
        row["size_mb"] = round(p.stat().st_size / (1024 * 1024), 3)
        row["sample_extensions"] = {p.suffix.lower() or "<none>": 1}
        row["sample_count"] = 1
        return row
    total = 0
    ext_counts: dict[str, int] = {}
    for idx, child in enumerate(p.rglob("*")):
        if idx >= 2000:
            break
        if child.is_file():
            total += child.stat().st_size
            ext = child.suffix.lower() or "<none>"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
    row["size_mb"] = round(total / (1024 * 1024), 3)
    row["sample_extensions"] = dict(sorted(ext_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8])
    row["sample_count"] = sum(ext_counts.values())
    return row


def _source_actions_by_target(source_action: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    for row in source_action.get("consolidated_actions", []):
        for key in [str(row.get("target", "")), str(row.get("domain", ""))]:
            if key:
                out.setdefault(key, []).append(row)
    return out


def _calibration_by_id(calibration: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in calibration.get("datasets", []):
        dataset_id = str(row.get("dataset_id", ""))
        if dataset_id:
            out[dataset_id] = row
    return out


def _target_row(target: Mapping[str, Any], source_actions: Mapping[str, list[Mapping[str, Any]]], calibration: Mapping[str, Mapping[str, Any]], unified_queue: Mapping[str, Any]) -> dict[str, Any]:
    path_rows = [_inspect_path(path) for path in target["paths"]]
    local_path_found = any(row["exists"] for row in path_rows)
    actions = list(source_actions.get(str(target["target_id"]), [])) + list(source_actions.get(str(target["domain"]), []))
    calibration_row = calibration.get(str(target["target_id"]), {})
    queue_count = int(unified_queue.get("summary", {}).get("unified_queue_count", 0) or 0)
    existing_converted = str(target["target_id"]) == "sdd" and any("stage21_sdd_world_state" in row["path"] and row["exists"] for row in path_rows)
    new_conversion_ready = False
    if queue_count > 0:
        new_conversion_ready = any(str(item.get("dataset_id", "")) == str(target["target_id"]) for item in unified_queue.get("unified_queue", []))
    blockers = []
    if not local_path_found:
        blockers.append("local_path_missing_or_not_verified")
    if not existing_converted:
        blockers.append("explicit_terms_or_source_identity_not_confirmed")
    if target["target_id"] in {"trajnetplusplus_official"}:
        blockers.append("h100_long_source_support_not_closed")
    if target["target_id"] in {"ucy_crowd_original"}:
        blockers.append("ucy_terms_and_h100_candidate_confirmation_missing")
    if target["target_id"] in {"aerialmpt_or_other_topdown"}:
        blockers.append("official_url_or_terms_not_verified")
    if target["target_id"] == "tgsim":
        blockers.append("diagnostic_only_not_topdown_pedestrian_claim")
    return {
        "target_id": target["target_id"],
        "domain": target["domain"],
        "official_url": target["official_url"],
        "data_role": target["data_role"],
        "result_source": "fresh_local_path_scan_plus_cached_verified_legal_state",
        "path_status": path_rows,
        "local_path_found": local_path_found,
        "existing_converted_artifact_found": existing_converted,
        "new_conversion_ready_now": new_conversion_ready,
        "new_conversion_executed": False,
        "new_evaluation_executed": False,
        "known_status": target["known_status"],
        "calibration_state": calibration_row.get("calibration_state", "not_verified_this_target"),
        "metric_claim_allowed": bool(calibration_row.get("metric_claim_allowed", False)) and False,
        "seconds_claim_allowed": bool(calibration_row.get("seconds_claim_allowed", False)) and False,
        "source_actions": [
            {
                "action_id": row.get("action_id"),
                "status": row.get("status"),
                "priority": row.get("priority"),
                "missing": row.get("missing", []),
                "next_user_action": row.get("next_user_action"),
            }
            for row in actions[:5]
        ],
        "blockers": blockers,
        "required_before_new_claim": list(target["required_before_new_claim"]),
        "next_action": _next_action(target, blockers, actions),
    }


def _next_action(target: Mapping[str, Any], blockers: list[str], actions: list[Mapping[str, Any]]) -> str:
    if target["target_id"] == "sdd":
        return "Do not relabel as metric/seconds-level; next useful action is homography/scale/FPS verification only."
    if target["target_id"] == "tgsim":
        return "Keep as traffic diagnostic-only evidence; do not convert or report it as pedestrian/top-down world-model success."
    if actions:
        return str(actions[0].get("next_user_action") or "complete source action template and rerun guarded queue")
    if "local_path_missing_or_not_verified" in blockers:
        return "Provide legal local path and official source/terms confirmation before conversion."
    return "Fill source terms confirmation template, rerun validator, guarded conversion queue, and no-leakage/source-CV."


def _summary(rows: list[Mapping[str, Any]], source_action: Mapping[str, Any], unified_queue: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "targets_audited": len(rows),
        "local_path_found_targets": sum(1 for row in rows if row.get("local_path_found")),
        "existing_converted_artifact_targets": sum(1 for row in rows if row.get("existing_converted_artifact_found")),
        "new_conversion_ready_targets": sum(1 for row in rows if row.get("new_conversion_ready_now")),
        "new_conversions_executed": 0,
        "new_evaluations_executed": 0,
        "source_action_conversion_ready_now": int(source_action.get("summary", {}).get("conversion_ready_now", -1)),
        "unified_queue_count": int(unified_queue.get("summary", {}).get("unified_queue_count", -1)),
        "highest_priority_next_action": source_action.get("summary", {}).get("highest_priority_blocker", "FW-TERMS-ucy_crowd_original"),
        "metric_claim_allowed": False,
        "seconds_claim_allowed": False,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "all_required_targets_audited": summary.get("targets_audited") >= 7,
        "local_scan_performed": summary.get("local_path_found_targets") >= 5,
        "data_calibration_loaded": payload.get("input_status", {}).get("data_calibration_exists") is True,
        "source_action_loaded": payload.get("input_status", {}).get("source_action_exists") is True,
        "unified_queue_loaded": payload.get("input_status", {}).get("unified_queue_exists") is True,
        "no_new_conversion_ready_overclaim": summary.get("new_conversion_ready_targets") == 0,
        "source_action_ready_zero_preserved": summary.get("source_action_conversion_ready_now") == 0,
        "every_blocked_target_has_next_action": all(row.get("next_action") for row in payload.get("target_rows", [])),
        "user_action_written": payload.get("user_action_required_written") is True,
        "no_download_conversion_eval": claim.get("download_executed") is False
        and claim.get("conversion_executed") is False
        and claim.get("evaluation_executed") is False,
        "no_metric_seconds_overclaim": claim.get("global_metric_claim_allowed") is False
        and claim.get("global_seconds_claim_allowed") is False,
        "no_true3d_foundation_overclaim": claim.get("true_3d") is False and claim.get("foundation_world_model") is False,
        "stage5c_false": claim.get("stage5c_executed") is False,
        "smc_false": claim.get("smc_enabled") is False,
    }
    passed = sum(bool(v) for v in gates.values())
    total = len(gates)
    return {
        "passed": passed,
        "total": total,
        "gates": gates,
        "verdict": "stage42_ga_live_source_calibration_recheck_pass" if passed == total else "stage42_ga_live_source_calibration_recheck_partial",
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    lines = [
        "# Stage42-GA Live Source / Calibration Recheck",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_ga_gate']['passed']} / {payload['stage42_ga_gate']['total']}`",
        f"- verdict: `{payload['stage42_ga_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        "- 当前不是 true 3D world model。",
        "- 当前不是 large-scale foundation world model。",
        "- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
        "- GA fresh-scans local paths and cached legal/calibration state; it does not download, convert, train, or evaluate.",
        "- Local files are not counted as legal conversion readiness without explicit terms/path/source identity confirmation.",
        "- Stage5C latent generative is not executed; SMC is not enabled.",
        "",
        "## Summary",
        "",
        f"- targets_audited: `{summary['targets_audited']}`",
        f"- local_path_found_targets: `{summary['local_path_found_targets']}`",
        f"- existing_converted_artifact_targets: `{summary['existing_converted_artifact_targets']}`",
        f"- new_conversion_ready_targets: `{summary['new_conversion_ready_targets']}`",
        f"- source_action_conversion_ready_now: `{summary['source_action_conversion_ready_now']}`",
        f"- unified_queue_count: `{summary['unified_queue_count']}`",
        f"- highest_priority_next_action: `{summary['highest_priority_next_action']}`",
        "",
        "## Target Rows",
        "",
        "| target | domain | local path | converted/cached | new ready | calibration | blockers | next action |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in payload["target_rows"]:
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | `{}` | {} | {} |".format(
                row["target_id"],
                row["domain"],
                row["local_path_found"],
                row["existing_converted_artifact_found"],
                row["new_conversion_ready_now"],
                row["calibration_state"],
                "<br>".join(row["blockers"]) or "none",
                row["next_action"],
            )
        )
    lines.extend(["", "## Gate", "", "| gate | pass |", "| --- | ---: |"])
    for gate, ok in payload["stage42_ga_gate"]["gates"].items():
        lines.append(f"| `{gate}` | {bool(ok)} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- 本机有多个 external/raw/cache artifact，但本轮没有任何新 source 达到 guarded conversion readiness。",
            "- SDD 仍是 pixel/raw-frame；external 仍是 dataset-local/raw-frame 或 diagnostic。",
            "- 下一步最高优先级仍是填 UCY official terms/path/source identity，然后 rerun terms validator、guarded queue、conversion/no-leakage/source-CV。",
        ]
    )
    return lines


def _render_user_actions(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-GA Source / Calibration",
        "",
        "- No new download, conversion, or evaluation was executed.",
        "- Local files are insufficient for new claims without source terms/path/source-identity confirmation.",
        "",
        "| target | official URL | action |",
        "| --- | --- | --- |",
    ]
    for row in payload["target_rows"]:
        if row["new_conversion_ready_now"]:
            continue
        lines.append(f"| `{row['target_id']}` | {row['official_url']} | {row['next_action']} |")
    lines.extend(
        [
            "",
            "## Suggested Commands After User Confirmation",
            "",
            "```bash",
            ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
            ".venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py",
            ".venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py",
            ".venv-pytorch/bin/python run_stage42_source_support_closure_audit.py",
            "```",
        ]
    )
    return lines


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-GA Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    write_md(GATE_MD, lines)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "## Stage42-GA Live Source / Calibration Recheck",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{payload['stage42_ga_gate']['passed']} / {payload['stage42_ga_gate']['total']}`; verdict `{payload['stage42_ga_gate']['verdict']}`.",
        "- role: fresh local path scan plus cached legal/calibration readiness recheck; no download, no conversion, no training, no evaluation.",
        f"- targets audited: `{s['targets_audited']}`; local-path-found targets `{s['local_path_found_targets']}`; existing converted/cache targets `{s['existing_converted_artifact_targets']}`.",
        f"- new conversion-ready targets: `{s['new_conversion_ready_targets']}`; source_action conversion_ready_now `{s['source_action_conversion_ready_now']}`; unified queue `{s['unified_queue_count']}`.",
        f"- highest-priority next action: `{s['highest_priority_next_action']}`.",
        "- boundary: local file presence is not legal conversion readiness; protected dataset-local/raw-frame 2.5D only; no true 3D/foundation/global metric/seconds/Stage5C/SMC claim.",
    ]
    for path in [README_RESULTS, M3W_README, ONE_FILE_SUMMARY]:
        _replace_section(path, "STAGE42_GA_LIVE_SOURCE_CALIBRATION_RECHECK", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-GA live source calibration recheck"
    state["current_verdict"] = payload["stage42_ga_gate"]["verdict"]
    state["stage42_ga_live_source_calibration_recheck"] = {
        "source": payload["source"],
        "path": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "updated_at": payload["generated_at_utc"],
        "gate": payload["stage42_ga_gate"],
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_live_source_calibration_recheck() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    source_action = read_json(SOURCE_ACTION_JSON, {})
    data_calibration = read_json(DATA_CALIBRATION_JSON, {})
    unified_queue = read_json(UNIFIED_QUEUE_JSON, {})
    source_actions = _source_actions_by_target(source_action)
    calibration = _calibration_by_id(data_calibration)
    target_rows = [_target_row(target, source_actions, calibration, unified_queue) for target in TARGETS]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([SOURCE_ACTION_JSON, DATA_CALIBRATION_JSON, UNIFIED_QUEUE_JSON, SOURCE_TERMS_TEMPLATE_JSON, UCY_H100_TEMPLATE_JSON]),
        "input_status": {
            "source_action_exists": SOURCE_ACTION_JSON.exists(),
            "data_calibration_exists": DATA_CALIBRATION_JSON.exists(),
            "unified_queue_exists": UNIFIED_QUEUE_JSON.exists(),
            "source_terms_template_exists": SOURCE_TERMS_TEMPLATE_JSON.exists(),
            "ucy_h100_template_exists": UCY_H100_TEMPLATE_JSON.exists(),
        },
        "summary": _summary(target_rows, source_action, unified_queue),
        "target_rows": target_rows,
        "claim_boundary": CLAIM_BOUNDARY,
        "user_action_required_written": True,
    }
    payload["stage42_ga_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_actions(payload))
    _write_gate(payload["stage42_ga_gate"])
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


__all__ = [
    "run_stage42_live_source_calibration_recheck",
    "_inspect_path",
    "_target_row",
    "_gate",
    "_summary",
]
