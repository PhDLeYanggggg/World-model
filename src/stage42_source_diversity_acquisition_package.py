from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
BV_JSON = OUT_DIR / "source_acquisition_status_stage42.json"
CC_JSON = OUT_DIR / "independent_t50_source_inventory_stage42.json"

REPORT_JSON = OUT_DIR / "source_diversity_acquisition_package_stage42.json"
REPORT_MD = OUT_DIR / "source_diversity_acquisition_package_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_cd_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_diversity_stage42.md"


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CD 是官方源获取与接入准备包，不下载数据，不转换数据，不训练模型。",
    "Stage42-CB/CC 已证明 t50 source diversity blocker 仍 active；本包不能把 blocker 包装成完成。",
    "OpenTraj toolkit / wrapper 许可不能自动覆盖 ETH/UCY/TrajNet 等底层第三方数据许可。",
    "future endpoints / waypoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "Stage5C 未执行，SMC 未启用。",
]


OFFICIAL_TARGETS = [
    {
        "id": "ucy_crowd_original",
        "name": "UCY Crowd Data original source",
        "official_url": "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
        "source_confidence": "official_url_known_manual_verification_required",
        "priority": "critical",
        "target_blocker": "UCY_students_t50_source_support",
        "expected_role": "independent t50-capable UCY/students/crowd source if legal long tracks are provided",
        "why_needed": "Stage42-CB/CC show UCY t50 evidence is source-concentrated; another independent UCY/students-family long source is needed for source-CV.",
        "requires_manual_terms_acceptance": True,
        "requires_login_or_application": False,
        "auto_download_allowed": False,
        "local_path_candidates": [
            "/Users/yangyue/Downloads/UCY",
            "/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY",
            "/Users/yangyue/Downloads/OpenTraj/datasets/UCY",
        ],
        "conversion_after_user_path": [
            "verify original dataset terms and source identity",
            "convert to external feature-store schema",
            "build source-level train/internal-val/final-test split without test endpoint goals",
            "rerun t50 source-CV and protected policy final test once",
        ],
        "claim_if_missing": "blocked_user_action_required",
    },
    {
        "id": "eth_biwi_original",
        "name": "ETH / BIWI walking pedestrians original source",
        "official_url": "https://vision.ee.ethz.ch/datsets.html",
        "source_confidence": "official_eth_page",
        "priority": "critical",
        "target_blocker": "ETH_seq_t50_source_support",
        "expected_role": "same-family ETH_seq long-track support with homography/metadata hints when legally usable",
        "why_needed": "ETH_seq t50 is still blocked because ETH-Person XML technical signal did not safely repair the ETH_seq holdout and terms remain unresolved.",
        "requires_manual_terms_acceptance": True,
        "requires_login_or_application": False,
        "auto_download_allowed": False,
        "local_path_candidates": [
            "/Users/yangyue/Downloads/ETH_UCY",
            "/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/ETH",
            "/Users/yangyue/Downloads/OpenTraj/datasets/ETH",
        ],
        "conversion_after_user_path": [
            "verify ETH/BIWI source terms and citation requirements",
            "separate alternate-format files from independent held-out sources",
            "audit frame step/FPS/homography only as source-specific evidence",
            "rerun ETH_seq t50 source-CV under validation-only policy",
        ],
        "claim_if_missing": "blocked_user_action_required",
    },
    {
        "id": "trajnetplusplus_official",
        "name": "TrajNet++ official challenge/source page",
        "official_url": "https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/",
        "secondary_url": "https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge",
        "source_confidence": "official_epfl_and_aicrowd_reference",
        "priority": "high",
        "target_blocker": "TrajNet_raw_long_t100_source_support",
        "expected_role": "scene-format diagnostics or long raw tracks only if legally provided; challenge snippets alone are too short for raw-frame t100 source-CV",
        "why_needed": "Stage42-BV/CC show local TrajNet rows are dominated by short snippets or synthetic diagnostic data, not legal raw long-track t100 support.",
        "requires_manual_terms_acceptance": True,
        "requires_login_or_application": True,
        "auto_download_allowed": False,
        "local_path_candidates": [
            "/Users/yangyue/Downloads/trajnetplusplusdataset",
            "/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet",
            "/Users/yangyue/Downloads/OpenTraj/datasets/TrajNet",
        ],
        "conversion_after_user_path": [
            "verify official challenge or source terms",
            "separate scene snippets from raw long tracks",
            "do not count synthetic ORCA diagnostic rows as real external evidence",
            "run raw-frame horizon audit before any t100 claim",
        ],
        "claim_if_missing": "short_scene_diagnostic_only",
    },
    {
        "id": "opentraj_toolkit",
        "name": "OpenTraj official GitHub toolkit",
        "official_url": "https://github.com/crowdbotp/OpenTraj",
        "source_confidence": "official_github_toolkit",
        "priority": "medium",
        "target_blocker": "loader_and_metadata_support",
        "expected_role": "loader/toolkit support and dataset indexing; not an independent data license by itself",
        "why_needed": "OpenTraj helps identify paths/loaders, but Stage42 must preserve underlying ETH/UCY/TrajNet terms and source identity.",
        "requires_manual_terms_acceptance": False,
        "requires_login_or_application": False,
        "auto_download_allowed": False,
        "local_path_candidates": [
            "/Users/yangyue/Downloads/World/external_data/OpenTraj",
            "/Users/yangyue/Downloads/OpenTraj",
        ],
        "conversion_after_user_path": [
            "use only as toolkit/index metadata",
            "resolve every underlying dataset to its own terms",
            "do not count toolkit clone as converted/evaluated data",
        ],
        "claim_if_missing": "toolkit_only_not_dataset_claim",
    },
    {
        "id": "aerialmpt_or_other_topdown",
        "name": "Additional legal top-down pedestrian/drone long-track source",
        "official_url": "user_or_web_verified_official_url_required",
        "source_confidence": "not_run_until_user_or_web_source_verified",
        "priority": "medium",
        "target_blocker": "external_source_diversity",
        "expected_role": "independent non-SDD top-down pedestrian/drone t50/t100 source if terms permit",
        "why_needed": "A non-ETH/UCY/TrajNet source would test whether M3W is broader than the current source families.",
        "requires_manual_terms_acceptance": True,
        "requires_login_or_application": True,
        "auto_download_allowed": False,
        "local_path_candidates": [
            "/Users/yangyue/Downloads/AerialMPT",
            "/Users/yangyue/Downloads/World/external_data/AerialMPT",
        ],
        "conversion_after_user_path": [
            "verify official source and terms",
            "audit coordinate unit, frame step, track length, scene identity, and agent type",
            "convert only if no-leakage source-level split is possible",
        ],
        "claim_if_missing": "future_external_expansion_only",
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _target_status(target: Mapping[str, Any], cc: Mapping[str, Any], bv: Mapping[str, Any]) -> dict[str, Any]:
    summary = cc["summary"]
    blocker_by_id = {row["blocker_id"]: row for row in bv["blocker_matrix"]}
    blocker = blocker_by_id.get(str(target["target_blocker"]), {})
    local_candidates = []
    for path in target["local_path_candidates"]:
        p = Path(path)
        local_candidates.append({"path": path, "exists": p.exists(), "kind": "directory" if p.is_dir() else "file" if p.exists() else "missing"})
    return {
        "id": target["id"],
        "priority": target["priority"],
        "target_blocker": target["target_blocker"],
        "blocker_status": blocker.get("status", "not_applicable"),
        "auto_download_allowed": target["auto_download_allowed"],
        "local_candidates": local_candidates,
        "local_path_found": any(row["exists"] for row in local_candidates),
        "can_claim_converted_now": False,
        "can_claim_source_diversity_repair_now": False,
        "reason": (
            "Stage42-CC found no unused independent ready-to-claim t50 source"
            if summary["unused_candidate_t50_sources"] == 0
            else "candidate files still require conversion/no-leakage/source-CV/final test"
        ),
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bv = _load_json(BV_JSON)
    cc = _load_json(CC_JSON)
    target_status = [_target_status(target, cc, bv) for target in OFFICIAL_TARGETS]
    summary = {
        "source": "fresh_stage42_cd_source_diversity_acquisition_package",
        "official_targets": len(OFFICIAL_TARGETS),
        "critical_targets": sum(1 for row in OFFICIAL_TARGETS if row["priority"] == "critical"),
        "auto_download_targets": sum(1 for row in OFFICIAL_TARGETS if row["auto_download_allowed"]),
        "manual_or_terms_targets": sum(
            1 for row in OFFICIAL_TARGETS if row["requires_manual_terms_acceptance"] or row["requires_login_or_application"]
        ),
        "local_paths_found": sum(1 for row in target_status if row["local_path_found"]),
        "converted_datasets_now": 0,
        "source_diversity_repair_ready_now": False,
        "broad_source_generalization_claim_allowed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": "fresh_stage42_cd_source_diversity_acquisition_package",
        "stage": "Stage42-CD Source Diversity Acquisition Package",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(BV_JSON), str(CC_JSON)]),
        "current_facts": CURRENT_FACTS,
        "input_reports": {
            "stage42_bv_verdict": bv["stage42_bv_gate"]["verdict"],
            "stage42_cc_verdict": cc["stage42_cc_gate"]["verdict"],
            "stage42_cc_unused_candidate_t50_sources": cc["summary"]["unused_candidate_t50_sources"],
            "stage42_cc_source_diversity_repair_ready": cc["summary"]["source_diversity_repair_ready"],
        },
        "summary": summary,
        "official_targets": OFFICIAL_TARGETS,
        "target_status": target_status,
        "next_commands_after_user_data": [
            "run a source-specific conversion script for the provided path",
            "run no-leakage audit with source-level train/internal-val/final-test split",
            "run protected t50/t100 source-CV only after terms and split are verified",
            "update paper package only after fresh final-test evidence exists",
        ],
        "user_action_required": [
            {
                "priority": row["priority"],
                "dataset": row["name"],
                "official_url": row["official_url"],
                "secondary_url": row.get("secondary_url"),
                "action": "Provide a legal local path or proof of accepted terms before conversion.",
                "why": row["why_needed"],
            }
            for row in OFFICIAL_TARGETS
            if row["requires_manual_terms_acceptance"] or row["requires_login_or_application"]
        ],
        "claim_boundary": {
            "inventory_counted_as_converted": False,
            "registry_only_counted_as_converted": False,
            "auto_download_executed": False,
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_cd_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    gates = {
        "bv_input_verified": payload["input_reports"]["stage42_bv_verdict"]
        == "stage42_bv_source_acquisition_status_pass_blockers_actionable",
        "cc_input_verified": payload["input_reports"]["stage42_cc_verdict"]
        == "stage42_cc_independent_t50_source_inventory_pass",
        "official_targets_present": summary["official_targets"] >= 4,
        "critical_targets_present": summary["critical_targets"] >= 2,
        "manual_terms_blockers_explicit": summary["manual_or_terms_targets"] >= 3,
        "no_auto_download_executed": payload["claim_boundary"]["auto_download_executed"] is False,
        "not_counted_as_converted": payload["claim_boundary"]["inventory_counted_as_converted"] is False
        and payload["claim_boundary"]["registry_only_counted_as_converted"] is False,
        "source_diversity_not_overclaimed": summary["broad_source_generalization_claim_allowed"] is False,
        "next_commands_defined": bool(payload["next_commands_after_user_data"]),
        "user_action_written": bool(payload["user_action_required"]),
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_cd_source_diversity_acquisition_package_pass" if passed == total else "stage42_cd_source_diversity_acquisition_package_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-CD Source Diversity Acquisition Package",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_cd_gate']['passed']} / {payload['stage42_cd_gate']['total']}`",
        f"- verdict: `{payload['stage42_cd_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- official_targets: `{s['official_targets']}`",
        f"- critical_targets: `{s['critical_targets']}`",
        f"- auto_download_targets: `{s['auto_download_targets']}`",
        f"- manual_or_terms_targets: `{s['manual_or_terms_targets']}`",
        f"- local_paths_found: `{s['local_paths_found']}`",
        f"- converted_datasets_now: `{s['converted_datasets_now']}`",
        f"- source_diversity_repair_ready_now: `{s['source_diversity_repair_ready_now']}`",
        f"- broad_source_generalization_claim_allowed: `{s['broad_source_generalization_claim_allowed']}`",
        "",
        "## Official / Manual Targets",
        "",
        "| id | priority | blocker | official URL | auto download | local path found | next claim status |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    status_by_id = {row["id"]: row for row in payload["target_status"]}
    for row in payload["official_targets"]:
        status = status_by_id[row["id"]]
        lines.append(
            f"| `{row['id']}` | `{row['priority']}` | `{row['target_blocker']}` | {row['official_url']} | "
            f"{row['auto_download_allowed']} | {status['local_path_found']} | `{row['claim_if_missing']}` |"
        )
    lines += [
        "",
        "## Required Pipeline After User Provides Data",
        "",
        *[f"- {cmd}" for cmd in payload["next_commands_after_user_data"]],
        "",
        "## Interpretation",
        "",
        "- Stage42-CD does not repair source diversity by itself.",
        "- No auto-download was executed because all priority source-diversity targets need manual terms/path verification or are toolkit/challenge references.",
        "- A file path, registry row, toolkit clone, or alternate representation is not counted as a converted/evaluated dataset.",
        "- Broad source-level generalization remains blocked until fresh conversion/no-leakage/source-CV/final-test evidence exists.",
    ]
    return lines


def _render_user_actions(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-CD Source Diversity",
        "",
        "Stage42-CD identifies the official/manual source actions needed to repair the current source-diversity blocker.",
        "",
    ]
    for row in payload["user_action_required"]:
        lines += [
            f"## {row['priority'].upper()} - {row['dataset']}",
            "",
            f"- official_url: {row['official_url']}",
            f"- secondary_url: {row.get('secondary_url') or 'none'}",
            f"- action: {row['action']}",
            f"- why: {row['why']}",
            "",
        ]
    lines.append("Do not upload or commit third-party raw data. Provide local paths only after terms are accepted.")
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_cd_gate"]
    lines = [
        "# Stage42-CD Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | {bool(value)} |")
    return lines


def run_stage42_source_diversity_acquisition_package() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_actions(payload))
    return payload


if __name__ == "__main__":
    result = run_stage42_source_diversity_acquisition_package()
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, sort_keys=True))
