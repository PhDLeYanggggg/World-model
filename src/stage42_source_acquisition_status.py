from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")

BR_JSON = OUT_DIR / "calibrated_t50_source_support_gap_stage42.json"
BT_JSON = OUT_DIR / "eth_seq_t50_support_dry_run_stage42.json"
BM_JSON = OUT_DIR / "eth_person_terms_audit_stage42.json"
BK_JSON = OUT_DIR / "post_bj_local_source_verification_stage42.json"
BU_JSON = OUT_DIR / "ucy_students_t50_source_support_stage42.json"
BJ_JSON = OUT_DIR / "post_bi_t100_source_package_stage42.json"
BN_JSON = OUT_DIR / "source_time_geometry_calibration_stage42.json"

REPORT_JSON = OUT_DIR / "source_acquisition_status_stage42.json"
REPORT_MD = OUT_DIR / "source_acquisition_status_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bv_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_acquisition_stage42.md"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BV 是 source acquisition / blocker matrix，不训练模型，不下载数据。",
    "OpenTraj toolkit MIT 许可证不能自动覆盖 ETH/UCY/TrajNet 等第三方数据许可。",
    "TrajNet++ challenge scene snippets 不能自动等同于 raw long-track t50/t100 support。",
    "future endpoints / waypoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

OFFICIAL_REFERENCES = [
    {
        "id": "trajnet_epfl_official",
        "name": "TrajNet++ EPFL official page",
        "url": "https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/",
        "source_confidence": "official",
        "use_for_m3w": "official challenge/source reference only; does not by itself provide long raw t50/t100 support",
        "notes": "EPFL describes TrajNet++ as an open benchmark/challenge with data and evaluation code on AIcrowd, and scripts under MIT license.",
    },
    {
        "id": "trajnet_aicrowd_official",
        "name": "TrajNet++ AIcrowd challenge",
        "url": "https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge",
        "source_confidence": "official_challenge_page",
        "use_for_m3w": "official scene-format reference and access/action page; local challenge snippets remain too short for raw-frame t100 repair",
        "notes": "AIcrowd describes 21-frame scenes with first 9 observed and last 12 predicted; useful benchmark, but not a long raw-track source in the current local inventory.",
    },
    {
        "id": "opentraj_github",
        "name": "OpenTraj GitHub",
        "url": "https://github.com/crowdbotp/OpenTraj",
        "source_confidence": "official_github",
        "use_for_m3w": "toolkit and local dataset index reference; underlying dataset terms must be audited separately",
        "notes": "OpenTraj is useful for loaders and metadata, but its toolkit license is not accepted as blanket permission for every third-party dataset.",
    },
    {
        "id": "eth_cvl_dataset_page",
        "name": "ETH CVL datasets page",
        "url": "https://vision.ee.ethz.ch/datsets.html",
        "source_confidence": "official_eth_page",
        "use_for_m3w": "official ETH data/terms reference; research-purpose boundary must be preserved",
        "notes": "ETH CVL page says maintained data are for research purposes unless stated differently and asks users to cite authors.",
    },
    {
        "id": "ucy_crowd_data_page",
        "name": "UCY crowd data page",
        "url": "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
        "source_confidence": "official_url_known_but_fetch_failed_in_current_web_tool",
        "use_for_m3w": "official UCY crowd-data action target; user/manual verification required if direct access fails",
        "notes": "The URL is widely referenced as the UCY data source, but the current web fetch failed, so no automatic download or terms claim is made.",
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_verdict(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key, {})
    return value.get("verdict") if isinstance(value, Mapping) else None


def _artifact(path: Path, verdict_key: str | None = None) -> dict[str, Any]:
    exists = path.exists()
    payload: dict[str, Any] = {}
    verdict = None
    if exists:
        payload = _load_json(path)
        if verdict_key:
            verdict = _safe_verdict(payload, verdict_key)
    return {
        "path": str(path),
        "exists": exists,
        "source": payload.get("source"),
        "verdict": verdict,
        "payload": payload,
    }


def _blocker_matrix(artifacts: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    br = artifacts["br"]["payload"]
    bt = artifacts["bt"]["payload"]
    bm = artifacts["bm"]["payload"]
    bk = artifacts["bk"]["payload"]
    bu = artifacts["bu"]["payload"]
    bj = artifacts["bj"]["payload"]
    bn = artifacts["bn"]["payload"]

    br_family = br.get("family_support", {})
    bu_summary = bu.get("summary", {})
    bt_summary = bt.get("summary", {})
    bm_summary = bm.get("summary", {})
    bk_summary = bk.get("summary", {})
    bj_summary = bj.get("summary", {})
    bj_support = bj.get("domain_support", {})
    calibrated_sources = bn.get("summary", {}).get("calibrated_sources", [])

    return [
        {
            "blocker_id": "ETH_seq_t50_source_support",
            "source": "cached_verified_plus_fresh_stage42_bv",
            "status": "blocked",
            "evidence": {
                "br_family_need": br_family.get("ETH_seq", {}).get("additional_calibrated_sources_needed"),
                "bt_candidate_sources": bt_summary.get("candidate_sources"),
                "bt_safe_positive_h50_fold_count": bt_summary.get("safe_positive_h50_fold_count"),
                "bt_eth_seq_eth_repaired": bt_summary.get("eth_seq_eth_repaired"),
                "eth_person_terms_verified": bm_summary.get("official_terms_verified"),
            },
            "root_cause": "ETH-Person XML has technical same-family h50 signal, but terms remain unverified and the validation-only dry-run does not safely repair the actual ETH_seq_eth holdout.",
            "next_action": "Verify ETH-Person terms or provide an official/legal source-compatible ETH_seq long-track source; then rerun conversion, no-leakage, source-CV, and t50 policy training.",
            "claim_allowed": "technical_blocker_only",
        },
        {
            "blocker_id": "UCY_students_t50_source_support",
            "source": "fresh_stage42_bu_verified",
            "status": "blocked_narrowed",
            "evidence": {
                "independent_t50_capable_sources": bu_summary.get("independent_t50_capable_sources"),
                "new_independent_t50_sources_found": bu_summary.get("new_independent_t50_sources_found"),
                "additional_independent_t50_sources_still_needed": bu_summary.get("additional_independent_t50_sources_still_needed"),
                "source_cv_ready": bu_summary.get("source_cv_ready"),
            },
            "root_cause": "students001 and students003 are t50-capable independent sources, but students002 is too short and duplicate formats cannot be counted as independent support.",
            "next_action": "Provide one more legal independent t50-capable UCY_students-family long-track source before train/val/holdout source-CV can be attempted.",
            "claim_allowed": "blocker_narrowed_not_positive_transfer",
        },
        {
            "blocker_id": "TrajNet_raw_long_t100_source_support",
            "source": "cached_verified_from_stage42_bk_bj",
            "status": "blocked",
            "evidence": {
                "trajnet_parsed_files": bk_summary.get("trajnet_parsed_files"),
                "trajnet_t100_capable_files": bk_summary.get("trajnet_t100_capable_files"),
                "trajnet_independent_sources": bj_support.get("TrajNet", {}).get("independent_sources"),
                "trajnet_additional_sources_needed": bj_summary.get("trajnet_additional_sources_needed"),
            },
            "root_cause": "Local TrajNet files parse as fixed short challenge snippets rather than raw long tracks, so they cannot repair raw-frame t100 source-CV.",
            "next_action": "Use TrajNet++ challenge data only for scene-format diagnostics, or provide legal raw long-track TrajNet-compatible sources if t100 source-CV is required.",
            "claim_allowed": "short_scene_diagnostic_only",
        },
        {
            "blocker_id": "ETH_UCY_global_t100_source_support",
            "source": "cached_verified_from_stage42_bj_bm",
            "status": "blocked",
            "evidence": {
                "eth_ucy_independent_sources": bj_support.get("ETH_UCY", {}).get("independent_sources"),
                "eth_ucy_additional_sources_needed": bj_summary.get("eth_ucy_additional_sources_needed"),
                "eth_person_terms_verified": bm_summary.get("official_terms_verified"),
                "eth_person_technical_t100_positive": bm_summary.get("bl_technical_t100_all_folds_safe_positive"),
            },
            "root_cause": "ETH-Person XML t100 dry-run is technically positive but still terms-unverified; official/deployable global t100 cannot advance without permission and rerun.",
            "next_action": "Confirm official ETH-Person terms or provide permitted ETH_UCY long raw sources, then rerun official conversion and strict source-CV.",
            "claim_allowed": "technical_dry_run_only",
        },
        {
            "blocker_id": "global_metric_seconds_claim",
            "source": "cached_verified_from_stage42_bn",
            "status": "blocked",
            "evidence": {
                "calibrated_sources": calibrated_sources,
                "global_metric_claim_allowed": False,
                "global_seconds_claim_allowed": False,
            },
            "root_cause": "Only source-specific calibration evidence exists; SDD remains pixel raw-frame and external data remain dataset-local or source-specific calibrated subsets.",
            "next_action": "Report metric/time only inside explicitly source-specific calibrated subsets after conversion/eval; keep global M3W raw-frame/dataset-local.",
            "claim_allowed": "source_specific_subset_only",
        },
    ]


def _user_actions(blockers: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "priority": "critical",
            "blocker_id": row["blocker_id"],
            "status": row["status"],
            "action": row["next_action"],
            "allowed_next_command": "rerun relevant Stage42 conversion/source-CV only after legal source/path/terms are provided",
        }
        for row in blockers
        if row["status"] != "ready"
    ]


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    artifacts = {
        "br": _artifact(BR_JSON, "stage42_br_gate"),
        "bt": _artifact(BT_JSON, "stage42_bt_gate"),
        "bm": _artifact(BM_JSON, "stage42_bm_gate"),
        "bk": _artifact(BK_JSON, "stage42_bk_gate"),
        "bu": _artifact(BU_JSON, "stage42_bu_gate"),
        "bj": _artifact(BJ_JSON, "stage42_bj_gate"),
        "bn": _artifact(BN_JSON, "stage42_bn_gate"),
    }
    blockers = _blocker_matrix(artifacts)
    actions = _user_actions(blockers)
    summary = {
        "source": "fresh_stage42_bv_source_acquisition_status",
        "blockers_total": len(blockers),
        "blockers_ready": sum(1 for row in blockers if row["status"] == "ready"),
        "blockers_active": sum(1 for row in blockers if row["status"] != "ready"),
        "ucy_students_blocker_narrowed": True,
        "eth_seq_blocker_resolved": False,
        "trajnet_raw_long_source_resolved": False,
        "global_t100_positive_claim_allowed": False,
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "auto_download_executed": False,
        "registry_only_counted_as_converted": False,
        "stage5c_executed": False,
        "smc_enabled": False,
        "next_best_actions": [
            "provide one more independent t50-capable UCY_students-family source",
            "verify ETH-Person terms before using XML dry-run as official evidence",
            "provide legal raw long TrajNet-compatible tracks if t100 source-CV is required",
        ],
    }
    payload: dict[str, Any] = {
        "source": "fresh_stage42_bv_source_acquisition_status",
        "stage": "Stage42-BV Source Acquisition Status / Blocker Matrix",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(path) for path in [BR_JSON, BT_JSON, BM_JSON, BK_JSON, BU_JSON, BJ_JSON, BN_JSON]]),
        "current_facts": CURRENT_FACTS,
        "official_references": OFFICIAL_REFERENCES,
        "artifact_inputs": {
            key: {k: v for k, v in artifact.items() if k != "payload"}
            for key, artifact in artifacts.items()
        },
        "blocker_matrix": blockers,
        "user_action_required": actions,
        "summary": summary,
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_bv_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    artifacts = payload["artifact_inputs"]
    blockers = payload["blocker_matrix"]
    claim = payload["claim_boundary"]
    gates = {
        "required_inputs_present": all(row["exists"] for row in artifacts.values()),
        "br_input_verified": artifacts["br"]["verdict"] == "stage42_br_calibrated_t50_source_support_gap_audit_pass",
        "bt_input_verified": artifacts["bt"]["verdict"] == "stage42_bt_eth_seq_t50_support_dry_run_pass_blocker_confirmed",
        "bm_input_verified": artifacts["bm"]["verdict"] == "stage42_bm_eth_person_terms_audit_pass_claim_blocked",
        "bu_input_verified": artifacts["bu"]["verdict"] == "stage42_bu_ucy_students_t50_source_support_pass_blocker_narrowed",
        "blocker_matrix_built": len(blockers) >= 5,
        "ucy_students_blocker_narrowed": summary["ucy_students_blocker_narrowed"] is True,
        "eth_seq_not_overclaimed": summary["eth_seq_blocker_resolved"] is False,
        "trajnet_not_overclaimed": summary["trajnet_raw_long_source_resolved"] is False,
        "user_actions_generated": len(payload["user_action_required"]) >= 4,
        "official_references_recorded": len(payload["official_references"]) >= 4,
        "no_auto_download": summary["auto_download_executed"] is False,
        "registry_only_not_counted_as_converted": summary["registry_only_counted_as_converted"] is False,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    verdict = "stage42_bv_source_acquisition_status_pass_blockers_actionable" if passed == total else "stage42_bv_source_acquisition_status_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    lines = [
        "# Stage42-BV Source Acquisition Status / Blocker Matrix",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bv_gate']['passed']} / {payload['stage42_bv_gate']['total']}`",
        f"- verdict: `{payload['stage42_bv_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- blockers_total: `{summary['blockers_total']}`",
        f"- blockers_active: `{summary['blockers_active']}`",
        f"- ucy_students_blocker_narrowed: `{summary['ucy_students_blocker_narrowed']}`",
        f"- eth_seq_blocker_resolved: `{summary['eth_seq_blocker_resolved']}`",
        f"- trajnet_raw_long_source_resolved: `{summary['trajnet_raw_long_source_resolved']}`",
        f"- global_t100_positive_claim_allowed: `{summary['global_t100_positive_claim_allowed']}`",
        f"- global_metric_claim_allowed: `{summary['global_metric_claim_allowed']}`",
        f"- auto_download_executed: `{summary['auto_download_executed']}`",
        "",
        "## Blocker Matrix",
        "",
        "| blocker | status | root cause | next action | allowed claim |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in payload["blocker_matrix"]:
        lines.append(
            f"| `{row['blocker_id']}` | `{row['status']}` | {row['root_cause']} | {row['next_action']} | `{row['claim_allowed']}` |"
        )
    lines.extend([
        "",
        "## Official / Source References",
        "",
        "| id | source confidence | url | M3W use |",
        "| --- | --- | --- | --- |",
    ])
    for ref in payload["official_references"]:
        lines.append(f"| `{ref['id']}` | `{ref['source_confidence']}` | {ref['url']} | {ref['use_for_m3w']} |")
    lines.extend([
        "",
        "## Next Best Actions",
        "",
        *[f"- {action}" for action in summary["next_best_actions"]],
        "",
        "## Interpretation",
        "",
        "- Stage42-BV does not repair any blocker by itself; it converts the remaining source-support problems into an executable acquisition/status matrix.",
        "- The strongest current deployable claim remains protected dataset-local/raw-frame 2.5D evidence, not true 3D, not foundation, not metric/seconds-level.",
        "- UCY_students is closer after BU because `students001` is now counted as a real t50-capable independent source, but one more independent students-family source is still required.",
        "- ETH_seq and ETH_UCY t100 remain blocked by source/terms support; TrajNet long-horizon repair remains blocked by local snippet length.",
    ])
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bv_gate"]
    lines = [
        "# Stage42-BV Gate",
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


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-BV User Action Required: External Source Support",
        "",
        "No automatic download was executed. Do not count these actions as converted/evaluated data until legal paths or terms are provided and the conversion/no-leakage/source-CV pipeline is rerun.",
        "",
        "| priority | blocker | action |",
        "| --- | --- | --- |",
    ]
    for row in payload["user_action_required"]:
        lines.append(f"| `{row['priority']}` | `{row['blocker_id']}` | {row['action']} |")
    return lines


def run_stage42_source_acquisition_status() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    return payload


if __name__ == "__main__":
    out = run_stage42_source_acquisition_status()
    print(json.dumps(out["summary"], indent=2, sort_keys=True))
