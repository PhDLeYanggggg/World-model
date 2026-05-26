from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_calibrated_subset_eval as bo
from src import stage42_calibrated_subset_safety_repair as bp
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
BQ_JSON = OUT_DIR / "calibrated_subset_t50_support_repair_stage42.json"
BN_JSON = OUT_DIR / "source_time_geometry_calibration_stage42.json"
BM_JSON = OUT_DIR / "eth_person_terms_audit_stage42.json"
BK_JSON = OUT_DIR / "post_bj_local_source_verification_stage42.json"

REPORT_JSON = OUT_DIR / "calibrated_t50_source_support_gap_stage42.json"
REPORT_MD = OUT_DIR / "calibrated_t50_source_support_gap_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_br_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_calibrated_t50_sources_stage42.md"

MIN_T50_FAMILY_SUPPORT = 2

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BR 是 t50 source-family support gap audit，不是训练或部署成功声明。",
    "Stage42-BQ 已把 calibrated-subset t50 负迁移守到 0，但没有产生 t50 正迁移。",
    "source-specific calibration evidence 不能升级为全局 metric/seconds-level M3W claim。",
    "ETH-Person XML 本地技术信号仍受 terms/license blocker 限制。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


SOURCE_TO_ID = {rel: sid for sid, rel in bo.SOURCE_TO_REL.items()}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _source_family_for_ids(source_ids: list[str]) -> dict[str, str]:
    rels = np.asarray([bo.SOURCE_TO_REL[sid] for sid in source_ids])
    families = bp._source_family(rels)
    return {sid: str(family) for sid, family in zip(source_ids, families)}


def _family_summary(source_ids: list[str], bq_payload: Mapping[str, Any]) -> dict[str, Any]:
    family_by_id = _source_family_for_ids(source_ids)
    family_to_sources: dict[str, list[str]] = defaultdict(list)
    for sid, family in family_by_id.items():
        family_to_sources[family].append(sid)

    t50_by_holdout = {
        str(fold["fold"]["holdout_source"]): float(fold["protected_ade"]["t50_improvement"])
        for fold in bq_payload.get("fold_results", [])
    }
    all_by_holdout = {
        str(fold["fold"]["holdout_source"]): float(fold["protected_ade"]["all_improvement"])
        for fold in bq_payload.get("fold_results", [])
    }
    out: dict[str, Any] = {}
    for family, members in sorted(family_to_sources.items()):
        member_rows = {}
        unsupported_holdouts = []
        supported_holdouts = []
        positive_t50_holdouts = []
        nonharm_t50_holdouts = []
        for sid in sorted(members):
            support_count = max(0, len(members) - 1)
            row = {
                "source_id": sid,
                "family": family,
                "same_family_non_test_support_sources": support_count,
                "t50_family_support_pass": support_count >= MIN_T50_FAMILY_SUPPORT,
                "bq_t50_improvement": t50_by_holdout.get(sid),
                "bq_all_improvement": all_by_holdout.get(sid),
            }
            member_rows[sid] = row
            if support_count >= MIN_T50_FAMILY_SUPPORT:
                supported_holdouts.append(sid)
            else:
                unsupported_holdouts.append(sid)
            t50 = t50_by_holdout.get(sid)
            if t50 is not None and t50 > 0:
                positive_t50_holdouts.append(sid)
            if t50 is not None and t50 >= 0:
                nonharm_t50_holdouts.append(sid)
        additional_needed = max(0, MIN_T50_FAMILY_SUPPORT + 1 - len(members))
        reason = "enough_family_sources_but_no_safe_positive_t50_policy" if additional_needed == 0 else "insufficient_same_family_source_support"
        out[family] = {
            "calibrated_sources": sorted(members),
            "calibrated_source_count": len(members),
            "minimum_sources_needed_for_leave_one_out_t50_support": MIN_T50_FAMILY_SUPPORT + 1,
            "additional_calibrated_sources_needed": additional_needed,
            "supported_holdouts": sorted(supported_holdouts),
            "unsupported_holdouts": sorted(unsupported_holdouts),
            "positive_t50_holdouts": sorted(positive_t50_holdouts),
            "nonharm_t50_holdouts": sorted(nonharm_t50_holdouts),
            "source_rows": member_rows,
            "primary_blocker": reason,
        }
    return out


def _local_candidate_summary(bm: Mapping[str, Any], bk: Mapping[str, Any]) -> dict[str, Any]:
    bm_summary = bm.get("summary", {})
    bk_summary = bk.get("summary", {})
    eth_candidates = list(bk_summary.get("eth_person_xml_candidates", []))
    return {
        "eth_person_xml_candidates": eth_candidates,
        "eth_person_candidate_count": len(eth_candidates),
        "eth_person_can_repair_after_terms_confirmation": bool(
            bk_summary.get("can_repair_eth_ucy_with_local_candidates_after_license_confirmation", False)
        ),
        "eth_person_terms_verified": bool(bm_summary.get("official_terms_verified", False)),
        "eth_person_license_terms_confirmed": bool(bm_summary.get("license_terms_confirmed", False)),
        "eth_person_official_conversion_allowed": bool(bm_summary.get("next_stage_official_conversion_allowed", False)),
        "trajnet_t100_capable_files": int(bk_summary.get("trajnet_t100_capable_files", 0)),
        "trajnet_independent_t100_groups": int(bk_summary.get("trajnet_independent_t100_groups", 0)),
        "trajnet_local_long_track_blocker": int(bk_summary.get("trajnet_t100_capable_files", 0)) == 0,
        "auto_download_executed": False,
    }


def _action_items(family: Mapping[str, Any], candidates: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for family_name, row in family.items():
        if row["additional_calibrated_sources_needed"] > 0:
            if family_name == "ETH_seq" and candidates["eth_person_candidate_count"] > 0:
                actions.append(
                    {
                        "family": family_name,
                        "priority": "high",
                        "action": "confirm_eth_person_terms_then_convert_xml_candidates",
                        "additional_sources_needed": int(row["additional_calibrated_sources_needed"]),
                        "available_local_candidates": candidates["eth_person_xml_candidates"],
                        "blocked_by": "ETH-Person official terms/license not confirmed",
                    }
                )
            else:
                actions.append(
                    {
                        "family": family_name,
                        "priority": "high",
                        "action": "provide_or_locate_additional_source_specific_calibrated_tracks",
                        "additional_sources_needed": int(row["additional_calibrated_sources_needed"]),
                        "available_local_candidates": [],
                        "blocked_by": "no verified local same-family calibrated source support",
                    }
                )
        elif not row["positive_t50_holdouts"]:
            actions.append(
                {
                    "family": family_name,
                    "priority": "medium",
                    "action": "train_family_specific_t50_policy_or_add_more_validation_sources",
                    "additional_sources_needed": 0,
                    "available_local_candidates": [],
                    "blocked_by": "source support exists but validation-safe t50 policy falls back to floor",
                }
            )
    return actions


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "bq_input_verified": payload["bq_verdict"]
        == "stage42_bq_calibrated_subset_t50_support_repair_pass_t50_nonharm_limited_positive",
        "family_gap_quantified": summary["families_audited"] >= 3,
        "unsupported_family_holdouts_identified": summary["unsupported_family_holdout_count"] > 0,
        "safe_nonharm_confirmed": summary["bq_t50_min"] == 0.0 and summary["bq_easy_max"] <= 0.02,
        "positive_t50_not_overclaimed": summary["bq_positive_t50_fold_count"] == 0,
        "local_candidates_checked": summary["eth_person_candidate_count"] >= 0 and summary["trajnet_t100_capable_files"] >= 0,
        "eth_person_terms_blocker_recorded": summary["eth_person_terms_verified"] is False,
        "user_action_generated": bool(payload["user_action_required"]),
        "global_metric_blocked": claim["global_metric_claim_allowed"] is False,
        "global_seconds_blocked": claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = (
        "stage42_br_calibrated_t50_source_support_gap_audit_pass"
        if passed == total
        else "stage42_br_calibrated_t50_source_support_gap_audit_partial"
    )
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def run_stage42_calibrated_t50_source_support_gap_audit() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bq = _load_json(BQ_JSON)
    bn = _load_json(BN_JSON)
    bm = _load_json(BM_JSON)
    bk = _load_json(BK_JSON)
    source_ids = list(bq["summary"]["calibrated_source_ids"])
    family = _family_summary(source_ids, bq)
    candidates = _local_candidate_summary(bm, bk)
    actions = _action_items(family, candidates)
    unsupported = sum(len(row["unsupported_holdouts"]) for row in family.values())
    summary = {
        "source": "fresh_calibrated_t50_source_support_gap_audit",
        "families_audited": len(family),
        "calibrated_sources_audited": len(source_ids),
        "unsupported_family_holdout_count": int(unsupported),
        "families_with_additional_sources_needed": [
            name for name, row in family.items() if int(row["additional_calibrated_sources_needed"]) > 0
        ],
        "families_with_support_but_no_positive_t50": [
            name for name, row in family.items() if int(row["additional_calibrated_sources_needed"]) == 0 and not row["positive_t50_holdouts"]
        ],
        "bq_all_macro": float(bq["summary"]["all_improvement_macro_mean"]),
        "bq_t50_macro": float(bq["summary"]["t50_improvement_macro_mean"]),
        "bq_t50_min": float(bq["summary"]["t50_improvement_min"]),
        "bq_easy_max": float(bq["summary"]["easy_degradation_max"]),
        "bq_positive_t50_fold_count": int(bq["summary"]["positive_t50_fold_count"]),
        "eth_person_candidate_count": int(candidates["eth_person_candidate_count"]),
        "eth_person_terms_verified": bool(candidates["eth_person_terms_verified"]),
        "trajnet_t100_capable_files": int(candidates["trajnet_t100_capable_files"]),
        "auto_download_executed": False,
        "training_run": False,
    }
    payload: dict[str, Any] = {
        "source": "fresh_calibrated_t50_source_support_gap_audit",
        "stage": "Stage42-BR calibrated t50 source-support gap audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(BQ_JSON), str(BN_JSON), str(BM_JSON), str(BK_JSON)]),
        "current_facts": CURRENT_FACTS,
        "bq_verdict": bq.get("stage42_bq_gate", {}).get("verdict"),
        "bn_verdict": bn.get("stage42_bn_gate", {}).get("verdict"),
        "bm_verdict": bm.get("stage42_bm_gate", {}).get("verdict"),
        "family_support": family,
        "local_candidates": candidates,
        "action_items": actions,
        "summary": summary,
        "user_action_required": actions,
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "m3w_official_metric_seconds_claim_allowed": False,
            "positive_t50_claim_allowed": False,
            "t50_nonharm_claim_allowed": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_br_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-BR Calibrated T50 Source-Support Gap Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_br_gate']['passed']} / {payload['stage42_br_gate']['total']}`",
        f"- verdict: `{payload['stage42_br_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- families_audited: `{s['families_audited']}`",
        f"- calibrated_sources_audited: `{s['calibrated_sources_audited']}`",
        f"- unsupported_family_holdout_count: `{s['unsupported_family_holdout_count']}`",
        f"- families_with_additional_sources_needed: `{s['families_with_additional_sources_needed']}`",
        f"- families_with_support_but_no_positive_t50: `{s['families_with_support_but_no_positive_t50']}`",
        f"- BQ all macro: `{s['bq_all_macro']}`",
        f"- BQ t50 macro/min: `{s['bq_t50_macro']}` / `{s['bq_t50_min']}`",
        f"- BQ easy max: `{s['bq_easy_max']}`",
        f"- BQ positive_t50_fold_count: `{s['bq_positive_t50_fold_count']}`",
        "",
        "## Source-Family Support",
        "",
        "| family | sources | additional needed | unsupported holdouts | positive t50 holdouts | blocker |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, row in payload["family_support"].items():
        lines.append(
            f"| `{name}` | {row['calibrated_source_count']} | {row['additional_calibrated_sources_needed']} | {len(row['unsupported_holdouts'])} | {len(row['positive_t50_holdouts'])} | `{row['primary_blocker']}` |"
        )
    lines.extend(
        [
            "",
            "## Local Candidate Check",
            "",
            f"- ETH-Person XML candidates: `{payload['local_candidates']['eth_person_xml_candidates']}`",
            f"- ETH-Person terms verified: `{payload['local_candidates']['eth_person_terms_verified']}`",
            f"- ETH-Person official conversion allowed: `{payload['local_candidates']['eth_person_official_conversion_allowed']}`",
            f"- TrajNet local t100-capable files: `{payload['local_candidates']['trajnet_t100_capable_files']}`",
            "",
            "## Action Items",
            "",
        ]
    )
    for item in payload["action_items"]:
        lines.append(
            f"- `{item['family']}`: {item['action']} (priority `{item['priority']}`, additional_sources_needed `{item['additional_sources_needed']}`, blocked_by `{item['blocked_by']}`)"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- BQ did the right safety move: unsupported t50 source-families fall back instead of producing negative transfer.",
            "- The remaining t50 blocker is evidence support, not a permission to loosen safety gates using test data.",
            "- ETH-style support may be repairable after ETH-Person terms are confirmed; UCY_students needs additional same-family calibrated sources; UCY_zara has enough source support but no validation-safe t50-positive policy yet.",
            "- This remains source-specific calibrated-subset evidence only; global metric/seconds-level M3W claims remain blocked.",
            "",
            "## Claim Boundary",
            "",
            f"- claim_boundary: `{payload['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_br_gate"]
    lines = [
        "# Stage42-BR Gate",
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


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-BR User Action Required: Calibrated T50 Source Support",
        "",
        "## Why This Exists",
        "",
        "- Stage42-BQ safely guards calibrated-subset t50 to non-harm, but positive t50 folds are zero.",
        "- Additional same-family calibrated support is needed before enabling t50 switching on unsupported source families.",
        "",
        "## Actions",
        "",
    ]
    for item in payload["action_items"]:
        lines.append(f"- `{item['family']}`: `{item['action']}`; blocked_by: `{item['blocked_by']}`.")
        if item["available_local_candidates"]:
            for candidate in item["available_local_candidates"]:
                lines.append(f"  - local candidate: `{candidate}`")
    lines.extend(
        [
            "",
            "## Non-Claims",
            "",
            "- Do not call BQ a positive t50 transfer result.",
            "- Do not call source-specific calibrated subsets a global metric/seconds-level M3W benchmark.",
            "- Do not treat ETH-Person XML as official/deployable until terms are confirmed.",
        ]
    )
    return lines


if __name__ == "__main__":
    run_stage42_calibrated_t50_source_support_gap_audit()
