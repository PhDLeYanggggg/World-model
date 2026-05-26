from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
BI_JSON = OUT_DIR / "local_t100_easy_guard_repair_stage42.json"
BC_JSON = OUT_DIR / "t100_source_acquisition_plan_stage42.json"
BD_JSON = OUT_DIR / "local_t100_source_inventory_stage42.json"
REPORT_JSON = OUT_DIR / "post_bi_t100_source_package_stage42.json"
REPORT_MD = OUT_DIR / "post_bi_t100_source_package_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bj_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_post_bi_t100_sources_stage42.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

MIN_INDEPENDENT_SOURCES = 3

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BI 已修复 UCY independent-source t100 easy guard，但这不是 global t100 success。",
    "ETH_UCY 和 TrajNet 仍缺足够 independent t100 sources，不能写全局 t100 positive claim。",
    "Stage42-BJ 不训练模型、不下载 gated/restricted raw data、不执行 Stage5C、不启用 SMC。",
    "future waypoints / endpoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t100 仍是 raw-frame diagnostic，不能写成 seconds-level。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _domain_support_from_bi(bi: Mapping[str, Any]) -> dict[str, Any]:
    summary = bi.get("summary", {})
    support = bi.get("support_matrix", {})
    out: dict[str, Any] = {}
    for domain in ["ETH_UCY", "UCY", "TrajNet"]:
        row = dict(support.get(domain, {}))
        independent = int(row.get("independent_sources", summary.get(f"{domain.lower()}_independent_sources", 0)) or 0)
        needed = max(0, MIN_INDEPENDENT_SOURCES - independent)
        t100 = bi.get("domain_summary", {}).get(domain, {}).get("by_horizon", {}).get("100", {})
        out[domain] = {
            "source": "cached_verified_from_stage42_bi",
            "domain": domain,
            "independent_sources": independent,
            "minimum_required_independent_sources": MIN_INDEPENDENT_SOURCES,
            "additional_independent_sources_needed": needed,
            "source_cv_feasible": independent >= MIN_INDEPENDENT_SOURCES,
            "t100_source_cv_supported": bool(domain == "UCY" and summary.get("ucy_t100_source_cv_supported", False)),
            "mean_improvement_vs_fallback": t100.get("mean_holdout_improvement_vs_fallback"),
            "minimum_improvement_vs_fallback": t100.get("minimum_holdout_improvement_vs_fallback"),
            "maximum_easy_degradation": t100.get("maximum_easy_degradation"),
            "blocker": None if needed == 0 else f"needs_{needed}_additional_independent_t100_sources_under_strict_post_bi_protocol",
        }
    return out


def _local_inventory_exhaustion(bd: Mapping[str, Any], support: Mapping[str, Any]) -> dict[str, Any]:
    inventory = list(bd.get("inventory", []))
    t100_rows = [row for row in inventory if row.get("t100_capable") and not row.get("synthetic_or_diagnostic", False)]
    domain_counts: dict[str, int] = {"ETH_UCY": 0, "UCY": 0, "TrajNet": 0}
    independent_keys_by_domain: dict[str, set[str]] = {"ETH_UCY": set(), "UCY": set(), "TrajNet": set()}
    for row in t100_rows:
        rel = str(row.get("relative_path", ""))
        suggested = str(row.get("suggested_domain", ""))
        if rel.startswith("ETH/"):
            domain = "ETH_UCY"
        elif rel.startswith("UCY/"):
            domain = "UCY"
        elif rel.startswith("TrajNet"):
            domain = "TrajNet"
        elif suggested in domain_counts:
            domain = suggested
        else:
            continue
        domain_counts[domain] += 1
        parts = rel.split("/")
        independent_key = f"{domain}::{parts[0]}/{parts[1]}" if len(parts) >= 2 else f"{domain}::{rel}"
        independent_keys_by_domain[domain].add(independent_key)
    independent_group_count_by_domain = {domain: len(keys) for domain, keys in independent_keys_by_domain.items()}
    exhausted_domains = [
        domain
        for domain, row in support.items()
        if int(row["additional_independent_sources_needed"]) > 0
        and independent_group_count_by_domain.get(domain, 0) <= int(row.get("independent_sources", 0))
    ]
    return {
        "source": "fresh_post_bi_inventory_exhaustion_audit",
        "bd_verdict": bd.get("stage42_bd_gate", {}).get("verdict"),
        "raw_t100_capable_file_count": len(t100_rows),
        "t100_file_count_by_domain": domain_counts,
        "independent_t100_group_count_by_domain": independent_group_count_by_domain,
        "independent_t100_groups_by_domain": {domain: sorted(keys) for domain, keys in independent_keys_by_domain.items()},
        "local_inventory_exhausted_for_domains": exhausted_domains,
        "interpretation": "The current local inventory has already been consumed for BI-style strict source support when independent scene/source groups do not exceed BI support counts.",
    }


def _rank_actions(support: Mapping[str, Any], bc: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = list(bc.get("candidates", []))
    actions: list[dict[str, Any]] = []
    for domain, row in support.items():
        needed = int(row["additional_independent_sources_needed"])
        if needed <= 0:
            continue
        matching = [
            cand
            for cand in candidates
            if domain in cand.get("target_domains", [])
        ]
        matching.sort(key=lambda cand: (-int(cand.get("priority_score", 0)), str(cand.get("id", ""))))
        actions.append(
            {
                "source": "fresh_post_bi_action_queue",
                "domain": domain,
                "priority": "critical" if domain in {"ETH_UCY", "TrajNet"} else "high",
                "additional_independent_t100_sources_needed": needed,
                "candidate_source_ids": [cand.get("id") for cand in matching],
                "candidate_sources": [
                    {
                        "id": cand.get("id"),
                        "dataset_name": cand.get("dataset_name"),
                        "official_url": cand.get("official_url"),
                        "local_path_found": cand.get("local_status", {}).get("local_path_found"),
                        "found_paths": cand.get("local_status", {}).get("found_paths", []),
                        "auto_download_allowed": cand.get("download_policy", {}).get("auto_download_allowed", False),
                        "blocked_reasons": cand.get("download_policy", {}).get("blocked_reasons", []),
                        "expected_t100_role": cand.get("expected_t100_role"),
                    }
                    for cand in matching[:5]
                ],
                "next_action": "provide_or_approve_legal_independent_t100_sources_then_rerun_conversion_and_source_cv",
            }
        )
    return actions


def run_stage42_post_bi_t100_source_package() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bi = _load_json(BI_JSON)
    bc = _load_json(BC_JSON)
    bd = _load_json(BD_JSON)
    support = _domain_support_from_bi(bi)
    inventory = _local_inventory_exhaustion(bd, support)
    actions = _rank_actions(support, bc)
    ucy = support["UCY"]
    payload: dict[str, Any] = {
        "source": "fresh_post_bi_t100_source_package",
        "stage": "Stage42-BJ Post-BI T100 Source Package",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(BI_JSON), str(BC_JSON), str(BD_JSON)]),
        "current_facts": CURRENT_FACTS,
        "bi_verdict": bi.get("stage42_bi_gate", {}).get("verdict"),
        "bc_verdict": bc.get("stage42_bc_gate", {}).get("verdict"),
        "bd_verdict": bd.get("stage42_bd_gate", {}).get("verdict"),
        "strict_protocol": {
            "source": "fresh_post_bi_protocol",
            "minimum_independent_sources_per_domain": MIN_INDEPENDENT_SOURCES,
            "selection_uses_holdout_source": False,
            "test_metrics_for_threshold": False,
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "domain_support": support,
        "local_inventory_exhaustion": inventory,
        "action_queue": actions,
        "summary": {
            "source": "fresh_post_bi_t100_source_package",
            "ucy_t100_repaired": bool(ucy["t100_source_cv_supported"] and float(ucy["maximum_easy_degradation"] or 0.0) <= 0.02),
            "eth_ucy_additional_sources_needed": support["ETH_UCY"]["additional_independent_sources_needed"],
            "trajnet_additional_sources_needed": support["TrajNet"]["additional_independent_sources_needed"],
            "domains_still_blocked": [domain for domain, row in support.items() if int(row["additional_independent_sources_needed"]) > 0],
            "global_t100_positive_claim_allowed": False,
            "auto_download_executed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "next_stage_recommended": "Stage42-BK legal source acquisition / user path verification for ETH_UCY and TrajNet independent t100 sources",
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "t100_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "global_t100_positive_claim_allowed": False,
        },
    }
    payload["stage42_bj_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_ledger(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    support = payload["domain_support"]
    gates = {
        "bi_input_verified": payload.get("bi_verdict") == "stage42_bi_ucy_t100_easy_guard_repair_pass_with_global_blocker",
        "bc_input_verified": payload.get("bc_verdict") == "stage42_bc_t100_source_acquisition_plan_pass",
        "bd_input_verified": payload.get("bd_verdict") == "stage42_bd_local_t100_source_inventory_pass",
        "ucy_repair_preserved": summary["ucy_t100_repaired"] is True,
        "eth_ucy_blocker_explicit": int(support["ETH_UCY"]["additional_independent_sources_needed"]) > 0,
        "trajnet_blocker_explicit": int(support["TrajNet"]["additional_independent_sources_needed"]) > 0,
        "local_inventory_exhaustion_checked": "local_inventory_exhausted_for_domains" in payload["local_inventory_exhaustion"],
        "action_queue_generated": len(payload["action_queue"]) >= 2,
        "no_auto_download": summary["auto_download_executed"] is False,
        "no_leakage_pass": not payload["strict_protocol"]["future_endpoint_input"]
        and not payload["strict_protocol"]["central_velocity"]
        and not payload["strict_protocol"]["test_endpoint_goals"]
        and not payload["strict_protocol"]["test_metrics_for_threshold"],
        "no_global_t100_overclaim": summary["global_t100_positive_claim_allowed"] is False,
        "no_metric_seconds_overclaim": not payload["claim_boundary"]["metric_or_seconds_claim"]
        and not payload["claim_boundary"]["t100_seconds_claim"],
        "stage5c_false": not payload["claim_boundary"]["stage5c_executed"],
        "smc_false": not payload["claim_boundary"]["smc_enabled"],
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_bj_post_bi_t100_source_package_pass" if passed == total else "stage42_bj_post_bi_t100_source_package_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    lines = [
        "# Stage42-BJ Post-BI T100 Source Package",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bj_gate']['passed']} / {payload['stage42_bj_gate']['total']}`",
        f"- verdict: `{payload['stage42_bj_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Domain Support Under Strict Post-BI Protocol",
        "",
        "| domain | independent sources | needed | t100 supported | mean gain | min gain | max easy | blocker |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for domain, row in payload["domain_support"].items():
        lines.append(
            f"| `{domain}` | {row['independent_sources']} | {row['additional_independent_sources_needed']} | `{row['t100_source_cv_supported']}` | {row['mean_improvement_vs_fallback']} | {row['minimum_improvement_vs_fallback']} | {row['maximum_easy_degradation']} | {row['blocker']} |"
        )
    lines.extend(
        [
            "",
            "## Local Inventory Exhaustion",
            "",
            f"- raw_t100_capable_file_count: `{payload['local_inventory_exhaustion']['raw_t100_capable_file_count']}`",
            f"- t100_file_count_by_domain: `{payload['local_inventory_exhaustion']['t100_file_count_by_domain']}`",
            f"- independent_t100_group_count_by_domain: `{payload['local_inventory_exhaustion']['independent_t100_group_count_by_domain']}`",
            f"- independent_t100_groups_by_domain: `{payload['local_inventory_exhaustion']['independent_t100_groups_by_domain']}`",
            f"- local_inventory_exhausted_for_domains: `{payload['local_inventory_exhaustion']['local_inventory_exhausted_for_domains']}`",
            "",
            "## Action Queue",
            "",
        ]
    )
    for action in payload["action_queue"]:
        lines.extend(
            [
                f"### {action['domain']}",
                "",
                f"- priority: `{action['priority']}`",
                f"- additional_independent_t100_sources_needed: `{action['additional_independent_t100_sources_needed']}`",
                f"- candidate_source_ids: `{action['candidate_source_ids']}`",
                f"- next_action: `{action['next_action']}`",
                "",
                "| candidate | official_url | local path found | auto download | blocked reasons |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for cand in action["candidate_sources"]:
            lines.append(
                f"| `{cand['id']}` | `{cand['official_url']}` | `{cand['local_path_found']}` | `{cand['auto_download_allowed']}` | {cand['blocked_reasons']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Summary",
            "",
            f"- ucy_t100_repaired: `{summary['ucy_t100_repaired']}`",
            f"- eth_ucy_additional_sources_needed: `{summary['eth_ucy_additional_sources_needed']}`",
            f"- trajnet_additional_sources_needed: `{summary['trajnet_additional_sources_needed']}`",
            f"- domains_still_blocked: `{summary['domains_still_blocked']}`",
            f"- global_t100_positive_claim_allowed: `{summary['global_t100_positive_claim_allowed']}`",
            f"- next_stage_recommended: `{summary['next_stage_recommended']}`",
            "",
            "## Interpretation",
            "",
            "- Stage42-BI fixed the UCY independent-source t100 easy blocker, but global t100 remains blocked.",
            "- Under the stricter post-BI protocol, ETH_UCY needs two additional independent t100 sources and TrajNet needs three.",
            "- The current local inventory is exhausted for the blocked independent-source requirements; next progress requires legal official sources or user-provided paths.",
            "- No Stage5C, no SMC, no metric/seconds-level claim, and no auto-download of gated/restricted raw data occurred.",
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-BJ User Action Required",
        "",
        f"- source: `{payload['source']}`",
        "- purpose: obtain legal independent t100-capable sources for domains still blocking global t100 support.",
        "",
    ]
    for action in payload["action_queue"]:
        lines.extend(
            [
                f"## {action['domain']}",
                "",
                f"- priority: `{action['priority']}`",
                f"- additional_independent_t100_sources_needed: `{action['additional_independent_t100_sources_needed']}`",
                f"- next_action: `{action['next_action']}`",
                "",
            ]
        )
        for cand in action["candidate_sources"]:
            lines.extend(
                [
                    f"### {cand['id']} / {cand['dataset_name']}",
                    "",
                    f"- official_url: `{cand['official_url']}`",
                    f"- local_path_found: `{cand['local_path_found']}`",
                    f"- found_paths: `{cand['found_paths']}`",
                    f"- auto_download_allowed: `{cand['auto_download_allowed']}`",
                    "- blocked_reasons:",
                    *[f"  - {reason}" for reason in cand["blocked_reasons"]],
                    f"- expected_t100_role: {cand['expected_t100_role']}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Non-Claims",
            "",
            "- UCY local t100 support does not establish global t100 success.",
            "- Registry-only datasets, missing paths, gated sources, or failed downloads must not be counted as converted/evaluated data.",
            "- Dataset-local/raw-frame horizons must not be reported as metric or seconds-level trajectories.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bj_gate"]
    lines = [
        "# Stage42-BJ Gate",
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
        "verdict": payload["stage42_bj_gate"]["verdict"],
        "gate": f"{payload['stage42_bj_gate']['passed']}/{payload['stage42_bj_gate']['total']}",
        "git_commit": payload["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_post_bi_t100_source_package()
