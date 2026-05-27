from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

SOURCE_TERMS_GAP_JSON = OUT_DIR / "source_terms_gap_audit_stage42.json"
SOURCE_CLOSURE_JSON = OUT_DIR / "source_support_closure_audit_stage42.json"
H100_QUEUE_JSON = OUT_DIR / "h100_source_support_repair_queue_stage42.json"
UNIFIED_QUEUE_JSON = OUT_DIR / "unified_guarded_conversion_queue_stage42.json"
OFFICIAL_LINKS_JSON = OUT_DIR / "official_source_link_audit_stage42.json"

REPORT_JSON = OUT_DIR / "source_action_consolidator_stage42.json"
REPORT_MD = OUT_DIR / "source_action_consolidator_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fw_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_action_consolidator_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
GOAL_LEDGER = Path("README_M3W_GOAL_EVIDENCE_LEDGER_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_source_action_consolidator_from_existing_blockers"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-FW consolidates existing source/legal/horizon blockers; it does not download, convert, train, or evaluate.",
    "Local paths, parseable files, and technical dry-runs are not legal conversion readiness.",
    "Every result remains labeled as fresh_run, cached_verified, or not_run.",
    "future endpoints / waypoints can only be supervised/evaluation labels, never inference inputs.",
    "No central velocity, no test endpoint goals, and no test metric threshold tuning are used.",
    "t+50 / t+100 remain raw-frame horizons, not seconds-level claims.",
    "dataset-local/raw-frame results are not global metric claims.",
    "Stage5C latent generative is not executed.",
    "SMC is not enabled.",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "download_executed": False,
    "conversion_executed": False,
    "evaluation_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _load_inputs() -> dict[str, Any]:
    return {
        "source_terms_gap": read_json(SOURCE_TERMS_GAP_JSON, {}),
        "source_closure": read_json(SOURCE_CLOSURE_JSON, {}),
        "h100_queue": read_json(H100_QUEUE_JSON, {}),
        "unified_queue": read_json(UNIFIED_QUEUE_JSON, {}),
        "official_links": read_json(OFFICIAL_LINKS_JSON, {}),
    }


def _official_url_map(official_links: Mapping[str, Any]) -> dict[str, str]:
    rows = official_links.get("official_source_rows", [])
    out: dict[str, str] = {}
    for row in rows:
        key = str(row.get("dataset_id") or row.get("dataset") or row.get("target") or "")
        url = str(row.get("official_url") or row.get("url") or "")
        if key and url:
            out[key] = url
    return out


def _terms_actions(inputs: Mapping[str, Any], official_urls: Mapping[str, str]) -> list[dict[str, Any]]:
    rows = inputs.get("source_terms_gap", {}).get("gap_rows", [])
    actions: list[dict[str, Any]] = []
    for row in rows:
        dataset_id = str(row.get("dataset_id", "unknown"))
        domain = str(row.get("domain", "unknown"))
        missing = list(row.get("missing_confirmation_fields", []))
        t50 = int(row.get("estimated_t50_windows_after_terms", 0) or 0)
        t100 = int(row.get("estimated_t100_windows_after_terms", 0) or 0)
        score = _score_terms_action(dataset_id, domain, t50, t100)
        actions.append(
            {
                "action_id": f"FW-TERMS-{dataset_id}",
                "target": dataset_id,
                "domain": domain,
                "category": "legal_terms_and_local_path",
                "result_source": "cached_verified",
                "status": "not_run_user_action_required",
                "priority": score,
                "official_url": official_urls.get(dataset_id, str(row.get("official_url", ""))),
                "blocking_claims": [
                    "official converted/evaluated external source",
                    "source-specific metric/time subset if calibration later passes",
                    "global t100 deployable claim",
                ],
                "missing": missing,
                "estimated_t50_rows_after_unblock": t50,
                "estimated_t100_rows_after_unblock": t100,
                "evidence_files": [str(SOURCE_TERMS_GAP_JSON), str(OFFICIAL_LINKS_JSON)],
                "next_user_action": _terms_next_action(dataset_id, missing),
                "next_commands_after_user_confirmation": [
                    ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
                    ".venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py",
                    ".venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py",
                ],
                "claim_guard": "Do not count this as converted/evaluated until validator, guarded conversion, no-leakage, and source-CV all pass.",
            }
        )
    return actions


def _score_terms_action(dataset_id: str, domain: str, t50: int, t100: int) -> int:
    score = 50 + min(30, (t50 + t100) // 400)
    if dataset_id == "ucy_crowd_original":
        score += 25
    if domain in {"UCY", "ETH_UCY"}:
        score += 8
    if "trajnet" in dataset_id.lower():
        score += 10
    return int(score)


def _terms_next_action(dataset_id: str, missing: list[str]) -> str:
    fields = ", ".join(missing) if missing else "terms/local path/source identity"
    return f"Confirm official terms and fill `{dataset_id}` fields in source_terms_confirmation_template_stage42.json: {fields}."


def _h100_actions(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    h100 = inputs.get("h100_queue", {})
    key_rows = h100.get("key_rows", {})
    actions: list[dict[str, Any]] = []
    for weak_key, row in key_rows.items():
        status = row.get("repair_status", {})
        candidates = list(row.get("top_candidates", []))
        target_bucket = row.get("target_bucket", "")
        top_candidates = [c.get("relative_path") for c in candidates[:6]]
        if str(status.get("status")) == "hard_blocker_no_local_trajnet_h100_long_source":
            priority = 98
        elif candidates:
            priority = 94
        else:
            priority = 86
        actions.append(
            {
                "action_id": f"FW-H100-{weak_key}",
                "target": weak_key,
                "domain": str(weak_key).split("|", 1)[0],
                "category": "h100_weak_horizon_source_support",
                "result_source": "cached_verified",
                "status": "not_run_user_action_required",
                "priority": priority,
                "official_url": "",
                "blocking_claims": [
                    "uniform horizon robustness",
                    "global t100 deployable claim",
                    "h100 raw-frame positive-safe claim for this source/horizon",
                ],
                "missing": _h100_missing(status, candidates),
                "target_bucket": target_bucket,
                "candidate_count": len(candidates),
                "top_candidate_paths": top_candidates,
                "evidence_files": [str(H100_QUEUE_JSON)],
                "next_user_action": _h100_next_action(str(weak_key), status, candidates),
                "next_commands_after_user_confirmation": [
                    ".venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py",
                    ".venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py",
                    ".venv-pytorch/bin/python run_stage42_h100_weak_horizon_source_support_audit.py",
                ],
                "claim_guard": "Candidate paths are inventory only; do not convert/evaluate or claim repair until legal terms, conversion, no-leakage, and source-CV pass.",
            }
        )
    return actions


def _h100_missing(status: Mapping[str, Any], candidates: list[Mapping[str, Any]]) -> list[str]:
    if str(status.get("status")) == "hard_blocker_no_local_trajnet_h100_long_source":
        return ["official longer TrajNet-compatible raw source", "timing/geometry evidence", "terms confirmation", "local path"]
    if candidates:
        return ["terms/license confirmation", "guarded conversion", "no-leakage audit", "train-only source-CV"]
    return ["source support", "terms confirmation", "local path"]


def _h100_next_action(weak_key: str, status: Mapping[str, Any], candidates: list[Mapping[str, Any]]) -> str:
    if str(status.get("status")) == "hard_blocker_no_local_trajnet_h100_long_source":
        return f"Provide or legally confirm a longer official raw source for `{weak_key}`; current local TrajNet snippets cannot support h100."
    if candidates:
        return f"Confirm terms/license for the listed local candidates for `{weak_key}`, then run guarded conversion and source-CV."
    return f"Find a legal source-support candidate for `{weak_key}` and confirm terms before conversion."


def _domain_actions(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = inputs.get("source_closure", {}).get("domain_status", [])
    actions: list[dict[str, Any]] = []
    for row in rows:
        domain = str(row.get("domain", "unknown"))
        blockers = list(row.get("blockers", []))
        if not blockers:
            continue
        actions.append(
            {
                "action_id": f"FW-DOMAIN-{domain}",
                "target": domain,
                "domain": domain,
                "category": "domain_closure",
                "result_source": "cached_verified",
                "status": "not_run_open_blocker",
                "priority": _score_domain_action(domain, blockers),
                "official_url": "",
                "blocking_claims": [
                    "domain closed for source/time support",
                    "metric/seconds subset claim if applicable",
                    "global source-level robustness claim",
                ],
                "missing": blockers,
                "partial_support": row.get("partial_support", {}),
                "evidence_files": [str(SOURCE_CLOSURE_JSON)],
                "next_user_action": str(row.get("next_action", "Close source support blockers and rerun closure audit.")),
                "next_commands_after_user_confirmation": [
                    ".venv-pytorch/bin/python run_stage42_source_support_closure_audit.py",
                ],
                "claim_guard": "Domain remains not_closed until this action passes with no leakage and no terms blocker.",
            }
        )
    return actions


def _score_domain_action(domain: str, blockers: list[str]) -> int:
    score = 70 + 3 * len(blockers)
    if domain == "UCY":
        score += 15
    if domain == "TrajNet":
        score += 12
    if domain == "ETH_UCY":
        score += 8
    return score


def _dedupe_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for action in sorted(actions, key=lambda row: (-int(row["priority"]), row["action_id"])):
        key = str(action["action_id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(action)
    return out


def _summary(actions: list[Mapping[str, Any]], inputs: Mapping[str, Any]) -> dict[str, Any]:
    unified = inputs.get("unified_queue", {})
    unified_summary = unified.get("summary", {})
    categories: dict[str, int] = {}
    for row in actions:
        categories[str(row["category"])] = categories.get(str(row["category"]), 0) + 1
    return {
        "source": SOURCE,
        "actions_total": len(actions),
        "categories": categories,
        "top_actions": [row["action_id"] for row in actions[:5]],
        "conversion_ready_now": int(unified_summary.get("unified_queue_count", 0) or 0),
        "blocked_action_count": int(len(unified.get("blocked_actions", []))),
        "downloads_or_conversions_executed": 0,
        "evaluations_executed": 0,
        "claim_ready_after_this_stage": False,
        "highest_priority_blocker": actions[0]["action_id"] if actions else "none",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    actions = payload["consolidated_actions"]
    by_id = {row["action_id"]: row for row in actions}
    gates = {
        "input_terms_gap_loaded": payload["input_status"]["source_terms_gap"]["exists"],
        "input_source_closure_loaded": payload["input_status"]["source_closure"]["exists"],
        "input_h100_queue_loaded": payload["input_status"]["h100_queue"]["exists"],
        "input_unified_queue_loaded": payload["input_status"]["unified_queue"]["exists"],
        "actions_consolidated": s["actions_total"] >= 8,
        "ucy_terms_action_present": "FW-TERMS-ucy_crowd_original" in by_id,
        "ucy_h100_action_present": "FW-H100-UCY|100" in by_id,
        "trajnet_h100_action_present": "FW-H100-TrajNet|100" in by_id,
        "conversion_queue_empty_preserved": s["conversion_ready_now"] == 0,
        "no_action_marked_complete": all(str(row["status"]).startswith("not_run") for row in actions),
        "all_actions_have_claim_guards": all(bool(row.get("claim_guard")) for row in actions),
        "user_action_written": payload["user_action_required_written"] is True,
        "no_download_conversion_eval": (
            payload["claim_boundary"]["download_executed"] is False
            and payload["claim_boundary"]["conversion_executed"] is False
            and payload["claim_boundary"]["evaluation_executed"] is False
        ),
        "no_metric_seconds_overclaim": (
            payload["claim_boundary"]["global_metric_claim_allowed"] is False
            and payload["claim_boundary"]["global_seconds_claim_allowed"] is False
        ),
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_fw_source_action_consolidator_pass" if passed == total else "stage42_fw_source_action_consolidator_partial"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _input_status(inputs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    paths = {
        "source_terms_gap": SOURCE_TERMS_GAP_JSON,
        "source_closure": SOURCE_CLOSURE_JSON,
        "h100_queue": H100_QUEUE_JSON,
        "unified_queue": UNIFIED_QUEUE_JSON,
        "official_links": OFFICIAL_LINKS_JSON,
    }
    out: dict[str, dict[str, Any]] = {}
    for key, path in paths.items():
        payload = inputs.get(key, {})
        out[key] = {
            "path": str(path),
            "exists": path.exists(),
            "source": payload.get("source"),
            "generated_at_utc": payload.get("generated_at_utc"),
            "verdict": _find_verdict(payload),
        }
    return out


def _find_verdict(payload: Mapping[str, Any]) -> Any:
    for key, value in payload.items():
        if key.endswith("_gate") and isinstance(value, Mapping):
            return value.get("verdict")
    return payload.get("verdict")


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-FW Source Action Consolidator",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_fw_gate']['passed']} / {payload['stage42_fw_gate']['total']}`",
        f"- verdict: `{payload['stage42_fw_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Summary",
        "",
        f"- actions_total: `{s['actions_total']}`",
        f"- categories: `{s['categories']}`",
        f"- top_actions: `{s['top_actions']}`",
        f"- conversion_ready_now: `{s['conversion_ready_now']}`",
        f"- blocked_action_count: `{s['blocked_action_count']}`",
        f"- downloads/conversions/evaluations executed: `{s['downloads_or_conversions_executed']}` / `{s['downloads_or_conversions_executed']}` / `{s['evaluations_executed']}`",
        f"- highest_priority_blocker: `{s['highest_priority_blocker']}`",
        "",
        "## Input Status",
        "",
        "| input | exists | verdict | generated_at_utc |",
        "| --- | ---: | --- | --- |",
    ]
    for key, row in payload["input_status"].items():
        lines.append(
            f"| `{key}` | `{row['exists']}` | `{row.get('verdict')}` | `{row.get('generated_at_utc')}` |"
        )
    lines += [
        "",
        "## Consolidated Actions",
        "",
        "| rank | action | category | target | domain | priority | status | missing | claim guard |",
        "| ---: | --- | --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for idx, row in enumerate(payload["consolidated_actions"], start=1):
        lines.append(
            f"| {idx} | `{row['action_id']}` | `{row['category']}` | `{row['target']}` | `{row['domain']}` | {row['priority']} | `{row['status']}` | {', '.join(map(str, row.get('missing', []))) or 'none'} | {row['claim_guard']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- This consolidator is an action router, not a conversion/evaluation/training stage.",
        "- UCY terms/path confirmation remains the highest-leverage unblocker because it can repair both source support and h100 weak slices.",
        "- TrajNet h100 remains a hard source-support blocker: current local TrajNet snippets are too short for raw-frame h100 support.",
        "- No external not_run item is counted as complete; all claims remain protected dataset-local/raw-frame 2.5D.",
        "- Stage5C remains false and SMC remains false.",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for gate, passed in payload["stage42_fw_gate"]["gates"].items():
        lines.append(f"| `{gate}` | {passed} |")
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-FW Consolidated Source Actions",
        "",
        "No source is newly conversion-ready in this stage. The rows below are the de-duplicated highest-impact actions required before any guarded external conversion/evaluation can be claimed.",
        "",
        "Do not convert, evaluate, or claim metric/seconds/source-closed results until the relevant validator, guarded conversion, no-leakage, and source-CV stages pass.",
        "",
    ]
    for idx, row in enumerate(payload["consolidated_actions"][:8], start=1):
        lines += [
            f"## {idx}. {row['action_id']}",
            "",
            f"- target: `{row['target']}`",
            f"- domain: `{row['domain']}`",
            f"- category: `{row['category']}`",
            f"- priority: `{row['priority']}`",
            f"- status: `{row['status']}`",
            f"- official_url: {row.get('official_url') or 'not_recorded_in_consolidated_inputs'}",
            f"- missing: `{row.get('missing', [])}`",
            f"- next user action: {row['next_user_action']}",
            f"- next commands after confirmation: `{row['next_commands_after_user_confirmation']}`",
            f"- claim guard: {row['claim_guard']}",
            "",
        ]
    return lines


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "## Stage42-FW Source Action Consolidator",
        "",
        "- source: `fresh_stage42_source_action_consolidator_from_existing_blockers`",
        f"- gate: `{payload['stage42_fw_gate']['passed']} / {payload['stage42_fw_gate']['total']}`; verdict `{payload['stage42_fw_gate']['verdict']}`",
        f"- consolidated actions: `{s['actions_total']}`; categories `{s['categories']}`",
        f"- top actions: `{s['top_actions']}`",
        f"- conversion_ready_now: `{s['conversion_ready_now']}`; blocked_action_count: `{s['blocked_action_count']}`",
        "- This is a source/legal/horizon action router only: no download, conversion, training, evaluation, metric/seconds claim, Stage5C execution, or SMC.",
        "- Highest-value path remains UCY terms/path confirmation plus guarded conversion/no-leakage/source-CV; TrajNet h100 needs a longer legal source because local snippets are too short.",
        "- Claim boundary unchanged: protected dataset-local/raw-frame 2.5D only; not true 3D, not foundation, not metric/seconds-level.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY, GOAL_LEDGER]:
        _replace_section(path, "STAGE42_FW_SOURCE_ACTION_CONSOLIDATOR", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    s = payload["summary"]
    state["current_stage"] = "Stage42-FW source action consolidator"
    state["current_verdict"] = payload["stage42_fw_gate"]["verdict"]
    state["stage42_fw_source_action_consolidator"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "report_json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_fw_gate"]["verdict"],
        "gates": f"{payload['stage42_fw_gate']['passed']}/{payload['stage42_fw_gate']['total']}",
        "actions_total": s["actions_total"],
        "top_actions": s["top_actions"],
        "conversion_ready_now": s["conversion_ready_now"],
        "result_source": "fresh_run_consolidation_from_cached_verified_blocker_reports",
        "downloads_or_conversions_executed": 0,
        "evaluations_executed": 0,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_source_action_consolidator() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    inputs = _load_inputs()
    official_urls = _official_url_map(inputs["official_links"])
    actions = _dedupe_actions(
        [
            *_terms_actions(inputs, official_urls),
            *_h100_actions(inputs),
            *_domain_actions(inputs),
        ]
    )
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FW source action consolidator",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([SOURCE_TERMS_GAP_JSON, SOURCE_CLOSURE_JSON, H100_QUEUE_JSON, UNIFIED_QUEUE_JSON, OFFICIAL_LINKS_JSON]),
        "current_facts": CURRENT_FACTS,
        "input_status": _input_status(inputs),
        "claim_boundary": CLAIM_BOUNDARY,
        "consolidated_actions": actions,
        "summary": _summary(actions, inputs),
        "user_action_required_written": True,
    }
    payload["stage42_fw_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_report(payload)[-20:])
    write_md(USER_ACTION_MD, _render_user_action(payload))
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_source_action_consolidator()
    gate = result["stage42_fw_gate"]
    print(f"Stage42-FW gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
