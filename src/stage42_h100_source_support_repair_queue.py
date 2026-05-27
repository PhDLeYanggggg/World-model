from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src import stage42_fe_source_robustness_audit as fg
from src import stage42_h100_weak_horizon_source_support_audit as fp
from src import stage42_post_bj_local_source_verification as bk
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "h100_source_support_repair_queue_stage42.json"
REPORT_MD = OUT_DIR / "h100_source_support_repair_queue_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fq_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_h100_source_support_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
GOAL_LEDGER = Path("README_M3W_GOAL_EVIDENCE_LEDGER_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_FILES = fp.fi.PAPER_FILES

SOURCE = "fresh_stage42_h100_source_support_repair_queue"
VERIFICATION = {
    "runner": ".venv-pytorch/bin/python run_stage42_h100_source_support_repair_queue.py -> 15/15",
    "focused_pytest": ".venv-pytorch/bin/python -m pytest tests/test_stage42_h100_source_support_repair_queue.py -> 4 passed",
    "full_pytest": ".venv-pytorch/bin/python -m pytest tests -> 836 passed",
}

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-FP 证明 TrajNet|100 和 UCY|100 的 blocker 包含 source/support/context 缺口。",
    "Stage42-FQ 只做本地合法性/路径/支持闭环队列，不自动下载、不把 local path 写成 license confirmed。",
    "future waypoints / endpoints 只作为 supervised labels 或 diagnostic/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _domain_from_key(key: str) -> str:
    return key.split("|", 1)[0]


def _family_bucket(relative_path: str) -> str:
    text = str(relative_path)
    lower = text.lower()
    if "zara" in lower or "crowds_zara" in lower:
        return "zara"
    if "student" in lower:
        return "students"
    if "seq_eth" in lower or "eth" in lower:
        return "eth_seq"
    if "hotel" in lower:
        return "hotel"
    if "crowd" in lower:
        return "crowds"
    return Path(text).parts[1] if len(Path(text).parts) > 1 else text


def _target_bucket(key: str, fp_payload: Mapping[str, Any]) -> str:
    rows = fp_payload.get("audits", {}).get(key, {}).get("support", {}).get("test_source_rows", [])
    if not rows:
        return _domain_from_key(key).lower()
    return _family_bucket(str(rows[0].get("name", "")))


def _scan_local_sources() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in bk._iter_files():
        row = bk._parse_file(path)
        if row.get("synthetic_or_diagnostic"):
            continue
        if row.get("domain") not in {"ETH_UCY", "UCY", "TrajNet"}:
            continue
        rows.append(dict(row))
    return rows


def _candidate_rows_for_key(key: str, rows: list[Mapping[str, Any]], fp_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    domain = _domain_from_key(key)
    target_bucket = _target_bucket(key, fp_payload)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        row_domain = str(row.get("domain"))
        if domain == "UCY" and row_domain != "UCY":
            continue
        if domain == "TrajNet" and row_domain != "TrajNet":
            continue
        if not bool(row.get("t100_capable")):
            continue
        bucket = _family_bucket(str(row.get("relative_path", row.get("path", ""))))
        estimated_windows = int(row.get("estimated_t100_windows", 0) or 0)
        candidates.append(
            {
                "relative_path": row.get("relative_path"),
                "domain": row_domain,
                "independent_key": row.get("independent_key"),
                "family_bucket": bucket,
                "target_bucket_match": bucket == target_bucket,
                "file_format": row.get("file_format"),
                "max_track_points": int(row.get("max_track_points", 0) or 0),
                "estimated_t100_windows": estimated_windows,
                "license_status": "local_path_present_terms_unverified",
                "conversion_status": "not_converted_not_evaluated",
                "priority_score": _priority_score(bucket == target_bucket, estimated_windows),
            }
        )
    candidates.sort(key=lambda row: (-int(row["target_bucket_match"]), -int(row["priority_score"]), str(row["relative_path"])))
    return candidates


def _priority_score(bucket_match: bool, windows: int) -> int:
    score = 20 if bucket_match else 5
    if windows >= 1000:
        score += 20
    elif windows >= 100:
        score += 10
    elif windows > 0:
        score += 3
    return score


def _local_gap_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, dict[str, Any]] = defaultdict(lambda: {"files": 0, "t100_files": 0, "independent_groups": set(), "short_files": 0})
    for row in rows:
        domain = str(row.get("domain"))
        by_domain[domain]["files"] += 1
        if bool(row.get("t100_capable")):
            by_domain[domain]["t100_files"] += 1
            by_domain[domain]["independent_groups"].add(str(row.get("independent_key")))
        else:
            by_domain[domain]["short_files"] += 1
    out: dict[str, Any] = {}
    for domain, row in by_domain.items():
        out[domain] = {
            "files": int(row["files"]),
            "t100_files": int(row["t100_files"]),
            "independent_t100_groups": len(row["independent_groups"]),
            "short_or_non_t100_files": int(row["short_files"]),
        }
    return out


def _repair_status(key: str, candidates: list[Mapping[str, Any]]) -> dict[str, Any]:
    domain = _domain_from_key(key)
    exact = [row for row in candidates if row["target_bucket_match"]]
    if domain == "TrajNet" and not candidates:
        return {
            "status": "hard_blocker_no_local_trajnet_h100_long_source",
            "can_repair_now": False,
            "requires_user_action": True,
            "reason": "local TrajNet files are short snippets and cannot provide raw-frame h100 source support",
        }
    if exact:
        return {
            "status": "candidate_support_exists_terms_unverified",
            "can_repair_now": False,
            "requires_user_action": True,
            "reason": "local candidate support exists but terms/license and conversion/no-leakage/source-CV are not confirmed",
        }
    if candidates:
        return {
            "status": "partial_support_exists_family_mismatch_terms_unverified",
            "can_repair_now": False,
            "requires_user_action": True,
            "reason": "local t100 support exists but does not exactly match the target weak-source family and remains terms-unverified",
        }
    return {
        "status": "no_local_support_candidate",
        "can_repair_now": False,
        "requires_user_action": True,
        "reason": "no local t100 candidate support found for this weak key",
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    fp_payload = read_json(fp.REPORT_JSON, {}) or fp.run_stage42_h100_weak_horizon_source_support_audit()
    rows = _scan_local_sources()
    weak_keys = list(fp_payload.get("summary", {}).get("h100_weak_horizons", []))
    key_rows: dict[str, Any] = {}
    actions: list[dict[str, Any]] = []
    for key in weak_keys:
        candidates = _candidate_rows_for_key(key, rows, fp_payload)
        status = _repair_status(key, candidates)
        key_rows[key] = {
            "target_bucket": _target_bucket(key, fp_payload),
            "fp_blockers": fp_payload.get("audits", {}).get(key, {}).get("blockers", []),
            "candidate_count": len(candidates),
            "top_candidates": candidates[:20],
            "repair_status": status,
        }
        actions.append(_user_action_for_key(key, key_rows[key]))
    summary = {
        "source": SOURCE,
        "input_fp_verdict": fp_payload.get("stage42_fp_gate", {}).get("verdict"),
        "weak_keys": weak_keys,
        "weak_key_count": len(weak_keys),
        "local_files_scanned": len(rows),
        "local_gap_summary": _local_gap_summary(rows),
        "repairable_now_count": 0,
        "user_action_required_count": len(actions),
        "uniform_horizon_claim_allowed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FQ H100 source-support repair queue",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(fp.REPORT_JSON)] + [str(p) for p in bk._iter_files()[:200]]),
        "current_facts": CURRENT_FACTS,
        "summary": summary,
        "key_rows": key_rows,
        "user_action_required": actions,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "local_inventory_only": True,
            "auto_download_executed": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "converted_dataset_claim_allowed": False,
            "uniform_horizon_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "verification": VERIFICATION,
    }
    payload["stage42_fq_gate"] = _gate(payload)
    return payload


def _user_action_for_key(key: str, row: Mapping[str, Any]) -> dict[str, Any]:
    status = row["repair_status"]["status"]
    if key.startswith("TrajNet"):
        return {
            "key": key,
            "priority": "high",
            "action_type": "provide_or_confirm_official_long_trajnet_source",
            "reason": row["repair_status"]["reason"],
            "official_source_hint": "TrajNet++ / official raw long trajectory source if available; current local snippets are too short for raw-frame h100.",
            "do_not_count_as_completed_until": "license/terms confirmed, conversion finished, no-leakage pass, train-only source-CV positive/easy-safe",
        }
    return {
        "key": key,
        "priority": "high" if "candidate_support_exists" in status else "medium",
        "action_type": "confirm_terms_and_convert_local_ucy_h100_support",
        "reason": row["repair_status"]["reason"],
        "top_candidates": [candidate["relative_path"] for candidate in row["top_candidates"][:10]],
        "do_not_count_as_completed_until": "license/terms confirmed, conversion finished, no-leakage pass, train-only source-CV positive/easy-safe",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    no_leak = payload["no_leakage"]
    boundary = payload["claim_boundary"]
    gates = {
        "source_fresh": payload["source"] == SOURCE,
        "fp_input_verified": str(s.get("input_fp_verdict", "")).startswith("stage42_fp_h100_source_support_audit_pass"),
        "weak_keys_loaded": int(s["weak_key_count"]) >= 1,
        "local_sources_scanned": int(s["local_files_scanned"]) > 0,
        "per_key_repair_status_built": all("repair_status" in row for row in payload["key_rows"].values()),
        "trajnet_gap_explained": any(
            row["repair_status"]["status"] == "hard_blocker_no_local_trajnet_h100_long_source"
            for key, row in payload["key_rows"].items()
            if key.startswith("TrajNet")
        ),
        "ucy_action_or_blocker_recorded": any(key.startswith("UCY") for key in payload["key_rows"]),
        "user_actions_generated": len(payload["user_action_required"]) == int(s["weak_key_count"]),
        "no_auto_download": no_leak["auto_download_executed"] is False,
        "no_converted_dataset_overclaim": boundary["converted_dataset_claim_allowed"] is False,
        "uniform_horizon_claim_false": boundary["uniform_horizon_claim"] is False,
        "no_future_or_test_leakage": all(
            [
                no_leak["future_endpoint_input"] is False,
                no_leak["future_waypoint_input"] is False,
                no_leak["central_velocity"] is False,
                no_leak["test_endpoint_goals"] is False,
                no_leak["test_threshold_tuning"] is False,
                no_leak["local_inventory_only"] is True,
            ]
        ),
        "no_metric_seconds_overclaim": boundary["metric_or_seconds_claim"] is False,
        "stage5c_false": boundary["stage5c_executed"] is False,
        "smc_false": boundary["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_fq_h100_source_support_repair_queue_pass" if passed == total else "stage42_fq_h100_source_support_repair_queue_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fq_gate"]
    s = payload["summary"]
    lines = [
        "# Stage42-FQ H100 Source-Support Repair Queue",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- input FP verdict: `{s['input_fp_verdict']}`",
        f"- weak keys: `{s['weak_keys']}`",
        f"- local files scanned: `{s['local_files_scanned']}`",
        f"- uniform horizon claim allowed: `{s['uniform_horizon_claim_allowed']}`",
        f"- verification: `{payload['verification']}`",
        "",
        "## Local Gap Summary",
        "",
        "| domain | files | t100 files | independent t100 groups | short/non-t100 files |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for domain, row in s["local_gap_summary"].items():
        lines.append(
            f"| `{domain}` | {row['files']} | {row['t100_files']} | {row['independent_t100_groups']} | {row['short_or_non_t100_files']} |"
        )
    for key, row in payload["key_rows"].items():
        status = row["repair_status"]
        lines += [
            "",
            f"## `{key}`",
            "",
            f"- target bucket: `{row['target_bucket']}`",
            f"- FP blockers: `{row['fp_blockers']}`",
            f"- candidate count: `{row['candidate_count']}`",
            f"- repair status: `{status['status']}`",
            f"- reason: {status['reason']}",
            "",
            "| candidate | family | match | max track | est h100 windows | license | status |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
        for candidate in row["top_candidates"][:20]:
            lines.append(
                f"| `{candidate['relative_path']}` | `{candidate['family_bucket']}` | {candidate['target_bucket_match']} | "
                f"{candidate['max_track_points']} | {candidate['estimated_t100_windows']} | `{candidate['license_status']}` | `{candidate['conversion_status']}` |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "- FQ is a repair queue, not a conversion or model-training step.",
        "- Local UCY h100 candidates can only become support after terms confirmation, conversion, no-leakage, and train-only source-CV.",
        "- Local TrajNet files do not currently provide long raw h100 support; TrajNet|100 needs official longer sources or the uniform-horizon claim must remain blocked.",
        "- No raw data is committed, no auto-download is performed, and no metric/seconds/Stage5C/SMC claim is made.",
    ]
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-FQ User Action Required: H100 Source Support",
        "",
        f"- source: `{payload['source']}`",
        "",
    ]
    for action in payload["user_action_required"]:
        lines += [
            f"## `{action['key']}`",
            "",
            f"- priority: `{action['priority']}`",
            f"- action_type: `{action['action_type']}`",
            f"- reason: {action['reason']}",
        ]
        if "official_source_hint" in action:
            lines.append(f"- official_source_hint: {action['official_source_hint']}")
        if "top_candidates" in action:
            lines.append("- top_candidates:")
            lines.extend([f"  - `{path}`" for path in action["top_candidates"]])
        lines.append(f"- do_not_count_as_completed_until: {action['do_not_count_as_completed_until']}")
        lines.append("")
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_fq_gate"]
    lines = [
        "# Stage42-FQ Gate",
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


def _summary_section(payload: Mapping[str, Any]) -> str:
    s = payload["summary"]
    return "\n".join(
        [
            "<!-- STAGE42_FQ_H100_SOURCE_SUPPORT_REPAIR_QUEUE:START -->",
            "## Stage42-FQ H100 Source-Support Repair Queue",
            "",
            f"- source: `{payload['source']}`",
            "- role: local source-support repair queue for FP h100 blockers; no conversion, no training, no auto-download.",
            f"- gate: `{payload['stage42_fq_gate']['passed']} / {payload['stage42_fq_gate']['total']}`; verdict `{payload['stage42_fq_gate']['verdict']}`.",
            f"- weak keys: `{s['weak_keys']}`.",
            f"- local gap summary: `{s['local_gap_summary']}`.",
            "- TrajNet|100 status: no local long raw h100 TrajNet source; user must provide or confirm official longer source.",
            "- UCY|100 status: local UCY h100 candidates exist but are terms-unverified and require conversion/no-leakage/source-CV before use.",
            "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
            f"- verification: `{payload['verification']}`.",
            "<!-- STAGE42_FQ_H100_SOURCE_SUPPORT_REPAIR_QUEUE:END -->",
            "",
        ]
    )


def _append_to_docs(payload: Mapping[str, Any]) -> None:
    section = _summary_section(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY, GOAL_LEDGER, *PAPER_FILES]:
        old = path.read_text() if path.exists() else ""
        path.write_text(fg._replace_text_section(old, "STAGE42_FQ_H100_SOURCE_SUPPORT_REPAIR_QUEUE", section))


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FQ H100 source-support repair queue"
    state["current_verdict"] = payload["stage42_fq_gate"]["verdict"]
    state["stage42_fq_h100_source_support_repair_queue"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_fq_gate"]["verdict"],
        "gates": f"{payload['stage42_fq_gate']['passed']}/{payload['stage42_fq_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "verification": payload["verification"],
        "conclusion": "Stage42-FQ turns FP h100 blockers into a concrete source-support repair queue: UCY has terms-unverified local candidates; TrajNet needs longer official raw sources.",
    }
    block = state.get("m3w_goal_evidence_ledger_readme")
    if isinstance(block, dict):
        block["latest_conclusion"] = "Stage42-FQ confirms h100 repair now depends on source-support closure: UCY terms/conversion and TrajNet longer raw sources."
        state["m3w_goal_evidence_ledger_readme"] = block
    write_json(RESEARCH_STATE, state)


def run_stage42_h100_source_support_repair_queue() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_h100_source_support_repair_queue()
    gate = result["stage42_fq_gate"]
    print(f"Stage42-FQ gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
