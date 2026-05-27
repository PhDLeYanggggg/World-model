from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_full_waypoint_bridge_shape_audit import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

HM_JSON = OUT_DIR / "restricted_metric_time_terms_intake_v2_stage42.json"
HM_MANIFEST_JSON = OUT_DIR / "restricted_metric_time_terms_intake_v2_manifest_stage42.json"
HM_TEMPLATE_JSON = OUT_DIR / "restricted_metric_time_terms_intake_v2_template_stage42.json"

REPORT_JSON = OUT_DIR / "restricted_metric_time_conversion_queue_v2_stage42.json"
REPORT_MD = OUT_DIR / "restricted_metric_time_conversion_queue_v2_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hn_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_restricted_metric_time_conversion_queue_v2_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
CURRENT_SUMMARY = Path("README_M3W_CURRENT_DETAILED_SUMMARY_2026_05_27_ZH.md")
A_JOURNAL_GAP = OUT_DIR / "a_journal_gap_stage42.md"
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_hn_restricted_metric_time_conversion_queue_v2"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HN 是 restricted metric/time guarded conversion queue v2，不下载、不转换、不训练、不评估。",
    "本阶段只读取 Stage42-HM source-level terms intake v2 manifest。",
    "只有 HM manifest 中 ready_candidates 非空时，才允许排队未来 guarded conversion；当前 ready_candidates 为 0。",
    "future endpoint / future waypoint 只可作为 supervised/evaluation labels，不可作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level；restricted metric/time 仍需转换后重新审计。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "restricted_metric_time_claim_allowed_now": False,
    "converted_dataset_claim_allowed": False,
    "download_executed": False,
    "conversion_executed": False,
    "feature_store_built": False,
    "no_leakage_audit_executed": False,
    "source_cv_executed": False,
    "evaluation_executed": False,
    "training_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}

REQUIRED_NEXT_CHECKS = [
    "source-specific parser",
    "row geometry reconstruction",
    "causal velocity only",
    "train/val/test or source-CV split rebuild",
    "train-only goals/prototypes if legal",
    "no future endpoint input",
    "no central velocity",
    "no test endpoint goals",
    "no test normalization statistics",
    "no-leakage audit",
    "source-CV/final-test evaluation",
    "metric/time claim guard rerun",
]


def _gate_passed(payload: Mapping[str, Any], gate_key: str) -> bool:
    gate = payload.get(gate_key, {})
    return bool(gate and gate.get("passed") == gate.get("total") and int(gate.get("total", 0)) > 0)


def queue_from_ready_candidates(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for row in manifest.get("ready_candidates", []):
        if row.get("conversion_ready") is not True:
            continue
        queue.append(
            {
                "queue_type": "restricted_metric_time_source_candidate",
                "candidate_id": row.get("candidate_id", ""),
                "source_id": row.get("source_id", ""),
                "domain": row.get("domain", ""),
                "terms_target_id": row.get("terms_target_id", ""),
                "t50_windows_after_terms": int(row.get("t50_windows_after_terms", 0) or 0),
                "t100_windows_after_terms": int(row.get("t100_windows_after_terms", 0) or 0),
                "source_cv_usable_after_terms": bool(row.get("source_cv_usable_after_terms")),
                "execution_in_stage42_hn": False,
                "download_executed": False,
                "conversion_executed": False,
                "feature_store_built": False,
                "no_leakage_audit_executed": False,
                "source_cv_executed": False,
                "evaluation_executed": False,
                "required_next_checks": REQUIRED_NEXT_CHECKS,
                "forbidden": [
                    "download without terms confirmation",
                    "convert without no-leakage audit",
                    "metric/seconds claim before restricted conversion/eval",
                    "future endpoint inference input",
                    "central velocity official input",
                    "test endpoint goal construction",
                    "Stage5C execution",
                    "SMC execution",
                ],
            }
        )
    return queue


def blocked_actions_from_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for row in manifest.get("blocked_candidates", []):
        actions.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "source_id": row.get("source_id", ""),
                "domain": row.get("domain", ""),
                "terms_target_id": row.get("terms_target_id", ""),
                "t50_windows_after_terms": int(row.get("t50_windows_after_terms", 0) or 0),
                "t100_windows_after_terms": int(row.get("t100_windows_after_terms", 0) or 0),
                "blockers": list(row.get("blockers", [])),
                "next_action": row.get(
                    "next_action",
                    "complete user-confirmed terms/path/source fields before conversion",
                ),
                "conversion_ready": False,
                "queued": False,
            }
        )
    return actions


def _summary(hm: Mapping[str, Any], manifest: Mapping[str, Any], queue: list[Mapping[str, Any]], blocked: list[Mapping[str, Any]]) -> dict[str, Any]:
    hm_summary = hm.get("summary", {})
    return {
        "source": SOURCE,
        "hm_verdict": hm.get("stage42_hm_gate", {}).get("verdict", ""),
        "hm_source_level_candidates": int(hm_summary.get("source_level_candidates", 0) or 0),
        "manifest_ready_candidates": len(manifest.get("ready_candidates", [])),
        "manifest_blocked_candidates": len(manifest.get("blocked_candidates", [])),
        "conversion_queue_count": len(queue),
        "blocked_action_count": len(blocked),
        "queued_t50_windows": sum(int(row.get("t50_windows_after_terms", 0) or 0) for row in queue),
        "queued_t100_windows": sum(int(row.get("t100_windows_after_terms", 0) or 0) for row in queue),
        "blocked_t50_windows_after_terms": sum(int(row.get("t50_windows_after_terms", 0) or 0) for row in blocked),
        "blocked_t100_windows_after_terms": sum(int(row.get("t100_windows_after_terms", 0) or 0) for row in blocked),
        "download_executed": False,
        "conversion_executed": False,
        "feature_store_built": False,
        "no_leakage_audit_executed": False,
        "source_cv_executed": False,
        "evaluation_executed": False,
        "training_executed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    hm = read_json(HM_JSON, {})
    manifest = read_json(HM_MANIFEST_JSON, {})
    queue = queue_from_ready_candidates(manifest)
    blocked = blocked_actions_from_manifest(manifest)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-HN Restricted Metric/Time Guarded Conversion Queue v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([HM_JSON, HM_MANIFEST_JSON, HM_TEMPLATE_JSON]),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "hm_gate_passed": _gate_passed(hm, "stage42_hm_gate"),
            "hm_manifest_source": manifest.get("source", ""),
            "hm_template_exists": HM_TEMPLATE_JSON.exists(),
        },
        "conversion_queue": queue,
        "blocked_actions": blocked,
        "summary": _summary(hm, manifest, queue, blocked),
        "claim_boundary": CLAIM_BOUNDARY,
        "no_leakage_preconditions": {
            "future_endpoint_input_allowed": False,
            "future_waypoint_input_allowed": False,
            "central_velocity_allowed": False,
            "test_endpoint_goals_allowed": False,
            "test_threshold_tuning_allowed": False,
        },
        "user_action_required": [
            "Fill and validate the Stage42-HM source-level terms intake template before any conversion can be queued.",
            "Rerun `.venv-pytorch/bin/python run_stage42_restricted_metric_time_terms_intake_v2.py --validate-only`.",
            "If ready candidates appear, rerun this queue and then a future guarded converter that performs parser/no-leakage/source-CV/final-test checks.",
        ],
    }
    payload["stage42_hn_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    c = payload["claim_boundary"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "hm_input_passed": payload["inputs"]["hm_gate_passed"] is True,
        "hm_manifest_loaded": payload["inputs"]["hm_manifest_source"] == "fresh_stage42_hm_restricted_metric_time_terms_intake_v2",
        "hm_template_exists": payload["inputs"]["hm_template_exists"] is True,
        "queue_count_matches_ready_candidates": s["conversion_queue_count"] == s["manifest_ready_candidates"],
        "blocked_actions_match_manifest": s["blocked_action_count"] == s["manifest_blocked_candidates"],
        "empty_ready_refuses_conversion": s["manifest_ready_candidates"] == 0
        and s["conversion_queue_count"] == 0
        and s["blocked_action_count"] > 0,
        "queue_entries_nonexecuting": all(
            row["conversion_executed"] is False
            and row["evaluation_executed"] is False
            and row["source_cv_executed"] is False
            for row in payload["conversion_queue"]
        ),
        "no_download_conversion_feature_store": not (
            s["download_executed"] or s["conversion_executed"] or s["feature_store_built"]
        ),
        "no_no_leakage_or_source_cv_claim": not (s["no_leakage_audit_executed"] or s["source_cv_executed"]),
        "no_training_eval_claim": not (s["training_executed"] or s["evaluation_executed"]),
        "no_future_or_test_leakage_allowed": all(value is False for value in payload["no_leakage_preconditions"].values()),
        "no_metric_seconds_or_converted_claim": c["global_metric_claim_allowed"] is False
        and c["global_seconds_claim_allowed"] is False
        and c["restricted_metric_time_claim_allowed_now"] is False
        and c["converted_dataset_claim_allowed"] is False,
        "stage5c_false": c["stage5c_executed"] is False,
        "smc_false": c["smc_enabled"] is False,
    }
    passed = sum(1 for value in gates.values() if value)
    total = len(gates)
    verdict = (
        "stage42_hn_restricted_metric_time_conversion_queue_v2_pass_blocked_until_ready_candidates"
        if passed == total
        else "stage42_hn_restricted_metric_time_conversion_queue_v2_partial"
    )
    return {"source": payload.get("source", SOURCE), "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hn_gate"]
    s = payload["summary"]
    lines = [
        "# Stage42-HN Restricted Metric/Time Guarded Conversion Queue v2",
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
        "## Summary",
        "",
        f"- HM verdict: `{s['hm_verdict']}`",
        f"- HM source-level candidates: `{s['hm_source_level_candidates']}`",
        f"- manifest ready / blocked candidates: `{s['manifest_ready_candidates']}` / `{s['manifest_blocked_candidates']}`",
        f"- conversion_queue_count: `{s['conversion_queue_count']}`",
        f"- blocked_action_count: `{s['blocked_action_count']}`",
        f"- queued t50/t100 windows: `{s['queued_t50_windows']}` / `{s['queued_t100_windows']}`",
        f"- blocked after-terms t50/t100 windows: `{s['blocked_t50_windows_after_terms']}` / `{s['blocked_t100_windows_after_terms']}`",
        "",
        "## Conversion Queue",
        "",
    ]
    if payload["conversion_queue"]:
        lines += [
            "| candidate | source | domain | t50 | t100 | executed |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
        for row in payload["conversion_queue"]:
            lines.append(
                f"| `{row['candidate_id']}` | `{row['source_id']}` | `{row['domain']}` | "
                f"{row['t50_windows_after_terms']} | {row['t100_windows_after_terms']} | {row['conversion_executed']} |"
            )
    else:
        lines.append("- No ready candidates. Conversion is refused, as intended.")
    lines += [
        "",
        "## Blocked Actions",
        "",
        "| candidate | source | domain | t50 | t100 | blockers |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in payload["blocked_actions"]:
        lines.append(
            f"| `{row['candidate_id']}` | `{row['source_id']}` | `{row['domain']}` | "
            f"{row['t50_windows_after_terms']} | {row['t100_windows_after_terms']} | "
            f"{', '.join(row['blockers']) or 'none'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- HN is the execution-side guard for HM. It refuses conversion while ready candidates are zero.",
        "- Blocked after-terms support is retained so the future conversion path is concrete once the user fills and validates terms/source identity/path.",
        "- This is not converted data, not evaluated data, not metric/seconds evidence, not Stage5C, and not SMC.",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hn_gate"]
    return [
        "# Stage42-HN Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    return [
        "# User Action Required: Stage42-HN Restricted Metric/Time Conversion Queue v2",
        "",
        "- Current conversion queue is empty because Stage42-HM has zero ready candidates.",
        "- Fill and validate `outputs/stage42_long_research/restricted_metric_time_terms_intake_v2_template_stage42.json` before rerunning this queue.",
        "- A future guarded converter must still run parser, no-leakage, source-CV/final-test, and claim guard checks before any restricted metric/time claim.",
        "",
        "Current status: no conversion, no evaluation, no metric/seconds claim, no Stage5C, no SMC.",
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hn_gate"]
    s = payload["summary"]
    return [
        "## Stage42-HN Restricted Metric/Time Conversion Queue v2",
        "",
        "- source: `fresh_stage42_hn_restricted_metric_time_conversion_queue_v2`",
        f"- verdict: `{gate['verdict']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- ready / blocked candidates: `{s['manifest_ready_candidates']}` / `{s['manifest_blocked_candidates']}`.",
        f"- conversion queue count: `{s['conversion_queue_count']}`.",
        f"- blocked after-terms t50/t100 windows retained: `{s['blocked_t50_windows_after_terms']}` / `{s['blocked_t100_windows_after_terms']}`.",
        "- conclusion: the restricted metric/time execution path is now guarded by HM ready-candidate validation; current conversion remains refused until user-confirmed terms/source identity/path are supplied.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, CURRENT_SUMMARY, A_JOURNAL_GAP]:
        _replace_section(path, "STAGE42_HN_RESTRICTED_METRIC_TIME_CONVERSION_QUEUE_V2", lines)


def _refresh_research_state(payload: Mapping[str, Any], verification: Mapping[str, Any] | None = None) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-HN restricted metric/time conversion queue v2"
    state["current_verdict"] = payload["stage42_hn_gate"]["verdict"]
    state["stage42_hn_restricted_metric_time_conversion_queue_v2"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_hn_gate"]["verdict"],
        "gates": f"{payload['stage42_hn_gate']['passed']}/{payload['stage42_hn_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "verification": dict(verification or {"status": "pending"}),
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_restricted_metric_time_conversion_queue_v2(
    *, refresh_readmes: bool = True, verification: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload, verification=verification)
    return payload


if __name__ == "__main__":
    run_stage42_restricted_metric_time_conversion_queue_v2()
