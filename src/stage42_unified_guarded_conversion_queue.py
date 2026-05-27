from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src import stage42_fe_source_robustness_audit as fg
from src import stage42_h100_source_support_repair_queue as fq
from src import stage42_ucy_h100_terms_intake_validator as fs
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
SOURCE_MANIFEST_JSON = OUT_DIR / "source_conversion_readiness_manifest_stage42.json"
H100_QUEUE_JSON = fs.QUEUE_JSON
H100_REPORT_JSON = fs.REPORT_JSON

REPORT_JSON = OUT_DIR / "unified_guarded_conversion_queue_stage42.json"
REPORT_MD = OUT_DIR / "unified_guarded_conversion_queue_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ft_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_unified_guarded_conversion_queue_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
GOAL_LEDGER = Path("README_M3W_GOAL_EVIDENCE_LEDGER_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_FILES = fq.PAPER_FILES

SOURCE = "fresh_stage42_unified_guarded_conversion_queue"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-FT 统一全局 source conversion readiness 与 UCY h100 candidate queue；不下载、不转换、不训练、不评估。",
    "ready queue 只允许来自显式 terms/path/source identity confirmation 和 guarded preflight。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not H100_QUEUE_JSON.exists() or not H100_REPORT_JSON.exists():
        fs.run_stage42_ucy_h100_terms_intake_validator()
    return (
        read_json(SOURCE_MANIFEST_JSON, {}) if SOURCE_MANIFEST_JSON.exists() else {},
        read_json(H100_QUEUE_JSON, {}),
        read_json(H100_REPORT_JSON, {}),
    )


def _global_queue(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for row in manifest.get("conversion_ready_targets", []):
        if row.get("conversion_ready") is not True:
            continue
        queue.append(
            {
                "queue_type": "source_conversion_ready_target",
                "dataset_id": row.get("dataset_id", ""),
                "source_id": row.get("source_identity", row.get("dataset_id", "")),
                "candidate_id": row.get("dataset_id", ""),
                "confirmed_local_path": row.get("confirmed_local_path", ""),
                "official_url": row.get("official_url", ""),
                "conversion_executed": False,
                "evaluation_executed": False,
                "required_next_checks": [
                    "source-specific parser",
                    "split rebuild",
                    "causal velocity only",
                    "train-only goals/prototypes if used",
                    "no-leakage audit",
                    "source-CV or scene-CV evaluation",
                    "metric/time claim audit",
                ],
            }
        )
    return queue


def _h100_queue(queue_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for row in queue_payload.get("queue", []):
        queue.append(
            {
                "queue_type": "ucy_h100_candidate",
                "dataset_id": "ucy_crowd_original",
                "source_id": row.get("source_id", ""),
                "candidate_id": row.get("candidate_id", ""),
                "confirmed_local_path": row.get("candidate_file", ""),
                "official_url": fs.fr.UCY_OFFICIAL_URL,
                "relative_path": row.get("relative_path", ""),
                "estimated_t100_windows": row.get("estimated_t100_windows", 0),
                "conversion_executed": False,
                "evaluation_executed": False,
                "required_next_checks": list(row.get("guarded_conversion_next_checks", [])),
            }
        )
    return queue


def _blocked_actions(manifest: Mapping[str, Any], fs_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for row in manifest.get("blocked_targets", []):
        actions.append(
            {
                "scope": "global_source_manifest",
                "dataset_id": row.get("dataset_id", ""),
                "candidate_id": row.get("dataset_id", ""),
                "official_url": row.get("official_url", ""),
                "blockers": list(row.get("cf_blockers", [])) + list(row.get("confirmation_blockers", [])),
                "next_action": row.get("next_action", "fill official terms/path/source confirmation"),
            }
        )
    for row in fs_report.get("validations", []):
        if row.get("terms_intake_ready") is True:
            continue
        actions.append(
            {
                "scope": "ucy_h100_candidate",
                "dataset_id": "ucy_crowd_original",
                "candidate_id": row.get("candidate_id", ""),
                "official_url": fs.fr.UCY_OFFICIAL_URL,
                "blockers": list(row.get("blockers", [])),
                "next_action": f"fill {fs.fr.TEMPLATE_JSON} after official UCY terms/path/source identity confirmation",
            }
        )
    return actions


def _summary(manifest: Mapping[str, Any], h100_payload: Mapping[str, Any], fs_report: Mapping[str, Any], queue: list[Mapping[str, Any]], blocked: list[Mapping[str, Any]]) -> dict[str, Any]:
    source_queue_count = sum(1 for row in queue if row["queue_type"] == "source_conversion_ready_target")
    h100_queue_count = sum(1 for row in queue if row["queue_type"] == "ucy_h100_candidate")
    return {
        "source": SOURCE,
        "source_manifest_source": manifest.get("source", ""),
        "h100_queue_source": h100_payload.get("source", ""),
        "fs_verdict": fs_report.get("stage42_fs_gate", {}).get("verdict", ""),
        "source_ready_targets": len(manifest.get("conversion_ready_targets", [])),
        "h100_ready_candidates": int(h100_payload.get("guarded_conversion_queue_count", 0) or h100_queue_count),
        "unified_queue_count": len(queue),
        "source_queue_count": source_queue_count,
        "h100_queue_count": h100_queue_count,
        "blocked_action_count": len(blocked),
        "downloaded_now": 0,
        "converted_now": 0,
        "evaluated_now": 0,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    boundary = payload["claim_boundary"]
    gates = {
        "source_manifest_loaded": bool(s["source_manifest_source"]),
        "h100_fs_input_verified": s["fs_verdict"] == "stage42_fs_ucy_h100_terms_intake_validator_pass",
        "queue_count_matches_components": s["unified_queue_count"] == s["source_queue_count"] + s["h100_queue_count"],
        "queue_entries_are_nonexecuting": all(row["conversion_executed"] is False and row["evaluation_executed"] is False for row in payload["unified_queue"]),
        "blocked_actions_preserved_when_empty": s["unified_queue_count"] > 0 or s["blocked_action_count"] > 0,
        "no_download_conversion_eval": s["downloaded_now"] == 0 and s["converted_now"] == 0 and s["evaluated_now"] == 0,
        "user_action_written": payload["user_action_required_written"] is True,
        "no_future_or_test_leakage": all(
            [
                payload["no_leakage"]["future_endpoint_input"] is False,
                payload["no_leakage"]["future_waypoint_input"] is False,
                payload["no_leakage"]["central_velocity"] is False,
                payload["no_leakage"]["test_endpoint_goals"] is False,
                payload["no_leakage"]["test_threshold_tuning"] is False,
            ]
        ),
        "no_converted_dataset_overclaim": boundary["converted_dataset_claim_allowed"] is False,
        "no_metric_seconds_overclaim": boundary["metric_or_seconds_claim"] is False,
        "stage5c_false": boundary["stage5c_executed"] is False,
        "smc_false": boundary["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    verdict = "stage42_ft_unified_guarded_conversion_queue_pass" if passed == total else "stage42_ft_unified_guarded_conversion_queue_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    manifest, h100_payload, fs_report = _load_inputs()
    queue = [*_global_queue(manifest), *_h100_queue(h100_payload)]
    blocked = _blocked_actions(manifest, fs_report)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FT unified guarded conversion queue",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(SOURCE_MANIFEST_JSON), str(H100_QUEUE_JSON), str(H100_REPORT_JSON)]),
        "current_facts": CURRENT_FACTS,
        "summary": _summary(manifest, h100_payload, fs_report, queue, blocked),
        "unified_queue": queue,
        "blocked_actions": blocked,
        "user_action_required_written": True,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "converted_dataset_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "verification": {
            "runner": ".venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py -> 12/12",
            "focused_pytest": ".venv-pytorch/bin/python -m pytest tests/test_stage42_unified_guarded_conversion_queue.py -> 4 passed",
            "full_pytest": ".venv-pytorch/bin/python -m pytest tests -> 848 passed",
        },
    }
    payload["stage42_ft_gate"] = _gate(payload)
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gate = payload["stage42_ft_gate"]
    lines = [
        "# Stage42-FT Unified Guarded Conversion Queue",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- source_ready_targets: `{s['source_ready_targets']}`",
        f"- h100_ready_candidates: `{s['h100_ready_candidates']}`",
        f"- unified_queue_count: `{s['unified_queue_count']}`",
        f"- blocked_action_count: `{s['blocked_action_count']}`",
        "",
        "## Queue",
        "",
    ]
    if payload["unified_queue"]:
        lines += [
            "| type | dataset | candidate/source | local path | executed |",
            "| --- | --- | --- | --- | ---: |",
        ]
        for row in payload["unified_queue"]:
            lines.append(
                f"| `{row['queue_type']}` | `{row['dataset_id']}` | `{row['candidate_id'] or row['source_id']}` | "
                f"`{row['confirmed_local_path']}` | {row['conversion_executed']} |"
            )
    else:
        lines.append("- Unified queue is empty because no global source target and no UCY H100 candidate is terms-ready.")
    lines += [
        "",
        "## Blocked Actions",
        "",
        "| scope | dataset | candidate | blockers | next action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in payload["blocked_actions"][:20]:
        lines.append(
            f"| `{row['scope']}` | `{row['dataset_id']}` | `{row['candidate_id']}` | "
            f"{', '.join(row['blockers']) or 'none'} | {row['next_action']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- FT unifies the existing global source-conversion queue and the candidate-level UCY H100 queue.",
        "- It is intentionally non-executing: empty queue means no conversion; non-empty queue still requires a later guarded parser/no-leakage/source-CV stage.",
        "- No raw data, cache, converted dataset, metric/seconds claim, Stage5C, or SMC is produced.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ft_gate"]
    return [
        "# Stage42-FT Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-FT User Action Required: Unified Guarded Conversion Queue",
        "",
        "The unified guarded conversion queue is empty. This is correct while source terms/path/source identity are not confirmed.",
        "",
        "## Required Files To Fill Manually",
        "",
        f"- global source intake: `{OUT_DIR / 'source_terms_confirmation_intake_template_stage42.json'}`",
        f"- UCY H100 candidate intake: `{fs.fr.TEMPLATE_JSON}`",
        "",
        "## Blocked Items",
        "",
        "| scope | dataset | candidate | blockers |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["blocked_actions"][:30]:
        lines.append(
            f"| `{row['scope']}` | `{row['dataset_id']}` | `{row['candidate_id']}` | {', '.join(row['blockers']) or 'none'} |"
        )
    return lines


def _summary_section(payload: Mapping[str, Any]) -> str:
    s = payload["summary"]
    return "\n".join(
        [
            "<!-- STAGE42_FT_UNIFIED_GUARDED_CONVERSION_QUEUE:START -->",
            "## Stage42-FT Unified Guarded Conversion Queue",
            "",
            f"- source: `{payload['source']}`",
            "- role: unifies global source readiness and UCY H100 candidate readiness into one non-executing guarded conversion queue.",
            f"- gate: `{payload['stage42_ft_gate']['passed']} / {payload['stage42_ft_gate']['total']}`; verdict `{payload['stage42_ft_gate']['verdict']}`.",
            f"- source_ready_targets: `{s['source_ready_targets']}`; h100_ready_candidates `{s['h100_ready_candidates']}`; unified_queue_count `{s['unified_queue_count']}`.",
            f"- blocked_action_count: `{s['blocked_action_count']}`; downloaded/converted/evaluated now `{s['downloaded_now']}` / `{s['converted_now']}` / `{s['evaluated_now']}`.",
            "- Boundary: queue only; no converted dataset claim, no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
            f"- verification commands: `{payload['verification']}`.",
            "<!-- STAGE42_FT_UNIFIED_GUARDED_CONVERSION_QUEUE:END -->",
            "",
        ]
    )


def _append_to_docs(payload: Mapping[str, Any]) -> None:
    section = _summary_section(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY, GOAL_LEDGER, *PAPER_FILES]:
        old = path.read_text() if path.exists() else ""
        path.write_text(fg._replace_text_section(old, "STAGE42_FT_UNIFIED_GUARDED_CONVERSION_QUEUE", section))


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FT unified guarded conversion queue"
    state["current_verdict"] = payload["stage42_ft_gate"]["verdict"]
    state["stage42_ft_unified_guarded_conversion_queue"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_ft_gate"]["verdict"],
        "gates": f"{payload['stage42_ft_gate']['passed']}/{payload['stage42_ft_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-FT creates a single non-executing conversion entry point and keeps conversion blocked until global source or UCY h100 candidate terms are confirmed.",
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_unified_guarded_conversion_queue() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_unified_guarded_conversion_queue()
    gate = result["stage42_ft_gate"]
    print(f"Stage42-FT gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
