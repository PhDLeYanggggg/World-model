from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
MANIFEST_JSON = OUT_DIR / "source_conversion_readiness_manifest_stage42.json"

REPORT_JSON = OUT_DIR / "guarded_source_conversion_launcher_stage42.json"
REPORT_MD = OUT_DIR / "guarded_source_conversion_launcher_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_guarded_source_conversion_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ej_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_guarded_source_conversion_launcher_from_stage42_ei_manifest"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EJ 是 guarded source conversion launcher；它只生成受保护转换队列，不下载、不转换、不训练、不评估。",
    "只有 source_conversion_readiness_manifest_stage42.json 中 conversion_ready_targets 非空时，才允许排队未来转换。",
    "空白 terms intake、local path、parseability、technical dry-run 都不等于 legal conversion readiness。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon；不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "converted_datasets_now": 0,
    "evaluated_datasets_now": 0,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _ready_targets(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in manifest.get("conversion_ready_targets", []) if row.get("conversion_ready") is True]


def _blocked_targets(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in manifest.get("blocked_targets", []) if row.get("conversion_ready") is not True]


def _build_conversion_queue(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for row in _ready_targets(manifest):
        queue.append(
            {
                "dataset_id": row.get("dataset_id", ""),
                "name": row.get("name", row.get("dataset_id", "")),
                "official_url": row.get("official_url", ""),
                "confirmed_local_path": row.get("confirmed_local_path", ""),
                "source_identity": row.get("source_identity", ""),
                "status": "queued_for_future_guarded_conversion",
                "execution_in_stage42_ej": False,
                "requires_next_stage_steps": [
                    "source-specific raw parser",
                    "causal velocity reconstruction",
                    "train/val/test split rebuild",
                    "train-only goal construction if legal",
                    "no-leakage audit",
                    "source-CV or scene-level evaluation",
                    "metric/time claim audit",
                ],
                "forbidden_in_queue": [
                    "download without terms confirmation",
                    "convert without no-leakage audit",
                    "evaluate before conversion",
                    "metric/seconds claim before calibration",
                    "Stage5C execution",
                    "SMC execution",
                ],
            }
        )
    return queue


def _blocked_actions(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for row in _blocked_targets(manifest):
        actions.append(
            {
                "dataset_id": row.get("dataset_id", ""),
                "name": row.get("name", row.get("dataset_id", "")),
                "official_url": row.get("official_url", ""),
                "cf_blockers": list(row.get("cf_blockers", [])),
                "confirmation_blockers": list(row.get("confirmation_blockers", [])),
                "next_action": row.get("next_action", "fill explicit official terms/path/source-identity confirmation before conversion"),
                "conversion_ready": False,
                "queued": False,
            }
        )
    return actions


def _summary(manifest: Mapping[str, Any], queue: list[Mapping[str, Any]], blocked: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "source": SOURCE,
        "manifest_source": manifest.get("source", ""),
        "manifest_generated_at_utc": manifest.get("generated_at_utc", ""),
        "ready_targets_in_manifest": len(_ready_targets(manifest)),
        "blocked_targets_in_manifest": len(blocked),
        "conversion_queue_count": len(queue),
        "conversion_executed": False,
        "evaluation_executed": False,
        "download_executed": False,
        "stage42_ej_is_launcher_only": True,
        "user_action_required_targets": len(blocked),
        "next_if_queue_nonempty": "run a future guarded converter that performs parser/no-leakage/source-CV evaluation; Stage42-EJ does not execute it",
        "next_if_queue_empty": "fill source_terms_confirmation_intake_template_stage42.json and rerun validator before conversion",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "manifest_loaded": bool(payload.get("input_manifest_hash")),
        "ready_targets_scanned": s["ready_targets_in_manifest"] >= 0,
        "blocked_targets_preserved": s["blocked_targets_in_manifest"] == len(payload["blocked_actions"]),
        "ready_targets_queued_only": s["conversion_queue_count"] == s["ready_targets_in_manifest"],
        "blank_intake_refuses_conversion": not (
            s["ready_targets_in_manifest"] == 0 and s["conversion_queue_count"] > 0
        ),
        "no_download_executed": s["download_executed"] is False,
        "no_conversion_executed": s["conversion_executed"] is False,
        "no_evaluation_executed": s["evaluation_executed"] is False,
        "user_action_written": payload["user_action_required_written"] is True,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for value in gates.values() if value)
    total = len(gates)
    verdict = "stage42_ej_guarded_source_conversion_launcher_pass" if passed == total else "stage42_ej_guarded_source_conversion_launcher_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-EJ Guarded Source Conversion Launcher",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_manifest_hash: `{payload['input_manifest_hash']}`",
        f"- gate: `{payload['stage42_ej_gate']['passed']} / {payload['stage42_ej_gate']['total']}`",
        f"- verdict: `{payload['stage42_ej_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in s.items()],
        "",
        "## Conversion Queue",
        "",
    ]
    if payload["conversion_queue"]:
        lines.extend(
            [
                "| dataset | status | local path | source identity |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in payload["conversion_queue"]:
            lines.append(
                f"| `{row['dataset_id']}` | `{row['status']}` | `{row['confirmed_local_path']}` | `{row['source_identity']}` |"
            )
    else:
        lines.append("- No conversion-ready targets. Stage42-EJ refused conversion, as intended.")
    lines.extend(
        [
            "",
            "## Blocked Targets",
            "",
            "| dataset | CF blockers | confirmation blockers | next action |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in payload["blocked_actions"]:
        lines.append(
            f"| `{row['dataset_id']}` | {', '.join(row['cf_blockers']) or 'none'} | "
            f"{', '.join(row['confirmation_blockers']) or 'none'} | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This launcher is intentionally non-executing. It prevents a later conversion stage from mistaking local files, parseability, or technical dry-runs for legal readiness.",
            "- Current queue count is zero because the Stage42-EH intake remains blank and the validator reports no conversion-ready targets.",
            "- If the user fills official terms/path/source identity and the validator later marks a target ready, this launcher will queue it for a future guarded converter; it still will not execute conversion itself.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_ej_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-EJ Guarded Source Conversion",
        "",
        "Stage42-EJ did not download, convert, train, or evaluate data. Conversion is blocked until the validator reports at least one conversion-ready target.",
        "",
        "| dataset | official URL | missing confirmation | source-CV blockers |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["blocked_actions"]:
        lines.append(
            f"| `{row['dataset_id']}` | {row['official_url']} | "
            f"{', '.join(row['confirmation_blockers']) or 'none'} | {', '.join(row['cf_blockers']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "Next safe steps:",
            "",
            "1. Fill `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json` manually after checking official terms and local source identity.",
            "2. Run `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`.",
            "3. Rerun `.venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py`.",
            "4. Only a later guarded converter may execute parser/no-leakage/source-CV work, and only for queued ready targets.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ej_gate"]
    return [
        "# Stage42-EJ Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    return [
        "## Stage42-EJ Guarded Source Conversion Launcher",
        "",
        "- source: `fresh_guarded_source_conversion_launcher_from_stage42_ei_manifest`",
        "- role: reads the validator readiness manifest and creates a non-executing guarded conversion queue.",
        f"- gate: `{payload['stage42_ej_gate']['passed']} / {payload['stage42_ej_gate']['total']}`; verdict `{payload['stage42_ej_gate']['verdict']}`.",
        f"- ready targets: `{s['ready_targets_in_manifest']}`; blocked targets: `{s['blocked_targets_in_manifest']}`; queued conversions: `{s['conversion_queue_count']}`.",
        f"- download/convert/evaluate executed: `{s['download_executed']}` / `{s['conversion_executed']}` / `{s['evaluation_executed']}`.",
        "- Current result preserves the legal blocker: no ready target means no conversion queue and no converted-data claim.",
        "- Boundary: no metric/seconds claim, no Stage5C, no SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_EJ_GUARDED_SOURCE_CONVERSION_LAUNCHER", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-EJ guarded source conversion launcher"
    state["current_verdict"] = payload["stage42_ej_gate"]["verdict"]
    state["stage42_ej_guarded_source_conversion_launcher"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_ej_gate"]["verdict"],
        "gates": f"{payload['stage42_ej_gate']['passed']}/{payload['stage42_ej_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_guarded_source_conversion_launcher(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    manifest = read_json(MANIFEST_JSON, {})
    queue = _build_conversion_queue(manifest)
    blocked = _blocked_actions(manifest)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-EJ",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_manifest_hash": _combined_hash([MANIFEST_JSON]) if MANIFEST_JSON.exists() else "",
        "current_facts": CURRENT_FACTS,
        "input_manifest_path": str(MANIFEST_JSON),
        "input_manifest_source": manifest.get("source", ""),
        "summary": _summary(manifest, queue, blocked),
        "conversion_queue": queue,
        "blocked_actions": blocked,
        "claim_boundary": CLAIM_BOUNDARY,
        "user_action_required_written": True,
    }
    payload["stage42_ej_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_guarded_source_conversion_launcher()
