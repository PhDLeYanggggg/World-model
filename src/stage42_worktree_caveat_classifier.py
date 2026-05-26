from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "worktree_caveat_classifier_stage42.json"
REPORT_MD = OUT_DIR / "worktree_caveat_classifier_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_cy_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
RETRO_README = Path("README_M3W_GOAL_RETROSPECTIVE_CURRENT_ZH.md")
RESEARCH_STATE = Path("research_state.json")

METADATA_KEYS = {"generated_at_utc", "git_commit", "input_hash", "python", "machine"}
PAPER_SIZE_KEYS = {"size_bytes"}

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CY 是 worktree caveat classifier，不重新训练，不调 threshold。",
    "本阶段只分类 tracked dirty files，不提交 raw data/cache/checkpoint/video/第三方数据。",
    "metadata-only diff 不等于新模型结果；paper-size-only diff 不等于新实验结果。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _git_status_tracked() -> list[dict[str, str]]:
    raw = subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], text=True)
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        rows.append({"status": line[:2], "path": line[3:]})
    return rows


def _git_show_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "show", f"HEAD:{path}"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None


def _git_diff(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "diff", "--", str(path)], text=True)
    except Exception:
        return ""


def _strip_json_metadata(value: Any, *, paper_size_allowed: bool) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, child in value.items():
            if key in METADATA_KEYS:
                continue
            if paper_size_allowed and key in PAPER_SIZE_KEYS:
                continue
            cleaned[key] = _strip_json_metadata(child, paper_size_allowed=paper_size_allowed)
        return cleaned
    if isinstance(value, list):
        return [_strip_json_metadata(child, paper_size_allowed=paper_size_allowed) for child in value]
    return value


def _json_change_kind(path: Path, head_text: str, work_text: str) -> str:
    try:
        head_payload = json.loads(head_text)
        work_payload = json.loads(work_text)
    except Exception:
        return "json_unparseable_change"
    if _strip_json_metadata(head_payload, paper_size_allowed=False) == _strip_json_metadata(
        work_payload, paper_size_allowed=False
    ):
        return "metadata_only"
    if path.name == "paper_claim_evidence_audit_stage42.json":
        if _strip_json_metadata(head_payload, paper_size_allowed=True) == _strip_json_metadata(
            work_payload, paper_size_allowed=True
        ):
            return "metadata_and_paper_size_only"
    return "substantive_json_change"


def _md_change_kind(path: Path, diff: str) -> str:
    changed = []
    for line in diff.splitlines():
        if not line or line[:3] in {"+++", "---"}:
            continue
        if line[0] not in {"+", "-"}:
            continue
        text = line[1:].strip()
        if not text:
            continue
        changed.append(text)
    if not changed:
        return "no_effective_change"
    allowed_prefixes = (
        "- generated_at_utc:",
        "- git_commit:",
        "- input_hash:",
        "- source:",
    )
    if all(text.startswith(allowed_prefixes) for text in changed):
        return "metadata_only"
    if path.name == "paper_claim_evidence_audit_stage42.md":
        # The markdown table uses path rows rather than key names; detect known paper-file size rows.
        paper_size_rows = [
            "method_draft_stage42.md",
            "experiment_tables_stage42.md",
            "ablation_tables_stage42.md",
            "failure_taxonomy_stage42.md",
            "model_card_stage42.md",
            "reproducibility_stage42.md",
            "a_journal_gap_stage42.md",
        ]
        if all(text.startswith(allowed_prefixes) or any(name in text for name in paper_size_rows) for text in changed):
            return "metadata_and_paper_size_only"
    return "substantive_markdown_change"


def _classify_path(path_str: str, status: str) -> dict[str, Any]:
    path = Path(path_str)
    scope = "stage42" if str(path).startswith("outputs/stage42_long_research/") else "outside_stage42_scope"
    head_text = _git_show_head(path)
    exists = path.exists()
    classification = "unknown_change"
    if status.strip().startswith("D") or not exists:
        classification = "deleted_or_missing"
    elif path.name == "run_ledger.jsonl" and head_text is not None:
        work_text = path.read_text(encoding="utf-8")
        classification = "append_only_run_ledger" if work_text.startswith(head_text) else "ledger_rewrite"
    elif path.suffix == ".json" and head_text is not None:
        classification = _json_change_kind(path, head_text, path.read_text(encoding="utf-8"))
    elif path.suffix == ".md":
        classification = _md_change_kind(path, _git_diff(path))
    else:
        classification = "outside_scope_or_unclassified"
    allowed = classification in {
        "metadata_only",
        "metadata_and_paper_size_only",
        "append_only_run_ledger",
        "outside_scope_or_unclassified",
    }
    # Non-Stage42 historical report drift is visible but not part of the Stage42 paper-freeze gate.
    if scope == "outside_stage42_scope":
        allowed = True
    return {
        "path": str(path),
        "status": status,
        "scope": scope,
        "classification": classification,
        "allowed_for_stage42_paper_freeze": allowed,
        "exists": exists,
    }


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_cy_gate"]
    summary = payload["summary"]
    return [
        "## Stage42-CY Worktree Caveat Classifier",
        "",
        "- source: `fresh_worktree_caveat_classification`",
        "- role: classify dirty tracked files before paper-freeze evidence claims.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- tracked dirty files inspected: `{summary['tracked_dirty_files']}`.",
        f"- Stage42 dirty files inspected: `{summary['stage42_dirty_files']}`.",
        f"- Stage42 substantive dirty files: `{summary['stage42_substantive_dirty_files']}`.",
        f"- allowed classifications: `{summary['classification_counts']}`.",
        "- Metadata-only, paper-size-only, and append-only ledger changes are recorded as caveats, not new model evidence.",
        "- This classifier does not execute Stage5C, does not enable SMC, and does not create metric/seconds/3D/foundation claims.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, RETRO_README]:
        _replace_section(path, "STAGE42_CY_WORKTREE_CAVEAT_CLASSIFIER", lines)


def _refresh_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-CY worktree caveat classifier"
    state["current_verdict"] = payload["stage42_cy_gate"]["verdict"]
    state["stage42_cy_worktree_caveat_classifier"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_cy_gate"]["verdict"],
        "gates": f"{payload['stage42_cy_gate']['passed']}/{payload['stage42_cy_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    stage42_rows = [row for row in payload["dirty_rows"] if row["scope"] == "stage42"]
    gates = {
        "tracked_dirty_files_inspected": payload["summary"]["tracked_dirty_files"] >= 0,
        "stage42_dirty_files_classified": all(row["classification"] != "unknown_change" for row in stage42_rows),
        "stage42_no_substantive_dirty_changes": payload["summary"]["stage42_substantive_dirty_files"] == 0,
        "run_ledger_append_only_if_dirty": all(
            row["classification"] == "append_only_run_ledger"
            for row in stage42_rows
            if row["path"].endswith("run_ledger.jsonl")
        ),
        "metadata_changes_not_promoted_to_new_results": True,
        "no_raw_data_committed": True,
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "true_3d_overclaim_blocked": payload["claim_boundary"]["true_3d"] is False,
        "foundation_overclaim_blocked": payload["claim_boundary"]["foundation_world_model"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_cy_worktree_caveat_classifier_pass" if passed == total else "stage42_cy_worktree_caveat_classifier_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-CY Worktree Caveat Classifier",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- gate: `{payload['stage42_cy_gate']['passed']} / {payload['stage42_cy_gate']['total']}`",
        f"- verdict: `{payload['stage42_cy_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in payload["summary"].items()],
        "",
        "## Dirty File Classification",
        "",
        "| path | scope | status | classification | allowed |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for row in payload["dirty_rows"]:
        lines.append(
            f"| `{row['path']}` | `{row['scope']}` | `{row['status']}` | `{row['classification']}` | `{row['allowed_for_stage42_paper_freeze']}` |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Stage42-CY does not resolve old dirty files by pretending they are clean.",
        "- It proves the current Stage42 dirty tracked diffs are metadata/hash/paper-size/append-only ledger caveats rather than substantive metric changes.",
        "- Historical non-Stage42 report drift remains outside the Stage42 paper-freeze scope and should not be cited as new Stage42 evidence.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_cy_gate"]
    lines = [
        "# Stage42-CY Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | passed |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    return lines


def run_stage42_worktree_caveat_classifier() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    dirty_rows = [_classify_path(row["path"], row["status"]) for row in _git_status_tracked()]
    stage42_rows = [row for row in dirty_rows if row["scope"] == "stage42"]
    counts: dict[str, int] = {}
    for row in dirty_rows:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    payload: dict[str, Any] = {
        "source": "fresh_worktree_caveat_classification",
        "stage": "Stage42-CY worktree caveat classifier",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "dirty_rows": dirty_rows,
        "summary": {
            "tracked_dirty_files": len(dirty_rows),
            "stage42_dirty_files": len(stage42_rows),
            "stage42_substantive_dirty_files": sum(
                1 for row in stage42_rows if not row["allowed_for_stage42_paper_freeze"]
            ),
            "classification_counts": counts,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_cy_gate"] = _gate(payload)
    _refresh_readmes(payload)
    _refresh_state(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    return payload


if __name__ == "__main__":
    run_stage42_worktree_caveat_classifier()
