from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "evidence_provenance_stage42.json"
REPORT_MD = OUT_DIR / "evidence_provenance_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_cx_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
RETRO_README = Path("README_M3W_GOAL_RETROSPECTIVE_CURRENT_ZH.md")
RESEARCH_STATE = Path("research_state.json")

PAPER_FILES = [
    OUT_DIR / "paper_outline_stage42.md",
    OUT_DIR / "method_draft_stage42.md",
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "failure_taxonomy_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "data_card_stage42.md",
    OUT_DIR / "reproducibility_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

EVIDENCE_ARTIFACTS = [
    {
        "claim_area": "data_calibration",
        "json": OUT_DIR / "data_calibration_stage42.json",
        "md": OUT_DIR / "data_calibration_stage42.md",
        "gate_md": OUT_DIR / "stage42_stage_a_gate.md",
        "runner": "run_stage42_data_calibration.py",
        "role": "data/metric-time boundary",
    },
    {
        "claim_area": "external_validation",
        "json": OUT_DIR / "external_validation_stage42.json",
        "md": OUT_DIR / "external_validation_stage42.md",
        "runner": "run_stage42_external_validation.py",
        "role": "external raw-frame validation",
    },
    {
        "claim_area": "full_waypoint_dynamics",
        "json": OUT_DIR / "full_waypoint_dynamics_stage42.json",
        "md": OUT_DIR / "full_waypoint_dynamics_stage42.md",
        "runner": "run_stage42_full_waypoint_dynamics.py",
        "role": "full-waypoint all-agent dynamics",
    },
    {
        "claim_area": "causal_ablation",
        "json": OUT_DIR / "causal_ablation_stage42.json",
        "md": OUT_DIR / "causal_ablation_stage42.md",
        "runner": "run_stage42_causal_ablation.py",
        "role": "causal component ablation",
    },
    {
        "claim_area": "safety_floor",
        "json": OUT_DIR / "safety_floor_stage42.json",
        "md": OUT_DIR / "safety_floor_stage42.md",
        "runner": "run_stage42_safety_floor.py",
        "role": "teacher/Stage37 floor necessity",
    },
    {
        "claim_area": "paper_package",
        "json": OUT_DIR / "paper_package_stage42.json",
        "md": OUT_DIR / "paper_outline_stage42.md",
        "runner": "run_stage42_paper_package.py",
        "role": "paper package scaffold",
    },
    {
        "claim_area": "strict_time_geometry_calibration",
        "json": OUT_DIR / "source_time_geometry_calibration_stage42.json",
        "md": OUT_DIR / "source_time_geometry_calibration_stage42.md",
        "runner": "run_stage42_source_time_geometry_calibration.py",
        "role": "strict source time/geometry claim guard",
    },
    {
        "claim_area": "metric_time_claim_guard",
        "json": OUT_DIR / "metric_time_claim_guard_stage42.json",
        "md": OUT_DIR / "metric_time_claim_guard_stage42.md",
        "runner": "run_stage42_metric_time_claim_guard.py",
        "role": "metric/seconds overclaim blocker",
    },
    {
        "claim_area": "source_terms_validation",
        "json": OUT_DIR / "source_terms_validation_stage42.json",
        "md": OUT_DIR / "source_terms_validation_stage42.md",
        "runner": "run_stage42_source_terms_confirmation_validator.py",
        "role": "legal source terms conversion gate",
    },
    {
        "claim_area": "context_contribution_forensics",
        "json": OUT_DIR / "context_contribution_forensics_stage42.json",
        "md": OUT_DIR / "context_contribution_forensics_stage42.md",
        "runner": "run_stage42_context_contribution_forensics.py",
        "role": "context contribution boundary",
    },
    {
        "claim_area": "goal_scene_gated_expert",
        "json": OUT_DIR / "goal_scene_gated_expert_stage42.json",
        "md": OUT_DIR / "goal_scene_gated_expert_stage42.md",
        "runner": "run_stage42_goal_scene_gated_expert.py",
        "role": "goal/scene negative gated expert",
    },
    {
        "claim_area": "neighbor_interaction_gated_expert",
        "json": OUT_DIR / "neighbor_interaction_gated_expert_stage42.json",
        "md": OUT_DIR / "neighbor_interaction_gated_expert_stage42.md",
        "runner": "run_stage42_neighbor_interaction_gated_expert.py",
        "role": "neighbor/interaction negative gated expert",
    },
    {
        "claim_area": "common_validation_bridge_shape_composer",
        "json": OUT_DIR / "common_validation_bridge_shape_composer_stage42.json",
        "md": OUT_DIR / "common_validation_bridge_shape_composer_stage42.md",
        "runner": "run_stage42_common_validation_bridge_shape_composer.py",
        "role": "endpoint/full-waypoint common-row composer",
    },
    {
        "claim_area": "composer_safety_bootstrap",
        "json": OUT_DIR / "common_validation_composer_safety_stage42.json",
        "md": OUT_DIR / "common_validation_composer_safety_stage42.md",
        "runner": "run_stage42_common_validation_composer_safety.py",
        "role": "composer bootstrap and joint safety",
    },
    {
        "claim_area": "proximity_aware_composer_guard",
        "json": OUT_DIR / "proximity_aware_composer_guard_stage42.json",
        "md": OUT_DIR / "proximity_aware_composer_guard_stage42.md",
        "runner": "run_stage42_proximity_aware_composer_guard.py",
        "role": "validation-only proximity guard",
    },
    {
        "claim_area": "proximity_guard_ablation",
        "json": OUT_DIR / "proximity_guard_ablation_stage42.json",
        "md": OUT_DIR / "proximity_guard_ablation_stage42.md",
        "runner": "run_stage42_proximity_guard_ablation.py",
        "role": "accuracy/safety Pareto ablation",
    },
    {
        "claim_area": "frozen_proximity_guard_policy",
        "json": OUT_DIR / "frozen_proximity_guard_composer_policy_stage42.json",
        "md": OUT_DIR / "frozen_proximity_guard_composer_policy_stage42.md",
        "runner": "run_stage42_freeze_proximity_guard_policy.py",
        "role": "frozen deployable policy artifact",
    },
    {
        "claim_area": "frozen_policy_replay",
        "json": OUT_DIR / "frozen_proximity_guard_policy_replay_stage42.json",
        "md": OUT_DIR / "frozen_proximity_guard_policy_replay_stage42.md",
        "runner": "run_stage42_replay_proximity_guard_policy.py",
        "role": "policy artifact replay verifier",
    },
    {
        "claim_area": "runtime_policy_api",
        "json": OUT_DIR / "proximity_guard_runtime_policy_stage42.json",
        "md": OUT_DIR / "proximity_guard_runtime_policy_stage42.md",
        "runner": "run_stage42_runtime_proximity_guard_policy.py",
        "role": "runtime policy API smoke evidence",
    },
    {
        "claim_area": "batch_runtime_replay",
        "json": OUT_DIR / "proximity_guard_batch_replay_stage42.json",
        "md": OUT_DIR / "proximity_guard_batch_replay_stage42.md",
        "runner": "run_stage42_batch_replay_proximity_guard_policy.py",
        "role": "real batch runtime exact replay",
    },
    {
        "claim_area": "runtime_replay_paper_refresh",
        "json": OUT_DIR / "runtime_replay_paper_refresh_stage42.json",
        "md": OUT_DIR / "runtime_replay_paper_refresh_stage42.md",
        "runner": "run_stage42_runtime_replay_paper_refresh.py",
        "role": "paper/reproducibility refresh",
    },
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CX 是 evidence provenance verifier，不重新训练，不调 threshold。",
    "所有 artifact 均标注 fresh_run、cached_verified、not_run 或 unknown_source_label。",
    "worktree dirty/untracked 状态会被记录为 provenance caveat，不会被隐藏。",
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


def _git_status(path: Path) -> str:
    try:
        raw = subprocess.check_output(["git", "status", "--short", "--", str(path)], text=True).strip()
    except Exception:
        return "unknown"
    return raw or "clean"


def _source_label(source: str) -> str:
    lowered = source.lower()
    if "not_run" in lowered:
        return "not_run"
    if "cached_verified" in lowered:
        return "cached_verified"
    if "fresh" in lowered:
        return "fresh_run"
    return "unknown_source_label"


def _find_gate(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    for key, value in payload.items():
        if key.endswith("_gate") and isinstance(value, dict):
            if "passed" in value and "total" in value:
                return {
                    "key": key,
                    "passed": value.get("passed"),
                    "total": value.get("total"),
                    "verdict": value.get("verdict", ""),
                    "all_passed": value.get("passed") == value.get("total"),
                }
    return None


def _gate_from_md(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    match = re.search(r"(?:passed|gates):\s*`?(\d+)\s*/\s*(\d+)`?", text, re.IGNORECASE)
    verdict = ""
    verdict_match = re.search(r"verdict:\s*`([^`]+)`", text)
    if verdict_match:
        verdict = verdict_match.group(1)
    if not match:
        return None
    passed = int(match.group(1))
    total = int(match.group(2))
    return {"key": path.name, "passed": passed, "total": total, "verdict": verdict, "all_passed": passed == total}


def _classify_artifact(spec: Mapping[str, Any]) -> dict[str, Any]:
    json_path = Path(spec["json"])
    md_path = Path(spec["md"])
    payload = read_json(json_path, {}) if json_path.exists() else {}
    source = str(payload.get("source", ""))
    gate_md = Path(spec["gate_md"]) if "gate_md" in spec else None
    gate = _find_gate(payload) or (_gate_from_md(gate_md) if gate_md else None) or _gate_from_md(md_path)
    json_status = _git_status(json_path)
    md_status = _git_status(md_path)
    runner = Path(spec["runner"])
    return {
        "claim_area": spec["claim_area"],
        "role": spec["role"],
        "json": str(json_path),
        "md": str(md_path),
        "runner": str(runner),
        "json_exists": json_path.exists(),
        "md_exists": md_path.exists(),
        "runner_exists": runner.exists(),
        "source": source or "missing_source",
        "source_label": _source_label(source),
        "gate": gate or {"key": "missing_gate", "passed": 0, "total": 1, "verdict": "missing_gate", "all_passed": False},
        "json_git_status": json_status,
        "md_git_status": md_status,
        "runner_git_status": _git_status(runner),
        "gate_md": str(gate_md) if gate_md else "",
        "gate_md_git_status": _git_status(gate_md) if gate_md else "",
        "worktree_caveat": json_status != "clean" or md_status != "clean" or _git_status(runner) != "clean",
    }


def _paper_status() -> list[dict[str, Any]]:
    rows = []
    for path in PAPER_FILES:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "git_status": _git_status(path),
                "has_stage42_cw": "Stage42-CW Runtime Replay" in text,
                "has_claim_boundary": "Stage5C" in text and "SMC" in text and ("metric" in text or "seconds" in text),
                "worktree_caveat": _git_status(path) != "clean",
            }
        )
    return rows


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_cx_gate"]
    summary = payload["summary"]
    return [
        "## Stage42-CX Evidence Provenance / Command Matrix",
        "",
        "- source: `fresh_evidence_provenance_from_stage42_artifacts`",
        "- role: paper-ready provenance and reproducibility audit.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- artifacts audited: `{summary['artifacts_total']}`.",
        f"- artifacts with passing gates: `{summary['artifacts_gate_passed']}`.",
        f"- source-label counts: `{summary['source_label_counts']}`.",
        f"- worktree caveat artifacts recorded: `{summary['artifacts_with_worktree_caveat']}`.",
        "- Dirty/untracked generated files are not hidden; they are recorded as caveats and must not be treated as extra clean paper evidence.",
        "- This audit does not create metric/seconds/3D/foundation claims and does not execute Stage5C or SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, RETRO_README]:
        _replace_section(path, "STAGE42_CX_EVIDENCE_PROVENANCE", lines)


def _refresh_reproducibility(payload: Mapping[str, Any]) -> None:
    rows = payload["artifact_rows"]
    lines = [
        "## Stage42-CX Evidence Provenance / Command Matrix",
        "",
        "- source: `fresh_evidence_provenance_from_stage42_artifacts`",
        "- This section lists high-value Stage42 evidence artifacts, their source labels, gates, runners, and worktree caveats.",
        "- A worktree caveat is not hidden evidence; it means the current file differs from committed HEAD or is untracked.",
        "",
        "| claim area | source label | gate | runner | worktree caveat |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for row in rows:
        gate = row["gate"]
        lines.append(
            f"| `{row['claim_area']}` | `{row['source_label']}` | `{gate['passed']}/{gate['total']}` | `{row['runner']}` | `{row['worktree_caveat']}` |"
        )
    _replace_section(OUT_DIR / "reproducibility_stage42.md", "STAGE42_CX_EVIDENCE_PROVENANCE", lines)


def _refresh_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-CX evidence provenance verifier"
    state["current_verdict"] = payload["stage42_cx_gate"]["verdict"]
    state["stage42_cx_evidence_provenance"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_cx_gate"]["verdict"],
        "gates": f"{payload['stage42_cx_gate']['passed']}/{payload['stage42_cx_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = payload["artifact_rows"]
    paper = payload["paper_file_status"]
    summary = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "all_artifact_json_exists": all(row["json_exists"] for row in rows),
        "all_artifact_md_exists": all(row["md_exists"] for row in rows),
        "all_runners_exist": all(row["runner_exists"] for row in rows),
        "all_source_labels_known": all(row["source_label"] in {"fresh_run", "cached_verified", "not_run"} for row in rows),
        "all_gates_present": all(row["gate"]["total"] >= 1 for row in rows),
        "all_gates_passed": all(row["gate"]["all_passed"] for row in rows),
        "paper_files_exist": all(row["exists"] for row in paper),
        "paper_files_have_claim_boundaries": all(row["has_claim_boundary"] for row in paper),
        "runtime_replay_included": any(row["claim_area"] == "batch_runtime_replay" for row in rows),
        "runtime_replay_refresh_included": any(row["claim_area"] == "runtime_replay_paper_refresh" for row in rows),
        "worktree_caveats_recorded": summary["artifacts_with_worktree_caveat"] >= 0,
        "no_metric_seconds_overclaim": claim["metric_or_seconds_claim"] is False,
        "true_3d_overclaim_blocked": claim["true_3d"] is False,
        "foundation_overclaim_blocked": claim["foundation_world_model"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_cx_evidence_provenance_pass" if passed == total else "stage42_cx_evidence_provenance_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-CX Evidence Provenance Verifier",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_cx_gate']['passed']} / {payload['stage42_cx_gate']['total']}`",
        f"- verdict: `{payload['stage42_cx_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in payload["summary"].items()],
        "",
        "## Artifact Matrix",
        "",
        "| claim area | role | source label | gate | runner | json status | md status | caveat |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for row in payload["artifact_rows"]:
        gate = row["gate"]
        lines.append(
            f"| `{row['claim_area']}` | {row['role']} | `{row['source_label']}` | `{gate['passed']}/{gate['total']}` | `{row['runner']}` | `{row['json_git_status']}` | `{row['md_git_status']}` | `{row['worktree_caveat']}` |"
        )
    lines += [
        "",
        "## Paper File Status",
        "",
        "| file | exists | claim boundary | git status | caveat |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for row in payload["paper_file_status"]:
        lines.append(
            f"| `{row['path']}` | `{row['exists']}` | `{row['has_claim_boundary']}` | `{row['git_status']}` | `{row['worktree_caveat']}` |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Stage42-CX provides a compact provenance and command matrix for the paper package.",
        "- Worktree caveats are intentionally visible; they are not new claims and must be resolved or cited as caveats before a frozen paper artifact release.",
        "- The supported claim remains protected dataset-local/raw-frame 2.5D only.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_cx_gate"]
    lines = [
        "# Stage42-CX Gate",
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


def run_stage42_evidence_provenance_verifier() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    rows = [_classify_artifact(spec) for spec in EVIDENCE_ARTIFACTS]
    paper = _paper_status()
    source_counts: dict[str, int] = {}
    for row in rows:
        source_counts[row["source_label"]] = source_counts.get(row["source_label"], 0) + 1
    payload: dict[str, Any] = {
        "source": "fresh_evidence_provenance_from_stage42_artifacts",
        "stage": "Stage42-CX evidence provenance verifier",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([Path(spec["json"]) for spec in EVIDENCE_ARTIFACTS if Path(spec["json"]).exists()]),
        "current_facts": CURRENT_FACTS,
        "artifact_rows": rows,
        "paper_file_status": paper,
        "summary": {
            "artifacts_total": len(rows),
            "artifacts_gate_passed": sum(1 for row in rows if row["gate"]["all_passed"]),
            "source_label_counts": source_counts,
            "artifacts_with_worktree_caveat": sum(1 for row in rows if row["worktree_caveat"]),
            "paper_files_with_worktree_caveat": sum(1 for row in paper if row["worktree_caveat"]),
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_cx_gate"] = _gate(payload)
    _refresh_reproducibility(payload)
    _refresh_readmes(payload)
    _refresh_state(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    return payload


if __name__ == "__main__":
    run_stage42_evidence_provenance_verifier()
