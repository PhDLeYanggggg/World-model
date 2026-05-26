from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
CX_JSON = OUT_DIR / "evidence_provenance_stage42.json"
CY_JSON = OUT_DIR / "worktree_caveat_classifier_stage42.json"
POLICY_JSON = OUT_DIR / "frozen_proximity_guard_composer_policy_stage42_policy.json"
GROUP_POLICY_JSON = OUT_DIR / "frozen_group_consistency_full_waypoint_policy_stage42_policy.json"

REPORT_JSON = OUT_DIR / "paper_freeze_candidate_manifest_stage42.json"
REPORT_MD = OUT_DIR / "paper_freeze_candidate_manifest_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_cz_gate.md"

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

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CZ 是 paper freeze candidate manifest，不重新训练，不调 threshold。",
    "manifest 冻结的是证据包候选，不是 broad foundation/3D/metric/seconds claim。",
    "若仍有 metadata-only caveats，状态必须写 candidate_with_metadata_caveats，而不是 final immutable release。",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_row(path: Path, role: str) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "role": role,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "sha256": _sha256(path) if exists else "",
        "git_status": _git_status(path),
    }


def _manifest_files(cx: Mapping[str, Any]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(path: Path, role: str) -> None:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        files.append(_file_row(path, role))

    for path in PAPER_FILES:
        add(path, "paper_file")
    add(POLICY_JSON, "frozen_runtime_policy_artifact")
    add(GROUP_POLICY_JSON, "frozen_group_consistency_policy_artifact")
    for row in cx.get("artifact_rows", []):
        add(Path(row["json"]), f"evidence_json:{row['claim_area']}")
        add(Path(row["md"]), f"evidence_md:{row['claim_area']}")
        add(Path(row["runner"]), f"runner:{row['claim_area']}")
    add(CX_JSON, "provenance_verifier_json")
    add(CY_JSON, "worktree_caveat_classifier_json")
    return files


def _status(cx: Mapping[str, Any], cy: Mapping[str, Any]) -> dict[str, Any]:
    cx_gate = cx.get("stage42_cx_gate", {})
    cy_gate = cy.get("stage42_cy_gate", {})
    cy_summary = cy.get("summary", {})
    metadata_caveats = int(cy_summary.get("stage42_dirty_files", 0))
    substantive_caveats = int(cy_summary.get("stage42_substantive_dirty_files", 0))
    cx_pass = cx_gate.get("passed") == cx_gate.get("total")
    cy_pass = cy_gate.get("passed") == cy_gate.get("total")
    if not cx_pass or not cy_pass or substantive_caveats:
        freeze_status = "not_freeze_ready"
    elif metadata_caveats:
        freeze_status = "candidate_with_metadata_caveats"
    else:
        freeze_status = "candidate_clean"
    return {
        "cx_pass": cx_pass,
        "cy_pass": cy_pass,
        "metadata_caveats": metadata_caveats,
        "substantive_caveats": substantive_caveats,
        "freeze_status": freeze_status,
        "final_immutable_release": freeze_status == "candidate_clean",
    }


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    status = payload["freeze_status"]
    gate = payload["stage42_cz_gate"]
    return [
        "## Stage42-CZ Paper Freeze Candidate Manifest",
        "",
        "- source: `fresh_freeze_candidate_manifest_from_cx_cy`",
        "- role: hash manifest for the current Stage42 paper evidence candidate.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- freeze status: `{status['freeze_status']}`.",
        f"- final immutable release: `{status['final_immutable_release']}`.",
        f"- files hashed: `{payload['summary']['files_total']}`.",
        f"- metadata caveats: `{status['metadata_caveats']}`; substantive caveats: `{status['substantive_caveats']}`.",
        "- This is a paper evidence freeze candidate under protected dataset-local/raw-frame 2.5D boundaries.",
        "- It is not true 3D, not foundation, not metric/seconds-level, not Stage5C, and not SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, RETRO_README]:
        _replace_section(path, "STAGE42_CZ_PAPER_FREEZE_MANIFEST", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-CZ paper freeze candidate manifest"
    state["current_verdict"] = payload["stage42_cz_gate"]["verdict"]
    state["stage42_cz_paper_freeze_candidate_manifest"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_cz_gate"]["verdict"],
        "gates": f"{payload['stage42_cz_gate']['passed']}/{payload['stage42_cz_gate']['total']}",
        "freeze_status": payload["freeze_status"],
        "summary": payload["summary"],
        "manifest_hash": payload["manifest_hash"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    status = payload["freeze_status"]
    claim = payload["claim_boundary"]
    gates = {
        "cx_gate_passed": status["cx_pass"],
        "cy_gate_passed": status["cy_pass"],
        "all_manifest_files_exist": all(row["exists"] for row in payload["files"]),
        "all_manifest_files_hashed": all(bool(row["sha256"]) for row in payload["files"]),
        "policy_artifact_included": any(row["role"] == "frozen_runtime_policy_artifact" for row in payload["files"]),
        "group_consistency_policy_artifact_included": any(
            row["role"] == "frozen_group_consistency_policy_artifact" for row in payload["files"]
        ),
        "paper_files_included": sum(1 for row in payload["files"] if row["role"] == "paper_file") == len(PAPER_FILES),
        "no_substantive_stage42_caveats": status["substantive_caveats"] == 0,
        "freeze_status_explicit": status["freeze_status"] in {
            "candidate_clean",
            "candidate_with_metadata_caveats",
            "not_freeze_ready",
        },
        "metadata_caveat_not_called_final_release": not (
            status["metadata_caveats"] > 0 and status["final_immutable_release"] is True
        ),
        "no_metric_seconds_overclaim": claim["metric_or_seconds_claim"] is False,
        "true_3d_overclaim_blocked": claim["true_3d"] is False,
        "foundation_overclaim_blocked": claim["foundation_world_model"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_cz_paper_freeze_candidate_manifest_pass" if passed == total else "stage42_cz_paper_freeze_candidate_manifest_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    status = payload["freeze_status"]
    lines = [
        "# Stage42-CZ Paper Freeze Candidate Manifest",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- manifest_hash: `{payload['manifest_hash']}`",
        f"- gate: `{payload['stage42_cz_gate']['passed']} / {payload['stage42_cz_gate']['total']}`",
        f"- verdict: `{payload['stage42_cz_gate']['verdict']}`",
        f"- freeze_status: `{status['freeze_status']}`",
        f"- final_immutable_release: `{status['final_immutable_release']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in payload["summary"].items()],
        "",
        "## Files",
        "",
        "| role | path | size | sha256 | git status |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in payload["files"]:
        lines.append(
            f"| `{row['role']}` | `{row['path']}` | {row['size_bytes']} | `{row['sha256']}` | `{row['git_status']}` |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Stage42-CZ creates a hash manifest for the current paper evidence candidate.",
        "- Because CY records metadata-only Stage42 caveats, this is a candidate manifest, not a final immutable release.",
        "- The supported claim remains protected dataset-local/raw-frame 2.5D multi-agent world-state evidence.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_cz_gate"]
    lines = [
        "# Stage42-CZ Gate",
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


def run_stage42_paper_freeze_candidate_manifest() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cx = read_json(CX_JSON, {})
    cy = read_json(CY_JSON, {})
    files = _manifest_files(cx)
    status = _status(cx, cy)
    manifest_hash = _combined_hash([Path(row["path"]) for row in files if row["exists"]])
    payload: dict[str, Any] = {
        "source": "fresh_freeze_candidate_manifest_from_cx_cy",
        "stage": "Stage42-CZ paper freeze candidate manifest",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "stage42_cx": {
                "path": str(CX_JSON),
                "gate": cx.get("stage42_cx_gate", {}),
                "summary": cx.get("summary", {}),
            },
            "stage42_cy": {
                "path": str(CY_JSON),
                "gate": cy.get("stage42_cy_gate", {}),
                "summary": cy.get("summary", {}),
            },
        },
        "freeze_status": status,
        "files": files,
        "manifest_hash": manifest_hash,
        "summary": {
            "files_total": len(files),
            "paper_files": sum(1 for row in files if row["role"] == "paper_file"),
            "evidence_json_files": sum(1 for row in files if row["role"].startswith("evidence_json")),
            "evidence_md_files": sum(1 for row in files if row["role"].startswith("evidence_md")),
            "runner_files": sum(1 for row in files if row["role"].startswith("runner")),
            "files_with_git_caveat": sum(1 for row in files if row["git_status"] != "clean"),
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_cz_gate"] = _gate(payload)
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    return payload


if __name__ == "__main__":
    run_stage42_paper_freeze_candidate_manifest()
