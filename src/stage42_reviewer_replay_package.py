from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "reviewer_replay_package_stage42.json"
REPORT_MD = OUT_DIR / "reviewer_replay_package_stage42.md"
COMMANDS_SH = OUT_DIR / "reviewer_replay_commands_stage42.sh"
GATE_MD = OUT_DIR / "stage42_stage_dm_gate.md"

CX_JSON = OUT_DIR / "evidence_provenance_stage42.json"
CZ_JSON = OUT_DIR / "paper_freeze_candidate_manifest_stage42.json"
CV_JSON = OUT_DIR / "proximity_guard_batch_replay_stage42.json"
DK_JSON = OUT_DIR / "group_consistency_policy_replay_stage42.json"
DL_JSON = OUT_DIR / "group_consistency_runtime_policy_stage42.json"
FU_JSON = OUT_DIR / "module_contribution_ledger_stage42.json"
FV_JSON = OUT_DIR / "claim_boundary_linter_stage42.json"
FW_JSON = OUT_DIR / "source_action_consolidator_stage42.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
RETRO_README = Path("README_M3W_GOAL_RETROSPECTIVE_CURRENT_ZH.md")
GOAL_SUMMARY_README = Path("README_M3W_CURRENT_GOAL_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DM 是 reviewer replay package，不重新训练，不调 threshold。",
    "reviewer replay 只复现 freeze manifest、provenance、runtime replay 和 policy exact-replay 证据。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

REPLAY_COMMANDS = [
    ".venv-pytorch/bin/python run_stage42_replay_proximity_guard_policy.py",
    ".venv-pytorch/bin/python run_stage42_batch_replay_proximity_guard_policy.py",
    ".venv-pytorch/bin/python run_stage42_replay_group_consistency_policy.py",
    ".venv-pytorch/bin/python run_stage42_group_consistency_runtime_policy.py",
    ".venv-pytorch/bin/python run_stage42_module_contribution_ledger.py",
    ".venv-pytorch/bin/python run_stage42_claim_boundary_linter.py",
    ".venv-pytorch/bin/python run_stage42_source_action_consolidator.py",
    ".venv-pytorch/bin/python run_stage42_evidence_provenance_verifier.py",
    ".venv-pytorch/bin/python run_stage42_paper_freeze_candidate_manifest.py",
    (
        ".venv-pytorch/bin/python -m pytest "
        "tests/test_stage42_proximity_guard_policy_replay.py "
        "tests/test_stage42_proximity_guard_batch_replay.py "
        "tests/test_stage42_group_consistency_policy_replay.py "
        "tests/test_stage42_group_consistency_runtime_policy.py "
        "tests/test_stage42_module_contribution_ledger.py "
        "tests/test_stage42_claim_boundary_linter.py "
        "tests/test_stage42_source_action_consolidator.py "
        "tests/test_stage42_evidence_provenance_verifier.py "
        "tests/test_stage42_paper_freeze_candidate_manifest.py"
    ),
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "metric_or_seconds_claim": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _passed_gate(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return bool(gate) and gate.get("passed") == gate.get("total")


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    cx = payload["inputs"]["evidence_provenance"]
    cz = payload["inputs"]["paper_freeze_manifest"]
    cv = payload["inputs"]["proximity_batch_replay"]
    dk = payload["inputs"]["group_consistency_replay"]
    dl = payload["inputs"]["group_consistency_runtime"]
    fu = payload["inputs"]["module_contribution_ledger"]
    fv = payload["inputs"]["claim_boundary_linter"]
    fw = payload["inputs"]["source_action_consolidator"]
    files = payload["required_files"]
    commands = payload["replay_commands"]
    claim = payload["claim_boundary"]
    dl_replay = dl.get("real_batch_replay", {})
    dl_metric = dl_replay.get("metric", {})
    dl_diag = dl_replay.get("diagnostics", {})
    forbidden_terms = ("stage5c", "smc", "train_", "train ")
    gates = {
        "required_inputs_exist": all(row["exists"] for row in files),
        "commands_file_written": payload["commands_file"]["exists"],
        "all_commands_use_arm64_venv": all(cmd.startswith(".venv-pytorch/bin/python") for cmd in commands),
        "minimal_replay_has_no_training_commands": not any(
            any(term in cmd.lower() for term in forbidden_terms) for cmd in commands
        ),
        "cx_provenance_passed": _passed_gate(cx, "stage42_cx_gate"),
        "cz_manifest_passed": _passed_gate(cz, "stage42_cz_gate"),
        "cv_batch_runtime_replay_passed": _passed_gate(cv, "stage42_cv_gate"),
        "dk_group_policy_replay_passed": _passed_gate(dk, "stage42_dk_gate"),
        "dl_runtime_policy_passed": _passed_gate(dl, "stage42_dl_gate"),
        "paper_manifest_candidate_clean": cz.get("freeze_status", {}).get("freeze_status") == "candidate_clean",
        "manifest_hash_recorded": bool(cz.get("manifest_hash")),
        "provenance_artifacts_count_ge_28": int(cx.get("summary", {}).get("artifacts_total", 0)) >= 28,
        "fu_module_ledger_passed": _passed_gate(fu, "stage42_fu_gate"),
        "fv_claim_linter_passed": _passed_gate(fv, "stage42_fv_gate"),
        "fv_claim_linter_zero_violations": int(fv.get("summary", {}).get("violations_total", -1)) == 0,
        "fw_source_action_passed": _passed_gate(fw, "stage42_fw_gate"),
        "fw_source_action_no_conversion_eval": fw.get("claim_boundary", {}).get("download_executed") is False
        and fw.get("claim_boundary", {}).get("conversion_executed") is False
        and fw.get("claim_boundary", {}).get("evaluation_executed") is False,
        "fw_source_action_not_claim_ready": fw.get("summary", {}).get("claim_ready_after_this_stage") is False,
        "group_runtime_exact_replay": dl_replay.get("selected_xy_max_abs_diff") == 0.0
        and dl_replay.get("selected_ade_max_abs_diff") == 0.0
        and dl_replay.get("selected_fde_max_abs_diff") == 0.0
        and dl_replay.get("switch_exact_match") is True,
        "group_runtime_positive_all_t50_hard": dl_metric.get("all_improvement", 0.0) > 0.0
        and dl_metric.get("t50_improvement", 0.0) > 0.0
        and dl_metric.get("hard_failure_improvement", 0.0) > 0.0,
        "group_runtime_t100_raw_reported": "t100_raw_frame_diagnostic_improvement" in dl_metric,
        "group_runtime_near_collision_reduced": dl_diag.get("final_near_005", 1.0) <= dl_diag.get("base_near_005", 0.0),
        "no_metric_seconds_overclaim": claim["metric_or_seconds_claim"] is False,
        "true_3d_overclaim_blocked": claim["true_3d"] is False,
        "foundation_overclaim_blocked": claim["foundation_world_model"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_dm_reviewer_replay_package_pass" if passed == total else "stage42_dm_reviewer_replay_package_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _file_row(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "sha256": _sha256(path) if exists else "",
        "size_bytes": path.stat().st_size if exists else 0,
    }


def _write_commands() -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Stage42-DM reviewer replay sequence.",
        "# This script regenerates/replays evidence only; it does not train models, execute Stage5C, or enable SMC.",
        "",
        *REPLAY_COMMANDS,
        "",
    ]
    COMMANDS_SH.write_text("\n".join(lines), encoding="utf-8")


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    dl_replay = payload["inputs"]["group_consistency_runtime"].get("real_batch_replay", {})
    dl_metric = dl_replay.get("metric", {})
    dl_diag = dl_replay.get("diagnostics", {})
    lines = [
        "# Stage42-DM Reviewer Replay Package",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- package_hash: `{payload['package_hash']}`",
        f"- gate: `{payload['stage42_dm_gate']['passed']} / {payload['stage42_dm_gate']['total']}`",
        f"- verdict: `{payload['stage42_dm_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Replay Commands",
        "",
        f"- commands file: `{payload['commands_file']['path']}`",
        "",
        "```bash",
        *payload["replay_commands"],
        "```",
        "",
        "## Evidence Inputs",
        "",
        "| file | exists | sha256 |",
        "| --- | --- | --- |",
        *[f"| `{row['path']}` | `{row['exists']}` | `{row['sha256']}` |" for row in payload["required_files"]],
        "",
        "## Key Replay Metrics",
        "",
        f"- group runtime rows: `{dl_replay.get('rows')}`",
        f"- switch exact match: `{dl_replay.get('switch_exact_match')}`",
        f"- selected_xy_max_abs_diff: `{dl_replay.get('selected_xy_max_abs_diff')}`",
        f"- selected_ade_max_abs_diff: `{dl_replay.get('selected_ade_max_abs_diff')}`",
        f"- selected_fde_max_abs_diff: `{dl_replay.get('selected_fde_max_abs_diff')}`",
        f"- all improvement: `{dl_metric.get('all_improvement')}`",
        f"- t50 improvement: `{dl_metric.get('t50_improvement')}`",
        f"- t100 raw-frame diagnostic improvement: `{dl_metric.get('t100_raw_frame_diagnostic_improvement')}`",
        f"- hard/failure improvement: `{dl_metric.get('hard_failure_improvement')}`",
        f"- easy degradation: `{dl_metric.get('easy_degradation')}`",
        f"- base near@0.05: `{dl_diag.get('base_near_005')}`",
        f"- final near@0.05: `{dl_diag.get('final_near_005')}`",
        f"- module ledger supported modules: `{payload['inputs']['module_contribution_ledger'].get('summary', {}).get('main_claim_allowed_modules')}`",
        f"- blocked modules: `{payload['inputs']['module_contribution_ledger'].get('summary', {}).get('blocked_or_auxiliary_modules')}`",
        f"- claim linter violations: `{payload['inputs']['claim_boundary_linter'].get('summary', {}).get('violations_total')}`",
        f"- source-action top actions: `{payload['inputs']['source_action_consolidator'].get('summary', {}).get('top_actions')}`",
        f"- source-action conversion_ready_now: `{payload['inputs']['source_action_consolidator'].get('summary', {}).get('conversion_ready_now')}`",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{key}` | `{value}` |" for key, value in payload["stage42_dm_gate"]["gates"].items()],
        "",
        "## Interpretation",
        "",
        "- Stage42-DM gives reviewers a minimal deterministic replay path for the current Stage42 evidence package.",
        "- It replays/freshens artifact checks and policy exact replay; it does not train, tune, or create new metric/seconds/3D/foundation claims.",
        "- The supported claim remains protected dataset-local/raw-frame 2.5D multi-agent world-state evidence.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_dm_gate"]
    return [
        "# Stage42-DM Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{key}` | `{value}` |" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_dm_gate"]
    dl_metric = payload["inputs"]["group_consistency_runtime"].get("real_batch_replay", {}).get("metric", {})
    return [
        "## Stage42-DM Reviewer Replay Package",
        "",
        "- source: `fresh_reviewer_replay_package_from_stage42_runtime_and_manifest_artifacts`",
        "- role: reviewer-facing minimal replay package for provenance, manifest, and runtime policy exact replay.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- commands file: `{payload['commands_file']['path']}`.",
        f"- group-consistency runtime all/t50/t100 raw/hard: `{dl_metric.get('all_improvement')}` / `{dl_metric.get('t50_improvement')}` / `{dl_metric.get('t100_raw_frame_diagnostic_improvement')}` / `{dl_metric.get('hard_failure_improvement')}`.",
        "- This is replay/provenance packaging only: no training, no threshold tuning, no Stage5C, no SMC, no metric/seconds-level claim.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, RETRO_README, GOAL_SUMMARY_README]:
        _replace_section(path, "STAGE42_DM_REVIEWER_REPLAY_PACKAGE", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-DM reviewer replay package"
    state["current_verdict"] = payload["stage42_dm_gate"]["verdict"]
    state["stage42_dm_reviewer_replay_package"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "commands_file": str(COMMANDS_SH),
        "verdict": payload["stage42_dm_gate"]["verdict"],
        "gates": f"{payload['stage42_dm_gate']['passed']}/{payload['stage42_dm_gate']['total']}",
        "package_hash": payload["package_hash"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_reviewer_replay_package() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    _write_commands()
    inputs = {
        "evidence_provenance": read_json(CX_JSON, {}),
        "paper_freeze_manifest": read_json(CZ_JSON, {}),
        "proximity_batch_replay": read_json(CV_JSON, {}),
        "group_consistency_replay": read_json(DK_JSON, {}),
        "group_consistency_runtime": read_json(DL_JSON, {}),
        "module_contribution_ledger": read_json(FU_JSON, {}),
        "claim_boundary_linter": read_json(FV_JSON, {}),
        "source_action_consolidator": read_json(FW_JSON, {}),
    }
    required_files = [_file_row(path) for path in [CX_JSON, CZ_JSON, CV_JSON, DK_JSON, DL_JSON, FU_JSON, FV_JSON, FW_JSON]]
    commands_file = _file_row(COMMANDS_SH)
    payload: dict[str, Any] = {
        "source": "fresh_reviewer_replay_package_from_stage42_runtime_and_manifest_artifacts",
        "stage": "Stage42-DM",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "inputs": inputs,
        "required_files": required_files,
        "commands_file": commands_file,
        "replay_commands": REPLAY_COMMANDS,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    package_hash_material = [row["sha256"] for row in required_files] + [commands_file["sha256"]]
    payload["package_hash"] = hashlib.sha256("|".join(package_hash_material).encode("utf-8")).hexdigest()
    payload["stage42_dm_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_reviewer_replay_package()
