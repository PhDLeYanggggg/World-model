from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
IK_JSON = OUT_DIR / "t50_ensemble_ucy_specialist_integration_stage42.json"
IL_JSON = OUT_DIR / "t50_ucy_specialist_claim_audit_stage42.json"
IM_JSON = OUT_DIR / "t50_source_specialist_policy_freeze_stage42.json"
POLICY_JSON = OUT_DIR / "frozen_t50_source_specialist_policy_stage42.json"

REPORT_JSON = OUT_DIR / "t50_source_specialist_reviewer_replay_stage42.json"
REPORT_MD = OUT_DIR / "t50_source_specialist_reviewer_replay_stage42.md"
COMMANDS_SH = OUT_DIR / "t50_source_specialist_replay_commands_stage42.sh"
GATE_MD = OUT_DIR / "stage42_stage_in_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_README = Path("README_M3W_MASTER_SUMMARY_2026_05_27_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE42_IN_T50_SOURCE_SPECIALIST_REVIEWER_REPLAY"
SOURCE = "cached_verified_stage42_ik_il_im_t50_source_specialist_reviewer_replay"

REPLAY_COMMANDS = [
    ".venv-pytorch/bin/python run_stage42_t50_ensemble_ucy_specialist_integration.py",
    ".venv-pytorch/bin/python run_stage42_t50_ucy_specialist_claim_audit.py",
    ".venv-pytorch/bin/python run_stage42_t50_source_specialist_policy_freeze.py",
    (
        ".venv-pytorch/bin/python -m pytest "
        "tests/test_stage42_t50_ensemble_ucy_specialist_integration.py "
        "tests/test_stage42_t50_ucy_specialist_claim_audit.py "
        "tests/test_stage42_t50_source_specialist_policy_freeze.py "
        "tests/test_stage42_t50_source_specialist_reviewer_replay.py"
    ),
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-IN 是 t50 source-specialist reviewer replay package，不训练新模型，不调 threshold。",
    "IN 只把 Stage42-IK/IL/IM 的 source-specialist composition evidence 固化为可复现命令、hash 和 claim boundary。",
    "future waypoints / endpoints 只作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "metric_or_seconds_claim": False,
    "stage5c_executed": False,
    "smc_enabled": False,
    "independent_new_domain_claim": False,
    "source_specialist_replay_only": True,
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


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _stable_hash(value: Any) -> str:
    blob = json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _gate_pass(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return bool(gate) and gate.get("passed") == gate.get("total")


def _file_row(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "sha256": _sha256(path) if exists else "",
        "size_bytes": path.stat().st_size if exists else 0,
    }


def _write_commands() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Stage42-IN t50 source-specialist reviewer replay.",
        "# Replays evidence only. No training, no threshold search, no Stage5C, no SMC.",
        "",
        *REPLAY_COMMANDS,
        "",
    ]
    COMMANDS_SH.write_text("\n".join(lines), encoding="utf-8")
    return _file_row(COMMANDS_SH)


def _build_replay_summary(ik: Mapping[str, Any], il: Mapping[str, Any], im: Mapping[str, Any]) -> dict[str, Any]:
    metric = im["frozen_policy"]["test_summary_vs_train_horizon_causal_floor"]
    il_summary = il["summary"]
    return {
        "source": "compact_review_replay_from_stage42_ik_il_im",
        "rows": int(metric["rows"]),
        "ade_all": float(metric["ade_all"]),
        "ade_t50": float(metric["ade_t50"]),
        "ade_t50_ci_low": float(metric["ade_t50_ci_low"]),
        "ade_t100_raw_frame_diagnostic": float(metric["ade_t100_raw_frame_diagnostic"]),
        "ade_hard_failure": float(metric["ade_hard_failure"]),
        "ade_easy_degradation": float(metric["ade_easy_degradation"]),
        "fde_t50": float(metric["fde_t50"]),
        "fde_t50_ci_low": float(metric["fde_t50_ci_low"]),
        "switch_rate": float(metric["switch_rate"]),
        "ucy_t50_before": float(il_summary["ucy_delta"]["before_t50"]),
        "ucy_t50_after": float(il_summary["ucy_delta"]["after_t50"]),
        "ucy_t50_delta": float(
            il_summary["ucy_delta"].get(
                "delta_t50",
                float(il_summary["ucy_delta"]["after_t50"]) - float(il_summary["ucy_delta"]["before_t50"]),
            )
        ),
        "non_ucy_max_abs_delta": float(il_summary["non_ucy_max_abs_delta"]),
        "source_rows": ik.get("source_rows", []),
        "domain_rows": ik.get("by_domain", {}),
        "policy_hash": im.get("policy_hash", ""),
        "policy_artifact_sha256": im.get("policy_artifact", {}).get("sha256", ""),
        "supported_claims": il.get("supported_claims", []),
        "blocked_claims": il.get("blocked_claims", []),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    inputs = payload["inputs"]
    replay = payload["reviewer_replay_summary"]
    claim = payload["claim_boundary"]
    commands = payload["replay_commands"]
    im_gate = inputs["stage42_im"].get("stage42_im_gate", {})
    no_leak = inputs["stage42_im"].get("frozen_policy", {}).get("no_leakage", {})
    gates = {
        "required_files_exist": all(row["exists"] for row in payload["required_files"]),
        "required_files_hashed": all(len(row["sha256"]) == 64 for row in payload["required_files"]),
        "commands_file_written": payload["commands_file"]["exists"] and len(payload["commands_file"]["sha256"]) == 64,
        "all_replay_commands_use_arm64_venv": all(cmd.startswith(".venv-pytorch/bin/python") for cmd in commands),
        "no_training_or_threshold_search_commands": not any(
            any(term in cmd.lower() for term in ("train_", "threshold", "stage5c", "smc")) for cmd in commands
        ),
        "ik_gate_passed": _gate_pass(inputs["stage42_ik"], "stage42_ik_gate"),
        "il_gate_passed": _gate_pass(inputs["stage42_il"], "stage42_il_gate"),
        "im_gate_passed": im_gate.get("passed") == im_gate.get("total"),
        "im_compact_replay_exact": inputs["stage42_im"].get("replay", {}).get("metric_summary_exact_replay") is True,
        "policy_hash_recorded": len(replay["policy_hash"]) == 64,
        "policy_artifact_hash_recorded": len(replay["policy_artifact_sha256"]) == 64,
        "ucy_specialist_repair_positive": replay["ucy_t50_after"] > 0.0 and replay["ucy_t50_delta"] > 0.0,
        "non_ucy_unchanged_with_tolerance": replay["non_ucy_max_abs_delta"] <= 1e-6,
        "global_all_t50_hard_positive": replay["ade_all"] > 0.0 and replay["ade_t50"] > 0.0 and replay["ade_hard_failure"] > 0.0,
        "t50_ci_low_positive": replay["ade_t50_ci_low"] > 0.0 and replay["fde_t50_ci_low"] > 0.0,
        "easy_preserved": replay["ade_easy_degradation"] <= 0.02,
        "no_future_or_test_leakage": no_leak.get("future_endpoint_input") is False
        and no_leak.get("future_waypoints_input") is False
        and no_leak.get("central_velocity") is False
        and no_leak.get("test_endpoint_goals") is False
        and no_leak.get("test_threshold_tuning") is False,
        "reviewer_supported_claims_present": len(replay["supported_claims"]) >= 3,
        "reviewer_blocked_claims_present": len(replay["blocked_claims"]) >= 3,
        "source_specialist_scope_only": claim["source_specialist_replay_only"] is True
        and claim["independent_new_domain_claim"] is False,
        "no_metric_seconds_overclaim": claim["metric_or_seconds_claim"] is False,
        "true_3d_overclaim_blocked": claim["true_3d"] is False,
        "foundation_overclaim_blocked": claim["foundation_world_model"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_in_t50_source_specialist_reviewer_replay_pass" if passed == total else "stage42_in_t50_source_specialist_reviewer_replay_partial"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_in_gate"]
    replay = payload["reviewer_replay_summary"]
    lines = [
        "# Stage42-IN T50 Source-Specialist Reviewer Replay Package",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- package_hash: `{payload['package_hash']}`",
        f"- commands_file: `{COMMANDS_SH}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Replay Commands",
        "",
        "```bash",
        *payload["replay_commands"],
        "```",
        "",
        "## Required Files",
        "",
        "| file | exists | sha256 |",
        "| --- | --- | --- |",
        *[f"| `{row['path']}` | `{row['exists']}` | `{row['sha256']}` |" for row in payload["required_files"]],
        "",
        "## Reviewer Replay Summary",
        "",
        f"- rows: `{replay['rows']}`",
        f"- ADE all / t50 / hard: `{replay['ade_all']:.6f}` / `{replay['ade_t50']:.6f}` / `{replay['ade_hard_failure']:.6f}`",
        f"- ADE t50 CI low: `{replay['ade_t50_ci_low']:.6f}`",
        f"- ADE t100 raw-frame diagnostic: `{replay['ade_t100_raw_frame_diagnostic']:.6f}`",
        f"- FDE t50 / CI low: `{replay['fde_t50']:.6f}` / `{replay['fde_t50_ci_low']:.6f}`",
        f"- easy degradation: `{replay['ade_easy_degradation']:.6f}`",
        f"- switch rate: `{replay['switch_rate']:.6f}`",
        f"- UCY t50 before / after / delta: `{replay['ucy_t50_before']:.6f}` / `{replay['ucy_t50_after']:.6f}` / `{replay['ucy_t50_delta']:.6f}`",
        f"- non-UCY max abs delta: `{replay['non_ucy_max_abs_delta']:.12f}`",
        f"- policy hash: `{replay['policy_hash']}`",
        "",
        "## Supported Claims",
        "",
        *[f"- {item}" for item in replay["supported_claims"]],
        "",
        "## Blocked Claims",
        "",
        *[f"- {item}" for item in replay["blocked_claims"]],
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{key}` | `{value}` |" for key, value in gate["gates"].items()],
        "",
        "## Interpretation",
        "",
        "- Stage42-IN turns the IK/IL/IM t50 source-specialist evidence into a compact reviewer replay package.",
        "- It supports source-specialist composition and replay claims only; it is not new training or new independent-domain evidence.",
        "- The result remains protected dataset-local/raw-frame 2.5D evidence, not metric, seconds-level, true 3D, foundation, Stage5C, or SMC evidence.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_in_gate"]
    return [
        "# Stage42-IN Gate",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{key}` | `{value}` |" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    replay = payload["reviewer_replay_summary"]
    gate = payload["stage42_in_gate"]
    return [
        "## Stage42-IN T50 Source-Specialist Reviewer Replay Package",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- commands file: `{COMMANDS_SH}`",
        f"- policy hash: `{replay['policy_hash']}`",
        f"- ADE all / t50 / hard: `{replay['ade_all']:.6f}` / `{replay['ade_t50']:.6f}` / `{replay['ade_hard_failure']:.6f}`",
        f"- UCY t50 before -> after: `{replay['ucy_t50_before']:.6f}` -> `{replay['ucy_t50_after']:.6f}`",
        "- boundary: reviewer replay package for source-specialist t50 evidence only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in (README_RESULTS, M3W_README, MASTER_README):
        _replace_section(path, SECTION, lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    files = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD, COMMANDS_SH]:
        text = str(path)
        if text not in files:
            files.append(text)
    state["current_stage"] = "Stage42-IN t50 source-specialist reviewer replay"
    state["current_verdict"] = payload["stage42_in_gate"]["verdict"]
    state.setdefault("stage42_long_research", {})["stage_in_t50_source_specialist_reviewer_replay"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "commands_file": str(COMMANDS_SH),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_in_gate"]["verdict"],
        "gates": f"{payload['stage42_in_gate']['passed']}/{payload['stage42_in_gate']['total']}",
        "package_hash": payload["package_hash"],
        "metric": payload["reviewer_replay_summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_t50_source_specialist_reviewer_replay() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ik = read_json(IK_JSON, {})
    il = read_json(IL_JSON, {})
    im = read_json(IM_JSON, {})
    commands_file = _write_commands()
    required_files = [_file_row(path) for path in [IK_JSON, IL_JSON, IM_JSON, POLICY_JSON]]
    replay = _build_replay_summary(ik, il, im)
    payload: dict[str, Any] = {
        "stage": "Stage42-IN",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "stage42_ik": ik,
            "stage42_il": il,
            "stage42_im": im,
        },
        "required_files": required_files,
        "commands_file": commands_file,
        "replay_commands": REPLAY_COMMANDS,
        "reviewer_replay_summary": replay,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["package_hash"] = _stable_hash(
        {
            "required_files": required_files,
            "commands": REPLAY_COMMANDS,
            "replay": replay,
            "claim_boundary": CLAIM_BOUNDARY,
        }
    )
    payload["stage42_in_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_t50_source_specialist_reviewer_replay()
