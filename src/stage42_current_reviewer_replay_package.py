from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
IV_JSON = OUT_DIR / "source_level_row_cache_integration_stage42.json"
IW_JSON = OUT_DIR / "source_level_row_cache_mechanism_audit_stage42.json"
AO_JSON = OUT_DIR / "source_level_incremental_ablation_stage42.json"
JS_JSON = OUT_DIR / "source_context_gain_harm_closure_stage42.json"
JT_JSON = OUT_DIR / "current_module_claim_refresh_stage42.json"

REPORT_JSON = OUT_DIR / "current_reviewer_replay_package_stage42.json"
REPORT_MD = OUT_DIR / "current_reviewer_replay_package_stage42.md"
COMMANDS_SH = OUT_DIR / "current_reviewer_replay_commands_stage42.sh"
GATE_MD = OUT_DIR / "stage42_stage_ju_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE42_JU_CURRENT_REVIEWER_REPLAY_PACKAGE"
SOURCE = "fresh_stage42_ju_current_reviewer_replay_package"

REPLAY_COMMANDS = [
    ".venv-pytorch/bin/python run_stage42_source_level_row_cache_integration.py",
    ".venv-pytorch/bin/python run_stage42_source_level_row_cache_mechanism_audit.py",
    ".venv-pytorch/bin/python run_stage42_source_level_incremental_ablation.py",
    ".venv-pytorch/bin/python run_stage42_source_context_gain_harm_closure.py",
    ".venv-pytorch/bin/python run_stage42_current_module_claim_refresh.py",
    (
        ".venv-pytorch/bin/python -m pytest "
        "tests/test_stage42_source_level_row_cache_integration.py "
        "tests/test_stage42_source_level_row_cache_mechanism_audit.py "
        "tests/test_stage42_source_level_incremental_ablation.py "
        "tests/test_stage42_source_context_gain_harm_closure.py "
        "tests/test_stage42_current_module_claim_refresh.py "
        "tests/test_stage42_current_reviewer_replay_package.py"
    ),
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JU 是 current reviewer replay package，不重新训练、不调 threshold、不执行 Stage5C/SMC。",
    "本包把当前 HEAD 的 IV/IW row-cache 正证据、AO incremental ablation 负证据、JS closure 和 JT claim refresh 串成可复现路径。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "# Stage42-JU current reviewer replay sequence.",
        "# This replays evidence and claim boundaries only; it does not train models, execute Stage5C, or enable SMC.",
        "",
        *REPLAY_COMMANDS,
        "",
    ]
    COMMANDS_SH.write_text("\n".join(lines), encoding="utf-8")
    return _file_row(COMMANDS_SH)


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    inputs = payload["input_status"]
    summary = payload["summary"]
    claim = payload["claim_boundary"]
    commands = payload["replay_commands"]
    forbidden = ("stage5c", "smc", "train_", "train ")
    gates = {
        "required_inputs_exist": all(row["exists"] for row in payload["required_files"]),
        "commands_file_written": payload["commands_file"]["exists"],
        "all_commands_use_arm64_venv": all(cmd.startswith(".venv-pytorch/bin/python") for cmd in commands),
        "replay_commands_have_no_training_or_forbidden_execution": not any(
            any(term in cmd.lower() for term in forbidden) for cmd in commands
        ),
        "iv_row_cache_passed": inputs["iv_verdict"] == "stage42_iv_source_level_row_cache_integration_pass",
        "iw_mechanism_passed": inputs["iw_verdict"] == "stage42_iw_row_cache_mechanism_audit_pass",
        "ao_negative_or_partial_recorded": inputs["ao_verdict"] == "stage42_ao_incremental_component_evidence_partial_or_negative",
        "js_closure_passed": inputs["js_verdict"] == "stage42_js_source_context_gain_harm_closure_pass",
        "jt_claim_refresh_passed": inputs["jt_verdict"] == "stage42_jt_current_module_claim_refresh_pass",
        "row_cache_positive_and_easy_safe": summary["row_cache"]["ade_all"] > 0.0
        and summary["row_cache"]["ade_t50"] > 0.0
        and summary["row_cache"]["ade_hard_failure"] > 0.0
        and summary["row_cache"]["easy_degradation"] <= 0.02,
        "safe_switch_floor_replay_recorded": summary["safe_switch"]["switch_rows"] > 0
        and summary["safe_switch"]["fallback_exact_floor_rate"] >= 0.999,
        "incremental_context_not_overclaimed": summary["ao"]["positive_incremental_context_variants"] == [],
        "blocked_claims_include_context_and_neural": all(
            item in summary["blocked_independent_claims"]
            for item in [
                "scene_goal_independent_main_claim",
                "neighbor_interaction_independent_main_claim",
                "JEPA_downstream_main_claim",
                "Transformer_independent_main_claim",
            ]
        ),
        "allowed_claims_are_protected_not_foundation": any("protected" in item for item in summary["allowed_claims"])
        and not any("foundation" in item.lower() for item in summary["allowed_claims"]),
        "no_metric_seconds_or_3d_overclaim": claim["metric_or_seconds_claim"] is False
        and claim["true_3d"] is False
        and claim["foundation_world_model"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    verdict = "stage42_ju_current_reviewer_replay_package_pass" if passed == total else "stage42_ju_current_reviewer_replay_package_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _input_status(iv: Mapping[str, Any], iw: Mapping[str, Any], ao: Mapping[str, Any], js: Mapping[str, Any], jt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "iv_verdict": iv.get("stage42_iv_gate", {}).get("verdict", ""),
        "iw_verdict": iw.get("stage42_iw_gate", {}).get("verdict", ""),
        "ao_verdict": ao.get("stage42_ao_gate", {}).get("verdict", ""),
        "js_verdict": js.get("stage42_js_gate", {}).get("verdict", ""),
        "jt_verdict": jt.get("stage42_jt_gate", {}).get("verdict", ""),
        "iv_generated_at_utc": iv.get("generated_at_utc", ""),
        "iw_generated_at_utc": iw.get("generated_at_utc", ""),
        "ao_generated_at_utc": ao.get("generated_at_utc", ""),
        "js_generated_at_utc": js.get("generated_at_utc", ""),
        "jt_generated_at_utc": jt.get("generated_at_utc", ""),
    }


def _summary(iv: Mapping[str, Any], iw: Mapping[str, Any], ao: Mapping[str, Any], js: Mapping[str, Any], jt: Mapping[str, Any]) -> dict[str, Any]:
    jt_summary = jt.get("summary", {})
    return {
        "row_cache": jt_summary.get("row_cache", {}),
        "safe_switch": jt_summary.get("safe_switch", {}),
        "waypoint_shape": jt_summary.get("waypoint_shape", {}),
        "ao": jt_summary.get("ao", {}),
        "js": jt_summary.get("js", js.get("summary", {})),
        "allowed_claims": jt_summary.get("allowed_claims", []),
        "blocked_independent_claims": jt_summary.get("blocked_independent_claims", []),
        "source_domains": iv.get("source_level_test_domains", {}),
        "mechanism_rows": iw.get("rows", 0),
        "package_interpretation": (
            "Current replay supports protected source-level full-waypoint row-cache evidence with safe-switch/floor. "
            "It explicitly blocks independent scene/goal, neighbor/interaction, JEPA, Transformer, ungated, metric/time, true-3D and foundation claims."
        ),
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    iv = read_json(IV_JSON, {})
    iw = read_json(IW_JSON, {})
    ao = read_json(AO_JSON, {})
    js = read_json(JS_JSON, {})
    jt = read_json(JT_JSON, {})
    payload: dict[str, Any] = {
        "stage": "Stage42-JU current reviewer replay package",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "input_hash": _combined_hash([IV_JSON, IW_JSON, AO_JSON, JS_JSON, JT_JSON]),
        "current_facts": CURRENT_FACTS,
        "required_files": [_file_row(path) for path in [IV_JSON, IW_JSON, AO_JSON, JS_JSON, JT_JSON]],
        "commands_file": _write_commands(),
        "replay_commands": REPLAY_COMMANDS,
        "input_status": _input_status(iv, iw, ao, js, jt),
        "summary": _summary(iv, iw, ao, js, jt),
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_ju_gate"] = _gate(payload)
    payload["package_hash"] = hashlib.sha256(
        json.dumps(_jsonable({"input_hash": payload["input_hash"], "gate": payload["stage42_ju_gate"], "summary": payload["summary"]}), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ju_gate"]
    s = payload["summary"]
    row = s["row_cache"]
    switch = s["safe_switch"]
    return [
        "# Stage42-JU Current Reviewer Replay Package",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- package_hash: `{payload['package_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
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
        "## Required Inputs",
        "",
        "| file | exists | sha256 |",
        "| --- | ---: | --- |",
        *[f"| `{item['path']}` | `{item['exists']}` | `{item['sha256']}` |" for item in payload["required_files"]],
        "",
        "## Evidence Summary",
        "",
        f"- source domains: `{s['source_domains']}`",
        f"- row-cache ADE all/t50/t100raw/hard: `{float(row.get('ade_all', 0.0)):.6f}` / `{float(row.get('ade_t50', 0.0)):.6f}` / `{float(row.get('ade_t100_raw_frame_diagnostic', 0.0)):.6f}` / `{float(row.get('ade_hard_failure', 0.0)):.6f}`",
        f"- easy degradation: `{float(row.get('easy_degradation', 0.0)):.6f}`",
        f"- t50 bootstrap CI: `{row.get('bootstrap_t50_ci')}`",
        f"- switch_rows: `{switch.get('switch_rows')}`; fallback_exact_floor_rate: `{switch.get('fallback_exact_floor_rate')}`",
        f"- AO positive standalone contexts: `{s['ao'].get('positive_standalone_context_variants')}`",
        f"- AO positive incremental contexts after baseline-family: `{s['ao'].get('positive_incremental_context_variants')}`",
        f"- JS decision: `{s['js'].get('decision')}`",
        "",
        "## Allowed Claims",
        "",
        *[f"- {item}" for item in s["allowed_claims"]],
        "",
        "## Blocked Independent Claims",
        "",
        *[f"- {item}" for item in s["blocked_independent_claims"]],
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
        "",
        "## Interpretation",
        "",
        f"- {s['package_interpretation']}",
        "- This package is for reviewer replay/provenance. It is not a new model training result and does not relax Stage42 claim boundaries.",
    ]


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ju_gate"]
    return [
        "# Stage42-JU Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
    ]


def _section_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ju_gate"]
    row = payload["summary"]["row_cache"]
    return [
        "## Stage42-JU Current Reviewer Replay Package",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`.",
        f"- replay commands: `{payload['commands_file']['path']}`.",
        f"- row-cache ADE all/t50/t100raw/hard: `{float(row.get('ade_all', 0.0)):.6f}` / `{float(row.get('ade_t50', 0.0)):.6f}` / `{float(row.get('ade_t100_raw_frame_diagnostic', 0.0)):.6f}` / `{float(row.get('ade_hard_failure', 0.0)):.6f}`.",
        "- current package locks the latest claim boundary: protected source-level full-waypoint row-cache + safe-switch/floor is supported; independent scene/goal, neighbor/interaction, JEPA, Transformer, ungated, metric/time, true-3D and foundation claims remain blocked.",
        "- public README remains a human project introduction; detailed replay/provenance stays in internal result files.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    block = _section_lines(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, block)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage42_ju_current_reviewer_replay_package"
    state["current_verdict"] = payload["stage42_ju_gate"]["verdict"]
    stage42 = state.setdefault("stage42", {})
    stage42["stage_ju_current_reviewer_replay_package"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "commands_file": str(COMMANDS_SH),
        "verdict": payload["stage42_ju_gate"]["verdict"],
        "gates": f"{payload['stage42_ju_gate']['passed']}/{payload['stage42_ju_gate']['total']}",
        "package_hash": payload["package_hash"],
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    state["last_successful_command"] = "python run_stage42_current_reviewer_replay_package.py"
    generated = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD, COMMANDS_SH]:
        item = str(path)
        if item not in generated:
            generated.append(item)
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": "Stage42-JU",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_ju_gate"]["verdict"],
                    "fresh_run": True,
                    "reviewer_replay_package": True,
                    "allowed_claim_count": len(payload["summary"]["allowed_claims"]),
                    "blocked_claim_count": len(payload["summary"]["blocked_independent_claims"]),
                    "stage5c_executed": False,
                    "smc_enabled": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_stage42_current_reviewer_replay_package(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _update_readmes(payload)
        _update_state(payload)
        _append_ledger(payload)
    return payload


def main() -> None:
    payload = run_stage42_current_reviewer_replay_package(refresh_readmes=True)
    gate = payload["stage42_ju_gate"]
    print(f"Stage42-JU current reviewer replay package: {gate['verdict']} ({gate['passed']}/{gate['total']})")


if __name__ == "__main__":
    main()
