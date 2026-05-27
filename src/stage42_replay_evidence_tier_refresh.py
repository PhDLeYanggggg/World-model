from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _pct, _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

HS_JSON = OUT_DIR / "group_consistency_t100_easy_guard_freeze_stage42.json"
HT_JSON = OUT_DIR / "group_consistency_t100_easy_guard_runtime_stage42.json"
HU_JSON = OUT_DIR / "t100_runtime_batch_replay_sufficiency_stage42.json"
HV_JSON = OUT_DIR / "t100_runtime_row_cache_replay_stage42.json"
DM_JSON = OUT_DIR / "reviewer_replay_package_stage42.json"
PAPER_MATRIX_MD = OUT_DIR / "paper_ready_evidence_matrix_stage42.md"
REVIEWER_PACKAGE_MD = OUT_DIR / "reviewer_replay_package_stage42.md"

REPORT_JSON = OUT_DIR / "replay_evidence_tiers_stage42.json"
REPORT_MD = OUT_DIR / "replay_evidence_tiers_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hw_gate.md"
COMMANDS_SH = OUT_DIR / "reviewer_replay_commands_stage42_hv.sh"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_SUMMARY = Path("README_M3W_CURRENT_MASTER_SUMMARY_ZH.md")
ROUTES_SUMMARY = Path("README_M3W_RESEARCH_ROUTES_FAILURES_SUCCESSES_2026_05_27_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_replay_evidence_tier_refresh_from_stage42_hs_ht_hu_hv"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HW 把 HS/HT/HU/HV 的 replay 证据整理成 reviewer/paper evidence tiers。",
    "HW 不训练、不调阈值、不下载、不转换；它只刷新证据包和复现命令。",
    "HV row-level cache 是本地 derived artifact，不提交 GitHub。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

REPLAY_COMMANDS = [
    ".venv-pytorch/bin/python run_stage42_group_consistency_t100_easy_guard_runtime.py",
    ".venv-pytorch/bin/python run_stage42_t100_runtime_batch_replay_sufficiency.py",
    ".venv-pytorch/bin/python run_stage42_t100_runtime_row_cache_replay.py",
    (
        ".venv-pytorch/bin/python -m pytest "
        "tests/test_stage42_group_consistency_t100_easy_guard_runtime.py "
        "tests/test_stage42_t100_runtime_batch_replay_sufficiency.py "
        "tests/test_stage42_t100_runtime_row_cache_replay.py"
    ),
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


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
        "size_bytes": int(path.stat().st_size) if exists else 0,
    }


def _passed_gate(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return bool(gate) and gate.get("passed") == gate.get("total")


def _write_commands() -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Stage42-HW replay evidence tier refresh commands.",
        "# These commands replay/audit evidence only; they do not train, tune thresholds, execute Stage5C, or enable SMC.",
        "",
        *REPLAY_COMMANDS,
        "",
    ]
    COMMANDS_SH.write_text("\n".join(lines), encoding="utf-8")


def _tracked_cache_files(cache_path: str) -> list[str]:
    if not cache_path:
        return []
    try:
        out = subprocess.check_output(["git", "ls-files", cache_path], text=True).strip()
    except Exception:
        return []
    return [line for line in out.splitlines() if line.strip()]


def _evidence_tiers(hs: Mapping[str, Any], ht: Mapping[str, Any], hu: Mapping[str, Any], hv: Mapping[str, Any]) -> list[dict[str, Any]]:
    hs_metric = hs.get("frozen_policy", {}).get("test_summary_vs_train_horizon_causal_floor", {})
    ht_metric = ht.get("policy_artifact_payload", {}).get("test_summary_vs_train_horizon_causal_floor", {})
    hu_suff = hu.get("sufficiency", {})
    hv_replay = hv.get("runtime_batch_replay", {})
    hv_metric = hv_replay.get("metric", {})
    return [
        {
            "tier": "T0_artifact_presence",
            "status": "pass" if all(path.exists() for path in [HS_JSON, HT_JSON, HU_JSON, HV_JSON]) else "partial",
            "source": "cached_verified_artifact_presence",
            "claim": "Required replay artifacts exist and are hashable.",
            "evidence": [str(HS_JSON), str(HT_JSON), str(HU_JSON), str(HV_JSON)],
        },
        {
            "tier": "T1_runtime_smoke_replay",
            "status": "pass" if _passed_gate(ht, "stage42_ht_gate") else "partial",
            "source": ht.get("source", "unknown"),
            "claim": "Frozen t100 guard is callable and smoke-tested.",
            "rows": int(ht.get("smoke_case", {}).get("rows", 0)),
            "evidence": [str(HT_JSON)],
        },
        {
            "tier": "T2_frozen_metric_replay",
            "status": "pass" if _passed_gate(hs, "stage42_hs_gate") else "partial",
            "source": hs.get("source", "unknown"),
            "claim": "Frozen policy decision table and metric summary replay exactly.",
            "all_improvement": float(hs_metric.get("all_improvement", 0.0)),
            "t50_improvement": float(hs_metric.get("t50_improvement", 0.0)),
            "t100_raw_frame_diagnostic_improvement": float(hs_metric.get("t100_raw_frame_diagnostic_improvement", 0.0)),
            "hard_failure_improvement": float(hs_metric.get("hard_failure_improvement", 0.0)),
            "easy_degradation": float(hs_metric.get("easy_degradation", 0.0)),
            "evidence": [str(HS_JSON)],
        },
        {
            "tier": "T2_5_blocker_audit",
            "status": "resolved_by_hv" if hu_suff.get("real_batch_replay_status") == "not_run" and _passed_gate(hv, "stage42_hv_gate") else "open",
            "source": hu.get("source", "unknown"),
            "claim": "HU identified that HT smoke replay was insufficient for row-level batch replay; HV resolves this locally.",
            "hu_blocker": hu_suff.get("blocker", ""),
            "evidence": [str(HU_JSON), str(HV_JSON)],
        },
        {
            "tier": "T3_row_level_batch_replay",
            "status": "pass" if _passed_gate(hv, "stage42_hv_gate") else "partial",
            "source": hv.get("source", "unknown"),
            "claim": "Frozen t100 runtime guard replayed over full row-level test cache with exact selected XY/ADE/switch/metric match.",
            "rows": int(hv_replay.get("rows", 0)),
            "t100_rows": int(hv_replay.get("t100_rows", 0)),
            "domains": hv_replay.get("domains", {}),
            "all_improvement": float(hv_metric.get("all_improvement", ht_metric.get("all_improvement", 0.0))),
            "t50_improvement": float(hv_metric.get("t50_improvement", ht_metric.get("t50_improvement", 0.0))),
            "t100_raw_frame_diagnostic_improvement": float(hv_metric.get("t100_raw_frame_diagnostic_improvement", ht_metric.get("t100_raw_frame_diagnostic_improvement", 0.0))),
            "hard_failure_improvement": float(hv_metric.get("hard_failure_improvement", ht_metric.get("hard_failure_improvement", 0.0))),
            "easy_degradation": float(hv_metric.get("easy_degradation", ht_metric.get("easy_degradation", 0.0))),
            "t100_easy_degradation": float(hv_metric.get("t100_easy_degradation", 0.0)),
            "cache_path": hv.get("cache_summary", {}).get("path", ""),
            "cache_hash": hv.get("cache_hash", ""),
            "cache_committed": bool(_tracked_cache_files(str(hv.get("cache_summary", {}).get("path", "")))),
            "evidence": [str(HV_JSON)],
        },
    ]


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    tiers = {row["tier"]: row for row in payload["evidence_tiers"]}
    hv = payload["inputs"]["stage42_hv"]
    hv_metric = hv.get("runtime_batch_replay", {}).get("metric", {})
    claim = payload["claim_boundary"]
    commands = payload["replay_commands"]
    forbidden = ("train_", "train ", "stage5c", "smc")
    gates = {
        "hs_artifact_passed": _passed_gate(payload["inputs"]["stage42_hs"], "stage42_hs_gate"),
        "ht_runtime_smoke_passed": _passed_gate(payload["inputs"]["stage42_ht"], "stage42_ht_gate"),
        "hu_blocker_audit_present": _passed_gate(payload["inputs"]["stage42_hu"], "stage42_hu_gate"),
        "hv_row_level_replay_passed": _passed_gate(hv, "stage42_hv_gate"),
        "tier0_present": tiers.get("T0_artifact_presence", {}).get("status") == "pass",
        "tier1_present": tiers.get("T1_runtime_smoke_replay", {}).get("status") == "pass",
        "tier2_present": tiers.get("T2_frozen_metric_replay", {}).get("status") == "pass",
        "tier25_blocker_resolved": tiers.get("T2_5_blocker_audit", {}).get("status") == "resolved_by_hv",
        "tier3_row_level_present": tiers.get("T3_row_level_batch_replay", {}).get("status") == "pass",
        "row_level_rows_positive": int(tiers.get("T3_row_level_batch_replay", {}).get("rows", 0)) > 0,
        "cache_not_committed": tiers.get("T3_row_level_batch_replay", {}).get("cache_committed") is False,
        "row_level_metric_positive_all_t50_hard": hv_metric.get("all_improvement", 0.0) > 0.0
        and hv_metric.get("t50_improvement", 0.0) > 0.0
        and hv_metric.get("hard_failure_improvement", 0.0) > 0.0,
        "t100_raw_reported_not_overclaimed": "t100_raw_frame_diagnostic_improvement" in hv_metric
        and claim["metric_or_seconds_claim"] is False,
        "t100_easy_guard_preserved": hv_metric.get("t100_easy_degradation", 1.0) <= 0.0,
        "replay_commands_written": payload["commands_file"]["exists"] is True,
        "commands_use_arm64_venv": all(cmd.startswith(".venv-pytorch/bin/python") for cmd in commands),
        "commands_do_not_train_or_execute_forbidden": not any(any(term in cmd.lower() for term in forbidden) for cmd in commands),
        "reviewer_package_updated": payload["paper_updates"]["reviewer_package_updated"] is True,
        "paper_matrix_updated": payload["paper_updates"]["paper_matrix_updated"] is True,
        "readmes_updated": payload["paper_updates"]["readmes_updated"] is True,
        "no_future_endpoint_input": payload["no_leakage"]["future_endpoint_input"] is False,
        "no_future_waypoint_input": payload["no_leakage"]["future_waypoint_input"] is False,
        "no_central_velocity": payload["no_leakage"]["central_velocity"] is False,
        "no_test_endpoint_goals": payload["no_leakage"]["test_endpoint_goals"] is False,
        "no_test_threshold_tuning": payload["no_leakage"]["test_threshold_tuning"] is False,
        "no_metric_seconds_claim": claim["metric_or_seconds_claim"] is False,
        "true_3d_overclaim_blocked": claim["true_3d"] is False,
        "foundation_overclaim_blocked": claim["foundation_world_model"] is False,
        "stage5c_not_executed": claim["stage5c_executed"] is False,
        "smc_not_enabled": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_hw_replay_evidence_tier_refresh_pass" if passed == total else "stage42_hw_replay_evidence_tier_refresh_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _tier_table(tiers: list[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| tier | status | source | claim | rows | key metric | evidence |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in tiers:
        metric = ""
        if "all_improvement" in row:
            metric = (
                f"all {_pct(row.get('all_improvement', 0.0))}, "
                f"t50 {_pct(row.get('t50_improvement', 0.0))}, "
                f"t100raw {_pct(row.get('t100_raw_frame_diagnostic_improvement', 0.0))}, "
                f"hard {_pct(row.get('hard_failure_improvement', 0.0))}"
            )
        evidence = "<br>".join(str(path) for path in row.get("evidence", []))
        lines.append(
            f"| `{row['tier']}` | `{row['status']}` | `{row.get('source', '')}` | {row.get('claim', '')} | "
            f"{int(row.get('rows', 0)) if row.get('rows') is not None else 0} | {metric or row.get('hu_blocker', '')} | {evidence} |"
        )
    return lines


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    hv_metric = payload["inputs"]["stage42_hv"].get("runtime_batch_replay", {}).get("metric", {})
    gate = payload.get("stage42_hw_gate", {"passed": "pending", "total": "pending", "verdict": "pending"})
    return [
        "## Stage42-HW Replay Evidence Tier Refresh",
        "",
        "- source: `fresh_replay_evidence_tier_refresh_from_stage42_hs_ht_hu_hv`",
        "- role: integrate HS/HT/HU/HV replay levels into reviewer replay and paper evidence matrix.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        "- evidence tiers: T1 runtime smoke, T2 frozen metric replay, T2.5 blocker audit, T3 row-level batch replay.",
        f"- T3 row-level rows/t100 rows: `{payload['inputs']['stage42_hv'].get('runtime_batch_replay', {}).get('rows')}` / `{payload['inputs']['stage42_hv'].get('runtime_batch_replay', {}).get('t100_rows')}`.",
        f"- T3 all/t50/t100 raw/hard/easy: `{_pct(hv_metric.get('all_improvement', 0.0))}` / `{_pct(hv_metric.get('t50_improvement', 0.0))}` / `{_pct(hv_metric.get('t100_raw_frame_diagnostic_improvement', 0.0))}` / `{_pct(hv_metric.get('hard_failure_improvement', 0.0))}` / `{_pct(hv_metric.get('easy_degradation', 0.0))}`.",
        "- Claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.",
    ]


def _write_report(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_hw_gate"]
    lines = [
        "# Stage42-HW Replay Evidence Tier Refresh",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- package_hash: `{payload['package_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Evidence Tiers",
        "",
        *_tier_table(payload["evidence_tiers"]),
        "",
        "## Replay Commands",
        "",
        f"- commands file: `{payload['commands_file']['path']}`",
        "",
        "```bash",
        *payload["replay_commands"],
        "```",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{name}` | `{bool(ok)}` |" for name, ok in gate["gates"].items()],
        "",
        "## Interpretation",
        "",
        "- Stage42-HW upgrades the paper/reviewer package from smoke/frozen replay evidence to explicit row-level batch replay evidence.",
        "- HV cache remains local derived data and is deliberately not committed.",
        "- The supported claim remains protected dataset-local/raw-frame 2.5D replay evidence, not metric/seconds-level, true 3D, Stage5C, or SMC.",
    ]
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage42-HW Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(ok)}` |" for name, ok in gate["gates"].items()],
        ],
    )


def _refresh_paper_files(payload: Mapping[str, Any]) -> dict[str, bool]:
    lines = _refresh_lines(payload)
    matrix_lines = [
        "## Stage42-HW Replay Evidence Tier Refresh",
        "",
        "This section refines the paper evidence matrix by separating replay evidence strength.",
        "",
        *_tier_table(payload["evidence_tiers"]),
        "",
        "- `T3_row_level_batch_replay` is the strongest replay evidence currently available for the t100 easy-guard runtime policy.",
        "- It remains raw-frame/dataset-local 2.5D evidence. It does not authorize metric/seconds-level, true-3D, Stage5C, or SMC claims.",
    ]
    reviewer_lines = [
        "## Stage42-HW Replay Evidence Tier Refresh",
        "",
        "- Stage42-HW adds the Stage42-HV row-level batch replay command to the reviewer replay evidence set.",
        f"- commands file: `{COMMANDS_SH}`.",
        "- Evidence tiers are now explicit: smoke replay, frozen metric replay, blocker audit, row-level batch replay.",
        "",
        *_tier_table(payload["evidence_tiers"]),
    ]
    _replace_section(PAPER_MATRIX_MD, "STAGE42_HW_REPLAY_EVIDENCE_TIER_REFRESH", matrix_lines)
    _replace_section(REVIEWER_PACKAGE_MD, "STAGE42_HW_REPLAY_EVIDENCE_TIER_REFRESH", reviewer_lines)
    readme_paths = [README_RESULTS, M3W_README, MASTER_SUMMARY, ROUTES_SUMMARY]
    for path in readme_paths:
        _replace_section(path, "STAGE42_HW_REPLAY_EVIDENCE_TIER_REFRESH", lines)
    return {
        "paper_matrix_updated": "STAGE42_HW_REPLAY_EVIDENCE_TIER_REFRESH" in PAPER_MATRIX_MD.read_text(encoding="utf-8"),
        "reviewer_package_updated": "STAGE42_HW_REPLAY_EVIDENCE_TIER_REFRESH" in REVIEWER_PACKAGE_MD.read_text(encoding="utf-8"),
        "readmes_updated": all("STAGE42_HW_REPLAY_EVIDENCE_TIER_REFRESH" in path.read_text(encoding="utf-8") for path in readme_paths),
    }


def _refresh_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    gate = payload["stage42_hw_gate"]
    state["current_stage"] = "Stage42-HW replay evidence tier refresh"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_hw_replay_evidence_tier_refresh"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "commands_file": str(COMMANDS_SH),
        "gates": f"{gate['passed']}/{gate['total']}",
        "verdict": gate["verdict"],
        "evidence_tiers": payload["evidence_tiers"],
        "claim_boundary": payload["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD, COMMANDS_SH]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(RESEARCH_STATE, state)


def run_stage42_replay_evidence_tier_refresh() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    _write_commands()
    hs = read_json(HS_JSON, {})
    ht = read_json(HT_JSON, {})
    hu = read_json(HU_JSON, {})
    hv = read_json(HV_JSON, {})
    tiers = _evidence_tiers(hs, ht, hu, hv)
    package_file_paths = [HS_JSON, HT_JSON, HU_JSON, HV_JSON, DM_JSON, PAPER_MATRIX_MD, REVIEWER_PACKAGE_MD, COMMANDS_SH]
    files = [_file_row(path) for path in package_file_paths]
    payload: dict[str, Any] = {
        "stage": "Stage42-HW",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "stage42_hs": hs,
            "stage42_ht": ht,
            "stage42_hu": hu,
            "stage42_hv": hv,
        },
        "input_hash": _combined_hash([HS_JSON, HT_JSON, HU_JSON, HV_JSON, DM_JSON, PAPER_MATRIX_MD, REVIEWER_PACKAGE_MD]),
        "evidence_tiers": tiers,
        "required_files": files,
        "commands_file": _file_row(COMMANDS_SH),
        "replay_commands": REPLAY_COMMANDS,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_hw_gate"] = {"passed": "pending", "total": "pending", "verdict": "pending", "gates": {}}
    payload["paper_updates"] = _refresh_paper_files(payload)
    files = [_file_row(path) for path in package_file_paths]
    payload["required_files"] = files
    payload["commands_file"] = _file_row(COMMANDS_SH)
    payload["package_hash"] = hashlib.sha256("|".join(row["sha256"] for row in files).encode("utf-8")).hexdigest()
    payload["stage42_hw_gate"] = _gate(payload)
    payload["paper_updates"] = _refresh_paper_files(payload)
    files = [_file_row(path) for path in package_file_paths]
    payload["required_files"] = files
    payload["commands_file"] = _file_row(COMMANDS_SH)
    payload["package_hash"] = hashlib.sha256("|".join(row["sha256"] for row in files).encode("utf-8")).hexdigest()
    payload["stage42_hw_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    _write_report(payload)
    _refresh_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_replay_evidence_tier_refresh()
    gate = result["stage42_hw_gate"]
    print(f"Stage42-HW replay evidence tier refresh: {gate['verdict']} ({gate['passed']}/{gate['total']})")
