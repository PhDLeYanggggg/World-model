from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_full_waypoint_bridge_shape_audit import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

HJ_JSON = OUT_DIR / "restricted_metric_time_source_cv_preflight_stage42.json"
HK_JSON = OUT_DIR / "restricted_metric_time_eth_ucy_source_support_stage42.json"
CH_JSON = OUT_DIR / "metric_time_claim_guard_stage42.json"
GQ_JSON = OUT_DIR / "source_terms_package_claim_linter_stage42.json"

REPORT_JSON = OUT_DIR / "restricted_metric_time_post_hk_claim_guard_stage42.json"
REPORT_MD = OUT_DIR / "restricted_metric_time_post_hk_claim_guard_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hl_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_restricted_metric_time_post_hk_claim_guard_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
CURRENT_SUMMARY = Path("README_M3W_CURRENT_DETAILED_SUMMARY_2026_05_27_ZH.md")
A_JOURNAL_GAP = OUT_DIR / "a_journal_gap_stage42.md"
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
    OUT_DIR / "paper_claim_evidence_audit_stage42.md",
    OUT_DIR / "paper_ready_evidence_matrix_stage42.md",
]
SCAN_FILES = [README_RESULTS, M3W_README, CURRENT_SUMMARY, *PAPER_FILES]
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_hl_restricted_metric_time_post_hk_claim_guard"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HL 是 post-HK restricted metric/time claim guard，不下载、不转换、不训练、不评估。",
    "Stage42-HK 只证明 ETH_UCY source support 在 terms 后技术上可修复，不证明 ready-now、converted、evaluated 或 metric/seconds-level 成功。",
    "ETH-Person XML local candidates 仍是 terms-unverified。",
    "restricted metric/time claim 需要用户确认 terms/source identity/path、guarded conversion、no-leakage、source-CV/final test 后才可重新审计。",
    "future endpoint / future waypoint 只可作为 supervised/evaluation labels，不可作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "restricted_metric_time_claim_allowed_now": False,
    "eth_ucy_restricted_metric_time_ready_now": False,
    "eth_ucy_conversion_executed": False,
    "eth_ucy_evaluation_executed": False,
    "download_executed": False,
    "training_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}

UNSAFE_PATTERNS: dict[str, list[str]] = {
    "global_metric_seconds_allowed_true": [
        r"\bglobal[_ -]metric[_ -]claim[_ -]allowed\b\s*[:=]\s*(true|yes|1)\b",
        r"\bglobal[_ -]seconds[_ -]claim[_ -]allowed\b\s*[:=]\s*(true|yes|1)\b",
    ],
    "restricted_metric_time_allowed_true": [
        r"\brestricted[_ -](subset[_ -])?metric[_ -]time[_ -]claim[_ -]allowed[_ -]now\b\s*[:=]\s*(true|yes|1)\b",
        r"\brestricted[_ -]metric[_ -]time[_ -]ready[_ -]now\b\s*[:=]\s*(true|yes|1)\b",
        r"\brestricted\s+metric/time\s+ready\s+now\b\s*[:=]\s*(true|yes|1)\b",
    ],
    "eth_ucy_ready_now_overclaim": [
        r"\beth[_ -]?ucy\b.*\brestricted[_ -]metric[_ -]time\b.*\bready[_ -]now\b\s*[:=]?\s*(true|yes|1|ready|complete|completed)\b",
        r"\beth[_ -]?ucy\b.*\brestricted\s+metric/time\b.*\bready\s+now\b\s*[:=]?\s*(true|yes|1|ready|complete|completed)\b",
        r"\beth[_ -]?ucy\b.*\b(metric|seconds)[-_ ]?(ready|calibrated|converted|evaluated|complete|completed)\b",
    ],
    "eth_person_terms_overclaim": [
        r"\beth[-_ ]?person\b.*\bterms\b.*\b(confirmed|accepted|ready|complete|completed)\b",
        r"\bagent\b.*\b(accepted|confirmed)\b.*\beth[-_ ]?person\b.*\bterms\b",
    ],
    "stage5c_smc_enabled": [
        r"\bstage5c\b.*\b(executed|enabled|ready)\b\s*[:=]?\s*(true|yes|1)\b",
        r"\bsmc\b.*\b(enabled|ready)\b\s*[:=]?\s*(true|yes|1)\b",
    ],
}

BOUNDARY_MARKERS = [
    "false",
    "not ",
    "not_",
    "no ",
    "never",
    "blocked",
    "forbidden",
    "disallowed",
    "requires",
    "until",
    "after terms",
    "preflight",
    "not ready",
    "ready now: `false`",
    "claim allowed now: `false`",
    "不能",
    "不是",
    "未",
    "没有",
    "禁止",
    "不得",
    "不允许",
    "仍需",
    "需要用户",
]


def _gate_passed(payload: Mapping[str, Any], gate_key: str) -> bool:
    gate = payload.get(gate_key, {})
    return bool(gate and gate.get("passed") == gate.get("total") and int(gate.get("total", 0)) > 0)


def _scan_files() -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for path in SCAN_FILES:
        if path.exists() and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _boundary_context(line: str, heading_stack: list[str], prior_lines: list[str]) -> bool:
    context = " ".join([*heading_stack[-3:], *prior_lines[-2:], line]).lower()
    return any(marker in context for marker in BOUNDARY_MARKERS)


def scan_file(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    violations: list[dict[str, Any]] = []
    headings: list[str] = []
    prior_lines: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            headings = headings[: max(level - 1, 0)]
            headings.append(stripped.lstrip("#").strip())
        if not stripped:
            continue
        for check, patterns in UNSAFE_PATTERNS.items():
            if _matches_any(stripped, patterns) and not _boundary_context(stripped, headings, prior_lines):
                violations.append(
                    {
                        "file": str(path),
                        "line": line_number,
                        "check": check,
                        "text": stripped,
                        "heading_context": headings[-3:],
                    }
                )
        prior_lines.append(stripped)
        prior_lines = prior_lines[-4:]
    return violations


def _build_scan_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _scan_files():
        violations = scan_file(path)
        rows.append(
            {
                "file": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size,
                "violation_count": len(violations),
                "violations": violations,
            }
        )
    return rows


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    hj = read_json(HJ_JSON, {})
    hk = read_json(HK_JSON, {})
    ch = read_json(CH_JSON, {})
    gq = read_json(GQ_JSON, {})
    scan_rows = _build_scan_rows()
    hk_summary = dict(hk.get("summary", {}))
    violations = [violation for row in scan_rows for violation in row.get("violations", [])]
    summary = {
        "source": SOURCE,
        "hj_verdict": hj.get("stage42_hj_gate", {}).get("verdict"),
        "hk_verdict": hk.get("stage42_hk_gate", {}).get("verdict"),
        "ch_verdict": ch.get("stage42_ch_gate", {}).get("verdict"),
        "gq_verdict": gq.get("stage42_gq_gate", {}).get("verdict"),
        "files_scanned": len(scan_rows),
        "files_with_violations": sum(1 for row in scan_rows if row.get("violation_count", 0) > 0),
        "violation_count": len(violations),
        "hk_terms_confirmed": bool(hk_summary.get("terms_confirmed")),
        "hk_ready_now": bool(hk_summary.get("restricted_metric_time_ready_now")),
        "hk_conversion_ready_targets_now": int(hk_summary.get("conversion_ready_targets_now", 0) or 0),
        "hk_augmented_sources_after_terms": int(hk_summary.get("augmented_eth_ucy_sources_after_terms", 0) or 0),
        "hk_augmented_t50_windows_after_terms": int(hk_summary.get("augmented_eth_ucy_t50_windows_after_terms", 0) or 0),
        "hk_augmented_t100_windows_after_terms": int(hk_summary.get("augmented_eth_ucy_t100_windows_after_terms", 0) or 0),
        "post_hk_claim_safe": len(violations) == 0,
        "download_executed": False,
        "conversion_executed": False,
        "evaluation_executed": False,
        "training_executed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-HL Restricted Metric/Time Post-HK Claim Guard",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([HJ_JSON, HK_JSON, CH_JSON, GQ_JSON, *SCAN_FILES]),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "hj_gate_passed": _gate_passed(hj, "stage42_hj_gate"),
            "hk_gate_passed": _gate_passed(hk, "stage42_hk_gate"),
            "ch_gate_passed": _gate_passed(ch, "stage42_ch_gate"),
            "gq_gate_passed": _gate_passed(gq, "stage42_gq_gate"),
        },
        "summary": summary,
        "claim_boundary": CLAIM_BOUNDARY,
        "scan_rows": scan_rows,
        "violations": violations,
        "unlock_checklist": [
            {
                "step": "user_terms_confirmation",
                "status": "not_run",
                "required_evidence": "User-confirmed official terms, source identity, and local path for ETH/BIWI, ETH-Person, and UCY candidates.",
            },
            {
                "step": "guarded_conversion",
                "status": "not_run",
                "required_evidence": "Source-specific parser run with causal velocity, train/val/test or source-CV split, and no test endpoint goal construction.",
            },
            {
                "step": "no_leakage_audit",
                "status": "not_run",
                "required_evidence": "No future endpoint input, no central velocity, no test endpoint goals, and no test normalization statistics.",
            },
            {
                "step": "restricted_metric_time_source_cv_eval",
                "status": "not_run",
                "required_evidence": "Fresh source-CV/final-test metrics on converted restricted subset with metric/time calibration provenance.",
            },
            {
                "step": "paper_claim_refresh",
                "status": "not_run",
                "required_evidence": "Claim guard rerun showing restricted subset wording only, not global metric/seconds claim.",
            },
        ],
        "user_action_required": [
            "Confirm official/source terms, source identity, and local paths for UCY/ETH/ETH-Person sources before any restricted metric/time conversion.",
            "Rerun the source terms validator and guarded conversion harness only after that confirmation.",
            "Keep all current M3W claims as protected dataset-local/raw-frame 2.5D until conversion, no-leakage, source-CV, and claim guard all pass.",
        ],
    }
    payload["stage42_hl_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    c = payload["claim_boundary"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "hj_input_passed": payload["inputs"]["hj_gate_passed"] is True,
        "hk_input_passed": payload["inputs"]["hk_gate_passed"] is True,
        "ch_input_passed": payload["inputs"]["ch_gate_passed"] is True,
        "gq_input_passed": payload["inputs"]["gq_gate_passed"] is True,
        "files_scanned": s["files_scanned"] >= 8,
        "no_post_hk_overclaim_found": s["violation_count"] == 0 and s["post_hk_claim_safe"] is True,
        "hk_terms_still_block_ready_now": s["hk_terms_confirmed"] is False
        and s["hk_ready_now"] is False
        and s["hk_conversion_ready_targets_now"] == 0,
        "hk_after_terms_support_recorded": s["hk_augmented_sources_after_terms"] >= 5
        and s["hk_augmented_t50_windows_after_terms"] > 0
        and s["hk_augmented_t100_windows_after_terms"] > 0,
        "unlock_checklist_nonexecuting": all(row["status"] == "not_run" for row in payload["unlock_checklist"]),
        "no_download_conversion_training_eval": not (
            s["download_executed"] or s["conversion_executed"] or s["training_executed"] or s["evaluation_executed"]
        ),
        "global_metric_seconds_blocked": c["global_metric_claim_allowed"] is False
        and c["global_seconds_claim_allowed"] is False
        and c["restricted_metric_time_claim_allowed_now"] is False,
        "eth_ucy_ready_now_blocked": c["eth_ucy_restricted_metric_time_ready_now"] is False
        and c["eth_ucy_conversion_executed"] is False
        and c["eth_ucy_evaluation_executed"] is False,
        "stage5c_false": c["stage5c_executed"] is False,
        "smc_false": c["smc_enabled"] is False,
    }
    passed = sum(1 for value in gates.values() if value)
    total = len(gates)
    verdict = (
        "stage42_hl_restricted_metric_time_post_hk_claim_guard_pass"
        if passed == total
        else "stage42_hl_restricted_metric_time_post_hk_claim_guard_partial"
    )
    return {"source": payload.get("source", SOURCE), "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hl_gate"]
    s = payload["summary"]
    lines = [
        "# Stage42-HL Restricted Metric/Time Post-HK Claim Guard",
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
        f"- files_scanned: `{s['files_scanned']}`",
        f"- files_with_violations: `{s['files_with_violations']}`",
        f"- violation_count: `{s['violation_count']}`",
        f"- HK terms confirmed: `{s['hk_terms_confirmed']}`",
        f"- HK restricted metric/time ready now: `{s['hk_ready_now']}`",
        f"- HK conversion ready targets now: `{s['hk_conversion_ready_targets_now']}`",
        f"- HK augmented after-terms sources: `{s['hk_augmented_sources_after_terms']}`",
        f"- HK augmented after-terms t50/t100 windows: `{s['hk_augmented_t50_windows_after_terms']}` / `{s['hk_augmented_t100_windows_after_terms']}`",
        "",
        "## Scan Results",
        "",
        "| file | bytes | violations |",
        "| --- | ---: | ---: |",
    ]
    for row in payload["scan_rows"]:
        lines.append(f"| `{row['file']}` | {row['size_bytes']} | {row['violation_count']} |")
    lines += [
        "",
        "## Unlock Checklist",
        "",
        "| step | status | required evidence |",
        "| --- | --- | --- |",
    ]
    for row in payload["unlock_checklist"]:
        lines.append(f"| `{row['step']}` | `{row['status']}` | {row['required_evidence']} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- The post-HK paper/README package is claim-safe for the current restricted metric/time boundary.",
        "- ETH_UCY source support is technically repairable after terms, but no restricted metric/time conversion or evaluation is ready now.",
        "- This result is a guardrail/evidence-packaging step, not new metric/time benchmark evidence.",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hl_gate"]
    return [
        "# Stage42-HL Gate",
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
        "# User Action Required: Stage42-HL Restricted Metric/Time Post-HK Guard",
        "",
        "- The package is guarded against overclaiming HK as ready-now metric/time evidence.",
        "- To unlock the next real conversion/evaluation step, confirm official/source terms, source identity, and local paths for UCY/ETH/ETH-Person candidates.",
        "- Then rerun the source terms validator, guarded conversion, no-leakage audit, source-CV/final test, and this claim guard.",
        "",
        "Current status remains: no restricted metric/time claim, no global metric/seconds claim, no Stage5C, and no SMC.",
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hl_gate"]
    s = payload["summary"]
    return [
        "## Stage42-HL Restricted Metric/Time Post-HK Claim Guard",
        "",
        "- source: `fresh_stage42_hl_restricted_metric_time_post_hk_claim_guard`",
        f"- verdict: `{gate['verdict']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- files scanned / violations: `{s['files_scanned']}` / `{s['violation_count']}`.",
        f"- HK after-terms source support: `{s['hk_augmented_sources_after_terms']}` sources, t50/t100 windows `{s['hk_augmented_t50_windows_after_terms']}` / `{s['hk_augmented_t100_windows_after_terms']}`.",
        f"- ready now: `{s['hk_ready_now']}`; conversion ready targets now: `{s['hk_conversion_ready_targets_now']}`.",
        "- conclusion: the paper/README package remains claim-safe after HK; ETH_UCY source support is technically repairable after terms, but restricted metric/time conversion/evaluation remains blocked until user confirmation and guarded rerun.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, CURRENT_SUMMARY, A_JOURNAL_GAP]:
        _replace_section(path, "STAGE42_HL_RESTRICTED_METRIC_TIME_POST_HK_CLAIM_GUARD", lines)


def _refresh_research_state(payload: Mapping[str, Any], verification: Mapping[str, Any] | None = None) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-HL restricted metric/time post-HK claim guard"
    state["current_verdict"] = payload["stage42_hl_gate"]["verdict"]
    state["stage42_hl_restricted_metric_time_post_hk_claim_guard"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_hl_gate"]["verdict"],
        "gates": f"{payload['stage42_hl_gate']['passed']}/{payload['stage42_hl_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "verification": dict(verification or {"status": "pending"}),
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_restricted_metric_time_post_hk_claim_guard(
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
    run_stage42_restricted_metric_time_post_hk_claim_guard()
