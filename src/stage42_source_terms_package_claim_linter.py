from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
GO_JSON = OUT_DIR / "official_source_terms_live_verifier_stage42.json"
GP_JSON = OUT_DIR / "source_terms_paper_claim_guard_stage42.json"

REPORT_JSON = OUT_DIR / "source_terms_package_claim_linter_stage42.json"
REPORT_MD = OUT_DIR / "source_terms_package_claim_linter_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_terms_package_claim_linter_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gq_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
ONE_FILE_SUMMARY = Path("README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_gq_source_terms_package_claim_linter"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-GQ 是 package-wide source/legal claim linter；不下载、不转换、不训练、不评估。",
    "OpenTraj toolkit MIT 许可不能写成 ETH/UCY/TrajNet/AerialMPT 底层数据许可。",
    "用户必须亲自确认 official terms、allowed use、local path、source identity；agent 不能代填 acceptance。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon；不能写成 seconds-level，除非未来 source-specific guard 通过。",
    "dataset-local/raw-frame 不能写成 global metric；restricted source-specific metric/time subset 也必须等 legal conversion 后再审计。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "restricted_metric_time_claim_allowed_now": False,
    "auto_download_allowed_now": False,
    "converted_dataset_claim_allowed": False,
    "download_executed": False,
    "conversion_executed": False,
    "training_executed": False,
    "evaluation_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}

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

SCAN_FILES = [README_RESULTS, M3W_README, ONE_FILE_SUMMARY, *PAPER_FILES]

UNSAFE_PATTERNS: dict[str, list[str]] = {
    "opentraj_license_overclaim": [
        r"opentraj\s+mit\s+(covers|permits|authorizes|grants).*(underlying|eth|ucy|trajnet)",
        r"opentraj.*license.*(covers|permits|authorizes|grants).*(underlying|eth|ucy|trajnet)",
    ],
    "auto_download_overclaim": [
        r"auto[- ]?download\s+allowed\s+now\s*[:=]\s*(true|1|yes)",
        r"auto[- ]?downloadable\s*[:=]\s*(true|1|yes)",
    ],
    "converted_dataset_overclaim": [
        r"converted\s+dataset\s+claim\s+allowed\s*[:=]\s*(true|1|yes)",
        r"(ucy|eth|biwi|trajnet|aerialmpt).*(has been|is)\s+(legally\s+)?converted",
        r"(ucy|eth|biwi|trajnet|aerialmpt).*(has been|is)\s+evaluated\s+data",
    ],
    "metric_time_overclaim": [
        r"restricted\s+metric\s+time\s+claim\s+allowed\s+now\s*[:=]\s*(true|1|yes)",
        r"global\s+metric\s+claim\s+allowed\s*[:=]\s*(true|1|yes)",
        r"global\s+seconds\s+claim\s+allowed\s*[:=]\s*(true|1|yes)",
        r"(ucy|eth|biwi|trajnet|aerialmpt).*(metric/seconds-calibrated|seconds-level|global metric)",
    ],
    "terms_acceptance_overclaim": [
        r"terms\s+(accepted|confirmed)\s+by\s+agent",
        r"agent\s+(accepted|confirmed)\s+.*terms",
    ],
}

BOUNDARY_MARKERS = [
    "do not",
    "not ",
    "not_",
    "no ",
    "cannot",
    "can't",
    "blocked",
    "disallowed",
    "forbidden",
    "candidate",
    "not counted",
    "separate",
    "not treated",
    "not verified",
    "user must",
    "must not",
    "until",
    "requires",
    "仍",
    "不是",
    "不能",
    "不允许",
    "禁止",
    "未",
    "没有",
    "不可",
    "不得",
    "候选",
    "需要用户",
]


def _scan_files() -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in SCAN_FILES:
        if path not in seen and path.exists():
            seen.add(path)
            out.append(path)
    return out


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _is_boundary_context(line: str, heading_stack: list[str], prior_lines: list[str]) -> bool:
    blob = " ".join([*heading_stack[-3:], *prior_lines[-2:], line]).lower()
    return any(marker in blob for marker in BOUNDARY_MARKERS)


def scan_file(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    violations: list[dict[str, Any]] = []
    heading_stack: list[str] = []
    prior_lines: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            heading_stack = heading_stack[: max(level - 1, 0)]
            heading_stack.append(stripped.lstrip("#").strip())
        if not stripped:
            continue
        for check, patterns in UNSAFE_PATTERNS.items():
            if _matches_any(stripped, patterns) and not _is_boundary_context(stripped, heading_stack, prior_lines):
                violations.append(
                    {
                        "file": str(path),
                        "line": line_number,
                        "check": check,
                        "text": stripped,
                        "heading_context": heading_stack[-3:],
                    }
                )
        prior_lines.append(stripped)
        prior_lines = prior_lines[-4:]
    return violations


def _summary(scan_rows: list[Mapping[str, Any]], go: Mapping[str, Any], gp: Mapping[str, Any]) -> dict[str, Any]:
    go_summary = go.get("summary", {})
    gp_summary = gp.get("summary", {})
    files_scanned = _scan_files()
    violations = [row for row in scan_rows if row.get("violation_count", 0) > 0]
    return {
        "source": SOURCE,
        "go_source": go.get("source", ""),
        "go_verdict": go.get("stage42_go_gate", {}).get("verdict", ""),
        "gp_source": gp.get("source", ""),
        "gp_verdict": gp.get("stage42_gp_gate", {}).get("verdict", ""),
        "files_scanned": len(files_scanned),
        "files_with_violations": len(violations),
        "violation_count": sum(int(row.get("violation_count", 0) or 0) for row in scan_rows),
        "underlying_data_license_confirmed": go_summary.get("underlying_data_license_confirmed", 0),
        "auto_download_allowed_now": go_summary.get("auto_download_allowed_now", 0),
        "contract_ready_now": go_summary.get("contract_ready_now", 0),
        "gp_paper_files_refreshed": gp_summary.get("paper_files_refreshed", []),
        "download_executed": False,
        "conversion_executed": False,
        "training_executed": False,
        "evaluation_executed": False,
        "next_required_action": "fix any future source/legal overclaim before treating paper package as claim-safe",
    }


def _build_scan_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _scan_files():
        violations = scan_file(path)
        rows.append(
            {
                "file": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "violation_count": len(violations),
                "violations": violations,
            }
        )
    return rows


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    c = payload["claim_boundary"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "go_loaded": payload.get("input_status", {}).get("go_exists") is True,
        "gp_loaded": payload.get("input_status", {}).get("gp_exists") is True,
        "go_gate_passed": payload.get("go_gate", {}).get("passed") == payload.get("go_gate", {}).get("total"),
        "gp_gate_passed": payload.get("gp_gate", {}).get("passed") == payload.get("gp_gate", {}).get("total"),
        "package_files_scanned": s["files_scanned"] >= 10,
        "no_source_terms_claim_violations": s["violation_count"] == 0,
        "no_license_or_auto_download_claim": s["underlying_data_license_confirmed"] == 0
        and s["auto_download_allowed_now"] == 0
        and s["contract_ready_now"] == 0,
        "no_download_conversion_training_eval": not (
            s["download_executed"] or s["conversion_executed"] or s["training_executed"] or s["evaluation_executed"]
        ),
        "user_action_written": payload.get("user_action_required_written") is True,
        "no_metric_seconds_overclaim": c["global_metric_claim_allowed"] is False
        and c["global_seconds_claim_allowed"] is False
        and c["restricted_metric_time_claim_allowed_now"] is False,
        "stage5c_false": c["stage5c_executed"] is False,
        "smc_false": c["smc_enabled"] is False,
    }
    passed = sum(bool(value) for value in gates.values())
    total = len(gates)
    verdict = (
        "stage42_gq_source_terms_package_claim_linter_pass"
        if passed == total
        else "stage42_gq_source_terms_package_claim_linter_partial"
    )
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-GQ Source Terms Package Claim Linter",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_gq_gate']['passed']} / {payload['stage42_gq_gate']['total']}`",
        f"- verdict: `{payload['stage42_gq_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in s.items()],
        "",
        "## Package Scan",
        "",
        "| file | size | violations |",
        "| --- | ---: | ---: |",
    ]
    for row in payload["scan_rows"]:
        lines.append(f"| `{row['file']}` | {row['size_bytes']} | {row['violation_count']} |")
    if s["violation_count"]:
        lines.extend(["", "## Violations", ""])
        for row in payload["scan_rows"]:
            for violation in row["violations"]:
                lines.append(
                    f"- `{violation['file']}:{violation['line']}` `{violation['check']}`: {violation['text']}"
                )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_gq_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    if payload["summary"]["violation_count"] == 0:
        return [
            "# User Action Required: Stage42-GQ Source Terms Package Claim Linter",
            "",
            "No package-wide source/legal claim violations were found.",
            "",
            "Keep using the same rule: do not claim converted/evaluated/metric/seconds-level source evidence until user-confirmed terms/path/source identity and guarded conversion/no-leakage/source-CV pass.",
        ]
    lines = [
        "# User Action Required: Stage42-GQ Source Terms Package Claim Linter",
        "",
        "Fix these source/legal claim violations before using the paper package:",
        "",
    ]
    for row in payload["scan_rows"]:
        for violation in row["violations"]:
            lines.append(f"- `{violation['file']}:{violation['line']}` `{violation['check']}`")
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_gq_gate"]
    return [
        "# Stage42-GQ Gate",
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
        "## Stage42-GQ Source Terms Package Claim Linter",
        "",
        "- source: `fresh_stage42_gq_source_terms_package_claim_linter`",
        "- role: scans README and Stage42 paper package for source/legal overclaims after GO/GP.",
        f"- gate: `{payload['stage42_gq_gate']['passed']} / {payload['stage42_gq_gate']['total']}`; verdict `{payload['stage42_gq_gate']['verdict']}`.",
        f"- files scanned: `{s['files_scanned']}`; violations: `{s['violation_count']}`.",
        "- No source is license-confirmed, auto-downloadable, conversion-ready, converted, trained, or evaluated by this step.",
        "- Boundary: package-wide claim lint only; not converted data, not metric/seconds evidence, not Stage5C, and not SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, ONE_FILE_SUMMARY]:
        _replace_section(path, "STAGE42_GQ_SOURCE_TERMS_PACKAGE_CLAIM_LINTER", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-GQ source terms package claim linter"
    state["current_verdict"] = payload["stage42_gq_gate"]["verdict"]
    state["stage42_gq_source_terms_package_claim_linter"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_gq_gate"]["verdict"],
        "gates": f"{payload['stage42_gq_gate']['passed']}/{payload['stage42_gq_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_source_terms_package_claim_linter(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    go = read_json(GO_JSON, {})
    gp = read_json(GP_JSON, {})
    scan_rows = _build_scan_rows()
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-GQ",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([GO_JSON, GP_JSON, *_scan_files()]),
        "input_status": {"go_exists": GO_JSON.exists(), "gp_exists": GP_JSON.exists()},
        "go_gate": go.get("stage42_go_gate", {}),
        "gp_gate": gp.get("stage42_gp_gate", {}),
        "current_facts": CURRENT_FACTS,
        "scan_rows": scan_rows,
        "summary": _summary(scan_rows, go, gp),
        "claim_boundary": CLAIM_BOUNDARY,
        "user_action_required_written": True,
    }
    payload["stage42_gq_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_source_terms_package_claim_linter()
