from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_full_waypoint_bridge_shape_audit import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

HF_JSON = OUT_DIR / "teacherless_gate_deployment_contract_stage42.json"
HE_JSON = OUT_DIR / "floor_free_proximity_guard_robustness_stage42.json"
HC_JSON = OUT_DIR / "floor_alternative_gate_stress_stage42.json"

REPORT_JSON = OUT_DIR / "teacherless_claim_linter_stage42.json"
REPORT_MD = OUT_DIR / "teacherless_claim_linter_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hg_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
USER_SUMMARY = Path("README_M3W_USER_DETAILED_SUMMARY_ZH.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
GOAL_SUMMARY = Path("README_M3W_GOAL_FULL_SUMMARY_ZH.md")
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
    OUT_DIR / "paper_ready_evidence_matrix_stage42.md",
]

SOURCE = "fresh_stage42_hg_teacherless_claim_linter"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HG 是 teacherless/floor-free claim linter，不训练、不转换、不调 threshold。",
    "允许表述：teacherless proximity-guarded switch gate with causal floor fallback。",
    "禁止表述：global floor-free neural deployment、causal floor removal、ungated neural deployment。",
    "future endpoint / future waypoint 只可作为 supervised/evaluation labels，不可作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

BOUNDARY_MARKERS = [
    "not ",
    "not_",
    "no ",
    "never",
    "false",
    "forbidden",
    "blocked",
    "rejected",
    "unsupported",
    "unsupported claims",
    "cannot",
    "can't",
    "must not",
    "not allowed",
    "diagnostic",
    "boundary",
    "caveat",
    "only with",
    "with causal floor fallback",
    "requires causal floor",
    "causal floor fallback",
    "protected",
    "not global",
    "not floor-free",
    "不是",
    "不能",
    "不得",
    "不允许",
    "禁止",
    "边界",
    "保护",
    "仍",
    "未",
    "不是全局",
]

CHECKS: dict[str, dict[str, Any]] = {
    "teacherless_as_global_floor_free": {
        "patterns": [
            r"teacherless.*global.*floor[- ]?free",
            r"teacherless.*causal floor removal",
            r"teacherless.*remove.*causal floor",
            r"teacherless.*without.*causal floor",
            r"teacherless.*无.*floor",
        ],
        "description": "Teacherless gate evidence must not be written as global causal-floor removal.",
    },
    "floor_free_deployable_overclaim": {
        "patterns": [
            r"global.*floor[- ]?free.*(deployable|deployment|ready|success|allowed|可部署|成功|允许)",
            r"floor[- ]?free.*global.*(deployable|deployment|ready|success|allowed|可部署|成功|允许)",
            r"floor[- ]?free neural.*(deployable|ready|success|可部署|成功)",
        ],
        "description": "Global floor-free neural deployment remains blocked.",
    },
    "ungated_neural_deployable": {
        "patterns": [
            r"ungated neural.*(deployable|deployment|safe|ready|success|可部署|安全|成功)",
            r"ungated.*world.*model.*(deployable|safe|ready|success|可部署|安全|成功)",
        ],
        "description": "Ungated neural deployment remains blocked.",
    },
    "causal_floor_removal_allowed": {
        "patterns": [
            r"causal floor removal.*(allowed|ready|success|deployable|pass|允许|成功|可部署)",
            r"remove.*causal floor.*(allowed|ready|success|deployable|pass|允许|成功|可部署)",
        ],
        "description": "Causal floor removal remains blocked except slice-bound diagnostic relaxation.",
    },
    "metric_seconds_true3d_foundation": {
        "patterns": [
            r"true[- ]?3d.*(ready|achieved|success|成功|完成)",
            r"foundation world model.*(ready|achieved|success|成功|完成)",
            r"seconds[- ]?level.*(ready|achieved|success|成功|完成)",
            r"global metric.*(ready|achieved|success|成功|完成)",
        ],
        "description": "No true-3D, foundation, metric, or seconds-level overclaim.",
    },
    "stage5c_smc_enabled": {
        "patterns": [
            r"stage5c.*(executed|enabled|ready|执行|启用|完成)",
            r"\bsmc\b.*(enabled|ready|执行|启用|完成)",
        ],
        "description": "Stage5C and SMC remain disabled.",
    },
}


def files_to_scan() -> list[Path]:
    candidates = [
        README_RESULTS,
        M3W_README,
        USER_SUMMARY,
        WORK_SUMMARY,
        GOAL_SUMMARY,
        OUT_DIR / "teacherless_gate_deployment_contract_stage42.md",
        OUT_DIR / "floor_free_proximity_guard_robustness_stage42.md",
        OUT_DIR / "floor_alternative_gate_stress_stage42.md",
        *PAPER_FILES,
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        if path.exists() and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def _matches_any(line: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in patterns)


def _is_boundary_context(line: str, heading_stack: list[str], prior_lines: list[str] | None = None) -> bool:
    context = " ".join([line, *heading_stack[-3:], *(prior_lines or [])]).lower()
    return any(marker in context for marker in BOUNDARY_MARKERS)


def _gate_passed(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return bool(gate and gate.get("passed") == gate.get("total") and int(gate.get("total", 0)) > 0)


def scan_file(path: Path) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    heading_stack: list[str] = []
    prior_lines: list[str] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            heading_stack = heading_stack[: max(level - 1, 0)]
            heading_stack.append(stripped.lstrip("#").strip())
        if not stripped:
            continue
        for check_name, check in CHECKS.items():
            if _matches_any(stripped, check["patterns"]) and not _is_boundary_context(stripped, heading_stack, prior_lines):
                violations.append(
                    {
                        "file": str(path),
                        "line": idx,
                        "check": check_name,
                        "text": stripped,
                        "heading_context": heading_stack[-3:],
                    }
                )
        prior_lines.append(stripped)
        prior_lines = prior_lines[-8:]
    return violations


def _summary_by_check(violations: list[Mapping[str, Any]]) -> dict[str, int]:
    out = {name: 0 for name in CHECKS}
    for row in violations:
        key = str(row["check"])
        out[key] = out.get(key, 0) + 1
    return out


def _decision(contract: Mapping[str, Any], request: str) -> Mapping[str, Any]:
    for row in contract.get("requests", []):
        if row.get("request") == request:
            return row
    return {}


def _contract_ok(hf: Mapping[str, Any]) -> bool:
    contract = hf.get("contract", {})
    return (
        _decision(contract, "teacherless_proximity_guarded_switch_gate").get("status") == "allowed_protected"
        and _decision(contract, "causal_floor_removal").get("allowed") is False
        and _decision(contract, "ungated_neural_or_floor_free_global_deployment").get("allowed") is False
        and _decision(contract, "metric_seconds_true3d_foundation_claim").get("allowed") is False
        and _decision(contract, "stage5c_execution_or_smc_enabled").get("allowed") is False
    )


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    violations = list(payload.get("violations", []))
    by_check = _summary_by_check(violations)
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "hf_contract_loaded": bool(payload.get("hf_contract_ok")),
        "he_input_passed": payload.get("inputs", {}).get("he_gate_passed") is True,
        "hc_input_passed": payload.get("inputs", {}).get("hc_gate_passed") is True,
        "files_scanned": payload.get("summary", {}).get("files_scanned", 0) >= 10,
        "no_teacherless_global_floor_free_overclaim": by_check.get("teacherless_as_global_floor_free", 0) == 0,
        "no_floor_free_deployable_overclaim": by_check.get("floor_free_deployable_overclaim", 0) == 0,
        "no_ungated_neural_deployable_overclaim": by_check.get("ungated_neural_deployable", 0) == 0,
        "no_causal_floor_removal_allowed_overclaim": by_check.get("causal_floor_removal_allowed", 0) == 0,
        "no_metric_seconds_true3d_foundation_overclaim": by_check.get("metric_seconds_true3d_foundation", 0) == 0,
        "no_stage5c_smc_overclaim": by_check.get("stage5c_smc_enabled", 0) == 0,
        "allowed_phrase_present": payload.get("summary", {}).get("allowed_phrase_hits", 0) > 0,
        "required_floor_phrase_present": payload.get("summary", {}).get("causal_floor_fallback_phrase_hits", 0) > 0,
        "stage5c_false": payload.get("claim_boundary", {}).get("stage5c_executed") is False,
        "smc_false": payload.get("claim_boundary", {}).get("smc_enabled") is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_hg_teacherless_claim_linter_pass" if passed == total else "stage42_hg_teacherless_claim_linter_partial"
    return {"source": payload.get("source", SOURCE), "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _text_hits(files: list[Path], phrase: str) -> int:
    hits = 0
    needle = phrase.lower()
    for path in files:
        text = path.read_text(encoding="utf-8").lower()
        hits += text.count(needle)
    return hits


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    hf = read_json(HF_JSON, {})
    he = read_json(HE_JSON, {})
    hc = read_json(HC_JSON, {})
    files = files_to_scan()
    violations: list[dict[str, Any]] = []
    for path in files:
        violations.extend(scan_file(path))
    by_check = _summary_by_check(violations)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-HG",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([HF_JSON, HE_JSON, HC_JSON, *files]),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "hf_gate_passed": _gate_passed(hf, "stage42_hf_gate"),
            "he_gate_passed": _gate_passed(he, "stage42_he_gate"),
            "hc_gate_passed": _gate_passed(hc, "stage42_hc_gate"),
        },
        "hf_contract_ok": _contract_ok(hf),
        "files_scanned": [str(path) for path in files],
        "violations": violations,
        "summary": {
            "files_scanned": len(files),
            "violations_total": len(violations),
            "violations_by_check": by_check,
            "allowed_phrase": "teacherless proximity-guarded switch gate",
            "allowed_phrase_hits": _text_hits(files, "teacherless proximity-guarded switch gate"),
            "causal_floor_fallback_phrase_hits": _text_hits(files, "causal floor fallback"),
            "global_floor_free_claim_allowed": False,
            "ungated_neural_deployment_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "claim_boundary": {
            "allowed_claim": "teacherless proximity-guarded switch gate with causal floor fallback",
            "blocked_claims": [
                "global floor-free neural deployment",
                "causal floor removal",
                "ungated neural deployment",
                "metric/seconds/true-3D/foundation claim",
                "Stage5C execution",
                "SMC readiness",
            ],
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_hg_gate"] = _gate(payload)
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hg_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage42-HG Teacherless / Floor-Free Claim Linter",
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
        f"- files_scanned: `{summary['files_scanned']}`",
        f"- violations_total: `{summary['violations_total']}`",
        f"- allowed_phrase_hits: `{summary['allowed_phrase_hits']}`",
        f"- causal_floor_fallback_phrase_hits: `{summary['causal_floor_fallback_phrase_hits']}`",
        f"- global_floor_free_claim_allowed: `{summary['global_floor_free_claim_allowed']}`",
        f"- ungated_neural_deployment_allowed: `{summary['ungated_neural_deployment_allowed']}`",
        "",
        "## Violations By Check",
        "",
        "| check | count |",
        "| --- | ---: |",
        *[f"| `{key}` | {value} |" for key, value in summary["violations_by_check"].items()],
        "",
        "## Violations",
        "",
    ]
    if payload["violations"]:
        lines += ["| file | line | check | text |", "| --- | ---: | --- | --- |"]
        for row in payload["violations"]:
            text = str(row["text"]).replace("|", "\\|")
            lines.append(f"| `{row['file']}` | {row['line']} | `{row['check']}` | {text} |")
    else:
        lines.append("- None.")
    lines += [
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hg_gate"]
    return [
        "# Stage42-HG Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hg_gate"]
    summary = payload["summary"]
    return [
        "## Stage42-HG Teacherless / Floor-Free Claim Linter",
        "",
        "- source: `fresh_stage42_hg_teacherless_claim_linter`",
        f"- verdict: `{gate['verdict']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- scanned files: `{summary['files_scanned']}`; violations: `{summary['violations_total']}`.",
        "- allowed phrase: `teacherless proximity-guarded switch gate with causal floor fallback`.",
        "- blocked: global floor-free neural deployment, causal floor removal, ungated neural deployment, metric/seconds/true-3D/foundation claims, Stage5C, and SMC.",
        "- role: applies Stage42-HF contract to the paper/README surface; this is not new training or threshold tuning.",
    ]


def _refresh_a_journal_gap(payload: Mapping[str, Any]) -> None:
    lines = [
        "## Stage42-HG Teacherless Claim Boundary Refresh",
        "",
        "- source: `fresh_stage42_hg_teacherless_claim_linter`",
        f"- verdict: `{payload['stage42_hg_gate']['verdict']}`",
        "- New supported wording: `teacherless proximity-guarded switch gate with causal floor fallback`.",
        "- This strengthens the safety-floor study by showing a teacher gate can be removed for the repaired proximity-guarded switch policy.",
        "- It does not support global causal floor removal, ungated neural deployment, metric/seconds-level prediction, true 3D, foundation model, Stage5C, or SMC.",
        "- A-journal gap after HG: still need legal source expansion / source diversity, metric-time calibration, and stronger independently positive scene/goal/interaction or neural dynamics evidence.",
    ]
    _replace_section(OUT_DIR / "a_journal_gap_stage42.md", "STAGE42_HG_TEACHERLESS_CLAIM_BOUNDARY", lines)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, USER_SUMMARY, WORK_SUMMARY, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_HG_TEACHERLESS_CLAIM_LINTER", lines)
    _refresh_a_journal_gap(payload)


def _refresh_research_state(payload: Mapping[str, Any], *, verification: Mapping[str, Any] | None = None) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-HG teacherless claim linter"
    state["current_verdict"] = payload["stage42_hg_gate"]["verdict"]
    state["stage42_hg_teacherless_claim_linter"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_hg_gate"]["verdict"],
        "gates": f"{payload['stage42_hg_gate']['passed']}/{payload['stage42_hg_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "verification": dict(verification or {"status": "pending"}),
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_teacherless_claim_linter(
    *,
    refresh_readmes: bool = True,
    verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload, verification=verification)
    return payload


if __name__ == "__main__":
    run_stage42_teacherless_claim_linter()
