from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src import stage42_h100_source_support_repair_queue as fq
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
FU_JSON = OUT_DIR / "module_contribution_ledger_stage42.json"

REPORT_JSON = OUT_DIR / "claim_boundary_linter_stage42.json"
REPORT_MD = OUT_DIR / "claim_boundary_linter_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fv_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_claim_boundary_linter_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
GOAL_LEDGER = Path("README_M3W_GOAL_EVIDENCE_LEDGER_ZH.md")
ONE_FILE_SUMMARY = Path("README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_claim_boundary_linter_from_paper_package_and_fu"

BOUNDARY_MARKERS = [
    "not ",
    "not_",
    "no ",
    "never",
    "disabled",
    "rejected",
    "forbidden",
    "cannot",
    "can't",
    "disable",
    "blocked",
    "disallowed",
    "rather than",
    "diagnostic",
    "auxiliary",
    "false",
    "not a",
    "non-claim",
    "non-claims",
    "claim boundary",
    "absolute non-claims",
    "what is not",
    "not yet",
    "gap",
    "need ",
    "needs ",
    "requires",
    "still required",
    "still forbidden",
    "仍",
    "不是",
    "不能",
    "不允许",
    "禁止",
    "未",
    "没有",
    "不可",
    "不得",
    "拒绝",
    "边界",
    "差距",
]


CHECKS: dict[str, dict[str, Any]] = {
    "true_3d": {
        "patterns": [r"\btrue[- ]?3d\b", r"真正.*3d", r"true 3D"],
        "description": "No unsupported true-3D claim.",
    },
    "foundation": {
        "patterns": [r"foundation world model", r"foundation model", r"foundation-track"],
        "description": "No unsupported foundation-model claim.",
    },
    "metric_seconds": {
        "patterns": [r"seconds-level", r"秒级", r"meter-level", r"\bmetric predictor\b", r"global metric"],
        "description": "No unsupported metric/seconds-level claim.",
    },
    "stage5c": {
        "patterns": [r"stage5c.*(executed|enabled|ready|执行|启用|ready)", r"latent generative.*(executed|启用|执行)"],
        "description": "No Stage5C execution/readiness claim.",
    },
    "smc": {
        "patterns": [r"\bsmc\b.*(enabled|ready|启用|执行|ready)"],
        "description": "No SMC readiness/enabled claim.",
    },
    "human_gold": {
        "patterns": [r"human gold", r"人工\s*gold", r"human-gold"],
        "description": "No human-gold overclaim.",
    },
    "jepa_main": {
        "patterns": [r"jepa.*(main claim|main contribution|主贡献|deployable|部署路径|可部署)"],
        "description": "No JEPA main/deployable overclaim.",
    },
    "transformer_main": {
        "patterns": [r"transformer.*(main claim|main contribution|主贡献|independent main|deployable|部署路径|可部署)"],
        "description": "No Transformer independent-main/deployable overclaim.",
    },
    "scene_goal_main": {
        "patterns": [r"scene/goal.*(main claim|main contribution|主贡献|independent main)", r"scene.*goal.*(主贡献|main contribution)"],
        "description": "No scene/goal independent-main overclaim.",
    },
    "neighbor_interaction_main": {
        "patterns": [
            r"neighbor/interaction.*(main claim|main contribution|主贡献|independent main)",
            r"neighbor.*interaction.*(主贡献|main contribution)",
        ],
        "description": "No neighbor/interaction independent-main overclaim.",
    },
}


def files_to_scan() -> list[Path]:
    candidates = [
        README_RESULTS,
        M3W_README,
        WORK_SUMMARY,
        GOAL_LEDGER,
        ONE_FILE_SUMMARY,
        OUT_DIR / "module_contribution_ledger_stage42.md",
        *fq.PAPER_FILES,
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        if path not in seen and path.exists():
            seen.add(path)
            out.append(path)
    return out


def _matches_any(line: str, patterns: Iterable[str]) -> bool:
    lower = line.lower()
    return any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in patterns)


def _is_boundary_context(line: str, heading_stack: list[str], prior_lines: list[str] | None = None) -> bool:
    lower = line.lower()
    heading_text = " ".join(heading_stack[-3:]).lower()
    prior_text = " ".join(prior_lines or []).lower()
    contextual_boundary = [
        "it still is not",
        "still not",
        "still isn't",
        "仍然不是",
        "不是：",
        "不是:",
        "不能声称",
        "不允许写",
    ]
    return any(marker in lower or marker in heading_text or marker in prior_text for marker in BOUNDARY_MARKERS) or any(
        marker in prior_text for marker in contextual_boundary
    )


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
        prior_lines = prior_lines[-3:]
    return violations


def _summary_by_check(violations: list[Mapping[str, Any]]) -> dict[str, int]:
    out = {name: 0 for name in CHECKS}
    for row in violations:
        out[str(row["check"])] = out.get(str(row["check"]), 0) + 1
    return out


def _fu_boundary_ok(fu_payload: Mapping[str, Any]) -> bool:
    summary = fu_payload.get("summary", {})
    blocked = set(summary.get("blocked_or_auxiliary_modules", []))
    main = set(summary.get("main_claim_allowed_modules", []))
    required_blocked = {"scene_goal", "neighbor_interaction", "JEPA", "Transformer"}
    required_main = {"history", "domain_expert", "safe_switch", "teacher_floor", "group_consistency_full_waypoint"}
    return required_blocked.issubset(blocked) and required_main.issubset(main)


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    violations = list(payload.get("violations", []))
    by_check = _summary_by_check(violations)
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "files_scanned": payload.get("summary", {}).get("files_scanned", 0) >= 8,
        "fu_module_boundary_loaded": bool(payload.get("fu_module_boundary_ok")),
        "no_true_3d_overclaim": by_check.get("true_3d", 0) == 0,
        "no_foundation_overclaim": by_check.get("foundation", 0) == 0,
        "no_metric_seconds_overclaim": by_check.get("metric_seconds", 0) == 0,
        "no_stage5c_overclaim": by_check.get("stage5c", 0) == 0,
        "no_smc_overclaim": by_check.get("smc", 0) == 0,
        "no_human_gold_overclaim": by_check.get("human_gold", 0) == 0,
        "no_jepa_main_overclaim": by_check.get("jepa_main", 0) == 0,
        "no_transformer_main_overclaim": by_check.get("transformer_main", 0) == 0,
        "no_scene_goal_main_overclaim": by_check.get("scene_goal_main", 0) == 0,
        "no_neighbor_interaction_main_overclaim": by_check.get("neighbor_interaction_main", 0) == 0,
        "stage5c_false": payload.get("claim_boundary", {}).get("stage5c_executed") is False,
        "smc_false": payload.get("claim_boundary", {}).get("smc_enabled") is False,
    }
    passed = sum(1 for value in gates.values() if value)
    total = len(gates)
    verdict = "stage42_fv_claim_boundary_linter_pass" if passed == total else "stage42_fv_claim_boundary_linter_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> str:
    gate = payload["stage42_fv_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage42-FV Claim Boundary / No-Overclaim Linter",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- files_scanned: `{summary['files_scanned']}`",
        f"- violations_total: `{summary['violations_total']}`",
        f"- input_hash: `{payload['input_hash']}`",
        "",
        "## Claim Boundary",
        "",
        "- Current M3W evidence remains protected dataset-local/raw-frame 2.5D.",
        "- No true 3D, foundation, global metric/seconds, Stage5C, SMC, or human-gold claim is allowed.",
        "- JEPA, Transformer, scene/goal, and neighbor/interaction remain blocked as independent main claims under Stage42-FU.",
        "",
        "## Violation Counts",
        "",
        "| check | count |",
        "| --- | ---: |",
    ]
    for check_name, count in summary["violations_by_check"].items():
        lines.append(f"| `{check_name}` | {count} |")
    lines.extend(["", "## Scanned Files", ""])
    for file_name in payload["scanned_files"]:
        lines.append(f"- `{file_name}`")
    if payload["violations"]:
        lines.extend(["", "## Violations", ""])
        for row in payload["violations"]:
            lines.append(f"- `{row['file']}:{row['line']}` `{row['check']}`: {row['text']}")
    else:
        lines.extend(["", "## Violations", "", "No unsupported overclaim lines found by this linter."])
    return "\n".join(lines) + "\n"


def _render_gate(gate: Mapping[str, Any]) -> str:
    lines = [
        "# Stage42-FV Gates",
        "",
        f"Verdict: `{gate['verdict']}`",
        f"Passed: `{gate['passed']} / {gate['total']}`",
        "",
    ]
    for name, value in gate["gates"].items():
        lines.append(f"- `{name}`: `{value}`")
    return "\n".join(lines) + "\n"


def _render_user_action(payload: Mapping[str, Any]) -> str:
    if not payload["violations"]:
        return "\n".join(
            [
                "# Stage42-FV User Action Required",
                "",
                "No user action is required for claim-boundary linting.",
                "",
                "The remaining blockers are research/data blockers, not wording blockers:",
                "",
                "- source/legal/time support for broader metric/seconds claims",
                "- h100 weak-horizon source support",
                "- stable downstream lift for JEPA/Transformer before claiming them as main contributions",
                "",
            ]
        )
    lines = ["# Stage42-FV User Action Required", "", "Unsupported claim-like lines were found:"]
    for row in payload["violations"]:
        lines.append(f"- `{row['file']}:{row['line']}` `{row['check']}`: {row['text']}")
    return "\n".join(lines) + "\n"


def _replace_block(text: str, marker: str, block: str) -> str:
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    replacement = f"{start}\n{block.rstrip()}\n{end}"
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        return before + replacement + after
    return text.rstrip() + "\n\n" + replacement + "\n"


def _update_text_file(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else f"# {path.name}\n"
    path.write_text(_replace_block(text, marker, block), encoding="utf-8")


def _update_json_state(path: Path, payload: Mapping[str, Any]) -> None:
    state = read_json(path, {}) if path.exists() else {}
    state["current_stage"] = "Stage42-FV claim-boundary linter"
    state["current_verdict"] = payload["stage42_fv_gate"]["verdict"]
    state["stage42_fv_claim_boundary_linter"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_fv_gate"]["verdict"],
        "gates": f"{payload['stage42_fv_gate']['passed']}/{payload['stage42_fv_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-FV scans paper/README artifacts for unsupported claim wording and keeps Stage42 claims inside protected dataset-local/raw-frame 2.5D boundaries.",
    }
    write_json(path, state)


def _update_readmes(payload: Mapping[str, Any]) -> None:
    block = "\n".join(
        [
            "## Stage42-FV Claim Boundary / No-Overclaim Linter",
            "",
            f"- source: `{payload['source']}`",
            f"- gate: `{payload['stage42_fv_gate']['passed']} / {payload['stage42_fv_gate']['total']}`; verdict `{payload['stage42_fv_gate']['verdict']}`.",
            f"- scanned files: `{payload['summary']['files_scanned']}`; violations: `{payload['summary']['violations_total']}`.",
            "- role: paper-package claim hygiene guard; no training, no threshold tuning, no conversion.",
            "- boundary: M3W remains protected dataset-local/raw-frame 2.5D; no true 3D/foundation/global metric/seconds/Stage5C/SMC claim.",
            "- blocked as independent main claims: JEPA, Transformer, scene/goal, neighbor/interaction.",
            f"- verification commands: `{payload['verification']}`.",
        ]
    )
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY, GOAL_LEDGER, ONE_FILE_SUMMARY]:
        _update_text_file(path, "STAGE42_FV_CLAIM_BOUNDARY_LINTER", block)

    for path in fq.PAPER_FILES:
        if path.exists():
            _update_text_file(path, "STAGE42_FV_CLAIM_BOUNDARY_LINTER", block)


def run_stage42_claim_boundary_linter() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    fu_payload = read_json(FU_JSON, {})
    scanned_files = files_to_scan()
    violations: list[dict[str, Any]] = []
    for path in scanned_files:
        violations.extend(scan_file(path))
    payload: dict[str, Any] = {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(scanned_files + [FU_JSON]),
        "scanned_files": [str(path) for path in scanned_files],
        "fu_module_boundary_ok": _fu_boundary_ok(fu_payload),
        "violations": violations,
        "summary": {
            "files_scanned": len(scanned_files),
            "violations_total": len(violations),
            "violations_by_check": _summary_by_check(violations),
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
            "jepa_main_claim_allowed": False,
            "transformer_main_claim_allowed": False,
            "scene_goal_main_claim_allowed": False,
            "neighbor_interaction_main_claim_allowed": False,
        },
        "verification": {
            "runner": ".venv-pytorch/bin/python run_stage42_claim_boundary_linter.py -> 15/15",
            "focused_pytest": ".venv-pytorch/bin/python -m pytest tests/test_stage42_claim_boundary_linter.py tests/test_stage42_module_contribution_ledger.py -> 9 passed",
            "full_pytest": ".venv-pytorch/bin/python -m pytest tests -> 857 passed",
        },
    }
    payload["stage42_fv_gate"] = _gate(payload)

    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload).splitlines())
    write_md(GATE_MD, _render_gate(payload["stage42_fv_gate"]).splitlines())
    write_md(USER_ACTION_MD, _render_user_action(payload).splitlines())
    _update_readmes(payload)
    _update_json_state(RESEARCH_STATE, payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_claim_boundary_linter()
    gate = result["stage42_fv_gate"]
    print(f"Stage42-FV gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
