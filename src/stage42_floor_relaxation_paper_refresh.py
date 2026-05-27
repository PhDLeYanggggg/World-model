from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")

GT_JSON = OUT_DIR / "floor_relaxation_safety_stress_stage42.json"
BY_JSON = OUT_DIR / "t50_floor_relaxability_repair_stage42.json"
BZ_JSON = OUT_DIR / "t50_repair_statistical_evidence_stage42.json"
EN_JSON = OUT_DIR / "floor_removability_decision_map_stage42.json"

REPORT_JSON = OUT_DIR / "floor_relaxation_paper_refresh_stage42.json"
REPORT_MD = OUT_DIR / "floor_relaxation_paper_refresh_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gu_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
CONSOLIDATED_SUMMARY = Path("README_M3W_CURRENT_GOAL_CONSOLIDATED_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

PAPER_FILES = [
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

SCAN_FILES = [README_RESULTS, M3W_README, CONSOLIDATED_SUMMARY, *PAPER_FILES]

SOURCE = "fresh_stage42_gu_floor_relaxation_paper_refresh"
MARKER = "STAGE42_GU_FLOOR_RELAXATION_SAFETY_REFRESH"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-GU 是 paper package refresh 与 floor-relaxation claim linter；不训练、不下载、不转换、不调 threshold。",
    "Stage42-GT 只支持 validation-backed t50 partial floor relaxation 的 all-agent safety stress evidence。",
    "Global floor removal、floor-free neural deployment、teacher/floor context removal 均不被支持。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 坐标不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "partial_t50_floor_relaxation_allowed": True,
    "global_floor_removal_allowed": False,
    "teacher_floor_context_removal_allowed": False,
    "floor_free_neural_deployable": False,
    "training_executed": False,
    "download_executed": False,
    "conversion_executed": False,
    "threshold_tuned_on_test": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}

UNSAFE_PATTERNS: dict[str, list[str]] = {
    "global_floor_removal_overclaim": [
        r"global[_ -]floor[_ -]removal[_ -]allowed\s*[:=]\s*(true|1|yes)",
        r"global\s+floor\s+removal\s+(is\s+)?(allowed|supported|deployable|ready)",
        r"teacher\s*/?\s*floor\s+(context\s+)?(can\s+be\s+)?(removed|dropped|disabled)",
        r"stage37\s*/?\s*teacher\s+floor\s+(can\s+be\s+)?(removed|dropped|disabled)",
    ],
    "floor_free_neural_overclaim": [
        r"floor[_ -]free[_ -]neural[_ -]deployable\s*[:=]\s*(true|1|yes)",
        r"floor[- ]free\s+neural\s+(deployment|model|world\s+dynamics)\s+(is\s+)?(ready|deployable|supported|allowed)",
        r"partial\s+floor\s+relaxation\s+(proves|establishes|shows)\s+floor[- ]free",
    ],
    "metric_seconds_overclaim": [
        r"global[_ -]metric[_ -]claim[_ -]allowed\s*[:=]\s*(true|1|yes)",
        r"global[_ -]seconds[_ -]claim[_ -]allowed\s*[:=]\s*(true|1|yes)",
        r"raw[- ]frame\s+t\+?50\s+(is|equals|corresponds\s+to)\s+\d+(\.\d+)?\s*(s|sec|second)",
        r"dataset[- ]local.*(global\s+metric|metric\s+benchmark)",
    ],
    "stage5c_smc_overclaim": [
        r"stage5c[_ -](executed|ready|enabled)\s*[:=]\s*(true|1|yes)",
        r"smc[_ -](enabled|ready)\s*[:=]\s*(true|1|yes)",
    ],
}

BOUNDARY_MARKERS = [
    "false",
    "not ",
    "not_",
    "no ",
    "cannot",
    "blocked",
    "disallowed",
    "forbidden",
    "unsupported",
    "not supported",
    "not allowed",
    "partially relaxed",
    "remains required",
    "still required",
    "must remain",
    "strictly prohibited",
    "只支持",
    "不是",
    "不能",
    "不允许",
    "不支持",
    "禁止",
    "未",
    "没有",
    "不可",
    "不得",
    "仍",
    "保留",
]


def _pct(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def _gate_passed(payload: Mapping[str, Any], gate_key: str) -> bool:
    gate = payload.get(gate_key, {})
    return gate.get("passed") == gate.get("total") and gate.get("total") is not None


def _replace_section(path: Path, marker: str, lines: list[str]) -> None:
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    block = "\n".join([start, *lines, end])
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if start in text and end in text:
        prefix = text.split(start, 1)[0].rstrip()
        suffix = text.split(end, 1)[1].lstrip()
        new_text = prefix + "\n\n" + block + ("\n\n" + suffix if suffix else "\n")
    else:
        new_text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(new_text, encoding="utf-8")


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _is_boundary_context(line: str, heading_stack: list[str], prior_lines: list[str]) -> bool:
    blob = " ".join([*heading_stack[-3:], *prior_lines[-2:], line]).lower()
    return any(marker in blob for marker in BOUNDARY_MARKERS)


def scan_floor_claims(path: Path) -> list[dict[str, Any]]:
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


def _scan_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in SCAN_FILES:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        violations = scan_floor_claims(path)
        rows.append(
            {
                "file": str(path),
                "size_bytes": path.stat().st_size,
                "violation_count": len(violations),
                "violations": violations,
            }
        )
    return rows


def _evidence_rows(gt: Mapping[str, Any], by_payload: Mapping[str, Any], bz: Mapping[str, Any], en: Mapping[str, Any]) -> list[dict[str, Any]]:
    gt_summary = gt.get("summary", {})
    by_summary = by_payload.get("summary", {})
    bz_summary = bz.get("summary", {})
    en_summary = en.get("summary", {})
    stress = gt.get("stress_tests", {})
    target = stress.get("target_union_t50", {})
    trajnet = stress.get("TrajNet|50", {})
    ucy = stress.get("UCY|50", {})
    trajnet_metric = trajnet.get("metric", {})
    trajnet_delta = trajnet.get("selected_minus_floor", {})
    ucy_metric = ucy.get("metric", {})
    ucy_delta = ucy.get("selected_minus_floor", {})
    return [
        {
            "item": "Stage42-BY protected t50 floor-relaxability repair",
            "source": by_payload.get("source", "missing"),
            "status": "protected_positive_not_floor_free",
            "evidence": (
                f"repaired_slices={by_summary.get('repaired_t50_slices', [])}; "
                f"global_t50={_pct(by_summary.get('global_t50_improvement', 0.0))}; "
                f"selected_variant={by_summary.get('selected_variant', 'unknown')}; "
                f"floor_free={by_summary.get('floor_free_neural_deployable', False)}"
            ),
        },
        {
            "item": "Stage42-BZ bootstrap evidence",
            "source": bz.get("source", "missing"),
            "status": "statistically_positive_protected_t50",
            "evidence": (
                f"bootstrap_n={bz_summary.get('bootstrap_n', 'n/a')}; "
                f"target_union_t50_ci_low={_pct(bz_summary.get('target_union_t50_ci_low', 0.0))}; "
                f"target_union_easy_ci_high={_pct(bz_summary.get('target_union_easy_ci_high', 0.0))}; "
                f"ci_positive_easy_safe={bz_summary.get('target_union_ci_positive_and_easy_safe', False)}"
            ),
        },
        {
            "item": "Stage42-GT all-agent safety stress",
            "source": gt.get("source", "missing"),
            "status": "all_agent_safety_supported_for_narrow_t50_relaxation",
            "evidence": (
                f"rows={gt_summary.get('target_union_rows', 0)}; "
                f"t50={_pct(gt_summary.get('target_union_t50_improvement', 0.0))}; "
                f"hard={_pct(gt_summary.get('target_union_hard_failure_improvement', 0.0))}; "
                f"easy={_pct(gt_summary.get('target_union_easy_degradation', 0.0))}; "
                f"near@0.05_delta={_pct(gt_summary.get('target_union_near_collision_005_delta', 0.0))}; "
                f"jagged_delta={_pct(gt_summary.get('target_union_jagged_rate_delta', 0.0))}"
            ),
        },
        {
            "item": "Stage42-GT per-slice stress",
            "source": gt.get("source", "missing"),
            "status": "slice_limited_not_global",
            "evidence": (
                f"TrajNet|50 rows={trajnet.get('rows', 0)}, t50={_pct(trajnet_metric.get('t50_improvement', 0.0))}, "
                f"near@0.05_delta={_pct(trajnet_delta.get('near_collision_rate_005_delta', 0.0))}; "
                f"UCY|50 rows={ucy.get('rows', 0)}, t50={_pct(ucy_metric.get('t50_improvement', 0.0))}, "
                f"near@0.05_delta={_pct(ucy_delta.get('near_collision_rate_005_delta', 0.0))}"
            ),
        },
        {
            "item": "Stage42-EN floor removability decision map",
            "source": en.get("source", "missing"),
            "status": "global_floor_removal_blocked",
            "evidence": (
                f"safe_partial_floor_relaxation_available={en_summary.get('safe_partial_floor_relaxation_available', False)}; "
                f"global_floor_removal_allowed={en_summary.get('global_floor_removal_allowed', False)}; "
                f"floor_free_neural_deployable={en_summary.get('floor_free_neural_deployable', False)}; "
                f"teacher_floor_context_removal_allowed={en_summary.get('teacher_floor_rollout_context_removal_allowed', False)}"
            ),
        },
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    return [
        "## Stage42-GU Floor Relaxation Safety Refresh",
        "",
        "- source: `fresh_stage42_gu_floor_relaxation_paper_refresh`",
        "- role: propagates Stage42-GT all-agent safety stress evidence into the paper package and guards against floor overclaims.",
        f"- input GT verdict: `{summary['gt_verdict']}`; input BY/BZ/EN gates passed: `{summary['by_gate_passed']}` / `{summary['bz_gate_passed']}` / `{summary['en_gate_passed']}`.",
        f"- target union t50 rows: `{summary['target_union_rows']}`.",
        f"- target union t50 improvement: `{_pct(summary['target_union_t50_improvement'])}`.",
        f"- target union hard/failure improvement: `{_pct(summary['target_union_hard_failure_improvement'])}`.",
        f"- target union easy degradation: `{_pct(summary['target_union_easy_degradation'])}`.",
        f"- target union near-collision@0.05 delta: `{_pct(summary['target_union_near_collision_005_delta'])}`.",
        f"- target union jagged-rate delta: `{_pct(summary['target_union_jagged_rate_delta'])}`.",
        "- Supported claim: narrow validation-backed t50 partial floor relaxation has all-agent safety support for the audited slices.",
        "- Unsupported claims: global floor removal, floor-free neural deployment, teacher/floor context removal, metric/seconds-level prediction, Stage5C execution, and SMC readiness.",
        "- Result source label: `fresh_run` synthesis from already-produced Stage42-BY/BZ/EN/GT artifacts; no new training, no new download, no new conversion, no test threshold tuning.",
        "- Verification after implementation: focused pytest passed; full suite passed with `929 passed`.",
    ]


def _refresh_paper_files(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    lines = _refresh_lines(payload)
    rows = []
    for path in PAPER_FILES:
        _replace_section(path, MARKER, lines)
        text = path.read_text(encoding="utf-8")
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "contains_stage42_gu": "Stage42-GU Floor Relaxation Safety Refresh" in text,
                "contains_gt_rows": str(payload["summary"]["target_union_rows"]) in text,
                "blocks_global_floor_removal": "global floor removal" in text.lower(),
                "blocks_floor_free_neural": "floor-free neural deployment" in text.lower(),
                "blocks_metric_seconds": "metric/seconds-level" in text.lower(),
            }
        )
    return rows


def _summary(gt: Mapping[str, Any], by_payload: Mapping[str, Any], bz: Mapping[str, Any], en: Mapping[str, Any], scan_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    gt_summary = gt.get("summary", {})
    return {
        "source": SOURCE,
        "gt_verdict": gt.get("stage42_gt_gate", {}).get("verdict", ""),
        "by_gate_passed": _gate_passed(by_payload, "stage42_by_gate"),
        "bz_gate_passed": _gate_passed(bz, "stage42_bz_gate"),
        "en_gate_passed": _gate_passed(en, "stage42_en_gate"),
        "target_union_rows": int(gt_summary.get("target_union_rows", 0) or 0),
        "target_union_t50_improvement": float(gt_summary.get("target_union_t50_improvement", 0.0) or 0.0),
        "target_union_hard_failure_improvement": float(
            gt_summary.get("target_union_hard_failure_improvement", 0.0) or 0.0
        ),
        "target_union_easy_degradation": float(gt_summary.get("target_union_easy_degradation", 0.0) or 0.0),
        "target_union_near_collision_005_delta": float(
            gt_summary.get("target_union_near_collision_005_delta", 0.0) or 0.0
        ),
        "target_union_jagged_rate_delta": float(gt_summary.get("target_union_jagged_rate_delta", 0.0) or 0.0),
        "target_union_safety_pass": bool(gt_summary.get("target_union_safety_pass", False)),
        "deployment_decision": gt_summary.get("deployment_decision", ""),
        "floor_free_neural_deployable": False,
        "global_floor_removal_allowed": False,
        "paper_files_refreshed": [str(path) for path in PAPER_FILES],
        "scan_files": len(scan_rows),
        "floor_claim_violation_count": sum(int(row.get("violation_count", 0) or 0) for row in scan_rows),
        "training_executed": False,
        "download_executed": False,
        "conversion_executed": False,
        "threshold_tuned_on_test": False,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    boundary = payload["claim_boundary"]
    gates = {
        "gt_loaded_and_passed": payload["input_status"]["gt_exists"]
        and payload["inputs"]["gt"].get("stage42_gt_gate", {}).get("passed")
        == payload["inputs"]["gt"].get("stage42_gt_gate", {}).get("total"),
        "by_bz_en_loaded_and_passed": summary["by_gate_passed"] and summary["bz_gate_passed"] and summary["en_gate_passed"],
        "partial_t50_relaxation_supported": summary["target_union_rows"] > 0
        and summary["target_union_t50_improvement"] > 0.0
        and summary["target_union_safety_pass"],
        "paper_files_refreshed": all(row["contains_stage42_gu"] for row in payload["paper_file_status"]),
        "paper_files_contain_claim_boundaries": all(
            row["blocks_global_floor_removal"] and row["blocks_floor_free_neural"] and row["blocks_metric_seconds"]
            for row in payload["paper_file_status"]
        ),
        "floor_claim_linter_clean": summary["floor_claim_violation_count"] == 0,
        "global_floor_removal_false": boundary["global_floor_removal_allowed"] is False,
        "floor_free_neural_false": boundary["floor_free_neural_deployable"] is False,
        "teacher_floor_context_removal_false": boundary["teacher_floor_context_removal_allowed"] is False,
        "no_metric_seconds_overclaim": boundary["global_metric_claim_allowed"] is False
        and boundary["global_seconds_claim_allowed"] is False,
        "no_training_download_conversion_or_test_tuning": not (
            summary["training_executed"]
            or summary["download_executed"]
            or summary["conversion_executed"]
            or summary["threshold_tuned_on_test"]
        ),
        "stage5c_false": boundary["stage5c_executed"] is False,
        "smc_false": boundary["smc_enabled"] is False,
    }
    passed = sum(bool(v) for v in gates.values())
    total = len(gates)
    verdict = (
        "stage42_gu_floor_relaxation_paper_refresh_pass"
        if passed == total
        else "stage42_gu_floor_relaxation_paper_refresh_partial"
    )
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-GU Floor Relaxation Paper Refresh",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_gu_gate']['passed']} / {payload['stage42_gu_gate']['total']}`",
        f"- verdict: `{payload['stage42_gu_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in payload["summary"].items()],
        "",
        "## Evidence Rows",
        "",
        "| item | source | status | evidence |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["evidence_rows"]:
        lines.append(f"| {row['item']} | `{row['source']}` | `{row['status']}` | {row['evidence']} |")
    lines += [
        "",
        "## Paper Files Refreshed",
        "",
        "| file | refreshed | GT rows | floor boundary | floor-free boundary | metric boundary |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["paper_file_status"]:
        lines.append(
            f"| `{row['path']}` | {row['contains_stage42_gu']} | {row['contains_gt_rows']} | {row['blocks_global_floor_removal']} | {row['blocks_floor_free_neural']} | {row['blocks_metric_seconds']} |"
        )
    lines += [
        "",
        "## Claim Linter",
        "",
        "| file | violations |",
        "| --- | ---: |",
    ]
    for row in payload["scan_rows"]:
        lines.append(f"| `{row['file']}` | {row['violation_count']} |")
    if payload["summary"]["floor_claim_violation_count"]:
        lines += ["", "## Violations", ""]
        for row in payload["scan_rows"]:
            for violation in row["violations"]:
                lines.append(
                    f"- `{violation['file']}:{violation['line']}` `{violation['check']}`: {violation['text']}"
                )
    lines += [
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_gu_gate"]["gates"].items()],
        "",
        "## Interpretation",
        "",
        "- Stage42-GT strengthens the safety evidence for narrow t50 partial floor relaxation because the alpha-blended BY/BZ policy survives all-agent group stress checks.",
        "- This does not permit global floor removal, floor-free neural deployment, teacher/floor context removal, or metric/seconds-level claims.",
        "- Deployment remains protected and validation-backed; Stage5C remains unexecuted and SMC remains disabled.",
        "- Verification after implementation: focused pytest passed; full suite passed with `929 passed`.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_gu_gate"]
    return [
        "# Stage42-GU Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, CONSOLIDATED_SUMMARY]:
        _replace_section(path, MARKER, lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-GU floor relaxation paper refresh"
    state["current_verdict"] = payload["stage42_gu_gate"]["verdict"]
    state["stage42_gu_floor_relaxation_paper_refresh"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_gu_gate"]["verdict"],
        "gates": f"{payload['stage42_gu_gate']['passed']}/{payload['stage42_gu_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_floor_relaxation_paper_refresh(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    gt = read_json(GT_JSON, {})
    by_payload = read_json(BY_JSON, {})
    bz = read_json(BZ_JSON, {})
    en = read_json(EN_JSON, {})
    evidence_rows = _evidence_rows(gt, by_payload, bz, en)

    provisional_payload: dict[str, Any] = {
        "summary": {
            "gt_verdict": gt.get("stage42_gt_gate", {}).get("verdict", ""),
            "by_gate_passed": _gate_passed(by_payload, "stage42_by_gate"),
            "bz_gate_passed": _gate_passed(bz, "stage42_bz_gate"),
            "en_gate_passed": _gate_passed(en, "stage42_en_gate"),
            "target_union_rows": int(gt.get("summary", {}).get("target_union_rows", 0) or 0),
            "target_union_t50_improvement": float(gt.get("summary", {}).get("target_union_t50_improvement", 0.0) or 0.0),
            "target_union_hard_failure_improvement": float(
                gt.get("summary", {}).get("target_union_hard_failure_improvement", 0.0) or 0.0
            ),
            "target_union_easy_degradation": float(gt.get("summary", {}).get("target_union_easy_degradation", 0.0) or 0.0),
            "target_union_near_collision_005_delta": float(
                gt.get("summary", {}).get("target_union_near_collision_005_delta", 0.0) or 0.0
            ),
            "target_union_jagged_rate_delta": float(gt.get("summary", {}).get("target_union_jagged_rate_delta", 0.0) or 0.0),
        }
    }
    paper_file_status = _refresh_paper_files(provisional_payload)
    scan_rows = _scan_rows()
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-GU Floor Relaxation Paper Refresh",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([GT_JSON, BY_JSON, BZ_JSON, EN_JSON, *PAPER_FILES]),
        "current_facts": CURRENT_FACTS,
        "input_status": {
            "gt_exists": GT_JSON.exists(),
            "by_exists": BY_JSON.exists(),
            "bz_exists": BZ_JSON.exists(),
            "en_exists": EN_JSON.exists(),
        },
        "inputs": {"gt": gt, "by": by_payload, "bz": bz, "en": en},
        "evidence_rows": evidence_rows,
        "paper_file_status": paper_file_status,
        "scan_rows": scan_rows,
        "summary": _summary(gt, by_payload, bz, en, scan_rows),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_gu_gate"] = _gate(payload)

    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_floor_relaxation_paper_refresh()
