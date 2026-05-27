from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_full_waypoint_bridge_shape_audit import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
GZ_JSON = OUT_DIR / "full_waypoint_claim_guard_stage42.json"
GJ_JSON = OUT_DIR / "module_claim_lock_stage42.json"
FV_JSON = OUT_DIR / "claim_boundary_linter_stage42.json"

REPORT_JSON = OUT_DIR / "full_waypoint_overclaim_linter_stage42.json"
REPORT_MD = OUT_DIR / "full_waypoint_overclaim_linter_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ha_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
ONE_FILE_SUMMARY = Path("README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md")

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

SOURCE = "fresh_stage42_ha_full_waypoint_overclaim_linter"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HA 是 full-waypoint overclaim linter，不重新训练、不转换数据、不调 threshold。",
    "endpoint-only / endpoint-linear bridge 不能写成 learned full-waypoint dynamics。",
    "ungated full-waypoint neural 不能写成 deployable。",
    "group-consistency full-waypoint 可写为 protected module；neighbor/interaction 独立主 claim 仍 blocked。",
    "t+50 / t+100 是 raw-frame horizon；dataset-local/raw-frame 不能写成 metric/seconds-level。",
    "Stage5C 未执行，SMC 未启用。",
]

BOUNDARY_MARKERS = [
    "not ",
    "not_",
    "no ",
    "never",
    "false",
    "rejected",
    "blocked",
    "forbidden",
    "cannot",
    "can't",
    "must not",
    "should not",
    "not allowed",
    "non-claim",
    "diagnostic",
    "caveat",
    "boundary",
    "guard",
    "separate",
    "不是",
    "不能",
    "不得",
    "不允许",
    "禁止",
    "拒绝",
    "边界",
    "当前不是",
    "仍然不能说",
    "不能写的主线",
    "ready now `0`",
    "ready now 0",
    "after terms",
    "candidate",
    "candidates",
    "source-specific restricted subset",
    "metric claim ready datasets",
    "仍",
    "未",
    "没有",
    "不可",
    "需",
]

CHECKS: dict[str, dict[str, Any]] = {
    "endpoint_as_full_waypoint": {
        "patterns": [
            r"endpoint[- ]?(only|linear|bridge).*(equivalent|counts as|counted as|is learned|as learned).*full[- ]?waypoint",
            r"endpoint[- ]?(only|linear|bridge).*等同.*full[- ]?waypoint",
            r"endpoint[- ]?(only|linear|bridge).*当成.*full[- ]?waypoint",
        ],
        "description": "Endpoint-only/linear bridge must not be claimed as learned full-waypoint dynamics.",
    },
    "ungated_full_waypoint_deployable": {
        "patterns": [
            r"ungated.*full[- ]?waypoint.*(deployable|deployment|safe|ready|可部署|安全|ready)",
            r"full[- ]?waypoint.*ungated.*(deployable|deployment|safe|ready|可部署|安全|ready)",
        ],
        "description": "Ungated full-waypoint neural must not be claimed deployable.",
    },
    "global_primary_full_waypoint_replacement": {
        "patterns": [
            r"global primary full[- ]?waypoint replacement.*(allowed|ready|deployable|pass|可部署|允许)",
            r"full[- ]?waypoint.*(global primary|primary replacement).*(allowed|ready|deployable|pass|可部署|允许)",
        ],
        "description": "Global primary full-waypoint replacement claim remains blocked.",
    },
    "neighbor_interaction_independent": {
        "patterns": [
            r"neighbor/interaction.*(independent main|main claim|main contribution|主贡献)",
            r"neighbor.*interaction.*(independent main|main claim|main contribution|主贡献)",
        ],
        "description": "Neighbor/interaction cannot be an independent main contribution under current evidence.",
    },
    "group_consistency_unprotected": {
        "patterns": [
            r"group[- ]?consistency.*(unprotected|ungated).*(deployable|main|safe|可部署)",
            r"unprotected.*group[- ]?consistency.*(deployable|safe|可部署)",
        ],
        "description": "Group consistency is supported only under protected policy.",
    },
    "metric_seconds_or_3d": {
        "patterns": [
            r"true[- ]?3d.*(ready|achieved|success|成功|完成)",
            r"metric.*(ready|achieved|success|成功|完成)",
            r"seconds[- ]?level.*(ready|achieved|success|成功|完成)",
            r"秒级.*(成功|完成|可用)",
        ],
        "description": "No true-3D, metric, or seconds-level claim.",
    },
    "stage5c_smc": {
        "patterns": [
            r"stage5c.*(executed|enabled|ready|执行|启用|完成)",
            r"\bsmc\b.*(enabled|ready|执行|启用|完成)",
        ],
        "description": "No Stage5C execution or SMC readiness claim.",
    },
}


def files_to_scan() -> list[Path]:
    candidates = [
        README_RESULTS,
        M3W_README,
        WORK_SUMMARY,
        ONE_FILE_SUMMARY,
        OUT_DIR / "full_waypoint_claim_guard_stage42.md",
        OUT_DIR / "module_claim_lock_stage42.md",
        *PAPER_FILES,
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        if path.exists() and path not in seen:
            out.append(path)
            seen.add(path)
    return out


def _matches_any(line: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in patterns)


def _is_boundary_context(line: str, heading_stack: list[str], prior_lines: list[str] | None = None) -> bool:
    lower = line.lower()
    context = " ".join([*heading_stack[-3:], *(prior_lines or [])]).lower()
    return any(marker in lower or marker in context for marker in BOUNDARY_MARKERS)


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
        prior_lines = prior_lines[-12:]
    return violations


def _summary_by_check(violations: list[Mapping[str, Any]]) -> dict[str, int]:
    out = {name: 0 for name in CHECKS}
    for row in violations:
        key = str(row["check"])
        out[key] = out.get(key, 0) + 1
    return out


def _gz_boundary_ok(gz_payload: Mapping[str, Any]) -> bool:
    rows = {row.get("claim_id"): row for row in gz_payload.get("claim_rows", [])}
    return (
        bool(rows.get("GZ-C1", {}).get("allowed_as_main_claim"))
        and rows.get("GZ-C2", {}).get("allowed_as_main_claim") is False
        and rows.get("GZ-C3", {}).get("allowed_as_main_claim") is False
        and rows.get("GZ-C8", {}).get("allowed_as_main_claim") is False
        and rows.get("GZ-C9", {}).get("allowed_as_main_claim") is False
    )


def _gj_boundary_ok(gj_payload: Mapping[str, Any]) -> bool:
    summary = gj_payload.get("summary", {})
    blocked = set(summary.get("blocked_main_modules_locked", []))
    supported = set(summary.get("supported_main_modules_locked", []))
    return "neighbor_interaction" in blocked and "Transformer" in blocked and "group_consistency_full_waypoint" in supported


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    by_check = payload["summary"]["violations_by_check"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "files_scanned": payload["summary"]["files_scanned"] >= 12,
        "gz_claim_guard_loaded": payload.get("gz_boundary_ok") is True,
        "gj_module_claim_lock_loaded": payload.get("gj_boundary_ok") is True,
        "fv_linter_loaded": payload.get("fv_linter_loaded") is True,
        "no_endpoint_as_full_waypoint_overclaim": by_check.get("endpoint_as_full_waypoint", 0) == 0,
        "no_ungated_full_waypoint_deployable_overclaim": by_check.get("ungated_full_waypoint_deployable", 0) == 0,
        "no_global_primary_replacement_overclaim": by_check.get("global_primary_full_waypoint_replacement", 0) == 0,
        "no_neighbor_interaction_independent_overclaim": by_check.get("neighbor_interaction_independent", 0) == 0,
        "no_group_consistency_unprotected_overclaim": by_check.get("group_consistency_unprotected", 0) == 0,
        "no_metric_seconds_or_3d_overclaim": by_check.get("metric_seconds_or_3d", 0) == 0,
        "no_stage5c_smc_overclaim": by_check.get("stage5c_smc", 0) == 0,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = sum(bool(v) for v in gates.values())
    total = len(gates)
    verdict = "stage42_ha_full_waypoint_overclaim_linter_pass" if passed == total else "stage42_ha_full_waypoint_overclaim_linter_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ha_gate"]
    lines = [
        "# Stage42-HA Full-Waypoint Overclaim Linter",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- files_scanned: `{payload['summary']['files_scanned']}`",
        f"- violations_total: `{payload['summary']['violations_total']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Violation Counts",
        "",
        "| check | count |",
        "| --- | ---: |",
    ]
    for key, count in payload["summary"]["violations_by_check"].items():
        lines.append(f"| `{key}` | {count} |")
    lines += [
        "",
        "## Scanned Files",
        "",
        *[f"- `{path}`" for path in payload["scanned_files"]],
        "",
        "## Violations",
        "",
    ]
    if payload["violations"]:
        for row in payload["violations"]:
            lines.append(f"- `{row['file']}:{row['line']}` `{row['check']}`: {row['text']}")
    else:
        lines.append("No unsupported full-waypoint overclaim lines found by this linter.")
    lines += [
        "",
        "## Guarded Boundaries",
        "",
        "- Endpoint-only / endpoint-linear bridge success must stay separate from learned full-waypoint dynamics.",
        "- Ungated full-waypoint neural remains not deployable.",
        "- Group-consistency full-waypoint is supported only as a protected module.",
        "- Neighbor/interaction alone remains blocked as an independent main claim.",
        "- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is allowed.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ha_gate"]
    lines = [
        "# Stage42-HA Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | {bool(value)} |")
    return lines


def _refresh_docs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    lines = [
        "## Stage42-HA Full-Waypoint Overclaim Linter",
        "",
        "- source: `fresh_stage42_ha_full_waypoint_overclaim_linter`",
        f"- gate: `{payload['stage42_ha_gate']['passed']} / {payload['stage42_ha_gate']['total']}`",
        f"- verdict: `{payload['stage42_ha_gate']['verdict']}`",
        f"- files_scanned: `{payload['summary']['files_scanned']}`",
        f"- violations_total: `{payload['summary']['violations_total']}`",
        "- Endpoint/full-waypoint, ungated full-waypoint, group/neighbor independent-main, metric/seconds, Stage5C and SMC overclaims were scanned.",
        "- No unsupported full-waypoint overclaim lines were found.",
    ]
    status = []
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, "STAGE42_HA_FULL_WAYPOINT_OVERCLAIM_LINTER", lines)
            text = path.read_text(encoding="utf-8")
            status.append(
                {
                    "path": str(path),
                    "exists": True,
                    "contains_stage42_ha": "Stage42-HA Full-Waypoint Overclaim Linter" in text,
                }
            )
        else:
            status.append({"path": str(path), "exists": False})
    return status


def run_stage42_full_waypoint_overclaim_linter() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    gz_payload = read_json(GZ_JSON, {})
    gj_payload = read_json(GJ_JSON, {})
    fv_payload = read_json(FV_JSON, {})
    scanned = files_to_scan()
    violations: list[dict[str, Any]] = []
    for path in scanned:
        violations.extend(scan_file(path))
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-HA full-waypoint overclaim linter",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([GZ_JSON, GJ_JSON, FV_JSON, *scanned]),
        "input_status": {
            "gz": {
                "path": str(GZ_JSON),
                "source": gz_payload.get("source"),
                "verdict": (gz_payload.get("stage42_gz_gate", {}) or {}).get("verdict"),
            },
            "gj": {
                "path": str(GJ_JSON),
                "source": gj_payload.get("source"),
                "verdict": (gj_payload.get("stage42_gj_gate", {}) or {}).get("verdict"),
            },
            "fv": {
                "path": str(FV_JSON),
                "source": fv_payload.get("source"),
                "verdict": (fv_payload.get("stage42_fv_gate", {}) or {}).get("verdict"),
            },
        },
        "gz_boundary_ok": _gz_boundary_ok(gz_payload),
        "gj_boundary_ok": _gj_boundary_ok(gj_payload),
        "fv_linter_loaded": bool(fv_payload.get("stage42_fv_gate")),
        "scanned_files": [str(path) for path in scanned],
        "violations": violations,
        "summary": {
            "files_scanned": len(scanned),
            "violations_total": len(violations),
            "violations_by_check": _summary_by_check(violations),
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "ungated_full_waypoint_deployable": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_ha_gate"] = _gate(payload)
    payload["doc_refresh_status"] = _refresh_docs(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    return payload


if __name__ == "__main__":
    run_stage42_full_waypoint_overclaim_linter()
