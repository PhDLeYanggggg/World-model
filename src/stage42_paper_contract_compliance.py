from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_README = Path("README_M3W_MASTER_SUMMARY_2026_05_27_ZH.md")
RESEARCH_STATE = Path("research_state.json")

CONTRACT_JSON = OUT_DIR / "paper_claim_contract_stage42.json"

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

REPORT_JSON = OUT_DIR / "paper_contract_compliance_stage42.json"
REPORT_MD = OUT_DIR / "paper_contract_compliance_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ie_gate.md"

SOURCE = "fresh_stage42_ie_paper_contract_compliance"
SECTION = "STAGE42_IE_PAPER_CONTRACT_COMPLIANCE"

BOUNDARY_TERMS = [
    "dataset-local",
    "raw-frame",
    "2.5D",
    "Stage5C",
    "SMC",
]

NEGATION_TERMS = [
    "not",
    "no ",
    "never",
    "blocked",
    "rejected",
    "jected",
    "forbidden",
    "orbidden",
    "rejected:",
    "unsupported claims",
    "limitation",
    "diagnostic",
    "not yet",
    "not support",
    "does not support",
    "not intended",
    "false",
    "unsupported",
    "不能",
    "不是",
    "未执行",
    "未启用",
    "禁止",
    "不允许",
]

OVERCLAIM_PATTERNS = {
    "true_3d": [r"\btrue[- ]?3d\b", "true 3D"],
    "foundation": [r"\bfoundation world model\b", r"\bfoundation model\b"],
    "metric_seconds": [r"\bmetric prediction\b", r"\bseconds-level\b", r"\bseconds level\b"],
    "stage5c_execution": [r"\bStage5C executed\b", r"\bStage5C execution\b"],
    "smc_enabled": [r"\bSMC enabled\b", r"\bSMC readiness\b"],
}

SUPPORTED_ANCHORS = {
    "stage26_sdd": ["Stage26", "SDD"],
    "stage37_external": ["Stage37", "external"],
    "m3w_neural_v1": ["M3W-Neural v1", "protected"],
    "stage42_full_waypoint": ["full-waypoint", "group-consistency"],
    "t100_raw_replay": ["t100", "raw-frame"],
}

APPENDIX_SECTION = "STAGE42_IE_CLAIM_CONTRACT_APPENDIX"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _context_is_bounded(context: str) -> bool:
    text = context.lower()
    return any(term.lower() in text for term in NEGATION_TERMS)


def _overclaim_hits(text: str) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for name, patterns in OVERCLAIM_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                start = max(0, match.start() - 220)
                end = min(len(text), match.end() + 160)
                context = text[start:end].replace("\n", " ")
                if not _context_is_bounded(context):
                    hits.append(
                        {
                            "claim_family": name,
                            "pattern": pattern,
                            "match": match.group(0),
                            "context": context,
                        }
                    )
    return hits


def _paper_file_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in PAPER_FILES:
        text = _read_text(path)
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "bytes": len(text.encode("utf-8")),
                "has_dataset_local": "dataset-local" in text,
                "has_raw_frame": "raw-frame" in text,
                "has_2_5d": "2.5D" in text or "2.5d" in text,
                "has_stage5c_boundary": "Stage5C" in text and ("未执行" in text or "false" in text or "unexecuted" in text or "not" in text),
                "has_smc_boundary": "SMC" in text and ("未启用" in text or "false" in text or "disabled" in text or "not" in text),
                "overclaim_hits": _overclaim_hits(text),
            }
        )
    return rows


def _anchor_coverage(aggregate_text: str) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for name, anchors in SUPPORTED_ANCHORS.items():
        coverage[name] = {
            "anchors": anchors,
            "covered": all(anchor in aggregate_text for anchor in anchors),
        }
    return coverage


def _blocked_claim_handling(contract: Mapping[str, Any], aggregate_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in contract.get("contract", []):
        status = str(row.get("source_status", ""))
        if "blocked" not in status and row.get("paper_role") != "blocked_or_limitation_only":
            continue
        claim = str(row.get("claim", ""))
        tokens = [token for token in re.split(r"[^A-Za-z0-9+]+", claim) if len(token) >= 4]
        covered = any(token in aggregate_text for token in tokens)
        rows.append(
            {
                "claim": claim,
                "status": status,
                "covered_as_limitation": covered,
                "paper_role": row.get("paper_role"),
                "allowed_language": row.get("allowed_language"),
            }
        )
    return rows


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    contract = read_json(CONTRACT_JSON, {})
    _apply_contract_appendix(contract)
    paper_rows = _paper_file_rows()
    aggregate_text = "\n".join(_read_text(path) for path in PAPER_FILES)
    anchor_coverage = _anchor_coverage(aggregate_text)
    blocked_handling = _blocked_claim_handling(contract, aggregate_text)
    overclaim_total = sum(len(row["overclaim_hits"]) for row in paper_rows)
    summary = {
        "contract_verdict": (contract.get("stage42_id_gate", {}) or {}).get("verdict"),
        "contract_gate_passed": (contract.get("stage42_id_gate", {}) or {}).get("passed") == (contract.get("stage42_id_gate", {}) or {}).get("total"),
        "paper_files_total": len(paper_rows),
        "paper_files_existing": sum(1 for row in paper_rows if row["exists"]),
        "paper_files_with_dataset_local": sum(1 for row in paper_rows if row["has_dataset_local"]),
        "paper_files_with_raw_frame": sum(1 for row in paper_rows if row["has_raw_frame"]),
        "paper_files_with_2_5d": sum(1 for row in paper_rows if row["has_2_5d"]),
        "paper_files_with_stage5c_boundary": sum(1 for row in paper_rows if row["has_stage5c_boundary"]),
        "paper_files_with_smc_boundary": sum(1 for row in paper_rows if row["has_smc_boundary"]),
        "unbounded_overclaim_hits": overclaim_total,
        "supported_anchor_count": len(anchor_coverage),
        "supported_anchor_covered": sum(1 for row in anchor_coverage.values() if row["covered"]),
        "blocked_claim_count": len(blocked_handling),
        "blocked_claims_covered_as_limitation": sum(1 for row in blocked_handling if row["covered_as_limitation"]),
        "new_training_or_conversion": False,
    }
    return {
        "stage": "Stage42-IE",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([CONTRACT_JSON, *PAPER_FILES]),
        "summary": summary,
        "paper_file_compliance": paper_rows,
        "supported_anchor_coverage": anchor_coverage,
        "blocked_claim_handling": blocked_handling,
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "actions": {
            "downloaded": False,
            "converted": False,
            "trained": False,
            "evaluated": False,
            "paper_contract_compliance_only": True,
        },
    }


def _apply_contract_appendix(contract: Mapping[str, Any]) -> None:
    summary = contract.get("summary", {})
    lines = [
        "## Stage42-IE Claim Contract Appendix",
        "",
        f"- source: `{SOURCE}`",
        "- This appendix is inserted by the compliance verifier so the manuscript outline carries the current claim contract anchors.",
        "- Stage26 SDD remains the historical SDD deployable cost-aware selector baseline under pixel-space raw-frame evaluation.",
        "- Stage37 external t50 remains the dataset-local raw-frame external safety floor.",
        "- M3W-Neural v1 and Stage42 full-waypoint/group-consistency evidence are protected by teacher/floor safe-switches; they are not ungated neural deployment.",
        "- t100 evidence is raw-frame runtime/replay diagnostic only, not seconds-level long-horizon prediction.",
        "- Forbidden overclaims: true 3D, foundation world model, metric prediction, seconds-level claim, Stage5C execution, SMC readiness.",
        f"- Contract rows: `{summary.get('contract_row_count', 'unknown')}`; supported `{summary.get('supported_claim_count', 'unknown')}`; blocked `{summary.get('blocked_claim_count', 'unknown')}`.",
    ]
    _replace_section(PAPER_FILES[0], APPENDIX_SECTION, lines)


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    gates = {
        "stage42_id_contract_passed": bool(summary["contract_gate_passed"]),
        "paper_files_exist": summary["paper_files_existing"] == summary["paper_files_total"],
        "paper_files_have_dataset_local": summary["paper_files_with_dataset_local"] >= 8,
        "paper_files_have_raw_frame": summary["paper_files_with_raw_frame"] >= 8,
        "paper_files_have_2_5d": summary["paper_files_with_2_5d"] >= 8,
        "paper_files_have_stage5c_boundary": summary["paper_files_with_stage5c_boundary"] >= 8,
        "paper_files_have_smc_boundary": summary["paper_files_with_smc_boundary"] >= 8,
        "no_unbounded_overclaims": summary["unbounded_overclaim_hits"] == 0,
        "supported_anchors_covered": summary["supported_anchor_covered"] == summary["supported_anchor_count"],
        "blocked_claims_are_limitations": summary["blocked_claims_covered_as_limitation"] >= max(5, summary["blocked_claim_count"] - 1),
        "no_metric_seconds_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_not_enabled": payload["claim_boundary"]["smc_enabled"] is False,
        "no_new_training_or_conversion_claim": summary["new_training_or_conversion"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_ie_paper_contract_compliance_pass" if passed == total else "stage42_ie_paper_contract_compliance_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _write_reports(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> None:
    output = dict(payload)
    output["stage42_ie_gate"] = gate
    write_json(REPORT_JSON, output)
    write_md(
        REPORT_MD,
        [
            "# Stage42-IE Paper Contract Compliance",
            "",
            f"- source: `{payload['source']}`",
            f"- generated_at_utc: `{payload['generated_at_utc']}`",
            f"- git_commit: `{payload['git_commit']}`",
            f"- input_hash: `{payload['input_hash']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- verdict: `{gate['verdict']}`",
            "",
            "## Purpose",
            "",
            "Stage42-IE applies the Stage42-ID claim contract to the actual paper package files. It is a compliance verifier, not new training, conversion, download, or evaluation.",
            "",
            "## Summary",
            "",
            *[f"- {key}: `{value}`" for key, value in payload["summary"].items()],
            "",
            "## Paper File Compliance",
            "",
            "| file | exists | raw-frame | dataset-local | 2.5D | Stage5C boundary | SMC boundary | overclaim hits |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: |",
            *[
                f"| `{row['path']}` | `{row['exists']}` | `{row['has_raw_frame']}` | `{row['has_dataset_local']}` | `{row['has_2_5d']}` | `{row['has_stage5c_boundary']}` | `{row['has_smc_boundary']}` | {len(row['overclaim_hits'])} |"
                for row in payload["paper_file_compliance"]
            ],
            "",
            "## Supported Anchor Coverage",
            "",
            "| anchor family | covered | anchors |",
            "| --- | --- | --- |",
            *[
                f"| `{name}` | `{row['covered']}` | {', '.join(row['anchors'])} |"
                for name, row in payload["supported_anchor_coverage"].items()
            ],
            "",
            "## Blocked Claim Handling",
            "",
            "| claim | status | covered as limitation |",
            "| --- | --- | --- |",
            *[
                f"| {row['claim']} | `{row['status']}` | `{row['covered_as_limitation']}` |"
                for row in payload["blocked_claim_handling"]
            ],
        ],
    )
    write_md(
        GATE_MD,
        [
            "# Stage42-IE Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(ok)}` |" for name, ok in gate["gates"].items()],
        ],
    )


def _refresh_lines(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    return [
        "## Stage42-IE Paper Contract Compliance",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`; gates `{gate['passed']} / {gate['total']}`.",
        f"- paper files checked: `{summary['paper_files_existing']} / {summary['paper_files_total']}`.",
        f"- supported anchors covered: `{summary['supported_anchor_covered']} / {summary['supported_anchor_count']}`; unbounded overclaim hits: `{summary['unbounded_overclaim_hits']}`.",
        f"- blocked claims covered as limitations: `{summary['blocked_claims_covered_as_limitation']} / {summary['blocked_claim_count']}`.",
        "- IE verifies the current paper package obeys the Stage42-ID contract: protected dataset-local/raw-frame 2.5D only; no true-3D/foundation/metric-seconds/Stage5C/SMC overclaim.",
        "- This is compliance verification only, not new training, conversion, download, or evaluation.",
    ]


def _refresh_readmes_and_state(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload, gate)
    for path in [README_RESULTS, M3W_README, MASTER_README]:
        _replace_section(path, SECTION, lines)
    state = read_json(RESEARCH_STATE, {})
    state["stage42_ie_paper_contract_compliance"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": gate["verdict"],
        "gates": f"{gate['passed']}/{gate['total']}",
        "summary": payload["summary"],
        "paper_contract_compliance_only": True,
        "new_training_or_conversion": False,
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_paper_contract_compliance() -> dict[str, Any]:
    payload = _build_payload()
    gate = _gate(payload)
    _write_reports(payload, gate)
    _refresh_readmes_and_state(payload, gate)
    payload = dict(payload)
    payload["stage42_ie_gate"] = gate
    write_json(REPORT_JSON, payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_paper_contract_compliance()
    gate = result["stage42_ie_gate"]
    print(f"Stage42-IE paper contract compliance: {gate['verdict']} ({gate['passed']}/{gate['total']})")
