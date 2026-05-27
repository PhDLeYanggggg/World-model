from __future__ import annotations

import csv
import io
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

CURRENT_CLAIM_JSON = OUT_DIR / "current_claim_evidence_closure_stage42.json"
PAPER_OUTLINE_MD = OUT_DIR / "paper_outline_stage42.md"
EXPERIMENT_TABLES_MD = OUT_DIR / "experiment_tables_stage42.md"
ABLATION_TABLES_MD = OUT_DIR / "ablation_tables_stage42.md"
FAILURE_TAXONOMY_MD = OUT_DIR / "failure_taxonomy_stage42.md"
MODEL_CARD_MD = OUT_DIR / "model_card_stage42.md"
DATA_CARD_MD = OUT_DIR / "data_card_stage42.md"
REPRODUCIBILITY_MD = OUT_DIR / "reproducibility_stage42.md"
AJOURNAL_GAP_MD = OUT_DIR / "a_journal_gap_stage42.md"

REPORT_JSON = OUT_DIR / "paper_claim_contract_stage42.json"
REPORT_MD = OUT_DIR / "paper_claim_contract_stage42.md"
REPORT_CSV = OUT_DIR / "paper_claim_contract_stage42.csv"
GATE_MD = OUT_DIR / "stage42_stage_id_gate.md"

SOURCE = "fresh_stage42_id_paper_claim_contract"
SECTION = "STAGE42_ID_PAPER_CLAIM_CONTRACT"

PAPER_FILES = [
    PAPER_OUTLINE_MD,
    EXPERIMENT_TABLES_MD,
    ABLATION_TABLES_MD,
    FAILURE_TAXONOMY_MD,
    MODEL_CARD_MD,
    DATA_CARD_MD,
    REPRODUCIBILITY_MD,
    AJOURNAL_GAP_MD,
]

FORBIDDEN_CLAIM_PHRASES = [
    "true 3D world model",
    "foundation world model",
    "metric prediction",
    "seconds-level",
    "Stage5C executed",
    "SMC enabled",
]

MANDATORY_CAVEATS = [
    "dataset-local/raw-frame",
    "2.5D",
    "not metric/seconds-level",
    "Stage5C false",
    "SMC false",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _gate_pass(gate: Mapping[str, Any]) -> bool:
    try:
        return int(gate.get("passed", -1)) == int(gate.get("total", 0)) and int(gate.get("total", 0)) > 0
    except Exception:
        return False


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _paper_file_status() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in PAPER_FILES:
        text = _read_text(path)
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "bytes": len(text.encode("utf-8")),
                "has_raw_frame_or_dataset_local_caveat": ("raw-frame" in text) or ("dataset-local" in text),
                "mentions_stage5c_or_smc_boundary": ("Stage5C" in text) and ("SMC" in text),
            }
        )
    return rows


def _contract_rows(closure: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    supported = list(closure.get("supported_claims", []))
    blocked = list(closure.get("blocked_claims", []))
    for claim in supported:
        text = str(claim.get("claim", ""))
        paper_use = str(claim.get("paper_use", ""))
        rows.append(
            {
                "claim": text,
                "source_status": claim.get("status", "unknown"),
                "paper_role": paper_use,
                "allowed_language": _allowed_language(text, paper_use),
                "forbidden_language": "; ".join(FORBIDDEN_CLAIM_PHRASES),
                "evidence": claim.get("evidence", ""),
                "claim_boundary": "must retain dataset-local/raw-frame 2.5D caveat unless future calibration verifies otherwise",
                "deployment_boundary": _deployment_boundary(text),
            }
        )
    for claim in blocked:
        text = str(claim.get("claim", ""))
        rows.append(
            {
                "claim": text,
                "source_status": claim.get("status", "blocked"),
                "paper_role": "blocked_or_limitation_only",
                "allowed_language": f"Limitation/blocker: {claim.get('reason', '')}",
                "forbidden_language": "do not present as achieved contribution",
                "evidence": claim.get("reason", ""),
                "claim_boundary": "blocked claim; may appear only in limitations or next steps",
                "deployment_boundary": "not deployable",
            }
        )
    return rows


def _allowed_language(claim: str, paper_use: str) -> str:
    if "Stage26" in claim:
        return "SDD deployable baseline under pixel-space raw-frame evaluation."
    if "Stage37" in claim:
        return "External dataset-local raw-frame t50 safety floor / deployable selector baseline."
    if "Neural" in claim:
        return "Protected neural world-state candidate; not ungated neural deployment."
    if "full-waypoint" in claim or "group-consistency" in claim:
        return "Protected full-waypoint/group-consistency 2.5D world-state evidence."
    if "t100" in claim:
        return "Raw-frame t100 replay/runtime evidence only, not seconds-level long-horizon claim."
    return paper_use or "Supported only with strict raw-frame/dataset-local caveats."


def _deployment_boundary(claim: str) -> str:
    if "Stage26" in claim:
        return "deployable for SDD fallback/selector use only"
    if "Stage37" in claim:
        return "deployable external t50 safety floor under current dataset-local setup"
    if "Neural" in claim or "full-waypoint" in claim or "group-consistency" in claim:
        return "protected deployment only with floor/safe-switch; not floor-free"
    if "t100" in claim:
        return "diagnostic raw-frame runtime replay"
    return "claim-specific protected use only"


def _csv_text(rows: list[Mapping[str, Any]]) -> str:
    fieldnames = [
        "claim",
        "source_status",
        "paper_role",
        "allowed_language",
        "forbidden_language",
        "evidence",
        "claim_boundary",
        "deployment_boundary",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return buffer.getvalue()


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    closure = read_json(CURRENT_CLAIM_JSON, {})
    closure_gate = closure.get("stage42_ic_gate", {})
    contract = _contract_rows(closure)
    paper_status = _paper_file_status()
    input_files = [CURRENT_CLAIM_JSON, *PAPER_FILES]
    summary = {
        "closure_verdict": closure_gate.get("verdict"),
        "closure_gate_passed": _gate_pass(closure_gate),
        "supported_claim_count": len(closure.get("supported_claims", [])),
        "blocked_claim_count": len(closure.get("blocked_claims", [])),
        "contract_row_count": len(contract),
        "paper_files_total": len(paper_status),
        "paper_files_existing": sum(1 for row in paper_status if row["exists"]),
        "paper_files_with_claim_caveat": sum(1 for row in paper_status if row["has_raw_frame_or_dataset_local_caveat"]),
        "paper_files_with_stage5c_smc_boundary": sum(1 for row in paper_status if row["mentions_stage5c_or_smc_boundary"]),
        "metric_seconds_claim_allowed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
        "new_training_or_conversion": False,
    }
    payload = {
        "stage": "Stage42-ID",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(input_files),
        "summary": summary,
        "contract": contract,
        "paper_file_status": paper_status,
        "mandatory_caveats": MANDATORY_CAVEATS,
        "forbidden_claim_phrases": FORBIDDEN_CLAIM_PHRASES,
        "actions": {
            "downloaded": False,
            "converted": False,
            "trained": False,
            "evaluated": False,
            "paper_claim_contract_only": True,
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
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    statuses = list(payload["paper_file_status"])
    contract = list(payload["contract"])
    supported_rows = [row for row in contract if "blocked" not in str(row["source_status"])]
    blocked_rows = [row for row in contract if "blocked" in str(row["source_status"]) or row["paper_role"] == "blocked_or_limitation_only"]
    gates = {
        "stage42_ic_closure_passed": bool(summary["closure_gate_passed"]),
        "supported_claims_imported": summary["supported_claim_count"] >= 5,
        "blocked_claims_imported": summary["blocked_claim_count"] >= 6,
        "contract_rows_complete": summary["contract_row_count"] == summary["supported_claim_count"] + summary["blocked_claim_count"],
        "paper_files_exist": summary["paper_files_existing"] == summary["paper_files_total"],
        "paper_files_have_claim_caveats": summary["paper_files_with_claim_caveat"] >= 6,
        "paper_files_have_stage5c_smc_boundary": summary["paper_files_with_stage5c_smc_boundary"] >= 6,
        "supported_rows_have_boundaries": all(row.get("claim_boundary") for row in supported_rows),
        "blocked_rows_are_limitation_only": all(row["paper_role"] == "blocked_or_limitation_only" for row in blocked_rows),
        "no_metric_seconds_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_not_enabled": payload["claim_boundary"]["smc_enabled"] is False,
        "no_new_training_or_conversion_claim": summary["new_training_or_conversion"] is False,
        "contract_has_deployment_boundaries": all(row.get("deployment_boundary") for row in contract),
        "paper_package_not_empty": all(row["bytes"] > 100 for row in statuses),
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_id_paper_claim_contract_pass" if passed == total else "stage42_id_paper_claim_contract_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _write_reports(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> None:
    output_payload = dict(payload)
    output_payload["stage42_id_gate"] = gate
    write_json(REPORT_JSON, output_payload)
    REPORT_CSV.write_text(_csv_text(list(payload["contract"])), encoding="utf-8")
    write_md(
        REPORT_MD,
        [
            "# Stage42-ID Paper Claim Contract",
            "",
            f"- source: `{payload['source']}`",
            f"- generated_at_utc: `{payload['generated_at_utc']}`",
            f"- git_commit: `{payload['git_commit']}`",
            f"- input_hash: `{payload['input_hash']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- verdict: `{gate['verdict']}`",
            "",
            "## Meaning",
            "",
            "This is a paper-package claim contract over the current Stage42 evidence closure. It does not train, convert, download, or evaluate data.",
            "It turns supported and blocked claims into explicit allowed language, forbidden language, deployment boundaries, and evidence pointers.",
            "",
            "## Mandatory Caveats",
            "",
            *[f"- `{item}`" for item in payload["mandatory_caveats"]],
            "",
            "## Contract",
            "",
            "| claim | source status | paper role | allowed language | deployment boundary |",
            "| --- | --- | --- | --- | --- |",
            *[
                f"| {row['claim']} | `{row['source_status']}` | {row['paper_role']} | {row['allowed_language']} | {row['deployment_boundary']} |"
                for row in payload["contract"]
            ],
            "",
            "## Paper File Status",
            "",
            "| file | exists | bytes | raw/dataset caveat | Stage5C/SMC boundary |",
            "| --- | --- | ---: | --- | --- |",
            *[
                f"| `{row['path']}` | `{row['exists']}` | {row['bytes']} | `{row['has_raw_frame_or_dataset_local_caveat']}` | `{row['mentions_stage5c_or_smc_boundary']}` |"
                for row in payload["paper_file_status"]
            ],
            "",
            "## Summary",
            "",
            *[f"- {key}: `{value}`" for key, value in payload["summary"].items()],
        ],
    )
    write_md(
        GATE_MD,
        [
            "# Stage42-ID Gate",
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
        "## Stage42-ID Paper Claim Contract",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`; gates `{gate['passed']} / {gate['total']}`.",
        f"- contract rows: `{summary['contract_row_count']}`; supported claims `{summary['supported_claim_count']}`; blocked claims `{summary['blocked_claim_count']}`.",
        f"- paper files existing: `{summary['paper_files_existing']} / {summary['paper_files_total']}`; files with raw/dataset caveat `{summary['paper_files_with_claim_caveat']}`; files with Stage5C/SMC boundary `{summary['paper_files_with_stage5c_smc_boundary']}`.",
        "- ID locks manuscript wording: supported claims are protected dataset-local/raw-frame 2.5D; true-3D/foundation/metric-seconds/Stage5C/SMC claims remain forbidden.",
        "- This is a paper-claim contract only, not new training, conversion, download, or evaluation.",
    ]


def _refresh_readmes_and_state(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload, gate)
    for path in [README_RESULTS, M3W_README, MASTER_README]:
        _replace_section(path, SECTION, lines)
    state = read_json(RESEARCH_STATE, {})
    state["stage42_id_paper_claim_contract"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "csv": str(REPORT_CSV),
        "gate": str(GATE_MD),
        "verdict": gate["verdict"],
        "gates": f"{gate['passed']}/{gate['total']}",
        "summary": payload["summary"],
        "paper_claim_contract_only": True,
        "new_training_or_conversion": False,
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_paper_claim_contract() -> dict[str, Any]:
    payload = _build_payload()
    gate = _gate(payload)
    _write_reports(payload, gate)
    _refresh_readmes_and_state(payload, gate)
    payload = dict(payload)
    payload["stage42_id_gate"] = gate
    write_json(REPORT_JSON, payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_paper_claim_contract()
    gate = result["stage42_id_gate"]
    print(f"Stage42-ID paper claim contract: {gate['verdict']} ({gate['passed']}/{gate['total']})")
