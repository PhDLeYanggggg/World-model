from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from .research_state import read_json, read_text


REPORT_DIR = Path("outputs/reports")


STAGE_REPORT_CANDIDATES = {
    "15": [REPORT_DIR / "report_stage15_final.md", Path("outputs/world_model_stage15_results/reports/report_stage15_final.md")],
    "14": [REPORT_DIR / "report_stage14_final.md", Path("outputs/world_model_stage14_results/reports/report_stage14_final.md")],
    "13": [REPORT_DIR / "report_stage13_final.md", Path("outputs/world_model_stage13_results/reports/report_stage13_final.md")],
    "12": [REPORT_DIR / "report_stage12_final.md", Path("outputs/world_model_stage12_results/reports/report_stage12_final.md")],
    "11": [REPORT_DIR / "report_stage11_final.md", Path("outputs/world_model_stage11_results/reports/report_stage11_final.md")],
    "10": [REPORT_DIR / "report_stage10_final.md", Path("outputs/world_model_stage10_results/reports/report_stage10_final.md")],
    "9": [REPORT_DIR / "report_stage9_final.md", Path("outputs/world_model_stage9_results/reports/report_stage9_final.md")],
}

GATE_REPORT_CANDIDATES = {
    "15": [REPORT_DIR / "world_model_gate_stage15.md", Path("outputs/world_model_stage15_results/reports/world_model_gate_stage15.md")],
    "14": [REPORT_DIR / "world_model_gate_stage14.md", Path("outputs/world_model_stage14_results/reports/world_model_gate_stage14.md")],
    "13": [REPORT_DIR / "world_model_gate_stage13.md", Path("outputs/world_model_stage13_results/reports/world_model_gate_stage13.md")],
    "12": [REPORT_DIR / "world_model_gate_stage12.md", Path("outputs/world_model_stage12_results/reports/world_model_gate_stage12.md")],
    "11": [REPORT_DIR / "world_model_gate_stage11.md", Path("outputs/world_model_stage11_results/reports/world_model_gate_stage11.md")],
    "10": [REPORT_DIR / "world_model_gate_stage10.md", Path("outputs/world_model_stage10_results/reports/world_model_gate_stage10.md")],
    "9": [REPORT_DIR / "world_model_gate_stage9.md", Path("outputs/world_model_stage9_results/reports/world_model_gate_stage9.md")],
}


def first_existing(paths: List[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def latest_completed_stage() -> Dict[str, Any]:
    missing = []
    for stage in ["15", "14", "13", "12", "11", "10", "9"]:
        report = first_existing(STAGE_REPORT_CANDIDATES[stage])
        gate = first_existing(GATE_REPORT_CANDIDATES[stage])
        if report:
            return {"stage": stage, "report_path": str(report), "gate_path": str(gate) if gate else None, "missing_newer": missing}
        missing.append(f"stage{stage}")
    return {"stage": "unknown", "report_path": None, "gate_path": None, "missing_newer": missing}


def extract_int(pattern: str, text: str, default: int = 0) -> int:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else default


def extract_bool(pattern: str, text: str, default: bool = False) -> bool:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return default
    return match.group(1).strip().lower() in {"true", "yes", "是", "pass"}


def extract_backtick_or_value(label: str, text: str, default: str = "unknown") -> str:
    patterns = [
        rf"{re.escape(label)}\s*[:：=]\s*`([^`]+)`",
        rf"{re.escape(label)}\s*[:：=]\s*([^\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return default


def parse_gate_table(md_text: str) -> Dict[str, List[str]]:
    passed: List[str] = []
    failed: List[str] = []
    for line in md_text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 3 or cells[0].lower() in {"gate", "---"}:
            continue
        gate, status, pass_cell = cells[:3]
        value = pass_cell.lower()
        if value in {"true", "pass", "yes"} or status.lower() == "pass":
            passed.append(gate)
        elif value in {"false", "fail", "no"} or status.lower() in {"fail", "failed"}:
            failed.append(gate)
    return {"passed": passed, "failed": failed}


def summarize_latest_reports() -> Dict[str, Any]:
    latest = latest_completed_stage()
    report_text = read_text(latest["report_path"]) if latest["report_path"] else ""
    gate_text = read_text(latest["gate_path"]) if latest["gate_path"] else ""
    gate_json_path = Path(str(latest["gate_path"]).replace(".md", ".json")) if latest["gate_path"] else None
    gate_json = read_json(gate_json_path, default={}) if gate_json_path else {}
    gate_lists = parse_gate_table(gate_text)

    score = extract_int(r"expert[_ ]audit[_ ]score(?:：|:| = |`?=)\s*`?(\d+)", report_text + "\n" + gate_text, 0)
    verdict = extract_backtick_or_value("当前 verdict", report_text, "unknown")
    if verdict == "unknown":
        verdict = extract_backtick_or_value("verdict", report_text + "\n" + gate_text, "unknown")

    return {
        "latest_stage": latest["stage"],
        "latest_report_path": latest["report_path"],
        "latest_gate_path": latest["gate_path"],
        "missing_newer_reports": latest["missing_newer"],
        "expert_audit_score": score,
        "verdict": verdict,
        "stage13_ready": "stage13_ready: `True`" in gate_text or "stage13_ready = true" in report_text,
        "latent_generative_ready": "latent_stage5c_ready = true" in report_text.lower() or "Stage 5C Readiness Gate | pass" in gate_text,
        "smc_ready": "smc_ready = true" in report_text.lower(),
        "gates_passed": gate_json.get("gates_passed") or gate_lists["passed"],
        "gates_failed": gate_json.get("gates_failed") or gate_lists["failed"],
        "report_text": report_text,
        "gate_text": gate_text,
    }
