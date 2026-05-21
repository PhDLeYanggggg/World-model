from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List


def audit_licenses(rows: Iterable[Dict]) -> List[Dict]:
    audited = []
    for row in rows:
        license_text = str(row.get("license", "unknown")).lower()
        status = "ok"
        risk = []
        if "unknown" in license_text or row.get("commercial_use_allowed") == "unknown":
            status = "needs_review"
            risk.append("license or commercial terms unknown")
        if str(row.get("download_status")) in {"gated", "requires_application"}:
            status = "gated"
            risk.append("requires registration/application; do not auto-download")
        if "noncommercial" in license_text or row.get("commercial_use_allowed") == "no":
            risk.append("non-commercial only")
        if "nd" in license_text or "no_derivatives" in str(row.get("redistribution_allowed")):
            risk.append("redistribution/derivatives restricted")
        audited.append(
            {
                "dataset_name": row.get("dataset_name"),
                "download_status": row.get("download_status"),
                "license": row.get("license"),
                "commercial_use_allowed": row.get("commercial_use_allowed"),
                "redistribution_allowed": row.get("redistribution_allowed"),
                "license_audit_status": status,
                "risk_notes": "; ".join(risk) or "no obvious registry risk",
            }
        )
    return audited


def write_license_report(rows: Iterable[Dict], path: str | Path = "outputs/reports/stage5_data/license_audit_stage5.md") -> List[Dict]:
    audited = audit_licenses(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Stage 5 License Audit", "", "| dataset | status | license | commercial | redistribution | risk |", "| --- | --- | --- | --- | --- | --- |"]
    for row in audited:
        lines.append(
            f"| {row['dataset_name']} | {row['license_audit_status']} | {row['license']} | {row['commercial_use_allowed']} | {row['redistribution_allowed']} | {row['risk_notes']} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audited
