from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List


def leakage_audit_dataset(dataset_name: str, metadata: Dict, official_velocity_source: str = "causal_fd") -> Dict:
    flags: List[str] = []
    if official_velocity_source in {"central", "central_fd"}:
        flags.append("official input uses central difference velocity")
    if "kf" in str(metadata.get("columns_inferred", {})).lower() and official_velocity_source == "native":
        flags.append("native Kalman-filtered velocity may include smoothing leakage")
    if metadata.get("normalization_uses_test_stats"):
        flags.append("normalization uses test statistics")
    return {
        "dataset_name": dataset_name,
        "official_velocity_source": official_velocity_source,
        "uses_central_difference_officially": official_velocity_source in {"central", "central_fd"},
        "native_velocity_leakage_risk": "kf" in str(metadata.get("columns_inferred", {})).lower(),
        "future_goal_used": False,
        "test_stats_used": bool(metadata.get("normalization_uses_test_stats", False)),
        "leakage_flags": flags,
        "passed": len(flags) == 0,
    }


def write_leakage_report(rows: Iterable[Dict], path: str | Path = "outputs/reports/leakage_audit_stage5.md") -> List[Dict]:
    audits = list(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Stage 5 Leakage Audit", "", "| dataset | passed | official velocity | native velocity risk | flags |", "| --- | --- | --- | --- | --- |"]
    for row in audits:
        lines.append(f"| {row['dataset_name']} | {row['passed']} | {row['official_velocity_source']} | {row['native_velocity_leakage_risk']} | {', '.join(row['leakage_flags'])} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audits
