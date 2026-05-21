from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.data.stage10_unified_pedestrian_loader import audit_stage10_sources
from src.stage10_common import REPORT_DIR, write_markdown_table


def build_horizon_audit() -> Dict:
    rows = []
    for item in audit_stage10_sources():
        dt = item.get("fps_or_dt")
        rows.append(
            {
                "dataset_name": item["dataset_name"],
                "original_fps": None if not dt else round(1.0 / dt, 6) if dt and dt > 0 else None,
                "dt_seconds": dt,
                "raw_frame_horizon": item.get("max_track_length", 0),
                "physical_time_t10": seconds(dt, 10),
                "physical_time_t25": seconds(dt, 25),
                "physical_time_t50": seconds(dt, 50),
                "physical_time_t100": seconds(dt, 100),
                "track_count": item.get("track_count", 0),
                "mean_track_length": item.get("mean_track_length", 0.0),
                "t50_sample_count": item.get("samples_t50", 0),
                "t100_sample_count": item.get("samples_t100", 0),
                "downsampling_used": False,
                "horizon_is_raw_or_downsampled": "raw_converted_rows",
                "downsampling_loses_interaction_detail": "not_applicable",
                "official_verified_t50": item.get("actual_verified_t50", False),
                "official_verified_t100": item.get("actual_verified_t100", False),
                "usable_for_stage11_training": bool(item.get("eligible_for_stage10") and item.get("samples_t10", 0) > 0),
                "honest_note": note(item),
            }
        )
    return {"stage": "10", "horizons": rows}


def seconds(dt, horizon: int):
    return None if dt is None else float(dt) * horizon


def note(item: Dict) -> str:
    if item.get("actual_verified_t50") or item.get("actual_verified_t100"):
        return "Verified long horizon exists in converted pedestrian/drone table."
    if item.get("samples_t10", 0) > 0:
        return "Only short-horizon pedestrian/drone evaluation is available; do not claim t+50/t+100."
    return item.get("failure_reason_if_not_eligible", "No usable local converted trajectories.")


def write_horizon_audit(payload: Dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage10_horizon_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_table(REPORT_DIR / "stage10_horizon_audit.md", "Stage 10 Horizon Audit", payload["horizons"])


def main() -> None:
    payload = build_horizon_audit()
    write_horizon_audit(payload)
    print(json.dumps({"datasets": len(payload["horizons"]), "verified_t50_or_t100": sum(r["official_verified_t50"] or r["official_verified_t100"] for r in payload["horizons"])}, indent=2))


if __name__ == "__main__":
    main()
