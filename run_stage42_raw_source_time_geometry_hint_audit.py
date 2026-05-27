from __future__ import annotations

from src.stage42_raw_source_time_geometry_hint_audit import (
    run_stage42_raw_source_time_geometry_hint_audit,
)


if __name__ == "__main__":
    payload = run_stage42_raw_source_time_geometry_hint_audit()
    gate = payload["stage42_du_gate"]
    print(
        "Stage42-DU raw source time/geometry hint audit: "
        f"{gate['passed']}/{gate['total']} {gate['verdict']}"
    )
