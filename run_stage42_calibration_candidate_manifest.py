from __future__ import annotations

from src.stage42_calibration_candidate_manifest import run_stage42_calibration_candidate_manifest


if __name__ == "__main__":
    payload = run_stage42_calibration_candidate_manifest()
    gate = payload["stage42_dv_gate"]
    print(f"Stage42-DV calibration candidate manifest: {gate['passed']}/{gate['total']} {gate['verdict']}")
