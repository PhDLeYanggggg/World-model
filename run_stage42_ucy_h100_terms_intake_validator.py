from __future__ import annotations

from src.stage42_ucy_h100_terms_intake_validator import run_stage42_ucy_h100_terms_intake_validator


if __name__ == "__main__":
    payload = run_stage42_ucy_h100_terms_intake_validator()
    gate = payload["stage42_fs_gate"]
    print(f"Stage42-FS gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
