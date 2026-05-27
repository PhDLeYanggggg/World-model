from __future__ import annotations

from src.stage42_ucy_supported_fe_composer import run_stage42_ucy_supported_fe_composer


if __name__ == "__main__":
    result = run_stage42_ucy_supported_fe_composer()
    gate = result["stage42_fh_gate"]
    print(f"Stage42-FH gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
