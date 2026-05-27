from __future__ import annotations

from src.stage42_source_specific_conversion_dry_run import run_stage42_source_specific_conversion_dry_run


if __name__ == "__main__":
    payload = run_stage42_source_specific_conversion_dry_run()
    gate = payload["stage42_dw_gate"]
    print(f"Stage42-DW source-specific conversion dry-run: {gate['passed']}/{gate['total']} {gate['verdict']}")
