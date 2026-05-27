from __future__ import annotations

from src.stage42_ucy_h100_terms_gated_conversion_preflight import (
    run_stage42_ucy_h100_terms_gated_conversion_preflight,
)


if __name__ == "__main__":
    payload = run_stage42_ucy_h100_terms_gated_conversion_preflight()
    gate = payload["stage42_fr_gate"]
    print(f"Stage42-FR gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
