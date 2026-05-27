from __future__ import annotations

from src.stage42_claim_boundary_linter import run_stage42_claim_boundary_linter


if __name__ == "__main__":
    payload = run_stage42_claim_boundary_linter()
    gate = payload["stage42_fv_gate"]
    print(f"Stage42-FV gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
