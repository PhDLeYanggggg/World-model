from __future__ import annotations

from src.stage42_module_claim_lock import run_stage42_module_claim_lock


if __name__ == "__main__":
    payload = run_stage42_module_claim_lock()
    gate = payload["stage42_gj_gate"]
    print(f"Stage42-GJ gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
