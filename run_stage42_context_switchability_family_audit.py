from __future__ import annotations

from src.stage42_context_switchability_family_audit import run_stage42_context_switchability_family_audit


if __name__ == "__main__":
    payload = run_stage42_context_switchability_family_audit()
    gate = payload["stage42_gk_gate"]
    print(f"Stage42-GK gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
