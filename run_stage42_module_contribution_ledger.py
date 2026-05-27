from __future__ import annotations

from src.stage42_module_contribution_ledger import run_stage42_module_contribution_ledger


if __name__ == "__main__":
    payload = run_stage42_module_contribution_ledger()
    gate = payload["stage42_fu_gate"]
    print(f"Stage42-FU gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
