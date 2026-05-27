from __future__ import annotations

from src.stage42_fh_policy_freeze_replay import run_stage42_fh_policy_freeze_replay


if __name__ == "__main__":
    result = run_stage42_fh_policy_freeze_replay()
    gate = result["stage42_fi_gate"]
    print(f"Stage42-FI gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
