from __future__ import annotations

from src.stage42_unified_guarded_conversion_queue import run_stage42_unified_guarded_conversion_queue


if __name__ == "__main__":
    payload = run_stage42_unified_guarded_conversion_queue()
    gate = payload["stage42_ft_gate"]
    print(f"Stage42-FT gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
