from __future__ import annotations

from src.stage42_h100_source_support_repair_queue import run_stage42_h100_source_support_repair_queue


if __name__ == "__main__":
    payload = run_stage42_h100_source_support_repair_queue()
    gate = payload["stage42_fq_gate"]
    print(f"Stage42-FQ gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
