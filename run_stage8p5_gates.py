from __future__ import annotations

from src.evaluation.stage8p5_gates import evaluate_gates, write_stage8p5_gate_report


if __name__ == "__main__":
    result = evaluate_gates()
    write_stage8p5_gate_report(result)
    print(f"Stage 8.5 gates: {result['passed']} / {result['total']}")
    print(result["verdict"])
