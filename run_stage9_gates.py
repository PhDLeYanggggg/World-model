from __future__ import annotations

from src.evaluation.stage9_gates import evaluate_stage9_gates, write_stage9_gates


if __name__ == "__main__":
    result = evaluate_stage9_gates()
    write_stage9_gates(result)
    print(f"Stage 9 gates: {result['passed']} / {result['total']}")
    print(result["verdict"])
