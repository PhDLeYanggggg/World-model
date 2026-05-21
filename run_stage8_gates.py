from __future__ import annotations

from src.evaluation.stage8_gates import evaluate_gates, write_report


if __name__ == "__main__":
    result = evaluate_gates()
    write_report(result)
    print(f"Stage 8 gates: {result['passed']} / {result['total']}")
    print(result["verdict"])
