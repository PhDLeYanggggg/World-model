from __future__ import annotations

import json

from src.evaluation.stage5b_gates import evaluate_stage5b_gates, write_gate_report


def main() -> int:
    result = evaluate_stage5b_gates()
    write_gate_report(result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
