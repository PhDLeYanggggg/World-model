from __future__ import annotations

import json

from src.evaluation.stage5b6_gates import evaluate_gates, write_report


def main() -> int:
    result = evaluate_gates()
    write_report(result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

