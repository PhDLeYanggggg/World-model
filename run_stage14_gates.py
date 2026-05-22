from __future__ import annotations

import json

from src.evaluation.stage14_gates import evaluate_gates


if __name__ == "__main__":
    print(json.dumps(evaluate_gates(), indent=2, ensure_ascii=False, default=str))
