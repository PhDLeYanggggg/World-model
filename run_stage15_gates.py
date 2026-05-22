from __future__ import annotations

import json

from src.evaluation.stage15_gates import evaluate


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, ensure_ascii=False, default=str))
