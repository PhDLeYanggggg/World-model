from __future__ import annotations

import json

from src.evaluation.stage15_benchmark import benchmark


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2, ensure_ascii=False, default=str))

