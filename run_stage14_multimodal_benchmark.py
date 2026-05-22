from __future__ import annotations

import json

from src.evaluation.stage14_multimodal_benchmark import run_benchmark


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2, ensure_ascii=False, default=str))

