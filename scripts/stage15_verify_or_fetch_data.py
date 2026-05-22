from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stage15_pipeline import run_stage15_data_verify


if __name__ == "__main__":
    print(json.dumps(run_stage15_data_verify(), indent=2, ensure_ascii=False, default=str))

