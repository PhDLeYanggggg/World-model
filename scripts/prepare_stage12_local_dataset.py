from __future__ import annotations

import json
from src.stage12_pipeline import run_full_stage12_data_pipeline


if __name__ == "__main__":
    print(json.dumps(run_full_stage12_data_pipeline(), indent=2))
