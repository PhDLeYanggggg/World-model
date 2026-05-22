from __future__ import annotations

import json

from src.stage14_pipeline import stage14_current_state


if __name__ == "__main__":
    print(json.dumps(stage14_current_state(), indent=2, ensure_ascii=False, default=str))
