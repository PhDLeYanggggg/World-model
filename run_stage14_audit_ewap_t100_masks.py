from __future__ import annotations

import json

from src.data_unification.stage14_per_agent_mask_audit import run_audit


if __name__ == "__main__":
    print(json.dumps(run_audit(), indent=2, ensure_ascii=False, default=str))

