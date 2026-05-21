#!/usr/bin/env python
from __future__ import annotations

import json

from src.evaluation.stage8_goalbench_gold import build_goalbench_gold, write_outputs


if __name__ == "__main__":
    payload = write_outputs(build_goalbench_gold())
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2))

