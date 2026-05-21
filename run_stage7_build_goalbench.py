#!/usr/bin/env python
from __future__ import annotations

import json

from src.evaluation.goalbench_builder import build_goalbench, write_outputs


if __name__ == "__main__":
    payload = write_outputs(build_goalbench())
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2))

