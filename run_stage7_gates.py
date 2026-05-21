#!/usr/bin/env python
from __future__ import annotations

import json

from src.evaluation.stage7_gates import evaluate_gates, write_report


if __name__ == "__main__":
    payload = evaluate_gates()
    write_report(payload)
    print(json.dumps(payload, indent=2))

