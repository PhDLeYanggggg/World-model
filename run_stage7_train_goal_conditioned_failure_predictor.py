#!/usr/bin/env python
from __future__ import annotations

import json

from src.training.train_stage7_failure_predictor import train_stage7_failure_predictors


if __name__ == "__main__":
    print(json.dumps(train_stage7_failure_predictors(), indent=2))

