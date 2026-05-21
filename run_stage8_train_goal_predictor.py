#!/usr/bin/env python
from __future__ import annotations

import json

from src.training.train_stage8_goal_predictor import train_goal_predictor_v2


if __name__ == "__main__":
    print(json.dumps(train_goal_predictor_v2(), indent=2))

