#!/usr/bin/env python
from __future__ import annotations

import json

from src.training.train_goal_intent_predictor import train_goal_predictor


if __name__ == "__main__":
    print(json.dumps(train_goal_predictor(), indent=2))

