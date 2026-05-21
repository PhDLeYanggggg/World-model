#!/usr/bin/env python
from __future__ import annotations

import json

from src.training.train_stage7_goal_conditioned_world_model import train_stage7_world_models


if __name__ == "__main__":
    print(json.dumps(train_stage7_world_models(), indent=2))

