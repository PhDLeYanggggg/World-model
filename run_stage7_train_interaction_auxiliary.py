#!/usr/bin/env python
from __future__ import annotations

import json

from src.training.train_stage7_interaction_auxiliary import train_interaction_auxiliary


if __name__ == "__main__":
    print(json.dumps(train_interaction_auxiliary(), indent=2))

