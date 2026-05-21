#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from src.data_unification.stage8_multiagent_episode_builder import build_stage8_multiagent_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all_available")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    datasets = None if args.datasets == "all_available" else [x.strip() for x in args.datasets.split(",") if x.strip()]
    print(json.dumps(build_stage8_multiagent_episodes(datasets=datasets, quick=True if args.quick else True), indent=2))

