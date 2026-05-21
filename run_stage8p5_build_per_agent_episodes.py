from __future__ import annotations

import argparse

from src.data_unification.stage8p5_per_agent_episode_builder import build_per_agent_episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all_available")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--future", type=int, default=100)
    args = parser.parse_args()
    datasets = None if args.datasets == "all_available" else [x.strip() for x in args.datasets.split(",") if x.strip()]
    payload = build_per_agent_episodes(datasets=datasets, future_horizon=args.future, quick=args.quick)
    print(payload)


if __name__ == "__main__":
    main()
