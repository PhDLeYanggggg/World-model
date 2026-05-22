from __future__ import annotations

import argparse

from src.stage18_pipeline import train_jepa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage18_jepa_quick.yaml")
    args = parser.parse_args()
    train_jepa(config=args.config)

