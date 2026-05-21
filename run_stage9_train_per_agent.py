from __future__ import annotations

from src.training.train_stage9_per_agent import train_stage9_models


if __name__ == "__main__":
    payload = train_stage9_models()
    print("trained variants:", [v["variant"] for v in payload["variants"]])
