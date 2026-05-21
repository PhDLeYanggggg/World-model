from __future__ import annotations

from src.training.train_stage8_world_model import train_stage8_world_models


if __name__ == "__main__":
    payload = train_stage8_world_models()
    print("Wrote outputs/reports/stage8_goal_conditioned_world_model_training.json")
    for row in payload["variants"]:
        print(row)
