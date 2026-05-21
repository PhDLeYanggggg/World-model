from __future__ import annotations

from src.training.train_stage8_interaction_auxiliary import train_stage8_interaction_auxiliary


if __name__ == "__main__":
    payload = train_stage8_interaction_auxiliary()
    print("Wrote outputs/reports/stage8_interaction_ablation.md")
    print(payload["metrics"])
