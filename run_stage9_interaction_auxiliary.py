from __future__ import annotations

from src.training.train_stage9_interaction_auxiliary import train_stage9_interaction_auxiliary


if __name__ == "__main__":
    payload = train_stage9_interaction_auxiliary()
    print(payload["test"])
