from __future__ import annotations

from src.training.train_stage8_failure_predictor import train_stage8_failure_predictors


if __name__ == "__main__":
    payload = train_stage8_failure_predictors()
    print("Wrote outputs/reports/stage8_failure_predictor_comparison.md")
    for name, row in payload["variants"].items():
        print(name, row["test"])
