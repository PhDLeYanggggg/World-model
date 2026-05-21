from __future__ import annotations

from src.models.stage8_goal_predictor_v2 import Stage8GoalPredictorV2
from src.training.train_stage8_goal_predictor import evaluate, load_records


def evaluate_checkpoint(path: str):
    return evaluate(Stage8GoalPredictorV2.load(path), [r for r in load_records() if r["split"] == "test"])

