from __future__ import annotations

from src.training.train_goal_intent_predictor import evaluate_goal_records, load_records
from src.models.goal_intent_predictor import GoalIntentPredictor


def evaluate_checkpoint(path: str):
    model = GoalIntentPredictor.load(path)
    records = load_records()
    return evaluate_goal_records(model, [r for r in records if r["split"] == "test"])

