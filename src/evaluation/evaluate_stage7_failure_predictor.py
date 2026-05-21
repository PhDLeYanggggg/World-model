from __future__ import annotations

from src.models.stage7_goal_conditioned_failure_predictor import Stage7GoalConditionedFailurePredictor
from src.training.stage7_common import collect_stage7_examples
from src.training.train_stage7_failure_predictor import evaluate_rows


def evaluate_checkpoint(path: str, split: str = "test"):
    model = Stage7GoalConditionedFailurePredictor.load(path)
    rows = collect_stage7_examples(split, model.payload.get("feature_mode", "goal_scene_interaction"))
    return evaluate_rows(model, rows)

