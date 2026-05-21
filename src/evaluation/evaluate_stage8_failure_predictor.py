from __future__ import annotations

from src.models.stage8_failure_predictor_v2 import Stage8FailurePredictorV2
from src.training.stage8_common import collect_stage8_examples
from src.training.train_stage8_failure_predictor import evaluate


def evaluate_checkpoint(path: str, split: str = "test"):
    return evaluate(Stage8FailurePredictorV2.load(path), collect_stage8_examples(split))
