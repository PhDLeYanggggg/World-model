from __future__ import annotations

from src.training.train_stage9_interaction_auxiliary import collect_aux_rows, evaluate_aux


def evaluate_stage9_interaction_auxiliary(model):
    return evaluate_aux(model, collect_aux_rows("test"))
