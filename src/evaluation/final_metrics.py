from __future__ import annotations

import numpy as np


def ade(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(pred) - np.asarray(true), axis=-1).mean())


def fde(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(pred)[-1] - np.asarray(true)[-1], axis=-1).mean())


def improvement(baseline_error: float, model_error: float) -> float:
    return float((baseline_error - model_error) / max(baseline_error, 1e-9))
