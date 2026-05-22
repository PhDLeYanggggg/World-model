from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class FallbackSelector:
    failure_threshold: float = 0.50
    alpha_threshold: float = 0.10
    residual_norm_threshold: float = 0.50
    force_dataset_baseline: bool = True

    def select(
        self,
        baseline_prediction: np.ndarray,
        learned_prediction: np.ndarray,
        alpha: np.ndarray,
        failure_probability: np.ndarray,
        residual_norm: np.ndarray,
        metadata: Dict,
    ) -> Dict[str, np.ndarray | list[str]]:
        n = baseline_prediction.shape[0]
        final = learned_prediction.copy()
        decisions = np.ones(n, dtype=bool)
        reasons: list[str] = []
        for i in range(n):
            reason = "use_learned_correction"
            if self.force_dataset_baseline or metadata.get("dataset_fallback_to_baseline", False):
                decisions[i] = False
                reason = "dataset_validation_prefers_strongest_baseline"
            elif failure_probability[i] < self.failure_threshold:
                decisions[i] = False
                reason = "failure_probability_below_threshold"
            elif alpha[i] < self.alpha_threshold:
                decisions[i] = False
                reason = "alpha_below_threshold"
            elif residual_norm[i] > self.residual_norm_threshold:
                decisions[i] = False
                reason = "residual_norm_too_large"
            if not decisions[i]:
                final[i] = baseline_prediction[i]
            reasons.append(reason)
        return {
            "final_prediction": final,
            "intervention_decision": decisions,
            "fallback_reason": reasons,
        }

