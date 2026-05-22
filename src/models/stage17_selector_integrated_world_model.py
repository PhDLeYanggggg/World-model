from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from src.models.stage17_baseline_selector import Stage17BaselineSelector
from src.models.stage17_correction_specialist import Stage17CorrectionSpecialist


@dataclass
class Stage17SelectorIntegratedWorldModel:
    selector: Stage17BaselineSelector
    specialist: Stage17CorrectionSpecialist

    def predict_endpoint(self, candidate_endpoints: Dict[str, np.ndarray], features: Dict[str, float]) -> Dict[str, Any]:
        baseline_id, confidence = self.selector.select(features)
        selected = candidate_endpoints.get(baseline_id, candidate_endpoints.get("constant_position"))
        correction = self.specialist.correct(selected, features)
        return {
            "selected_baseline": baseline_id,
            "selector_confidence": confidence,
            "selected_baseline_trajectory": selected,
            "final_trajectory": correction["prediction"],
            "alpha": correction["alpha"],
            "correction_applied": correction["correction_applied"],
            "fallback_reason": correction["fallback_reason"],
        }

