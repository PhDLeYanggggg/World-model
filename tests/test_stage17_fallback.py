from __future__ import annotations

import numpy as np

from src.models.stage17_baseline_selector import Stage17BaselineSelector
from src.models.stage17_correction_specialist import Stage17CorrectionSpecialist
from src.models.stage17_selector_integrated_world_model import Stage17SelectorIntegratedWorldModel


def test_stage17_correction_specialist_defaults_to_noop_fallback():
    specialist = Stage17CorrectionSpecialist(enabled=False)
    endpoint = np.array([1.0, 2.0])
    out = specialist.correct(endpoint, {"failure_probability": 1.0})
    assert out["correction_applied"] is False
    assert out["alpha"] == 0.0
    assert np.allclose(out["prediction"], endpoint)


def test_stage17_selector_integrated_outputs_selected_baseline():
    selector = Stage17BaselineSelector(
        rules=[{"baseline_id": "damped_velocity", "feature": "past_heading_change", "op": "<", "threshold": 0.2, "confidence": 0.7}]
    )
    model = Stage17SelectorIntegratedWorldModel(selector=selector, specialist=Stage17CorrectionSpecialist(enabled=False))
    endpoints = {
        "constant_position": np.array([0.0, 0.0]),
        "damped_velocity": np.array([1.0, 0.0]),
    }
    out = model.predict_endpoint(endpoints, {"past_heading_change": 0.1})
    assert out["selected_baseline"] == "damped_velocity"
    assert out["correction_applied"] is False
    assert np.allclose(out["final_trajectory"], endpoints["damped_velocity"])

