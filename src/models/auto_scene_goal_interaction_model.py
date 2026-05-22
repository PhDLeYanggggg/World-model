from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AutoSceneGoalInteractionModelSpec:
    name: str = "auto_scene_goal_interaction_deterministic"
    prediction_form: str = "prediction_i,h = strongest_causal_baseline_i,h + alpha_i,h * bounded_residual_i,h"
    latent_enabled: bool = False
    smc_enabled: bool = False
    supports_variable_agent_masks: bool = True
    notes: str = "Specification only; training is gated by strongest-baseline repair readiness."


def model_spec() -> dict:
    return AutoSceneGoalInteractionModelSpec().__dict__

