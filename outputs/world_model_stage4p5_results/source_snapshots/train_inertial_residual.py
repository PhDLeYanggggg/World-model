from __future__ import annotations

from typing import Dict, List

from src.models.inertial_residual_model import InertialResidualModel, train_inertial_residual


def train_stage4p5_residual_models(episodes: List[Dict], quick: bool = False) -> Dict[str, InertialResidualModel]:
    max_samples = 1600 if quick else 12000
    return {
        "residual_over_constant_velocity": train_inertial_residual(episodes, "constant_velocity", multi_step=False, max_samples=max_samples),
        "residual_over_constant_acceleration": train_inertial_residual(episodes, "constant_acceleration", multi_step=False, max_samples=max_samples),
        "residual_over_tuned_hand_physics": train_inertial_residual(episodes, "tuned_hand_physics", multi_step=False, max_samples=max_samples),
        "residual_over_constant_velocity_with_multistep_loss": train_inertial_residual(episodes, "constant_velocity", multi_step=True, max_samples=max_samples),
    }
