from __future__ import annotations

from typing import Dict


def stage5_loss_terms() -> list[str]:
    return [
        "position_loss",
        "velocity_loss",
        "acceleration_loss",
        "heading_loss",
        "turn_rate_loss",
        "endpoint_loss",
        "rollout_loss@10",
        "rollout_loss@25",
        "rollout_loss@50",
        "rollout_loss@100",
        "residual_loss",
        "map_violation_loss",
        "collision_loss",
        "speed_limit_loss",
        "acceleration_limit_loss",
        "smoothness_loss",
        "uncertainty_nll_loss",
        "domain_balance_loss",
        "latent_KL_loss_when_enabled",
        "diversity_loss_when_enabled",
    ]


def train_stage5_foundation_quick(config: Dict) -> Dict:
    return {
        "status": "not_trained_in_data_discovery_dry_run",
        "reason": "Stage 5-Data gates require larger converted data lake before deterministic foundation training.",
        "loss_terms": stage5_loss_terms(),
        "scheduled_sampling": {"teacher_forcing_start": 1.0, "teacher_forcing_end": config.get("teacher_forcing_end", 0.5)},
        "curriculum": [10, 25, 50, 100],
    }
