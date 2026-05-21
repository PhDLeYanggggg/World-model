from __future__ import annotations

from src.training.train_stage5_foundation import train_stage5_foundation_quick


def pretrain_stage5(config: dict) -> dict:
    return train_stage5_foundation_quick(config)
