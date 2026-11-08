"""Factor missing-step training debit and positive-harm weighting."""

import torch


ARMS = ("grid_plain", "masked_harm", "masked_plain")


def objective(forecast, floor, target, valid, scale, arm):
    if arm not in (*ARMS, "grid_harm"):
        raise ValueError("Unknown registered loss")
    clean = torch.where(valid[..., None], target, 0.)
    diff = torch.linalg.vector_norm(forecast-clean, dim=-1) - torch.linalg.vector_norm(floor-clean, dim=-1)
    debit = torch.linalg.vector_norm(forecast-floor, dim=-1) if arm.startswith("grid") else 0.
    # Preserve the same step denominator even for incomplete rows.
    per_row = torch.where(valid, diff, debit).mean(1) / (12*scale)
    return (per_row + per_row.relu()).mean() if arm.endswith("harm") else per_row.mean()
