"""Train a forecast against the same missing-step debit used in evaluation."""

import torch


OBJECTIVES = ("masked", "cv_completion", "dual_completion")


def loss_terms(pred, y, valid, scale, cv, ols):
    """Per-batch path cost and signed completion regret, in past-scale units."""
    scale = scale[:, None, None]

    def distance(a, b):
        return torch.linalg.vector_norm((a - b) / scale, dim=-1)

    # Mask the target before arithmetic: unobserved coordinates are immaterial.
    safe_y = torch.where(valid[..., None], y, torch.zeros_like(y))
    observed = torch.where(valid, distance(pred, safe_y), 0.)
    cost = observed.mean()
    regrets = []
    for ref in (cv, ols):
        ref_error = torch.where(valid, distance(ref, safe_y), 0.)
        debit = torch.where(valid, 0., distance(pred, ref))
        regrets.append((observed - ref_error + debit).mean())
    return cost, *regrets


def objective(arm, pred, y, valid, scale, cv, ols):
    cost, cv_regret, ols_regret = loss_terms(pred, y, valid, scale, cv, ols)
    if arm == "masked":
        return cost
    if arm == "cv_completion":
        return cv_regret
    if arm == "dual_completion":
        return torch.maximum(cv_regret, ols_regret)
    raise ValueError(f"Unknown training objective: {arm}")
