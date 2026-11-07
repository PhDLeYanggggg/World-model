"""Masked training loss for the relative displacement of two agents."""
import torch


def relative_path_loss(left, right, target_left, target_right, mask_left, mask_right, scale):
    mask = mask_left & mask_right
    # Mask before forming differences, including arbitrary missing-label fillers.
    predicted = torch.where(mask[..., None], left-right, 0.)
    truth = torch.where(mask[..., None], target_left-target_right, 0.)
    errors = torch.linalg.vector_norm(predicted-truth, dim=-1)
    rows = errors.sum(1)/mask.sum(1).clamp_min(1)
    return (rows/scale).mean()
