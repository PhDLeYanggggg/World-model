"""Train an emitted CV/hindcast blend with optional positive-harm supervision."""
import torch

from src.world_model.m3w_opendd_velocity_candidate import decode

ALPHA = .05
ARMS = ('observed', 'completion', 'easy_guard')


def emitted(raw, anchor):
    # Bounded heading-frame velocity increments, integrated by the shared decoder.
    return anchor + (.25 * ALPHA) * decode(raw)


def objective(pred, cv, anchor, future, valid, easy, arm):
    if arm not in ARMS:
        raise ValueError(f'Unknown objective: {arm}')
    target = torch.where(valid[..., None], future, 0.)
    # Vector norms have a finite zero subgradient at the exact anchor.
    err = torch.linalg.vector_norm(pred-target, dim=-1)
    masked = torch.where(valid, err, 0.)
    terms = masked
    if arm != 'observed':
        missing = torch.linalg.vector_norm(pred-anchor, dim=-1)
        terms = torch.where(valid, err, missing)
    loss = terms.mean()
    if arm == 'easy_guard':
        cv_err = torch.where(valid, torch.linalg.vector_norm(cv-target, dim=-1), 0.)
        signed = (masked-cv_err).sum(1) / valid.sum(1).clamp_min(1)
        easy_harm = torch.where(easy & valid.any(1), torch.relu(signed), 0.)
        cv_gap = torch.linalg.vector_norm(pred-cv, dim=-1).amax(1)
        loss = loss + 2.*(easy_harm + torch.where(valid.any(1), 0., cv_gap)).mean()
    return loss
