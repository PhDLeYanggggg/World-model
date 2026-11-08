"""Finite easy harm relative to the same-radius incumbent."""

import torch

from src.world_model.m3w_cv_tangent import directional_derivative, losses


def terms(offset, incumbent, cv, target, valid):
    clean = torch.where(valid[..., None], target, cv)
    e = cv-clean
    count = valid.sum(1).clamp_min(1)

    def finite(d):
        return (((e+d).norm(dim=-1)-e.norm(dim=-1))*valid).sum(1)/count

    diff, old_diff = finite(offset), finite(incumbent)
    excess = (diff.clamp_min(0.)-old_diff.clamp_min(0.)).clamp_min(0.)
    curvature = diff-directional_derivative(e, offset, valid)
    prior_curvature = old_diff-directional_derivative(e, incumbent, valid)
    extra_curvature = (curvature-prior_curvature).clamp_min(0.)
    return excess, extra_curvature


def objective(offset, logit, incumbent, cv, target, valid, easy, norm, arm):
    result = losses(offset, logit, cv, target, valid, easy, with_tangent=True)
    if arm == "continued":
        return result
    if arm not in ("finite_guard", "curvature_guard"):
        raise ValueError("Unknown continuation objective")
    excess, curvature = terms(offset, incumbent, cv, target, valid)
    known = valid.any(1)
    easy_factor = easy*norm
    # Easy excess is normalized by fit-record easy reference mass per row.
    # A smaller unnormalized all-row term also penalizes harm migration.
    result = result + 1.5*(easy_factor*excess).sum()/easy.sum().clamp_min(1)
    result = result + .25*excess.sum()/known.sum().clamp_min(1)
    if arm == "curvature_guard":
        result = result + .5*(easy_factor*(excess.detach() > 0.)*curvature).sum()/easy.sum().clamp_min(1)
    return result
