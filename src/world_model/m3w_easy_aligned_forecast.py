"""One-objective warm continuation of the existing deterministic forecaster."""
import torch

from src.world_model.m3w_native_forecast import masked_objective


def objective(prediction, target, mask, factors, causal_cv, easy_cut, *, aligned):
    base, ade = masked_objective(prediction,target,mask,factors)
    with torch.no_grad():
        _, cv_ade = masked_objective(causal_cv,target,mask,factors)
        easy = mask.any(1) & (cv_ade>0) & (cv_ade<=easy_cut)
    penalty = (factors * easy * torch.relu(ade-cv_ade)).mean()
    return base+2*penalty if aligned else base, base.detach(), penalty.detach()
