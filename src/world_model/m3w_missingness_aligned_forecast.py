"""Train against path disagreement without inventing missing positions."""
import torch

from src.world_model.m3w_easy_aligned_forecast import objective as known_objective


def objective(prediction,target,mask,factors,cv,easy_cut):
    known,ade,easy=known_objective(prediction,target,mask,factors,cv,easy_cut,aligned=True)
    radius=torch.linalg.vector_norm(prediction-cv,dim=-1).amax(1)
    missing=(~mask.any(1))
    penalty=(factors*missing*radius).mean()
    return known+penalty,dict(ade=ade,easy_penalty=easy,unknown_radius_penalty=penalty.detach())
