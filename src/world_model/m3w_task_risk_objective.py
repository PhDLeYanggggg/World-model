"""Smooth decision-regret and empirical risk constraints, not risk calibration."""
import math

import torch

from src.world_model.m3w_inner_separability import signed

BUDGET = .02


def soft_action(prediction, eligible, rms, temperature):
    if (prediction.ndim != 2 or prediction.shape[1] != 5
            or not torch.isfinite(prediction).all() or (prediction < 0).any()
            or eligible.shape != (len(prediction),) or eligible.dtype != torch.bool
            or rms.shape != (8,) or not torch.isfinite(rms).all() or (rms <= 0).any()
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('Finite causal moments, masks, TRAIN scales and positive temperature required')
    margins = signed(prediction)*prediction.new_tensor([1., -1., -1.])
    return eligible.to(prediction.dtype)*torch.sigmoid(margins/rms[5:]/temperature).prod(1)


def query_mean(values, segments, count):
    if (segments.dtype != torch.int64 or segments.shape != (len(values),) or count <= 0
            or len(values) == 0 or (segments < 0).any() or (segments >= count).any()):
        raise ValueError('Aligned nonempty query segments required')
    sizes = torch.bincount(segments, minlength=count).to(values.dtype)
    if (sizes == 0).any():
        raise ValueError('Missing query cannot be silently dropped')
    if values.ndim == 1:
        values = values[:, None]
    return (values.new_zeros((count, values.shape[1])).index_add(0, segments, values)/sizes[:, None]).mean(0)


def terms(prediction, target, segments, count, eligible, rms, temperature):
    if (target.shape != prediction.shape or target.requires_grad
            or not torch.isfinite(target).all() or (target < 0).any()):
        raise ValueError('Known supervised targets only; unknowns are not imputed safe')
    action = soft_action(prediction, eligible, rms, temperature)
    regret = (1-action)*target[:, 0]+action*target[:, 1]
    # Positive-harm/reference constraints keep their original 2% definition.
    excess = action[:, None]*torch.stack((target[:, 1]-BUDGET*target[:, 2],
                                         target[:, 4]-BUDGET*target[:, 3]), 1)
    return dict(action=action, regret=query_mean(regret, segments, count)[0],
                risk_excess=query_mean(excess, segments, count),
                action_mean=query_mean(action, segments, count)[0])


def objective(result, dual):
    if dual.shape != (2,) or not torch.isfinite(dual).all() or (dual < 0).any():
        raise ValueError('Finite nonnegative empirical Lagrange weights required')
    return result['regret']+(dual.detach()*result['risk_excess']).sum()


def update_dual(dual, excess, learning_rate, maximum):
    if (dual.shape != (2,) or excess.shape != (2,)
            or not torch.isfinite(dual).all() or not torch.isfinite(excess).all()
            or (dual < 0).any() or not math.isfinite(learning_rate)
            or not math.isfinite(maximum) or learning_rate <= 0 or maximum <= 0):
        raise ValueError('Finite bounded dual update required')
    return (dual.detach()+learning_rate*excess.detach()).clamp(0, maximum)
