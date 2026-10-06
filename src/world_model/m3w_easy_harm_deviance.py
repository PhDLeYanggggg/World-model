"""Candidate one-factor cost objective; not yet a trained or calibrated policy."""
import torch
from torch.nn import functional as F


def log_easy_harm(raw, envelope):
    """Stable log of the existing bounded decoder's fifth moment."""
    if (raw.ndim != 2 or raw.shape[1] != 5 or envelope.shape != (len(raw),)
            or not torch.isfinite(raw).all() or not torch.isfinite(envelope).all()
            or (envelope < 0).any()):
        raise ValueError('Finite five decoder logits and nonnegative causal envelope required')
    fractions = torch.log_softmax(torch.cat((raw[:, :2], torch.zeros_like(raw[:, :1])), 1), 1)
    # Zero-envelope rows have identically zero possible harm. The finite log
    # placeholder is masked by easy_deviance below, never a prediction.
    safe = torch.where(envelope > 0, envelope, torch.ones_like(envelope))
    return safe.log()+fractions[:, 1]+F.logsigmoid(raw[:, 4])


def easy_deviance(log_prediction, target, envelope, rms):
    if (log_prediction.shape != target.shape or target.shape != envelope.shape
            or not torch.isfinite(log_prediction).all() or not torch.isfinite(target).all()
            or not torch.isfinite(envelope).all() or (target < 0).any() or (envelope < 0).any()
            or (target > envelope+1e-5).any() or not torch.isfinite(rms) or rms <= 0
            or ((envelope == 0) & (target != 0)).any()):
        raise ValueError('Known nonnegative harm, matching envelope and frozen positive RMS required')
    log_u = log_prediction-rms.log()
    v = target/rms
    # xlogy(0, 0)=0: zeros are real labels, not missing-outcome imputation.
    result = 2*(log_u.exp()-v+torch.xlogy(v, v)-v*log_u)
    return torch.where(envelope > 0, result, result*0)


def objective(prediction, target, log_prediction, envelope, segments, count, rms):
    """Replace only easy-harm quadratic term; all/easy signed scores stay squared."""
    from src.world_model.m3w_inner_separability import signed
    if (prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 5
            or not torch.isfinite(prediction).all() or not torch.isfinite(target).all()
            or (prediction < 0).any() or (target < 0).any() or rms.shape != (8,)
            or not torch.isfinite(rms).all() or (rms <= 0).any()
            or segments.shape != (len(target),) or count <= 0
            or segments.dtype != torch.int64 or (segments < 0).any() or (segments >= count).any()):
        raise ValueError('Matched finite moment targets, query segments and original RMS required')
    sizes = torch.bincount(segments, minlength=count).to(prediction.dtype)
    if (sizes == 0).any():
        raise ValueError('Every sampled query must be nonempty')
    delta = torch.cat((prediction-target, signed(prediction)-signed(target)), 1)/rms
    square = delta.square()
    dev = easy_deviance(log_prediction, target[:, 4], envelope, rms[4])
    terms = torch.cat((square[:, :4], dev[:, None], square[:, 5:]), 1)
    grouped = terms.new_zeros((count, 8)).index_add(0, segments, terms)/sizes[:, None]
    moments, scores = grouped[:, :5].mean(), grouped[:, 5:].mean()
    return dict(total=.5*(moments+scores), moments=moments, decision_scores=scores,
                easy_harm_deviance=grouped[:, 4].mean())
