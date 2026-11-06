"""Complete-risk training contributions; inference features remain past-only."""
import numpy as np
import torch

from src.world_model import m3w_amplitude_cost as amplitude
from src.world_model import m3w_donor_augmentation as donor


def signed_targets(packet, target, valid, easy_cut, scales):
    truth = amplitude.targets(packet, target, valid, easy_cut)
    scales = np.asarray(scales)
    if scales.shape != (len(target),) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Positive past-only row scales required')
    normalized = torch.from_numpy((truth['moments']/scales[:, None, None]).astype(np.float32))
    signed = (normalized[..., [1, 4]]-.02*normalized[..., [2, 3]]).numpy()
    return signed, truth['observed_future']


def unknown_targets(envelope, scales):
    envelope, scales = np.asarray(envelope), np.asarray(scales)
    if (envelope.ndim != 1 or scales.shape != envelope.shape or (envelope < 0).any()
            or (scales <= 0).any() or not np.isfinite(envelope).all() or not np.isfinite(scales).all()):
        raise ValueError('Finite causal envelope and scale required')
    upper = (envelope/scales).astype(np.float32)
    return np.column_stack((upper, upper))


def append_unknown(known, unknown):
    keys = ('x', 'target', 'sites', 'query', 'ids')
    if len(set(known['ids']) & set(unknown['ids'])):
        raise ValueError('Known and bound-only training rows must be disjoint')
    result = {k: np.concatenate((known[k], unknown[k])) for k in keys}
    if result['x'].shape != (len(result['ids']), 120) or not np.isfinite(result['target']).all():
        raise ValueError('Complete finite training contributions required')
    result['weights'] = donor.parent.base.balanced_weights(result['sites'], result['query'])
    result['observed_future'] = np.r_[np.ones(len(known['ids']), bool), np.zeros(len(unknown['ids']), bool)]
    return result
