"""Training-only moment contributions for the existing partial-label risk rule."""
import numpy as np

from src.world_model.m3w_selection_controller import labels


def supported_bound_targets(rollouts, target, valid, floor_index, easy_cut):
    """Unknown contributions are conservative bounds, never fabricated ADE labels.

    R/ER zero means no observed reference contribution in the evaluator's
    denominator. It does not mean an unknown agent's physical error is zero.
    """
    p = np.asarray(rollouts)
    if p.ndim != 4 or p.shape[1:] != (7, 12, 2) or floor_index not in range(6):
        raise ValueError('The original seven-action/twelve-step moment schema is required')
    moments, observed = labels(p, target, valid, floor_index, easy_cut)
    radius = np.linalg.norm(p-p[:, floor_index, None], axis=-1).max(-1)
    missing = ~observed
    moments[missing] = 0
    moments[missing, :, 1] = radius[missing]
    moments[missing, :, 4] = radius[missing]
    if not np.isfinite(moments).all() or (moments < 0).any():
        raise ValueError('Finite nonnegative observed moments or explicit bound contributions required')
    return dict(moments=moments, observed_future=observed,
        unknown_bound_supervision=missing, target_defined=np.ones(len(p), bool),
        semantics=('benefit_lower', 'harm_upper', 'known_reference_contribution',
                   'known_easy_reference_contribution', 'easy_harm_upper'))
