"""Continuous conservative costs for an already chosen intervention set."""
import numpy as np

from src.evaluation.m3w_selection_forecast_readout import errors


NAMES = ('lower_gain', 'all_harm', 'easy_harm', 'all_reference', 'easy_reference')
BUDGET = .02


def action_targets(floor, pred, future, valid, easy_cut):
    known = valid.any(1)
    ref = np.where(known, errors(floor, future, valid)[0], 0.)
    gain = np.where(known, ref-errors(pred, future, valid)[0], 0.)
    easy = known & (ref > 0.) & (ref <= easy_cut)
    missing = np.where(known, 0., np.linalg.norm(pred-floor, axis=-1).max(1))
    harm = np.maximum(-gain, 0.)
    return np.column_stack((gain-missing, harm+missing, easy*harm+missing, ref, easy*ref))


def option_targets(action, q, slate, scale):
    mask = slate >= 0
    local = np.where(mask)[1].reshape(len(slate), -1)
    y = action[q[local], slate[mask].reshape(local.shape)].sum(1)
    return y/scale[local].mean(1, keepdims=True)


def score(prediction):
    value, all_harm, easy_harm, ref, easy_ref = prediction.T
    all_harm = all_harm.clip(0.)
    ref = ref.clip(0.)
    all_excess = all_harm-BUDGET*ref
    easy_excess = np.minimum(easy_harm.clip(0.), all_harm)-BUDGET*np.minimum(easy_ref.clip(0.), ref)
    return np.column_stack((value, value-all_excess.clip(0.)-easy_excess.clip(0.)))
