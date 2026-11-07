"""Source-deletion sensitivity for learned veto cost corrections."""
import numpy as np

from src.world_model import m3w_veto_risk_refit as veto
from src.world_model import m3w_causal_record_ledger as ledger


def fit(moments):
    if len(moments) < 3:
        raise ValueError('Require at least three fitting sources')
    full = veto.coefficients(moments)
    deleted = np.stack([veto.coefficients(moments[:i] + moments[i+1:])
                        for i in range(len(moments))])
    return full, deleted


def correction(x, full, deleted):
    point = np.einsum('nad,adc->nac', x, full)
    predictions = np.einsum('nad,sadc->snac', x, deleted)
    centered = predictions - predictions.mean(axis=0)
    # Jackknife sensitivity is a model feature, not a coverage guarantee.
    sensitivity = np.sqrt((1. - 1. / len(deleted)) * (centered**2).sum(axis=0))
    return point, np.minimum(point + sensitivity, np.maximum(point, 0.))


def allocate(groups, value, a, e, legal, witness, before, vetoed, delta, scale, radius):
    base = np.stack((a, e), axis=-1)
    limit = (radius + .02 * scale[:, None])[..., None]
    proposed = base + np.clip(delta * scale[:, None, None], -limit, limit)
    release = vetoed & (proposed <= before[:, None] + ledger.EPS).all(axis=-1)
    rows = np.flatnonzero(witness >= 0)
    release[rows, witness[rows]] = False
    costs = np.where(release[..., None], proposed, base)
    counts = [int((witness[g] >= 0).sum()) for g in groups]
    choice, _ = ledger.stream(groups, value, costs[..., 0], costs[..., 1], legal,
                              counts=counts, witness=witness)
    return choice, release
