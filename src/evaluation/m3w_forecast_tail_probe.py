"""Candidate-bank tail attribution and source-held causal information test."""
import numpy as np
from src.evaluation import m3w_peer_path_probe as parent

FRACTIONS = (.01, .05, .10, .20)
SCORES = ('signed_risk', 'positive_tail', 'envelope', 'oracle_diagnostic')


def capture(score, cost, weights, fraction):
    """Fractional score-tie allocation; a ranking diagnostic, never a policy."""
    score, cost, weights = map(lambda a: np.asarray(a, float), (score, cost, weights))
    if (score.ndim != 1 or cost.shape != score.shape or weights.shape != score.shape
            or not len(score) or not all(np.isfinite(v).all() for v in (score, cost, weights))
            or (cost < 0).any() or (weights < 0).any() or weights.sum() <= 0
            or not 0 < fraction <= 1):
        raise ValueError('Finite aligned scores, nonnegative costs and positive weights required')
    w = weights/weights.sum(); total = float(w@cost)
    if total <= 0:
        return None
    levels, inverse = np.unique(score, return_inverse=True)
    mass = np.bincount(inverse, weights=w, minlength=len(levels))
    value = np.bincount(inverse, weights=w*cost, minlength=len(levels))
    mass, value = mass[::-1], value[::-1]
    before = np.r_[0., np.cumsum(mass)[:-1]]
    take = np.divide(np.clip(fraction-before, 0, mass), mass,
        out=np.zeros_like(mass), where=mass > 0)
    return float(np.dot(take, value)/total)


def fit(x, y, train, weights):
    if (x.ndim != 2 or y.shape != (len(x), 2) or train.shape != (len(x),)
            or train.dtype != bool or weights.shape != train.shape
            or not all(np.isfinite(v).all() for v in (x, y, weights)) or (weights < 0).any()):
        raise ValueError('Separate finite two-channel supervision and source mask required')
    # Positivity is a supervision transform, not a new inference feature.
    return parent.fit_predict(x, np.maximum(y, 0), train, ~train, weights, 'extra_trees')


def score_table(y, weights, predictions, envelope):
    costs = np.maximum(y, 0)
    if set(predictions) != {'signed_risk', 'positive_tail'}:
        raise ValueError('Fixed signed and positive-target comparison required')
    rows = {}
    for channel, name in enumerate(('all', 'easy')):
        scores = {k:v[:, channel] for k,v in predictions.items()}
        scores.update(envelope=envelope, oracle_diagnostic=costs[:, channel])
        rows[name] = {key:{str(f):capture(score, costs[:, channel], weights, f) for f in FRACTIONS}
            for key,score in scores.items()}
    return rows


def group_masses(y, weights, group):
    w = weights/weights.sum(); out = []
    positive = np.maximum(y, 0); negative = np.maximum(-y, 0)
    total = (w[:, None]*positive).sum(0)
    for value in sorted(set(group)):
        at = group == value; mass = (w[at, None]*positive[at]).sum(0)
        out.append(dict(group=str(value), rows=int(at.sum()), weight=float(w[at].sum()),
            positive_mass=mass.tolist(), negative_mass=(w[at, None]*negative[at]).sum(0).tolist(),
            positive_mass_share=[float(a/b) if b > 0 else None for a,b in zip(mass,total)]))
    return out


def source_summary(folds):
    table = {}
    for channel in ('all', 'easy'):
        table[channel] = {}
        for score in SCORES:
            values = [r['ranking'][channel][score]['0.1'] for r in folds]
            table[channel][score] = (None if any(v is None for v in values) else float(np.mean(values)))
    easy = table['easy']; new = easy['positive_tail']
    passed = (new is not None and easy['signed_risk'] is not None and easy['envelope'] is not None
        and new >= .5 and new-easy['signed_risk'] >= .05 and new-easy['envelope'] >= .05)
    return dict(source_macro_capture_at_10percent=table, passes_information_screen=bool(passed),
        safety_certificate=False, no_policy_selected=True)
