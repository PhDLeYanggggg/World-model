"""Causal sparse exchanges scored by dense paired gain and support predictions."""
import numpy as np

from src.world_model.m3w_quarter_allocation import predict, train


CAP = 4
THRESHOLDS = (0., .025, .10, .25)
SUPPORT_WEIGHT = .25
FITTING_PAIRS_PER_QUERY = 128


def roster(groups, choices, legal):
    chunks, queries = [], []
    for j, q in enumerate(groups):
        drop = q[choices[q] >= 0]
        i, f = np.where((choices[q, None] < 0) & legal[q])
        if len(drop) and len(i):
            chunks.append(np.column_stack((np.tile(q[i], len(drop)), np.tile(f, len(drop)),
                                           np.repeat(drop, len(i)))))
            queries.append(np.full(len(chunks[-1]), j, dtype=int))
    return (np.concatenate(chunks), np.concatenate(queries)) if chunks else (
        np.empty((0, 3), dtype=int), np.empty(0, dtype=int))


def pair_features(action_features, choices, moves, scale):
    i, family, drop = moves.T
    s = (scale[i]+scale[drop])*.5
    a = action_features[i, family]
    b = action_features[drop, choices[drop]]
    return np.column_stack((a, b, np.log(scale[i]/scale[drop]))), s


def first_options(prediction, query, *, support):
    """One positive-gain candidate per query, retaining its causal time index."""
    gain, all_risk, easy_risk, signed, delta = prediction.T
    score = gain - np.maximum(all_risk, 0.) - np.maximum(easy_risk, 0.) - np.maximum(signed, 0.)
    if support:
        score += SUPPORT_WEIGHT * delta.clip(-1., 1.)
    score = np.where(gain >= 0., score, -np.inf)
    if not len(score):
        return np.empty(0, dtype=int), score
    starts = np.r_[0, np.flatnonzero(np.diff(query))+1, len(query)]
    best = np.asarray([i + np.argmax(score[i:j]) for i, j in zip(starts[:-1], starts[1:])])
    return best, score[best]


def apply(choices, moves, best, scores, threshold):
    result = choices.copy()
    if threshold is None:
        return result
    take = best[scores > threshold][:CAP]
    i, family, drop = moves[take].T
    result[i] = family
    result[drop] = -1
    return result
