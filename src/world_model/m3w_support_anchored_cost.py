"""Candidate-choice scores that price deviation from a frozen reference."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.world_model.m3w_seed_dispersion_cost import matched_choices

ARMS = ('reference', 'dispersion', 'visibility_only', 'constant_anchor', 'learned_anchor')


def fit_support(features, unknown, weights):
    if np.unique(unknown).size != 2:
        raise ValueError('Both observed and wholly unknown fitting rows are required')
    clf = HistGradientBoostingClassifier(max_iter=64, max_leaf_nodes=15, max_bins=63,
        l2_regularization=1., early_stopping=False, random_state=20261107)
    clf.fit(features, unknown, sample_weight=weights / weights.mean())
    return clf, float(np.average(unknown, weights=weights))


def anchored_score(value, radius, p_unknown, reference):
    # Remove the terms constant in the new binary action from
    # (a-r)(1-p)*value - |a-r|*p*radius.
    return (1-p_unknown)*value + np.where(reference, 1., -1.)*p_unknown*radius


def choices(query, ids, predictions, radius, p_unknown, base_rate):
    def top(score):
        return matched_choices(query, ids, np.column_stack((score, np.zeros_like(score))))
    ref = matched_choices(query, ids, predictions[0])
    value = predictions[1, :, 0]-predictions[1, :, 1]
    return np.stack((ref, top(value), top((1-p_unknown)*value),
        top(anchored_score(value, radius, base_rate, ref)),
        top(anchored_score(value, radius, p_unknown, ref))))
