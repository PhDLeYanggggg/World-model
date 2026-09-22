"""Fixed risk cross-check, with an equal-count nomination-only control."""
import numpy as np
from src.evaluation.m3w_conditional_cost_audit import strict_bits

POLICIES = ('nomination', 'reviewed', 'matched_nomination')


def choices(nomination_score, native_score, fraction_score, past, distance, ids):
    scores = [np.asarray(v) for v in (nomination_score, native_score, fraction_score)]
    ids = np.asarray(ids)
    n = len(distance)
    if (ids.shape != (n,) or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != n
            or any(v.shape != (n, 2) or not np.isfinite(v).all() or (v < 0).any() for v in scores)):
        raise ValueError('Three aligned nonnegative cost pairs and unique causal query IDs required')
    nominated = strict_bits(scores[0], past, distance)
    review_harm = np.maximum.reduce([v[:, 1] for v in scores])
    reviewed = nominated & (review_harm <= .1*scores[0][:, 0])
    pool = np.flatnonzero(nominated)
    gain = scores[0][:, 0]-scores[0][:, 1]
    order = np.lexsort((ids[pool], -gain[pool]))
    matched = np.zeros(n, bool)
    matched[pool[order[:int(reviewed.sum())]]] = True
    return dict(nomination=nominated, reviewed=reviewed, matched_nomination=matched)


def empirical_gate(summary, contrast):
    return dict(positive_matched_count_ci=contrast['ci95_pp'][0] > 0,
        positive_vs_cv_ci=summary['ADE']['scene_bootstrap_ci95'][0] > 0,
        each_seed_positive_cv=all(r['ADE']['equal_scene_gain_percent'] > 0 for r in summary['seeds'].values()),
        aggregate_easy=all(-r['subsets']['positive_easy']['equal_scene_gain_percent'] <= 2 for r in summary['seeds'].values()),
        each_scene_seed_easy=all(-v['gain_percent'] <= 2 for r in summary['seeds'].values()
            for v in r['subsets']['positive_easy']['by_scene'].values()),
        exact_zero=all(r['zero_CV_harmed'] == 0 for r in summary['seeds'].values()))
