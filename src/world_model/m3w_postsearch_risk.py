"""Estimate risk conditional on the assignment selected by path search."""
import numpy as np

from src.world_model.m3w_joint_path_portfolio import options


QUANTILES = (0., .25, .5, .75, .9, .95)
RIDGE = .02


def search(history, cv, ols, bank, scale, u, a, e, support, legal, count):
    who, how, add, joint, admit = options(history, cv, ols, bank, scale, u, a, e, support, legal, count)
    eligible = np.flatnonzero(admit)
    if not len(eligible):
        return None
    j = eligible[joint[eligible].argmax()]
    ii, kk = who[j], how[j]
    s = scale[ii].sum()
    target = bank[ii, kk]
    cv_radius = np.linalg.norm(target-cv[ii], axis=-1).max(1)
    ols_radius = np.linalg.norm(target-ols[ii], axis=-1).max(1)
    all_slack, easy_slack = -a[ii, kk].sum()/s, -e[ii, kk].sum()/s
    features = np.array([all_slack, easy_slack, support[ii, kk].sum()/s, u[ii, kk].sum()/s,
        cv_radius.sum()/s, ols_radius.sum()/s, joint[j], joint[j]-add[j],
        np.log1p(len(eligible)), np.log1p(len(who)), np.log1p(len(cv)), np.log1p(s)])
    x = np.r_[1., np.arcsinh(features), np.tanh(features)]
    return ii, kk, x, min(all_slack, easy_slack)


def fit(x, y, source):
    if len(x) == 0:
        raise ValueError('No cross-fitted search choices')
    _, inv, counts = np.unique(source, return_inverse=True, return_counts=True)
    w = 1./counts[inv]
    w /= w.sum()
    penalty = np.eye(x.shape[1])*RIDGE
    penalty[0, 0] = 0.
    return np.linalg.solve(x.T@(w[:, None]*x)+penalty, x.T@(w[:, None]*y))


def learned_score(x, coefficient):
    p = (x@coefficient).clip(0., 1.)
    return p[:, 0]*(.5+.5*p[:, 1]), p


def levels(score):
    return np.r_[-np.inf, np.quantile(score, QUANTILES)]


def rank_count(control, selected, recording, count):
    """Offline comparator: exactly the treatment's counts by recording and cardinality."""
    result = np.zeros(len(control), dtype=bool)
    for r in np.unique(recording):
        for k in (1, 2):
            ix = np.flatnonzero((recording == r) & (count == k))
            n = int(selected[ix].sum())
            best = ix[np.argsort(-control[ix], kind='stable')[:n]]
            result[best] = True
    return result
