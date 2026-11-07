"""Fit additive risk masses on selected assignments and act in frame order."""
import numpy as np

from src.evaluation.m3w_recording_switch_ceiling import BUDGET, moments


FACTORS = (.25, .5, .75, 1.)
RIDGE = .02


def targets(cv, path, y, valid, easy_cut):
    m = moments(cv, path, y, valid, easy_cut)
    return np.array([m['all_excess']+BUDGET*m['base'],
                     m['easy_excess']+BUDGET*m['ref_easy'], m['base'], m['ref_easy']]).sum(1)


def statistics(x, y, source):
    _, inv, count = np.unique(source, return_inverse=True, return_counts=True)
    w = 1./count[inv]
    w /= w.sum()
    return x.T@(w[:, None]*x), x.T@(w[:, None]*y)


def fit(x, raw, scale, recording, source):
    """Match pointwise fits, adding exact full-record residuals in one arm."""
    gram, rhs = statistics(x, raw/scale[:, None], source)
    design, mass, sites = [], [], []
    for r in np.unique(recording):
        at = recording == r
        if len(set(source[at])) != 1:
            raise ValueError('A recording cannot cross sources')
        design.append((scale[at]@x[at])/scale[at].sum())
        mass.append(raw[at].sum(0)/scale[at].sum())
        sites.append(source[at][0])
    g, b = statistics(np.stack(design), np.stack(mass), np.asarray(sites))
    penalty = np.eye(x.shape[1])*RIDGE
    penalty[0, 0] = 0.
    return np.stack((np.linalg.solve(gram+penalty, rhs),
                     np.linalg.solve(.5*(gram+g)+penalty, .5*(rhs+b))))


def predict(x, scale, coef):
    p = np.maximum((x@coef)*scale[:, None], 0.)
    p[:, 1] = np.minimum(p[:, 1], p[:, 0])
    p[:, 3] = np.minimum(p[:, 3], p[:, 2])
    return p


def excess(p, factor=1.):
    return p[:, :2]-BUDGET*factor*p[:, 2:]


def ledger(p, recording, frame, factor):
    """Do not spend expected budget earned by later queries."""
    costs = excess(p, factor)
    take = np.zeros(len(p), dtype=bool)
    for r in np.unique(recording):
        rows = np.flatnonzero(recording == r)
        if len(np.unique(frame[rows])) != len(rows):
            raise ValueError('One search choice per recording frame')
        balance = np.zeros(2)
        for k in rows[np.argsort(frame[rows], kind='stable')]:
            next_balance = balance+costs[k]
            if (next_balance <= 0.).all():
                take[k] = True
                balance = next_balance
    return take
