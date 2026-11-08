"""CV-anchored choice and out-of-site calibration for the OpenDD diagnostic."""
import numpy as np

from src.evaluation.m3w_opendd_location import choose, paired


def edit(d, score, *, threshold=None, allowed_queries=None):
    offsets = d['offsets']
    legal = d['legal'].copy()
    if threshold is not None:
        legal &= score > threshold
    if allowed_queries is not None:
        legal &= np.repeat(allowed_queries, np.diff(offsets))[:, None]
    selected = choose(score, legal, offsets)
    rows = np.flatnonzero(selected >= 0)
    chosen = d['cv'].copy()
    chosen[rows] = d['bank'][rows, selected[rows]]
    by_query = np.zeros(len(offsets)-1, dtype=bool)
    by_query[np.searchsorted(offsets[1:], rows, side='right')] = True
    return chosen, by_query


def metrics(d, pred, switched, easy_cut, *, ref=None):
    valid, y = d['valid'], d['y']
    cv_error = np.linalg.norm(d['cv'] - y, axis=-1)
    err = np.linalg.norm(pred - y, axis=-1)
    cv_ade = cv_error.mean(1)
    easy = valid.all(1) & (cv_ade > 0) & (cv_ade <= easy_cut)
    easy_ref = cv_ade[easy].sum()
    easy_change = (err.mean(1) - cv_ade)[easy]
    observed, lower = paired(d['cv'] if ref is None else ref, pred, y, valid)
    denom = np.nansum(cv_error)
    return dict(observed_pp=float(observed.sum()*100/denom), lower_pp=float(lower.sum()*100/denom),
                switches=int(switched.sum()), easy=int(easy.sum()),
                signed_easy_pct=float(easy_change.sum()*100/easy_ref) if easy_ref else None,
                positive_easy_pct=float(np.maximum(easy_change, 0).sum()*100/easy_ref) if easy_ref else None,
                ade=float(np.nanmean(err)), fde=float(np.nanmean(err[:, -1])))


def select_threshold(trials):
    def valid(row):
        values = row['sites']
        return (sum(x['switches'] > 0 for x in values) >= 3
                and all(x['signed_easy_pct'] is not None and x['signed_easy_pct'] <= 2 for x in values)
                and np.mean([x['lower_pp'] for x in values]) > 0)

    feasible = [row for row in trials if valid(row)]
    best = max(feasible, key=lambda row: (np.mean([x['lower_pp'] for x in row['sites']]), -row['threshold']),
               default=None)
    return None if best is None else best['threshold']
