"""Fitting-source-only calibration readout for an OLS4 proposal score."""
import numpy as np


THRESHOLDS = (0., .0025, .005, .01, .02, .04, .08)


def targets(cv, ols, bank, label, valid):
    clean = np.where(valid[...,None],label,0.)
    count = valid.sum(1)
    known = count > 0

    def ade(p):
        dist = np.linalg.norm(p-clean[:,None],axis=3)
        return np.where(valid[:,None],dist,0.).sum(2)/np.maximum(count[:,None],1)

    cv_mass = ade(cv[:,None])[:,0]
    observed_gain = ade(ols[:,None])-ade(bank)
    debit = np.linalg.norm(bank-ols[:,None],axis=3).max(2)
    gain = np.where(known[:,None],observed_gain,-debit)
    harm = np.where(known[:,None],np.maximum(-observed_gain,0.),debit)
    return cv_mass,gain,harm,known


def intervals(cells, draws):
    if cells.ndim != 2 or not np.isfinite(cells).all():
        raise ValueError('Expected a complete source by seed matrix')
    mean = cells.mean(1)
    return dict(point=float(mean.mean()),interval95=np.quantile(mean[draws].mean(1),[.025,.975]).tolist(),
                positive_sources=int((mean > 0.).sum()),zero_sources=int((mean == 0.).sum()))


def choose_threshold(rows, *, min_sources):
    """Predeclared utility diagnostic, not an external or deployment gate."""
    good = [r for r in rows if r['switches'] and r['lower_gain_pp']['interval95'][0] > 0.
            and r['lower_gain_pp']['positive_sources'] >= min_sources]
    return min(good,key=lambda r:(-r['lower_gain_pp']['point'],r['switches'],-r['threshold']))['threshold'] if good else None


def score_bins(score, realized, weights):
    boundaries = np.asarray(THRESHOLDS)
    ids = np.searchsorted(boundaries,score,side='left')
    out = []
    for i in range(len(boundaries)+1):
        use = ids == i
        mass = float(weights[use].sum())
        row = dict(lower=None if i == 0 else float(boundaries[i-1]),
                   upper=None if i == len(boundaries) else float(boundaries[i]),rows=int(use.sum()),weight=mass)
        row.update(predicted=float(weights[use]@score[use]/mass) if mass else None,
                   realized=float(weights[use]@realized[use]/mass) if mass else None)
        out.append(row)
    return out
